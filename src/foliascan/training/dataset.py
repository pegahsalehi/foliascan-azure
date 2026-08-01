"""Manifest-driven image dataset utilities."""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

SplitName = Literal["train", "validation", "test"]
SourceSplitName = Literal["train", "test"]

MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "class_name",
    "split",
    "leaf_id",
    "source_split",
)
SPLIT_ORDER: Final[tuple[SplitName, ...]] = ("train", "validation", "test")


class TrainingDataError(ValueError):
    """Raised when training data or manifest rows are invalid."""


class ImageTransform(Protocol):
    """Callable image transform returning a PyTorch tensor."""

    def __call__(self, image: Image.Image) -> Tensor:
        """Transform one Pillow image into a tensor."""


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """Immutable training manifest row."""

    relative_path: Path
    class_name: str
    split: SplitName
    leaf_id: str
    source_split: SourceSplitName


@dataclass(frozen=True, slots=True)
class ClassMapping:
    """Deterministic class-name mapping shared across all splits."""

    class_to_index: Mapping[str, int]
    index_to_class: tuple[str, ...]

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the mapping."""

        return len(self.index_to_class)


class ManifestImageDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch dataset that loads images lazily from a FoliaScan manifest."""

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        data_dir: Path,
        class_mapping: ClassMapping,
        image_transform: ImageTransform,
    ) -> None:
        self._records = tuple(records)
        self._data_dir = data_dir
        self._data_dir_resolved = data_dir.resolve()
        self._class_mapping = class_mapping
        self._image_transform = image_transform
        _validate_records_have_class_indexes(self._records, class_mapping)

    @property
    def records(self) -> tuple[ManifestRecord, ...]:
        """Return manifest records represented by this dataset."""

        return self._records

    def __len__(self) -> int:
        """Return dataset length without loading image bytes."""

        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Tensor, int]:
        """Load and transform one image-target pair."""

        record = self._records[index]
        image_path = self._image_path(record)
        if not image_path.exists():
            msg = f"Manifest image does not exist: {image_path}"
            raise TrainingDataError(msg)

        try:
            with Image.open(image_path) as image:
                image_tensor = self._image_transform(image.convert("RGB"))
        except OSError as exc:
            msg = f"Unable to read manifest image: {image_path}"
            raise TrainingDataError(msg) from exc

        return image_tensor, self._class_mapping.class_to_index[record.class_name]

    def _image_path(self, record: ManifestRecord) -> Path:
        if record.relative_path.is_absolute():
            msg = f"Manifest path must be relative: {record.relative_path}"
            raise TrainingDataError(msg)

        image_path = (self._data_dir / record.relative_path).resolve()
        try:
            image_path.relative_to(self._data_dir_resolved)
        except ValueError as exc:
            msg = f"Manifest image path escapes dataset root: {record.relative_path}"
            raise TrainingDataError(msg) from exc
        return image_path


def read_training_manifest(manifest_path: Path) -> tuple[ManifestRecord, ...]:
    """Read and validate a FoliaScan training manifest CSV."""

    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames != list(MANIFEST_COLUMNS):
            msg = (
                "Training manifest must have columns: "
                f"{', '.join(MANIFEST_COLUMNS)}"
            )
            raise TrainingDataError(msg)
        records = tuple(_record_from_row(row) for row in reader)

    _validate_manifest_records(records)
    return tuple(sorted(records, key=_manifest_record_sort_key))


def build_class_mapping(records: Sequence[ManifestRecord]) -> ClassMapping:
    """Create a deterministic class-name to integer-index mapping."""

    class_names = tuple(
        sorted({record.class_name for record in records}, key=_deterministic_text_key)
    )
    if not class_names:
        msg = "Cannot build a class mapping from an empty manifest."
        raise TrainingDataError(msg)

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    return ClassMapping(
        class_to_index=MappingProxyType(class_to_index),
        index_to_class=class_names,
    )


def records_for_split(
    records: Sequence[ManifestRecord],
    split: SplitName,
) -> tuple[ManifestRecord, ...]:
    """Return deterministic manifest records for one split."""

    return tuple(record for record in records if record.split == split)


def _record_from_row(row: Mapping[str, str]) -> ManifestRecord:
    return ManifestRecord(
        relative_path=_relative_manifest_path(row["relative_path"]),
        class_name=_non_empty_csv_value(row, "class_name"),
        split=_normalize_split(row["split"]),
        leaf_id=_non_empty_csv_value(row, "leaf_id"),
        source_split=_normalize_source_split(row["source_split"]),
    )


def _validate_manifest_records(records: Sequence[ManifestRecord]) -> None:
    seen_paths: set[str] = set()
    for record in records:
        relative_path = record.relative_path.as_posix()
        if relative_path in seen_paths:
            msg = f"Duplicate training manifest relative_path: {relative_path}"
            raise TrainingDataError(msg)
        seen_paths.add(relative_path)


def _validate_records_have_class_indexes(
    records: Sequence[ManifestRecord],
    class_mapping: ClassMapping,
) -> None:
    missing_classes = sorted(
        {
            record.class_name
            for record in records
            if record.class_name not in class_mapping.class_to_index
        },
        key=_deterministic_text_key,
    )
    if missing_classes:
        msg = "Manifest class is missing from class mapping: " + missing_classes[0]
        raise TrainingDataError(msg)


def _relative_manifest_path(value: str) -> Path:
    path = Path(value)
    if not value:
        msg = "Training manifest relative_path must not be empty."
        raise TrainingDataError(msg)
    if path.is_absolute():
        msg = f"Training manifest path must be relative: {value}"
        raise TrainingDataError(msg)
    return path


def _non_empty_csv_value(row: Mapping[str, str], key: str) -> str:
    value = row.get(key, "")
    if not value:
        msg = f"Training manifest row is missing required value: {key}"
        raise TrainingDataError(msg)
    return value


def _normalize_split(value: str) -> SplitName:
    if value in SPLIT_ORDER:
        return value
    msg = f"Unsupported FoliaScan split: {value}"
    raise TrainingDataError(msg)


def _normalize_source_split(value: str) -> SourceSplitName:
    if value == "train":
        return "train"
    if value == "test":
        return "test"
    msg = f"Unsupported source_split: {value}"
    raise TrainingDataError(msg)


def _manifest_record_sort_key(record: ManifestRecord) -> tuple[int, str, str, str]:
    return (
        SPLIT_ORDER.index(record.split),
        record.class_name.casefold(),
        record.leaf_id,
        record.relative_path.as_posix(),
    )


def _deterministic_text_key(value: str) -> tuple[str, str]:
    return (value.casefold(), value)
