"""PlantVillage tomato ingestion and leaf-group-aware splitting."""

from __future__ import annotations

import csv
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Literal, TypeAlias

from PIL import Image

PlantVillageSplit = Literal["train", "validation", "test"]
SourceSplit = Literal["train", "test"]
DatasetRows: TypeAlias = Mapping[str, Iterable[Mapping[str, object]]]

DATASET_ID: Final[str] = "mohanty/PlantVillage"
DATASET_CONFIG: Final[str] = "color"
HUGGINGFACE_SCRIPT_FILENAME: Final[str] = "plant_village.py"
HUGGINGFACE_SCRIPT_BUILDER_CONFIG: Final[str] = "default"
CROP_NAME: Final[str] = "Tomato"
IMAGE_FORMAT: Final[str] = "JPEG"
IMAGE_EXTENSION: Final[str] = ".jpg"
SOURCE_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "class_name",
    "source_split",
    "leaf_id",
)
FOLIASCAN_MANIFEST_COLUMNS: Final[tuple[str, ...]] = (
    "relative_path",
    "class_name",
    "split",
    "leaf_id",
    "source_split",
)
EXPECTED_TOMATO_CLASSES: Final[tuple[str, ...]] = (
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)


class PlantVillageError(ValueError):
    """Raised when PlantVillage export or splitting cannot continue safely."""


@dataclass(frozen=True, slots=True)
class PlantVillageRecord:
    """A normalized PlantVillage source record."""

    image: object
    class_name: str
    crop: str
    disease: str
    leaf_id: str
    source_split: SourceSplit
    source_index: int


@dataclass(frozen=True, slots=True)
class SourceManifestRecord:
    """A source manifest row for one exported image."""

    relative_path: Path
    class_name: str
    source_split: SourceSplit
    leaf_id: str


@dataclass(frozen=True, slots=True)
class FoliaScanManifestRecord:
    """A FoliaScan manifest row for one exported image."""

    relative_path: Path
    class_name: str
    split: PlantVillageSplit
    leaf_id: str
    source_split: SourceSplit


@dataclass(frozen=True, slots=True)
class PlantVillageExportSummary:
    """Summary returned after exporting the PlantVillage tomato subset."""

    dataset_id: str
    dataset_config: str
    output_dir: Path
    source_manifest_path: Path
    total_records_seen: int
    tomato_records_exported: int
    class_names: tuple[str, ...]
    source_split_counts: tuple[tuple[SourceSplit, int], ...]
    image_format: str


def load_official_plantvillage_dataset() -> Any:
    """Load the official Hugging Face PlantVillage color dataset."""

    from datasets import load_dataset  # type: ignore[import-untyped]
    from huggingface_hub import hf_hub_download

    # The current upstream builder maps "default" to the color image variant.
    dataset_script_path = hf_hub_download(
        repo_id=DATASET_ID,
        filename=HUGGINGFACE_SCRIPT_FILENAME,
        repo_type="dataset",
    )
    return load_dataset(
        dataset_script_path,
        HUGGINGFACE_SCRIPT_BUILDER_CONFIG,
        trust_remote_code=True,
    )


def export_tomato_subset(
    output_dir: Path,
    source_manifest_path: Path,
    *,
    overwrite: bool = False,
    dataset: Any | None = None,
    expected_classes: Sequence[str] | None = EXPECTED_TOMATO_CLASSES,
) -> PlantVillageExportSummary:
    """Export PlantVillage Tomato records as RGB JPEGs and write a source manifest."""

    loaded_dataset = (
        load_official_plantvillage_dataset() if dataset is None else dataset
    )
    records = normalize_plantvillage_records(loaded_dataset)
    tomato_records = tuple(record for record in records if record.crop == CROP_NAME)

    if not tomato_records:
        msg = "No Tomato records were found in the PlantVillage dataset."
        raise PlantVillageError(msg)

    manifest_records = tuple(
        SourceManifestRecord(
            relative_path=_export_relative_path(record),
            class_name=record.class_name,
            source_split=record.source_split,
            leaf_id=record.leaf_id,
        )
        for record in tomato_records
    )
    _validate_source_manifest_records(manifest_records)
    _validate_expected_classes(manifest_records, expected_classes)
    _ensure_outputs_can_be_written(
        output_dir,
        source_manifest_path,
        manifest_records,
        overwrite,
    )

    for record, manifest_record in zip(tomato_records, manifest_records, strict=True):
        export_path = output_dir / manifest_record.relative_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        _save_rgb_jpeg(record.image, export_path)

    write_source_manifest(
        manifest_records,
        source_manifest_path,
        overwrite=overwrite,
    )

    return PlantVillageExportSummary(
        dataset_id=DATASET_ID,
        dataset_config=DATASET_CONFIG,
        output_dir=output_dir,
        source_manifest_path=source_manifest_path,
        total_records_seen=len(records),
        tomato_records_exported=len(tomato_records),
        class_names=_class_names(manifest_records),
        source_split_counts=_source_split_counts(manifest_records),
        image_format=IMAGE_FORMAT,
    )


def normalize_plantvillage_records(dataset: Any) -> tuple[PlantVillageRecord, ...]:
    """Normalize supported PlantVillage dataset objects into typed records."""

    split_rows = _iter_dataset_rows(dataset)
    label_names_by_split = _label_names_by_split(dataset)
    records: list[PlantVillageRecord] = []

    for split_name, rows in split_rows.items():
        source_split = _normalize_source_split(split_name)
        label_names = label_names_by_split.get(split_name, ())
        for source_index, row in enumerate(rows):
            class_name = _class_name_from_row(row, label_names)
            crop = _required_string(row, "crop")
            disease = _required_string(row, "disease")
            leaf_id = _required_string(row, "leaf_id")
            image = _required_value(row, "image")
            records.append(
                PlantVillageRecord(
                    image=image,
                    class_name=class_name,
                    crop=crop,
                    disease=disease,
                    leaf_id=leaf_id,
                    source_split=source_split,
                    source_index=source_index,
                )
            )

    return tuple(records)


def write_source_manifest(
    records: Sequence[SourceManifestRecord],
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the PlantVillage source manifest as UTF-8 CSV."""

    if manifest_path.exists() and not overwrite:
        msg = f"Source manifest already exists: {manifest_path}"
        raise FileExistsError(msg)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=SOURCE_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in sorted(records, key=_source_record_sort_key):
            writer.writerow(
                {
                    "relative_path": record.relative_path.as_posix(),
                    "class_name": record.class_name,
                    "source_split": record.source_split,
                    "leaf_id": record.leaf_id,
                }
            )


def read_source_manifest(manifest_path: Path) -> tuple[SourceManifestRecord, ...]:
    """Read and validate a PlantVillage source manifest."""

    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames != list(SOURCE_MANIFEST_COLUMNS):
            msg = (
                "Source manifest must have columns: "
                f"{', '.join(SOURCE_MANIFEST_COLUMNS)}"
            )
            raise PlantVillageError(msg)
        records = tuple(
            SourceManifestRecord(
                relative_path=_relative_manifest_path(row["relative_path"]),
                class_name=_non_empty_csv_value(row, "class_name"),
                source_split=_normalize_source_split(row["source_split"]),
                leaf_id=_non_empty_csv_value(row, "leaf_id"),
            )
            for row in reader
        )

    _validate_source_manifest_records(records)
    return records


def create_leaf_group_manifest(
    source_records: Sequence[SourceManifestRecord],
    *,
    validation_ratio: float,
    random_seed: int,
    expected_classes: Sequence[str] | None = EXPECTED_TOMATO_CLASSES,
) -> tuple[FoliaScanManifestRecord, ...]:
    """Create a FoliaScan manifest preserving official test and leaf groups."""

    _validate_validation_ratio(validation_ratio)
    if not source_records:
        msg = "Cannot create a FoliaScan manifest from an empty source manifest."
        raise PlantVillageError(msg)

    _validate_source_manifest_records(source_records)
    _validate_expected_classes(source_records, expected_classes)
    official_test_leaf_ids = _official_test_leaf_ids(source_records)
    train_source_records = tuple(
        record for record in source_records if record.source_split == "train"
    )
    _validate_leaf_classes(train_source_records)
    train_validation_source_records = tuple(
        record
        for record in train_source_records
        if record.leaf_id not in official_test_leaf_ids
    )

    leaf_split_by_id = _assign_train_validation_leaf_groups(
        train_validation_source_records,
        validation_ratio=validation_ratio,
        random_seed=random_seed,
    )

    foliascan_records: list[FoliaScanManifestRecord] = []
    for record in source_records:
        split: PlantVillageSplit = (
            "test"
            if record.leaf_id in official_test_leaf_ids
            else leaf_split_by_id[record.leaf_id]
        )
        foliascan_records.append(
            FoliaScanManifestRecord(
                relative_path=record.relative_path,
                class_name=record.class_name,
                split=split,
                leaf_id=record.leaf_id,
                source_split=record.source_split,
            )
        )

    result = tuple(sorted(foliascan_records, key=_foliascan_record_sort_key))
    validate_leaf_group_manifest(
        result,
        source_records,
        expected_classes=expected_classes,
    )
    return result


def write_foliascan_manifest(
    records: Sequence[FoliaScanManifestRecord],
    manifest_path: Path,
    *,
    overwrite: bool = False,
) -> None:
    """Write the leakage-safe FoliaScan dataset manifest as UTF-8 CSV."""

    if manifest_path.exists() and not overwrite:
        msg = f"FoliaScan manifest already exists: {manifest_path}"
        raise FileExistsError(msg)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=FOLIASCAN_MANIFEST_COLUMNS)
        writer.writeheader()
        for record in sorted(records, key=_foliascan_record_sort_key):
            writer.writerow(
                {
                    "relative_path": record.relative_path.as_posix(),
                    "class_name": record.class_name,
                    "split": record.split,
                    "leaf_id": record.leaf_id,
                    "source_split": record.source_split,
                }
            )


def create_and_write_leaf_group_manifest(
    source_manifest_path: Path,
    output_path: Path,
    *,
    validation_ratio: float,
    random_seed: int,
    overwrite: bool = False,
    expected_classes: Sequence[str] | None = EXPECTED_TOMATO_CLASSES,
) -> tuple[FoliaScanManifestRecord, ...]:
    """Read a source manifest, create a leaf-group split, and write it."""

    records = create_leaf_group_manifest(
        read_source_manifest(source_manifest_path),
        validation_ratio=validation_ratio,
        random_seed=random_seed,
        expected_classes=expected_classes,
    )
    write_foliascan_manifest(records, output_path, overwrite=overwrite)
    return records


def validate_leaf_group_manifest(
    records: Sequence[FoliaScanManifestRecord],
    source_records: Sequence[SourceManifestRecord],
    *,
    expected_classes: Sequence[str] | None = EXPECTED_TOMATO_CLASSES,
) -> None:
    """Validate leakage, coverage, official test preservation, and class coverage."""

    source_by_path = {
        record.relative_path.as_posix(): record for record in source_records
    }
    manifest_by_path = {record.relative_path.as_posix(): record for record in records}
    if len(source_by_path) != len(source_records):
        msg = "Source manifest contains duplicate relative_path values."
        raise PlantVillageError(msg)
    if len(manifest_by_path) != len(records):
        msg = "FoliaScan manifest contains duplicate relative_path values."
        raise PlantVillageError(msg)
    if set(source_by_path) != set(manifest_by_path):
        msg = "Every exported source record must appear exactly once in the manifest."
        raise PlantVillageError(msg)

    official_test_leaf_ids = _official_test_leaf_ids(source_records)
    splits_by_leaf_id: defaultdict[str, set[PlantVillageSplit]] = defaultdict(set)
    for record in records:
        if record.relative_path.is_absolute():
            msg = f"Manifest path must be relative: {record.relative_path}"
            raise PlantVillageError(msg)
        source_record = source_by_path[record.relative_path.as_posix()]
        if source_record.source_split == "test" and record.split != "test":
            msg = (
                "Official PlantVillage test records must remain in the "
                f"FoliaScan test split: {record.relative_path}"
            )
            raise PlantVillageError(msg)
        if (
            source_record.source_split == "train"
            and record.split == "test"
            and source_record.leaf_id not in official_test_leaf_ids
        ):
            msg = (
                "Official PlantVillage training records may only move to "
                "FoliaScan test when their leaf_id also appears in the official "
                f"test split: {record.relative_path}"
            )
            raise PlantVillageError(msg)
        splits_by_leaf_id[record.leaf_id].add(record.split)

    leaked_leaf_ids = sorted(
        leaf_id for leaf_id, splits in splits_by_leaf_id.items() if len(splits) > 1
    )
    if leaked_leaf_ids:
        msg = f"leaf_id appears in multiple FoliaScan splits: {leaked_leaf_ids[0]}"
        raise PlantVillageError(msg)

    _validate_expected_classes(source_records, expected_classes)


def manifest_counts_by_split(
    records: Sequence[FoliaScanManifestRecord],
) -> tuple[tuple[PlantVillageSplit, int], ...]:
    """Return deterministic FoliaScan split counts."""

    counter: Counter[PlantVillageSplit] = Counter(record.split for record in records)
    split_names: tuple[PlantVillageSplit, ...] = ("train", "validation", "test")
    return tuple((split, counter[split]) for split in split_names)


def _official_test_leaf_ids(records: Sequence[SourceManifestRecord]) -> set[str]:
    return {record.leaf_id for record in records if record.source_split == "test"}


def _iter_dataset_rows(dataset: Any) -> dict[str, Iterable[Mapping[str, object]]]:
    if isinstance(dataset, Mapping):
        return {str(split_name): rows for split_name, rows in dataset.items()}

    msg = "PlantVillage dataset must provide train/test splits."
    raise PlantVillageError(msg)


def _label_names_by_split(dataset: Any) -> dict[str, tuple[str, ...]]:
    if not isinstance(dataset, Mapping):
        return {}

    return {
        str(split_name): _label_names_from_split(split_dataset)
        for split_name, split_dataset in dataset.items()
    }


def _label_names_from_split(split_dataset: object) -> tuple[str, ...]:
    features = getattr(split_dataset, "features", None)
    if not isinstance(features, Mapping):
        return ()
    label_feature = features.get("label")
    names = getattr(label_feature, "names", None)
    if isinstance(names, Sequence) and not isinstance(names, str):
        return tuple(str(name) for name in names)
    return ()


def _class_name_from_row(
    row: Mapping[str, object],
    label_names: Sequence[str],
) -> str:
    for field_name in ("class_name", "label_name", "class"):
        value = row.get(field_name)
        if isinstance(value, str) and value:
            return value

    label = row.get("label")
    if isinstance(label, str) and label:
        return label
    if isinstance(label, int) and label_names:
        try:
            return label_names[label]
        except IndexError as exc:
            msg = f"Label id {label} is not present in dataset label metadata."
            raise PlantVillageError(msg) from exc

    crop = row.get("crop")
    disease = row.get("disease")
    if isinstance(crop, str) and crop and isinstance(disease, str) and disease:
        return f"{crop}___{disease}"

    msg = "PlantVillage record is missing class metadata."
    raise PlantVillageError(msg)


def _required_string(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        msg = f"PlantVillage record is missing required metadata: {key}"
        raise PlantVillageError(msg)
    return value


def _required_value(row: Mapping[str, object], key: str) -> object:
    value = row.get(key)
    if value is None:
        msg = f"PlantVillage record is missing required metadata: {key}"
        raise PlantVillageError(msg)
    return value


def _normalize_source_split(value: str) -> SourceSplit:
    if value in {"train", "training"}:
        return "train"
    if value in {"test", "testing"}:
        return "test"
    msg = f"Unsupported PlantVillage source split: {value}"
    raise PlantVillageError(msg)


def _export_relative_path(record: PlantVillageRecord) -> Path:
    leaf_id = _safe_filename_part(record.leaf_id)
    filename = (
        f"{record.source_split}_{leaf_id}_{record.source_index:06d}{IMAGE_EXTENSION}"
    )
    return Path(record.class_name) / filename


def _safe_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "leaf"


def _save_rgb_jpeg(image_value: object, export_path: Path) -> None:
    try:
        with _open_image(image_value) as image:
            rgb_image = image.convert("RGB")
            rgb_image.save(export_path, IMAGE_FORMAT, quality=95)
    except Exception as exc:
        msg = f"Failed to export image to {export_path}: {exc}"
        raise PlantVillageError(msg) from exc


def _open_image(image_value: object) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value.copy()
    if isinstance(image_value, Mapping):
        path = image_value.get("path")
        if isinstance(path, str):
            return Image.open(path)
        image_bytes = image_value.get("bytes")
        if isinstance(image_bytes, bytes):
            return Image.open(BytesIO(image_bytes))

    msg = "Unsupported PlantVillage image value; expected a Pillow image or image dict."
    raise PlantVillageError(msg)


def _ensure_outputs_can_be_written(
    output_dir: Path,
    source_manifest_path: Path,
    records: Sequence[SourceManifestRecord],
    overwrite: bool,
) -> None:
    if source_manifest_path.exists() and not overwrite:
        msg = f"Source manifest already exists: {source_manifest_path}"
        raise FileExistsError(msg)

    existing_images = tuple(
        output_dir / record.relative_path
        for record in records
        if (output_dir / record.relative_path).exists()
    )
    if existing_images and not overwrite:
        first_path = existing_images[0]
        msg = (
            "Export output contains existing images; use --overwrite to replace "
            f"them. First existing image: {first_path}"
        )
        raise FileExistsError(msg)


def _validate_source_manifest_records(
    records: Sequence[SourceManifestRecord],
) -> None:
    if not records:
        msg = "Source manifest contains no records."
        raise PlantVillageError(msg)

    seen_paths: set[str] = set()
    for record in records:
        if record.relative_path.is_absolute():
            msg = f"Source manifest path must be relative: {record.relative_path}"
            raise PlantVillageError(msg)
        if not record.class_name:
            msg = f"Missing class_name for source manifest row: {record.relative_path}"
            raise PlantVillageError(msg)
        if not record.leaf_id:
            msg = f"Missing leaf_id for source manifest row: {record.relative_path}"
            raise PlantVillageError(msg)
        relative_path = record.relative_path.as_posix()
        if relative_path in seen_paths:
            msg = f"Duplicate source manifest relative_path: {relative_path}"
            raise PlantVillageError(msg)
        seen_paths.add(relative_path)


def _validate_expected_classes(
    records: Sequence[SourceManifestRecord],
    expected_classes: Sequence[str] | None,
) -> None:
    if expected_classes is None:
        return

    present_classes = {record.class_name for record in records}
    missing_classes = [
        class_name
        for class_name in expected_classes
        if class_name not in present_classes
    ]
    if missing_classes:
        msg = "Expected tomato classes are missing: " + ", ".join(missing_classes)
        raise PlantVillageError(msg)


def _validate_validation_ratio(validation_ratio: float) -> None:
    if validation_ratio <= 0 or validation_ratio >= 1:
        msg = "Validation ratio must be greater than 0 and less than 1."
        raise PlantVillageError(msg)


def _validate_leaf_classes(records: Sequence[SourceManifestRecord]) -> None:
    classes_by_leaf_id: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        classes_by_leaf_id[record.leaf_id].add(record.class_name)

    for leaf_id, class_names in classes_by_leaf_id.items():
        if len(class_names) > 1:
            msg = (
                "A leaf_id appears under multiple classes in the official training "
                f"split, which is unsafe to split: {leaf_id}"
            )
            raise PlantVillageError(msg)


def _assign_train_validation_leaf_groups(
    records: Sequence[SourceManifestRecord],
    *,
    validation_ratio: float,
    random_seed: int,
) -> dict[str, Literal["train", "validation"]]:
    leaf_ids_by_class: defaultdict[str, set[str]] = defaultdict(set)
    for record in records:
        leaf_ids_by_class[record.class_name].add(record.leaf_id)

    assignments: dict[str, Literal["train", "validation"]] = {}
    for class_name in sorted(
        leaf_ids_by_class,
        key=lambda value: (value.casefold(), value),
    ):
        leaf_ids = sorted(leaf_ids_by_class[class_name])
        if len(leaf_ids) < 2:
            msg = (
                f"Class '{class_name}' has {len(leaf_ids)} distinct training leaf "
                "groups; at least 2 are required for train/validation splitting."
            )
            raise PlantVillageError(msg)

        validation_count = _validation_leaf_count(len(leaf_ids), validation_ratio)
        shuffled_leaf_ids = list(leaf_ids)
        random.Random(f"{random_seed}:{class_name}").shuffle(shuffled_leaf_ids)
        validation_leaf_ids = set(shuffled_leaf_ids[:validation_count])

        for leaf_id in leaf_ids:
            assignments[leaf_id] = (
                "validation" if leaf_id in validation_leaf_ids else "train"
            )

    return assignments


def _validation_leaf_count(leaf_count: int, validation_ratio: float) -> int:
    validation_count = round(leaf_count * validation_ratio)
    return min(max(validation_count, 1), leaf_count - 1)


def _relative_manifest_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        msg = f"Manifest relative_path must be relative: {value}"
        raise PlantVillageError(msg)
    if not value:
        msg = "Manifest relative_path must not be empty."
        raise PlantVillageError(msg)
    return path


def _non_empty_csv_value(row: Mapping[str, str], key: str) -> str:
    value = row.get(key, "")
    if not value:
        msg = f"Manifest row is missing required value: {key}"
        raise PlantVillageError(msg)
    return value


def _class_names(records: Sequence[SourceManifestRecord]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {record.class_name for record in records},
            key=lambda value: (value.casefold(), value),
        )
    )


def _source_split_counts(
    records: Sequence[SourceManifestRecord],
) -> tuple[tuple[SourceSplit, int], ...]:
    counter: Counter[SourceSplit] = Counter(record.source_split for record in records)
    return (("train", counter["train"]), ("test", counter["test"]))


def _source_record_sort_key(record: SourceManifestRecord) -> tuple[str, str, str, str]:
    source_split_order = {"train": "0", "test": "1"}
    return (
        source_split_order[record.source_split],
        record.class_name.casefold(),
        record.leaf_id,
        record.relative_path.as_posix(),
    )


def _foliascan_record_sort_key(
    record: FoliaScanManifestRecord,
) -> tuple[int, str, str, str]:
    split_order = {"train": 0, "validation": 1, "test": 2}
    return (
        split_order[record.split],
        record.class_name.casefold(),
        record.leaf_id,
        record.relative_path.as_posix(),
    )
