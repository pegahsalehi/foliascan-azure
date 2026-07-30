"""Image validation and dataset summary utilities."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from PIL import Image

from leafsignal.data.discovery import ImageRecord

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


@dataclass(frozen=True, slots=True)
class ImageValidationResult:
    """Validation metadata for one discovered image."""

    record: ImageRecord
    is_valid: bool
    width: int | None
    height: int | None
    mode: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class CountRecord:
    """A deterministic count for a string key."""

    key: str
    count: int


@dataclass(frozen=True, slots=True)
class ImageSizeCount:
    """A deterministic count for an image width and height."""

    width: int
    height: int
    count: int


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    """Summary statistics for discovered and validated images."""

    total_class_count: int
    total_discovered_images: int
    total_valid_images: int
    total_corrupted_images: int
    image_count_per_class: tuple[CountRecord, ...]
    corrupted_images: tuple[ImageValidationResult, ...]
    extension_counts: tuple[CountRecord, ...]
    image_mode_counts: tuple[CountRecord, ...]
    image_size_counts: tuple[ImageSizeCount, ...]
    smallest_class_size: int
    largest_class_size: int
    class_imbalance_ratio: float | None


def validate_image(record: ImageRecord) -> ImageValidationResult:
    """Validate one image file without modifying it."""

    try:
        with Image.open(record.path) as image:
            image.verify()

        with Image.open(record.path) as image:
            width, height = image.size
            mode = image.mode

    except (OSError, SyntaxError, ValueError) as exc:
        return ImageValidationResult(
            record=record,
            is_valid=False,
            width=None,
            height=None,
            mode=None,
            error=f"{type(exc).__name__}: {exc}",
        )

    return ImageValidationResult(
        record=record,
        is_valid=True,
        width=width,
        height=height,
        mode=mode,
        error=None,
    )


def validate_images(
    records: Sequence[ImageRecord],
) -> tuple[ImageValidationResult, ...]:
    """Validate discovered images and return one result per image."""

    return tuple(validate_image(record) for record in records)


def summarize_dataset(
    records: Sequence[ImageRecord],
    validation_results: Sequence[ImageValidationResult],
    class_names: Sequence[str] | None = None,
) -> DatasetSummary:
    """Build deterministic summary statistics for a dataset inspection."""

    class_counter: Counter[str] = Counter()
    if class_names is not None:
        for class_name in class_names:
            class_counter[class_name] += 0

    for record in records:
        class_counter[record.class_name] += 1

    extension_counter: Counter[str] = Counter(
        record.path.suffix.lower() for record in records
    )
    mode_counter: Counter[str] = Counter(
        result.mode
        for result in validation_results
        if result.is_valid and result.mode is not None
    )
    size_counter: Counter[tuple[int, int]] = Counter(
        (result.width, result.height)
        for result in validation_results
        if result.is_valid and result.width is not None and result.height is not None
    )
    corrupted_images = tuple(
        sorted(
            (result for result in validation_results if not result.is_valid),
            key=_validation_result_sort_key,
        )
    )
    class_sizes = list(class_counter.values())
    smallest_class_size = min(class_sizes) if class_sizes else 0
    largest_class_size = max(class_sizes) if class_sizes else 0
    imbalance_ratio = (
        largest_class_size / smallest_class_size
        if smallest_class_size > 0
        else None
    )

    return DatasetSummary(
        total_class_count=len(class_counter),
        total_discovered_images=len(records),
        total_valid_images=sum(result.is_valid for result in validation_results),
        total_corrupted_images=len(corrupted_images),
        image_count_per_class=_count_records(class_counter),
        corrupted_images=corrupted_images,
        extension_counts=_count_records(extension_counter),
        image_mode_counts=_count_records(mode_counter),
        image_size_counts=_image_size_count_records(size_counter),
        smallest_class_size=smallest_class_size,
        largest_class_size=largest_class_size,
        class_imbalance_ratio=imbalance_ratio,
    )


def validation_result_to_dict(
    result: ImageValidationResult,
    dataset_root: Path | None = None,
) -> dict[str, JsonValue]:
    """Serialize an image validation result with an optional relative path."""

    return {
        "path": _format_path(result.record.path, dataset_root),
        "class_name": result.record.class_name,
        "is_valid": result.is_valid,
        "width": result.width,
        "height": result.height,
        "mode": result.mode,
        "error": result.error,
    }


def summary_to_dict(
    summary: DatasetSummary,
    dataset_root: Path | None = None,
) -> dict[str, JsonValue]:
    """Serialize a dataset summary with deterministic key and row ordering."""

    return {
        "total_class_count": summary.total_class_count,
        "total_discovered_images": summary.total_discovered_images,
        "total_valid_images": summary.total_valid_images,
        "total_corrupted_images": summary.total_corrupted_images,
        "image_count_per_class": {
            record.key: record.count for record in summary.image_count_per_class
        },
        "corrupted_images": [
            validation_result_to_dict(result, dataset_root)
            for result in summary.corrupted_images
        ],
        "extension_counts": {
            record.key: record.count for record in summary.extension_counts
        },
        "image_mode_counts": {
            record.key: record.count for record in summary.image_mode_counts
        },
        "image_size_counts": [
            {"width": record.width, "height": record.height, "count": record.count}
            for record in summary.image_size_counts
        ],
        "smallest_class_size": summary.smallest_class_size,
        "largest_class_size": summary.largest_class_size,
        "class_imbalance_ratio": summary.class_imbalance_ratio,
    }


def write_summary_report(
    summary: DatasetSummary,
    report_path: Path,
    dataset_root: Path | None = None,
) -> None:
    """Write a JSON dataset inspection report."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary_to_dict(summary, dataset_root), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _count_records(counter: Counter[str]) -> tuple[CountRecord, ...]:
    return tuple(
        CountRecord(key=key, count=counter[key])
        for key in sorted(counter, key=lambda value: (value.casefold(), value))
    )


def _image_size_count_records(
    counter: Counter[tuple[int, int]],
) -> tuple[ImageSizeCount, ...]:
    return tuple(
        ImageSizeCount(width=width, height=height, count=counter[(width, height)])
        for width, height in sorted(counter)
    )


def _validation_result_sort_key(result: ImageValidationResult) -> tuple[str, str]:
    return (result.record.class_name.casefold(), result.record.path.as_posix())


def _format_path(path: Path, dataset_root: Path | None) -> str:
    if dataset_root is None:
        return path.as_posix()

    try:
        return path.resolve().relative_to(dataset_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
