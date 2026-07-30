"""Reproducible stratified dataset splitting and manifest writing."""

from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

from leafsignal.data.discovery import ImageRecord

SplitName = Literal["train", "validation", "test"]

RATIO_TOLERANCE: Final[float] = 1e-6
SPLIT_ORDER: Final[tuple[SplitName, ...]] = ("train", "validation", "test")


@dataclass(frozen=True, slots=True)
class SplitRecord:
    """A relative-path split assignment for one image."""

    relative_path: Path
    class_name: str
    split: SplitName


def create_split_assignments(
    records: Sequence[ImageRecord],
    dataset_root: Path,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int,
) -> tuple[SplitRecord, ...]:
    """Create deterministic, class-stratified train/validation/test assignments."""

    validate_split_ratios(train_ratio, validation_ratio, test_ratio)
    if not records:
        msg = "Cannot split an empty dataset."
        raise ValueError(msg)

    records_by_class: defaultdict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        records_by_class[record.class_name].append(record)

    assignments: list[SplitRecord] = []
    ratios = (train_ratio, validation_ratio, test_ratio)

    for class_name in sorted(
        records_by_class,
        key=lambda value: (value.casefold(), value),
    ):
        class_records = sorted(
            records_by_class[class_name],
            key=lambda record: _relative_path(record.path, dataset_root).as_posix(),
        )
        split_counts = _split_counts(len(class_records), ratios, class_name)
        shuffled_records = list(class_records)
        random.Random(f"{random_seed}:{class_name}").shuffle(shuffled_records)

        cursor = 0
        for split_name, split_count in zip(SPLIT_ORDER, split_counts, strict=True):
            for record in shuffled_records[cursor : cursor + split_count]:
                assignments.append(
                    SplitRecord(
                        relative_path=_relative_path(record.path, dataset_root),
                        class_name=record.class_name,
                        split=split_name,
                    )
                )
            cursor += split_count

    return tuple(sorted(assignments, key=_split_record_sort_key))


def validate_split_ratios(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> None:
    """Validate positive three-way split ratios that sum to one."""

    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio <= 0 for ratio in ratios):
        msg = "Train, validation, and test ratios must all be greater than zero."
        raise ValueError(msg)

    ratio_sum = sum(ratios)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=RATIO_TOLERANCE):
        msg = (
            "Train, validation, and test ratios must sum to 1.0 "
            f"(received {ratio_sum:.6f})."
        )
        raise ValueError(msg)


def write_manifest(
    split_records: Sequence[SplitRecord],
    manifest_path: Path,
    overwrite: bool = False,
) -> None:
    """Write split assignments to a UTF-8 CSV manifest."""

    if manifest_path.exists() and not overwrite:
        msg = f"Manifest already exists: {manifest_path}"
        raise FileExistsError(msg)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_records = sorted(split_records, key=_split_record_sort_key)

    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("relative_path", "class_name", "split"),
        )
        writer.writeheader()
        for record in sorted_records:
            writer.writerow(
                {
                    "relative_path": record.relative_path.as_posix(),
                    "class_name": record.class_name,
                    "split": record.split,
                }
            )


def split_counts(split_records: Sequence[SplitRecord]) -> tuple[CountRecord, ...]:
    """Return deterministic counts for split names."""

    counter: Counter[str] = Counter(record.split for record in split_records)
    return tuple(
        CountRecord(split_name, counter[split_name]) for split_name in SPLIT_ORDER
    )


@dataclass(frozen=True, slots=True)
class CountRecord:
    """A deterministic count for one split name."""

    key: str
    count: int


def _split_counts(
    sample_count: int,
    ratios: tuple[float, float, float],
    class_name: str,
) -> tuple[int, int, int]:
    split_count = len(SPLIT_ORDER)
    if sample_count < split_count:
        msg = (
            f"Class '{class_name}' has {sample_count} samples; at least "
            f"{split_count} are required for a three-way stratified split."
        )
        raise ValueError(msg)

    counts = [1 for _ in SPLIT_ORDER]
    targets = [sample_count * ratio for ratio in ratios]

    while sum(counts) < sample_count:
        split_index = max(
            range(split_count),
            key=lambda index: (
                targets[index] - counts[index],
                ratios[index],
                -index,
            ),
        )
        counts[split_index] += 1

    return (counts[0], counts[1], counts[2])


def _relative_path(path: Path, dataset_root: Path) -> Path:
    try:
        return path.resolve().relative_to(dataset_root.resolve())
    except ValueError as exc:
        msg = f"Image path is not inside dataset root: {path}"
        raise ValueError(msg) from exc


def _split_record_sort_key(record: SplitRecord) -> tuple[int, str, str]:
    return (
        SPLIT_ORDER.index(record.split),
        record.class_name.casefold(),
        record.relative_path.as_posix(),
    )
