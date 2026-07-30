import csv
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from foliascan.data import cli
from foliascan.data.discovery import ImageRecord
from foliascan.data.splitting import (
    SplitRecord,
    create_split_assignments,
    write_manifest,
)


def test_create_split_assignments_is_deterministic_stratified_and_exhaustive(
    tmp_path: Path,
) -> None:
    records = _make_records(tmp_path, class_names=("a", "b"), images_per_class=10)

    first = create_split_assignments(records, tmp_path, 0.6, 0.2, 0.2, 123)
    second = create_split_assignments(records, tmp_path, 0.6, 0.2, 0.2, 123)

    assert first == second
    assert len(first) == len(records)
    assert {record.relative_path for record in first} == {
        record.path.relative_to(tmp_path) for record in records
    }
    assert len({record.relative_path for record in first}) == len(records)
    assert _counts_by_class_and_split(first) == {
        ("a", "train"): 6,
        ("a", "validation"): 2,
        ("a", "test"): 2,
        ("b", "train"): 6,
        ("b", "validation"): 2,
        ("b", "test"): 2,
    }


def test_create_split_assignments_rejects_invalid_ratios(tmp_path: Path) -> None:
    records = _make_records(tmp_path, class_names=("a",), images_per_class=3)

    with pytest.raises(ValueError, match="sum to 1.0"):
        create_split_assignments(records, tmp_path, 0.8, 0.1, 0.2, 42)


def test_create_split_assignments_rejects_too_few_samples_per_class(
    tmp_path: Path,
) -> None:
    records = _make_records(tmp_path, class_names=("a",), images_per_class=2)

    with pytest.raises(ValueError, match="at least 3"):
        create_split_assignments(records, tmp_path, 0.6, 0.2, 0.2, 42)


def test_write_manifest_uses_relative_paths_and_protects_existing(
    tmp_path: Path,
) -> None:
    records = _make_records(tmp_path, class_names=("a",), images_per_class=3)
    split_records = create_split_assignments(records, tmp_path, 0.5, 0.25, 0.25, 42)
    manifest_path = tmp_path / "processed" / "manifest.csv"

    write_manifest(split_records, manifest_path)
    rows = _read_manifest(manifest_path)

    assert manifest_path.exists()
    assert rows[0].keys() == {"relative_path", "class_name", "split"}
    assert all(not Path(row["relative_path"]).is_absolute() for row in rows)
    assert {row["relative_path"] for row in rows} == {
        record.path.relative_to(tmp_path).as_posix() for record in records
    }

    with pytest.raises(FileExistsError, match="already exists"):
        write_manifest(split_records, manifest_path)

    write_manifest(split_records, manifest_path, overwrite=True)


def test_cli_inspect_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _make_image_dataset(tmp_path, class_names=("a", "b"), images_per_class=1)
    report_path = tmp_path / "report.json"

    exit_status = cli.main(
        ["inspect", "--data-dir", str(tmp_path), "--json-report", str(report_path)]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "Dataset inspection complete" in captured.out
    assert "Valid images: 2" in captured.out
    assert report_path.exists()


def test_cli_split_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _make_image_dataset(tmp_path, class_names=("a", "b"), images_per_class=4)
    manifest_path = tmp_path / "processed" / "manifest.csv"

    exit_status = cli.main(
        [
            "split",
            "--data-dir",
            str(tmp_path),
            "--output",
            str(manifest_path),
            "--train-ratio",
            "0.5",
            "--validation-ratio",
            "0.25",
            "--test-ratio",
            "0.25",
            "--random-seed",
            "7",
        ]
    )

    captured = capsys.readouterr()
    rows = _read_manifest(manifest_path)
    assert exit_status == 0
    assert manifest_path.exists()
    assert len(rows) == 8
    assert "Manifest written:" in captured.out
    assert "train: 4" in captured.out
    assert "validation: 2" in captured.out
    assert "test: 2" in captured.out


def _make_records(
    dataset_root: Path,
    class_names: tuple[str, ...],
    images_per_class: int,
) -> tuple[ImageRecord, ...]:
    records: list[ImageRecord] = []
    for class_name in class_names:
        class_dir = dataset_root / class_name
        class_dir.mkdir()
        for index in range(images_per_class):
            path = class_dir / f"image_{index:03}.jpg"
            path.touch()
            records.append(ImageRecord(path, class_name))
    return tuple(records)


def _make_image_dataset(
    dataset_root: Path,
    class_names: tuple[str, ...],
    images_per_class: int,
) -> None:
    for class_name in class_names:
        class_dir = dataset_root / class_name
        class_dir.mkdir()
        for index in range(images_per_class):
            image_path = class_dir / f"image_{index:03}.jpg"
            Image.new("RGB", (8, 8)).save(image_path)


def _counts_by_class_and_split(
    split_records: tuple[SplitRecord, ...],
) -> dict[tuple[str, str], int]:
    return dict(
        Counter((record.class_name, record.split) for record in split_records)
    )


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        return list(csv.DictReader(manifest_file))
