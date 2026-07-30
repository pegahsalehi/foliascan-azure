from pathlib import Path

import pytest

from foliascan.data.discovery import (
    DatasetDiscoveryError,
    ImageRecord,
    discover_class_directories,
    discover_class_names,
    discover_image_records,
)


def test_discover_class_names_are_sorted_and_ignore_hidden_dirs(
    tmp_path: Path,
) -> None:
    (tmp_path / "Tomato___healthy").mkdir()
    (tmp_path / ".hidden_class").mkdir()
    (tmp_path / "Tomato___Bacterial_spot").mkdir()

    assert discover_class_names(tmp_path) == (
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
    )


def test_discover_image_records_filters_files_and_extensions(tmp_path: Path) -> None:
    bacterial = tmp_path / "Tomato___Bacterial_spot"
    healthy = tmp_path / "Tomato___healthy"
    bacterial.mkdir()
    healthy.mkdir()

    (bacterial / "image_002.PNG").touch()
    (bacterial / "image_001.JPG").touch()
    (bacterial / ".hidden.jpg").touch()
    (bacterial / "notes.txt").touch()
    nested = bacterial / "nested"
    nested.mkdir()
    (nested / "nested_image.jpg").touch()
    (healthy / "image_003.jpeg").touch()

    assert discover_image_records(tmp_path) == (
        ImageRecord(bacterial / "image_001.JPG", "Tomato___Bacterial_spot"),
        ImageRecord(bacterial / "image_002.PNG", "Tomato___Bacterial_spot"),
        ImageRecord(healthy / "image_003.jpeg", "Tomato___healthy"),
    )


def test_discover_class_directories_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(DatasetDiscoveryError, match="does not exist"):
        discover_class_directories(tmp_path / "missing")


def test_discover_class_directories_rejects_file_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset.txt"
    dataset_root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(DatasetDiscoveryError, match="not a directory"):
        discover_class_directories(dataset_root)


def test_discover_class_directories_rejects_root_without_classes(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.txt").write_text("no class directories", encoding="utf-8")

    with pytest.raises(DatasetDiscoveryError, match="No class directories"):
        discover_class_directories(tmp_path)
