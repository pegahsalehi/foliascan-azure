import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor

from foliascan.training.dataset import (
    MANIFEST_COLUMNS,
    ClassMapping,
    ManifestImageDataset,
    ManifestRecord,
    TrainingDataError,
    build_class_mapping,
    read_training_manifest,
)


def test_read_training_manifest_validates_and_sorts_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            ("b/image.jpg", "class_b", "test", "leaf_b", "test"),
            ("a/image.jpg", "class_a", "train", "leaf_a", "train"),
            ("c/image.jpg", "class_c", "validation", "leaf_c", "train"),
        ],
    )

    records = read_training_manifest(manifest_path)

    assert [record.relative_path for record in records] == [
        Path("a/image.jpg"),
        Path("c/image.jpg"),
        Path("b/image.jpg"),
    ]
    assert all(isinstance(record.relative_path, Path) for record in records)


def test_read_training_manifest_rejects_absolute_paths(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.csv"
    _write_manifest(
        manifest_path,
        [(str(tmp_path / "image.jpg"), "class_a", "train", "leaf_a", "train")],
    )

    with pytest.raises(TrainingDataError, match="relative"):
        read_training_manifest(manifest_path)


def test_read_training_manifest_rejects_duplicate_relative_paths(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "dataset_manifest.csv"
    _write_manifest(
        manifest_path,
        [
            ("a/image.jpg", "class_a", "train", "leaf_a", "train"),
            ("a/image.jpg", "class_a", "validation", "leaf_b", "train"),
        ],
    )

    with pytest.raises(TrainingDataError, match="Duplicate"):
        read_training_manifest(manifest_path)


@pytest.mark.parametrize(
    ("row", "error_match"),
    [
        (("a/image.jpg", "", "train", "leaf_a", "train"), "class_name"),
        (("a/image.jpg", "class_a", "holdout", "leaf_a", "train"), "split"),
        (("a/image.jpg", "class_a", "train", "", "train"), "leaf_id"),
        (("a/image.jpg", "class_a", "train", "leaf_a", "validation"), "source"),
    ],
)
def test_read_training_manifest_rejects_invalid_rows(
    tmp_path: Path,
    row: tuple[str, str, str, str, str],
    error_match: str,
) -> None:
    manifest_path = tmp_path / "dataset_manifest.csv"
    _write_manifest(manifest_path, [row])

    with pytest.raises(TrainingDataError, match=error_match):
        read_training_manifest(manifest_path)


def test_read_training_manifest_rejects_unexpected_columns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "dataset_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=("relative_path", "split"))
        writer.writeheader()
        writer.writerow({"relative_path": "a/image.jpg", "split": "train"})

    with pytest.raises(TrainingDataError, match="columns"):
        read_training_manifest(manifest_path)


def test_build_class_mapping_is_deterministic_and_zero_based() -> None:
    records = (
        _record("b/image.jpg", "Tomato___healthy", "train", "leaf_b"),
        _record("a/image.jpg", "Tomato___Bacterial_spot", "validation", "leaf_a"),
    )

    mapping = build_class_mapping(records)

    assert mapping.index_to_class == (
        "Tomato___Bacterial_spot",
        "Tomato___healthy",
    )
    assert dict(mapping.class_to_index) == {
        "Tomato___Bacterial_spot": 0,
        "Tomato___healthy": 1,
    }


def test_build_class_mapping_rejects_no_classes() -> None:
    with pytest.raises(TrainingDataError, match="empty manifest"):
        build_class_mapping(())


def test_manifest_image_dataset_converts_to_rgb_and_returns_tensor(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "Tomato___healthy" / "image.jpg"
    image_path.parent.mkdir()
    Image.new("L", (10, 12), color=128).save(image_path)
    seen_modes: list[str] = []

    def transform(image: Image.Image) -> Tensor:
        seen_modes.append(image.mode)
        return torch.zeros((3, image.height, image.width))

    dataset = ManifestImageDataset(
        [_record("Tomato___healthy/image.jpg", "Tomato___healthy", "train", "leaf")],
        tmp_path,
        ClassMapping({"Tomato___healthy": 0}, ("Tomato___healthy",)),
        transform,
    )

    image_tensor, target = dataset[0]

    assert seen_modes == ["RGB"]
    assert image_tensor.shape == (3, 12, 10)
    assert target == 0


def test_manifest_image_dataset_reports_missing_images(tmp_path: Path) -> None:
    dataset = ManifestImageDataset(
        [_record("Tomato___healthy/missing.jpg", "Tomato___healthy", "train", "leaf")],
        tmp_path,
        ClassMapping({"Tomato___healthy": 0}, ("Tomato___healthy",)),
        lambda image: torch.zeros((3, image.height, image.width)),
    )

    with pytest.raises(TrainingDataError, match="does not exist"):
        dataset[0]


def test_manifest_image_dataset_rejects_paths_that_escape_root(tmp_path: Path) -> None:
    dataset = ManifestImageDataset(
        [_record("../outside.jpg", "Tomato___healthy", "train", "leaf")],
        tmp_path,
        ClassMapping({"Tomato___healthy": 0}, ("Tomato___healthy",)),
        lambda image: torch.zeros((3, image.height, image.width)),
    )

    with pytest.raises(TrainingDataError, match="escapes"):
        dataset[0]


def _record(
    relative_path: str,
    class_name: str,
    split: str,
    leaf_id: str,
) -> ManifestRecord:
    return ManifestRecord(
        Path(relative_path),
        class_name,
        split,  # type: ignore[arg-type]
        leaf_id,
        "test" if split == "test" else "train",
    )


def _write_manifest(
    manifest_path: Path,
    rows: list[tuple[str, str, str, str, str]],
) -> None:
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for relative_path, class_name, split, leaf_id, source_split in rows:
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "class_name": class_name,
                    "split": split,
                    "leaf_id": leaf_id,
                    "source_split": source_split,
                }
            )

