from pathlib import Path

from PIL import Image

from leafsignal.data.discovery import ImageRecord
from leafsignal.data.validation import (
    CountRecord,
    ImageSizeCount,
    summarize_dataset,
    validate_images,
)


def test_validate_images_collects_valid_metadata(tmp_path: Path) -> None:
    class_dir = tmp_path / "Tomato___healthy"
    class_dir.mkdir()
    rgb_path = class_dir / "rgb.jpg"
    gray_path = class_dir / "gray.png"
    _create_image(rgb_path, size=(8, 6), mode="RGB")
    _create_image(gray_path, size=(5, 7), mode="L")
    records = (
        ImageRecord(rgb_path, "Tomato___healthy"),
        ImageRecord(gray_path, "Tomato___healthy"),
    )

    results = validate_images(records)
    summary = summarize_dataset(records, results, ["Tomato___healthy"])

    assert all(result.is_valid for result in results)
    assert {(result.width, result.height, result.mode) for result in results} == {
        (8, 6, "RGB"),
        (5, 7, "L"),
    }
    assert summary.total_valid_images == 2
    assert summary.image_mode_counts == (
        CountRecord("L", 1),
        CountRecord("RGB", 1),
    )
    assert summary.image_size_counts == (
        ImageSizeCount(width=5, height=7, count=1),
        ImageSizeCount(width=8, height=6, count=1),
    )


def test_validate_images_records_corrupted_images(tmp_path: Path) -> None:
    class_dir = tmp_path / "Tomato___Early_blight"
    class_dir.mkdir()
    valid_path = class_dir / "valid.png"
    corrupted_path = class_dir / "corrupted.jpg"
    _create_image(valid_path, size=(4, 4), mode="RGB")
    corrupted_path.write_text("not an image", encoding="utf-8")
    records = (
        ImageRecord(corrupted_path, "Tomato___Early_blight"),
        ImageRecord(valid_path, "Tomato___Early_blight"),
    )

    results = validate_images(records)
    summary = summarize_dataset(records, results, ["Tomato___Early_blight"])

    corrupted = [result for result in results if not result.is_valid]
    assert len(corrupted) == 1
    assert corrupted[0].record.path == corrupted_path
    assert corrupted[0].error is not None
    assert summary.total_discovered_images == 2
    assert summary.total_valid_images == 1
    assert summary.total_corrupted_images == 1
    assert summary.corrupted_images == (corrupted[0],)


def test_summarize_dataset_counts_classes_extensions_and_imbalance(
    tmp_path: Path,
) -> None:
    healthy = tmp_path / "Tomato___healthy"
    bacterial = tmp_path / "Tomato___Bacterial_spot"
    empty = tmp_path / "Tomato___Late_blight"
    healthy.mkdir()
    bacterial.mkdir()
    empty.mkdir()
    healthy_image = healthy / "healthy.JPG"
    bacterial_image = bacterial / "bacterial.png"
    _create_image(healthy_image, size=(10, 10), mode="RGB")
    _create_image(bacterial_image, size=(10, 10), mode="RGB")
    records = (
        ImageRecord(bacterial_image, "Tomato___Bacterial_spot"),
        ImageRecord(healthy_image, "Tomato___healthy"),
    )

    summary = summarize_dataset(
        records,
        validate_images(records),
        [
            "Tomato___Bacterial_spot",
            "Tomato___Late_blight",
            "Tomato___healthy",
        ],
    )

    assert summary.total_class_count == 3
    assert summary.image_count_per_class == (
        CountRecord("Tomato___Bacterial_spot", 1),
        CountRecord("Tomato___healthy", 1),
        CountRecord("Tomato___Late_blight", 0),
    )
    assert summary.extension_counts == (
        CountRecord(".jpg", 1),
        CountRecord(".png", 1),
    )
    assert summary.smallest_class_size == 0
    assert summary.largest_class_size == 1
    assert summary.class_imbalance_ratio is None


def _create_image(path: Path, size: tuple[int, int], mode: str) -> None:
    image = Image.new(mode, size)
    image.save(path)

