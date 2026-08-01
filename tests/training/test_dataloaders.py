from pathlib import Path

from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from foliascan.training.config import TrainingConfig
from foliascan.training.dataloaders import (
    create_dataloader,
    create_dataloaders,
    dataset_from_loader,
)
from foliascan.training.dataset import ManifestRecord, build_class_mapping


def test_create_dataloaders_filters_splits_and_shuffles_train_only(
    tmp_path: Path,
) -> None:
    records = _records()
    _write_images(tmp_path, records)
    mapping = build_class_mapping(records)
    config = _config(batch_size=2, image_size=16, random_seed=123)

    dataloaders = create_dataloaders(records, tmp_path, mapping, config)

    assert len(dataset_from_loader(dataloaders.train).records) == 2
    assert len(dataset_from_loader(dataloaders.validation).records) == 1
    assert len(dataset_from_loader(dataloaders.test).records) == 1
    assert isinstance(dataloaders.train.sampler, RandomSampler)
    assert isinstance(dataloaders.validation.sampler, SequentialSampler)
    assert isinstance(dataloaders.test.sampler, SequentialSampler)


def test_create_dataloader_uses_deterministic_train_generator(tmp_path: Path) -> None:
    records = tuple(
        ManifestRecord(
            Path(f"class_{index}/image.jpg"),
            f"class_{index}",
            "train",
            f"leaf_{index}",
            "train",
        )
        for index in range(6)
    )
    _write_images(tmp_path, records)
    mapping = build_class_mapping(records)

    first_loader = create_dataloader(
        records=records,
        data_dir=tmp_path,
        class_mapping=mapping,
        split="train",
        batch_size=1,
        num_workers=0,
        random_seed=77,
        image_size=16,
    )
    second_loader = create_dataloader(
        records=records,
        data_dir=tmp_path,
        class_mapping=mapping,
        split="train",
        batch_size=1,
        num_workers=0,
        random_seed=77,
        image_size=16,
    )

    assert _target_order(first_loader) == _target_order(second_loader)


def test_create_dataloader_rejects_empty_split(tmp_path: Path) -> None:
    records = (
        ManifestRecord(Path("class_a/image.jpg"), "class_a", "train", "leaf", "train"),
    )
    _write_images(tmp_path, records)
    mapping = build_class_mapping(records)

    try:
        create_dataloader(
            records=records,
            data_dir=tmp_path,
            class_mapping=mapping,
            split="validation",
            batch_size=1,
            num_workers=0,
            random_seed=42,
            image_size=16,
        )
    except ValueError as exc:
        assert "validation" in str(exc)
    else:
        raise AssertionError("Expected empty validation split to fail.")


def _target_order(loader: object) -> list[int]:
    return [
        int(targets.item())
        for _, targets in loader  # type: ignore[union-attr]
    ]


def _config(batch_size: int, image_size: int, random_seed: int) -> TrainingConfig:
    return TrainingConfig(
        random_seed=random_seed,
        image_size=image_size,
        batch_size=batch_size,
        learning_rate=0.001,
        epochs=1,
        model_name="resnet18",
        num_workers=0,
        pretrained=False,
        freeze_backbone=False,
    )


def _records() -> tuple[ManifestRecord, ...]:
    return (
        ManifestRecord(Path("class_a/train_a.jpg"), "class_a", "train", "a1", "train"),
        ManifestRecord(Path("class_b/train_b.jpg"), "class_b", "train", "b1", "train"),
        ManifestRecord(
            Path("class_a/validation_a.jpg"),
            "class_a",
            "validation",
            "a2",
            "train",
        ),
        ManifestRecord(Path("class_b/test_b.jpg"), "class_b", "test", "b2", "test"),
    )


def _write_images(root: Path, records: tuple[ManifestRecord, ...]) -> None:
    for record in records:
        image_path = root / record.relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), color=(10, 20, 30)).save(image_path)
