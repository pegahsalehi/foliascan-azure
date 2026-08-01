"""PyTorch DataLoader creation for manifest-driven training data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from foliascan.training.config import TrainingConfig
from foliascan.training.dataset import (
    ClassMapping,
    ManifestImageDataset,
    ManifestRecord,
    SplitName,
    records_for_split,
)
from foliascan.training.transforms import create_eval_transform, create_train_transform


class DataLoaderError(ValueError):
    """Raised when a requested training DataLoader cannot be created."""


@dataclass(frozen=True, slots=True)
class SplitDataLoaders:
    """Train, validation, and test DataLoaders."""

    train: DataLoader[tuple[Tensor, int]]
    validation: DataLoader[tuple[Tensor, int]]
    test: DataLoader[tuple[Tensor, int]]


def create_dataloaders(
    records: tuple[ManifestRecord, ...],
    data_dir: Path,
    class_mapping: ClassMapping,
    config: TrainingConfig,
) -> SplitDataLoaders:
    """Create train, validation, and test DataLoaders from manifest records."""

    return SplitDataLoaders(
        train=create_dataloader(
            records=records,
            data_dir=data_dir,
            class_mapping=class_mapping,
            split="train",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_seed=config.random_seed,
            image_size=config.image_size,
        ),
        validation=create_dataloader(
            records=records,
            data_dir=data_dir,
            class_mapping=class_mapping,
            split="validation",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_seed=config.random_seed,
            image_size=config.image_size,
        ),
        test=create_dataloader(
            records=records,
            data_dir=data_dir,
            class_mapping=class_mapping,
            split="test",
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            random_seed=config.random_seed,
            image_size=config.image_size,
        ),
    )


def create_dataloader(
    *,
    records: tuple[ManifestRecord, ...],
    data_dir: Path,
    class_mapping: ClassMapping,
    split: SplitName,
    batch_size: int,
    num_workers: int,
    random_seed: int,
    image_size: int,
) -> DataLoader[tuple[Tensor, int]]:
    """Create a DataLoader for one manifest split."""

    _validate_loader_options(batch_size, num_workers, random_seed)
    split_records = records_for_split(records, split)
    if not split_records:
        msg = f"Training manifest contains no records for split: {split}"
        raise DataLoaderError(msg)

    image_transform = (
        create_train_transform(image_size)
        if split == "train"
        else create_eval_transform(image_size)
    )
    dataset = ManifestImageDataset(
        split_records,
        data_dir,
        class_mapping,
        image_transform,
    )
    generator = torch.Generator()
    generator.manual_seed(random_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        generator=generator,
    )


def loader_for_split(
    dataloaders: SplitDataLoaders,
    split: SplitName,
) -> DataLoader[tuple[Tensor, int]]:
    """Select one DataLoader by split name."""

    if split == "train":
        return dataloaders.train
    if split == "validation":
        return dataloaders.validation
    if split == "test":
        return dataloaders.test
    msg = f"Unsupported FoliaScan split: {split}"
    raise DataLoaderError(msg)


def dataset_from_loader(
    dataloader: DataLoader[tuple[Tensor, int]],
) -> ManifestImageDataset:
    """Return the typed image dataset backing a DataLoader."""

    return cast(ManifestImageDataset, dataloader.dataset)


def _validate_loader_options(
    batch_size: int,
    num_workers: int,
    random_seed: int,
) -> None:
    if batch_size <= 0:
        msg = "batch_size must be greater than zero."
        raise DataLoaderError(msg)
    if num_workers < 0:
        msg = "num_workers must not be negative."
        raise DataLoaderError(msg)
    if random_seed < 0:
        msg = "random_seed must not be negative."
        raise DataLoaderError(msg)
