"""Forward-pass smoke test for the local FoliaScan training foundation."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import Tensor

from foliascan.training.config import TrainingConfigError, load_training_config
from foliascan.training.dataloaders import (
    DataLoaderError,
    create_dataloaders,
    loader_for_split,
)
from foliascan.training.dataset import (
    SplitName,
    TrainingDataError,
    build_class_mapping,
    read_training_manifest,
)
from foliascan.training.model import ModelFactoryError, create_model


class SmokeTestError(ValueError):
    """Raised when the forward-pass smoke test cannot complete."""


@dataclass(frozen=True, slots=True)
class SmokeTestSummary:
    """Concise smoke-test result summary."""

    split: SplitName
    batch_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    num_classes: int
    output_shape: tuple[int, ...]
    device: str


def run_smoke_test(
    *,
    manifest_path: Path,
    data_dir: Path,
    config_path: Path,
    split: SplitName,
    device_name: str = "cpu",
) -> SmokeTestSummary:
    """Run one model forward pass from one manifest-driven DataLoader batch."""

    config = load_training_config(config_path)
    records = read_training_manifest(manifest_path)
    class_mapping = build_class_mapping(records)
    dataloaders = create_dataloaders(records, data_dir, class_mapping, config)
    dataloader = loader_for_split(dataloaders, split)
    device = _resolve_device(device_name)

    try:
        batch = next(iter(dataloader))
    except StopIteration as exc:
        msg = f"Training manifest contains no records for split: {split}"
        raise SmokeTestError(msg) from exc

    images = cast(Tensor, batch[0]).to(device)
    targets = cast(Tensor, batch[1]).to(device)
    model = create_model(
        model_name=config.model_name,
        num_classes=class_mapping.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(images)

    expected_output_shape = (images.shape[0], class_mapping.num_classes)
    if tuple(outputs.shape) != expected_output_shape:
        msg = (
            "Smoke-test output shape mismatch: "
            f"expected {expected_output_shape}, got {tuple(outputs.shape)}"
        )
        raise SmokeTestError(msg)

    return SmokeTestSummary(
        split=split,
        batch_shape=_tensor_shape(images),
        target_shape=_tensor_shape(targets),
        num_classes=class_mapping.num_classes,
        output_shape=_tensor_shape(outputs),
        device=str(device),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the smoke-test CLI and return a process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        summary = run_smoke_test(
            manifest_path=_namespace_path(args, "manifest"),
            data_dir=_namespace_path(args, "data_dir"),
            config_path=_namespace_path(args, "config"),
            split=_namespace_split(args, "split"),
            device_name=_namespace_string(args, "device"),
        )
    except (
        DataLoaderError,
        FileNotFoundError,
        ModelFactoryError,
        OSError,
        SmokeTestError,
        TrainingConfigError,
        TrainingDataError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_summary(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.training.smoke_test",
        description="Run one manifest-driven model forward-pass smoke test.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="CSV FoliaScan dataset manifest path.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Exported image dataset root.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Training YAML configuration path.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="train",
        help="Manifest split to use for the smoke-test batch.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to use. Defaults to CPU.",
    )
    return parser


def _print_summary(summary: SmokeTestSummary) -> None:
    print("FoliaScan training smoke test complete")
    print(f"split: {summary.split}")
    print(f"batch tensor shape: {summary.batch_shape}")
    print(f"target shape: {summary.target_shape}")
    print(f"classes: {summary.num_classes}")
    print(f"model output shape: {summary.output_shape}")
    print(f"device: {summary.device}")


def _resolve_device(device_name: str) -> torch.device:
    try:
        device = torch.device(device_name)
    except RuntimeError as exc:
        msg = f"Unsupported torch device: {device_name}"
        raise SmokeTestError(msg) from exc

    if device.type == "cuda" and not torch.cuda.is_available():
        msg = "CUDA was requested but is not available."
        raise SmokeTestError(msg)
    return device


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    msg = f"Expected path argument for {name}."
    raise TypeError(msg)


def _namespace_split(args: argparse.Namespace, name: str) -> SplitName:
    value = getattr(args, name)
    if value in {"train", "validation", "test"}:
        return cast(SplitName, value)
    msg = f"Expected split argument for {name}."
    raise TypeError(msg)


def _namespace_string(args: argparse.Namespace, name: str) -> str:
    value = getattr(args, name)
    if isinstance(value, str):
        return value
    msg = f"Expected string argument for {name}."
    raise TypeError(msg)


def _tensor_shape(tensor: Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


if __name__ == "__main__":
    raise SystemExit(main())
