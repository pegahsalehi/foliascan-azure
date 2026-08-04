"""Local baseline model training CLI."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from foliascan.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    EpochHistory,
    TrainingArtifactError,
    prepare_output_dir,
    reset_managed_artifacts,
    save_checkpoint,
    write_history,
)
from foliascan.training.config import (
    DeviceName,
    TrainingConfig,
    TrainingConfigError,
    load_training_config,
    training_config_with_overrides,
)
from foliascan.training.dataloaders import (
    DataLoaderError,
    create_train_validation_dataloaders,
)
from foliascan.training.dataset import (
    TrainingDataError,
    build_class_mapping,
    read_training_manifest,
)
from foliascan.training.engine import (
    TrainingEngineError,
    evaluate_one_epoch,
    train_one_epoch,
)
from foliascan.training.model import ModelFactoryError, create_model
from foliascan.training.reproducibility import (
    DeviceResolutionError,
    resolve_device,
    seed_everything,
)


class TrainingRunError(ValueError):
    """Raised when a local training run cannot complete."""


@dataclass(frozen=True, slots=True)
class TrainingSummary:
    """Summary returned by a completed local training run."""

    completed_epochs: int
    best_epoch: int
    best_validation_loss: float
    best_validation_accuracy: float
    output_dir: Path
    device: str
    num_classes: int
    early_stopped: bool
    max_train_batches: int | None = None
    max_validation_batches: int | None = None


def run_training(
    *,
    manifest_path: Path,
    data_dir: Path,
    config_path: Path,
    output_dir_override: Path | None = None,
    device_override: DeviceName | None = None,
    epochs_override: int | None = None,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    overwrite: bool = False,
    on_epoch: Callable[[EpochHistory], None] | None = None,
) -> TrainingSummary:
    """Train the local baseline model using train and validation splits only."""

    _validate_optional_batch_limit("max_train_batches", max_train_batches)
    _validate_optional_batch_limit("max_validation_batches", max_validation_batches)
    config = training_config_with_overrides(
        load_training_config(config_path),
        output_dir=output_dir_override,
        device=device_override,
        epochs=epochs_override,
    )
    records = read_training_manifest(manifest_path)
    class_mapping = build_class_mapping(records)
    dataloaders = create_train_validation_dataloaders(
        records,
        data_dir,
        class_mapping,
        config,
    )
    device = resolve_device(config.device)
    prepare_output_dir(config.output_dir, overwrite=overwrite)
    if overwrite:
        reset_managed_artifacts(config.output_dir)

    seed_everything(config.random_seed)
    model = create_model(
        model_name=config.model_name,
        num_classes=class_mapping.num_classes,
        pretrained=config.pretrained,
        freeze_backbone=config.freeze_backbone,
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)

    history: list[EpochHistory] = []
    best_epoch = 0
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    epochs_without_improvement = 0
    early_stopped = False

    for epoch in range(1, config.epochs + 1):
        start_time = time.perf_counter()
        train_metrics = train_one_epoch(
            model=model,
            dataloader=dataloaders.train,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            max_batches=max_train_batches,
        )
        validation_metrics = evaluate_one_epoch(
            model=model,
            dataloader=dataloaders.validation,
            loss_fn=loss_fn,
            device=device,
            max_batches=max_validation_batches,
        )
        elapsed_seconds = time.perf_counter() - start_time

        improved = validation_metrics.average_loss < best_validation_loss
        if improved:
            best_epoch = epoch
            best_validation_loss = validation_metrics.average_loss
            best_validation_accuracy = validation_metrics.accuracy
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_checkpoint(
            output_path=config.output_dir / LAST_CHECKPOINT_FILENAME,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            class_mapping=class_mapping,
            config=config,
            train_metrics=train_metrics,
            validation_metrics=validation_metrics,
            best_validation_loss=best_validation_loss,
        )
        if improved:
            save_checkpoint(
                output_path=config.output_dir / BEST_CHECKPOINT_FILENAME,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                class_mapping=class_mapping,
                config=config,
                train_metrics=train_metrics,
                validation_metrics=validation_metrics,
                best_validation_loss=best_validation_loss,
            )

        history_record = EpochHistory(
            epoch=epoch,
            train_loss=train_metrics.average_loss,
            train_accuracy=train_metrics.accuracy,
            validation_loss=validation_metrics.average_loss,
            validation_accuracy=validation_metrics.accuracy,
            learning_rate=current_learning_rate(optimizer),
            elapsed_seconds=elapsed_seconds,
        )
        history.append(history_record)
        write_history(tuple(history), config.output_dir)
        if on_epoch is not None:
            on_epoch(history_record)

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            early_stopped = True
            break

    if not history:
        msg = "Training did not complete any epochs."
        raise TrainingRunError(msg)

    return TrainingSummary(
        completed_epochs=len(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        best_validation_accuracy=best_validation_accuracy,
        output_dir=config.output_dir,
        device=str(device),
        num_classes=class_mapping.num_classes,
        early_stopped=early_stopped,
        max_train_batches=max_train_batches,
        max_validation_batches=max_validation_batches,
    )


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Create an optimizer for trainable model parameters."""

    if config.optimizer_name != "adamw":
        msg = f"Unsupported optimizer_name: {config.optimizer_name}"
        raise TrainingRunError(msg)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        msg = "Cannot create optimizer because the model has no trainable parameters."
        raise TrainingRunError(msg)

    return torch.optim.AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Return the current learning rate from the first optimizer group."""

    if not optimizer.param_groups:
        msg = "Optimizer has no parameter groups."
        raise TrainingRunError(msg)
    learning_rate = optimizer.param_groups[0].get("lr")
    if not isinstance(learning_rate, int | float):
        msg = "Optimizer learning rate is not numeric."
        raise TrainingRunError(msg)
    return float(learning_rate)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local training CLI and return a process exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config_preview = training_config_with_overrides(
            load_training_config(_namespace_path(args, "config")),
            output_dir=_namespace_optional_path(args, "output_dir"),
            device=_namespace_optional_device(args, "device"),
            epochs=_namespace_optional_int(args, "epochs"),
        )
        max_train_batches = _namespace_optional_int(args, "max_train_batches")
        max_validation_batches = _namespace_optional_int(
            args,
            "max_validation_batches",
        )
        _print_config_summary(
            config_preview,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
        )
        summary = run_training(
            manifest_path=_namespace_path(args, "manifest"),
            data_dir=_namespace_path(args, "data_dir"),
            config_path=_namespace_path(args, "config"),
            output_dir_override=_namespace_optional_path(args, "output_dir"),
            device_override=_namespace_optional_device(args, "device"),
            epochs_override=_namespace_optional_int(args, "epochs"),
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            overwrite=_namespace_bool(args, "overwrite"),
            on_epoch=_print_epoch,
        )
    except (
        DataLoaderError,
        DeviceResolutionError,
        FileExistsError,
        FileNotFoundError,
        ModelFactoryError,
        OSError,
        TrainingArtifactError,
        TrainingConfigError,
        TrainingDataError,
        TrainingEngineError,
        TrainingRunError,
        ValueError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_final_summary(summary)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m foliascan.training.train",
        description="Train the local ResNet18 FoliaScan baseline.",
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
        "--output-dir",
        type=Path,
        help="Override the training output directory from the config.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        help="Override the configured device.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Override the configured epoch count.",
    )
    parser.add_argument(
        "--max-train-batches",
        type=_positive_int,
        help="Optional smoke-test limit for train batches processed per epoch.",
    )
    parser.add_argument(
        "--max-validation-batches",
        type=_positive_int,
        help="Optional smoke-test limit for validation batches processed per epoch.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty training output directory.",
    )
    return parser


def _print_config_summary(
    config: TrainingConfig,
    *,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
) -> None:
    print("FoliaScan local training")
    print(f"model: {config.model_name}")
    print(f"epochs: {config.epochs}")
    print(f"batch_size: {config.batch_size}")
    print(f"optimizer: {config.optimizer_name}")
    print(f"learning_rate: {config.learning_rate}")
    print(f"weight_decay: {config.weight_decay}")
    print(f"device: {config.device}")
    print(f"output_dir: {config.output_dir}")
    print(f"max_train_batches: {_format_optional_limit(max_train_batches)}")
    print(
        "max_validation_batches: "
        f"{_format_optional_limit(max_validation_batches)}"
    )


def _print_epoch(record: EpochHistory) -> None:
    print(
        f"epoch {record.epoch}: "
        f"train_loss={record.train_loss:.4f} "
        f"train_acc={record.train_accuracy:.4f} "
        f"val_loss={record.validation_loss:.4f} "
        f"val_acc={record.validation_accuracy:.4f}"
    )


def _print_final_summary(summary: TrainingSummary) -> None:
    print("Training complete")
    print(f"completed_epochs: {summary.completed_epochs}")
    print(f"best_epoch: {summary.best_epoch}")
    print(f"best_validation_loss: {summary.best_validation_loss:.6f}")
    print(f"best_validation_accuracy: {summary.best_validation_accuracy:.6f}")
    print(f"output_dir: {summary.output_dir}")
    print(f"device: {summary.device}")
    print(f"classes: {summary.num_classes}")
    print(f"early_stopped: {summary.early_stopped}")
    print(f"max_train_batches: {_format_optional_limit(summary.max_train_batches)}")
    print(
        "max_validation_batches: "
        f"{_format_optional_limit(summary.max_validation_batches)}"
    )


def _namespace_path(args: argparse.Namespace, name: str) -> Path:
    value = getattr(args, name)
    if isinstance(value, Path):
        return value
    msg = f"Expected path argument for {name}."
    raise TypeError(msg)


def _namespace_optional_path(args: argparse.Namespace, name: str) -> Path | None:
    value = getattr(args, name)
    if value is None or isinstance(value, Path):
        return value
    msg = f"Expected optional path argument for {name}."
    raise TypeError(msg)


def _namespace_optional_device(
    args: argparse.Namespace,
    name: str,
) -> DeviceName | None:
    value = getattr(args, name)
    if value is None:
        return None
    if value in {"auto", "cpu", "cuda"}:
        return cast(DeviceName, value)
    msg = f"Expected device argument for {name}."
    raise TypeError(msg)


def _namespace_optional_int(args: argparse.Namespace, name: str) -> int | None:
    value = getattr(args, name)
    if value is None or isinstance(value, int):
        return value
    msg = f"Expected optional integer argument for {name}."
    raise TypeError(msg)


def _namespace_bool(args: argparse.Namespace, name: str) -> bool:
    value = getattr(args, name)
    if isinstance(value, bool):
        return value
    msg = f"Expected boolean argument for {name}."
    raise TypeError(msg)


def _positive_int(value: str) -> int:
    try:
        parsed_value = int(value)
    except ValueError as exc:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg) from exc
    if parsed_value <= 0:
        msg = "must be a positive integer"
        raise argparse.ArgumentTypeError(msg)
    return parsed_value


def _validate_optional_batch_limit(name: str, value: int | None) -> None:
    if value is None:
        return
    if value <= 0:
        msg = f"{name} must be a positive integer when supplied."
        raise TrainingRunError(msg)


def _format_optional_limit(value: int | None) -> str:
    return "all" if value is None else str(value)


if __name__ == "__main__":
    raise SystemExit(main())
