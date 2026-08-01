"""Training history and checkpoint artifacts."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from foliascan.training.config import TrainingConfig
from foliascan.training.dataset import ClassMapping
from foliascan.training.engine import EpochMetrics

HISTORY_COLUMNS: tuple[str, ...] = (
    "epoch",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "learning_rate",
    "elapsed_seconds",
)
BEST_CHECKPOINT_FILENAME = "best_model.pt"
LAST_CHECKPOINT_FILENAME = "last_model.pt"
HISTORY_CSV_FILENAME = "history.csv"
HISTORY_JSON_FILENAME = "history.json"


class TrainingArtifactError(ValueError):
    """Raised when training artifacts cannot be written safely."""


@dataclass(frozen=True, slots=True)
class EpochHistory:
    """Immutable per-epoch training history row."""

    epoch: int
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    learning_rate: float
    elapsed_seconds: float


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Create or validate a training output directory."""

    if output_dir.exists() and not output_dir.is_dir():
        msg = f"Training output path is not a directory: {output_dir}"
        raise TrainingArtifactError(msg)
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        msg = (
            "Training output directory is not empty; use --overwrite to write "
            f"into it: {output_dir}"
        )
        raise TrainingArtifactError(msg)
    output_dir.mkdir(parents=True, exist_ok=True)


def reset_managed_artifacts(output_dir: Path) -> None:
    """Remove artifacts managed by this training run when overwrite is enabled."""

    for filename in (
        BEST_CHECKPOINT_FILENAME,
        LAST_CHECKPOINT_FILENAME,
        HISTORY_CSV_FILENAME,
        HISTORY_JSON_FILENAME,
    ):
        artifact_path = output_dir / filename
        if artifact_path.exists():
            if artifact_path.is_dir():
                shutil.rmtree(artifact_path)
            else:
                artifact_path.unlink()


def write_history(
    history: tuple[EpochHistory, ...],
    output_dir: Path,
) -> None:
    """Write stable UTF-8 CSV and JSON history artifacts."""

    csv_path = output_dir / HISTORY_CSV_FILENAME
    with csv_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for record in history:
            writer.writerow(_history_csv_row(record))

    json_path = output_dir / HISTORY_JSON_FILENAME
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump([asdict(record) for record in history], json_file, indent=2)
        json_file.write("\n")


def save_checkpoint(
    *,
    output_path: Path,
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    class_mapping: ClassMapping,
    config: TrainingConfig,
    train_metrics: EpochMetrics,
    validation_metrics: EpochMetrics,
    best_validation_loss: float,
) -> None:
    """Save one model checkpoint."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_name": config.model_name,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "class_to_index": dict(class_mapping.class_to_index),
        "training_config": _config_dict(config),
        "train_metrics": asdict(train_metrics),
        "validation_metrics": asdict(validation_metrics),
        "best_validation_loss": best_validation_loss,
        "random_seed": config.random_seed,
    }
    torch.save(checkpoint, output_path)


def _history_csv_row(record: EpochHistory) -> dict[str, str]:
    return {
        "epoch": str(record.epoch),
        "train_loss": f"{record.train_loss:.12g}",
        "train_accuracy": f"{record.train_accuracy:.12g}",
        "validation_loss": f"{record.validation_loss:.12g}",
        "validation_accuracy": f"{record.validation_accuracy:.12g}",
        "learning_rate": f"{record.learning_rate:.12g}",
        "elapsed_seconds": f"{record.elapsed_seconds:.12g}",
    }


def _config_dict(config: TrainingConfig) -> dict[str, object]:
    config_values = asdict(config)
    config_values["output_dir"] = config.output_dir.as_posix()
    return config_values

