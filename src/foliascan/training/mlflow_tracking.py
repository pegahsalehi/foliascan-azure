"""Opt-in MLflow tracking helpers for FoliaScan training."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final, Protocol, cast

from foliascan.training.checkpoints import (
    HISTORY_CSV_FILENAME,
    HISTORY_JSON_FILENAME,
    EpochHistory,
)
from foliascan.training.config import DeviceName, TrainingConfig

MLFLOW_RUN_ID_ENVIRONMENT_VARIABLE: Final[str] = "MLFLOW_RUN_ID"


class MlflowTrackingError(ValueError):
    """Raised when explicit MLflow tracking cannot complete."""


class MlflowModule(Protocol):
    """Subset of the MLflow fluent API used by FoliaScan."""

    def active_run(self) -> object | None:
        """Return the currently active MLflow run, if any."""

    def log_params(self, params: Mapping[str, object]) -> None:
        """Log run parameters."""

    def log_metrics(
        self,
        metrics: Mapping[str, float],
        step: int | None = None,
    ) -> None:
        """Log run metrics."""

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        """Log one local artifact."""


MLFLOW: MlflowModule | None = None


class MlflowRunTracker:
    """Explicit MLflow logger that uses the existing active run context."""

    def __init__(self, mlflow_module: MlflowModule) -> None:
        self._mlflow = mlflow_module

    def log_training_parameters(
        self,
        *,
        config: TrainingConfig,
        requested_device: DeviceName,
        max_train_batches: int | None,
        max_validation_batches: int | None,
    ) -> None:
        """Log stable training parameters once."""

        params: dict[str, object] = {
            "model_name": config.model_name,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "optimizer_name": config.optimizer_name,
            "weight_decay": config.weight_decay,
            "image_size": config.image_size,
            "pretrained": config.pretrained,
            "freeze_backbone": config.freeze_backbone,
            "random_seed": config.random_seed,
            "requested_device": requested_device,
            "max_train_batches": _optional_limit_value(max_train_batches),
            "max_validation_batches": _optional_limit_value(max_validation_batches),
        }
        _run_mlflow_operation(
            "logging training parameters",
            lambda: self._mlflow.log_params(params),
        )

    def log_epoch_metrics(self, history: EpochHistory) -> None:
        """Log one epoch of training and validation metrics."""

        metrics = {
            "train_loss": history.train_loss,
            "train_accuracy": history.train_accuracy,
            "validation_loss": history.validation_loss,
            "validation_accuracy": history.validation_accuracy,
            "learning_rate": history.learning_rate,
            "elapsed_seconds": history.elapsed_seconds,
        }
        _run_mlflow_operation(
            f"logging metrics for epoch {history.epoch}",
            lambda: self._mlflow.log_metrics(metrics, step=history.epoch),
        )

    def log_training_summary(
        self,
        *,
        completed_epochs: int,
        best_epoch: int,
        best_validation_loss: float,
        best_validation_accuracy: float,
        early_stopped: bool,
    ) -> None:
        """Log final successful training summary metrics."""

        metrics = {
            "completed_epochs": float(completed_epochs),
            "best_epoch": float(best_epoch),
            "best_validation_loss": best_validation_loss,
            "best_validation_accuracy": best_validation_accuracy,
            "early_stopped": float(int(early_stopped)),
        }
        _run_mlflow_operation(
            "logging final training summary",
            lambda: self._mlflow.log_metrics(metrics),
        )

    def log_lightweight_artifacts(
        self,
        *,
        config_path: Path,
        output_dir: Path,
    ) -> None:
        """Log lightweight training artifacts without duplicating checkpoints."""

        artifacts = (
            (config_path, "config"),
            (output_dir / HISTORY_CSV_FILENAME, "history"),
            (output_dir / HISTORY_JSON_FILENAME, "history"),
        )
        for artifact_path, mlflow_artifact_path in artifacts:
            _ensure_artifact_exists(artifact_path)
            self._log_artifact(artifact_path, mlflow_artifact_path)

    def _log_artifact(self, path: Path, artifact_path: str) -> None:
        _run_mlflow_operation(
            f"logging artifact {path.name}",
            lambda: self._mlflow.log_artifact(
                path.as_posix(),
                artifact_path=artifact_path,
            ),
        )


def create_mlflow_tracker(
    environ: Mapping[str, str] | None = None,
) -> MlflowRunTracker:
    """Create an MLflow tracker using the current run context."""

    mlflow_module = _mlflow_module()
    environment = os.environ if environ is None else environ
    try:
        active_run = mlflow_module.active_run()
    except Exception as exc:
        msg = "MLflow tracking failed while checking the active run context."
        raise MlflowTrackingError(msg) from exc

    if active_run is None and not environment.get(MLFLOW_RUN_ID_ENVIRONMENT_VARIABLE):
        msg = (
            "MLflow tracking was enabled, but no active MLflow run context was "
            "found. Run inside an Azure ML command job or start an MLflow run "
            "before enabling tracking."
        )
        raise MlflowTrackingError(msg)

    return MlflowRunTracker(mlflow_module)


def _mlflow_module() -> MlflowModule:
    if MLFLOW is not None:
        return MLFLOW
    try:
        mlflow_module = importlib.import_module("mlflow")
    except ImportError as exc:
        msg = (
            "MLflow dependencies are not installed in this Poetry environment. "
            "Run 'poetry install' before enabling MLflow tracking."
        )
        raise MlflowTrackingError(msg) from exc
    return cast(MlflowModule, mlflow_module)


def _run_mlflow_operation(operation: str, action: Callable[[], None]) -> None:
    try:
        action()
    except Exception as exc:
        msg = f"MLflow tracking failed while {operation}: {exc}"
        raise MlflowTrackingError(msg) from exc


def _ensure_artifact_exists(path: Path) -> None:
    if not path.exists() or not path.is_file():
        msg = f"MLflow artifact does not exist: {path}"
        raise MlflowTrackingError(msg)


def _optional_limit_value(value: int | None) -> int | str:
    return "all" if value is None else value
