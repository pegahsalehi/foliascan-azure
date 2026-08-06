from pathlib import Path

import pytest

from foliascan.training import mlflow_tracking
from foliascan.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    HISTORY_CSV_FILENAME,
    HISTORY_JSON_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    EpochHistory,
)
from foliascan.training.config import TrainingConfig
from foliascan.training.mlflow_tracking import (
    MLFLOW_RUN_ID_ENVIRONMENT_VARIABLE,
    MlflowRunTracker,
    MlflowTrackingError,
    create_mlflow_tracker,
)


class FakeMlflow:
    def __init__(
        self,
        *,
        active_run: object | None = object(),
        fail_on: str | None = None,
    ) -> None:
        self.active_run_result = active_run
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.params: list[dict[str, object]] = []
        self.metrics: list[tuple[dict[str, float], int | None]] = []
        self.artifacts: list[tuple[str, str | None]] = []

    def active_run(self) -> object | None:
        self.calls.append("active_run")
        return self.active_run_result

    def log_params(self, params: dict[str, object]) -> None:
        self.calls.append("log_params")
        if self.fail_on == "log_params":
            raise RuntimeError("parameter sink unavailable")
        self.params.append(dict(params))

    def log_metrics(
        self,
        metrics: dict[str, float],
        step: int | None = None,
    ) -> None:
        self.calls.append("log_metrics")
        if self.fail_on == "log_metrics":
            raise RuntimeError("metric sink unavailable")
        self.metrics.append((dict(metrics), step))

    def log_artifact(
        self,
        local_path: str,
        artifact_path: str | None = None,
    ) -> None:
        self.calls.append("log_artifact")
        if self.fail_on == "log_artifact":
            raise RuntimeError("artifact sink unavailable")
        self.artifacts.append((local_path, artifact_path))

    def start_run(self) -> None:
        self.calls.append("start_run")
        raise AssertionError("training tracking must not start a second run")

    def set_experiment(self, name: str) -> None:
        self.calls.append(f"set_experiment:{name}")
        raise AssertionError("Azure YAML controls the experiment name")

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(f"set_tracking_uri:{uri}")
        raise AssertionError("Azure ML supplies the tracking URI")

    def autolog(self) -> None:
        self.calls.append("autolog")
        raise AssertionError("autologging is intentionally not used")


def test_create_tracker_uses_existing_active_run_without_creating_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow(active_run=object())
    monkeypatch.setattr(mlflow_tracking, "MLFLOW", fake_mlflow)

    tracker = create_mlflow_tracker(environ={})

    assert isinstance(tracker, MlflowRunTracker)
    assert fake_mlflow.calls == ["active_run"]


def test_create_tracker_accepts_azure_run_id_environment_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow(active_run=None)
    monkeypatch.setattr(mlflow_tracking, "MLFLOW", fake_mlflow)

    tracker = create_mlflow_tracker(
        environ={MLFLOW_RUN_ID_ENVIRONMENT_VARIABLE: "azure-job-run"}
    )

    assert isinstance(tracker, MlflowRunTracker)
    assert fake_mlflow.calls == ["active_run"]


def test_create_tracker_rejects_missing_run_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mlflow = FakeMlflow(active_run=None)
    monkeypatch.setattr(mlflow_tracking, "MLFLOW", fake_mlflow)

    with pytest.raises(MlflowTrackingError, match="no active MLflow run context"):
        create_mlflow_tracker(environ={})


def test_tracker_logs_parameters_metrics_summary_and_lightweight_artifacts(
    tmp_path: Path,
) -> None:
    fake_mlflow = FakeMlflow()
    tracker = MlflowRunTracker(fake_mlflow)
    config = _config(tmp_path / "training-output")
    config_path = tmp_path / "training.yaml"
    config_path.write_text("model_name: resnet18\n", encoding="utf-8")
    output_dir = config.output_dir
    output_dir.mkdir()
    (output_dir / HISTORY_CSV_FILENAME).write_text(
        "epoch,train_loss\n",
        encoding="utf-8",
    )
    (output_dir / HISTORY_JSON_FILENAME).write_text("[]\n", encoding="utf-8")
    (output_dir / BEST_CHECKPOINT_FILENAME).write_text(
        "large checkpoint",
        encoding="utf-8",
    )
    (output_dir / LAST_CHECKPOINT_FILENAME).write_text(
        "large checkpoint",
        encoding="utf-8",
    )

    tracker.log_training_parameters(
        config=config,
        requested_device="cuda",
        max_train_batches=2,
        max_validation_batches=None,
    )
    tracker.log_epoch_metrics(
        EpochHistory(
            epoch=3,
            train_loss=0.7,
            train_accuracy=0.8,
            validation_loss=0.9,
            validation_accuracy=0.6,
            learning_rate=0.001,
            elapsed_seconds=12.5,
        )
    )
    tracker.log_training_summary(
        completed_epochs=3,
        best_epoch=2,
        best_validation_loss=0.5,
        best_validation_accuracy=0.85,
        early_stopped=True,
    )
    tracker.log_lightweight_artifacts(config_path=config_path, output_dir=output_dir)

    params = fake_mlflow.params[0]
    assert params["model_name"] == "resnet18"
    assert params["epochs"] == 4
    assert params["batch_size"] == 8
    assert params["learning_rate"] == 0.01
    assert params["optimizer_name"] == "adamw"
    assert params["weight_decay"] == 0.001
    assert params["image_size"] == 32
    assert params["pretrained"] is False
    assert params["freeze_backbone"] is True
    assert params["random_seed"] == 123
    assert params["requested_device"] == "cuda"
    assert params["max_train_batches"] == 2
    assert params["max_validation_batches"] == "all"

    epoch_metrics, epoch_step = fake_mlflow.metrics[0]
    assert epoch_step == 3
    assert epoch_metrics == {
        "train_loss": 0.7,
        "train_accuracy": 0.8,
        "validation_loss": 0.9,
        "validation_accuracy": 0.6,
        "learning_rate": 0.001,
        "elapsed_seconds": 12.5,
    }
    summary_metrics, summary_step = fake_mlflow.metrics[1]
    assert summary_step is None
    assert summary_metrics["completed_epochs"] == 3.0
    assert summary_metrics["best_epoch"] == 2.0
    assert summary_metrics["best_validation_loss"] == 0.5
    assert summary_metrics["best_validation_accuracy"] == 0.85
    assert summary_metrics["early_stopped"] == 1.0

    artifact_names = {Path(path).name for path, _ in fake_mlflow.artifacts}
    assert artifact_names == {
        "training.yaml",
        HISTORY_CSV_FILENAME,
        HISTORY_JSON_FILENAME,
    }
    assert BEST_CHECKPOINT_FILENAME not in artifact_names
    assert LAST_CHECKPOINT_FILENAME not in artifact_names


def test_tracking_failures_are_reported_clearly(tmp_path: Path) -> None:
    fake_mlflow = FakeMlflow(fail_on="log_params")
    tracker = MlflowRunTracker(fake_mlflow)

    with pytest.raises(MlflowTrackingError, match="logging training parameters"):
        tracker.log_training_parameters(
            config=_config(tmp_path / "training-output"),
            requested_device="cpu",
            max_train_batches=None,
            max_validation_batches=None,
        )


def _config(output_dir: Path) -> TrainingConfig:
    return TrainingConfig(
        random_seed=123,
        image_size=32,
        batch_size=8,
        learning_rate=0.01,
        epochs=4,
        model_name="resnet18",
        num_workers=0,
        pretrained=False,
        freeze_backbone=True,
        optimizer_name="adamw",
        weight_decay=0.001,
        early_stopping_patience=2,
        device="auto",
        output_dir=output_dir,
    )
