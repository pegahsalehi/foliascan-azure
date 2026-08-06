import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from foliascan.training import train as train_module
from foliascan.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    HISTORY_CSV_FILENAME,
    HISTORY_JSON_FILENAME,
    LAST_CHECKPOINT_FILENAME,
    EpochHistory,
)
from foliascan.training.config import TrainingConfig
from foliascan.training.dataset import MANIFEST_COLUMNS
from foliascan.training.engine import EpochMetrics
from foliascan.training.train import TrainingRunError, TrainingSummary


def test_create_optimizer_uses_only_trainable_parameters() -> None:
    model = nn.Sequential(nn.Linear(2, 3), nn.Linear(3, 2))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    config = _config(Path("artifacts/training/test"))

    optimizer = train_module.create_optimizer(model, config)

    optimizer_param_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_param_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    frozen_param_ids = {
        id(parameter) for parameter in model.parameters() if not parameter.requires_grad
    }
    assert optimizer_param_ids == trainable_param_ids
    assert optimizer_param_ids.isdisjoint(frozen_param_ids)


def test_create_optimizer_rejects_unsupported_optimizer() -> None:
    config = _config(Path("artifacts/training/test"), optimizer_name="sgd")

    with pytest.raises(TrainingRunError, match="Unsupported optimizer_name"):
        train_module.create_optimizer(nn.Linear(2, 2), config)


def test_create_optimizer_rejects_model_with_no_trainable_parameters() -> None:
    model = nn.Linear(2, 2)
    for parameter in model.parameters():
        parameter.requires_grad = False

    with pytest.raises(TrainingRunError, match="no trainable"):
        train_module.create_optimizer(model, _config(Path("artifacts/training/test")))


def test_run_training_completes_tiny_run_and_does_not_use_test_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=2,
        patience=0,
    )
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)
    monkeypatch.setattr(
        train_module,
        "create_mlflow_tracker",
        _unexpected_mlflow_tracker_factory,
    )
    epoch_rows: list[int] = []

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
        overwrite=False,
        on_epoch=lambda record: epoch_rows.append(record.epoch),
    )

    assert summary.completed_epochs == 2
    assert summary.best_epoch in {1, 2}
    assert summary.output_dir == tmp_path / "artifacts" / "training"
    assert summary.device == "cpu"
    assert summary.num_classes == 2
    assert summary.early_stopped is False
    assert epoch_rows == [1, 2]
    assert (summary.output_dir / BEST_CHECKPOINT_FILENAME).exists()
    assert (summary.output_dir / LAST_CHECKPOINT_FILENAME).exists()
    assert (summary.output_dir / "history.csv").exists()
    assert (summary.output_dir / "history.json").exists()
    assert not (data_dir / "class_a" / "missing_test.jpg").exists()


def test_run_training_accepts_absolute_paths_and_existing_empty_output_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configured_output_dir = tmp_path / "configured-output"
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
        output_dir=configured_output_dir,
    )
    output_dir = (tmp_path / "azure-managed-output").resolve()
    output_dir.mkdir()
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)

    assert manifest_path.is_absolute()
    assert data_dir.is_absolute()
    assert config_path.is_absolute()
    assert output_dir.is_absolute()

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
        output_dir_override=output_dir,
    )

    assert summary.output_dir == output_dir
    _assert_managed_artifacts_under(output_dir)
    assert not configured_output_dir.exists()


def test_run_training_batch_limits_control_processed_sample_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
        train_records_per_class=2,
        validation_records_per_class=2,
        batch_size=2,
    )
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
        max_train_batches=1,
        max_validation_batches=1,
    )

    checkpoint = torch.load(
        summary.output_dir / LAST_CHECKPOINT_FILENAME,
        map_location="cpu",
    )
    assert checkpoint["train_metrics"]["sample_count"] == 2
    assert checkpoint["train_metrics"]["batch_count"] == 1
    assert checkpoint["validation_metrics"]["sample_count"] == 2
    assert checkpoint["validation_metrics"]["batch_count"] == 1
    assert summary.max_train_batches == 1
    assert summary.max_validation_batches == 1


def test_run_training_without_batch_limits_processes_all_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
        train_records_per_class=2,
        validation_records_per_class=2,
        batch_size=2,
    )
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
    )

    checkpoint = torch.load(
        summary.output_dir / LAST_CHECKPOINT_FILENAME,
        map_location="cpu",
    )
    assert checkpoint["train_metrics"]["sample_count"] == 4
    assert checkpoint["train_metrics"]["batch_count"] == 2
    assert checkpoint["validation_metrics"]["sample_count"] == 4
    assert checkpoint["validation_metrics"]["batch_count"] == 2
    assert summary.max_train_batches is None
    assert summary.max_validation_batches is None


def test_run_training_logs_to_mlflow_only_when_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
        train_records_per_class=2,
        validation_records_per_class=2,
        batch_size=2,
    )
    tracker = FakeMlflowTracker()
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)
    monkeypatch.setattr(train_module, "create_mlflow_tracker", lambda: tracker)

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
        max_train_batches=1,
        max_validation_batches=1,
        enable_mlflow=True,
    )

    assert summary.mlflow_enabled is True
    assert tracker.parameter_calls == [
        {
            "model_name": "resnet18",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "optimizer_name": "adamw",
            "weight_decay": 0.0001,
            "image_size": 16,
            "pretrained": False,
            "freeze_backbone": False,
            "random_seed": 42,
            "requested_device": "cpu",
            "max_train_batches": 1,
            "max_validation_batches": 1,
        }
    ]
    assert [record.epoch for record in tracker.epoch_metric_calls] == [1]
    assert tracker.epoch_metric_calls[0].train_loss >= 0
    assert tracker.summary_calls == [
        {
            "completed_epochs": 1,
            "best_epoch": summary.best_epoch,
            "best_validation_loss": summary.best_validation_loss,
            "best_validation_accuracy": summary.best_validation_accuracy,
            "early_stopped": False,
        }
    ]
    assert tracker.artifact_calls == [
        {
            "config_path": config_path,
            "output_dir": summary.output_dir,
        }
    ]
    _assert_managed_artifacts_under(summary.output_dir)


def test_run_training_rejects_invalid_batch_limits(tmp_path: Path) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
    )

    with pytest.raises(TrainingRunError, match="max_train_batches"):
        train_module.run_training(
            manifest_path=manifest_path,
            data_dir=data_dir,
            config_path=config_path,
            max_train_batches=0,
        )

    with pytest.raises(TrainingRunError, match="max_validation_batches"):
        train_module.run_training(
            manifest_path=manifest_path,
            data_dir=data_dir,
            config_path=config_path,
            max_validation_batches=-1,
        )


def test_run_training_stops_early_and_selects_best_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=5,
        patience=2,
    )
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)
    validation_losses = iter((1.0, 1.1, 1.2, 1.3))

    def fake_train_one_epoch(**kwargs: object) -> EpochMetrics:
        return EpochMetrics(
            average_loss=0.8,
            accuracy=0.5,
            sample_count=2,
            batch_count=1,
        )

    def fake_evaluate_one_epoch(**kwargs: object) -> EpochMetrics:
        return EpochMetrics(
            average_loss=next(validation_losses),
            accuracy=0.25,
            sample_count=2,
            batch_count=1,
        )

    monkeypatch.setattr(train_module, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_module, "evaluate_one_epoch", fake_evaluate_one_epoch)

    summary = train_module.run_training(
        manifest_path=manifest_path,
        data_dir=data_dir,
        config_path=config_path,
    )

    best_checkpoint = torch.load(
        summary.output_dir / BEST_CHECKPOINT_FILENAME,
        map_location="cpu",
    )
    last_checkpoint = torch.load(
        summary.output_dir / LAST_CHECKPOINT_FILENAME,
        map_location="cpu",
    )
    assert summary.completed_epochs == 3
    assert summary.best_epoch == 1
    assert summary.best_validation_loss == 1.0
    assert summary.early_stopped is True
    assert best_checkpoint["epoch"] == 1
    assert last_checkpoint["epoch"] == 3


def test_run_training_rejects_non_empty_output_dir_without_overwrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir, manifest_path, config_path = _write_tiny_training_inputs(
        tmp_path,
        epochs=1,
        patience=0,
    )
    output_dir = tmp_path / "artifacts" / "training"
    output_dir.mkdir(parents=True)
    (output_dir / "existing.txt").write_text("already here", encoding="utf-8")
    monkeypatch.setattr(train_module, "create_model", _tiny_model_factory)

    with pytest.raises(ValueError, match="not empty"):
        train_module.run_training(
            manifest_path=manifest_path,
            data_dir=data_dir,
            config_path=config_path,
            overwrite=False,
        )


def test_train_cli_parses_arguments_and_prints_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, epochs=4, patience=0)

    def fake_run_training(
        *,
        manifest_path: Path,
        data_dir: Path,
        config_path: Path,
        output_dir_override: Path | None = None,
        device_override: str | None = None,
        epochs_override: int | None = None,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
        enable_mlflow: bool = False,
        overwrite: bool = False,
        on_epoch: object = None,
    ) -> TrainingSummary:
        assert manifest_path == tmp_path / "dataset_manifest.csv"
        assert data_dir == tmp_path / "raw"
        assert config_path == tmp_path / "training.yaml"
        assert output_dir_override == tmp_path / "override"
        assert device_override == "cpu"
        assert epochs_override == 1
        assert max_train_batches == 3
        assert max_validation_batches == 4
        assert enable_mlflow is True
        assert overwrite is True
        return TrainingSummary(
            completed_epochs=1,
            best_epoch=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.75,
            output_dir=tmp_path / "override",
            device="cpu",
            num_classes=2,
            early_stopped=False,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            mlflow_enabled=enable_mlflow,
        )

    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    exit_status = train_module.main(
        [
            "--manifest",
            str(tmp_path / "dataset_manifest.csv"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "override"),
            "--device",
            "cpu",
            "--epochs",
            "1",
            "--max-train-batches",
            "3",
            "--max-validation-batches",
            "4",
            "--enable-mlflow",
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "FoliaScan local training" in captured.out
    assert "max_train_batches: 3" in captured.out
    assert "max_validation_batches: 4" in captured.out
    assert "Training complete" in captured.out


def test_train_cli_forwards_linux_mount_style_paths_without_project_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = _write_config(tmp_path, epochs=1, patience=0)
    seen_paths: dict[str, Path | None] = {}

    def fake_run_training(
        *,
        manifest_path: Path,
        data_dir: Path,
        config_path: Path,
        output_dir_override: Path | None = None,
        device_override: str | None = None,
        epochs_override: int | None = None,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
        enable_mlflow: bool = False,
        overwrite: bool = False,
        on_epoch: object = None,
    ) -> TrainingSummary:
        seen_paths["manifest"] = manifest_path
        seen_paths["data_dir"] = data_dir
        seen_paths["config"] = config_path
        seen_paths["output_dir"] = output_dir_override
        return TrainingSummary(
            completed_epochs=1,
            best_epoch=1,
            best_validation_loss=0.5,
            best_validation_accuracy=0.75,
            output_dir=output_dir_override or Path("unused"),
            device="cpu",
            num_classes=2,
            early_stopped=False,
            max_train_batches=max_train_batches,
            max_validation_batches=max_validation_batches,
            mlflow_enabled=enable_mlflow,
        )

    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    exit_status = train_module.main(
        [
            "--manifest",
            "/mnt/azureml/inputs/manifest/dataset_manifest.csv",
            "--data-dir",
            "/mnt/azureml/inputs/images",
            "--config",
            str(config_path),
            "--output-dir",
            "/mnt/azureml/outputs/model",
        ]
    )

    assert exit_status == 0
    manifest = seen_paths["manifest"]
    data_dir = seen_paths["data_dir"]
    output_dir = seen_paths["output_dir"]
    assert manifest is not None
    assert data_dir is not None
    assert output_dir is not None
    assert manifest.as_posix() == (
        "/mnt/azureml/inputs/manifest/dataset_manifest.csv"
    )
    assert data_dir.as_posix() == "/mnt/azureml/inputs/images"
    assert seen_paths["config"] == config_path
    assert output_dir.as_posix() == "/mnt/azureml/outputs/model"


def test_train_cli_rejects_invalid_batch_limits_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, epochs=1, patience=0)

    with pytest.raises(SystemExit) as exc_info:
        train_module.main(
            [
                "--manifest",
                str(tmp_path / "dataset_manifest.csv"),
                "--data-dir",
                str(tmp_path / "raw"),
                "--config",
                str(config_path),
                "--max-train-batches",
                "0",
            ]
        )

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert "must be a positive integer" in captured.err
    assert "Traceback" not in captured.err


def test_train_cli_reports_expected_errors_without_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = _write_config(tmp_path, epochs=1, patience=0)

    def fake_run_training(**kwargs: object) -> TrainingSummary:
        raise TrainingRunError("bad user input")

    monkeypatch.setattr(train_module, "run_training", fake_run_training)

    exit_status = train_module.main(
        [
            "--manifest",
            str(tmp_path / "dataset_manifest.csv"),
            "--data-dir",
            str(tmp_path / "raw"),
            "--config",
            str(config_path),
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 2
    assert "error: bad user input" in captured.err
    assert "Traceback" not in captured.err


class FakeMlflowTracker:
    def __init__(self) -> None:
        self.parameter_calls: list[dict[str, object]] = []
        self.epoch_metric_calls: list[EpochHistory] = []
        self.summary_calls: list[dict[str, object]] = []
        self.artifact_calls: list[dict[str, Path]] = []

    def log_training_parameters(
        self,
        *,
        config: TrainingConfig,
        requested_device: str,
        max_train_batches: int | None,
        max_validation_batches: int | None,
    ) -> None:
        self.parameter_calls.append(
            {
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
                "max_train_batches": max_train_batches,
                "max_validation_batches": max_validation_batches,
            }
        )

    def log_epoch_metrics(self, history: EpochHistory) -> None:
        self.epoch_metric_calls.append(history)

    def log_training_summary(
        self,
        *,
        completed_epochs: int,
        best_epoch: int,
        best_validation_loss: float,
        best_validation_accuracy: float,
        early_stopped: bool,
    ) -> None:
        self.summary_calls.append(
            {
                "completed_epochs": completed_epochs,
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "best_validation_accuracy": best_validation_accuracy,
                "early_stopped": early_stopped,
            }
        )

    def log_lightweight_artifacts(
        self,
        *,
        config_path: Path,
        output_dir: Path,
    ) -> None:
        self.artifact_calls.append(
            {
                "config_path": config_path,
                "output_dir": output_dir,
            }
        )


def _unexpected_mlflow_tracker_factory() -> object:
    raise AssertionError("MLflow tracking must remain disabled without the flag.")


def _tiny_model_factory(
    *,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    assert model_name == "resnet18"
    assert pretrained is False
    assert freeze_backbone is False
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, num_classes))


def _write_tiny_training_inputs(
    tmp_path: Path,
    *,
    epochs: int,
    patience: int,
    train_records_per_class: int = 1,
    validation_records_per_class: int = 1,
    batch_size: int = 2,
    output_dir: Path | None = None,
) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "raw"
    manifest_path = tmp_path / "dataset_manifest.csv"
    config_path = _write_config(
        tmp_path,
        epochs=epochs,
        patience=patience,
        batch_size=batch_size,
        output_dir=output_dir,
    )
    rows: list[tuple[str, str, str, str, str]] = []
    for class_name in ("class_a", "class_b"):
        for index in range(train_records_per_class):
            rows.append(
                (
                    f"{class_name}/train_{index}.jpg",
                    class_name,
                    "train",
                    f"{class_name}_train_leaf_{index}",
                    "train",
                )
            )
        for index in range(validation_records_per_class):
            rows.append(
                (
                    f"{class_name}/validation_{index}.jpg",
                    class_name,
                    "validation",
                    f"{class_name}_validation_leaf_{index}",
                    "train",
                )
            )
    rows.append(("class_a/missing_test.jpg", "class_a", "test", "leaf_test", "test"))
    for relative_path, _, split, _, _ in rows:
        if split == "test":
            continue
        image_path = data_dir / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (24, 24), color=(20, 80, 120)).save(image_path)

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
    return data_dir, manifest_path, config_path


def _write_config(
    tmp_path: Path,
    *,
    epochs: int,
    patience: int,
    batch_size: int = 2,
    output_dir: Path | None = None,
) -> Path:
    config_path = tmp_path / "training.yaml"
    configured_output_dir = output_dir or tmp_path / "artifacts" / "training"
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "image_size: 16",
                f"batch_size: {batch_size}",
                "learning_rate: 0.001",
                f"epochs: {epochs}",
                "model_name: resnet18",
                "num_workers: 0",
                "pretrained: false",
                "freeze_backbone: false",
                "optimizer_name: adamw",
                "weight_decay: 0.0001",
                f"early_stopping_patience: {patience}",
                "device: cpu",
                f"output_dir: {configured_output_dir.as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def _assert_managed_artifacts_under(output_dir: Path) -> None:
    for filename in (
        BEST_CHECKPOINT_FILENAME,
        LAST_CHECKPOINT_FILENAME,
        HISTORY_CSV_FILENAME,
        HISTORY_JSON_FILENAME,
    ):
        artifact_path = output_dir / filename
        assert artifact_path.exists()
        assert artifact_path.resolve().is_relative_to(output_dir.resolve())


def _config(
    output_dir: Path,
    *,
    optimizer_name: str = "adamw",
) -> TrainingConfig:
    return TrainingConfig(
        random_seed=42,
        image_size=16,
        batch_size=2,
        learning_rate=0.001,
        epochs=1,
        model_name="resnet18",
        num_workers=0,
        pretrained=False,
        freeze_backbone=False,
        optimizer_name=optimizer_name,  # type: ignore[arg-type]
        weight_decay=0.0001,
        early_stopping_patience=0,
        device="cpu",
        output_dir=output_dir,
    )
