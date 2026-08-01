import csv
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from foliascan.training import train as train_module
from foliascan.training.checkpoints import (
    BEST_CHECKPOINT_FILENAME,
    LAST_CHECKPOINT_FILENAME,
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
        overwrite: bool = False,
        on_epoch: object = None,
    ) -> TrainingSummary:
        assert manifest_path == tmp_path / "dataset_manifest.csv"
        assert data_dir == tmp_path / "raw"
        assert config_path == tmp_path / "training.yaml"
        assert output_dir_override == tmp_path / "override"
        assert device_override == "cpu"
        assert epochs_override == 1
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
            "--overwrite",
        ]
    )

    captured = capsys.readouterr()
    assert exit_status == 0
    assert "FoliaScan local training" in captured.out
    assert "Training complete" in captured.out


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
) -> tuple[Path, Path, Path]:
    data_dir = tmp_path / "raw"
    manifest_path = tmp_path / "dataset_manifest.csv"
    config_path = _write_config(tmp_path, epochs=epochs, patience=patience)
    rows = [
        ("class_a/train_a.jpg", "class_a", "train", "leaf_a", "train"),
        ("class_b/train_b.jpg", "class_b", "train", "leaf_b", "train"),
        ("class_a/validation_a.jpg", "class_a", "validation", "leaf_c", "train"),
        ("class_b/validation_b.jpg", "class_b", "validation", "leaf_d", "train"),
        ("class_a/missing_test.jpg", "class_a", "test", "leaf_e", "test"),
    ]
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


def _write_config(tmp_path: Path, *, epochs: int, patience: int) -> Path:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "image_size: 16",
                "batch_size: 2",
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
                f"output_dir: {(tmp_path / 'artifacts' / 'training').as_posix()}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


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
