import csv
import json
from pathlib import Path

import pytest
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
from foliascan.training.config import TrainingConfig
from foliascan.training.dataset import ClassMapping
from foliascan.training.engine import EpochMetrics


def test_write_history_creates_stable_csv_and_json(tmp_path: Path) -> None:
    history = (
        EpochHistory(
            epoch=1,
            train_loss=0.75,
            train_accuracy=0.5,
            validation_loss=0.8,
            validation_accuracy=0.25,
            learning_rate=0.001,
            elapsed_seconds=1.25,
        ),
    )

    write_history(history, tmp_path)

    with (tmp_path / "history.csv").open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    with (tmp_path / "history.json").open(encoding="utf-8") as json_file:
        json_rows = json.load(json_file)

    assert rows == [
        {
            "epoch": "1",
            "train_loss": "0.75",
            "train_accuracy": "0.5",
            "validation_loss": "0.8",
            "validation_accuracy": "0.25",
            "learning_rate": "0.001",
            "elapsed_seconds": "1.25",
        }
    ]
    assert json_rows[0]["epoch"] == 1
    assert json_rows[0]["validation_loss"] == 0.8


def test_save_checkpoint_writes_metadata_for_best_and_last(tmp_path: Path) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    config = _config(tmp_path)
    mapping = ClassMapping({"class_a": 0, "class_b": 1}, ("class_a", "class_b"))
    metrics = EpochMetrics(
        average_loss=0.5,
        accuracy=0.75,
        sample_count=4,
        batch_count=2,
    )

    for filename in (BEST_CHECKPOINT_FILENAME, LAST_CHECKPOINT_FILENAME):
        save_checkpoint(
            output_path=tmp_path / filename,
            epoch=2,
            model=model,
            optimizer=optimizer,
            class_mapping=mapping,
            config=config,
            train_metrics=metrics,
            validation_metrics=metrics,
            best_validation_loss=0.5,
        )

    checkpoint = torch.load(tmp_path / BEST_CHECKPOINT_FILENAME, map_location="cpu")
    assert checkpoint["epoch"] == 2
    assert checkpoint["model_name"] == "resnet18"
    assert checkpoint["class_to_index"] == {"class_a": 0, "class_b": 1}
    assert checkpoint["training_config"]["output_dir"] == tmp_path.as_posix()
    assert checkpoint["train_metrics"]["sample_count"] == 4
    assert checkpoint["validation_metrics"]["accuracy"] == 0.75
    assert checkpoint["best_validation_loss"] == 0.5
    assert checkpoint["random_seed"] == 42
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert (tmp_path / LAST_CHECKPOINT_FILENAME).exists()


def test_prepare_output_dir_rejects_non_empty_directory_without_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    (output_dir / "existing.txt").write_text("already here", encoding="utf-8")

    with pytest.raises(TrainingArtifactError, match="not empty"):
        prepare_output_dir(output_dir, overwrite=False)

    prepare_output_dir(output_dir, overwrite=True)


def test_reset_managed_artifacts_removes_known_training_outputs(tmp_path: Path) -> None:
    for filename in (
        BEST_CHECKPOINT_FILENAME,
        LAST_CHECKPOINT_FILENAME,
        "history.csv",
        "history.json",
    ):
        (tmp_path / filename).write_text("old", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("keep", encoding="utf-8")

    reset_managed_artifacts(tmp_path)

    assert not (tmp_path / BEST_CHECKPOINT_FILENAME).exists()
    assert not (tmp_path / LAST_CHECKPOINT_FILENAME).exists()
    assert not (tmp_path / "history.csv").exists()
    assert not (tmp_path / "history.json").exists()
    assert (tmp_path / "notes.txt").exists()


def _config(output_dir: Path) -> TrainingConfig:
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
        optimizer_name="adamw",
        weight_decay=0.0001,
        early_stopping_patience=0,
        device="cpu",
        output_dir=output_dir,
    )

