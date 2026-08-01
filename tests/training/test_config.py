from pathlib import Path

import pytest

from foliascan.training.config import (
    TrainingConfig,
    TrainingConfigError,
    load_training_config,
    training_config_from_mapping,
)


def test_load_training_config_returns_immutable_typed_config(tmp_path: Path) -> None:
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "random_seed: 42",
                "image_size: 224",
                "batch_size: 32",
                "learning_rate: 0.001",
                "epochs: 10",
                "model_name: resnet18",
                "num_workers: 0",
                "pretrained: false",
                "freeze_backbone: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_training_config(config_path)

    assert config == TrainingConfig(
        random_seed=42,
        image_size=224,
        batch_size=32,
        learning_rate=0.001,
        epochs=10,
        model_name="resnet18",
        num_workers=0,
        pretrained=False,
        freeze_backbone=True,
    )


def test_training_config_rejects_missing_field() -> None:
    with pytest.raises(TrainingConfigError, match="batch_size"):
        training_config_from_mapping(
            {
                "random_seed": 42,
                "image_size": 224,
                "learning_rate": 0.001,
                "epochs": 10,
                "model_name": "resnet18",
                "num_workers": 0,
                "pretrained": False,
                "freeze_backbone": False,
            }
        )


@pytest.mark.parametrize(
    ("field_name", "value", "error_match"),
    [
        ("image_size", 0, "positive integer"),
        ("batch_size", -1, "positive integer"),
        ("learning_rate", 0.0, "positive number"),
        ("epochs", 0, "positive integer"),
        ("num_workers", -1, "non-negative integer"),
        ("pretrained", "no", "true or false"),
        ("freeze_backbone", "yes", "true or false"),
        ("model_name", "", "non-empty string"),
    ],
)
def test_training_config_rejects_invalid_values(
    field_name: str,
    value: object,
    error_match: str,
) -> None:
    config: dict[str, object] = {
        "random_seed": 42,
        "image_size": 224,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "model_name": "resnet18",
        "num_workers": 0,
        "pretrained": False,
        "freeze_backbone": False,
    }
    config[field_name] = value

    with pytest.raises(TrainingConfigError, match=error_match):
        training_config_from_mapping(config)

