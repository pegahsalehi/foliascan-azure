"""Typed training configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final, Literal

import yaml  # type: ignore[import-untyped]

DeviceName = Literal["auto", "cpu", "cuda"]
OptimizerName = Literal["adamw"]

REQUIRED_CONFIG_FIELDS: Final[tuple[str, ...]] = (
    "random_seed",
    "image_size",
    "batch_size",
    "learning_rate",
    "epochs",
    "model_name",
    "num_workers",
    "pretrained",
    "freeze_backbone",
    "optimizer_name",
    "weight_decay",
    "early_stopping_patience",
    "device",
    "output_dir",
)


class TrainingConfigError(ValueError):
    """Raised when a training configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Immutable local training configuration."""

    random_seed: int
    image_size: int
    batch_size: int
    learning_rate: float
    epochs: int
    model_name: str
    num_workers: int
    pretrained: bool
    freeze_backbone: bool
    optimizer_name: OptimizerName
    weight_decay: float
    early_stopping_patience: int
    device: DeviceName
    output_dir: Path


def load_training_config(config_path: Path) -> TrainingConfig:
    """Load and validate a training configuration from YAML."""

    with config_path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)

    if not isinstance(loaded, Mapping):
        msg = f"Training config must be a YAML mapping: {config_path}"
        raise TrainingConfigError(msg)

    return training_config_from_mapping(loaded)


def training_config_from_mapping(config: Mapping[str, Any]) -> TrainingConfig:
    """Validate a mapping and return an immutable training configuration."""

    missing_fields = [
        field_name for field_name in REQUIRED_CONFIG_FIELDS if field_name not in config
    ]
    if missing_fields:
        msg = "Training config is missing required field: " + missing_fields[0]
        raise TrainingConfigError(msg)

    return TrainingConfig(
        random_seed=_required_non_negative_int(config, "random_seed"),
        image_size=_required_positive_int(config, "image_size"),
        batch_size=_required_positive_int(config, "batch_size"),
        learning_rate=_required_positive_float(config, "learning_rate"),
        epochs=_required_positive_int(config, "epochs"),
        model_name=_required_non_empty_string(config, "model_name"),
        num_workers=_required_non_negative_int(config, "num_workers"),
        pretrained=_required_bool(config, "pretrained"),
        freeze_backbone=_required_bool(config, "freeze_backbone"),
        optimizer_name=_required_optimizer_name(config, "optimizer_name"),
        weight_decay=_required_non_negative_float(config, "weight_decay"),
        early_stopping_patience=_required_non_negative_int(
            config,
            "early_stopping_patience",
        ),
        device=_required_device_name(config, "device"),
        output_dir=_required_path(config, "output_dir"),
    )


def training_config_with_overrides(
    config: TrainingConfig,
    *,
    output_dir: Path | None = None,
    device: DeviceName | None = None,
    epochs: int | None = None,
) -> TrainingConfig:
    """Return a training configuration with validated CLI overrides applied."""

    if epochs is not None and epochs <= 0:
        msg = "Training config field 'epochs' must be a positive integer."
        raise TrainingConfigError(msg)

    return replace(
        config,
        output_dir=config.output_dir if output_dir is None else output_dir,
        device=config.device if device is None else device,
        epochs=config.epochs if epochs is None else epochs,
    )


def _required_positive_int(config: Mapping[str, Any], field_name: str) -> int:
    value = config[field_name]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        msg = f"Training config field '{field_name}' must be a positive integer."
        raise TrainingConfigError(msg)
    return value


def _required_non_negative_int(config: Mapping[str, Any], field_name: str) -> int:
    value = config[field_name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        msg = f"Training config field '{field_name}' must be a non-negative integer."
        raise TrainingConfigError(msg)
    return value


def _required_positive_float(config: Mapping[str, Any], field_name: str) -> float:
    value = config[field_name]
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"Training config field '{field_name}' must be a positive number."
        raise TrainingConfigError(msg)
    numeric_value = float(value)
    if numeric_value <= 0:
        msg = f"Training config field '{field_name}' must be a positive number."
        raise TrainingConfigError(msg)
    return numeric_value


def _required_non_negative_float(config: Mapping[str, Any], field_name: str) -> float:
    value = config[field_name]
    if not isinstance(value, int | float) or isinstance(value, bool):
        msg = f"Training config field '{field_name}' must be a non-negative number."
        raise TrainingConfigError(msg)
    numeric_value = float(value)
    if numeric_value < 0:
        msg = f"Training config field '{field_name}' must be a non-negative number."
        raise TrainingConfigError(msg)
    return numeric_value


def _required_non_empty_string(config: Mapping[str, Any], field_name: str) -> str:
    value = config[field_name]
    if not isinstance(value, str) or not value:
        msg = f"Training config field '{field_name}' must be a non-empty string."
        raise TrainingConfigError(msg)
    return value


def _required_optimizer_name(
    config: Mapping[str, Any],
    field_name: str,
) -> OptimizerName:
    value = _required_non_empty_string(config, field_name)
    if value == "adamw":
        return "adamw"
    msg = f"Unsupported optimizer_name: {value}"
    raise TrainingConfigError(msg)


def _required_device_name(config: Mapping[str, Any], field_name: str) -> DeviceName:
    value = _required_non_empty_string(config, field_name)
    if value == "auto":
        return "auto"
    if value == "cpu":
        return "cpu"
    if value == "cuda":
        return "cuda"
    msg = f"Unsupported device: {value}"
    raise TrainingConfigError(msg)


def _required_path(config: Mapping[str, Any], field_name: str) -> Path:
    value = config[field_name]
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        path = Path(value)
    else:
        msg = f"Training config field '{field_name}' must be a path string."
        raise TrainingConfigError(msg)

    if not path.as_posix():
        msg = f"Training config field '{field_name}' must not be empty."
        raise TrainingConfigError(msg)
    return path


def _required_bool(config: Mapping[str, Any], field_name: str) -> bool:
    value = config[field_name]
    if not isinstance(value, bool):
        msg = f"Training config field '{field_name}' must be true or false."
        raise TrainingConfigError(msg)
    return value
