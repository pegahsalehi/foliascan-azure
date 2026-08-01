"""Typed training configuration loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml  # type: ignore[import-untyped]

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


def _required_non_empty_string(config: Mapping[str, Any], field_name: str) -> str:
    value = config[field_name]
    if not isinstance(value, str) or not value:
        msg = f"Training config field '{field_name}' must be a non-empty string."
        raise TrainingConfigError(msg)
    return value


def _required_bool(config: Mapping[str, Any], field_name: str) -> bool:
    value = config[field_name]
    if not isinstance(value, bool):
        msg = f"Training config field '{field_name}' must be true or false."
        raise TrainingConfigError(msg)
    return value
