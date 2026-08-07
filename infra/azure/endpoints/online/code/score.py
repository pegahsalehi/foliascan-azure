"""Azure ML online endpoint scoring script for FoliaScan."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from foliascan.inference.predictor import (
    BEST_CHECKPOINT_FILENAME,
    FoliaScanPredictor,
    PredictionError,
    PredictionResponse,
)

LOGGER = logging.getLogger(__name__)
AZUREML_MODEL_DIR_ENVIRONMENT_VARIABLE = "AZUREML_MODEL_DIR"

_PREDICTOR: FoliaScanPredictor | None = None


class ScoringError(ValueError):
    """Raised when scoring setup or request handling fails."""


def init() -> None:
    """Initialize the module-level predictor from the Azure ML model directory."""

    global _PREDICTOR  # noqa: PLW0603

    if _PREDICTOR is not None:
        LOGGER.info("FoliaScan predictor already initialized.")
        return

    model_dir_text = os.environ.get(AZUREML_MODEL_DIR_ENVIRONMENT_VARIABLE, "").strip()
    if not model_dir_text:
        msg = "AZUREML_MODEL_DIR is not set."
        raise ScoringError(msg)

    checkpoint_path = _find_single_checkpoint(Path(model_dir_text))
    LOGGER.info("Initializing FoliaScan predictor from Azure ML model directory.")
    _PREDICTOR = FoliaScanPredictor(checkpoint_path)
    LOGGER.info("FoliaScan predictor initialized.")


def run(raw_data: object) -> PredictionResponse:
    """Run one JSON scoring request and return a JSON-serializable response."""

    if _PREDICTOR is None:
        init()

    if _PREDICTOR is None:
        msg = "FoliaScan predictor is not initialized."
        raise ScoringError(msg)

    request = _parse_request(raw_data)
    image_base64 = request.get("image_base64")
    if not isinstance(image_base64, str):
        msg = "Request field 'image_base64' must be a non-empty string."
        raise ScoringError(msg)
    if not image_base64:
        msg = "Request field 'image_base64' must be a non-empty string."
        raise ScoringError(msg)

    LOGGER.info("Running FoliaScan single-image prediction.")
    try:
        return _PREDICTOR.predict_base64(image_base64)
    except PredictionError as exc:
        raise ScoringError(str(exc)) from exc


def _find_single_checkpoint(model_dir: Path) -> Path:
    if not model_dir.exists() or not model_dir.is_dir():
        msg = f"Azure ML model directory does not exist: {model_dir}"
        raise ScoringError(msg)

    checkpoint_paths = tuple(sorted(model_dir.rglob(BEST_CHECKPOINT_FILENAME)))
    if not checkpoint_paths:
        msg = f"No {BEST_CHECKPOINT_FILENAME} file found under AZUREML_MODEL_DIR."
        raise ScoringError(msg)
    if len(checkpoint_paths) > 1:
        msg = (
            f"Multiple {BEST_CHECKPOINT_FILENAME} files found under "
            "AZUREML_MODEL_DIR."
        )
        raise ScoringError(msg)
    return checkpoint_paths[0]


def _parse_request(raw_data: object) -> dict[str, Any]:
    if isinstance(raw_data, dict):
        return raw_data
    if isinstance(raw_data, bytes):
        try:
            raw_data = raw_data.decode("utf-8")
        except UnicodeDecodeError as exc:
            msg = "Request body bytes must be UTF-8 JSON."
            raise ScoringError(msg) from exc
    if isinstance(raw_data, str):
        try:
            parsed = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            msg = "Request body must be valid JSON."
            raise ScoringError(msg) from exc
        if not isinstance(parsed, dict):
            msg = "Request JSON must be an object."
            raise ScoringError(msg)
        return parsed

    msg = "Request body must be a JSON string, UTF-8 bytes, or dictionary."
    raise ScoringError(msg)
