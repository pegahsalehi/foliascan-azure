"""Single-image prediction for registered FoliaScan PyTorch checkpoints."""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, TypeAlias, cast

import torch
from PIL import Image, UnidentifiedImageError
from torch import Tensor, nn

from foliascan.training.checkpoints import BEST_CHECKPOINT_FILENAME
from foliascan.training.model import ModelFactoryError, create_model
from foliascan.training.transforms import create_eval_transform

JsonPredictionValue: TypeAlias = str | int | float | dict[str, float]
PredictionResponse: TypeAlias = dict[str, JsonPredictionValue]

REQUIRED_CHECKPOINT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "model_name",
        "model_state_dict",
        "class_to_index",
        "training_config",
        "validation_metrics",
        "best_validation_loss",
        "random_seed",
    }
)


class PredictionError(ValueError):
    """Raised when prediction setup or inference cannot complete."""


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Validated checkpoint metadata used for inference."""

    model_name: str
    model_state_dict: Mapping[str, object]
    class_to_index: Mapping[str, int]
    index_to_class: tuple[str, ...]
    image_size: int


class FoliaScanPredictor:
    """Reusable one-image classifier backed by a FoliaScan checkpoint."""

    def __init__(self, checkpoint_path: Path) -> None:
        self._checkpoint_path = checkpoint_path
        checkpoint = load_checkpoint(checkpoint_path)
        self._index_to_class = checkpoint.index_to_class
        self._image_transform = create_eval_transform(checkpoint.image_size)
        self._model = _create_inference_model(checkpoint)

    @property
    def model(self) -> nn.Module:
        """Return the initialized PyTorch model."""

        return self._model

    @property
    def index_to_class(self) -> tuple[str, ...]:
        """Return class names in class-index order."""

        return self._index_to_class

    @property
    def checkpoint_path(self) -> Path:
        """Return the checkpoint path used by this predictor."""

        return self._checkpoint_path

    def predict_base64(self, image_base64: str) -> PredictionResponse:
        """Predict one raw base64-encoded JPEG or PNG image."""

        image = decode_base64_image(image_base64)
        image_tensor = self._image_transform(image).unsqueeze(0)
        with torch.inference_mode():
            logits = self._model(image_tensor)

        probabilities_tensor = _probabilities_from_logits(
            logits,
            expected_class_count=len(self._index_to_class),
        )
        probabilities = [
            float(probability)
            for probability in probabilities_tensor.squeeze(0).tolist()
        ]
        predicted_index = int(torch.argmax(probabilities_tensor, dim=1).item())
        predicted_class = self._index_to_class[predicted_index]
        confidence = probabilities[predicted_index]

        return {
            "predicted_class": predicted_class,
            "predicted_index": predicted_index,
            "confidence": confidence,
            "probabilities": {
                class_name: probabilities[index]
                for index, class_name in enumerate(self._index_to_class)
            },
        }


def load_checkpoint(checkpoint_path: Path) -> LoadedCheckpoint:
    """Load and validate the raw PyTorch training checkpoint."""

    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        msg = f"Checkpoint file does not exist: {checkpoint_path}"
        raise PredictionError(msg)

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        msg = f"Unable to load checkpoint: {checkpoint_path}"
        raise PredictionError(msg) from exc

    if not isinstance(checkpoint, Mapping):
        msg = "Checkpoint must be a mapping."
        raise PredictionError(msg)

    missing_fields = sorted(REQUIRED_CHECKPOINT_FIELDS.difference(checkpoint))
    if missing_fields:
        msg = "Checkpoint is missing required field: " + missing_fields[0]
        raise PredictionError(msg)

    model_name = _checkpoint_string(checkpoint, "model_name")
    model_state_dict = _checkpoint_state_dict(checkpoint)
    class_to_index = _validate_class_mapping(
        _checkpoint_mapping(checkpoint, "class_to_index")
    )
    training_config = _checkpoint_mapping(checkpoint, "training_config")
    image_size = _training_config_image_size(training_config)

    return LoadedCheckpoint(
        model_name=model_name,
        model_state_dict=model_state_dict,
        class_to_index=class_to_index,
        index_to_class=_index_to_class(class_to_index),
        image_size=image_size,
    )


def decode_base64_image(image_base64: str) -> Image.Image:
    """Decode strict base64 image content and return an RGB Pillow image."""

    if not isinstance(image_base64, str) or not image_base64:
        msg = "image_base64 must be a non-empty string."
        raise PredictionError(msg)

    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        msg = "image_base64 is not valid strict Base64."
        raise PredictionError(msg) from exc

    if not image_bytes:
        msg = "image_base64 decoded to empty bytes."
        raise PredictionError(msg)

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            return cast(Image.Image, image.convert("RGB"))
    except (OSError, UnidentifiedImageError) as exc:
        msg = "image_base64 does not contain a valid image."
        raise PredictionError(msg) from exc


def _create_inference_model(checkpoint: LoadedCheckpoint) -> nn.Module:
    try:
        model = create_model(
            model_name=checkpoint.model_name,
            num_classes=len(checkpoint.index_to_class),
            pretrained=False,
            freeze_backbone=False,
        )
    except ModelFactoryError as exc:
        raise PredictionError(str(exc)) from exc

    try:
        model.load_state_dict(checkpoint.model_state_dict, strict=True)
    except RuntimeError as exc:
        msg = "Checkpoint model_state_dict is not compatible with the model."
        raise PredictionError(msg) from exc

    model.eval()
    return model


def _probabilities_from_logits(
    logits: object,
    *,
    expected_class_count: int,
) -> Tensor:
    if not isinstance(logits, Tensor):
        msg = "Model output must be a tensor."
        raise PredictionError(msg)
    if logits.ndim != 2 or logits.shape[0] != 1:
        msg = "Model output must have shape [1, num_classes]."
        raise PredictionError(msg)
    if int(logits.shape[1]) != expected_class_count:
        msg = (
            "Model output class count does not match checkpoint class mapping: "
            f"expected {expected_class_count}, found {int(logits.shape[1])}."
        )
        raise PredictionError(msg)
    return torch.softmax(logits, dim=1)


def _checkpoint_string(checkpoint: Mapping[object, object], field_name: str) -> str:
    value = checkpoint[field_name]
    if not isinstance(value, str) or not value:
        msg = f"Checkpoint field '{field_name}' must be a non-empty string."
        raise PredictionError(msg)
    return value


def _checkpoint_mapping(
    checkpoint: Mapping[object, object],
    field_name: str,
) -> Mapping[object, object]:
    value = checkpoint[field_name]
    if not isinstance(value, Mapping):
        msg = f"Checkpoint field '{field_name}' must be a mapping."
        raise PredictionError(msg)
    return value


def _checkpoint_state_dict(
    checkpoint: Mapping[object, object],
) -> Mapping[str, object]:
    state_dict = _checkpoint_mapping(checkpoint, "model_state_dict")
    for key in state_dict:
        if not isinstance(key, str) or not key:
            msg = "Checkpoint model_state_dict keys must be non-empty strings."
            raise PredictionError(msg)
    return cast(Mapping[str, object], state_dict)


def _training_config_image_size(training_config: Mapping[object, object]) -> int:
    image_size = training_config.get("image_size")
    if not isinstance(image_size, int) or isinstance(image_size, bool):
        msg = "Checkpoint training_config.image_size must be a positive integer."
        raise PredictionError(msg)
    if image_size <= 0:
        msg = "Checkpoint training_config.image_size must be a positive integer."
        raise PredictionError(msg)
    return image_size


def _validate_class_mapping(
    class_to_index_value: Mapping[object, object],
) -> Mapping[str, int]:
    if not class_to_index_value:
        msg = "Checkpoint class_to_index must not be empty."
        raise PredictionError(msg)

    class_to_index: dict[str, int] = {}
    for class_name, class_index in class_to_index_value.items():
        if not isinstance(class_name, str) or not class_name:
            msg = "Checkpoint class_to_index keys must be non-empty class names."
            raise PredictionError(msg)
        if not isinstance(class_index, int) or isinstance(class_index, bool):
            msg = "Checkpoint class_to_index values must be integer indices."
            raise PredictionError(msg)
        class_to_index[class_name] = class_index

    indices = tuple(class_to_index.values())
    if len(set(indices)) != len(indices):
        msg = "Checkpoint class_to_index contains duplicate indices."
        raise PredictionError(msg)

    expected_indices = tuple(range(len(indices)))
    if tuple(sorted(indices)) != expected_indices:
        msg = (
            "Checkpoint class_to_index indices must be contiguous and begin at zero."
        )
        raise PredictionError(msg)

    return class_to_index


def _index_to_class(class_to_index: Mapping[str, int]) -> tuple[str, ...]:
    ordered_classes = [""] * len(class_to_index)
    for class_name, class_index in class_to_index.items():
        ordered_classes[class_index] = class_name
    return tuple(ordered_classes)


__all__ = [
    "BEST_CHECKPOINT_FILENAME",
    "FoliaScanPredictor",
    "PredictionError",
    "PredictionResponse",
    "decode_base64_image",
    "load_checkpoint",
]
