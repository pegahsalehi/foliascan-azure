import base64
from io import BytesIO
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from foliascan.inference import predictor as predictor_module
from foliascan.inference.predictor import (
    FoliaScanPredictor,
    PredictionError,
    decode_base64_image,
)


def test_predictor_initializes_checkpoint_and_eval_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_module, "create_model", _tiny_model_factory)
    checkpoint_path = _write_checkpoint(tmp_path)

    predictor = FoliaScanPredictor(checkpoint_path)

    assert predictor.checkpoint_path == checkpoint_path
    assert predictor.index_to_class == _class_names()
    assert predictor.model.training is False


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_predictor_returns_valid_single_image_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_format: str,
) -> None:
    monkeypatch.setattr(predictor_module, "create_model", _tiny_model_factory)
    predictor = FoliaScanPredictor(_write_checkpoint(tmp_path))

    response = predictor.predict_base64(_encoded_image(image_format))

    assert set(response) == {
        "predicted_class",
        "predicted_index",
        "confidence",
        "probabilities",
    }
    assert response["predicted_class"] == "class_9"
    assert response["predicted_index"] == 9
    assert isinstance(response["confidence"], float)
    probabilities = response["probabilities"]
    assert isinstance(probabilities, dict)
    assert tuple(probabilities) == _class_names()
    assert len(probabilities) == 10
    assert probabilities[response["predicted_class"]] == response["confidence"]
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_invalid_base64_fails_clearly() -> None:
    with pytest.raises(PredictionError, match="valid strict Base64"):
        decode_base64_image("not valid base64")


def test_base64_non_image_bytes_fail_clearly() -> None:
    encoded = base64.b64encode(b"not an image").decode("ascii")

    with pytest.raises(PredictionError, match="valid image"):
        decode_base64_image(encoded)


@pytest.mark.parametrize(
    ("class_to_index", "error_match"),
    [
        ({}, "must not be empty"),
        ({"class_a": 0, "class_b": 0}, "duplicate"),
        ({"class_a": 1}, "contiguous"),
        ({"class_a": "0"}, "integer"),
    ],
)
def test_invalid_class_mapping_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_to_index: dict[str, object],
    error_match: str,
) -> None:
    monkeypatch.setattr(predictor_module, "create_model", _tiny_model_factory)
    checkpoint_path = _write_checkpoint(tmp_path, class_to_index=class_to_index)

    with pytest.raises(PredictionError, match=error_match):
        FoliaScanPredictor(checkpoint_path)


def test_model_state_mismatch_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(predictor_module, "create_model", _tiny_model_factory)
    checkpoint_path = _write_checkpoint(
        tmp_path,
        model_state_dict={"unexpected.weight": torch.zeros((1, 1))},
    )

    with pytest.raises(PredictionError, match="model_state_dict"):
        FoliaScanPredictor(checkpoint_path)


class TinyClassifier(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(images.flatten(start_dim=1))


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
    return TinyClassifier(num_classes)


def _write_checkpoint(
    tmp_path: Path,
    *,
    class_to_index: dict[str, object] | None = None,
    model_state_dict: dict[str, object] | None = None,
) -> Path:
    checkpoint_path = tmp_path / "best_model.pt"
    effective_class_to_index = (
        {class_name: index for index, class_name in enumerate(_class_names())}
        if class_to_index is None
        else class_to_index
    )
    state_dict = (
        _tiny_state_dict(max(len(effective_class_to_index), 1))
        if model_state_dict is None
        else model_state_dict
    )
    torch.save(
        {
            "model_name": "resnet18",
            "model_state_dict": state_dict,
            "class_to_index": effective_class_to_index,
            "training_config": {"image_size": 8},
            "validation_metrics": {"accuracy": 0.5},
            "best_validation_loss": 0.7,
            "random_seed": 42,
        },
        checkpoint_path,
    )
    return checkpoint_path


def _tiny_state_dict(num_classes: int) -> dict[str, object]:
    model = TinyClassifier(num_classes)
    with torch.no_grad():
        model.classifier.weight.zero_()
        model.classifier.bias.copy_(torch.arange(num_classes, dtype=torch.float32))
    return dict(model.state_dict())


def _encoded_image(image_format: str) -> str:
    image = Image.new("RGB", (12, 12), color=(120, 40, 200))
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _class_names() -> tuple[str, ...]:
    return tuple(f"class_{index}" for index in range(10))
