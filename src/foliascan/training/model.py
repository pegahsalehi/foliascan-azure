"""Model factory for local FoliaScan baselines."""

from __future__ import annotations

from typing import Any, cast

from torch import nn
from torchvision.models import (  # type: ignore[import-untyped]
    ResNet18_Weights,
    resnet18,
)


class ModelFactoryError(ValueError):
    """Raised when a model cannot be created from requested options."""


def create_model(
    *,
    model_name: str,
    num_classes: int,
    pretrained: bool,
    freeze_backbone: bool,
) -> nn.Module:
    """Create a supported image classification model."""

    if model_name != "resnet18":
        msg = f"Unsupported model_name: {model_name}"
        raise ModelFactoryError(msg)
    if num_classes <= 0:
        msg = "num_classes must be greater than zero."
        raise ModelFactoryError(msg)

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = cast(Any, resnet18(weights=weights))
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc.parameters():
            parameter.requires_grad = True

    return cast(nn.Module, model)
