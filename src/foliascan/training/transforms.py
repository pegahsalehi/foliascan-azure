"""Image transforms for local FoliaScan training foundations."""

from __future__ import annotations

from typing import Final, cast

from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.transforms import InterpolationMode  # type: ignore[import-untyped]

from foliascan.training.dataset import ImageTransform

IMAGENET_MEAN: Final[tuple[float, float, float]] = (0.485, 0.456, 0.406)
IMAGENET_STD: Final[tuple[float, float, float]] = (0.229, 0.224, 0.225)


def create_train_transform(image_size: int) -> ImageTransform:
    """Create conservative randomized training image transforms."""

    _validate_image_size(image_size)
    resize_size = _resize_size(image_size)
    return cast(
        ImageTransform,
        transforms.Compose(
            [
                transforms.Resize(
                    resize_size,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.9, 1.0),
                    ratio=(0.95, 1.05),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(
                    degrees=10,
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    )


def create_eval_transform(image_size: int) -> ImageTransform:
    """Create deterministic validation and test image transforms."""

    _validate_image_size(image_size)
    return cast(
        ImageTransform,
        transforms.Compose(
            [
                transforms.Resize(
                    _resize_size(image_size),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        ),
    )


def _resize_size(image_size: int) -> int:
    return max(image_size, round(image_size * 1.15))


def _validate_image_size(image_size: int) -> None:
    if image_size <= 0:
        msg = "image_size must be greater than zero."
        raise ValueError(msg)
