"""Directory-based image dataset discovery."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path
from typing import Final

SUPPORTED_IMAGE_EXTENSIONS: Final[tuple[str, ...]] = (".jpg", ".jpeg", ".png")


class DatasetDiscoveryError(ValueError):
    """Raised when a directory-based image dataset cannot be discovered."""


@dataclass(frozen=True, slots=True)
class ImageRecord:
    """An image path and the class assigned by its parent directory."""

    path: Path
    class_name: str


def discover_class_directories(dataset_root: Path) -> tuple[Path, ...]:
    """Return immediate, non-hidden class directories in sorted order."""

    _ensure_dataset_root(dataset_root)
    class_directories = tuple(
        sorted(
            (
                path
                for path in dataset_root.iterdir()
                if path.is_dir() and not _is_hidden(path)
            ),
            key=_path_sort_key,
        )
    )

    if not class_directories:
        msg = f"No class directories found under dataset root: {dataset_root}"
        raise DatasetDiscoveryError(msg)

    return class_directories


def discover_class_names(dataset_root: Path) -> tuple[str, ...]:
    """Return discovered class names in deterministic sorted order."""

    return tuple(path.name for path in discover_class_directories(dataset_root))


def discover_image_records(
    dataset_root: Path,
    supported_extensions: Collection[str] = SUPPORTED_IMAGE_EXTENSIONS,
) -> tuple[ImageRecord, ...]:
    """Return supported image records for immediate class-directory files."""

    normalized_extensions = _normalize_extensions(supported_extensions)
    records: list[ImageRecord] = []

    for class_directory in discover_class_directories(dataset_root):
        image_paths = sorted(
            (
                path
                for path in class_directory.iterdir()
                if path.is_file()
                and not _is_hidden(path)
                and path.suffix.lower() in normalized_extensions
            ),
            key=_path_sort_key,
        )
        records.extend(
            ImageRecord(path=image_path, class_name=class_directory.name)
            for image_path in image_paths
        )

    return tuple(records)


def _ensure_dataset_root(dataset_root: Path) -> None:
    if not dataset_root.exists():
        msg = f"Dataset root does not exist: {dataset_root}"
        raise DatasetDiscoveryError(msg)

    if not dataset_root.is_dir():
        msg = f"Dataset root is not a directory: {dataset_root}"
        raise DatasetDiscoveryError(msg)


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")


def _normalize_extensions(supported_extensions: Collection[str]) -> frozenset[str]:
    return frozenset(
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in supported_extensions
    )


def _path_sort_key(path: Path) -> tuple[str, str]:
    name = path.name
    return (name.casefold(), name)

