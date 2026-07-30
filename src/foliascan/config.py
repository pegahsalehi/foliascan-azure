"""Project-level paths for FoliaScan."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
RAW_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
SAMPLE_IMAGES_DIR: Final[Path] = PROJECT_ROOT / "sample_images"
TRAINING_CONFIG_PATH: Final[Path] = PROJECT_ROOT / "configs" / "training.example.yaml"
