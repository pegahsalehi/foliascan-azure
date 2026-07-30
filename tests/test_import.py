from pathlib import Path

import foliascan
from foliascan.config import (
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    SAMPLE_IMAGES_DIR,
    TRAINING_CONFIG_PATH,
)


def test_package_imports() -> None:
    assert foliascan.__version__


def test_project_root_exists() -> None:
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.is_dir()


def test_configured_paths_are_inside_project_root() -> None:
    configured_paths: tuple[Path, ...] = (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        SAMPLE_IMAGES_DIR,
        TRAINING_CONFIG_PATH,
    )

    for path in configured_paths:
        assert path.is_relative_to(PROJECT_ROOT)
