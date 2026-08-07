import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from foliascan.inference.predictor import PredictionError

SCORE_PATH = (
    Path(__file__).resolve().parents[2]
    / "infra"
    / "azure"
    / "endpoints"
    / "online"
    / "code"
    / "score.py"
)


class FakePredictor:
    instances: list["FakePredictor"] = []

    def __init__(self, checkpoint_path: Path) -> None:
        self.checkpoint_path = checkpoint_path
        FakePredictor.instances.append(self)

    def predict_base64(self, image_base64: str) -> dict[str, object]:
        if image_base64 == "invalid-base64":
            raise PredictionError("image_base64 is not valid strict Base64.")
        if image_base64 == "not-image":
            raise PredictionError("image_base64 does not contain a valid image.")
        return {
            "predicted_class": "class_1",
            "predicted_index": 1,
            "confidence": 0.75,
            "probabilities": {"class_0": 0.25, "class_1": 0.75},
        }


def test_init_finds_direct_checkpoint_and_initializes_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _load_score_module(monkeypatch)
    checkpoint_path = tmp_path / "best_model.pt"
    checkpoint_path.touch()
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(score, "FoliaScanPredictor", FakePredictor)
    FakePredictor.instances.clear()

    score.init()
    score.init()

    assert len(FakePredictor.instances) == 1
    assert FakePredictor.instances[0].checkpoint_path == checkpoint_path


def test_init_finds_nested_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _load_score_module(monkeypatch)
    checkpoint_path = tmp_path / "azure-model" / "1" / "best_model.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint_path.touch()
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(score, "FoliaScanPredictor", FakePredictor)
    FakePredictor.instances.clear()

    score.init()

    assert FakePredictor.instances[0].checkpoint_path == checkpoint_path


def test_missing_model_dir_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    score = _load_score_module(monkeypatch)
    monkeypatch.delenv("AZUREML_MODEL_DIR", raising=False)

    with pytest.raises(score.ScoringError, match="AZUREML_MODEL_DIR"):
        score.init()


def test_missing_checkpoint_fails_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _load_score_module(monkeypatch)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))

    with pytest.raises(score.ScoringError, match="No best_model.pt"):
        score.init()


def test_multiple_checkpoints_fail_clearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _load_score_module(monkeypatch)
    first_checkpoint = tmp_path / "model-a" / "best_model.pt"
    second_checkpoint = tmp_path / "model-b" / "best_model.pt"
    first_checkpoint.parent.mkdir()
    second_checkpoint.parent.mkdir()
    first_checkpoint.touch()
    second_checkpoint.touch()
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))

    with pytest.raises(score.ScoringError, match="Multiple best_model.pt"):
        score.init()


@pytest.mark.parametrize(
    "raw_data",
    [
        {"image_base64": "valid"},
        json.dumps({"image_base64": "valid"}),
        json.dumps({"image_base64": "valid"}).encode("utf-8"),
    ],
)
def test_run_accepts_dictionary_string_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_data: object,
) -> None:
    score = _load_score_module(monkeypatch)
    score._PREDICTOR = FakePredictor(tmp_path / "best_model.pt")

    response = score.run(raw_data)

    assert response["predicted_class"] == "class_1"
    assert response["predicted_index"] == 1
    assert response["probabilities"] == {"class_0": 0.25, "class_1": 0.75}


def test_run_rejects_malformed_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    score = _load_score_module(monkeypatch)
    score._PREDICTOR = FakePredictor(tmp_path / "best_model.pt")

    with pytest.raises(score.ScoringError, match="valid JSON"):
        score.run("{bad json")


@pytest.mark.parametrize(
    ("raw_data", "error_match"),
    [
        ({}, "image_base64"),
        ({"image_base64": ""}, "non-empty string"),
    ],
)
def test_run_rejects_missing_or_empty_image_base64(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    raw_data: dict[str, object],
    error_match: str,
) -> None:
    score = _load_score_module(monkeypatch)
    score._PREDICTOR = FakePredictor(tmp_path / "best_model.pt")

    with pytest.raises(score.ScoringError, match=error_match):
        score.run(raw_data)


@pytest.mark.parametrize(
    ("image_base64", "error_match"),
    [
        ("invalid-base64", "valid strict Base64"),
        ("not-image", "valid image"),
    ],
)
def test_run_surfaces_prediction_input_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    image_base64: str,
    error_match: str,
) -> None:
    score = _load_score_module(monkeypatch)
    score._PREDICTOR = FakePredictor(tmp_path / "best_model.pt")

    with pytest.raises(score.ScoringError, match=error_match):
        score.run({"image_base64": image_base64})


def _load_score_module(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module_name = "foliascan_score_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SCORE_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load score.py module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_PREDICTOR", None)
    return module
