import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from foliascan.client.azure_endpoint import PredictionResponse

STREAMLIT_APP_PATH = (
    Path(__file__).resolve().parents[2] / "app" / "streamlit_app.py"
)


def test_display_class_name_removes_plantvillage_tomato_prefix() -> None:
    streamlit_app = _load_streamlit_app()

    assert streamlit_app.display_class_name("Tomato___Late_blight") == "Late Blight"
    assert (
        streamlit_app.display_class_name("Tomato___Bacterial_spot")
        == "Bacterial Spot"
    )


def test_display_class_name_keeps_meaningful_inner_tomato_name() -> None:
    streamlit_app = _load_streamlit_app()

    assert (
        streamlit_app.display_class_name("Tomato___Tomato_Yellow_Leaf_Curl_Virus")
        == "Tomato Yellow Leaf Curl Virus"
    )


def test_display_class_name_handles_plain_or_empty_names() -> None:
    streamlit_app = _load_streamlit_app()

    assert streamlit_app.display_class_name("Tomato healthy") == "Healthy"
    assert streamlit_app.display_class_name("  ") == "Unknown"


def test_probability_rows_sort_and_limit_display_values() -> None:
    streamlit_app = _load_streamlit_app()
    prediction: PredictionResponse = {
        "predicted_class": "Tomato___Late_blight",
        "predicted_index": 2,
        "confidence": 0.7,
        "probabilities": {
            "Tomato___Bacterial_spot": 0.2,
            "Tomato___Late_blight": 0.7,
            "Tomato___healthy": 0.1,
        },
    }

    assert streamlit_app.probability_rows(prediction, limit=2) == [
        {"Class": "Late Blight", "Probability": "70.0%"},
        {"Class": "Bacterial Spot", "Probability": "20.0%"},
    ]


def _load_streamlit_app() -> ModuleType:
    module_name = "foliascan_streamlit_app_test_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, STREAMLIT_APP_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError("Unable to load Streamlit app module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
