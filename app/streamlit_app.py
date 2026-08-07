"""Streamlit interface for invoking a FoliaScan Azure ML endpoint."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Final

import streamlit as st

from foliascan.client.azure_endpoint import (
    AzureEndpointClient,
    FoliaScanEndpointError,
    PredictionResponse,
)

SUPPORTED_IMAGE_TYPES: Final[tuple[str, ...]] = ("jpg", "jpeg", "png")
APP_DIR: Final[Path] = Path(__file__).resolve().parent
PROJECT_ROOT: Final[Path] = APP_DIR.parent
ASSET_ROOT: Final[Path] = PROJECT_ROOT / "assets"
LOGO_PATH: Final[Path] = ASSET_ROOT / "branding" / "foliascan-logo.png"
ICON_PATH: Final[Path] = ASSET_ROOT / "branding" / "foliascan-icon.png"
DISCLAIMER: Final[str] = (
    "Educational use only. FoliaScan is not a professional agricultural "
    "diagnosis tool."
)


def main() -> None:
    """Render the FoliaScan Streamlit application."""

    st.set_page_config(page_title="FoliaScan", page_icon=str(ICON_PATH))
    _apply_styles()
    _render_header()

    uploaded_file = st.file_uploader(
        "Upload a tomato-leaf image",
        type=SUPPORTED_IMAGE_TYPES,
        help="JPEG and PNG files are supported.",
    )
    if uploaded_file is None:
        st.info("Upload a JPEG or PNG tomato-leaf image to begin.")
        st.caption(DISCLAIMER)
        return

    image_bytes = uploaded_file.getvalue()
    preview_column, _ = st.columns([0.62, 0.38])
    preview_column.image(
        image_bytes,
        caption="Image preview",
        use_container_width=True,
    )

    if st.button("Analyze", type="primary"):
        _run_inference(image_bytes)

    st.caption(DISCLAIMER)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 860px;
            padding-top: 2.5rem;
        }
        .foliascan-intro {
            color: #425348;
            font-size: 1.02rem;
            line-height: 1.6;
            margin-bottom: 1.35rem;
        }
        .foliascan-result {
            border: 1px solid #d7e8d7;
            border-left: 0.35rem solid #2f7d46;
            border-radius: 0.5rem;
            padding: 1rem 1.1rem;
            margin: 1.25rem 0 1rem;
            background: #fbfdf9;
        }
        .foliascan-result-label {
            color: #587061;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0;
            margin: 0 0 0.2rem;
            text-transform: uppercase;
        }
        .foliascan-result-title {
            color: #173d24;
            font-size: 1.75rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0 0 0.45rem;
        }
        .foliascan-result-confidence {
            color: #243228;
            font-size: 1rem;
            margin: 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_header() -> None:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=220, output_format="PNG")

    st.title("FoliaScan")

    st.markdown(
        """
        <p class="foliascan-intro">
        FoliaScan performs AI-based tomato-leaf image classification using a
        computer vision model served through Azure ML. Upload one clear leaf
        image, preview it, then run inference against the configured endpoint.
        </p>
        """,
        unsafe_allow_html=True,
    )


def _run_inference(image_bytes: bytes) -> None:
    with st.spinner("Analyzing image..."):
        try:
            prediction = AzureEndpointClient.from_environment().predict(image_bytes)
        except FoliaScanEndpointError as exc:
            st.error(str(exc))
            return

    _render_prediction(prediction)


def _render_prediction(prediction: PredictionResponse) -> None:
    display_name = display_class_name(prediction["predicted_class"])
    st.markdown(
        f"""
        <section class="foliascan-result" role="status"
            aria-label="FoliaScan prediction result">
            <p class="foliascan-result-label">Prediction</p>
            <h2 class="foliascan-result-title">{escape(display_name)}</h2>
            <p class="foliascan-result-confidence">
                Confidence: <strong>{prediction["confidence"]:.1%}</strong>
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top class probabilities")
    st.dataframe(
        probability_rows(prediction, limit=3),
        hide_index=True,
        use_container_width=True,
    )

    with st.expander("View all class probabilities"):
        st.dataframe(
            probability_rows(prediction),
            hide_index=True,
            use_container_width=True,
        )


def probability_rows(
    prediction: PredictionResponse,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Return display-ready class probabilities sorted highest to lowest."""

    sorted_items = sorted(
        prediction["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
    if limit is not None:
        sorted_items = sorted_items[:limit]

    return [
        {"Class": display_class_name(class_name), "Probability": f"{probability:.1%}"}
        for class_name, probability in sorted_items
    ]


def display_class_name(raw_class_name: str) -> str:
    """Convert backend PlantVillage class names into readable UI labels."""

    class_name = raw_class_name.strip()
    if not class_name:
        return "Unknown"

    if "___" in class_name:
        _, _, class_name = class_name.partition("___")
    elif class_name.lower().startswith("tomato "):
        class_name = class_name[7:]
    elif class_name.lower().startswith("tomato_"):
        class_name = class_name[7:]

    words = class_name.replace("_", " ").replace("-", " ").split()
    if not words:
        return "Unknown"
    return " ".join(word.capitalize() for word in words)


if __name__ == "__main__":
    main()
