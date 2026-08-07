"""Streamlit interface for invoking a FoliaScan Azure ML endpoint."""

from __future__ import annotations

from typing import Final

import streamlit as st

from foliascan.client.azure_endpoint import (
    AzureEndpointClient,
    FoliaScanEndpointError,
    PredictionResponse,
)

SUPPORTED_IMAGE_TYPES: Final[tuple[str, ...]] = ("jpg", "jpeg", "png")
DISCLAIMER: Final[str] = (
    "Educational use only. FoliaScan results are not professional agricultural "
    "diagnosis."
)


def main() -> None:
    """Render the FoliaScan Streamlit application."""

    st.set_page_config(page_title="FoliaScan")
    st.title("FoliaScan")
    st.write(
        "Classifies one uploaded tomato-leaf JPEG or PNG image using a configured "
        "Azure ML Managed Online Endpoint."
    )

    uploaded_file = st.file_uploader(
        "Upload a tomato-leaf image",
        type=SUPPORTED_IMAGE_TYPES,
    )
    if uploaded_file is None:
        st.info("Upload a JPEG or PNG image to begin.")
    else:
        image_bytes = uploaded_file.getvalue()
        st.image(image_bytes, caption="Image preview", use_container_width=True)
        if st.button("Analyze", type="primary"):
            _run_inference(image_bytes)

    st.caption(DISCLAIMER)


def _run_inference(image_bytes: bytes) -> None:
    with st.spinner("Analyzing image..."):
        try:
            prediction = AzureEndpointClient.from_environment().predict(image_bytes)
        except FoliaScanEndpointError as exc:
            st.error(str(exc))
            return

    _render_prediction(prediction)


def _render_prediction(prediction: PredictionResponse) -> None:
    left_column, right_column = st.columns(2)
    left_column.metric("Predicted class", prediction["predicted_class"])
    right_column.metric("Confidence", f"{prediction['confidence']:.1%}")

    st.subheader("Class probabilities")
    st.dataframe(
        _probability_rows(prediction),
        hide_index=True,
        use_container_width=True,
    )


def _probability_rows(prediction: PredictionResponse) -> list[dict[str, str]]:
    return [
        {"Class": class_name, "Probability": f"{probability:.1%}"}
        for class_name, probability in sorted(
            prediction["probabilities"].items(),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


if __name__ == "__main__":
    main()
