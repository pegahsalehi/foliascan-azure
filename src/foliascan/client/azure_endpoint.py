"""HTTPS client for a FoliaScan Azure ML Managed Online Endpoint."""

from __future__ import annotations

import base64
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Final, TypedDict, cast

ENDPOINT_URL_ENVIRONMENT_VARIABLE: Final[str] = "FOLIASCAN_ENDPOINT_URL"
ENDPOINT_KEY_ENVIRONMENT_VARIABLE: Final[str] = "FOLIASCAN_ENDPOINT_KEY"
DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
REQUEST_IMAGE_FIELD: Final[str] = "image_base64"
REQUIRED_RESPONSE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "predicted_class",
        "predicted_index",
        "confidence",
        "probabilities",
    }
)


class PredictionResponse(TypedDict):
    """Validated response returned by the FoliaScan endpoint."""

    predicted_class: str
    predicted_index: int
    confidence: float
    probabilities: dict[str, float]


class FoliaScanEndpointError(RuntimeError):
    """Base error for application-level endpoint invocation failures."""


class EndpointConfigurationError(FoliaScanEndpointError):
    """Raised when endpoint configuration is incomplete or invalid."""


class EndpointRequestError(FoliaScanEndpointError):
    """Raised when a request cannot be sent to the endpoint."""


class EndpointResponseError(FoliaScanEndpointError):
    """Raised when the endpoint returns an unusable response."""


@dataclass(frozen=True, slots=True)
class AzureEndpointClient:
    """Reusable HTTPS client for one-image FoliaScan predictions."""

    endpoint_url: str
    endpoint_key: str = field(repr=False)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        endpoint_url = self.endpoint_url.strip()
        endpoint_key = self.endpoint_key.strip()

        if not endpoint_url:
            msg = f"{ENDPOINT_URL_ENVIRONMENT_VARIABLE} is not configured."
            raise EndpointConfigurationError(msg)
        if not endpoint_key:
            msg = f"{ENDPOINT_KEY_ENVIRONMENT_VARIABLE} is not configured."
            raise EndpointConfigurationError(msg)
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            msg = "Endpoint HTTP timeout must be greater than zero seconds."
            raise EndpointConfigurationError(msg)

        object.__setattr__(self, "endpoint_url", endpoint_url)
        object.__setattr__(self, "endpoint_key", endpoint_key)

    @classmethod
    def from_environment(
        cls,
        *,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> AzureEndpointClient:
        """Create a client from FoliaScan endpoint environment variables."""

        return cls(
            endpoint_url=os.environ.get(ENDPOINT_URL_ENVIRONMENT_VARIABLE, ""),
            endpoint_key=os.environ.get(ENDPOINT_KEY_ENVIRONMENT_VARIABLE, ""),
            timeout_seconds=timeout_seconds,
        )

    def predict(self, image_bytes: bytes) -> PredictionResponse:
        """Send one image to the configured endpoint and validate the response."""

        if not image_bytes:
            msg = "Image bytes must be non-empty."
            raise EndpointRequestError(msg)

        request = _prediction_request(
            endpoint_url=self.endpoint_url,
            endpoint_key=self.endpoint_key,
            image_bytes=image_bytes,
        )
        response_body = _send_request(request, timeout_seconds=self.timeout_seconds)
        parsed_response = _parse_response_json(response_body)
        return _validate_prediction_response(parsed_response)


def _prediction_request(
    *,
    endpoint_url: str,
    endpoint_key: str,
    image_bytes: bytes,
) -> urllib.request.Request:
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    payload = json.dumps({REQUEST_IMAGE_FIELD: image_base64}).encode("utf-8")
    return urllib.request.Request(
        endpoint_url,
        data=payload,
        headers={
            "Authorization": f"Bearer {endpoint_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )


def _send_request(
    request: urllib.request.Request,
    *,
    timeout_seconds: float,
) -> bytes:
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            status_code: object = response.getcode()
            response_body: object = response.read()
    except TimeoutError as exc:
        msg = "FoliaScan endpoint request timed out."
        raise EndpointRequestError(msg) from exc
    except urllib.error.HTTPError as exc:
        msg = f"FoliaScan endpoint returned HTTP {exc.code}."
        raise EndpointResponseError(msg) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            msg = "FoliaScan endpoint request timed out."
            raise EndpointRequestError(msg) from exc
        msg = "Unable to reach FoliaScan endpoint."
        raise EndpointRequestError(msg) from exc
    except OSError as exc:
        msg = "Unable to reach FoliaScan endpoint."
        raise EndpointRequestError(msg) from exc

    if not isinstance(status_code, int) or isinstance(status_code, bool):
        msg = "FoliaScan endpoint did not return an HTTP status code."
        raise EndpointResponseError(msg)
    if status_code < 200 or status_code >= 300:
        msg = f"FoliaScan endpoint returned HTTP {status_code}."
        raise EndpointResponseError(msg)
    if not isinstance(response_body, bytes):
        msg = "FoliaScan endpoint did not return a byte response body."
        raise EndpointResponseError(msg)
    return response_body


def _parse_response_json(response_body: bytes) -> object:
    try:
        parsed_response: object = json.loads(response_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        msg = "FoliaScan endpoint returned invalid JSON."
        raise EndpointResponseError(msg) from exc
    return parsed_response


def _validate_prediction_response(response: object) -> PredictionResponse:
    if not isinstance(response, Mapping):
        msg = "FoliaScan endpoint response JSON must be an object."
        raise EndpointResponseError(msg)

    response_mapping = cast(Mapping[object, object], response)
    missing_fields = sorted(
        field_name
        for field_name in REQUIRED_RESPONSE_FIELDS
        if field_name not in response_mapping
    )
    if missing_fields:
        msg = (
            "FoliaScan endpoint response is missing required prediction field: "
            f"{missing_fields[0]}."
        )
        raise EndpointResponseError(msg)

    predicted_class = _required_string(
        response_mapping["predicted_class"],
        "predicted_class",
    )
    predicted_index = _required_integer(
        response_mapping["predicted_index"],
        "predicted_index",
    )
    confidence = _required_float(response_mapping["confidence"], "confidence")
    probabilities = _required_probabilities(response_mapping["probabilities"])

    return {
        "predicted_class": predicted_class,
        "predicted_index": predicted_index,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        msg = f"FoliaScan endpoint response field '{field_name}' must be a string."
        raise EndpointResponseError(msg)
    return value


def _required_integer(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        msg = f"FoliaScan endpoint response field '{field_name}' must be an integer."
        raise EndpointResponseError(msg)
    return value


def _required_float(value: object, field_name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        msg = f"FoliaScan endpoint response field '{field_name}' must be a number."
        raise EndpointResponseError(msg)
    return float(value)


def _required_probabilities(value: object) -> dict[str, float]:
    if not isinstance(value, Mapping):
        msg = "FoliaScan endpoint response field 'probabilities' must be an object."
        raise EndpointResponseError(msg)

    probabilities_mapping = cast(Mapping[object, object], value)
    probabilities: dict[str, float] = {}
    for class_name, probability in probabilities_mapping.items():
        if not isinstance(class_name, str) or not class_name:
            msg = (
                "FoliaScan endpoint response field 'probabilities' must use "
                "non-empty class names."
            )
            raise EndpointResponseError(msg)
        probabilities[class_name] = _required_float(
            probability,
            f"probabilities.{class_name}",
        )

    if not probabilities:
        msg = "FoliaScan endpoint response field 'probabilities' must not be empty."
        raise EndpointResponseError(msg)
    return probabilities


__all__ = [
    "AzureEndpointClient",
    "DEFAULT_TIMEOUT_SECONDS",
    "ENDPOINT_KEY_ENVIRONMENT_VARIABLE",
    "ENDPOINT_URL_ENVIRONMENT_VARIABLE",
    "EndpointConfigurationError",
    "EndpointRequestError",
    "EndpointResponseError",
    "FoliaScanEndpointError",
    "PredictionResponse",
]
