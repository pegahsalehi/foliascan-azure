import base64
import json
import urllib.error
import urllib.request

import pytest

from foliascan.client.azure_endpoint import (
    DEFAULT_TIMEOUT_SECONDS,
    AzureEndpointClient,
    EndpointConfigurationError,
    EndpointRequestError,
    EndpointResponseError,
    PredictionResponse,
)


class FakeHTTPResponse:
    def __init__(self, body: bytes, status_code: int = 200) -> None:
        self._body = body
        self._status_code = status_code

    def __enter__(self) -> "FakeHTTPResponse":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        return None

    def getcode(self) -> int:
        return self._status_code

    def read(self) -> bytes:
        return self._body


def test_successful_request(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    response = client.predict(b"image bytes")

    assert response == _valid_prediction_response()
    assert len(calls) == 1


def test_generates_base64_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    client.predict(b"raw image bytes")

    request = calls[0][0]
    assert request.data is not None
    payload = json.loads(request.data.decode("utf-8"))
    assert payload == {
        "image_base64": base64.b64encode(b"raw image bytes").decode("ascii")
    }


def test_sets_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    client.predict(b"image bytes")

    request = calls[0][0]
    assert request.get_header("Authorization") == "Bearer secret-key"
    assert request.get_header("Content-type") == "application/json"


def test_configures_http_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
        timeout_seconds=12.5,
    )

    client.predict(b"image bytes")

    assert calls[0][1] == 12.5


def test_default_timeout_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    client.predict(b"image bytes")

    assert calls[0][1] == DEFAULT_TIMEOUT_SECONDS


def test_missing_endpoint_url_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FOLIASCAN_ENDPOINT_URL", raising=False)
    monkeypatch.setenv("FOLIASCAN_ENDPOINT_KEY", "secret-key")

    with pytest.raises(EndpointConfigurationError, match="FOLIASCAN_ENDPOINT_URL"):
        AzureEndpointClient.from_environment()


def test_missing_endpoint_key_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FOLIASCAN_ENDPOINT_URL", "https://example.test/score")
    monkeypatch.delenv("FOLIASCAN_ENDPOINT_KEY", raising=False)

    with pytest.raises(EndpointConfigurationError, match="FOLIASCAN_ENDPOINT_KEY"):
        AzureEndpointClient.from_environment()


def test_empty_image_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_urlopen(monkeypatch, _json_response(_valid_prediction_response()))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointRequestError, match="non-empty"):
        client.predict(b"")


def test_network_error_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointRequestError, match="Unable to reach"):
        client.predict(b"image bytes")


def test_timeout_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointRequestError, match="timed out"):
        client.predict(b"image bytes")


def test_non_success_http_status_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_urlopen(
        monkeypatch,
        b'{"error": "failed"}',
        status_code=503,
    )
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointResponseError, match="HTTP 503"):
        client.predict(b"image bytes")


def test_invalid_json_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_urlopen(monkeypatch, b"not json")
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointResponseError, match="invalid JSON"):
        client.predict(b"image bytes")


@pytest.mark.parametrize(
    "field_name",
    [
        "predicted_class",
        "predicted_index",
        "confidence",
        "probabilities",
    ],
)
def test_missing_required_response_fields_fail_clearly(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
) -> None:
    response = _valid_prediction_response()
    del response[field_name]
    _mock_urlopen(monkeypatch, _json_response(response))
    client = AzureEndpointClient(
        endpoint_url="https://example.test/score",
        endpoint_key="secret-key",
    )

    with pytest.raises(EndpointResponseError, match=field_name):
        client.predict(b"image bytes")


def _mock_urlopen(
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    *,
    status_code: int = 200,
) -> list[tuple[urllib.request.Request, float]]:
    calls: list[tuple[urllib.request.Request, float]] = []

    def fake_urlopen(
        request: urllib.request.Request,
        *,
        timeout: float,
    ) -> FakeHTTPResponse:
        calls.append((request, timeout))
        return FakeHTTPResponse(body, status_code=status_code)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return calls


def _json_response(response: dict[str, object]) -> bytes:
    return json.dumps(response).encode("utf-8")


def _valid_prediction_response() -> PredictionResponse:
    return {
        "predicted_class": "Tomato healthy",
        "predicted_index": 1,
        "confidence": 0.875,
        "probabilities": {
            "Tomato healthy": 0.875,
            "Tomato early blight": 0.125,
        },
    }
