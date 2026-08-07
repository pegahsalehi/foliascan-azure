# FoliaScan Client Application

The FoliaScan client application is a small Streamlit interface for invoking a
previously deployed Azure ML Managed Online Endpoint. It is intentionally a thin
presentation layer: image upload, preview, endpoint invocation, prediction
display, and clear user-facing errors.

The application does not create Azure resources, deploy endpoints, train models,
store uploads, keep history, or authenticate users. The Phase 8 cloud endpoint
was validated end to end and then deleted to avoid ongoing Azure cost.

## Client Architecture

The application has two layers:

- `app/streamlit_app.py`: Streamlit UI, asset loading, image preview, result
  presentation, and display-only class-name formatting.
- `src/foliascan/client/azure_endpoint.py`: reusable HTTPS client for the
  Azure ML endpoint request and response contract.

The endpoint client uses ordinary HTTPS through Python standard-library
networking. It does not require Azure SDK packages for inference. The Streamlit
app imports the client but does not create the client or send a request until a
user uploads an image and selects **Analyze**.

## Environment Variables

Two environment variables configure inference:

```powershell
$env:FOLIASCAN_ENDPOINT_URL = "<managed-online-endpoint-scoring-url>"
$env:FOLIASCAN_ENDPOINT_KEY = "<managed-online-endpoint-key>"
```

Do not commit real values. Do not print them in logs, screenshots, notebooks, or
documentation. The app displays only application-level errors and prediction
results; it does not expose endpoint URLs, endpoint keys, Base64 payloads,
subscription identifiers, or Azure resource identifiers.

## Local Streamlit Execution

Run the app locally from the repository root after installing dependencies:

```powershell
poetry install
poetry run streamlit run app/streamlit_app.py
```

If the endpoint environment variables are missing, the app still starts and
allows image upload. Configuration errors appear only when **Analyze** is used.

## Request Flow

The workflow is deliberately simple:

1. The user uploads one JPEG or PNG tomato-leaf image.
2. Streamlit previews the uploaded image.
3. The user selects **Analyze**.
4. The app reads endpoint configuration from environment variables.
5. The client Base64-encodes the image bytes.
6. The client sends a POST request to the endpoint.
7. The client validates the JSON response.
8. The UI displays the prediction, confidence, top three probabilities, and an
   expandable table of all class probabilities.

The request body follows the Phase 7 endpoint contract:

```json
{
  "image_base64": "<raw base64 JPEG or PNG>"
}
```

The response is expected to contain:

```json
{
  "predicted_class": "<class name>",
  "predicted_index": 0,
  "confidence": 0.0,
  "probabilities": {
    "<class name>": 0.0
  }
}
```

Class names are humanized only in the UI. The backend response and endpoint
contract remain unchanged.

## Error Handling

The endpoint client raises clear application-level errors for:

- missing endpoint URL
- missing endpoint key
- empty image bytes
- network failure
- timeout
- HTTP 4xx or 5xx status
- invalid JSON response
- response missing required prediction fields
- invalid response field types

The Streamlit app catches these errors and shows concise user-facing messages
without exposing secrets or Azure identifiers.

## End-To-End Validation

Phase 8 validation included a successful real flow from the Streamlit app to an
Azure ML Managed Online Endpoint and back to the UI. That proved the uploaded
image path, Base64 request body, authorization header, endpoint scoring script,
prediction response validation, and Streamlit result presentation worked
together.

The cloud endpoint was deleted after validation to avoid ongoing cost. Reusing
the app later requires a newly deployed compatible Azure ML endpoint and fresh
local environment variables.

## Security Considerations

The endpoint key is read from the local environment and is never hard-coded. The
client dataclass excludes it from `repr`, and neither the UI nor tests log it.

The app does not persist uploaded images or predictions. Streamlit still holds
uploaded bytes in the active session while the app is running, so avoid uploading
sensitive images to untrusted environments.

Keep screenshots and demos free of browser developer tools or terminal windows
that could reveal configuration values. Rotate endpoint keys if they are ever
shown accidentally.
