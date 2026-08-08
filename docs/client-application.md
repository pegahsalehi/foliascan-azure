# FoliaScan Client Application

FoliaScan includes a lightweight Streamlit interface for sending tomato-leaf images to an Azure ML Managed Online Endpoint and displaying the prediction results.

The application is intentionally simple: it handles image upload, preview, endpoint invocation, result presentation, and user-facing errors. It does not create Azure resources, deploy endpoints, train models, persist uploads, or manage users.

## Architecture

The client has two layers:

- `app/streamlit_app.py` — Streamlit UI, image preview, result presentation, branding, and display-only class-name formatting
- `src/foliascan/client/azure_endpoint.py` — reusable HTTPS client for the Azure ML endpoint

The endpoint client uses standard HTTPS and does not require the Azure SDK for inference.

The Streamlit app does not send a request until the user uploads an image and selects **Analyze**.

## Configuration

Inference requires two environment variables:

```powershell
$env:FOLIASCAN_ENDPOINT_URL = "<managed-online-endpoint-scoring-url>"
$env:FOLIASCAN_ENDPOINT_KEY = "<managed-online-endpoint-key>"
```

Real values should never be committed to the repository or exposed in logs, screenshots, or documentation.

The application does not display endpoint URLs, endpoint keys, Base64 payloads, subscription identifiers, or Azure resource identifiers.

## Run Locally

Install dependencies and start Streamlit from the repository root:

```powershell
poetry install
poetry run streamlit run app/streamlit_app.py
```

The app can start without endpoint configuration. Missing configuration is reported only when **Analyze** is used.

## Request Flow

The inference flow is:

1. Upload one JPEG or PNG tomato-leaf image.
2. Preview the image in Streamlit.
3. Select **Analyze**.
4. Read the endpoint URL and key from environment variables.
5. Base64-encode the uploaded image.
6. Send an authenticated POST request to the Azure ML endpoint.
7. Validate the JSON response.
8. Display the prediction, confidence, top three probabilities, and the full probability table.

The request body is:

```json
{
  "image_base64": "<raw base64 JPEG or PNG>"
}
```

The expected response contains:

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

Internal model labels are converted to user-friendly names only in the presentation layer. For example:

```text
Tomato___Late_blight
→
Late Blight
```

The endpoint contract itself remains unchanged.

## Error Handling

The endpoint client handles:

- missing endpoint configuration
- empty image input
- network failures
- request timeouts
- HTTP errors
- invalid JSON
- missing prediction fields
- invalid response field types

The Streamlit app catches these errors and presents concise messages without exposing sensitive configuration.

## End-to-End Validation

The application was tested against a real Azure ML Managed Online Endpoint.

The complete flow was validated:

```text
Streamlit
   ↓
HTTPS client
   ↓
Azure ML Managed Online Endpoint
   ↓
model inference
   ↓
prediction response
   ↓
Streamlit result
```

After validation, the cloud endpoint was deleted to avoid ongoing compute cost.

Running real inference again requires a compatible deployed endpoint and fresh local environment variables.

## Security

The endpoint key is read from the local environment and is never hard-coded.

The client excludes the key from its object representation, and neither the UI nor tests log it.

Uploaded images and prediction results are not intentionally persisted by the application. Uploaded bytes remain available in the active Streamlit session while the app is running.

Avoid including credentials, endpoint details, or sensitive configuration in screenshots and demos.