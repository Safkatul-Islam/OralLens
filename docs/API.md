# OralLens AI API Contract

## Purpose

The backend provides a stable HTTP contract between the frontend and the oral-image screening pipeline. The current adapter returns deterministic mock data behind the same response shape planned for the trained detector.

Default local base URL:

```text
http://127.0.0.1:8000
```

Interactive OpenAPI documentation is available at `/docs` while the API is running.

## Request IDs

Clients may send a request ID:

```text
X-Request-ID: portfolio-request-1
```

The value must contain 1-128 letters, numbers, dots, underscores, or hyphens. Invalid or missing IDs are replaced with a UUID. Every response returns the accepted/generated request ID in the same header.

## CORS

CORS uses an explicit allowlist. The default development origins are:

```text
http://127.0.0.1:5173
http://localhost:5173
```

Production origins must be configured with `ORALLENS_CORS_ALLOWED_ORIGINS` as a JSON array. Wildcard origins are not enabled by default.

## Error Schema

All handled errors use this shape:

```json
{
  "code": "validation_error",
  "detail": "Request validation failed.",
  "request_id": "f490f47b-fc04-449f-9a5a-dd0c80a2c25f"
}
```

Unexpected errors return HTTP 500 with a generic message. Stack traces, local paths, and exception details are not returned to clients.

## Endpoints

### `GET /health`

Returns service status, version, and environment.

### `POST /scans`

Accepts one multipart field named `file`.

Accepted formats in default mock mode:

- JPEG: `.jpg` or `.jpeg`
- PNG: `.png`
- WebP: `.webp`

ML mode currently supports JPEG and PNG. WebP receives HTTP 415 in ML mode until
the model-backed image loader supports it.

Validation checks:

- Declared content type
- Filename extension
- Image magic bytes
- Configured maximum upload size
- Non-empty content

Successful response: HTTP 201 with scan metadata, prediction metadata, optional detection boxes, evidence summary, and responsible-AI report.

Prediction shape:

```json
{
  "label": "possible_tartar_buildup",
  "display_name": "Possible tartar buildup",
  "confidence": 0.73,
  "severity": "medium",
  "is_mock": true,
  "model_name": "deterministic-mock-v1",
  "prediction_count": 0,
  "detections": []
}
```

Model-backed detection entries use:

```json
{
  "box_xyxy": [1.0, 2.0, 5.0, 6.0],
  "label": 1,
  "score": 0.42
}
```

The active backend adapter is still mock inference, so it returns an empty
`detections` array. The ML MVP checkpoint already writes the same detection
fields in its prediction JSON artifact.

Possible errors:

- `400`: empty content or content/signature mismatch
- `413`: upload exceeds configured limit
- `415`: unsupported content type or extension
- `422`: missing or structurally invalid request
- `500`: unexpected internal error with safe public message

Raw uploaded image bytes are not persisted. The local store contains metadata and
a SHA-256 hash. ML mode uses a controlled temporary image file during inference
and deletes it after the adapter returns.
Clients cannot provide model paths or checkpoint paths.

### `GET /scans`

Returns stored scan metadata in creation order.

### `GET /scans/{scan_id}`

Returns one scan record or HTTP 404 when the ID does not exist.

## Configuration

Settings use the `ORALLENS_` environment prefix.

```text
ORALLENS_ENVIRONMENT=local
ORALLENS_LOG_LEVEL=INFO
ORALLENS_INFERENCE_MODE=mock
ORALLENS_MAX_UPLOAD_BYTES=5242880
ORALLENS_CORS_ALLOWED_ORIGINS=["http://127.0.0.1:5173","http://localhost:5173"]
ORALLENS_CORS_ALLOW_CREDENTIALS=false
ORALLENS_ML_SOURCE_PATH=../ml/src
ORALLENS_ML_DETECTION_CONFIG_PATH=../ml/configs/orthodontic_plaque_detection_mvp_predict.toml
ORALLENS_ML_TEMP_DIR=var/ml-inputs
```

`ORALLENS_INFERENCE_MODE=ml` enables the optional local MVP detector adapter.
The backend still owns upload validation and temp-file placement; clients cannot
choose model paths, checkpoint paths, or prediction output directories.

The optional backend-to-ML smoke is skipped by default and must be enabled
explicitly from the repository root:

```powershell
$env:ORALLENS_RUN_ML_INTEGRATION = "1"
ml\.venv\Scripts\python.exe -m pytest backend\tests\test_ml_integration_smoke.py --basetemp "ml\.uv-cache\pytest-backend-ml" -p no:cacheprovider
```

## Logging Safety

Application logs are emitted as JSON. Request logs include method, URL path, status code, duration, and request ID. Request bodies, uploaded image bytes, authorization values, cookies, and query strings are not logged.
