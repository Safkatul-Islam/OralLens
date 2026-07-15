# Backend

The backend exposes the API used by the frontend to upload oral images, run screening-support inference, and return structured results.

The current implementation uses deterministic mock inference behind the same
response contract the trained detector will use. That gives the frontend and API
tests a stable contract while keeping the heavy ML runtime outside the backend
MVP process.

## Architecture

```text
app/
  main.py                  FastAPI app factory and dependency wiring
  config.py                Typed settings loaded from ORALLENS_* env vars
  error_handlers.py        Safe, consistent public error responses
  logging_config.py        JSON application logging
  middleware.py            Request IDs, timing, and request logs
  schemas.py               Pydantic request/response models
  storage.py               JSON-backed local scan store
  api/
    health.py              GET /health
    scans.py               scan upload, list, and detail endpoints
  services/
    scan_service.py        upload validation, inference orchestration, reporting
  pipeline/
    inference.py           inference protocol and deterministic mock adapter
tests/
  test_hardening.py
  test_health.py
  test_scans.py
```

## API Endpoints

```text
GET  /health
POST /scans
GET  /scans
GET  /scans/{scan_id}
```

## Security Decisions

- Uploads are accepted only as JPEG, PNG, or WebP.
- File extension, declared content type, and image magic bytes are validated.
- Upload size is capped by `ORALLENS_MAX_UPLOAD_BYTES`.
- User filenames are never used as filesystem paths.
- The backend persists scan metadata and hashes, not raw uploaded images. ML mode
  uses a controlled temporary image file during inference and deletes it after
  the adapter returns.
- User uploads cannot choose model paths, checkpoint paths, or output paths.
- Error responses avoid leaking stack traces or internal paths.
- CORS uses an explicit origin allowlist and does not default to `*`.
- Request IDs are validated before they are echoed or logged.
- Request logs exclude bodies, uploaded bytes, query strings, and credentials.

## Runtime Configuration

Settings use the `ORALLENS_` environment prefix. Important values include:

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

See `../docs/API.md` for the endpoint contract, validation behavior, error schema, and CORS details.

## ML Inference Contract

The backend scan service depends on a small inference interface:

```text
predict(image_bytes, content_type) -> InferenceResult
```

The response includes:

- `label`, `display_name`, `confidence`, and `severity`
- `is_mock` and `model_name`
- `prediction_count`
- `detections[]` with `box_xyxy`, `label`, and `score`

The default adapter is deterministic mock inference, so `detections` is empty.
Set `ORALLENS_INFERENCE_MODE=ml` to use the optional local MVP detector adapter.
ML mode lazy-imports the project-local `orallens_ml` package, writes validated
upload bytes to a controlled backend temp directory, calls the configured
checkpoint-backed prediction config, and maps returned boxes into the same API
schema.

Current ML mode supports JPEG and PNG uploads. WebP remains accepted in mock
mode, but returns a clean unsupported-media response in ML mode until the ML
image loader supports it.

Start the local backend in ML mode from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

## Local ML Integration Smoke

The real backend-to-ML smoke requires one Python environment that can import both
the backend package and the project-local ML package. The latest local smoke used
`ml/.venv` with the backend runtime package installed into that project-local
environment.

Smoke result:

```text
POST /scans
inference_mode: ml
status: 201
model_name: orthodontic-plaque-mvp
is_mock: false
prediction_count: 25
first_score: 0.19762550294399261
```

The adapter deleted the temporary uploaded image copy after inference. The scan
metadata was written to `backend/var/integration-smoke/scans.json`; model
prediction artifacts remain under ignored `ml/runs/detection/` outputs.

Run the optional integration smoke from the repository root only after the MVP
checkpoint exists:

```powershell
$env:ORALLENS_RUN_ML_INTEGRATION = "1"
ml\.venv\Scripts\python.exe -m pytest backend\tests\test_ml_integration_smoke.py --basetemp "ml\.uv-cache\pytest-backend-ml" -p no:cacheprovider
```

The normal backend suite skips this smoke:

```powershell
ml\.venv\Scripts\python.exe -m pytest backend\tests --basetemp "ml\.uv-cache\pytest-backend" -p no:cacheprovider
```

## Quality Checklist

### Is the implementation secure?

For this backend slice, yes for local portfolio scope. It validates file type, extension, file signature, size, request IDs, and allowed browser origins. It avoids path traversal and returns generic messages for unexpected failures.

### Is the code clean and efficiently written?

The code separates routing, settings, schemas, middleware, error handling, logging, storage, service orchestration, and inference adapter logic. The scan service depends on an inference protocol rather than a concrete model runtime. The app factory allows tests to inject temporary storage and environment-specific settings.

### Is the documentation clear?

This README documents architecture and operations. The project-level API contract documents endpoints, validation, errors, configuration, CORS, request IDs, and logging safety.

### Are there enough tests?

Tests cover health, image upload validation, scan persistence, inference response contract, CORS allowlisting, request ID handling, safe validation errors, and unexpected-error redaction.

## Local Development

Install dependencies from inside `backend/` using a project-local virtual environment.

```text
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Run tests:

```text
pytest
```

Run the API:

```text
fastapi dev app/main.py
```

## Learning Goals

- Understand what an API does
- Learn request and response schemas
- Learn input validation
- Learn backend error handling
- Learn how a backend calls ML code
- Learn how tests protect behavior

## Status

Backend skeleton hardened with explicit CORS, structured logging, request IDs, centralized error responses, a stable inference contract, mock adapter, and tests.
