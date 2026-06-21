# Backend

The backend exposes the API used by the frontend to upload oral images, run screening-support inference, and return structured results.

The current implementation uses deterministic mock inference. That gives the frontend and API tests a stable contract while the real ML model is built.

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
    inference.py           deterministic mock inference adapter
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
- The backend stores scan metadata and hashes, not raw uploaded images.
- Error responses avoid leaking stack traces or internal paths.
- CORS uses an explicit origin allowlist and does not default to `*`.
- Request IDs are validated before they are echoed or logged.
- Request logs exclude bodies, uploaded bytes, query strings, and credentials.

## Runtime Configuration

Settings use the `ORALLENS_` environment prefix. Important values include:

```text
ORALLENS_ENVIRONMENT=local
ORALLENS_LOG_LEVEL=INFO
ORALLENS_MAX_UPLOAD_BYTES=5242880
ORALLENS_CORS_ALLOWED_ORIGINS=["http://localhost:5173"]
ORALLENS_CORS_ALLOW_CREDENTIALS=false
```

See `../docs/API.md` for the endpoint contract, validation behavior, error schema, and CORS details.

## Quality Checklist

### Is the implementation secure?

For this backend slice, yes for local portfolio scope. It validates file type, extension, file signature, size, request IDs, and allowed browser origins. It avoids path traversal and returns generic messages for unexpected failures.

### Is the code clean and efficiently written?

The code separates routing, settings, schemas, middleware, error handling, logging, storage, service orchestration, and inference adapter logic. The app factory allows tests to inject temporary storage and environment-specific settings.

### Is the documentation clear?

This README documents architecture and operations. The project-level API contract documents endpoints, validation, errors, configuration, CORS, request IDs, and logging safety.

### Are there enough tests?

Tests cover health, image upload validation, scan persistence, CORS allowlisting, request ID handling, safe validation errors, and unexpected-error redaction.

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

Backend skeleton hardened with explicit CORS, structured logging, request IDs, centralized error responses, mock inference, and tests.
