# Backend

The backend exposes the API used by the frontend to upload oral images, run screening-support inference, and return structured results.

The current implementation uses deterministic mock inference. That gives the frontend and API tests a stable contract while the real ML model is built.

## Architecture

```text
app/
  main.py                  FastAPI app factory and dependency wiring
  config.py                Typed settings loaded from ORALLENS_* env vars
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

## Quality Checklist

### Is the implementation secure?

For this first backend slice, yes for local portfolio scope. It validates file type, extension, file signature, and size. It avoids path traversal by not writing user-provided filenames to disk.

### Is the code clean and efficiently written?

The code separates routing, settings, schemas, storage, service orchestration, and inference adapter logic. The app factory allows tests to inject temporary storage.

### Is the documentation clear?

This README documents the architecture, endpoints, security decisions, and learning goals. More API details will be added after the frontend and real ML pipeline connect.

### Are there enough tests?

The first test set covers health, valid image upload, unsupported file type, oversized upload, file signature mismatch, scan listing, scan detail, and missing scan behavior.

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

Backend skeleton implemented with mock inference and tests.
