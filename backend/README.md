# OralLens AI Backend

FastAPI service for validated oral-image uploads, configurable mock or real-model inference, structured screening-support reports, and local scan persistence.

The backend is part of an experimental portfolio project. Its output is not a diagnosis, clinical validation, medical-device result, or treatment recommendation.

## Responsibilities

- expose health and scan HTTP contracts
- validate upload metadata, size, extension, and file signature
- coordinate inference through a stable adapter protocol
- translate prediction output into typed evidence and report schemas
- persist local scan records atomically
- return structured, client-safe errors with request IDs

The step-by-step request and model flow is in [Pipeline](../docs/PIPELINE.md). System boundaries are in [Architecture](../docs/ARCHITECTURE.md), and endpoint details are in [API](../docs/API.md).

## Structure

```text
backend/
  app/
    api/             HTTP routes
    pipeline/        mock and ML inference adapters
    services/        upload, inference, report, and storage orchestration
    config.py        typed ORALLENS_ settings
    schemas.py       request/response data contracts
    storage.py       local JSON scan store
    error_handlers.py
    middleware.py
    main.py          application factory and dependency wiring
  scripts/           real-ML launchers
  tests/             unit, API, CORS, config, storage, and integration coverage
```

## Runtime modes

`ORALLENS_INFERENCE_MODE` accepts:

- `mock` — default deterministic placeholder for lightweight development and tests
- `ml` — v3 TorchVision plaque-candidate detector through the project-local ML package

The default v3 ML config is:

`ml/configs/orthodontic_plaque_detection_mvp_v3_predict.toml`

The backend does not hard-code checkpoint logic in routes. Model selection remains behind typed configuration and the inference adapter.

## Run with the real v3 detector

Prerequisites:

- project-local `ml/.venv` with backend and ML runtime dependencies
- local v3 checkpoint at the path referenced by the prediction config
- prepared project-local data only when running the integration smoke

From the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

The script selects `ml` mode, uses the v3 config, and starts Uvicorn on `http://127.0.0.1:8000`.

## Run in mock mode

From the repository root, with backend dependencies available in the project-local environment:

```powershell
ml\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

No inference environment override is required because `mock` is the safe default.

## Configuration

Settings use the `ORALLENS_` prefix and may be loaded from a local `.env`, which is excluded from git.

| Setting | Default | Purpose |
|---|---|---|
| `ORALLENS_INFERENCE_MODE` | `mock` | inference adapter selection |
| `ORALLENS_MAX_UPLOAD_BYTES` | `5242880` | upload-size boundary |
| `ORALLENS_STORAGE_PATH` | `backend/var/scans.json` | local record store |
| `ORALLENS_ML_SOURCE_PATH` | `ml/src` | project-local ML import boundary |
| `ORALLENS_ML_DETECTION_CONFIG_PATH` | v3 predict config | checkpoint, model identity, and inference settings |
| `ORALLENS_ML_TEMP_DIR` | `backend/var/ml-inputs` | constrained temporary inputs |
| `ORALLENS_LOG_LEVEL` | `INFO` | application logging level |

Local CORS origins are explicitly limited to `http://127.0.0.1:5173` and `http://localhost:5173`.

## Upload behavior

The general backend boundary recognizes JPEG, PNG, and WebP metadata/signatures. The real ML adapter intentionally supports JPEG and PNG only and translates WebP to a concise HTTP `415`. The frontend is aligned to that real-model subset.

The service:

1. checks MIME type and filename extension
2. reads the upload in 1 MiB chunks while enforcing the total limit
3. rejects empty input and signature mismatches
4. hashes validated bytes
5. runs the selected adapter
6. stores a typed scan record

User-provided filenames are reduced to safe display basenames and never used as storage paths.

## Tests

Complete backend suite from the repository root:

```powershell
ml\.venv\Scripts\python.exe -B -m pytest backend\tests
```

Latest result: `24 passed, 1 skipped`. The skip is the opt-in real-model integration test.

Run that smoke explicitly when the dataset, v3 checkpoint, and GPU/CPU ML runtime are available:

```powershell
$env:ORALLENS_RUN_ML_INTEGRATION = "1"
ml\.venv\Scripts\python.exe -B -m pytest backend\tests\test_ml_integration_smoke.py
```

Latest real integration result: `1 passed`.

Coverage includes uploads, signature mismatches, size limits, filenames, storage, schemas, configured/unconfigured CORS behavior, ML adapter selection, explicit model identity, detector-score semantics, unsupported ML input, cleanup, error boundaries, and real checkpoint-backed output.

## Local data and production boundary

`backend/var` is ignored and intentionally retains local scan evidence. `JSONScanStore` is appropriate for this single-user demonstration, not for protected health information, multiple processes, or external deployment.

Before deployment, add authentication, authorization, rate limiting, encrypted policy-governed storage, retention/deletion controls, observability, deployment configuration, and security review. These are documented gaps, not implied features.
