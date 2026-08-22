# OralLens AI Backend

FastAPI backend for the OralLens AI experimental screening-support workflow.

The backend validates uploads, selects mock or model-backed inference, creates structured evidence/report content, and persists local scan records. It does not provide diagnosis or treatment recommendations.

## Structure

```text
backend/
  app/
    api/                 HTTP routes
    pipeline/            mock and ML inference boundary
    services/            scan orchestration and validation
    config.py            typed ORALLENS_ settings
    schemas.py           response contracts
    storage.py           local JSON persistence
    main.py              application wiring
  scripts/               local startup scripts
  tests/                 API, config, storage, and integration tests
```

## Runtime modes

| Mode | Purpose |
|---|---|
| `mock` | deterministic API development without loading Torch |
| `ml` | frozen Faster R-CNN inference through the project-local ML package |

The default application mode is `mock`. Both real-ML startup scripts set `ml` and select:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml`

That config points to the frozen epoch-9 checkpoint, threshold `0.80`, and model identity `orthodontic-plaque-mvp-v4-originals-online-aug-epoch9`.

## Run

From the repository root, start the lightweight default backend:

```powershell
ml\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

Start the model-backed backend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

The ML workflow requires the local prepared dataset, frozen checkpoint, and `ml\.venv`. Neither server should be assumed to be running when a session starts.

## Configuration

Settings use the `ORALLENS_` prefix.

| Setting | Default | Purpose |
|---|---|---|
| `ORALLENS_INFERENCE_MODE` | `mock` | selects mock or ML adapter |
| `ORALLENS_MAX_UPLOAD_BYTES` | 5 MiB | bounded upload size |
| `ORALLENS_STORAGE_PATH` | `backend/var/scans.json` | local scan records |
| `ORALLENS_ML_SOURCE_PATH` | `ml/src` | project-local ML package |
| `ORALLENS_ML_DETECTION_CONFIG_PATH` | final Experiment 1 predict config | checkpoint and inference contract |
| `ORALLENS_ML_TEMP_DIR` | `backend/var/ml-inputs` | contained temporary uploads |

Environment-specific values belong in ignored environment files, not source control.

## Request behavior

`POST /scans`:

1. validates MIME type and filename extension
2. reads content in bounded chunks
3. rejects empty or oversized uploads
4. verifies JPEG/PNG/WebP magic bytes
5. hashes the validated content
6. invokes the selected inference adapter
7. constructs typed evidence and report content
8. saves and returns a `ScanRecord`

The ML adapter supports JPEG and PNG. It writes a UUID temporary file below the configured directory, verifies containment, invokes the ML package, and removes the input in `finally`.

Production class semantics:

- detector label `1`: plaque-positive peri-tooth region candidate
- machine-compatible response label: `possible_plaque`
- zero boxes: `no_detection`, not a plaque-free finding
- score: experimental detector ranking value, not clinical confidence

## Input assessment

ML mode assesses decoded dimensions, luminance, and contrast before model construction. Unsupported images return HTTP `201` with `prediction = null`, reason codes, and retake guidance. This prevents an unsupported image from being represented as a negative result.

The assessment does not prove mouth/teeth presence, oral framing, absence of glare/blur, or semantic in-distribution status.

## Storage

`JSONScanStore` uses an in-process lock and temporary-file replacement. Stored fields include display filename, metadata, digest, assessment, predictions, and report. Image bytes are not stored by the scan store.

`backend/var` is ignored by git. The store is local demonstration infrastructure, not authenticated, encrypted, multi-user, or regulated storage.

## Tests

Run the backend suite with workspace-local Python:

```powershell
ml\.venv\Scripts\python.exe -B -m pytest backend\tests
```

Run the opt-in real-model smoke when the local checkpoint and dataset are available:

```powershell
$env:ORALLENS_RUN_ML_INTEGRATION = "1"
ml\.venv\Scripts\python.exe -B -m pytest backend\tests\test_ml_integration_smoke.py
```

Latest complete backend result: `32 passed, 1 skipped`. The skipped case is the opt-in real-model integration test.

Final promotion verification also includes 13 focused deployment/inference contract cases and one separately enabled real model-backed `POST /scans` smoke. The smoke returned 14 class-1 detections, all above `0.80`, and verified temporary-input cleanup.

## Security boundaries

- explicit CORS origins, methods, and headers
- bounded reads and upload-size enforcement
- MIME, extension, and signature agreement
- basename-only display filenames
- constrained temporary paths and symlink rejection
- generic external errors without raw stack traces or local paths
- typed settings and response schemas
- safe checkpoint loading and architecture compatibility in the ML package

See [API](../docs/API.md), [Pipeline](../docs/PIPELINE.md), and [Architecture](../docs/ARCHITECTURE.md).
