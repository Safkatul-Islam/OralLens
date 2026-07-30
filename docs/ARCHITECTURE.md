# OralLens AI Architecture

## Purpose

This document is the architectural source of truth for the current OralLens AI portfolio MVP. It explains component boundaries and which boundaries would need to change before a more trustworthy or deployable system could be claimed. For the numbered operational flow and direct source-file locator, see [Pipeline](PIPELINE.md).

OralLens AI provides experimental screening support only. The architecture does not confer clinical validity or medical-device status.

## System context

```text
User
  |
  v
React + TypeScript + Vite
  |  POST /scans (multipart/form-data)
  v
FastAPI route
  |
  v
ScanService ----------------------------------+
  |                                           |
  +-> upload validation                       +-> report construction
  +-> content digest                          +-> JSONScanStore
  |
  v
InferencePipeline protocol
  |
  +-> MockInferencePipeline
  |
  +-> MLDetectionInferencePipeline
          |
          v
     orallens_ml inference
          |
          v
     Faster R-CNN checkpoint + v2 predict config
```

## Component responsibilities

| Component | Responsibility | Deliberately does not own |
|---|---|---|
| Frontend | File selection, preview, request submission, result presentation, box overlay | Model execution, clinical interpretation, persistent storage |
| API routes | HTTP contracts and dependency access | Business logic and ML details |
| `ScanService` | Upload validation, hashing, inference orchestration, report construction | Framework startup and detector internals |
| Inference adapters | Stable backend-to-inference contract; mock or ML implementation | HTTP responses and storage |
| `JSONScanStore` | Atomic local scan-record persistence | Multi-user concurrency, authentication, regulated retention |
| ML package | Dataset contracts, target conversion, model construction, training, evaluation, and prediction | Browser/API behavior |
| TOML configs | Reproducible experiment and inference parameters | Source data or secrets |

## Request lifecycle

1. The frontend accepts JPEG or PNG and creates a multipart request.
2. FastAPI receives the upload at `POST /scans`.
3. `ScanService` checks the declared MIME type and extension, reads in bounded chunks, enforces the size limit, rejects empty content, and checks the file signature.
4. The service hashes the validated bytes with SHA-256.
5. The selected inference adapter runs:
   - `mock` returns a deterministic placeholder based on the input hash.
   - `ml` writes a UUID-named temporary JPEG/PNG within the configured directory, loads the v2 inference config, runs the detector, and removes the temporary input in a `finally` block.
6. The service converts inference output into typed prediction, evidence, report, limitation, and disclaimer schemas.
7. `JSONScanStore` appends the record through a temporary-file replacement.
8. The API returns HTTP `201`; the frontend renders metadata and overlays each `xyxy` detection box on the natural image coordinate system.

## Runtime modes and configuration

The backend uses typed `ORALLENS_` environment settings.

| Mode | Selection | Intended use |
|---|---|---|
| `mock` | Default `ORALLENS_INFERENCE_MODE` | Fast API development and deterministic tests without loading Torch |
| `ml` | Set by `backend\scripts\run-ml-server.ps1` or `.cmd` | Local GPU/CPU inference with the trained v2 detector |

The current default ML config boundary is:

`ml/configs/orthodontic_plaque_detection_mvp_v2_predict.toml`

That config points to the ignored local v2 checkpoint, fixes the score threshold at `0.65`, and caps returned detections at 25. Model promotion is performed through configuration/startup boundaries rather than route changes.

## ML architecture and data flow

The detector is TorchVision Faster R-CNN with ResNet-50 FPN and two classes: background and plaque candidate. Prepared manifest rows reference images under the controlled dataset root and carry normalized center-width-height annotations.

```text
verified source material
        |
        v
patient-aware prepared manifest
        |
        +-> dataset validation
        +-> normalized box conversion and bounded clipping policy
        v
TorchVision Dataset/DataLoader
        |
        +-> training -> state-dict checkpoint + loss metrics
        +-> validation -> threshold sweep
        +-> held-out test -> one frozen-threshold evaluation
        +-> prediction -> local JSON artifact
```

The validation split selected threshold `0.65`. The held-out test used exactly that threshold and must not be used for further tuning.

## Trust and security boundaries

### External input

- The backend does not trust user filenames, extensions, MIME declarations, or raw bytes independently.
- Upload size is bounded while reading, not only after buffering.
- The real ML adapter supports JPEG and PNG only; unsupported formats fail with a concise `415` response.
- Pydantic schemas constrain scores, labels, counts, and record shape.

### Filesystem

- Display filenames are reduced to a basename.
- Temporary ML files use generated names inside a configured directory.
- The ML source and temporary directories reject unsafe symlink/path behavior.
- Dataset paths must be safe relative POSIX paths contained by the configured root; missing files and symlinks are rejected.
- Datasets, checkpoints, run output, local scans, and temporary inputs are ignored by git.

### Model and evaluation

- Checkpoints are state dictionaries loaded with `weights_only=True`.
- Annotation fields must be finite, normalized, and structurally valid.
- Derived corners may exceed the boundary only within the documented `1e-6` tolerance, after which boxes are clipped and rechecked for positive area.
- Grossly invalid or degenerate boxes fail closed.
- Shared target-validation failures are translated into evaluation-domain errors so expected contract failures do not expose raw tracebacks.
- Validation chooses the operating point; the test split measures it once.

### API and browser

- CORS uses explicit local origins: `http://127.0.0.1:5173` and `http://localhost:5173`.
- Error handlers return structured messages and request IDs without exposing internal exception details.
- The UI repeats the non-diagnostic limitations and directs health concerns to a licensed professional.

## Persistence and artifacts

`JSONScanStore` is a local demonstration store under `backend/var`. ML checkpoints, metrics, predictions, and prepared/raw data live under ignored `ml` directories. These artifacts support reproducibility and local evidence, but are not committed because they may be large, machine-specific, or derived from restricted data.

The current storage design has no authentication, encryption-at-rest policy, retention controls, audit log, database migrations, or multi-process coordination. It must not be used for sensitive clinical records.

## Failure boundaries

Expected failures are translated at the closest appropriate boundary:

- upload contract failures -> concise HTTP `400`, `413`, or `415`
- missing scan -> HTTP `404`
- unsupported ML input -> upload-domain `415`
- internal inference failure -> generic API error without internal details
- dataset/target failure during evaluation -> evaluation-domain error and concise CLI exit

Unexpected failures remain observable to developers through controlled logging, while user-facing output avoids raw stack traces and local paths.

## Deployment boundary

The current system is a local portfolio MVP, not a production deployment design. Before broader use, the architecture would need at least authenticated access, encrypted and policy-governed storage, explicit data retention, observability, rate limiting, worker/process-safe persistence, packaged model/version provenance, environment separation, backup/recovery, and independent security review.

Clinical or regulated use would additionally require a defined intended use, representative multi-site evidence, risk management, human-factors work, monitoring, quality systems, and applicable regulatory review. Those are outside the current project claim.

## Related documents

- [Pipeline](PIPELINE.md)
- [Project brief](PROJECT_BRIEF.md)
- [API contract](API.md)
- [Training and evaluation](TRAINING.md)
- [Dataset card](DATASET_CARD.md)
- [Data acquisition](DATA_ACQUISITION.md)
- [ML environment](ML_ENVIRONMENT.md)
- [Backend guide](../backend/README.md)
- [Frontend guide](../frontend/README.md)
- [ML guide](../ml/README.md)
