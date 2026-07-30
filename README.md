# OralLens AI

OralLens AI is an end-to-end portfolio project for **experimental oral-image screening support**. A React interface uploads a JPEG or PNG image to a FastAPI API, which can run a TorchVision Faster R-CNN detector and return candidate plaque boxes, confidence information, evidence text, limitations, and suggested next steps.

> OralLens AI is a learning and engineering project. It is not a diagnostic system, a clinically validated model, a medical device, or a source of treatment recommendations.

## Current status

The local MVP works end to end:

1. The browser validates the selected file type and shows an image preview.
2. `POST /scans` validates upload metadata, size, extension, and file signature.
3. The backend invokes either a deterministic mock adapter or the project-local v2 ML detector.
4. The API stores a scan record and returns structured detections and report content.
5. The frontend overlays the returned boxes and presents the screening-support result.

The GPU-backed workflow has been manually verified from the browser through FastAPI and the real v2 checkpoint. The latest verification returned HTTP `201`, rendered 17 candidate boxes, and displayed the expected limitations and disclaimer. Generated scan and prediction artifacts remain local evidence and are excluded from version control.

## Model evidence snapshot

The current detector is a binary TorchVision Faster R-CNN with a ResNet-50 FPN backbone:

- class `0`: background
- class `1`: plaque candidate
- training data: verified Part 2 orthodontic-plaque dataset
- split policy: patient-aware train, validation, and test groups
- operating threshold: `0.65`, selected on the complete validation split
- held-out test at IoU `0.5`: precision `0.7582`, recall `0.7037`, F1 `0.7299`
- mean IoU for matched held-out detections: `0.7541`

These are dataset-specific object-detection results, not clinical-performance claims. See [Training and evaluation](docs/TRAINING.md) for the complete experiment record and [Dataset card](docs/DATASET_CARD.md) for data limitations.

## Architecture

```text
React/Vite UI
     |
     | multipart image upload
     v
FastAPI routes -> ScanService -> validation -> inference adapter
                                      |             |
                                      |             +-> mock adapter
                                      |             +-> TorchVision detector
                                      v
                               JSON scan store
                                      |
                                      v
                      boxes + evidence + report -> UI
```

Responsibilities are separated across routes, schemas, services, storage, backend-to-ML integration, and the project-local ML package. See [Pipeline](docs/PIPELINE.md) for the step-by-step operational flow and [Architecture](docs/ARCHITECTURE.md) for system boundaries and deployment context.

## Repository layout

```text
backend/   FastAPI application, services, storage, integration adapter, and tests
frontend/  React/TypeScript/Vite screening-support interface
ml/        dataset contracts, detector training/evaluation/inference, configs, and tests
docs/      architecture, API, dataset, training, environment, and decision records
```

Datasets, checkpoints, run output, local scan records, virtual environments, and frontend build output are intentionally ignored by git.

## Run the local MVP

The real-ML workflow requires the prepared dataset, the trained v2 checkpoint, and the project-local `ml\.venv`. Setup and reproducibility details are in [ML environment](docs/ML_ENVIRONMENT.md).

From the repository root, start the ML-backed API:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

In a second terminal, start the frontend:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend\scripts\run-dev.ps1
```

Open `http://127.0.0.1:5173`, select a JPEG or PNG image, and run a scan. The backend listens on `http://127.0.0.1:8000`.

For the lightweight backend without loading the ML model, use its default `mock` inference mode as described in [Backend guide](backend/README.md).

## Verification status

Latest completed checks:

- ML suite: `132 passed, 1 skipped`
- backend suite: `23 passed, 1 skipped`
- real backend-to-ML integration smoke: `1 passed`
- frontend production build: passed
- manual browser-to-backend-to-v2 scan: passed

The skipped ML test requires Windows symbolic-link privileges. The skipped backend test is an opt-in real-ML integration case that passes when explicitly enabled.

## Security and reliability controls

- explicit CORS origins for both local loopback hostnames
- bounded upload size and chunked reads
- MIME type, extension, and magic-byte validation
- sanitized display filenames and SHA-256 content digests
- constrained temporary input paths with symlink and containment checks
- state-dict-only checkpoint loading with `weights_only=True`
- strict manifest path, annotation, numeric, file, and symlink validation
- expected errors translated at API and evaluation boundaries instead of exposing raw tracebacks
- frozen held-out threshold; test results are not used for retuning

These controls make the portfolio MVP more disciplined; they do not make it a production medical system.

## Known limitations and next phase

The model was trained on one specialized orthodontic-plaque dataset and has not been validated across clinics, devices, demographics, acquisition conditions, or clinically representative tasks. Confidence scores are detector scores, not calibrated probabilities of disease. Local JSON storage is suitable for demonstration, not multi-user deployment or sensitive clinical records.

The next phase is to make trust more measurable through data-quality reporting, robustness and subgroup evaluation where metadata permits, calibration analysis, failure-case review, stronger privacy and deployment boundaries, and documented human oversight. The goal is earned reliability for an experimental research prototype—not a clinical claim.

## Documentation map

- [Architecture](docs/ARCHITECTURE.md) — components, data flow, runtime modes, and trust boundaries
- [Pipeline](docs/PIPELINE.md) — end-to-end runtime, ML lifecycle, artifacts, and source ownership
- [Project brief](docs/PROJECT_BRIEF.md) — product scope, achieved MVP, exclusions, and next phase
- [API](docs/API.md) — endpoint contracts and error behavior
- [Training and evaluation](docs/TRAINING.md) — experiments, threshold selection, and held-out results
- [Dataset card](docs/DATASET_CARD.md) — provenance, splits, validation, and limitations
- [Data acquisition](docs/DATA_ACQUISITION.md) — controlled preparation workflow
- [ML environment](docs/ML_ENVIRONMENT.md) — reproducible local environment and commands
- [Learning log](docs/LEARNING_LOG.md) — engineering decisions and lessons
- [Backend guide](backend/README.md), [Frontend guide](frontend/README.md), [ML guide](ml/README.md)
