# OralLens AI

OralLens AI is an end-to-end portfolio project for **experimental oral-image screening support**. A React interface uploads a JPEG or PNG image to a FastAPI API, which can run a TorchVision Faster R-CNN detector and return candidate plaque boxes, detector scores, evidence text, limitations, and suggested next steps.

> OralLens AI is a learning and engineering project. It is not a diagnostic system, a clinically validated model, a medical device, or a source of treatment recommendations.

## Current status

The local MVP works end to end:

1. The browser validates the selected file type and shows an image preview.
2. `POST /scans` validates upload metadata, size, extension, and file signature.
3. The backend invokes either a deterministic mock adapter or the project-local v3 ML detector.
4. The API stores a scan record and returns structured detections and report content.
5. The frontend overlays the returned boxes and presents the screening-support result.

The GPU-backed workflow has been manually verified from the browser through FastAPI and the real v3 checkpoint. The latest verification returned HTTP `201`, rendered 15 candidate boxes, exposed the explicit `orthodontic-plaque-mvp-v3` model identity, and displayed the expected score limitation and disclaimer. Generated scan and prediction artifacts remain local evidence and are excluded from version control.

The application plumbing is working, but the active v3 model is not a credible application-facing model. A source-label audit found that historical v1-v3 target conversion treated 1,170 plaque-absent annotated regions as plaque objects, and manual consumer-style challenge images exposed severe domain shift. V3 remains active only as historical end-to-end engineering evidence.

## Model status and evidence correction

The active application model is v3, a binary TorchVision Faster R-CNN with a ResNet-50 FPN backbone. Its runtime contract is:

- class `0`: background
- class `1`: plaque candidate
- training data: verified Part 2 orthodontic-plaque dataset
- split policy: patient-aware train, validation, and test groups
- v3 training: three complete train/validation epochs with durable last/best checkpoints
- v3 operating threshold: `0.85`, selected on the complete validation split

The source annotations use `0` for plaque absent in an annotated region and `1` for plaque present. Historical conversion assigned detector label `1` to both values. All published v1-v3 precision, recall, F1, IoU, calibration, and patient-level numbers are therefore contaminated and retained only as experiment-history artifacts. They are not valid current plaque-only performance evidence.

The conversion is now corrected and regression-tested: every source box is validated, only source label `1` becomes a plaque target, and images with no positive targets are supported. A fresh v4 run on the same source-aware AIRC image distribution was stopped after two completed epochs: training loss fell while validation loss worsened, and epoch 1 remained the best checkpoint. V4 has not been evaluated, integrated, or promoted and must not be resumed. The next experiment is blocked on a condition-specific task contract, admission of genuinely complementary target-domain data, and a CUDA-enforced training preflight. See [Data strategy](docs/DATA_STRATEGY.md), [Training and evaluation](docs/TRAINING.md), and [Dataset card](docs/DATASET_CARD.md).

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

The real-ML workflow requires the prepared dataset, the trained v3 checkpoint, and the project-local `ml\.venv`. Setup and reproducibility details are in [ML environment](docs/ML_ENVIRONMENT.md).

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

- ML suite: `174 passed, 1 skipped`
- backend suite: `24 passed, 1 skipped`
- real backend-to-ML integration smoke: `1 passed`
- frontend Playwright/axe browser suite: `5 passed`
- frontend production build: passed
- manual browser-to-backend-to-v3 scan: passed
- frontend dependency audit: `0` known vulnerabilities across `82` dependencies

The skipped ML test requires Windows symbolic-link privileges. The skipped backend test is an opt-in real-ML integration case that passes when explicitly enabled.

## Security and reliability controls

- explicit CORS origins for both local loopback hostnames
- browser-side JPEG/PNG MIME-and-extension validation before transmission; backend byte-level validation remains authoritative
- bounded upload size and chunked reads
- MIME type, extension, and magic-byte validation
- sanitized display filenames and SHA-256 content digests
- constrained temporary input paths with symlink and containment checks
- state-dict-only checkpoint loading with `weights_only=True`
- strict manifest path, annotation, numeric, file, and symlink validation
- expected errors translated at API and evaluation boundaries instead of exposing raw tracebacks
- validation-selected historical operating threshold; future locked results must not be used for retuning
- checkpoint, manifest, and config hashes recorded in local trustworthiness reports
- deployed-cap parity and score-to-match reliability measured on validation and test
- automated Chromium interaction and WCAG A/AA scans for critical frontend states

These controls make the portfolio MVP more disciplined; they do not make it a production medical system.

## Known limitations and next phase

The model was trained on one specialized orthodontic-plaque dataset under an incorrect historical target mapping that has since been corrected in source code. It has not been validated across consumer devices, framing, lighting, appliance status, demographics, or representative acquisition conditions. The UI displays the maximum detector score as a raw three-decimal ranking value, not a percentage or calibrated probability of disease. Local JSON storage is suitable for demonstration, not multi-user deployment or sensitive clinical records.

The historical measurable-trust reports remain useful for pipeline provenance and failure-analysis mechanics, but their model-quality numbers inherit the target defect. Manual testing on clear internet photographs also produced very poor results, especially when lips or surrounding facial skin were visible. This is concrete evidence that the current standardized orthodontic data does not cover the application-facing image domain.

Automated accessibility checks cover semantics, keyboard focus, live status, target size, reduced motion, and axe rules for critical UI states. They do not replace manual screen-reader, zoom/reflow, high-contrast, touch, or small-screen evaluation.

The next phase is a realistic portfolio-quality correction, not a medical-device program or another long run on the same images:

1. freeze separate task contracts for visible plaque and supragingival calculus; do not merge them into one label
2. audit the ODS/Oralformer release as the first calculus data-feasibility candidate, without downloading or training until provenance, license, patient identity, acquisition context, mask semantics, and redistribution terms pass
3. continue access review for the strongest non-orthodontic plaque sources
4. add a separate oral ROI, image-quality, OOD, and abstention boundary for consumer-style inputs
5. require CUDA explicitly, record device/utilization/throughput evidence, and run a monitored smoke test before any future pilot
6. run only a small predeclared pilot after the data and evaluation contracts are fixed; stop on validation reversal or failure to beat the baseline
7. select policy on validation only and evaluate once on a locked, rights-safe challenge boundary before considering promotion

This can make the screening-support demonstration more credible; it cannot support diagnosis, rule-out, or clinical-validation claims.

## Documentation map

- [Architecture](docs/ARCHITECTURE.md) - components, data flow, runtime modes, and trust boundaries
- [Pipeline](docs/PIPELINE.md) - end-to-end runtime, ML lifecycle, artifacts, and source ownership
- [Project brief](docs/PROJECT_BRIEF.md) - product scope, achieved MVP, exclusions, and next phase
- [Intended use and claims](docs/INTENDED_USE_AND_CLAIMS.md) - current claim boundary and staged research target
- [Clinical evidence plan](docs/CLINICAL_EVIDENCE_PLAN.md) - external, prospective, and human-AI evidence ladder
- [Data acquisition and annotation](docs/DATA_ACQUISITION_AND_ANNOTATION.md) - future multi-site clinical data contract
- [Draft external validation protocol](docs/EXTERNAL_VALIDATION_PROTOCOL.md) - Stage 1 study design, model lock, endpoints, and approval gates
- [Clinical data dictionary](docs/CLINICAL_DATA_DICTIONARY.md) - pseudonymous entities, vocabularies, integrity rules, and release contract
- [Clinical evidence readiness checklist](docs/CLINICAL_EVIDENCE_READINESS_CHECKLIST.md) - accountable prerequisites and current readiness status
- [API](docs/API.md) - endpoint contracts and error behavior
- [Training and evaluation](docs/TRAINING.md) - experiments, threshold selection, and internal test results
- [Dataset card](docs/DATASET_CARD.md) - provenance, splits, validation, and limitations
- [Data acquisition](docs/DATA_ACQUISITION.md) - controlled preparation workflow for the current source dataset
- [Data strategy](docs/DATA_STRATEGY.md) - condition-specific data roles, source decisions, admission gates, and future training criteria
- [ML environment](docs/ML_ENVIRONMENT.md) - reproducible local environment and commands
- [Learning log](docs/LEARNING_LOG.md) - engineering decisions and lessons
- [Backend guide](backend/README.md), [Frontend guide](frontend/README.md), [ML guide](ml/README.md)
