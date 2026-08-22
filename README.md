# OralLens AI

OralLens AI is an end-to-end AI/ML engineering project for experimental oral-image screening support. A React interface uploads an oral image to FastAPI, which validates the request, runs a project-local TorchVision Faster R-CNN detector, and returns localized plaque-positive peri-tooth region candidates with detector scores, evidence, limitations, and next steps.

> OralLens AI is not a diagnostic system, clinically validated model, medical device, or treatment-recommendation system. Detector scores are ranking values, not disease probabilities.

## Current status

The current portfolio scope is complete:

- patient- and original-family-safe dataset partitions
- corrected foreground-label semantics
- controlled baseline and improvement experiment
- validation-only checkpoint and threshold selection
- one frozen originals-only test evaluation
- frozen epoch-9 checkpoint integrated into FastAPI
- automated frontend, backend, ML, and real-model integration coverage

The deployed model is `orthodontic-plaque-mvp-v4-originals-online-aug-epoch9`. It uses score threshold `0.80` and returns class `1` as a plaque-positive peri-tooth region candidate.

## Case-study result

Experiment 1 replaced independently sampled pre-generated derivatives with original-family-balanced online augmentation while keeping Faster R-CNN and the optimizer exposure approximately constant.

| Patient-held-out originals-only test metric | Corrected v4 baseline | Final model | Relative change |
|---|---:|---:|---:|
| AP@0.50 | 0.8034 | **0.8335** | +3.7% |
| mAP@0.50:0.95 | 0.4198 | **0.5164** | +23.0% |
| Precision | 0.7206 | **0.7830** | +8.7% |
| Recall | 0.7858 | **0.8067** | +2.7% |
| F1 | 0.7518 | **0.7947** | +5.7% |
| False positives/image | 3.6822 | **2.7009** | -26.6% |
| Localization failures | 95 | **58** | -38.9% |
| Mean matched IoU | 0.7808 | **0.8240** | +5.5% |

The fixed test population contains 107 genuine original images, 12 held-out patients, and 1,293 foreground regions. The threshold and epoch were frozen using validation only; the test result was not used for retuning.

See [Training and evaluation](docs/TRAINING.md) for the experiment contract, exact counts, metrics, and limitations.

## Architecture

```text
React/Vite UI
    |
    | POST /scans (multipart image)
    v
FastAPI route -> ScanService -> upload validation + SHA-256
                                      |
                                      v
                           InferencePipeline boundary
                             |                  |
                             | mock             | ml
                             v                  v
                    deterministic result   input assessment
                                                |
                                      supported | unsupported
                                                v
                                  Faster R-CNN / abstention
                                                |
                                                v
                     typed response + local JSON scan record
                                                |
                                                v
                              report + bounding-box overlay
```

Responsibilities remain separated across routes, schemas, services, storage, inference adapters, and the project-local ML package. See [Pipeline](docs/PIPELINE.md) and [Architecture](docs/ARCHITECTURE.md).

## Dataset boundary

Only the verified Part 2 AIRC/LabDen orthodontic-plaque source is used.

| Split | Genuine originals | Patients | Foreground regions |
|---|---:|---:|---:|
| Train | 480 | 55 | 5,502 |
| Validation | 58 | 7 | 710 |
| Test | 107 | 12 | 1,293 |

All images belonging to one patient or derivative/original family remain in one partition. Stored brightness, flip, and rotation variants remain available as source provenance but are not independent samples in the final experiment.

## Repository layout

```text
backend/   FastAPI routes, services, storage, ML adapter, scripts, and tests
frontend/  React/TypeScript/Vite upload and visualization interface
ml/        dataset contracts, Faster R-CNN lifecycle, configs, and tests
docs/      architecture, pipeline, API, dataset, training, and environment guides
```

Datasets, checkpoints, run output, scan records, virtual environments, and frontend build artifacts are ignored by git.

## Run locally

Requirements:

- project-local Python environment at `ml\.venv`
- local prepared dataset and frozen checkpoint at the paths declared by the inference config
- project-local frontend dependencies

Start the model-backed API from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File backend\scripts\run-ml-server.ps1
```

Start the frontend in a second terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File frontend\scripts\run-dev.ps1
```

Open `http://127.0.0.1:5173`. The backend listens on `http://127.0.0.1:8000`.

The backend defaults to deterministic mock mode when it is not launched through the real-ML script.

## Verification

Latest relevant verification:

- complete ML suite after Experiment 1 implementation: `215 passed, 2 skipped`
- complete backend suite after final promotion: `32 passed, 1 skipped`
- focused final deployment configuration and adapter contracts: `13 passed`
- real FastAPI-to-frozen-model integration smoke: `1 passed`
- automated Chromium interaction/accessibility suite: `5 passed`
- frontend production build: passed

The final real-model smoke returned HTTP `201`, model identity `orthodontic-plaque-mvp-v4-originals-online-aug-epoch9`, and 14 class-1 detections with scores above the frozen `0.80` threshold.

## Security and reliability controls

- explicit local CORS allowlist
- bounded upload size and chunked reads
- MIME, extension, and magic-byte validation
- sanitized display filenames and SHA-256 digests
- constrained temporary paths with symlink and containment checks
- safe image decoding and technical-quality abstention
- checkpoint loading with `weights_only=True`
- manifest schema, path, numeric, box, label, and split validation
- generic user-facing errors without stack traces or internal paths
- ignored datasets, checkpoints, predictions, scans, and secrets

## Honest limitations

- The target is an AIRC plaque-positive peri-tooth region, not a discrete visible-plaque deposit.
- Training uses 480 genuine originals from one orthodontic source family.
- The held-out test contains 12 patients and is not an external or consumer-photo cohort.
- Representative clean-mouth and consumer hard-negative images are absent.
- Consumer selfie, device, lighting, demographic, and acquisition-site generalization is unproven.
- The technical input gate can reject obvious dimension/luminance failures but does not establish oral ROI, blur, glare, or semantic in-distribution status.
- Local JSON storage is demonstration infrastructure, not a multi-user clinical record system.

The measured result supports an object-detection engineering case study. It does not support diagnosis, clinical rule-out, prevalence estimates, specificity/NPV claims, or medical-device status.

## Documentation map

- [Pipeline](docs/PIPELINE.md) - runtime and offline flow with source ownership
- [Architecture](docs/ARCHITECTURE.md) - component and trust boundaries
- [API](docs/API.md) - endpoint and error contracts
- [Training and evaluation](docs/TRAINING.md) - controlled experiment and exact evidence
- [Dataset card](docs/DATASET_CARD.md) - source provenance and dataset constraints
- [ML environment](docs/ML_ENVIRONMENT.md) - project-local setup
- [Backend guide](backend/README.md), [Frontend guide](frontend/README.md), [ML guide](ml/README.md)
