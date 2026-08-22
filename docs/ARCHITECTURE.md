# OralLens AI Architecture

## Purpose and scope

This document is the architectural source of truth for the current portfolio application. OralLens AI localizes AIRC plaque-positive peri-tooth region candidates for experimental screening support. Architecture quality does not imply clinical validity.

## System context

```text
User
  |
  v
React + TypeScript + Vite
  |  POST /scans
  v
FastAPI route
  |
  v
ScanService ------------------------------+
  |                                       |
  +-> upload validation + SHA-256         +-> report construction
  |                                       +-> JSONScanStore
  v
InferencePipeline
  +-> MockInferencePipeline
  +-> MLDetectionInferencePipeline
          |
          +-> technical input assessment -> structured abstention
          |
          +-> Faster R-CNN + epoch-9 checkpoint + threshold 0.80
```

## Component boundaries

| Component | Owns | Does not own |
|---|---|---|
| Frontend | selection preflight, preview, request state, results, box overlay | authoritative validation, model execution, clinical interpretation |
| API routes | HTTP contract and dependency access | business logic and ML internals |
| `ScanService` | validation, hashing, orchestration, report construction | framework startup and model construction |
| Inference adapters | stable mock/ML boundary | HTTP persistence and UI behavior |
| `JSONScanStore` | atomic local JSON persistence | authentication, encryption, multi-process or regulated storage |
| ML package | data contracts, model construction, training, evaluation, input assessment, inference | HTTP and browser concerns |
| TOML configs | reproducible model/runtime selection | source data, checkpoints, or secrets |

## Runtime configuration

The backend uses typed `ORALLENS_` settings.

| Mode | Selection | Purpose |
|---|---|---|
| `mock` | default | deterministic API development without Torch inference |
| `ml` | real-ML startup scripts | local frozen-model inference |

Application model selection occurs at one configuration boundary:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml`

It fixes:

- model identity `orthodontic-plaque-mvp-v4-originals-online-aug-epoch9`
- epoch-9 `checkpoint_best.pt`
- Faster R-CNN ResNet-50 FPN
- two classes and `512–768` resizing
- threshold `0.80`
- maximum 100 detections
- technical input-assessment policy

The default backend settings and both startup scripts point to this config. Promotion did not require route or response-schema changes.

## ML and data contract

The v4 prepared manifest records dataset/source identity, patient split group, derivative family, publisher variant, relative image path, and structured annotations.

Data invariants:

- patient groups remain in one split
- original/derivative families remain in one split
- paths remain relative and contained by the dataset root
- symlinks and missing files fail closed
- coordinates and labels are finite and structurally valid
- source labels are restricted to `0` or `1`
- only source label `1` becomes a detector foreground object
- tolerated boundary crossings are clipped and degenerate boxes are rejected

The final model uses 480/58/107 genuine originals for train/validation/test. Stored publisher transformations are provenance, not independent final-experiment samples.

## Model lifecycle boundary

Training, validation, test evaluation, and production inference use the same shared model constructor and image-size contract. Checkpoint compatibility validates class count and image sizes before loading.

Selection discipline:

1. training exposure was fixed before the run
2. epoch 9 was selected using originals-only validation loss
3. threshold `0.80` was selected using validation F1
4. the 107-image test set was evaluated once
5. the test result did not change the checkpoint or threshold
6. the frozen model was promoted through configuration

## Trust boundaries

### External requests

- size is bounded during reading
- MIME type, extension, and magic bytes are checked
- filenames are reduced to a basename
- invalid uploads return concise 4xx errors
- technical assessment can abstain before model construction
- response schemas constrain labels, scores, coordinates, and counts

### Filesystem and artifacts

- temporary ML inputs use UUID names under a contained directory
- symlinked or escaping paths are rejected
- temporary inputs are deleted in `finally`
- datasets, checkpoints, run output, scans, and secrets are ignored by git
- the local JSON store uses temporary-file replacement

### Model execution

- checkpoints contain state dictionaries and structured metadata
- loading uses `weights_only=True`
- runtime and checkpoint architecture contracts must match
- user-facing errors omit stack traces and local paths
- returned boxes must be finite, nondegenerate, class-valid, and score-valid

## Deployment characteristics

This is a local portfolio architecture, not a production clinical deployment:

- no authentication or authorization
- no encrypted clinical record store
- no multi-process persistence guarantees
- no model registry or remote artifact delivery
- no monitoring, drift detection, or external validation
- no representative consumer-image input boundary

The current technical assessment catches simple decoding, dimension, luminance, and contrast failures. It does not prove mouth presence, adequate oral framing, absence of blur/glare, or semantic in-distribution status.

## Verification boundaries

- ML tests cover data, split, box, label, augmentation, checkpoint, evaluation, and inference contracts.
- Backend tests cover upload validation, schemas, storage, adapter conversion, errors, and deployment configuration.
- A real `POST /scans` smoke loaded the frozen checkpoint and returned class-1 detections at the `0.80` threshold.
- Frontend tests cover upload interaction, result rendering, errors, accessibility rules, and overlay geometry with mocked API responses.

See [Pipeline](PIPELINE.md), [API](API.md), and [Training and evaluation](TRAINING.md).
