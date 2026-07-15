# ML Pipeline

The ML module provides the reproducible data, training, evaluation, and
inference workflow for OralLens AI. The current block has a verified Part 2
orthodontic plaque manifest, a safe PyTorch dataset loader, and a TorchVision
Faster R-CNN baseline training and evaluation entrypoint.

## Responsibilities

- Prepare datasets
- Split data into train, validation, and test sets
- Train a baseline model
- Evaluate model performance
- Save model artifacts
- Run inference on uploaded images
- Generate visual evidence where possible
- Document limitations in a model card

## Current Structure

```text
ml/
  configs/
    orthodontic_plaque_detection_baseline.toml
    orthodontic_plaque_detection_eval_smoke.toml
    orthodontic_plaque_detection_mvp.toml
    orthodontic_plaque_detection_mvp_eval.toml
    orthodontic_plaque_detection_mvp_predict.toml
    orthodontic_plaque_detection_predict_smoke.toml
    orthodontic_plaque_detection_smoke.toml
  src/orallens_ml/data/
    acquisition.py
    archive.py
    manifest.py
    audit.py
    splits.py
    orthodontic_plaque.py
    orthodontic_plaque_dataset.py
  src/orallens_ml/training/
    detection.py
  src/orallens_ml/modeling/
    detection.py
  src/orallens_ml/evaluation/
    detection.py
  src/orallens_ml/inference/
    detection.py
  src/orallens_ml/cli/
    dataset.py
    evaluate.py
    predict.py
    train.py
  tests/
  .python-version
  pyproject.toml
```

Training outputs are written under `ml/runs/`, which is ignored by git.

## Normalized Data Flow

1. Download and verify the exact dataset version and license.
2. Inspect the publisher's metadata and archive layout.
3. Adapt source metadata into the documented normalized CSV manifest.
4. Run the read-only dataset audit.
5. Create deterministic patient-grouped split assignments.
6. Validate retained images and load real manifest samples through PyTorch.
7. Train a bounded smoke run before any longer baseline run.
8. Train a bounded pretrained MVP run before any longer full baseline run.
9. Evaluate checkpoints with transparent IoU-threshold validation metrics.
10. Run checkpoint-backed one-image inference and write prediction artifacts.

The dataset is not downloaded automatically. This avoids hidden network access,
license ambiguity, and accidental multi-gigabyte repository content.
The same v1 policy applies to model weights: smoke runs use
`pretrained_weights = "none"`, and baseline runs using TorchVision default
weights require the official weight file to already exist in the local Torch
cache before model construction.

See [`../docs/DATASET_CARD.md`](../docs/DATASET_CARD.md) for provenance,
limitations, required fields, and leakage controls.

The reviewed acquisition configuration and CLI verify publisher hashes and
inspect the outer ZIP before extracting its expected 7z member. Inner 7z
extraction remains a manual approval boundary until the real member listing has
been reviewed. See
[`../docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md).

See [`../docs/TRAINING.md`](../docs/TRAINING.md) for the training architecture,
configs, smoke command, baseline command, artifact contract, and current limits.

## Environment Setup

The ML module uses Python 3.13 and uv. From the repository root, create the
project-local environment only after the lockfile has been generated and
reviewed:

```powershell
uv sync --project ml --group training --python C:\Python313\python.exe --no-python-downloads --cache-dir ml\.uv-cache --locked
```

Dependencies are separated by responsibility:

- Base: Pillow for upcoming image-content validation.
- Development: Pytest.
- Training: PyTorch, TorchVision, Captum, and scikit-learn.

Run the tests through the locked ML environment:

```powershell
uv run --project ml --group training python -m pytest
```

See [`../docs/ML_ENVIRONMENT.md`](../docs/ML_ENVIRONMENT.md) for version
rationale, official sources, CUDA index isolation, verification gates, and the
dependency update policy.

Latest verification in the locked ML environment on Windows with Python 3.13.5
and Pytest 9.1.1:

```text
127 passed, 1 skipped
```

The skipped case requires permission to create symbolic links on Windows. The
platform-independent path-escape test passed, so dataset-root containment is
still exercised on this environment.

The CUDA verification also passed with PyTorch 2.12.1+cu126 and TorchVision
0.27.1+cu126 on an NVIDIA GeForce RTX 4070 Laptop GPU.

## Learning Goals

- Understand supervised learning
- Understand image preprocessing
- Learn why train/validation/test splits matter
- Learn evaluation metrics beyond accuracy
- Learn model artifact management
- Learn how training code differs from inference code

## Status

Dataset acquisition gates, Part 2 manifest generation, image validation, the
PyTorch dataset loader, shared detection model/checkpoint utilities, baseline
detection training CLI, detection evaluation CLI, and detection inference module
are implemented and verified. A bounded pretrained MVP checkpoint has been
trained and evaluated from the verified Part 2 manifest. Pretrained-weight
handling is intentionally MVP-scoped: no implicit downloads during training,
with pretrained runs requiring a reviewed official TorchVision cache file.
Dataset Part 1 remains excluded because official 7-Zip integrity testing fails
inside the nested archive. Part 2 is the only current training scope.
