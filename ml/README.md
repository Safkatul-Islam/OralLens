# ML Pipeline

The ML module provides the reproducible data, training, evaluation, and
inference workflow for OralLens AI. The current block establishes the normalized
dataset contract and patient-level leakage controls before any model is trained.

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
  src/orallens_ml/data/
    acquisition.py
    archive.py
    manifest.py
    audit.py
    splits.py
  tests/
  .python-version
  pyproject.toml
```

Training, evaluation, inference, configuration, and model-artifact directories
will be added only when their implementation starts.

## Normalized Data Flow

1. Download and verify the exact dataset version and license.
2. Inspect the publisher's metadata and archive layout.
3. Adapt source metadata into the documented normalized CSV manifest.
4. Run the read-only dataset audit.
5. Create deterministic patient-grouped split assignments.
6. Review class and patient distributions before choosing training settings.

The dataset is not downloaded automatically. This avoids hidden network access,
license ambiguity, and accidental multi-gigabyte repository content.

See [`../docs/DATASET_CARD.md`](../docs/DATASET_CARD.md) for provenance,
limitations, required fields, and leakage controls.

The reviewed acquisition configuration and CLI verify publisher hashes and
inspect the outer ZIP before extracting its expected 7z member. Inner 7z
extraction remains a manual approval boundary until the real member listing has
been reviewed. See
[`../docs/DATA_ACQUISITION.md`](../docs/DATA_ACQUISITION.md).

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
45 passed, 1 skipped
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

Data manifest, audit, and patient-grouped split implementation added and
verified. The locked Python 3.13 ML environment and CUDA execution are also
verified. Dataset download and source-metadata adaptation remain pending.
