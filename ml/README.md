# OralLens AI ML

Project-local machine-learning package for orthodontic plaque-candidate object detection. It owns dataset validation, target conversion, model construction, training, evaluation, prediction, configs, and regression tests.

The detector is an experimental portfolio artifact, not a diagnostic model, clinically validated system, or medical device.

## Current model

- TorchVision Faster R-CNN with ResNet-50 FPN
- binary contract: background `0`, plaque candidate `1`
- official TorchVision pretrained initialization
- state-dict-only project checkpoints loaded with `weights_only=True`
- v2 inference threshold `0.65`, selected on complete validation
- maximum inference results: 25

Held-out test at score threshold `0.65` and IoU `0.5`:

| Precision | Recall | F1 | Mean matched IoU |
|---:|---:|---:|---:|
| 0.7582 | 0.7037 | 0.7299 | 0.7541 |

These are annotation-matching metrics on the project dataset and are not clinical-performance measures.

## Package layout

```text
ml/
  configs/                       versioned train/evaluate/predict settings
  src/orallens_ml/
    cli/                         command-line entry modules
    data/                        manifest and image dataset contracts
    modeling/                    Faster R-CNN construction/checkpoint behavior
    training/                    target conversion and training loop
    evaluation/                  thresholded matching and metrics
    inference/                   prediction config and JSON output
  tests/                         data, model, CLI, security, and regression tests
  data/                          ignored raw/prepared local data
  runs/                          ignored checkpoints, metrics, and predictions
```

## Data in use

Only the verified Part 2 orthodontic-plaque dataset is active.

Manifest:

`ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |

Part 1 remains excluded after its nested archive failed official integrity validation. See [Dataset card](../docs/DATASET_CARD.md) and [Data acquisition](../docs/DATA_ACQUISITION.md).

## Environment

The project targets Python 3.13 and uses `ml/.venv`. Create/synchronize it with project-local dependencies only:

```powershell
Push-Location ml
uv sync --group dev --group training
Pop-Location
```

Detailed versions and GPU notes are in [ML environment](../docs/ML_ENVIRONMENT.md).

## v2 workflow

Run commands from the repository root.

### Train

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp_v2.toml"
```

Outputs:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/checkpoint_last.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/metrics.json`

### Select the operating point on validation

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_eval.toml"
```

Complete-validation F1 peaked at tested threshold `0.65`. Do not select a new threshold from test results.

### Evaluate the frozen held-out test

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_test.toml"
```

The test config fixes split `test`, IoU `0.5`, score threshold `0.65`, and no batch cap.

### Predict one image

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_predict.toml" --image "C:\path\to\image.jpg"
```

The backend real-ML startup script uses this same prediction config.

## Annotation boundary policy

Manifest annotations use normalized center-width-height fields. Target conversion:

1. verifies finite normalized input
2. derives corner coordinates
3. allows only `1e-6` of numerical boundary tolerance
4. clips tolerated crossings with TorchVision
5. rejects boxes that are grossly invalid or non-positive after clipping

The full audit found two tiny boundary crossings in one rotated test image, with no degenerate result. Source annotations are retained unchanged and no sample is silently skipped.

## Tests

```powershell
ml\.venv\Scripts\python.exe -B -m pytest ml\tests
```

Latest result: `132 passed, 1 skipped`.

The skip requires Windows symbolic-link creation privileges. Tests that do not require that privilege continue to enforce path containment.

Coverage includes:

- manifest/schema/numeric/path validation
- image decoding and symlink boundaries
- normalized and derived box validation
- tolerated clipping, gross invalidity, and degeneration
- model/checkpoint contracts
- training, evaluation, inference, and CLI errors
- concise evaluation error translation without raw tracebacks
- config and generated-artifact behavior

## Artifact policy

Raw/prepared data, checkpoints, metrics, prediction output, logs, and caches are excluded by the root `.gitignore`. They remain local for reproducibility and evidence. Do not commit them or silently delete/overwrite them.

## Limitations and next work

The present evidence comes from one specialized dataset and a bounded experiment. High-priority research work is error analysis, score calibration assessment, acquisition-quality robustness, source/subgroup stratification where metadata supports it, and external validation design. None of those should be converted into clinical claims without appropriate representative evidence and review.

## Related documents

- [Training and evaluation](../docs/TRAINING.md)
- [Dataset card](../docs/DATASET_CARD.md)
- [ML environment](../docs/ML_ENVIRONMENT.md)
- [Pipeline](../docs/PIPELINE.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [Backend integration](../backend/README.md)
