# OralLens AI ML Package

Project-local object-detection package for the OralLens AI portfolio case study.

## Final model contract

- architecture: TorchVision Faster R-CNN with ResNet-50 FPN
- class `0`: implicit background
- class `1`: AIRC plaque-positive peri-tooth region
- model resize: 512–768
- final checkpoint: Experiment 1 epoch 9
- inference threshold: `0.80`
- inference cap: 100
- checkpoint loading: state dictionary with `weights_only=True`

This detector does not localize a clinically verified disease state or necessarily trace discrete visible-plaque deposits.

## Package layout

```text
ml/
  configs/                       versioned training/evaluation/inference settings
  src/orallens_ml/
    cli/                         command-line entry points
    data/                        source, manifest, split, and dataset contracts
    modeling/                    Faster R-CNN and checkpoint handling
    training/                    target conversion, augmentation, and training loop
    evaluation/                  matching, AP, metrics, and error analysis
    inference/                   input assessment and prediction
  tests/                         ML contract and regression tests
  data/                          ignored local raw/prepared data
  runs/                          ignored local checkpoints and evidence
```

## Data

Only verified AIRC/LabDen Part 2 is used. The source-aware v4 manifest is:

`ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`

Final experiment population:

| Split | Genuine originals | Patients | Foreground regions |
|---|---:|---:|---:|
| Train | 480 | 55 | 5,502 |
| Validation | 58 | 7 | 710 |
| Test | 107 | 12 | 1,293 |

The filter requires `variant="original"` and excludes `_blur`, `_dark`, and `_light` sample suffixes. Patient and derivative-family leakage checks return zero cross-partition groups.

The loader validates schema, encoding, paths, containment, symlinks, files, annotations, labels, and numeric ranges. Target conversion validates both source classes but emits only source label `1` as Faster R-CNN foreground.

## Experiment 1

Baseline problem: 3,834 stored training rows treated publisher-generated variants as independent samples, and validation loss reversed after the first completed epoch.

Intervention:

- train on 480 genuine originals
- apply conservative horizontal flip and color jitter online
- retain the architecture and SGD settings
- use a comparable optimizer budget: 7,680 versus 7,668 steps
- select checkpoint and threshold on 58 genuine validation originals

Selection:

- checkpoint: epoch 9, validation loss `0.657865`
- threshold: `0.80`, validation F1 `0.809192`

Final test:

| Metric | Result |
|---|---:|
| AP@0.50 | 0.833486 |
| mAP@0.50:0.95 | 0.516427 |
| Precision | 0.783033 |
| Recall | 0.806651 |
| F1 | 0.794667 |
| FP/image | 2.700935 |
| Mean matched IoU | 0.823955 |

Compared with the corrected baseline, mAP improved 23.01%, F1 improved 5.71%, false positives/image decreased 26.65%, and localization failures decreased 38.95%.

Full methodology and exact counts: [Training and evaluation](../docs/TRAINING.md).

## Frozen configuration

| Purpose | Config |
|---|---|
| corrected baseline | `configs/orthodontic_plaque_detection_mvp_v4.toml` |
| baseline test methodology | `configs/orthodontic_plaque_detection_mvp_v4_baseline_test.toml` |
| Experiment 1 training | `configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug.toml` |
| Experiment 1 validation | `configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_validation.toml` |
| final inference | `configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml` |

Final checkpoint:

`runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug/checkpoint_best.pt`

SHA-256:

`79CBB3B99D56F77E852B5096446649EFF17858A25B5FD40DAB9408FA7A2F19D7`

Data, checkpoints, metrics, and predictions are ignored local artifacts. They must not be committed or overwritten.

## Environment

The project targets Python 3.13 and uses `ml/.venv`.

```powershell
Push-Location ml
uv sync --group dev --group training
Pop-Location
```

Use project-local dependencies only. See [ML environment](../docs/ML_ENVIRONMENT.md).

## Inference

Run from the repository root:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml" --image "C:\path\to\image.jpg"
```

Inference safely decodes and technically assesses the image, constructs the same model contract used during evaluation, loads epoch 9, filters scores at `0.80`, and writes local JSON output.

The final model, threshold, and test result are frozen. The training and evaluation commands are retained in versioned configs for provenance, not as authorization for another run.

## Tests

```powershell
ml\.venv\Scripts\python.exe -B -m pytest ml\tests
```

Most recent complete suite after Experiment 1 implementation: `215 passed, 2 skipped`. Windows symlink-privilege cases may skip while platform-independent containment tests continue to run.

Coverage includes:

- source identity and leakage prevention
- dataset paths, annotations, labels, and image decoding
- valid, clipped, invalid, and degenerate boxes
- original-only augmentation and transformed-box validity
- model/checkpoint compatibility and safe loading
- training resume and checkpoint selection
- AP, threshold metrics, matching, and error analysis
- inference filtering, technical abstention, and concise CLI errors

## Limitations

- one source family, 74 patients, and 480 genuine training originals
- no representative image-level negative/hard-negative population
- no external or consumer-photo validation
- no device/site/demographic subgroup evaluation
- annotations encode plaque-positive peri-tooth regions, not exact visible deposits
- technical assessment does not provide oral ROI or semantic OOD detection
- scores are not calibrated clinical probabilities

Related: [Pipeline](../docs/PIPELINE.md), [Architecture](../docs/ARCHITECTURE.md), and [Dataset card](../docs/DATASET_CARD.md).
