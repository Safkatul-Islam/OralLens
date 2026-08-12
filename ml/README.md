# OralLens AI ML

Project-local machine-learning package for orthodontic plaque-candidate object detection. It owns dataset validation, target conversion, model construction, training, evaluation, prediction, configs, and regression tests.

The detector is an experimental portfolio artifact, not a diagnostic model, clinically validated system, or medical device.

## Current model

- TorchVision Faster R-CNN with ResNet-50 FPN
- source labels: plaque absent `0`, plaque present `1`
- detector contract: implicit background `0`, plaque candidate `1`
- official TorchVision pretrained initialization
- state-dict-based project checkpoints loaded with `weights_only=True`
- v3 atomic last/best training-state checkpoints with optimizer and RNG resume state
- v3 active threshold `0.85`, selected on complete validation
- maximum inference results: 25

The backend selects the v3 prediction config through its typed configuration and startup-script boundary. V3 is now a historical application model: its training converter incorrectly mapped all source boxes to plaque candidate `1`, including 1,170 source regions labeled plaque absent. Its reported metrics are contaminated and must not be used as plaque-only performance evidence.

The corrected converter validates every source box and label, retains only source label `1` as detector objects, and supports empty positive targets. A fresh corrected v4 diagnostic completed two epochs on the same underlying AIRC distribution and was stopped after validation-loss reversal. It is not a promoted or evaluated model.

The stopped v4 training config is
`ml/configs/orthodontic_plaque_detection_mvp_v4.toml`; it preserves the v3
architecture, optimizer, seed, and full-split budget while using the corrected
source-aware manifest and an isolated output directory. The existing v4
evaluation and resume configs are not authorized for execution. V4 has no
threshold, test result, prediction config, backend integration, or promotion.

| Completed epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | `0.7966761603586116` | `0.8249539341299962` |
| 2 | `0.6149270393257503` | `1.028658177671779` |

`checkpoint_best.pt` is epoch 1 and `checkpoint_last.pt` is epoch 2. Both were
inspected read-only with `weights_only=True` and record `device_type="cuda"`
plus CUDA RNG state. The experiment did not record GPU utilization, throughput,
data-loading time, or peak memory, so future training requires explicit CUDA and
runtime telemetry. Do not resume epoch 3 or interpret these losses as localization
performance.

A one-train-batch/one-validation-batch CUDA smoke completed through the v4
manifest and target-conversion path. It used random initialization to avoid
external cache access, so its loss values and checkpoints are pipeline evidence
only and must not be interpreted, resumed, evaluated, or promoted as model
quality evidence.

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

| Split | Source label `0` | Source label `1` |
|---|---:|---:|
| Train | 996 | 47,102 |
| Validation | 30 | 6,040 |
| Test | 144 | 10,926 |
| **Total** | **1,170** | **64,068** |

### Source-aware manifest contract

New prepared detection manifests use detection-manifest schema `1`. In addition
to the image and annotation fields, every row records:

- dataset ID and version
- source-family and source-artifact IDs
- the split-isolation group (`split_group_id`), which is the patient ID for the
  current AIRC/LabDEN source
- the original/derived image family (`derivative_group_id`)
- the exact publisher variant, such as `original`, `brightness-down`, or
  `rotate-right-15`

Preparation fails closed if a source-family split group appears in more than one
split, if a derivative family crosses split groups or splits, if a derivative
variant is duplicated, or if an identity is malformed. This prevents original
images and their publisher-generated brightness, flip, and rotation variants
from being divided across train, validation, and test.

The existing v3 prepared manifest is retained unchanged as historical evidence,
and the current loader remains compatible with it. The source-aware v4 manifest
was generated separately at:

`ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`

It contains 5,160 samples across 74 patient split groups and 860 derivative
families. Every derivative family contains the original plus five declared
publisher variants, with no sample, patient, or derivative-family leakage.
Preparation recorded 14 excluded empty-label samples and 42 filtered
non-positive annotations in the adjacent `exclusions.csv`; it did not alter the
source data. All 5,160 referenced images subsequently decoded successfully.

## Environment

The project targets Python 3.13 and uses `ml/.venv`. Create/synchronize it with project-local dependencies only:

```powershell
Push-Location ml
uv sync --group dev --group training
Pop-Location
```

Detailed versions and GPU notes are in [ML environment](../docs/ML_ENVIRONMENT.md).

## Historical v3 workflow

Run commands from the repository root.

These commands reproduce or invoke the current historical v3 artifacts. Do not resume v3 or use its evaluation output as corrected plaque-only evidence.

### Train

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp_v3.toml"
```

Outputs:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/checkpoint_last.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/checkpoint_best.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/metrics.json`

V3 processes the complete train and validation splits in every epoch. A completed epoch is persisted atomically. To resume an interrupted run, use an explicitly configured `resume_checkpoint_path` inside the same output directory; incompatible training settings fail closed. Fresh training refuses to overwrite existing final artifacts.

### Select the operating point on validation

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_eval.toml"
```

Under the incorrect historical target mapping, complete-validation F1 peaked at tested threshold `0.85`. Do not select a new threshold from test results.

### Evaluate the fixed-threshold internal test benchmark

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_test.toml"
```

The test config fixes split `test`, IoU `0.5`, score threshold `0.85`, and no batch cap. Its output inherits the historical target defect. The cohort has also been inspected across model generations, so v4 requires a new locked challenge boundary.

### Predict one image

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_predict.toml" --image "C:\path\to\image.jpg"
```

This is the active v3 prediction config. It declares the checkpoint, threshold, result cap, and explicit `orthodontic-plaque-mvp-v3` model identity used by the backend.

## Trustworthiness reporting

The v3 trustworthiness reports below are historical pipeline artifacts. Their provenance, deployment-cap, and failure-report structures remain useful engineering evidence, but their matching and calibration values inherit the incorrect target mapping.

The trustworthiness evaluator measures the exact local deployment policy without
changing the model or operating point. It compares score-ordered, one-to-one box
matching before and after the deployed 25-detection cap, retains per-image and
per-patient evidence, and records SHA-256 identities for the checkpoint,
manifest, and configs.

Run the complete validation report:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection-trustworthiness --config "ml\configs\orthodontic_plaque_detection_mvp_v3_trust_validation.toml"
```

Run the fixed-threshold internal test report without retuning:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection-trustworthiness --config "ml\configs\orthodontic_plaque_detection_mvp_v3_trust_test.toml"
```

Both runs used score threshold `0.85`, IoU `0.5`, and maximum 25 detections per
image. No validation image reached the cap. One test image exceeded it by one
false positive; no true positive was truncated.

| Evidence | Validation | Frozen test |
|---|---:|---:|
| Images | 468 | 858 |
| Patients | 7 | 12 |
| Uncapped predictions | 6,309 | 11,434 |
| Cap-affected images | 0 | 1 |
| Score-to-match ECE | 0.1940 | 0.1878 |
| Score-to-match MCE | 0.4290 | 0.4046 |
| Score-to-match Brier score | 0.2075 | 0.2035 |
| Patient-level F1 range | 0.6769-0.9245 | 0.6824-0.8889 |

Score-to-match reliability compares a displayed detector score with whether that
box matched one dataset annotation under the declared policy. It is conditional
on the `0.85` threshold and 25-result cap. It is not a calibrated probability of
plaque, disease, clinical risk, or patient outcome.

Historically, v3 improved the old-contract detection metrics over v2 but worsened
score-to-match ECE, MCE, and Brier score. Dark, blur, and rotation variants were
recurring failure cases. Manual consumer-style challenges also exposed severe
framing and domain-shift failures. These motivate a separate oral-ROI,
input-quality, OOD, and abstention boundary.

Generated reports remain local ignored evidence:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3_trust_validation/trustworthiness_report.json`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3_trust_test/trustworthiness_report.json`

## Annotation boundary policy

Manifest annotations use normalized center-width-height fields. Target conversion:

1. verifies integer source labels in `{0, 1}`
2. verifies finite normalized input for every source box
3. derives corner coordinates
4. allows only `1e-6` of numerical boundary tolerance
5. clips tolerated crossings with TorchVision
6. rejects boxes that are grossly invalid or non-positive after clipping
7. retains only source label `1` boxes as plaque detector targets

The full audit found two tiny boundary crossings in one rotated test image, with no degenerate result. Source annotations are retained unchanged and no sample is silently skipped.

## Tests

```powershell
ml\.venv\Scripts\python.exe -B -m pytest ml\tests
```

Latest result: `174 passed, 1 skipped`.

The skip requires Windows symbolic-link creation privileges. Tests that do not require that privilege continue to enforce path containment.

Coverage includes:

- manifest/schema/numeric/path validation
- source identity, derivative-family tracking, and split-leakage prevention
- image decoding and symlink boundaries
- normalized and derived box validation
- source-label validation, plaque-only filtering, and empty positive targets
- tolerated clipping, gross invalidity, and degeneration
- model/checkpoint contracts
- training, evaluation, inference, and CLI errors
- deployment-cap parity, per-image matching evidence, provenance, and score reliability
- concise evaluation error translation without raw tracebacks
- config and generated-artifact behavior

## Artifact policy

Raw/prepared data, checkpoints, metrics, prediction output, logs, and caches are excluded by the root `.gitignore`. They remain local for reproducibility and evidence. Do not commit them or silently delete/overwrite them.

## Limitations and next work

The present data comes from one standardized orthodontic source with 74 patient groups and at least one plaque-present region on every manifest row. V3 also learned the incorrect historical target mapping. Additional epochs on that checkpoint cannot correct the labels or consumer-image domain shift.

The practical next work is to keep visible plaque and supragingival calculus as separate task contracts, complete the ODS/Oralformer admission audit as the first calculus feasibility check, continue ordinary-RGB plaque source review, and build an independent oral-ROI and image-quality/OOD abstention stage. Only then may a new experiment run under explicit CUDA with telemetry, a monitored smoke test, and a small predeclared pilot. V4 remains stopped and must not be resumed. See [Data strategy](../docs/DATA_STRATEGY.md) and [Intended use and claims](../docs/INTENDED_USE_AND_CLAIMS.md).

## Related documents

- [Training and evaluation](../docs/TRAINING.md)
- [Dataset card](../docs/DATASET_CARD.md)
- [Data strategy](../docs/DATA_STRATEGY.md)
- [Intended use and claims](../docs/INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](../docs/CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](../docs/DATA_ACQUISITION_AND_ANNOTATION.md)
- [ML environment](../docs/ML_ENVIRONMENT.md)
- [Pipeline](../docs/PIPELINE.md)
- [Architecture](../docs/ARCHITECTURE.md)
- [Backend integration](../backend/README.md)
