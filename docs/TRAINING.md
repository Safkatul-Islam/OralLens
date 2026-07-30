# Training and Evaluation

## Claim boundary

This document records reproducible object-detection experiments for a portfolio and learning project. Reported precision, recall, F1, and IoU describe annotation matching on one orthodontic-plaque dataset. They are not sensitivity, specificity, diagnostic accuracy, patient risk, or clinical validation.

## Model contract

- framework: PyTorch and TorchVision
- architecture: Faster R-CNN with ResNet-50 FPN
- classes: `0` background, `1` plaque candidate
- initialization: official TorchVision default pretrained weights
- image size range: 512 to 768 pixels
- trainable backbone layers: 3
- checkpoint format: model state dictionary loaded with `weights_only=True`

## Data contract

Only the verified Part 2 orthodontic-plaque material is used. The prepared manifest has patient-aware splits:

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |

The manifest loader validates schema, split values, safe paths, extensions, JSON annotations, finite numeric fields, normalized values, positive annotation counts, root containment, file existence, and symlink policy.

### Boundary policy for normalized boxes

Source annotations use normalized `x_center`, `y_center`, `width`, and `height`. Target conversion derives corner coordinates.

- fully in-range boxes remain unchanged
- a derived corner may cross an image boundary by at most `1e-6`, covering tiny numerical/rotation edge effects
- tolerated crossings are clipped with TorchVision's box operation
- positive width and height are revalidated after clipping
- grossly out-of-range or degenerate boxes fail closed

The complete manifest audit found two affected annotations in one test image. The maximum crossing was approximately `5e-7`; neither box became degenerate after clipping. No manifest row was deleted, skipped, or silently altered.

## Reproducibility controls

- TOML configs define data, model, optimizer, device, seed, batch caps, checkpoint, thresholds, and outputs
- v2 seed: `20260711`
- patient-aware split assignments remain fixed
- validation selects the score threshold
- held-out test uses one frozen threshold and is not used for retuning
- generated checkpoints, metrics, predictions, and data are local ignored artifacts
- evaluator errors translate shared target-validation failures into concise evaluation-domain failures

Run commands below from the repository root with the project-local ML environment.

## v1 baseline

Config: `ml/configs/orthodontic_plaque_detection_mvp.toml`

The intentionally small baseline used one epoch, batch size 1, 64 training batches, and 16 validation batches.

| Metric | Result |
|---|---:|
| Training loss | 1.2272 |
| Validation loss | 0.9620 |

A 64-image validation calibration selected threshold `0.15` from the tested values:

| Precision | Recall | F1 | TP | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0.0489 | 0.1361 | 0.0720 | 86 | 1,671 | 546 |

This weak baseline established that the pipeline ran but was not a useful final operating point.

## v2 training experiment

Config: `ml/configs/orthodontic_plaque_detection_mvp_v2.toml`

Key settings:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 1 |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Maximum training batches per epoch | 512 |
| Maximum validation batches per epoch | 64 |
| Workers | 0 |
| Device | automatic |

Command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp_v2.toml"
```

Loss history:

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 1.0039 | 0.6238 |
| 2 | 0.8170 | 0.6192 |
| 3 | 0.7662 | 0.6145 |

Both losses decreased across the bounded run; no validation-loss reversal was observed. This is useful experiment evidence, not proof of generalization.

Local outputs:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/checkpoint_last.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/metrics.json`

## Full-validation threshold selection

Config: `ml/configs/orthodontic_plaque_detection_mvp_v2_eval.toml`

Command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_eval.toml"
```

The complete validation split covered 468 images and 6,070 targets at IoU `0.5`. The broad tested F1 maximum was `0.65`; no finer search was performed to avoid unnecessary validation overfitting.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.60 | 0.7375 | 0.7783 | 0.7574 |
| **0.65** | **0.7652** | **0.7516** | **0.7583** |
| 0.70 | 0.7924 | 0.7180 | 0.7533 |

At `0.65`: TP 4,562; FP 1,400; FN 1,508; prediction count 5,962; mean matched IoU `0.7487`.

The selected threshold was copied to the prediction and held-out test configs before test evaluation.

## Frozen held-out evaluation

Config: `ml/configs/orthodontic_plaque_detection_mvp_v2_test.toml`

Command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_test.toml"
```

The final run covered all 858 test images and 11,070 targets at the frozen score threshold `0.65` and IoU `0.5`.

| Precision | Recall | F1 | Mean matched IoU |
|---:|---:|---:|---:|
| 0.7582 | 0.7037 | 0.7299 | 0.7541 |

Counts: TP 7,790; FP 2,484; FN 3,280; predictions 10,274.

Artifact: `ml/runs/detection/orthodontic_plaque_part2_mvp_v2_test/evaluation_metrics.json`

The test result is a final measurement at the validation-selected operating point. It must not drive a threshold change.

## Inference

Config: `ml/configs/orthodontic_plaque_detection_mvp_v2_predict.toml`

It fixes score threshold `0.65` and caps returned detections at 25.

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v2_predict.toml" --image "C:\path\to\image.jpg"
```

The backend ML launcher selects this same config. Prediction JSON is retained locally under the configured ignored run directory.

## Verification

The latest complete ML suite: `132 passed, 1 skipped`. The skip is limited to a Windows symbolic-link case when the current account lacks link-creation privileges; platform-independent containment tests still run.

Covered behavior includes manifest and path validation, normalized-box handling, training/evaluation/inference contracts, checkpoint safety, boundary clipping and degeneration, concise CLI errors, and regressions.

## Limitations

- one specialized dataset and task
- bounded training rather than a comprehensive hyperparameter study
- no external, multi-site, temporal, or prospective validation
- no demographic or acquisition-device subgroup evidence
- no calibration claim for detector confidence
- incomplete characterization of image-quality and distribution-shift failures
- annotations and source augmentation may encode dataset-specific conventions

See [Dataset card](DATASET_CARD.md), [Architecture](ARCHITECTURE.md), and [ML guide](../ml/README.md).
