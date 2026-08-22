# Training and Evaluation

## Current evidence boundary

The final OralLens model is a single-class object detector for **AIRC plaque-positive peri-tooth regions**.

- detector class `0`: implicit background
- detector class `1`: plaque-positive peri-tooth region
- architecture: TorchVision Faster R-CNN with ResNet-50 FPN
- input: RGB `float32` `[0,1]`
- model resize: minimum 512, maximum 768
- final checkpoint: Experiment 1 epoch 9
- final score threshold: `0.80`

This target is not equivalent to a discrete visible-plaque deposit. Metrics below measure agreement with the AIRC region annotations, not diagnosis, disease probability, or consumer-photo performance.

Historical v1-v3 experiments used an incorrect conversion that mapped source label `0` and source label `1` boxes to foreground. Their metrics remain historical pipeline evidence only and are not compared with the corrected final detector.

## Dataset and leakage controls

Source: verified AIRC/LabDen Part 2 orthodontic-plaque dataset.

The source-aware v4 manifest contains 5,160 stored rows across 74 patients and six publisher variants. The final experiment requires both:

- `variant = "original"`
- exclusion of sample IDs ending in `_blur`, `_dark`, or `_light`

This produces the genuinely original population:

| Split | Images | Patients | Foreground class-1 regions | Source class-0 regions |
|---|---:|---:|---:|---:|
| Train | 480 | 55 | 5,502 | 124 |
| Validation | 58 | 7 | 710 | 4 |
| Test | 107 | 12 | 1,293 | 23 |

Patient groups and derivative/original families are isolated to one partition. Source class `0` regions are validated but are not emitted as detector objects.

Evidence:

- local manifest: `ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`
- dataset loader: `ml/src/orallens_ml/data/orthodontic_plaque_dataset.py`
- target conversion: `ml/src/orallens_ml/training/detection.py`

## Corrected v4 baseline

Config:

`ml/configs/orthodontic_plaque_detection_mvp_v4.toml`

The baseline started from official TorchVision pretrained weights and used all 3,834 stored training rows as independent samples. It preserved the established optimizer settings:

| Setting | Value |
|---|---:|
| Batch size | 1 |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Trainable backbone layers | 3 |
| Image range | 512–768 |
| Seed | 20260711 |

Two epochs completed before the run was stopped:

| Epoch | Train loss | Validation loss |
|---:|---:|---:|
| **1** | 0.796676 | **0.824954** |
| 2 | 0.614927 | 1.028658 |

Training loss decreased while validation loss worsened. Epoch 1 remained the lowest-loss checkpoint.

### Baseline evaluation contract

- checkpoint: corrected v4 epoch 1
- test: genuine originals only
- patients: 12
- images: 107
- foreground targets: 1,293
- IoU operating threshold: 0.50
- AP IoUs: 0.50 through 0.95 in 0.05 increments
- score threshold: `0.75`, selected on originals-only validation F1

Config:

`ml/configs/orthodontic_plaque_detection_mvp_v4_baseline_test.toml`

## Problem identified

The prepared training population contained publisher-generated brightness, flip, and rotation derivatives. Treating them as independent samples caused original families to be repeatedly represented through highly correlated transformations. Even rows marked `variant="original"` included source files ending in `_blur`, `_dark`, or `_light` unless those suffixes were explicitly excluded.

The observed baseline reversal, limited 480-image genuine-original population, and error analysis motivated one controlled experiment rather than more unstructured training.

Baseline held-out errors at threshold `0.75`:

- localization failures: 95
- low-confidence matches: 216
- background false positives: 292
- duplicate detections: 7

## Experiment 1: original-family-balanced online augmentation

Training config:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug.toml`

Controlled change:

1. retain only 480 genuine training originals
2. remove stored derivatives as independent samples
3. apply conservative augmentation online
4. keep Faster R-CNN and optimizer settings unchanged
5. keep optimizer exposure approximately equal to the completed baseline

Online augmentation:

| Transform | Setting |
|---|---:|
| Horizontal flip | probability 0.50 |
| Color jitter | probability 0.80 |
| Brightness | 0.12 |
| Contrast | 0.12 |
| Saturation | 0.08 |
| Hue | 0.02 |
| Gaussian blur | disabled |

Exposure comparison:

- corrected baseline: 3,834 samples x 2 completed epochs = 7,668 optimizer steps
- Experiment 1: 480 samples x 16 maximum epochs = 7,680 optimizer steps

Sixteen epochs were a maximum comparable-exposure budget, not an instruction to select the last epoch.

## Checkpoint selection

The checkpoint was selected by lowest loss on the 58-image originals-only validation set.

| Epoch | Validation loss | Note |
|---:|---:|---|
| 1 | 0.875290 | initial |
| 5 | 0.690110 | improving |
| **9** | **0.657865** | selected minimum |
| 10 | 0.986758 | transient reversal |
| 11 | 0.667123 | recovered |
| 16 | 0.693573 | last epoch, not selected |

After epoch 9, training loss continued decreasing while validation loss fluctuated above the minimum. Online augmentation delayed overfitting but did not eliminate it.

Checkpoint:

`ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug/checkpoint_best.pt`

- selected epoch: 9
- size: 330,069,425 bytes
- SHA-256: `79CBB3B99D56F77E852B5096446649EFF17858A25B5FD40DAB9408FA7A2F19D7`

The checkpoint is a local ignored artifact and is loaded with `weights_only=True`.

## Threshold selection

Validation config:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_validation.toml`

The complete originals-only validation sweep selected `0.80` by maximum tested F1:

| Threshold | Precision | Recall | F1 | FP/image |
|---:|---:|---:|---:|---:|
| 0.75 | 0.764781 | 0.838028 | 0.799731 | 3.155 |
| **0.80** | **0.800275** | **0.818310** | **0.809192** | **2.500** |
| 0.85 | 0.822222 | 0.781690 | 0.801444 | 2.069 |

Validation AP for the selected checkpoint:

- AP@0.50: `0.837588`
- mAP@0.50:0.95: `0.513031`

The epoch and threshold were frozen before accessing the final test result.

## Final held-out test

The final candidate was evaluated once on the same 107-image originals-only, 12-patient test population used by the corrected baseline. No threshold, checkpoint, architecture, or preprocessing decision was made from test examples or results.

| Metric | Baseline | Final | Absolute change | Relative change |
|---|---:|---:|---:|---:|
| AP@0.50 | 0.803432 | **0.833486** | +0.030054 | +3.74% |
| mAP@0.50:0.95 | 0.419837 | **0.516427** | +0.096591 | +23.01% |
| Precision | 0.720567 | **0.783033** | +0.062466 | +8.67% |
| Recall | 0.785770 | **0.806651** | +0.020882 | +2.66% |
| F1 | 0.751757 | **0.794667** | +0.042909 | +5.71% |
| False positives/image | 3.682243 | **2.700935** | -0.981308 | -26.65% |
| Mean matched IoU | 0.780843 | **0.823955** | +0.043112 | +5.52% |

Counts:

| Evidence | Baseline | Final | Change |
|---|---:|---:|---:|
| True positives | 1,016 | **1,043** | +27 |
| False positives | 394 | **289** | -105 |
| False negatives | 277 | **250** | -27 |
| Localization failures | 95 | **58** | -37 |
| Low-confidence matches | 216 | **179** | -37 |
| Background false positives | 292 | **230** | -62 |
| Duplicate detections | 7 | **1** | -6 |

The AP improvement is threshold-independent, while the operating-point metrics use each model's validation-selected threshold. The consistent gains support a genuine in-distribution detector improvement.

Final artifact:

`ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug_test/evaluation_metrics.json`

The final test configuration was created in memory from the stored baseline-test methodology with the frozen checkpoint, `0.80` threshold, and final output directory substituted. The metrics artifact records the completed population and operating point. No additional test evaluation is authorized.

## Production integration

Inference config:

`ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml`

Production uses the same model constructor, class count, image sizing, RGB tensor conversion, checkpoint, and threshold. `max_detections = 100` matches the detector's bounded returned population used by evaluation.

A real FastAPI `POST /scans` smoke:

- returned HTTP `201`
- identified the final epoch-9 model
- passed technical input assessment
- returned 14 class-1 detections
- produced scores from `0.836087` to `0.987487`
- removed the temporary upload

Production additionally applies technical-quality abstention; offline evaluation did not. That is the only intentional behavioral boundary.

## Verification

Latest relevant checks:

- complete ML suite after Experiment 1 implementation: `215 passed, 2 skipped`
- complete backend suite after final promotion: `32 passed, 1 skipped`
- final backend config/inference contracts: `13 passed`
- real backend-to-model integration smoke: `1 passed`

The skipped ML cases are Windows symlink-privilege dependent; platform-independent containment checks remain active.

## Artifact policy

Raw/prepared data, checkpoints, metrics, predictions, caches, and scans are ignored by git. They remain local evidence and must not be committed or silently overwritten.

Key local artifacts:

- baseline metrics: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4/metrics.json`
- baseline test: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4_baseline_test/evaluation_metrics.json`
- final training: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug/metrics.json`
- final validation: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug_validation/evaluation_metrics.json`
- final test: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug_test/evaluation_metrics.json`
- production smoke: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug_predictions/`

## Limitations

- one source family and 74 patients overall
- 480 genuine training originals
- only 12 held-out test patients
- no independent external or consumer-photo evaluation
- no representative clean-mouth or consumer hard-negative population
- annotation semantics describe plaque-positive peri-tooth regions rather than exact visible deposits
- test results cannot estimate clinical specificity, NPV, prevalence performance, or diagnostic safety
- no demographic, acquisition-device, or site subgroup evidence
- technical abstention does not establish oral ROI or semantic in-distribution status
- detector scores are not calibrated plaque or disease probabilities

The final result is suitable for an honest object-detection engineering case study. It is not evidence of clinical validation.
