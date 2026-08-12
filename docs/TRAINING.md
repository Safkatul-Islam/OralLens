# Training and Evaluation

## Claim boundary

This document records reproducible object-detection experiments for a portfolio and learning project. Reported precision, recall, F1, and IoU describe annotation matching on one orthodontic-plaque dataset. They are not sensitivity, specificity, diagnostic accuracy, patient risk, or clinical validation.

> **Historical-results warning:** v1-v3 were trained and evaluated before a source-label semantic defect was corrected. Source label `0` means plaque absent in an annotated region, but the historical converter assigned every source box detector label `1`. The 1,170 affected annotations contaminate all v1-v3 losses and metrics. Those numbers are retained below to reproduce the experiment history, not as valid plaque-only performance evidence. No corrected post-defect model has yet been trained.

## Model contract

- framework: PyTorch and TorchVision
- architecture: Faster R-CNN with ResNet-50 FPN
- source labels: `0` plaque absent in the annotated region, `1` plaque present
- detector classes: `0` implicit background, `1` plaque candidate
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

Target conversion validates all source labels and boxes, keeps only source label `1` as plaque objects, and supports an empty positive target when a valid image contains no plaque-present regions. The prepared manifest contains 1,170 source label `0` annotations and 64,068 source label `1` annotations; every current image row contains at least one label `1`.

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
- each fixed-threshold internal test uses the validation-selected operating point and is not used for retuning
- generated checkpoints, metrics, predictions, and data are local ignored artifacts
- evaluator errors translate shared target-validation failures into concise evaluation-domain failures
- v3 writes atomic last/best training-state checkpoints after every completed epoch
- v3 resume restores model, optimizer, and RNG state and rejects incompatible configs or external output paths
- fresh training rejects existing final artifacts instead of silently overwriting an experiment

Run commands below from the repository root with the project-local ML environment.

## Historical v1 baseline

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

This weak baseline established that the historical pipeline ran but was not a useful final operating point. Its targets included plaque-absent source regions as positive objects.

## Historical v2 training experiment

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

Both losses decreased across the bounded run; no validation-loss reversal was observed. Because the targets used the incorrect source-label mapping, this is historical optimization evidence rather than evidence of a valid plaque detector.

Local outputs:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/checkpoint_last.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2/metrics.json`

## Historical v2 full-validation threshold selection

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

## Historical v2 fixed-threshold test evaluation

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

## Historical v2 trustworthiness evaluation

The trustworthiness evaluator measured the then-application-facing v2 policy without changing the frozen model or threshold. It references the historical evaluation configs and v2 prediction config, then compares score threshold `0.65`, IoU `0.5`, and maximum 25 detections per image with the uncapped evaluator.

Validation config: `ml/configs/orthodontic_plaque_detection_mvp_v2_trust_validation.toml`

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection-trustworthiness --config "ml\configs\orthodontic_plaque_detection_mvp_v2_trust_validation.toml"
```

Frozen test config: `ml/configs/orthodontic_plaque_detection_mvp_v2_trust_test.toml`

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection-trustworthiness --config "ml\configs\orthodontic_plaque_detection_mvp_v2_trust_test.toml"
```

| Evidence | Validation | Frozen test |
|---|---:|---:|
| Images | 468 | 858 |
| Patients | 7 | 12 |
| Displayed predictions | 5,962 | 10,274 |
| Cap-affected images | 0 | 0 |
| Truncated predictions | 0 | 0 |
| Score-to-match ECE | 0.0949 | 0.0971 |
| Score-to-match MCE | 0.2331 | 0.2245 |
| Score-to-match Brier score | 0.1640 | 0.1708 |
| Patient-level F1 range | 0.6950-0.8951 | 0.6332-0.8675 |

Because no image reached 25 displayed detections, deployed and uncapped precision, recall, F1, counts, and matched IoU are identical to the historical aggregate results.

### Score-to-match reliability

Each displayed prediction is labeled correct only when it matches one available annotation of the same class under score-ordered, one-to-one matching at IoU `0.5`. Equal-width bins then compare mean detector score with empirical box-match rate.

| Displayed score range | Validation mean score | Validation match rate | Test mean score | Test match rate |
|---|---:|---:|---:|---:|
| 0.65-0.70 | 0.6746 | 0.4416 | 0.6751 | 0.4507 |
| 0.70-0.80 | 0.7515 | 0.5458 | 0.7535 | 0.5843 |
| 0.80-0.90 | 0.8545 | 0.7335 | 0.8554 | 0.7464 |
| 0.90-1.00 | 0.9438 | 0.9392 | 0.9441 | 0.9188 |

The similar validation and test pattern shows that scores below `0.9` are overconfident as annotation-match indicators. The highest bin is closer to its empirical match rate. This is conditional score reliability among displayed boxes, not comprehensive object-detector calibration and never a probability of plaque, disease, clinical risk, or patient outcome.

### Failure and provenance evidence

- Dark brightness-down samples repeatedly appeared among images with the most false negatives on validation and test.
- Blur appeared among the largest held-out false-negative cases.
- Some highest-score false positives narrowly missed IoU `0.5`; others had almost no annotated overlap.
- Patient-level variation is substantial and is not visible in aggregate F1 alone.
- Reports record SHA-256 identities for the checkpoint, manifest, evaluation config, inference config, and trust config, plus runtime and GPU versions.
- Reports retain identifiers, boxes, matching outcomes, and coordinate-space metadata without copying source image bytes.

Local ignored artifacts:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2_trust_validation/trustworthiness_report.json`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v2_trust_test/trustworthiness_report.json`

## Historical v3 full-coverage experiment

Config: `ml/configs/orthodontic_plaque_detection_mvp_v3.toml`

V2 processed 512 training batches and 64 validation batches per epoch. V3 preserves the v2 architecture, optimizer, seed, image sizes, and three-epoch budget while removing both caps. Each epoch therefore processes all 3,834 training images and all 468 validation images. V3 starts fresh from the official pretrained TorchVision weights; it is not a continuation of v2.

Key settings:

| Setting | Value |
|---|---:|
| Epochs | 3 |
| Batch size | 1 |
| Learning rate | 0.005 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Training batches per epoch | 3,834 |
| Validation batches per epoch | 468 |
| Workers | 0 |
| Device | automatic |

Command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp_v3.toml"
```

Training completed in approximately 73 minutes 56 seconds on the local RTX 4070 Laptop GPU.

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.7727 | 0.7848 |
| 2 | 0.6232 | 0.7451 |
| 3 | 0.5418 | 0.6757 |

Both losses decreased through epoch 3. The lowest validation loss occurred at epoch 3, so `checkpoint_best.pt` and `checkpoint_last.pt` contain the same model state. The checkpoint SHA-256 recorded after training was:

```text
0782117A3B3F868F5DB941D3D54D2988A46BEBC9DD7B5DA75BF0414CE55F987F
```

Local ignored outputs:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/checkpoint_last.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/checkpoint_best.pt`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3/metrics.json`

The general training checkpoint includes the model state, optimizer state, completed epoch, metrics, device type, and CPU/CUDA RNG state. It is loaded with `weights_only=True`. This supports interruption recovery. Because the checkpoint learned the incorrect target mapping, v4 must initialize fresh from official pretrained weights and must not resume v3.

### Historical v3 validation threshold selection

Config: `ml/configs/orthodontic_plaque_detection_mvp_v3_eval.toml`

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_eval.toml"
```

The complete 468-image validation split selected score threshold `0.85` at IoU `0.5`. The tested peak was broad: threshold `0.80` produced F1 `0.7771`, only `0.0005` below the `0.85` result. No finer search was performed.

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.80 | 0.7304 | 0.8301 | 0.7771 |
| **0.85** | **0.7629** | **0.7929** | **0.7776** |
| 0.90 | 0.8040 | 0.7392 | 0.7702 |

At `0.85`: TP 4,813; FP 1,496; FN 1,257; predictions 6,309; mean matched IoU `0.8034`.

Compared with the selected v2 validation point, v3 changed:

- precision: `-0.0023`
- recall: `+0.0414`
- F1: `+0.0193`
- mean matched IoU: `+0.0547`

The selected threshold was copied to the v3 prediction and test configs before the v3 test run.

### Historical v3 fixed-threshold internal test benchmark

Config: `ml/configs/orthodontic_plaque_detection_mvp_v3_test.toml`

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_test.toml"
```

The fixed-threshold run covered all 858 test images and 11,070 targets at score threshold `0.85` and IoU `0.5`.

| Precision | Recall | F1 | Mean matched IoU |
|---:|---:|---:|---:|
| 0.7671 | 0.7923 | 0.7795 | 0.8102 |

Counts: TP 8,771; FP 2,663; FN 2,299; predictions 11,434.

Compared with v2 on the same internal cohort, v3 improved precision by `0.0089`, recall by `0.0886`, F1 by `0.0496`, and mean matched IoU by `0.0561`, with 981 fewer false negatives.

Artifact: `ml/runs/detection/orthodontic_plaque_part2_mvp_v3_test/evaluation_metrics.json`

The test did not change the threshold or model. However, the cohort has now been inspected for v2 and v3. It remains useful as an internal fixed-policy benchmark, but it is not a pristine external evaluation boundary for future model generations or medical claims.

### Historical v3 trustworthiness evidence

Validation config: `ml/configs/orthodontic_plaque_detection_mvp_v3_trust_validation.toml`

Test config: `ml/configs/orthodontic_plaque_detection_mvp_v3_trust_test.toml`

| Evidence | Validation | Internal test benchmark |
|---|---:|---:|
| Images | 468 | 858 |
| Patients | 7 | 12 |
| Uncapped predictions | 6,309 | 11,434 |
| Cap-affected images | 0 | 1 |
| Truncated predictions | 0 | 1 false positive |
| Score-to-match ECE | 0.1940 | 0.1878 |
| Score-to-match MCE | 0.4290 | 0.4046 |
| Score-to-match Brier score | 0.2075 | 0.2035 |
| Patient-level F1 range | 0.6769-0.9245 | 0.6824-0.8889 |

The cap removed one false positive and no true positive on the single affected test image. Aggregate deployed-cap F1 was `0.7795` after rounding, effectively identical to uncapped F1.

V3's ECE, MCE, and Brier score are worse than v2's despite better precision, recall, F1, matched IoU, and internal patient-level test bounds. This is an explicit tradeoff: v3 is the stronger detector, but its raw scores are more overconfident as annotation-match indicators.

Dark, blurred, and rotated variants remain recurring false-negative or false-positive cases. These source-dataset variants identify robustness hypotheses; they do not quantify real multi-clinic acquisition performance.

Local ignored artifacts:

- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3_trust_validation/trustworthiness_report.json`
- `ml/runs/detection/orthodontic_plaque_part2_mvp_v3_trust_test/trustworthiness_report.json`

## Inference

Active v3 config: `ml/configs/orthodontic_plaque_detection_mvp_v3_predict.toml`

It declares model identity `orthodontic-plaque-mvp-v3`, fixes score threshold `0.85`, and caps returned detections at 25.

V3 remains wired into the local application as historical end-to-end engineering evidence. It is not a corrected plaque model, and its poor behavior on consumer-style challenge images is consistent with both the target defect and the narrow standardized training domain. Do not promote its historical metrics as present model quality.

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_predict.toml" --image "C:\path\to\image.jpg"
```

Prediction JSON is retained locally under the configured ignored run directory. The backend default and both ML launchers select this v3 config. Promotion was verified with the real checkpoint integration smoke, full backend and ML suites, frontend build, and manual browser scan.

## Verification

The latest complete ML suite after the target-conversion, source-admission, and source-aware manifest corrections: `174 passed, 1 skipped`. The skip is limited to a Windows symbolic-link case when the current account lacks link-creation privileges; platform-independent containment tests still run.

Covered behavior includes manifest and path validation, normalized-box handling, source-label validation and plaque-only filtering, empty positive targets, training/evaluation/inference contracts, checkpoint safety, boundary clipping and degeneration, concise CLI errors, and regressions.

## Stopped v4 diagnostic

V4 started fresh with the corrected plaque-present target conversion and source-aware manifest. It still used the same underlying AIRC image distribution. The user stopped training before epoch 3 after the second completed epoch showed a clear validation-loss reversal.

Config: `ml/configs/orthodontic_plaque_detection_mvp_v4.toml`

Local ignored output directory: `ml/runs/detection/orthodontic_plaque_part2_mvp_v4`

| Completed epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | `0.7966761603586116` | `0.8249539341299962` |
| 2 | `0.6149270393257503` | `1.028658177671779` |

Training loss decreased by approximately `22.8%` from epoch 1 to epoch 2, while validation loss increased by approximately `24.7%`. `checkpoint_best.pt` is epoch 1; `checkpoint_last.pt` is epoch 2. Read-only `weights_only=True` inspection confirmed both checkpoint structures, `device_type="cuda"`, and one recorded CUDA RNG state. No temporary checkpoint files remained after the stop.

The device record confirms use of the CUDA code path for the completed checkpoints. It does not prove sustained GPU utilization because the trainer did not record utilization, throughput, data-loading time, or peak memory. The v4 config also used `batch_size = 1`, `num_workers = 0`, and `device = "auto"`, so future training needs an explicit CUDA requirement and telemetry rather than inference from Task Manager alone.

V4 has no authorized validation sweep, threshold, fixed-threshold test result, prediction result, backend integration, or promotion. The existing evaluation and resume configs must not be run. Do not resume epoch 3 or cite these losses as condition-localization performance.

## Authorization gate for any future experiment

Before another model run:

1. freeze one condition-specific task: ordinary-RGB visible plaque or ordinary-RGB supragingival calculus
2. admit genuinely complementary target-domain data through the provenance, license, patient/source, acquisition, and annotation gate
3. keep plaque and calculus labels separate and use spatial supervision compatible with the chosen output
4. define patient-, source-, and derivative-aware development and locked challenge boundaries
5. require CUDA explicitly and log device, model/tensor placement, batch/epoch timing, throughput, utilization samples, and peak GPU memory
6. run a monitored smoke test before a small pilot
7. predeclare the pilot baseline, budget, success criteria, and early-stop rules
8. create a fresh experiment identity and output directory; do not resume v1-v4
9. select policy on validation only and preserve the locked challenge boundary

Dataset admission and the go/no-go gate are defined in [Data strategy](DATA_STRATEGY.md).

## Limitations

- one specialized dataset and task
- only 74 patient groups despite thousands of images and annotations
- the active v3 model and all v1-v3 metrics use the incorrect historical target mapping
- the corrected v4 run stopped after early validation-loss reversal and has no evaluation or promotion evidence
- every current manifest row contains positive annotations, so the dataset cannot estimate clinical specificity or NPV in a representative negative population
- no external, multi-site, temporal, or prospective validation
- no demographic or acquisition-device subgroup evidence
- score-to-match reliability is measured only for displayed in-dataset predictions; comprehensive calibration across location, scale, and shift remains incomplete
- dark/blur failure signals are identified, but controlled image-quality and distribution-shift characterization remains pending
- annotations and source augmentation may encode dataset-specific conventions
- the internal test cohort is no longer a pristine evidence boundary for future model generations

See [Data strategy](DATA_STRATEGY.md), [Dataset card](DATASET_CARD.md), [Intended use and claims](INTENDED_USE_AND_CLAIMS.md), and [ML guide](../ml/README.md).
