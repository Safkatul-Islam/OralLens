# Training

## Scope

The current ML training path is a baseline object detector for the verified
Part 2 orthodontic plaque manifest. It uses TorchVision Faster R-CNN and the
project's manifest loader to convert normalized source annotations into
TorchVision pixel-space detection targets.

This baseline is an engineering and learning milestone. It is not a validated
clinical model, must not be described as diagnostic, and must not be used for
treatment decisions.

## Architecture

Training code is separated from dataset acquisition:

```text
ml/src/orallens_ml/
  data/
    orthodontic_plaque_dataset.py
  modeling/
    detection.py
  training/
    detection.py
  evaluation/
    detection.py
  inference/
    detection.py
  cli/
    evaluate.py
    predict.py
    train.py
```

Responsibilities:

- `orthodontic_plaque_dataset.py` loads the generated Part 2 manifest, validates
  paths, decodes images, and returns normalized boxes.
- `modeling/detection.py` owns shared detector construction, state-dict-only
  checkpoint writing, checkpoint loading, and checkpoint/config compatibility
  validation.
- `training/detection.py` validates training config, adapts targets to
  TorchVision format, runs training, and writes artifacts.
- `evaluation/detection.py` loads checkpoints and reports transparent
  IoU-threshold validation metrics.
- `inference/detection.py` loads checkpoints, validates one input image, filters
  predictions by score, and writes JSON prediction artifacts.
- `cli/train.py` exposes the training entrypoint without mixing it into dataset
  acquisition commands.

## Configs

Two configs are intentionally kept separate:

| Config | Purpose |
| --- | --- |
| `ml/configs/orthodontic_plaque_detection_smoke.toml` | Fast CPU smoke run against real data. Uses one train batch and one validation batch. |
| `ml/configs/orthodontic_plaque_detection_eval_smoke.toml` | Fast CPU evaluation smoke run against the smoke checkpoint and one validation batch. |
| `ml/configs/orthodontic_plaque_detection_predict_smoke.toml` | Fast CPU one-image prediction smoke run against the smoke checkpoint. |
| `ml/configs/orthodontic_plaque_detection_mvp.toml` | Bounded pretrained MVP run against real Part 2 data. Uses official TorchVision weights, one epoch, and capped train/validation batches. |
| `ml/configs/orthodontic_plaque_detection_mvp_eval.toml` | Bounded validation evaluation for the MVP checkpoint. |
| `ml/configs/orthodontic_plaque_detection_mvp_predict.toml` | One-image prediction run for the MVP checkpoint. |
| `ml/configs/orthodontic_plaque_detection_baseline.toml` | Baseline training run against the verified Part 2 manifest. Uses default TorchVision weights and full configured epochs. |

These configs keep outputs under `ml/runs/detection/`, which is ignored by git.
Dataset files, checkpoints, and metrics artifacts must not be committed.

## Smoke Run

Run the bounded smoke command first after any training-code change:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_smoke.toml"
```

Expected outputs:

```text
ml/runs/detection/orthodontic_plaque_part2_smoke/
  checkpoint_last.pt
  metrics.json
```

The current smoke run completed successfully with:

```text
status: detection-baseline-trained
train_loss: 1.9249593019485474
validation_loss: 1.2211699485778809
```

These loss values only prove that the command, loader, model, optimizer,
checkpoint writer, and metrics writer executed on real data. They are not model
quality claims.

## Baseline Run

The baseline command is:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_baseline.toml"
```

Expected outputs:

```text
ml/runs/detection/orthodontic_plaque_part2_baseline/
  checkpoint_last.pt
  metrics.json
```

The baseline config uses TorchVision's default Faster R-CNN weights. If weights
are not already present in the local Torch cache, the run fails before
TorchVision can start an implicit download. This keeps baseline training
explicit and reviewable: approve an official TorchVision weights download first,
or switch `pretrained_weights` to `"none"` for smoke/offline development.

The cache path follows Torch's documented behavior:

```text
torch.hub.get_dir()/checkpoints/<official-weight-filename>
```

## MVP Pretrained Run

The MVP run uses the same verified Part 2 data and official TorchVision weights,
but caps the number of batches so the project gets a real checkpoint quickly
without over-optimizing before the product path is complete:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp.toml"
```

Current bounded MVP result:

```text
status: detection-baseline-trained
train_loss: 1.2272009430453181
validation_loss: 0.9619900770485401
```

This checkpoint is useful for backend and inference integration. It is not a
final model-quality claim.

## Evaluation

The first evaluation layer reports transparent IoU-threshold detection metrics
instead of full COCO mAP. It uses `torchvision.ops.box_iou` to greedily match
predictions to ground-truth boxes of the same class after score filtering.

Smoke evaluation command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_eval_smoke.toml"
```

Expected output:

```text
ml/runs/detection/orthodontic_plaque_part2_smoke_eval/
  evaluation_metrics.json
```

The current smoke evaluation completed successfully against the smoke
checkpoint and one validation batch:

```text
score_threshold=0.05: precision=0.0 recall=0.0 f1=0.0 tp=0 fp=1 fn=8
score_threshold=0.5: precision=0.0 recall=0.0 f1=0.0 tp=0 fp=0 fn=8
```

This is expected for an untrained smoke checkpoint. The result validates
checkpoint loading, model inference, metric aggregation, and JSON output
writing, not model quality.

Bounded MVP evaluation command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_eval.toml"
```

Current bounded MVP validation result at IoU `0.5`:

| Score threshold | Precision | Recall | F1 | TP | FP | FN | Predictions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.050 | 0.0278 | 0.2816 | 0.0506 | 178 | 6222 | 454 | 6400 |
| 0.075 | 0.0278 | 0.2816 | 0.0506 | 178 | 6222 | 454 | 6400 |
| 0.100 | 0.0278 | 0.2816 | 0.0506 | 178 | 6222 | 454 | 6400 |
| 0.125 | 0.0319 | 0.2627 | 0.0569 | 166 | 5041 | 466 | 5207 |
| **0.150** | **0.0489** | **0.1361** | **0.0720** | **86** | **1671** | **546** | **1757** |
| 0.175 | 0.0502 | 0.0348 | 0.0411 | 22 | 416 | 610 | 438 |
| 0.200 | 0.0116 | 0.0016 | 0.0028 | 1 | 85 | 631 | 86 |
| 0.225 | 0.0000 | 0.0000 | 0.0000 | 0 | 15 | 632 | 15 |
| 0.250 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 632 | 0 |

The MVP inference threshold is `0.15` because it produced the highest measured
validation F1 in this sweep. Compared with `0.05`, it reduced false positives
and retained predictions by approximately 73%, while increasing precision and
F1. Recall decreased from approximately 28% to 14%, so this is an MVP operating
point rather than evidence of a high-quality detector. Raising the threshold to
`0.175` produced only a small precision gain and reduced recall to approximately
3%.

These results show that the pipeline is learning a weak objectness signal, but
the detector remains poorly calibrated after the intentionally bounded run.

Reported fields include IoU threshold, score threshold, true positives, false
positives, false negatives, precision, recall, F1, prediction count, target
count, and mean matched IoU. These validation metrics are useful for engineering
iteration, but they are not final held-out test metrics and are not diagnostic
claims.

## Inference

The inference path runs checkpoint-backed prediction on a single image and
writes a JSON artifact. It validates the checkpoint path, image file, extension,
device setting, thresholds, prediction tensor shapes, score finiteness, label
dtypes, and output path.

Smoke prediction command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_predict_smoke.toml" --image "ml\data\raw\orthodontic_plaque\v3\extracted\part-2\mendeley-dataset-materials_Part_2\data\images\patient0144\patient0144_20260118_bottom-left.jpg"
```

Expected output:

```text
ml/runs/detection/orthodontic_plaque_part2_smoke_predictions/
  patient0144_20260118_bottom-left.json
```

The current smoke prediction completed successfully with zero predictions at
the configured `0.5` score threshold. That is expected for the untrained smoke
checkpoint and validates the runtime path, not model quality.

MVP prediction command:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_predict.toml" --image "ml\data\raw\orthodontic_plaque\v3\extracted\part-2\mendeley-dataset-materials_Part_2\data\images\patient0144\patient0144_20260118_bottom-left.jpg"
```

The current MVP prediction completed successfully with `25` retained
predictions at the configured `0.15` score threshold. This image still reaches
the configured 25-detection cap because at least 25 candidate scores exceed the
threshold. The validation-wide reduction in false positives therefore does not
guarantee a smaller result on every image. Scores remain low after the
intentionally bounded run, so these predictions are an integration artifact,
not a reliable clinical output.

## Security Controls

- Dataset roots and manifest paths are validated before training.
- Manifest image paths must be relative POSIX paths and resolve inside the
  dataset root.
- Checkpoint loads use PyTorch `weights_only=True`; the implementation fails
  closed instead of falling back to unrestricted pickle loading.
- Evaluation and inference reject checkpoints whose saved architecture contract
  does not match the runtime config for class count or image-size policy.
- Faster R-CNN default pretrained weights must already exist in the local Torch
  cache; the project does not trigger hidden weight downloads during model
  construction.
- Output paths are constrained to named files inside the configured output
  directory.
- Existing symlink output files are rejected.
- The training CLI reports concise errors instead of exposing stack traces for
  normal validation failures.
- Raw data, prepared manifests, checkpoints, and metrics are kept out of git by
  project ignore rules.

## Verification

The training block is covered by tests for:

- Config parsing and validation failures.
- Detection-target conversion from normalized boxes to pixel boxes.
- Binary v1 label mapping from preserved publisher `class_id` values to
  TorchVision foreground label `1`.
- Detection collate behavior for variable image sizes.
- Label bounds validation.
- Output-path safety.
- A tiny CPU training smoke path that writes checkpoint and metrics artifacts.
- IoU matching, score filtering, empty prediction handling, evaluation config
  validation, output-path safety, and a tiny evaluation smoke path.
- One-image inference config validation, image validation, prediction filtering,
  non-finite score rejection, label dtype validation, output-path safety, and a
  tiny inference smoke path.
- Shared model/checkpoint helpers, incompatible-checkpoint rejection, and direct
  train/evaluate/predict CLI entrypoint JSON/error behavior.
- Pretrained-weight cache policy for offline smoke runs and baseline runs that
  use TorchVision default weights.

Latest complete ML suite:

```text
127 passed, 1 skipped
```

The skipped case requires Windows symbolic-link creation permission. The
platform-independent path-containment tests still run.

## Current Limitations

- Full detection mAP and patient-level confidence intervals are a later block.
- The held-out test split must remain untouched until final evaluation.
- Validation loss is computed through the TorchVision training-loss path with
  BatchNorm modules forced to eval mode during validation batches.
- Part 1 is excluded because its nested 7z fails official integrity testing.
- The training labels currently use a single foreground detection class plus
  preserved tooth-position metadata for auditability.
