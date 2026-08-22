# ML Environment and Reproducibility

## Supported local environment

The ML package targets Python `>=3.13,<3.14`; completed experiments used Python 3.13.5 on Windows with an NVIDIA RTX 4070-class laptop GPU and approximately 8 GB of GPU memory.

The project-local environment is:

`ml/.venv`

Do not install ML packages globally.

## Declared dependencies

`ml/pyproject.toml` is the dependency source of truth.

| Group | Packages | Purpose |
|---|---|---|
| base | Pillow 12.2.0 | image decoding and validation |
| dev | pytest 9.1.1 | automated tests |
| training | PyTorch 2.12.1, TorchVision 0.27.1 | training, evaluation, box operations, and inference |

PyTorch and TorchVision resolve from the official PyTorch CUDA 12.6 package index on Windows/Linux. The lockfile and project-local environment form the reproducible installation boundary.

## Create or synchronize the environment

Run from the repository root:

```powershell
Push-Location ml
uv sync --group dev --group training
Pop-Location
```

Review dependency and lockfile changes before accepting them.

## Verify the runtime

```powershell
ml\.venv\Scripts\python.exe --version
ml\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

Training configurations may select CUDA through `device = "auto"` when available. Completed checkpoint metadata and run evidence—not Task Manager alone—must be used to verify the execution device.

## Official initialization weights and checkpoints

The detector initializes from official TorchVision Faster R-CNN ResNet-50 FPN weights. TorchVision may populate its normal user cache on first acquisition. Any cached official file outside the repository is read-only and requires explicit access approval.

Project checkpoints are written below ignored `ml/runs` paths and loaded as state dictionaries with `weights_only=True`.

Frozen final checkpoint:

`ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug/checkpoint_best.pt`

SHA-256:

`79CBB3B99D56F77E852B5096446649EFF17858A25B5FD40DAB9408FA7A2F19D7`

The checkpoint records Experiment 1 epoch 9. It is paired with threshold `0.80` and must not be replaced by the last epoch.

## Frozen configuration set

- training provenance: `ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug.toml`
- validation selection: `ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_validation.toml`
- production prediction: `ml/configs/orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml`

Training and evaluation are complete for the current scope. These configurations remain as provenance, not authorization to retrain, retune, or re-evaluate the frozen test set.

## Prediction

Run from the repository root:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml" --image "C:\path\to\image.jpg"
```

Inference performs safe decoding and technical input assessment, builds the same Faster R-CNN contract used during evaluation, loads epoch 9, and filters detections at `0.80`.

## Tests

```powershell
ml\.venv\Scripts\python.exe -B -m pytest ml\tests
```

Latest relevant verification:

- complete ML suite after Experiment 1: `215 passed, 2 skipped`
- complete backend suite after final promotion: `32 passed, 1 skipped`
- real FastAPI-to-model integration smoke: `1 passed`
- frontend Chromium interaction/accessibility suite: `5 passed`
- frontend production build: passed

The Windows skips are limited to symbolic-link privilege constraints while platform-independent containment tests continue to run.

## Generated artifacts

The following are local evidence and are excluded from version control:

- raw and prepared data under `ml/data`
- checkpoints and run metrics under `ml/runs`
- validation and final test evaluation artifacts
- prediction JSON output
- package caches such as `ml/.uv-cache`

Do not commit datasets, checkpoints, run outputs, user images, or temporary artifacts.

## Reproducibility rules

- use repository-root-relative paths in configs
- keep the model architecture, preprocessing, checkpoint, threshold, dataset population, and metrics artifact identified together
- select checkpoints and thresholds on validation only
- do not use the frozen test result for tuning
- retain patient and original-family isolation
- preserve exact checkpoint and configuration identities for production parity
- report hardware/runtime metadata for any future experiment

See [training and evaluation](TRAINING.md), the [dataset card](DATASET_CARD.md), and the [ML guide](../ml/README.md).

