# ML Environment and Reproducibility

## Supported local environment

The ML package is configured for Python `>=3.13,<3.14`; the project version file selects Python 3.13. The completed experiments used Python 3.13.5 on Windows with an NVIDIA RTX 4070-class laptop GPU and approximately 8 GB of available GPU memory.

The project-local environment is:

`ml/.venv`

Do not install ML packages globally.

## Declared dependencies

`ml/pyproject.toml` is the dependency source of truth.

| Group | Packages | Purpose |
|---|---|---|
| base | Pillow 12.2.0 | image decoding and validation |
| dev | pytest 9.1.1 | automated tests |
| training | PyTorch 2.12.1, TorchVision 0.27.1 | detector training, evaluation, box operations, and inference |

PyTorch and TorchVision are resolved from the official PyTorch CUDA 12.6 package index on Windows/Linux. The lockfile and project-local environment should be treated as the reproducible installation boundary.

## Create or synchronize the environment

With `uv` available, run from the repository root:

```powershell
Push-Location ml
uv sync --group dev --group training
Pop-Location
```

This should create or update `ml/.venv`. Review dependency changes before accepting lockfile modifications.

## Verify the runtime

```powershell
ml\.venv\Scripts\python.exe --version
ml\.venv\Scripts\python.exe -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
nvidia-smi
```

GPU execution is selected by `device = "auto"` when CUDA is available. CPU execution remains possible but is slower.

## Official pretrained weights

The detector initializes from official TorchVision Faster R-CNN ResNet-50 FPN weights. TorchVision may populate the normal user cache on first acquisition. A cached official checkpoint outside the repository may be used read-only when explicitly approved; it must not be copied into source control.

Trained project checkpoints are written under ignored `ml/runs` paths and loaded as state dictionaries with `weights_only=True`.

## Common commands

Run from the repository root.

Training:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.train detection-baseline --config "ml\configs\orthodontic_plaque_detection_mvp_v3.toml"
```

Validation sweep:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_eval.toml"
```

Fixed-threshold internal test:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.evaluate detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_test.toml"
```

Prediction:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.predict detection --config "ml\configs\orthodontic_plaque_detection_mvp_v3_predict.toml" --image "C:\path\to\image.jpg"
```

Test suite:

```powershell
ml\.venv\Scripts\python.exe -B -m pytest ml\tests
```

## Generated artifacts

The following are local, reproducible evidence rather than repository source:

- raw and prepared datasets under `ml/data`
- trained checkpoints and loss metrics under `ml/runs/detection/...`
- validation and fixed-threshold internal evaluation metrics
- prediction JSON output
- package caches such as `ml/.uv-cache`

They are excluded by the root `.gitignore`. Do not commit data, checkpoints, run output, or temporary test artifacts.

## Verification baseline

Latest completed checks:

- ML suite: `150 passed, 1 skipped`
- v3 full-coverage training: completed
- complete validation threshold sweep: completed
- fixed-threshold internal test: completed
- representative v3 prediction: completed with 15 detections and explicit v3 model identity
- real backend-to-ML smoke: `1 passed`
- frontend build and manual browser-to-v3 flow: passed

The Windows skip is limited to symbolic-link creation permissions.

## Reproducibility notes

- configs use repository-root-relative data and output paths
- v3 preserves training seed `20260711`
- workers are fixed at zero for the current Windows/GPU experiment
- the threshold is selected on validation and frozen for test/inference
- checkpoint, config, dataset version, and metrics artifact must be reported together
- exact GPU kernels can remain nondeterministic unless deterministic algorithms are explicitly enabled and validated

See [Training and evaluation](TRAINING.md), [Dataset card](DATASET_CARD.md), and [ML guide](../ml/README.md).
