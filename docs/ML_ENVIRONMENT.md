# ML Environment

## Purpose

The OralLens AI ML environment is isolated from the backend environment. It
provides reproducible image validation, training, evaluation, explainability,
and testing dependencies without installing packages globally.

## Supported Python

The project uses Python 3.13 and declares `>=3.13,<3.14`. This narrow range
prevents an environment from silently moving to a newer Python minor release
before the ML dependency stack has been verified against it.

The local interpreter selected during the feasibility pass is:

```text
C:\Python313\python.exe
```

uv commands use `--no-python-downloads` so environment creation cannot download
or select a different interpreter without review.

## Direct Dependencies

| Group | Package | Version | Direct purpose |
| --- | --- | --- | --- |
| Base | Pillow | 12.2.0 | Decode and validate image content |
| Training | PyTorch | 2.12.1 | Model training and tensor computation |
| Training | TorchVision | 0.27.1 | ResNet18, image transforms, and vision utilities |
| Training | Captum | 0.9.0 | Maintained model-explainability algorithms |
| Training | scikit-learn | 1.9.0 | Evaluation metrics and grouped data utilities |
| Development | Pytest | 9.1.1 | Automated test execution |

Only packages imported directly by project code or its test suite are declared
as direct dependencies. Transitive dependencies are selected and recorded by
uv in `uv.lock`; they must not be copied into `pyproject.toml` without a direct
project use.

## Package Sources

General dependencies resolve from the default Python Package Index. PyTorch and
TorchVision resolve from the official PyTorch CUDA 12.6 wheel index on Windows
and Linux.

The PyTorch index is configured with `explicit = true`. This prevents unrelated
packages from being resolved from that secondary index and limits dependency
confusion risk. macOS falls back to the CPU packages published on PyPI because
PyTorch does not publish CUDA builds for macOS.

## Verified Official Metadata

The dependency versions and Python constraints were checked on June 23, 2026:

- [PyTorch on PyPI](https://pypi.org/project/torch/)
- [TorchVision on PyPI](https://pypi.org/project/torchvision/)
- [Captum on PyPI](https://pypi.org/project/captum/)
- [scikit-learn on PyPI](https://pypi.org/project/scikit-learn/)
- [Pillow on PyPI](https://pypi.org/project/Pillow/)
- [Pytest on PyPI](https://pypi.org/project/pytest/)
- [Official PyTorch installation selector](https://pytorch.org/get-started/locally/)
- [Official uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/)

The visible TorchVision compatibility table had not yet been updated for the
new PyTorch 2.12.1 and TorchVision 0.27.1 releases. Their pairing is therefore
not accepted based on matching release dates. Successful uv resolution against
the published wheel metadata is a required gate before installation.

## Reproducible Workflow

Run these commands from the repository root after each command is reviewed and
approved:

```powershell
uv lock --project ml --python C:\Python313\python.exe --no-python-downloads --cache-dir ml\.uv-cache
uv sync --project ml --group training --python C:\Python313\python.exe --no-python-downloads --cache-dir ml\.uv-cache --locked
uv run --project ml --group training python -m pytest
```

The lockfile is source-controlled. The project-local `.venv` and `.uv-cache`
directories are generated and ignored.

## Verification Gates

An environment is accepted only when all of the following succeed:

- uv resolves the exact direct versions without overriding constraints.
- `uv sync --locked` completes without source builds or global installation.
- Package imports report the expected versions.
- Existing ML tests pass inside the ML environment.
- PyTorch reports CUDA available on the training workstation.
- A small tensor operation executes on the GPU.

## Verified Environment

Verification completed on June 23, 2026:

- Python: 3.13.5
- Locked dependency graph: 56 packages
- Installed environment: 35 packages
- PyTorch: 2.12.1+cu126
- TorchVision: 0.27.1+cu126
- Captum: 0.9.0
- scikit-learn: 1.9.0
- Pillow: 12.2.0
- Pytest: 9.1.1
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 4070 Laptop GPU
- CUDA tensor verification: passed with expected result `14.0`
- Automated tests: 45 passed, 1 capability-gated skip

The skipped test requires Windows permission to create symbolic links. A
platform-independent path-escape test passed, so dataset-root containment
remains covered on this workstation.

## Update Policy

Dependency updates are deliberate maintenance work, not automatic upgrades. For
each update:

1. Recheck official release metadata and Python compatibility.
2. Review security and breaking-change notices.
3. Update direct pins intentionally.
4. Regenerate and review `uv.lock`.
5. Run the full ML test and CUDA verification suite.
6. Record the decision and remaining limitations.

## Known Boundary

The existing setuptools build backend declaration remains unchanged in this
block. Build-tool pinning must be reviewed before publishing the ML package as a
distribution. It does not block the current project-local editable workflow.
