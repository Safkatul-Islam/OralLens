# Learning Log

This file will track what is learned while building OralLens AI. The goal is to understand the system, not blindly copy code.

## How To Use This Log

For each build phase, record:

- What was built
- Why it matters
- New concepts learned
- Mistakes or blockers
- How the issue was solved
- Resume-ready takeaway

## Phase 1: Project Design

### What We Are Building

An end-to-end AI/ML application that screens oral images and returns explainable model results through a web interface.

### Why It Matters

Hiring managers want to see that a candidate can connect ML work to a usable product. A model alone is not enough. A strong project shows data handling, evaluation, serving, UI, documentation, and responsible use.

### Concepts

- End-to-end ML project: a project that covers data, model, API, UI, and deployment thinking.
- Computer vision: machine learning applied to images.
- Inference: using a trained model to make a prediction on new input.
- Confidence score: the model's estimated certainty for a prediction.
- Explainability: tools or outputs that help humans understand why a model produced a result.
- Responsible AI: building AI systems with clear limitations, safety boundaries, and honest claims.

### Current Status

The project concept and initial documentation structure have been created.

## Phase 2: Repository Structure

Planned learning goals:

- Understand how production projects separate frontend, backend, ML, and documentation.
- Learn why clear folder structure helps recruiters, collaborators, and future maintainers.

## Phase 3: Backend API

Planned learning goals:

- Learn what an API is.
- Learn how a frontend sends an image to a backend.
- Learn request validation, error handling, and response schemas.
- Learn how tests protect backend behavior.

## Phase 4: ML Baseline

Planned learning goals:

- Learn dataset splits.
- Learn image preprocessing.
- Train or wire a baseline model.
- Understand accuracy, precision, recall, F1, and confusion matrix.

### Data Foundation

Before training, the project defines a normalized manifest instead of coupling
model code directly to an external archive layout. This creates a stable boundary
between publisher-specific metadata and the rest of the ML system.

Key lessons:

- A manifest makes sample identity, patient grouping, labels, source images, and
  augmentations explicit and reviewable.
- Input validation matters in offline ML pipelines too. Unsafe paths, missing
  files, conflicting metadata, and duplicate records can invalidate an experiment.
- Image-level random splitting is unsafe when multiple views or augmentations come
  from one patient. All records for a patient must remain in one split.
- Reproducibility requires stable split logic. Python's process-randomized hash is
  unsuitable, so the splitter orders patient IDs with a seeded SHA-256 digest.
- A deterministic split is not automatically a balanced split. Class and patient
  distributions must be audited before deciding whether grouped stratification is
  required.

Current status: manifest, audit, and grouped-split code has been added. Automated
verification completed with 19 passing tests and one capability-gated skip. The
skipped test covers symbolic-link escape on Windows, where this account lacks
permission to create symbolic links; a platform-independent path traversal test
still verifies dataset-root containment.

### Reproducible ML Environment

The ML environment is intentionally separate from the backend environment. A
web API and a model-training workstation have different dependency sizes,
hardware needs, and deployment risks, so sharing one environment would create
unnecessary coupling.

Key lessons:

- A direct dependency is a package imported by project code. Transitive
  dependencies belong in the lockfile and should not be declared without a
  direct use.
- Exact direct pins make model experiments easier to reproduce, while `uv.lock`
  records the complete resolved dependency graph.
- A secondary package index can create dependency-confusion risk. Marking the
  official PyTorch CUDA index as explicit restricts it to PyTorch packages.
- Python minor versions are part of the experiment environment. Restricting the
  project to Python 3.13 prevents unreviewed interpreter upgrades.
- Dependency resolution and package installation are separate gates. A lock must
  resolve successfully before a large CUDA environment is installed.

Current status: the 56-package dependency graph is locked and the project-local
environment is installed. Exact import checks passed for all direct packages,
CUDA executed a real tensor operation on the RTX 4070 Laptop GPU, and the ML
suite completed with 19 passing tests and one capability-gated Windows symlink
skip.

### Secure Dataset Acquisition

Dataset archives are untrusted input even when they come from a reputable
publisher. A valid license and HTTPS connection do not prevent corruption,
unexpected archive members, path traversal, or decompression bombs.

Key lessons:

- A publisher checksum proves artifact identity only after the complete file is
  hashed and matched. Incomplete downloads retain a `.part` suffix.
- Resumable downloads and automatic retries are different concerns. Manual curl
  resume preserves operator visibility when a multi-gigabyte transfer fails.
- Archive extraction must follow inspection. Member paths, links, encryption,
  duplicate names, declared sizes, and compression ratios are validated first.
- Nested archives create a second trust boundary. Verifying the outer ZIP does
  not make the inner 7z contents safe to extract automatically.
- Supply-chain isolation includes tools. The existing Windows bsdtar/libarchive
  support avoids introducing an unnecessary archive dependency.

Current status: acquisition metadata, checksum verification, outer-ZIP policy,
safe extraction, CLI commands, documentation, and synthetic security tests have
been added. The complete ML suite passed with 45 tests and one capability-gated
Windows symlink skip. No dataset has been downloaded.

## Phase 5: Frontend

Planned learning goals:

- Build a polished upload and result workflow.
- Understand loading states, error states, and empty states.
- Learn how UI design affects recruiter perception.

## Phase 6: Evaluation and Model Card

Planned learning goals:

- Learn how to present model performance honestly.
- Document known limitations.
- Identify failure cases.

## Phase 7: Production Readiness

Planned learning goals:

- Learn Docker basics.
- Learn environment configuration.
- Learn logging and health checks.
- Create clear run instructions.

## Phase 8: Resume Packaging

Planned learning goals:

- Convert the project into strong resume bullets.
- Prepare a GitHub README that is easy to skim.
- Create a demo script for interviews.
