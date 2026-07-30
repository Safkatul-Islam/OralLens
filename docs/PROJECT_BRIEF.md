# OralLens AI Project Brief

## Project statement

OralLens AI is a resume-quality, end-to-end AI/ML engineering project that demonstrates how an oral image can move through a secure upload boundary, an object-detection pipeline, structured API contracts, local persistence, and an evidence-oriented user interface.

The system supports experimentation and learning. It does not diagnose disease, establish clinical risk, recommend treatment, or claim medical-device or clinical-validation status.

## User and need

The primary audience is a reviewer, recruiter, engineer, or ML practitioner evaluating a complete applied-ML workflow. The interface is also designed to make experimental detector output understandable to a non-specialist without presenting it as medical advice.

The project addresses a common portfolio gap: many ML demonstrations stop at a notebook or a metric. OralLens AI connects data contracts, reproducible experiments, inference integration, API security, persistence, frontend visualization, tests, and limitations into one inspectable system.

## Current user story

As a local user, I can upload a JPEG or PNG oral image and receive:

- a screening-support label and detector confidence
- candidate plaque bounding boxes
- a concise evidence summary
- an automatically structured report
- explicit limitations, disclaimer, and appropriate next steps

## Achieved MVP scope

- React, TypeScript, and Vite frontend with preview and box overlays
- FastAPI backend with typed configuration, schemas, services, storage, and error handling
- deterministic mock mode for lightweight development
- project-local TorchVision Faster R-CNN inference adapter
- verified Part 2 orthodontic-plaque dataset with patient-aware splits
- reproducible training, full-validation threshold selection, frozen held-out evaluation, and prediction CLIs
- strict input, path, annotation, checkpoint, and output validation
- local JSON scan persistence and ignored ML evidence artifacts
- automated ML and backend suites, real integration smoke test, frontend build, and manual browser E2E verification

## Evidence of completion

The v2 detector improved training and validation loss over three bounded epochs. Its `0.65` score threshold was selected on all 468 validation images and then evaluated once on 858 held-out test images.

| Evidence | Result |
|---|---|
| Full validation at threshold `0.65`, IoU `0.5` | precision `0.7652`, recall `0.7516`, F1 `0.7583` |
| Held-out test at frozen threshold, IoU `0.5` | precision `0.7582`, recall `0.7037`, F1 `0.7299` |
| ML test suite | `132 passed, 1 skipped` |
| Backend test suite | `23 passed, 1 skipped` |
| Real backend-to-ML integration | `1 passed` |
| Frontend production build | passed |
| Manual browser E2E | passed |

These measurements describe annotation matching on this dataset. They do not measure disease diagnosis, patient outcomes, or clinical safety.

## Design principles

- **Honest claims:** separate detector metrics from clinical meaning.
- **Explicit boundaries:** routes, services, storage, integration, and ML code have distinct responsibilities.
- **Fail closed:** malformed uploads, paths, annotations, boxes, and checkpoints are rejected.
- **Reproducibility:** configs, seeds, commands, and locally generated metrics define each experiment.
- **Test meaningful behavior:** cover validation, security, error translation, integration, and regression paths.
- **MVP discipline:** avoid infrastructure and dependencies that do not serve a present requirement.

## Out of scope for the current claim

- diagnosis or differential diagnosis
- treatment or triage recommendations
- clinical validation or claims of generalization to patients
- handling protected health information
- authentication, accounts, or multi-user storage
- cloud deployment, monitoring, or regulated operations
- mobile capture guidance or image-quality scoring
- additional oral conditions unsupported by verified data

## Next phase: measurable trust

The remaining work is not to soften disclaimers; it is to generate stronger evidence and safer system behavior. The next phase should prioritize:

1. structured error and false-positive/false-negative review
2. calibration analysis and honest interpretation of detector scores
3. robustness evaluation across image quality, framing, rotation, and acquisition variation
4. subgroup or source-stratified analysis where valid metadata exists
5. dataset lineage, exclusion, and boundary-policy reporting
6. model/version provenance and reproducible evaluation summaries
7. privacy, authentication, retention, observability, and deployment threat modeling before any external hosting
8. documented human oversight and clear escalation when images are unsupported or unreliable

This can make OralLens AI more trustworthy as an experimental research prototype. It would still not establish clinical validity without representative studies and appropriate independent review.

## Success criteria for the next milestone

- reliability claims are tied to reproducible artifacts and defined datasets
- known failure modes are categorized and visible
- confidence presentation does not imply calibrated disease probability
- unsupported inputs and low-quality conditions fail or warn explicitly
- model, config, threshold, dataset, and evaluation provenance are traceable
- security and privacy gaps are explicit before deployment decisions
- documentation remains synchronized with executable behavior

## References

- [Architecture](ARCHITECTURE.md)
- [Training and evaluation](TRAINING.md)
- [Dataset card](DATASET_CARD.md)
- [API contract](API.md)
- [Root project guide](../README.md)
