# OralLens AI Project Brief

## Project statement

OralLens AI is a resume-quality, end-to-end AI/ML engineering project that demonstrates how an oral image can move through a secure upload boundary, an object-detection pipeline, structured API contracts, local persistence, and an evidence-oriented user interface.

The system supports experimentation and learning. It does not diagnose disease, establish clinical risk, recommend treatment, or claim medical-device or clinical-validation status.

## User and need

The primary audience is a reviewer, recruiter, engineer, or ML practitioner evaluating a complete applied-ML workflow. The interface is also designed to make experimental detector output understandable to a non-specialist without presenting it as medical advice.

The project addresses a common portfolio gap: many ML demonstrations stop at a notebook or a metric. OralLens AI connects data contracts, reproducible experiments, inference integration, API security, persistence, frontend visualization, tests, and limitations into one inspectable system.

## Current user story

As a local user, I can upload a JPEG or PNG oral image and receive:

- a screening-support label and explicitly non-probabilistic detector score
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
- reproducible historical v1-v3 training, full-validation threshold selection, fixed-threshold internal evaluation, and prediction CLIs
- deployment-parity, score-reliability, provenance, patient-level, and failure-case reporting
- strict input, path, annotation, checkpoint, and output validation
- local JSON scan persistence and ignored ML evidence artifacts
- automated ML and backend suites, real integration smoke test, frontend build, and manual browser E2E verification

## Engineering evidence and model-status correction

V3 replaced the capped v2 development experiment with three complete train/validation epochs and durable last/best training-state checkpoints. It is active in the local application through the existing typed configuration and startup-script boundary.

| Evidence | Result |
|---|---|
| End-to-end application | real v3 checkpoint returned HTTP `201`, typed report content, and rendered overlays in the browser |
| Source-label audit | 1,170 plaque-absent and 64,068 plaque-present region annotations |
| Corrected target conversion | validates all boxes/labels, filters to plaque-present targets, and supports empty positive targets |
| Historical model evidence | all v1-v3 model-quality metrics are contaminated by the old source-label conversion |
| Corrected v4 diagnostic | stopped after two completed epochs when training loss fell from `0.7967` to `0.6149` while validation loss worsened from `0.8250` to `1.0287`; epoch 1 remained best |
| Consumer-style challenge | severe failures observed, including images with surrounding lips/facial skin |
| ML test suite | `174 passed, 1 skipped` |
| Backend test suite | `24 passed, 1 skipped` |
| Real backend-to-ML integration | `1 passed` |
| Frontend production build | passed |
| Manual browser E2E | v3 identity, 15 detections/overlays, report, limitations, and raw detector-score wording passed |

The engineering workflow is complete and testable, but current model quality is not acceptable. Historical conversion assigned detector label `1` to source label `0` regions even though the source defines `0` as plaque absent. The corrected v4 run produced isolated diagnostic checkpoints, but its validation loss reversed on the same narrow source distribution. It was stopped before epoch 3 and is not authorized for resumption, evaluation, integration, promotion, or a model-quality claim. V3 remains historical application evidence rather than a credible plaque-only model.

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

## Historical measurable-trust milestone

The v3 trustworthiness reports evaluate the candidate policy at score threshold `0.85`, IoU `0.5`, and maximum 25 detections. No validation image reached the cap. One internal test image exceeded it by one false positive, producing a negligible aggregate change.

The reports retain per-image and per-patient match evidence, structured failure cases, score-to-match reliability bins, and SHA-256 identities for the checkpoint, manifest, and configs. Their matching and calibration numbers inherit the historical target defect and are not current plaque-only evidence. Detector scores remain ranking values, not disease probabilities.

## Next phase: credible portfolio screening support

The remaining work is not another long run on v3 or v4 and is not a clinical-study program. The next phase should prioritize:

1. freeze separate visible-plaque and supragingival-calculus task contracts
2. audit ODS/Oralformer as the first calculus feasibility candidate and admit no data until provenance, rights, patient identity, acquisition context, and mask semantics pass
3. continue access review for the strongest ordinary-RGB plaque sources
4. add oral ROI, image-quality, OOD, and abstention behavior for consumer-style inputs
5. create patient- and source-aware development boundaries and a locked challenge set
6. require CUDA and runtime telemetry, then run only a monitored smoke test and small predeclared pilot
7. report aggregate, per-source, per-patient, and failure-mode evidence before any promotion decision

The target is a more believable experimental screening-support portfolio system for documented oral photographs. Diagnostic, medical-device, autonomous rule-out, and treatment claims remain out of scope.

## Success criteria for the next milestone

- dataset role, provenance, license, label semantics, and patient identity pass the admission gate
- no patient, source, or derived image leaks across protected boundaries
- unsupported or low-quality inputs produce abstention rather than a misleading no-box result
- stopped v4 artifacts remain isolated and are not resumed or used as model-quality evidence
- any future model uses newly admitted condition-compatible data and a fresh experiment identity
- CUDA execution, model/tensor placement, throughput, timing, and peak memory are recorded before a long run
- operating policy is selected on validation only
- a rights-safe challenge boundary remains locked until final evaluation
- remaining failure modes and scope limits are reported plainly

## References

- [Architecture](ARCHITECTURE.md)
- [Training and evaluation](TRAINING.md)
- [Dataset card](DATASET_CARD.md)
- [Data strategy](DATA_STRATEGY.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [API contract](API.md)
- [Root project guide](../README.md)
