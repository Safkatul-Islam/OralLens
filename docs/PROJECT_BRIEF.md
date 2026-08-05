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
- reproducible v1-v3 training, full-validation threshold selection, fixed-threshold internal evaluation, and prediction CLIs
- deployment-parity, score-reliability, provenance, patient-level, and failure-case reporting
- strict input, path, annotation, checkpoint, and output validation
- local JSON scan persistence and ignored ML evidence artifacts
- automated ML and backend suites, real integration smoke test, frontend build, and manual browser E2E verification

## Evidence of completion

V3 replaced the capped v2 development experiment with three complete train/validation epochs and durable last/best training-state checkpoints. Its `0.85` score threshold was selected on all 468 validation images and then measured for v3 on the 858-image internal test benchmark. The threshold was not retuned from test results. Because the same test cohort has now been examined across model generations, it is not pristine external evidence. V3 is now active in the local application through the existing typed configuration and startup-script boundary.

| Evidence | Result |
|---|---|
| V3 full validation at threshold `0.85`, IoU `0.5` | precision `0.7629`, recall `0.7929`, F1 `0.7776` |
| V3 internal test benchmark at frozen threshold, IoU `0.5` | precision `0.7671`, recall `0.7923`, F1 `0.7795` |
| V3 deployment cap impact | `0` validation images; `1` test image and `1` false positive truncated |
| V3 score-to-match ECE | validation `0.1940`; internal test `0.1878` |
| V3 patient-level F1 range | validation `0.6769-0.9245`; internal test `0.6824-0.8889` |
| ML test suite | `150 passed, 1 skipped` |
| Backend test suite | `24 passed, 1 skipped` |
| Real backend-to-ML integration | `1 passed` |
| Frontend production build | passed |
| Manual browser E2E | v3 identity, 15 detections/overlays, report, limitations, and raw detector-score wording passed |

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

## Measurable-trust milestone

The v3 trustworthiness reports evaluate the candidate policy at score threshold `0.85`, IoU `0.5`, and maximum 25 detections. No validation image reached the cap. One internal test image exceeded it by one false positive, producing a negligible aggregate change.

The reports retain per-image and per-patient match evidence, structured failure cases, score-to-match reliability bins, and SHA-256 identities for the checkpoint, manifest, and configs. V3 improved internal detection metrics over v2, but score-to-match ECE worsened to `0.1940` on validation and `0.1878` on test. Detector scores remain ranking values, not disease probabilities.

## Next phase: claim-first clinical evidence

The remaining work is not another long run on the same 74 patients. The next phase should prioritize:

1. freeze a narrow research target, intended user, population, acquisition protocol, output, and clinical action
2. appoint dental, statistical, and data-governance owners before clinical collection
3. define a fit-for-purpose reference standard and adjudication workflow
4. collect representative positive, negative, confounding, multi-site, and real acquisition-quality cases
5. reserve separate development, calibration, external test, and prospective cohorts
6. develop v4 without accessing the new locked evaluation boundary
7. report external diagnostic-accuracy evidence transparently before considering a human-AI study

The target is an externally validated, dentist-supervised research system. A diagnostic or medical-device claim remains a later evidence and regulatory question.

## Success criteria for the next milestone

- intended use and claim wording are frozen before data design
- data governance and usage rights are approved before collection
- sample size is justified at the patient level
- positive, negative, hard-negative, site, device, and subgroup coverage are measurable
- clinician reference labels, disagreements, and adjudication are traceable
- no patient, site, encounter, or derived image leaks across protected boundaries
- external evaluation is pre-specified and reported with patient-level uncertainty

## References

- [Architecture](ARCHITECTURE.md)
- [Training and evaluation](TRAINING.md)
- [Dataset card](DATASET_CARD.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [API contract](API.md)
- [Root project guide](../README.md)
