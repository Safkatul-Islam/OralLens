# Clinical Evidence Plan

## Purpose and status

This document defines the staged evidence required to move OralLens AI from an internal object-detection demonstration toward a credible dental-professional research tool.

It is a planning document. It does not authorize a clinical investigation, assert regulatory compliance, replace a statistical analysis plan, or claim that OralLens is clinically validated.

## Current evidence baseline

The strongest current model candidate is v3, trained on the complete patient-aware training and validation splits of the verified Part 2 orthodontic-plaque dataset.

At the validation-selected threshold `0.85` and IoU `0.5`:

| Evidence | Validation | Internal test benchmark |
|---|---:|---:|
| Images | 468 | 858 |
| Patients | 7 | 12 |
| Precision | 0.7629 | 0.7671 |
| Recall | 0.7929 | 0.7923 |
| F1 | 0.7776 | 0.7795 |
| Mean matched IoU | 0.8034 | 0.8102 |
| Patient-level F1 range | 0.6769-0.9245 | 0.6824-0.8889 |
| Score-to-match ECE | 0.1940 | 0.1878 |

The test threshold was not retuned. However, this cohort has now been examined across model generations, so it is an internal locked benchmark rather than pristine external evidence.

The present evidence does not include a representative negative population, independent clinical site, prospective capture, dentist-plus-AI evaluation, clinical outcomes, or medical-device quality system.

## Evidence questions

The program must answer three different questions without treating them as interchangeable.

### 1. Valid clinical association

Is the visual target detected by the model meaningfully associated with the exact clinical concept named in the intended use?

This requires dental consensus on the plaque definition, capture conditions, examination procedure, labeling unit, and limitations. Dataset boxes alone do not establish the association.

### 2. Technical and analytical validation

Does the frozen system accurately and reliably produce the specified output from the specified input under clinically relevant conditions?

This includes localization, patient/tooth/surface classification where applicable, calibration, quality failures, subgroup performance, robustness, reproducibility, and software behavior.

### 3. Clinical validation and utility

Does the system achieve the intended purpose when used by the intended user in the intended workflow, with acceptable benefits and risks?

This requires human-AI and prospective evidence. Offline F1 cannot answer it.

## Stage 0: internal engineering evidence

Status: achieved for the portfolio MVP.

Evidence includes:

- secure upload and inference boundaries
- patient-aware internal splits
- reproducible v1, v2, and v3 experiments
- validation-only threshold selection
- internal test-benchmark measurement
- per-patient and failure-case evidence
- score-to-match reliability and deployment-cap parity
- automated ML/backend tests and local end-to-end verification

Permitted conclusion: the engineering and internal evaluation pipeline works for its documented dataset-specific task.

## Stage 1: claim-aligned retrospective external validation

Objective: establish credible technical performance for the narrow research target in data not used for model development.

### Prerequisites

- intended use and population frozen
- dental clinical lead and biostatistician assigned
- ethics, consent/waiver, privacy, and data-use approvals complete
- reference-standard procedure finalized
- primary and secondary endpoints pre-specified
- sample size justified at the patient level
- model, preprocessing, threshold, and calibration policy frozen before external-test access

### Cohort design

- multiple acquisition sites where feasible
- at least one site or temporal cohort isolated from development
- unique patients with no cross-partition images or derived variants
- positive, negative, and hard-negative cases
- capture-device, operator, lighting, framing, and quality variation representative of intended use
- clinically relevant population and subgroup representation

### Required measurements

The exact primary endpoint must be chosen with clinical and statistical owners. Candidate measurements include:

- patient-, tooth-, or surface-level sensitivity and specificity
- PPV and NPV at the observed and target prevalence
- lesion-level precision, recall, F1, and localization overlap
- precision-recall curves and performance at clinically chosen false-positive or false-negative constraints
- calibration and uncertainty for any output represented as probability
- invalid/unsupported input rate and quality-gate performance
- performance with 95% confidence intervals using the patient as the independent unit
- subgroup and site estimates with uncertainty
- pre-specified failure and safety analyses

Image boxes and lesions are clustered within patients. Analyses must account for that clustering rather than treating every annotation as an independent clinical subject.

### Stage 1 exit gate

Advance only if:

- external performance meets pre-specified clinically justified goals
- confidence intervals are sufficiently precise for the target claim
- no critical site or subgroup failure is hidden by aggregate performance
- exclusions and missing/invalid inputs are transparent
- calibration and score wording are appropriate
- an independent clinical/statistical review accepts the conclusion

## Stage 2: prospective silent-mode validation

Objective: measure real acquisition and workflow behavior without exposing model output to clinical decision-makers.

The frozen model runs prospectively in the intended environment while normal care continues unchanged. The study should measure:

- real input quality and unsupported-image frequency
- device, site, operator, and workflow variation
- prospective technical performance against the reference standard
- latency, failure recovery, logging, and availability
- data drift relative to development cohorts
- privacy, security, and operational incidents

Silent validation reduces the risk of model output influencing the reference standard or patient care while revealing implementation failures absent from retrospective data.

### Stage 2 exit gate

- prospectively captured performance remains within pre-specified bounds
- operational and quality failures are understood and controlled
- drift and subgroup monitoring are feasible
- the system is safe enough to study with users under an approved protocol

## Stage 3: human-AI clinical evaluation

Objective: determine whether OralLens improves the defined task when used by dental professionals.

A suitable early study may compare:

- dental professional alone
- dental professional with OralLens assistance

The design should address reader order, washout, case order, learning effects, clinician experience, blinding, and clustered cases. Outcomes should include:

- task accuracy and clinically important misses
- unnecessary alerts or follow-up
- time and workflow burden
- agreement and consistency
- automation bias and overreliance
- user understanding of scores, limitations, and unsupported inputs
- error recovery and appropriate override behavior

The study must evaluate the human-AI team rather than assume better standalone F1 produces better care.

### Stage 3 exit gate

- pre-specified human-AI benefit is demonstrated without unacceptable harm
- users understand output limitations
- foreseeable misuse and overreliance are mitigated
- workflow and usability findings support the exact assistance claim

## Stage 4: clinical outcome and regulated-use evidence

Objective: support any medical, screening, or commercial claim proportionate to its risk.

Potential requirements include:

- formal regulatory classification and strategy
- quality management and design controls
- risk management and human-factors engineering
- cybersecurity and privacy controls
- validated deployment, monitoring, incident, and change-control processes
- clinical investigation or controlled trial appropriate to the intended use
- post-deployment performance and drift monitoring

Regulatory requirements depend on jurisdiction and claim. Scientific reporting guidelines do not replace regulatory review.

## Dataset partition and access policy

Future data should be partitioned by patient before modeling and, where possible, by site or time:

| Partition | Purpose | Permitted decisions |
|---|---|---|
| Development train | Parameter optimization | Model fitting and augmentation |
| Development validation | Architecture and training policy | Early stopping and controlled model selection |
| Calibration | Frozen-output calibration | Calibration mapping only |
| External locked test | Claim-aligned technical evidence | Measurement only; no tuning |
| Prospective cohort | Real-world and workflow evidence | Measurement under the registered protocol |

Access to locked cohorts should be logged and limited. A failed result is not permission to revise the model and reread the same test as if it were fresh evidence.

## Statistical analysis principles

- define one primary analysis and endpoint before locked-data access
- use the patient as the primary independent sampling unit
- account for multiple images, teeth, surfaces, and lesions within patients
- calculate sample size from desired precision, prevalence, subgroup goals, and clinical performance targets
- report confidence intervals, not only point estimates
- state how indeterminate, missing, corrupted, and unsupported inputs are handled
- pre-specify exclusions and sensitivity analyses
- distinguish per-image, per-lesion, per-tooth, per-surface, and per-patient metrics
- report subgroup results only where metadata quality and sample size support interpretation
- separate model-score calibration from disease-probability calibration
- preserve all negative and unexpected results

No universal patient count is declared here. A biostatistician must calculate it after the target outcome, unit of analysis, performance goal, confidence-interval width, prevalence, and clustering assumptions are fixed.

## Model and change control

Each evidence-producing model must have:

- immutable model and config identifiers
- checkpoint, manifest, and config hashes
- source-data and split versions
- training seed and environment versions
- pre-specified operating threshold
- calibration version where applicable
- a documented reason for every change
- a statement identifying which prior evidence remains applicable

Retraining after locked-test review creates a new model generation. Strong evidence for that generation requires a new independent evaluation boundary.

## Reporting package

Each external or prospective study should retain:

- protocol and statistical analysis plan
- ethics and data-governance approvals
- participant flow and exclusions
- site/device/population characterization
- reference-standard methods and agreement
- model/data/config provenance
- aggregate, patient-level, subgroup, robustness, and calibration results
- confidence intervals and sensitivity analyses
- failure cases, harms, deviations, and missing data
- claim wording reviewed against the evidence

Diagnostic-accuracy reporting should follow STARD-AI where applicable. Early live clinical decision-support studies should use DECIDE-AI. Later clinical trials should use the appropriate SPIRIT/CONSORT extensions and regulatory requirements.

## Immediate execution sequence

1. confirm the narrow research target in [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
2. appoint a dental clinical lead and biostatistician
3. review and approve the [draft external validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md)
4. approve the [clinical data dictionary](CLINICAL_DATA_DICTIONARY.md) and its minimum-necessary fields
5. define the reference standard, endpoints, performance goals, and patient-level sample size
6. secure ethics, privacy, consent/waiver, site, and data-use approval
7. satisfy the applicable [clinical evidence readiness gates](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md)
8. collect and quality-check representative positive, negative, indeterminate, and confounding cases
9. freeze patient/site partitions before v4 model development
10. train and calibrate without accessing the external locked cohort
11. perform one registered external evaluation and preserve the result regardless of outcome
12. decide whether prospective silent and human-AI studies are justified

## Authoritative frameworks

- [IMDRF Good machine learning practice guiding principles](https://www.imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf)
- [IMDRF Software as a Medical Device: Clinical Evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation)
- [STARD-AI diagnostic-accuracy reporting guideline](https://www.nature.com/articles/s41591-025-03953-8)
- [DECIDE-AI early clinical evaluation guideline](https://www.nature.com/articles/s41591-022-01772-9)
- [WHO ethics and governance of AI for health](https://www.who.int/publications/i/item/9789240037403)

## Related documents

- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Draft external validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md)
- [Clinical data dictionary](CLINICAL_DATA_DICTIONARY.md)
- [Clinical evidence readiness checklist](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md)
- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
