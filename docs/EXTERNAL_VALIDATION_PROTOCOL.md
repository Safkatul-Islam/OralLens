# Draft External Validation Protocol

## Document status

| Field | Value |
|---|---|
| Project | OralLens AI |
| Protocol stage | Stage 1 retrospective external validation |
| Status | Engineering draft; not approved for clinical data collection or study conduct |
| Version | `0.1-draft` |
| Dental clinical lead | Not recorded; required before approval |
| Biostatistician | Not recorded; required before approval |
| Privacy/data-governance owner | Not recorded; required before approval |
| Site principal investigator | Not recorded; required for each participating site |
| Model generation | To be frozen before access to the external evaluation cohort |

This document turns the evidence strategy into a reviewable study-protocol draft. It is not ethics or IRB approval, informed consent, a statistical analysis plan, a data-use agreement, trial registration, or authorization to collect or access patient data.

No study activity may begin until the blocking decisions and approvals in this document and the [Clinical evidence readiness checklist](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md) are complete.

## Protocol objective

The proposed study will estimate how a frozen OralLens model localizes a pre-specified dental-plaque target in independently sourced intraoral photographs collected from the defined population and acquisition setting.

The proposed research statement is:

> OralLens AI is being studied as a tool that highlights regions visually consistent with a pre-specified, reference-standard definition of dental plaque in defined intraoral photographs for review by dental professionals.

This is a research objective, not a current performance, diagnostic, screening, medical-device, safety, or clinical-utility claim. The study does not evaluate autonomous diagnosis, treatment recommendation, consumer self-assessment, or replacement of a dental examination.

## Decisions that must be frozen before approval

| Decision | Required owner | Status |
|---|---|---|
| Exact clinical concept and operational plaque definition | Dental clinical lead | Not recorded |
| Intended user and review workflow | Dental clinical lead and human-factors owner | Not recorded |
| Population, care setting, appliance status, and age range | Dental clinical lead | Not recorded |
| Supported views, devices, and acquisition-quality envelope | Dental clinical lead and imaging/site lead | Not recorded |
| Clinical reference-standard method and timing | Dental clinical lead | Not recorded |
| Primary endpoint and success criterion | Dental clinical lead and biostatistician | Not recorded |
| Analysis unit and clustering method | Biostatistician | Not recorded |
| Patient-level sample size and subgroup plan | Biostatistician | Not recorded |
| Ethics, consent/waiver, privacy, and data-use basis | Site PI and governance/privacy owner | Not recorded |
| Model, preprocessing, threshold, and calibration policy | ML lead | Not frozen for this study |
| External-cohort custodian and access procedure | Independent data owner | Not recorded |

An unfilled decision is a blocker, not permission to choose a convenient default during analysis.

## Study design

### Proposed design

- retrospective, observational, multi-site technical validation
- independently sourced patients not used for OralLens development, calibration, or model selection
- patient-level isolation across all data partitions and model generations
- frozen-model evaluation with no output exposed to clinical care or reference assessors
- representative target-positive, target-negative, indeterminate, difficult, and unusable inputs
- a reference standard produced independently of model predictions

At least one site or a pre-specified temporal cohort must remain inaccessible to the ML development team until the model, threshold, preprocessing, analysis code, and statistical analysis plan are locked.

### Research questions

1. How reliably does the frozen system localize reference-standard plaque regions under the approved acquisition protocol?
2. How often does it return false candidate regions on clinically plaque-negative and hard-negative images?
3. How often are inputs unusable or outside the supported acquisition envelope?
4. How does performance vary by site, device, operator, image quality, appliance status, and other approved subgroups?
5. Are detector scores suitable only for ranking boxes, or does a separately developed outcome-specific calibration support any stronger interpretation?

The fifth question must not be answered using the current detector-score reliability analysis alone. Current scores are not clinical probabilities.

## Study population

### Inclusion criteria template

The clinical and statistical owners must replace the following placeholders with precise criteria:

- participant belongs to the intended-use population: **TBD**
- encounter occurs in the intended care or research setting: **TBD**
- required intraoral view and anatomical region are available
- image was captured using an eligible device and approved procedure
- reference assessment is available within the permitted time interval
- consent, waiver, and permitted-use requirements are satisfied
- the participant is unique within the study identity system

### Exclusion criteria template

- consent, permission, or data-use requirements are not satisfied
- participant identity cannot be safely isolated or duplicate identity cannot be resolved
- required reference-standard data are missing
- capture falls outside criteria that the protocol defines as non-evaluable
- image integrity cannot be verified

Poor-quality, difficult, negative, or model-failing cases must not be excluded solely because they reduce performance. The protocol must distinguish:

- **evaluable:** included in the primary analysis
- **limited but evaluable:** included and flagged for sensitivity analysis
- **indeterminate reference:** handled by the pre-specified indeterminate policy
- **unsupported input:** counted in the unsupported-input endpoint
- **administrative exclusion:** excluded for a documented non-performance reason

## Sampling and cohort construction

The patient is the primary sampling unit. Images, teeth, surfaces, and plaque regions within a patient are correlated observations.

The sampling plan must:

- estimate target-positive and target-negative prevalence in the intended setting
- include consecutive or otherwise defensibly sampled patients where feasible
- prevent enrichment from being mistaken for real-world prevalence
- represent relevant sites, devices, operators, lighting, framing, and image quality
- include hard negatives defined by dental experts
- measure repeated encounters and repeated images without treating them as independent patients
- retain a participant-flow record from available population through final analysis
- document reasons and counts for every exclusion

If an enriched case-control design is necessary, sensitivity and specificity may be estimable under the approved analysis, but PPV and NPV must not be presented as population values without prevalence adjustment and justification.

## Image-acquisition contract

Before data are accepted, the site manual must define:

- required views and anatomical coverage
- device eligibility and versioned capture settings where material
- orientation, distance, focus, illumination, flash, and white-balance guidance
- resolution and compression limits
- handling of mirrors, retractors, saliva, glare, and occlusion
- relationship to brushing, eating, cleaning, treatment, or plaque-disclosing procedures
- maximum interval between photography and reference assessment
- retake, repeated-image, and failed-capture rules
- operator training and quality checks

Synthetic transformations may support model training but are not independent clinical observations and may not appear across partitions derived from the same source image or patient.

## Reference standard

The dental clinical lead must define the target independently of the existing dataset annotations. The approved rubric must specify:

- clinical definition of plaque and non-plaque
- assessment method, including any accepted index or disclosing procedure
- patient-, tooth-, surface-, image-, and region-level label relationships
- spatial representation and coordinate convention for localization
- handling of non-visible anatomy, mixed findings, uncertainty, and poor image quality
- allowed interval between image capture and clinical assessment
- clinically relevant hard negatives and confounders

### Assessment workflow

1. Train at least two qualified dental assessors on the versioned rubric.
2. Assess cases independently and blinded to model output, dataset partition, and other assessors.
3. Preserve every original assessment and its rubric version.
4. Quantify agreement using measures selected for the outcome type.
5. Adjudicate disagreements through a pre-specified assessor or panel.
6. Preserve the adjudication decision, reason, and final reference label.
7. Audit annotation drift and a random quality-control sample during the study.

Model-assisted labeling is prohibited for the primary reference standard unless separately justified, evaluated for bias, and approved in the protocol.

## Endpoints

### Primary endpoint proposal

The working endpoint for clinical/statistical review is patient-clustered lesion-level localization sensitivity at the frozen operating policy, paired with a pre-specified false-positive-region burden. A reference region is detected only under a pre-specified class and spatial-matching rule.

The dental clinical lead and biostatistician must approve or replace this endpoint, define the clinically acceptable target, and specify the confidence-interval criterion before sample-size calculation or locked-cohort access. The current IoU `0.5` engineering convention is not automatically the clinically correct matching rule.

### Candidate secondary endpoints

- localization precision and overlap under approved matching rules
- false-positive regions per image and per patient
- patient-, tooth-, surface-, or image-level sensitivity and specificity where supported by the reference standard
- PPV and NPV with observed prevalence and any justified target-prevalence analysis
- unsupported-input and unusable-image rates
- performance by site, device, operator, quality category, appliance status, and pre-specified subgroups
- inter-assessor agreement and adjudication frequency
- model failure, timeout, and invalid-output rate
- calibration only for an explicitly defined outcome and separately reserved calibration data

All endpoint names must state their unit. Annotation-match precision and recall must not be relabeled as clinical sensitivity and specificity.

## Sample-size and statistical analysis

The biostatistician must issue a separate, versioned statistical analysis plan before external-cohort access. It must define:

- one primary endpoint and hypothesis or precision objective
- patient-level sample-size calculation
- expected prevalence, event counts, and unusable-data allowance
- handling of clustering within patient, site, tooth, surface, and image
- confidence-interval method and confidence level
- multiplicity and subgroup interpretation
- missing, indeterminate, unsupported, and excluded input handling
- sensitivity analyses for reference uncertainty and quality restrictions
- prevalence treatment for predictive values
- model-score calibration analysis, if applicable
- software and analysis-code versioning

Point estimates without patient-level uncertainty are insufficient. Annotation counts must not be used as the effective sample size.

## Model, threshold, and software freeze

Before anyone authorized for model development can access external evaluation labels or predictions, record:

- model-generation identifier and checkpoint SHA-256
- source-data and development-split versions
- model architecture and preprocessing configuration
- supported input contract and quality-gate version
- operating threshold and maximum displayed detections
- calibration mapping and cohort, if applicable
- inference and evaluation code version
- environment and dependency versions
- complete analysis configuration and expected report schema

The external custodian should verify the package identity and run the registered analysis once. Any post-access model, threshold, matching, exclusion, or endpoint change creates a new exploratory analysis and cannot replace the registered primary result. Evaluating a revised model as fresh external evidence requires a new independent cohort.

## Data separation, privacy, and security

- direct identifiers and re-identification keys remain in controlled clinical systems
- the ML workspace receives pseudonymous study identifiers only
- role-based access follows least privilege
- transfers, storage, backups, and deletion follow the approved security plan
- access to labels, partitions, model outputs, and audit logs is recorded
- repository, ordinary scan storage, and ML run folders must not contain protected health information
- site-specific minimum-necessary metadata are defined before export
- dates, locations, and rare attributes are generalized or retained only where justified
- withdrawals, corrections, incidents, and retention expiry have traceable procedures

The current local demonstration store is not approved for clinical study records.

## Bias, robustness, and failure analysis

The analysis must report both aggregate and site-stratified evidence and examine approved subgroups with uncertainty. Pre-specified robustness dimensions should include:

- device and operator
- focus, darkness, overexposure, glare, occlusion, framing, and compression
- appliance and hardware status
- relevant non-target findings and hard negatives
- repeated views and incomplete anatomical coverage
- unsupported and unusable inputs

Small subgroups must be reported as underpowered rather than interpreted from unstable point estimates. Failure examples must be de-identified and handled under the approved data-use terms.

## Study conduct and deviations

The study log must retain:

- participant and image flow
- protocol deviations and their impact classification
- missing or corrected data
- reference-standard disagreements and adjudication
- locked-cohort access events
- software, model, and configuration identity
- inference and analysis failures
- planned and unplanned analyses

Unexpected results, negative results, and site-specific failures must be preserved. A failed primary endpoint must not be converted into success by post-hoc threshold or cohort changes.

## Reporting

The final report should include:

- protocol and statistical analysis plan versions
- ethics, consent/waiver, privacy, and data-use status
- intended-use definition and participant flow
- site, device, population, and acquisition characteristics
- reference-standard method and assessor agreement
- model, configuration, data, and analysis provenance
- primary and secondary results with confidence intervals
- exclusions, missing data, deviations, sensitivity analyses, and subgroup limits
- robustness, failure, and safety findings
- exact claim language supported and unsupported by the evidence

Reporting should follow STARD-AI where applicable. This retrospective study cannot establish human-AI benefit, prospective workflow safety, or clinical outcomes.

## Approval and start gates

The protocol is ready for data activity only when all of the following are documented:

1. intended use and operational target definition approved
2. accountable clinical, statistical, privacy, site, and ML owners assigned
3. final protocol and statistical analysis plan approved
4. ethics/IRB determination, consent/waiver, privacy, and data-use requirements complete
5. acquisition manual and assessor rubric versioned and trained
6. data dictionary, validation rules, access controls, and audit procedure accepted
7. sample size and site contribution targets approved
8. external cohort custodian and partition-lock procedure established
9. model package and analysis plan frozen before external-label access
10. deviations, incident response, reporting, and publication policy approved

Current readiness: **not ready for clinical data collection or external clinical evaluation**. The repository contains an engineering draft and internal evidence only.

## Related documents

- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Clinical data dictionary](CLINICAL_DATA_DICTIONARY.md)
- [Clinical evidence readiness checklist](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md)
- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
