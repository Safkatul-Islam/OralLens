# Data Acquisition and Annotation Protocol

## Purpose and status

This document is the engineering and evidence-planning contract for future claim-aligned oral-image data. It is not ethics approval, informed consent, a data-use agreement, permission to collect patient data, or a finalized clinical study protocol.

No new clinical data should be collected until the intended use, clinical reference standard, statistical plan, privacy controls, and responsible owners are approved.

## Design principle

The next dataset must be fit for the intended claim, not merely large.

The current Part 2 dataset is valuable for model development but contains only 74 patient groups, comes from one specialized orthodontic-plaque source, includes transformed image variants, and requires at least one positive annotation per manifest row. It cannot independently establish clinical specificity, negative predictive value, cross-site generalization, or human-use safety.

## Governance prerequisites

Before acquisition begins, record and approve:

- intended use, user, population, environment, output, and clinical action
- dental clinical lead and site principal investigator where applicable
- study protocol and statistical analysis plan
- ethics/IRB determination and approval where required
- consent process or documented waiver
- privacy impact and de-identification procedure
- data-use, licensing, publication, and model-training rights
- access roles, encryption, transfer, retention, backup, and deletion policy
- incident, withdrawal, correction, and audit procedure
- rules for cross-border or multi-jurisdiction data where applicable

Public availability does not by itself establish permission for clinical use, redistribution, or model training.

## Unit of collection

The primary sampling unit is the unique patient, not the image, box, tooth, or augmented variant.

Recommended hierarchy:

```text
site
  patient
    encounter
      acquisition session
        image
          tooth or surface
            reference label and localization
```

Every identifier exposed to the ML workspace must be pseudonymous. Direct identifiers and re-identification keys belong in separately controlled clinical systems, never the repository or ordinary ML run artifacts.

## Population and sampling matrix

The final protocol must define inclusion and exclusion criteria appropriate to the intended claim. Sampling should deliberately cover:

- plaque-positive and clinically plaque-negative cases
- clinically relevant plaque severity and distribution
- fixed appliances, other appliances, and no appliance if included in the claim
- age ranges and demographic characteristics needed for the intended population
- relevant oral conditions and common confounders
- multiple clinical sites, operators, and acquisition workflows
- multiple supported camera/phone models and image resolutions
- expected lighting, framing, focus, glare, saliva, and occlusion variation
- geographic and temporal variation proportionate to the claim

Counts should be monitored by unique patients and encounters as well as images. Thousands of derived images from a small patient cohort do not provide thousands of independent clinical observations.

## Negative and hard-negative cases

A screening or probability claim requires representative negative evidence. The future dataset should include plaque-negative images and hard negatives that can visually compete with plaque candidates, such as:

- calculus or tartar
- stains and pigmentation
- saliva, bubbles, glare, and specular highlights
- restorative materials and dental hardware
- braces, wires, brackets, retainers, and attachments
- food debris or transient material where clinically relevant
- gingival inflammation and other non-target oral findings
- caries or lesions not belonging to the target definition
- blur, darkness, overexposure, compression, and poor framing
- normal anatomical structures

The dental clinical lead must confirm the relevant confounder list and how each category is labeled. Hard negatives must not be silently discarded because they reduce model metrics.

## Image acquisition protocol

The study protocol should define:

- required intraoral views and anatomical coverage
- camera/device eligibility and metadata
- orientation and distance guidance
- focus, illumination, flash, and white-balance expectations
- minimum usable resolution and compression limits
- whether mirrors, retractors, or plaque-disclosing procedures are used
- whether cleaning, eating, brushing, or treatment timing affects the target
- acceptable delay between image capture and the clinical reference assessment
- retake policy and handling of repeated images
- infection-control and operator-safety procedures owned by the clinical site

Real acquisition variation belongs in development and evaluation data. Synthetic transforms may augment training but must not substitute for external clinical capture in validation or test evidence.

## Minimum metadata

Collect only metadata justified by the intended use and analysis plan.

| Category | Example fields |
|---|---|
| Provenance | pseudonymous site, patient, encounter, session, and image identifiers |
| Acquisition | device family, view, resolution, operator role, lighting/flash, retake status |
| Population | only approved demographic and clinical variables needed for representativeness or subgroup analysis |
| Oral context | appliance status, relevant confounders, tooth/surface identifiers where applicable |
| Reference standard | assessor identifiers, assessment method, timing, labels, adjudication status |
| Quality | focus, exposure, framing, occlusion, glare, unsupported-input reason |
| Governance | consent/waiver category, data-use version, allowed purposes, retention class |

Exact dates, locations, and rare combinations may create re-identification risk. Store or transform them according to the approved privacy plan rather than copying them into model manifests by default.

## Reference-standard design

The reference standard must be selected for the intended claim with dental expertise.

Before labeling, define:

- the operational definition of plaque and non-plaque
- whether the standard is visual examination, an accepted index, a disclosing procedure, another clinical assessment, or a justified combination
- patient-, tooth-, surface-, lesion-, and image-level label relationships
- how uncertainty, non-visibility, mixed findings, and poor quality are recorded
- how reference limitations may bias the study
- the time relationship between image capture and examination

Bounding boxes alone are not sufficient for a patient-level screening claim. If localization remains part of the task, retain boxes or another justified region representation while also recording the clinically relevant patient/tooth/surface outcome and true negative status.

## Annotation workflow

Recommended workflow:

1. train and calibrate at least two dental assessors on the written rubric
2. label cases independently and blinded to model output
3. preserve each assessor's original labels
4. quantify inter-rater agreement using a measure appropriate to the label type
5. route disagreements and uncertain cases to a pre-specified adjudicator or panel
6. record adjudication reason and final reference label
7. perform periodic drift and quality checks during annotation
8. prevent model predictions from becoming the reference standard

Annotators should be blinded to dataset partition and model score. If model-assisted annotation is ever used for efficiency, it must be studied and documented separately because it can bias the reference standard.

## Image and annotation quality control

Automated checks should cover:

- decodability, format, dimensions, and color-channel contract
- duplicate and near-duplicate detection
- patient/encounter/site identifier integrity
- path containment and symlink policy
- finite, valid coordinates and positive region extent
- label vocabulary and tooth/surface consistency
- required metadata and reference-standard completion
- cross-partition patient, encounter, source, and derived-image leakage

Clinical review should cover:

- correct view and anatomical coverage
- target visibility and image usability
- plausible labels and region placement
- uncertainty and confounder handling
- consistency across repeated or related images

Exclusions must be recorded with structured reasons. Do not silently delete difficult, negative, or low-quality cases.

## Dataset partitions

Partition before model development using the patient as the minimum isolation unit.

Required protections:

- no patient or encounter crosses partitions
- no crop, transform, burst frame, or near-duplicate crosses partitions
- site and acquisition dependence are considered explicitly
- calibration data remain separate from model/threshold selection
- at least one external site or temporal cohort remains locked
- prospective data remain separate from retrospective development

Preferred partition roles:

| Partition | Role |
|---|---|
| Development train | model fitting and training-only augmentation |
| Development validation | early stopping and controlled model selection |
| Calibration | score/probability calibration after model freeze |
| External locked test | one claim-aligned technical evaluation |
| Prospective cohort | real acquisition, workflow, and human-use evidence |

## Sample-size planning

Do not select sample size from a convenient image count.

The biostatistical calculation must consider:

- primary clinical endpoint and unit of analysis
- target sensitivity/specificity or other clinically justified goals
- desired confidence-interval precision
- target-population prevalence
- clustering of images, teeth, surfaces, and lesions within patients
- site and operator effects
- expected unusable or indeterminate inputs
- planned subgroup estimates and multiplicity
- prospective study design and reader effects where applicable

The current 7-patient validation and 12-patient test groups are valuable engineering checks but are not an adequate basis for broad clinical generalization.

## Manifest and provenance contract

Every prepared release should have:

- immutable dataset and schema version
- source/site and acquisition provenance
- patient-aware partition assignment
- reference-standard and annotation-rubric version
- structured inclusion/exclusion reason
- checksums or equivalent integrity controls
- creation tool and environment version
- documented license and permitted uses
- audit summary covering duplicates, leakage, paths, labels, and quality

Source clinical data must remain immutable. Corrections should produce a new version with a traceable reason rather than rewriting history.

## Release gates

A dataset release may enter model development only when:

1. governance approvals and usage rights are documented
2. intended population and acquisition protocol are defined
3. reference-standard and annotation procedures are complete
4. required positive, negative, confounding, site, and subgroup coverage is measured
5. patient/site/derived-image leakage checks pass
6. exclusions and missing data are reported
7. quality-control failures are resolved or retained with explicit unsupported status
8. the clinical lead, statistician, data owner, and ML owner approve the release for their respective responsibilities

Release for model development does not authorize clinical deployment.

## Prohibited practices

- collecting patient images without the required governance and consent/waiver pathway
- storing direct identifiers or protected health information in the repository
- mixing public datasets without checking license, population, label definition, and acquisition compatibility
- placing transformed versions of one patient across data partitions
- tuning on calibration, external-test, or prospective cohorts beyond their declared role
- silently excluding negative, difficult, minority, or low-quality cases
- treating synthetic augmentation as independent clinical evidence
- describing detector scores as probabilities without outcome-specific calibration
- declaring success from annotation count while ignoring patient count and clustering

## Current-to-future gap summary

| Requirement | Current Part 2 evidence | Future requirement |
|---|---|---|
| Unique patients | 74 total | Patient-level sample-size justification |
| Sources | One specialized dataset | Independent sites and/or temporal cohorts |
| Negatives | Manifest rows require positive annotations | Representative negative and hard-negative cases |
| Reference standard | Source annotations | Clinically justified, multi-assessor standard and adjudication |
| Acquisition | Source-specific plus transforms | Real intended-use devices, operators, and conditions |
| Calibration | Box score-to-match analysis | Separate outcome-specific calibration cohort |
| Test boundary | Patient-aware internal benchmark | New locked external cohort |
| Clinical utility | Not measured | Prospective human-AI evaluation |

## Operationalization boundary

This document defines acquisition and annotation policy. The [draft external validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md) translates that policy into a proposed Stage 1 study design, the [clinical data dictionary](CLINICAL_DATA_DICTIONARY.md) defines the future pseudonymous entity and validation contract, and the [clinical evidence readiness checklist](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md) controls progression between evidence stages.

These documents remain engineering drafts. A template, unchecked box, or repository review does not replace approval by the dental clinical lead, biostatistician, site PI, privacy/data-governance owner, ethics authority, or other accountable reviewer.

## Related documents

- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Draft external validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md)
- [Clinical data dictionary](CLINICAL_DATA_DICTIONARY.md)
- [Clinical evidence readiness checklist](CLINICAL_EVIDENCE_READINESS_CHECKLIST.md)
- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
