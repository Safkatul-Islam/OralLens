# Intended Use and Claims

## Status

This document defines the evidence boundary and the proposed next research target for OralLens AI. It is not a regulatory intended-use statement, an authorized indication, a clinical protocol approval, or evidence that a medical device has been developed.

The working product remains an experimental portfolio application. Its current output must not be used to diagnose disease, rule out disease, recommend treatment, or replace examination by a licensed dental professional.

## Current demonstrated use

OralLens AI currently demonstrates an end-to-end engineering workflow:

1. a local user uploads a JPEG or PNG oral image
2. the backend validates the upload and invokes a plaque-candidate detector
3. the detector returns bounding boxes and detector scores
4. the interface overlays candidates and presents limitations and next steps

The active local application model is v3. A later source audit established that historical v1-v3 target conversion incorrectly treated 1,170 plaque-absent annotated regions as plaque objects. Its previously reported F1, IoU, and calibration values are therefore not valid plaque-only evidence. Manual consumer-style challenge images also demonstrated severe acquisition-domain failures. A corrected v4 run on the same source distribution was stopped after two completed epochs because validation loss worsened while training loss fell. V4 has not been evaluated or promoted and must not be resumed. V3 remains an end-to-end engineering artifact, not a credible condition model.

## Proposed next research target

The narrowest defensible portfolio target for a future admitted model is:

> OralLens AI is an experimental tool that highlights candidate regions for one explicitly named visible-deposit condition in supported, properly framed oral photographs and abstains when image quality or scope is unsupported.

This is a research target, not a current product claim. It deliberately limits:

- each model task to one defined condition and annotation contract
- the input to a documented oral-photograph quality and framing envelope
- the user to a portfolio reviewer or informed local user who sees explicit limitations
- the output to reviewable candidate regions and non-probabilistic detector scores
- the role of the system to experimental screening support rather than autonomous decision-making

This target does not require a clinical data-collection program. It does require verified public/research data rights, explicit label semantics, patient/source isolation, a documented input envelope, and honest challenge testing. Any later attempt to make a clinical, diagnostic, or medical-device claim would be a separate project phase requiring appropriate clinical, statistical, governance, and regulatory expertise.

## Condition and output contract

Visible plaque and supragingival calculus/tartar are related oral deposits, but they are not interchangeable labels. Data for one condition must not be silently merged into the other condition's positive class.

| Condition | Minimum compatible supervision for localization | Current decision |
|---|---|---|
| Visible plaque | Spatial plaque annotation on ordinary, unstained RGB oral photographs, with a documented plaque definition and source/patient identity | Keep as a separate future task. AIRC peri-tooth plaque-status regions may support restricted auxiliary research, but they are not precise plaque-deposit outlines and do not establish consumer-image performance. |
| Supragingival calculus/tartar | Spatial calculus annotation, preferably masks or tightly defined lesion regions, on ordinary RGB oral photographs with provenance and patient/source grouping | First data-feasibility candidate. ODS/Oralformer must pass a full admission audit before download, training, or any claim. |

A classification label, peri-tooth status region, disclosed-plaque image, stained-plaque mask, tooth mask, or gingival-condition label is not automatically compatible with either localization contract. Annotation modality and acquisition appearance are part of the task definition, not implementation details.

## Intended-use elements that must be fixed before a study

| Element | Required decision |
|---|---|
| Clinical purpose | Exact plaque-localization or screening-support question the study will evaluate |
| Intended user | Portfolio reviewer or informed local demonstration user |
| Population | Dataset scope, appliance status, age range where known, and explicit non-generalization limits |
| Input | Accepted views, capture devices, lighting, framing, resolution, and image-quality limits |
| Reference standard | Verified source plaque definition and annotation procedure |
| Output | Candidate location, score interpretation, unsupported-input warning, and failure behavior |
| User action | Review an experimental visualization; seek professional care for actual health concerns |
| Benefits and harms | Intended benefit, missed candidates, false alerts, automation bias, delay, and misuse |
| Environment | Local portfolio demonstration on non-sensitive images |
| Autonomy | No autonomous diagnosis, rule-out, triage, or treatment action |

Changing any of these elements can change the required dataset, performance target, risk assessment, and regulatory status. Evidence must not be carried across a materially changed intended use without justification.

## Claims ladder

### Level 0: current engineering claim

Supported now:

> OralLens AI is an experimental end-to-end oral-image candidate-localization application with tested software boundaries, historical plaque-detector artifacts, and documented model-data limitations.

Required language:

- experimental screening support
- historical dataset-specific results are not current plaque-only evidence
- not a diagnosis, medical device, or treatment system
- detector scores are not probabilities of plaque or disease

### Level 1: credible portfolio screening-support claim

Potentially supportable only after a condition-specific model and a locked, rights-safe challenge evaluation:

> On the documented datasets and supported image envelope, the frozen OralLens model localized annotations for the named visible-deposit condition with the reported source-stratified performance and abstained on the tested unsupported inputs.

Prerequisites include verified condition and annotation semantics, compatible target-domain sources, patient/source isolation, an input-quality and OOD gate, validation-only policy selection, a locked challenge boundary, patient-level uncertainty where possible, and transparent failure reporting. This remains a portfolio research claim, not clinical validation.

### Level 2: clinical or dental-professional assistance claim

Potentially supportable only after live or prospective human-use evidence:

> When used under the study protocol, OralLens assisted dental professionals with the defined plaque-assessment task.

This requires evaluating the human-AI team, including clinician-alone versus clinician-plus-AI performance, usability, overreliance, error recovery, workflow effects, and safety.

Level 2 is outside the current project roadmap. Clinical collaborators and study infrastructure become necessary only if this claim is deliberately pursued later.

### Level 3: diagnostic, screening, or medical-device claim

Not currently supportable. This level would require a formal intended use, regulatory assessment, quality and risk-management systems, representative clinical evidence, security and lifecycle controls, and any applicable authorization before marketing or clinical deployment.

## Claims that are not supported

OralLens AI must not currently claim that it:

- diagnoses plaque, gingivitis, periodontitis, caries, oral cancer, or any other condition
- confirms that a patient is plaque-free or disease-free
- estimates a calibrated probability of plaque, disease, clinical risk, or outcome
- provides clinical sensitivity, specificity, PPV, or NPV
- is validated across clinics, devices, demographics, geographies, or the general population
- improves dentist performance, workflow, treatment decisions, or patient outcomes
- recommends treatment, urgency, referral, or self-care
- is safe for unsupervised consumer use
- is a clinically validated system or medical device

The current dataset contains one or more positive annotations per manifest row. It therefore cannot establish patient-level specificity or negative predictive value in a representative plaque-negative population.

## Score interpretation

The displayed v3 value is a detector score used to rank and filter predicted boxes. It is not a clinical probability, and it comes from the historical model with the incorrect target mapping.

The historical v3 trust report recorded:

| Evidence | Validation | Internal test benchmark |
|---|---:|---:|
| ECE | 0.1940 | 0.1878 |
| MCE | 0.4290 | 0.4046 |
| Brier score | 0.2075 | 0.2035 |

This reliability definition asked whether a displayed box matched one available historical target at IoU `0.5`. Because those targets included source label `0` regions as plaque objects, the values are not valid plaque-match calibration and do not calibrate disease presence. The current portfolio roadmap does not claim or target a clinical probability.

## Unsupported and high-risk use

The current system is unsupported for:

- emergency assessment or urgent triage
- autonomous or consumer self-diagnosis
- treatment selection or monitoring treatment response
- replacing a clinical examination
- ruling out a condition because no box was returned
- images outside a documented acquisition and quality envelope
- protected health information in the local demonstration store
- clinical deployment without authentication, privacy, retention, audit, and security controls

## Boundary for any future clinical work

The current portfolio roadmap does not include clinical data collection or a medical claim. If that scope changes, the following roles would be required before collection or clinical evaluation begins:

- dental clinical lead: clinical purpose, population, reference standard, harms, and workflow
- biostatistician: endpoints, sample size, clustering, confidence intervals, and analysis plan
- data governance/privacy owner: consent or waiver, de-identification, access, retention, and data-use terms
- ML lead: model freeze, leakage controls, reproducibility, calibration, and monitoring
- regulatory/quality advisor when a medical or commercial claim is contemplated

These roles are not prerequisites for the current public/research-dataset portfolio plan and have not been appointed.

## Claim review gate

A claim may advance only when:

1. its intended-use elements are frozen
2. the supporting study was designed for that claim
3. the reference standard and analysis plan were pre-specified
4. the evaluated population and inputs match the claim
5. uncertainty, subgroup results, failures, and exclusions are reported
6. no model or threshold decision used the locked evaluation cohort
7. clinical, statistical, privacy, and regulatory reviewers approve the wording appropriate to their roles

## Authoritative frameworks

- [IMDRF Good machine learning practice guiding principles](https://www.imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf)
- [IMDRF Software as a Medical Device: Clinical Evaluation](https://www.imdrf.org/documents/software-medical-device-samd-clinical-evaluation)
- [FDA Clinical Decision Support Software guidance](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software)
- [STARD-AI diagnostic-accuracy reporting guideline](https://www.nature.com/articles/s41591-025-03953-8)
- [DECIDE-AI early clinical evaluation guideline](https://www.nature.com/articles/s41591-022-01772-9)

## Related documents

- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation protocol](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Training and evaluation](TRAINING.md)
- [Dataset card](DATASET_CARD.md)
- [Data strategy](DATA_STRATEGY.md)
- [Architecture](ARCHITECTURE.md)
