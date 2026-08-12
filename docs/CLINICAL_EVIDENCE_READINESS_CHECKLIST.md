# Clinical Evidence Readiness Checklist

## Purpose and status

This checklist controls progression from OralLens AI's internal engineering evidence toward a retrospective external-validation study. It records what must be demonstrated before data collection, model development, locked evaluation, or stronger claims.

It is not a quality-system certificate, ethics approval, regulatory determination, or statement that OralLens is clinically validated.

## Status vocabulary

| Status | Meaning |
|---|---|
| `complete` | Required evidence is recorded, reviewed, and approved by the accountable owner |
| `in progress` | Work has started, but the gate is not satisfied |
| `not recorded` | The repository contains no evidence that the requirement is complete |
| `blocked` | A prerequisite or accountable external decision is missing |
| `not applicable` | Exclusion is justified and approved; never assumed |

Only an accountable owner may mark a clinical, statistical, privacy, site, ethics, quality, or regulatory item complete. Repository documentation can prepare the decision but cannot substitute for that sign-off.

## Current readiness verdict

| Activity | Current status | Reason |
|---|---|---|
| Portfolio MVP demonstration | `complete` | End-to-end engineering workflow and internal tests are documented |
| Internal model-development evidence | `complete` | V1-v3 training, validation selection, internal benchmark, and trust reports are recorded |
| Promote v3 into local application | `complete` | Typed config/startup boundaries, real integration, suites, build, and browser E2E are recorded |
| Collect or import clinical study data | `blocked` | Clinical/statistical owners and governance approvals are not recorded |
| Start a retrospective external validation | `blocked` | Protocol, reference standard, sample size, sites, approvals, and locked cohort are not approved |
| Make clinical-performance claims | `blocked` | No independent claim-aligned clinical evidence exists |
| Study dental-professional benefit | `blocked` | Requires prospective human-AI protocol after earlier evidence gates |

The current blocker is not another training epoch. It is accountable clinical study definition and access to governed, representative, independently sourced data.

## Gate 0: current engineering evidence

Owner: ML and engineering leads.

- [x] end-to-end upload, validation, inference, report, and visualization workflow demonstrated
- [x] active v3 runtime and historical model versions clearly distinguished
- [x] patient-aware development splits documented
- [x] validation-only threshold selection documented
- [x] fixed-threshold internal benchmark documented
- [x] checkpoint/config/data provenance recorded for v3 evidence
- [x] patient-level variation and recurring failure modes reported
- [x] score-to-match reliability separated from clinical probability
- [x] existing dataset's positive-only and 74-patient limitations documented
- [x] internal test cohort identified as already examined and unsuitable as fresh external evidence
- [x] current product language remains experimental and non-diagnostic

Gate 0 status: **complete for the portfolio MVP, not clinical validation**.

## Gate 1: accountable ownership and intended use

Owners: project sponsor, dental clinical lead, biostatistician, privacy/data-governance owner, ML lead, and regulatory/quality advisor when applicable.

- [ ] named dental clinical lead and responsibilities recorded
- [ ] named biostatistician and responsibilities recorded
- [ ] named privacy/data-governance owner recorded
- [ ] named site PI/data custodian for each proposed site recorded
- [ ] ML model owner and evidence-release authority recorded
- [ ] jurisdiction-specific regulatory/quality review need assessed
- [ ] exact clinical purpose approved
- [ ] intended user and workflow approved
- [ ] population, care setting, appliance status, and age range approved
- [ ] supported image views, devices, and quality envelope approved
- [ ] model output, user action, and failure behavior approved
- [ ] expected benefit, foreseeable harms, and unsupported uses reviewed

Required evidence:

- signed or controlled intended-use decision record
- responsibility matrix with role, authority, and escalation path
- dated review of claim wording against the [claims ladder](INTENDED_USE_AND_CLAIMS.md)

Gate 1 status: **blocked; accountable assignments and approvals are not recorded in the repository**.

## Gate 2: governance, ethics, privacy, and data rights

Owners: site PI, ethics/IRB authority, privacy/data-governance owner, and legal/data-rights owner.

- [ ] study classified under applicable ethics/IRB requirements
- [ ] approval, exemption, or other formal determination recorded
- [ ] consent process or waiver approved
- [ ] minimum-necessary data fields approved
- [ ] de-identification/pseudonymization procedure validated
- [ ] direct identifiers and linkage key isolated from ML systems
- [ ] data-use agreement and model-training rights approved
- [ ] publication, retention, withdrawal, correction, and deletion rules approved
- [ ] encryption, transfer, access, backup, and incident controls approved
- [ ] cross-site and cross-jurisdiction requirements resolved
- [ ] repository and demonstration storage explicitly prohibited for PHI
- [ ] breach/incident and participant-withdrawal workflows tested by responsible owners

Required evidence:

- controlled approval references, not PHI or confidential approval files committed to git
- data-flow diagram and access matrix
- retention/deletion schedule and incident procedure
- site export and de-identification acceptance record

Gate 2 status: **blocked; no clinical data activity is authorized by repository documentation**.

## Gate 3: protocol and statistical analysis

Owners: dental clinical lead and biostatistician.

- [ ] [External validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md) reviewed and version-frozen
- [ ] one primary endpoint and analysis unit approved
- [ ] clinically meaningful success criterion approved
- [ ] lesion/region matching rule clinically justified
- [ ] false-positive burden definition approved
- [ ] secondary and safety endpoints pre-specified
- [ ] patient-level sample-size calculation approved
- [ ] prevalence and enrichment assumptions documented
- [ ] clustering by patient/site/image/tooth/surface addressed
- [ ] confidence-interval method approved
- [ ] subgroup and multiplicity policy approved
- [ ] missing, indeterminate, unsupported, and excluded input rules approved
- [ ] sensitivity analyses pre-specified
- [ ] analysis software, code, and report schema versioning defined
- [ ] protocol deviation and unplanned-analysis policy approved

Required evidence:

- final protocol
- versioned statistical analysis plan
- sample-size report with assumptions and sensitivity calculations
- endpoint and denominator definitions understandable without reading source code

Gate 3 status: **in progress at engineering-draft level; clinical/statistical approval is absent**.

## Gate 4: acquisition and site readiness

Owners: site PI, dental clinical lead, imaging lead, and privacy/data owner.

- [ ] sites selected for intended population and independent evaluation role
- [ ] site contribution and recruitment/sampling targets approved
- [ ] consecutive or otherwise defensible sampling method documented
- [ ] target-positive, target-negative, hard-negative, and indeterminate coverage planned
- [ ] acquisition views, devices, settings, and quality envelope frozen
- [ ] timing relative to examination, brushing, cleaning, eating, treatment, or disclosure defined
- [ ] operator training and competency checks complete
- [ ] retake, burst, repeated-image, and failed-capture handling defined
- [ ] site export and pseudonymous identity mapping tested
- [ ] file integrity, de-identification, and metadata stripping tested
- [ ] site incident and correction workflow rehearsed

Required evidence:

- site manual and acquisition checklist
- device and operator training log
- pilot export acceptance report containing no direct identifiers
- signed site readiness decision

Gate 4 status: **not recorded**.

## Gate 5: reference standard and annotation readiness

Owners: dental clinical lead, assessor lead, and biostatistician.

- [ ] operational definition of plaque/non-plaque approved
- [ ] patient/image/tooth/surface/region label relationships defined
- [ ] reference assessment method and timing approved
- [ ] uncertainty, non-visibility, mixed findings, and poor quality defined
- [ ] hard-negative and confounder vocabulary approved
- [ ] at least two qualified assessors trained on versioned rubric
- [ ] assessors blinded to model output and partition
- [ ] independent-label preservation verified
- [ ] adjudicator/panel and disagreement rules approved
- [ ] agreement measure and acceptance target approved
- [ ] drift checks and annotation QC sampling defined
- [ ] model-assisted annotation prohibited or separately evaluated and approved

Required evidence:

- versioned assessor manual and target ontology
- training/calibration exercise results
- inter-assessor agreement and adjudication pilot report
- reference-standard approval signed by clinical and statistical owners

Gate 5 status: **not recorded**.

## Gate 6: data contract and secure validation system

Owners: privacy/data owner, data engineer, ML lead, and independent evaluation custodian.

- [ ] [Clinical data dictionary](CLINICAL_DATA_DICTIONARY.md) clinically and statistically approved
- [ ] machine-readable schema versioned separately from the current Part 2 manifest
- [ ] true negatives represented explicitly with zero target regions
- [ ] indeterminate/not-assessable statuses cannot become negatives
- [ ] assessor, adjudicated reference, and prediction entities isolated
- [ ] safe paths, supported formats, integrity hashes, and file decoding validated
- [ ] finite/bounded geometry and positive area validated
- [ ] patient/encounter/site/derived-image leakage checks implemented
- [ ] exact and near-duplicate detection implemented
- [ ] immutable release and correction/supersession process implemented
- [ ] access, partition, export, and locked-cohort events audited
- [ ] validation failures are concise, deterministic, and fail closed
- [ ] success, validation failure, privacy boundary, and leakage behavior tested

Required evidence:

- approved schema and vocabulary versions
- automated validator and security-focused test suite
- example synthetic records only; no PHI in fixtures
- release-audit report proving zero prohibited leakage

Gate 6 status: **in progress at documentation level; no future clinical schema or validator has been implemented**.

## Gate 7: pilot and dataset-release readiness

Owners: all Stage 1 owners.

- [ ] governance-approved pilot completed before full acquisition/import
- [ ] participant and image flow reconciled
- [ ] positive, negative, hard-negative, indeterminate, and unusable counts reviewed
- [ ] site/device/operator/view/quality coverage reviewed
- [ ] missingness and exclusion reasons reviewed
- [ ] reference agreement and adjudication burden acceptable
- [ ] exact/near-duplicate and cross-partition leakage audit passes
- [ ] paths, files, hashes, geometry, and metadata validation passes
- [ ] patient-level sample-size and subgroup targets remain achievable
- [ ] source licenses and permitted-use versions recorded
- [ ] clinical, statistical, privacy, site, and ML release sign-offs complete

Required evidence:

- pilot report with deviations and corrective actions
- immutable dataset release manifest and audit summary
- documented decision to proceed, revise, or stop

Gate 7 status: **blocked by Gates 1-6**.

## Gate 8: model-development and freeze readiness

Owners: ML lead and evidence-release authority.

- [ ] patient/site/related-image partitions frozen before development
- [ ] external locked labels inaccessible to model developers
- [ ] training, validation, and calibration roles enforced
- [ ] model architecture, initialization, preprocessing, and augmentation recorded
- [ ] development decisions and negative results retained
- [ ] calibration uses only its declared cohort
- [ ] operating threshold chosen without external-test access
- [ ] unsupported-input and failure behavior frozen
- [ ] checkpoint, config, data release, schema, and environment hashes recorded
- [ ] inference/evaluation parity and detection-cap effects tested
- [ ] final model card and known limitations reviewed
- [ ] frozen evaluation package accepted by independent custodian

Required evidence:

- immutable model/evaluation package
- reproducibility and deployment-parity report
- signed freeze record listing every identity/hash

Gate 8 status: **not started for a future claim-aligned model; v3 internal evidence does not satisfy this external-study gate**.

## Gate 9: locked external evaluation

Owners: independent custodian, biostatistician, dental clinical lead, and evidence-release authority.

- [ ] protocol, analysis plan, model, and report schema registered/frozen
- [ ] locked-cohort access authorization recorded
- [ ] package identity verified before execution
- [ ] registered analysis executed once
- [ ] inference and analysis failures retained
- [ ] participant flow, exclusions, missingness, and deviations reconciled
- [ ] primary result and confidence interval reported regardless of outcome
- [ ] site, subgroup, robustness, and failure analyses reported with uncertainty
- [ ] unplanned analyses labeled exploratory
- [ ] no post-access threshold/model/cohort change replaces the primary result
- [ ] supported and unsupported claim wording approved by accountable reviewers
- [ ] STARD-AI-aligned report prepared where applicable

Required evidence:

- locked evaluation report and provenance package
- signed clinical/statistical interpretation
- explicit pass, fail, or inconclusive decision against the pre-specified gate

Gate 9 status: **blocked by Gates 1-8**.

## Gate 10: prospective and human-AI readiness

Owners: clinical sponsor/PI, human-factors lead, statistician, privacy/security owners, regulatory/quality advisor, and ML lead.

- [ ] retrospective external evidence justifies further study
- [ ] prospective silent-mode protocol approved
- [ ] operational security, privacy, monitoring, latency, and failure recovery validated
- [ ] drift and unsupported-input monitoring defined
- [ ] human-AI benefit and harm endpoints approved
- [ ] reader order, washout, learning, experience, and clustering addressed
- [ ] automation bias, overreliance, explanation, and override behavior studied
- [ ] clinical incident and stop rules approved
- [ ] applicable registration, regulatory, and quality requirements satisfied

Gate 10 status: **out of scope until credible external retrospective evidence exists**.

## Promotion and claim rules

- Promoting v3 into the local portfolio application is an engineering deployment decision, not a clinical-evidence gate.
- More epochs or a higher internal F1 do not satisfy missing clinical, statistical, privacy, site, or external-validation evidence.
- A new model after locked-cohort access requires a new independent evaluation boundary for confirmatory claims.
- A failed or inconclusive study remains part of the evidence record.
- No checkbox may be marked complete solely because a document template exists.

## Next accountable actions

1. record the dental clinical lead, biostatistician, and privacy/data-governance owner
2. conduct a controlled review of the proposed research target and intended-use table
3. resolve the blocking decisions in the draft external validation protocol
4. commission the sample-size and statistical analysis plan
5. identify candidate sites and begin ethics/privacy/data-rights feasibility review
6. approve the minimum clinical data dictionary before implementing any schema or intake tooling

## Related documents

- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [External validation protocol](EXTERNAL_VALIDATION_PROTOCOL.md)
- [Clinical data dictionary](CLINICAL_DATA_DICTIONARY.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Dataset card](DATASET_CARD.md)
- [Architecture](ARCHITECTURE.md)
