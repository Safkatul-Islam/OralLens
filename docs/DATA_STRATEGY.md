# Data Strategy

## Purpose

This document defines which data OralLens AI may use, what role each source may play, and what evidence is required before a new model result is presented. It is the decision record for the next model generation; acquisition mechanics remain in [Data acquisition](DATA_ACQUISITION.md), and experiment results remain in [Training and evaluation](TRAINING.md).

OralLens AI is a portfolio-grade experimental screening-support project. The practical target is credible, condition-specific visible-deposit localization on properly framed oral photographs, with explicit rejection or abstention for unsupported inputs. Plaque and supragingival calculus are separate task contracts and must not share a positive label merely because both are deposits. The project is not a diagnostic system, a medical device, or a substitute for a dental examination.

## Why the current model is not sufficient

The deployed v3 model was trained on one standardized fixed-orthodontic image source. Manual challenge images exposed severe domain shift: ordinary web and consumer-style photographs differ in framing, visible lips and facial skin, device characteristics, lighting, scale, and appliance status.

A separate semantic audit found that the source annotations use `0` for an annotated region without plaque and `1` for plaque presence. Historical target conversion mapped every source box to detector class `1`. Consequently, all v1-v3 annotation-matching metrics are contaminated and must not be used as clean plaque-only performance evidence. The corrected conversion keeps only source label `1` as plaque objects; source label `0` is not a TorchVision object class.

The corrected source-aware v4 experiment still used the same underlying AIRC images. It was stopped after two completed epochs: training loss improved from `0.7967` to `0.6149`, while validation loss worsened from `0.8250` to `1.0287`. Epoch 1 remained the best checkpoint. This is early overfitting evidence on a narrow source distribution, not a model-quality result. V4 must not be resumed, evaluated, integrated, or promoted.

The next credible result therefore requires a condition-specific task, genuinely complementary target-domain data, and a controlled pilot. More epochs on v3 or v4 would not solve the acquisition-domain mismatch.

## Dataset roles

Every admitted dataset must have exactly one or more declared roles. A source suitable for one role is not automatically valid for another.

| Role | Permitted use | Required evidence |
|---|---|---|
| Plaque supervision | Train or evaluate plaque localization | Explicit plaque label definition, spatial annotations, provenance, usage rights, and patient/group identity |
| Calculus supervision | Train or evaluate supragingival calculus localization | Explicit calculus definition, spatial annotations, ordinary-RGB acquisition context, provenance, usage rights, and patient/group identity |
| Mouth/teeth ROI | Learn or test whether a usable oral region is present | Compatible oral-region labels and capture conditions; never relabel as plaque-negative |
| Input quality | Train or evaluate blur, darkness, framing, and visibility gates | Defined quality criteria and representative capture failures |
| OOD/abstention | Test rejection of unsupported inputs | Source identity and a documented reason each group is outside the plaque detector's supported domain |
| Locked challenge evaluation | Final model comparison only | Independent patient/source boundary, frozen protocol, no threshold or model tuning |

Non-plaque datasets must never be treated as plaque-negative examples merely because they lack plaque annotations. An unlabeled plaque region is not a confirmed negative.

## Candidate source decisions

| Source | Evidence relevant to this project | Decision and permitted role |
|---|---|---|
| AIRC-LABDEN orthodontic-plaque Parts 1 and 2 | Standardized fixed-orthodontic images with peri-tooth region labels encoding plaque absence/presence rather than precise deposit outlines | **Restricted auxiliary plaque source.** Preserve corrected semantics and patient/source-family isolation. Do not use as the sole consumer-image foundation or represent its boxes as precise plaque masks. Part 1 remains excluded while its official nested archive fails integrity validation. |
| ODS/Oralformer visible oral-deposit segmentation release | The paper and official repository report 2,602 image-mask pairs across plaque, calculus, and caries, but the public release lacks an explicit license, data card, provenance/grouping record, annotation protocol, checksums, and documented split policy | **Deferred after admission audit; do not download or train.** Reconsider only after the maintainers clarify usage rights, per-condition contents, source/patient identity, acquisition context, annotation quality control, duplicate handling, and redistribution. Plaque and calculus masks must remain separate. |
| Plaque/teeth/gingiva segmentation dataset, 504 images from 168 subjects | Three standardized views, ages 8-75, patient-wise development and holdout boundaries, and multi-class masks; the source explicitly includes dental calculus inside the plaque mask | **Conditional auxiliary plaque candidate, not admitted.** Access is by reasonable request. Require dataset-specific terms and separate plaque/calculus semantics or an approved reannotation policy before use. It cannot serve as clean plaque-only or calculus-only supervision in its published form. |
| Undyed biofilm dataset, 480 images from 160 people | Three DSLR/retractor views per subject, people with and without appliances, explicit undyed-biofilm polygons, disclosed-plaque agreement evidence, and patient-sized folds | **Highest-priority plaque access candidate, not admitted.** Request data and terms from the corresponding author. Useful for plaque segmentation if patient IDs and rights are supplied, but the controlled professional-camera domain cannot be the sole consumer-image foundation. The 2023 and 2025 papers describe the same experimental dataset, not two independent sources. |
| SDPSeg stained/disclosed plaque segmentation | Spatial plaque supervision, but staining/disclosure changes the visual target | **Separate stained-plaque task only.** Audit rights and download integrity before use; do not merge with ordinary unstained RGB plaque without a declared cross-domain protocol. |
| MIO gingival-condition images | Ordinary oral photographs with image-level healthy/gingivitis/periodontitis labels, not deposit-localization annotations | **Conditional ROI/OOD source only.** Never treat unannotated plaque or calculus as negative supervision. |
| PKNU calculus collection | Small collection assembled from internet sources with weak patient identity/provenance and mixed label concerns | **Rejected as core training or evaluation evidence.** It may not define a protected benchmark. |
| SMART-OM, smartphone caries, OralCam, SegmentAnyTooth, IO150K, Kaggle oral-disease collections, and other adjacent sources | Potentially useful for ROI ideas or qualitative challenge cases, but current evidence has incompatible labels, unclear rights/provenance, synthetic content, or inadequate grouping | **Rejected or deferred for deposit supervision.** Reconsider only after a role-specific admission record resolves the gap. |

Primary research records used for this decision include the [AIRC-LABDEN publication](https://pmc.ncbi.nlm.nih.gov/articles/PMC13224366/), the [504-image plaque/gingiva segmentation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC12976221/), and the [480-image undyed biofilm study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11972646/). Publication access does not by itself grant access to or permission to redistribute the underlying dataset.

## ODS/Oralformer admission audit

Audit date: `2026-08-11`

Official records reviewed:

- [peer-reviewed PubMed record](https://pubmed.ncbi.nlm.nih.gov/40036515/) for IEEE Transactions on Image Processing, DOI `10.1109/TIP.2025.3544139`
- [paper-designated Oralformer repository](https://github.com/LintaoPeng/Oralformer)
- repository-linked Google Drive landing page for `ODS_dataset.zip`; the archive was not downloaded

| Admission field | Verified evidence | Status |
|---|---|---|
| Official identity | Paper, DOI, authors, repository, and aggregate 2,602 image-pair claim agree | Pass |
| Target relevance | Paper and repository name plaque, calculus, and caries with image-mask pairs | Conditional; per-condition counts and mask semantics are not exposed |
| Dataset license | No dataset license or usage/redistribution statement is visible in the official repository or Drive landing page | Fail closed |
| Version and integrity | One ZIP link is exposed, without release version, size, checksum, or signed manifest | Fail closed |
| Image provenance | Original collection sources, sites, devices, and source chain were not verifiable from the accessible official records | Fail closed |
| Patient/group identity | No subject count, patient keys, encounter grouping, or patient-level split mechanism was verifiable | Fail closed |
| Acquisition envelope | The records claim all age groups but do not expose the capture protocol or representative consumer-device distribution | Fail closed |
| Annotation contract | Pixel-paired segmentation is claimed, but annotator qualifications, class schema, overlap rules, and quality control were not verifiable | Fail closed |
| Split and leakage controls | No official train/validation/test manifest, grouping rule, or duplicate/near-duplicate audit was exposed | Fail closed |
| Reproducibility | The public repository contained only `.gitignore` and `README.md` during the audit; no code, preparation scripts, or data card were visible | Fail closed |

Decision: **deferred, not admitted**. Public download availability does not establish permission or fitness for this project. Do not download, prepare, train, evaluate, or cite ODS as project evidence until the maintainers answer the blocking questions and any supplied terms permit the intended portfolio/research use.

Required maintainer clarification:

1. What license governs the images, masks, and derived model artifacts, and is portfolio/research training permitted?
2. May the dataset or transformed annotations be redistributed, or must all artifacts remain local?
3. How many original images and independent subjects belong to plaque, calculus, and caries respectively?
4. Were any images sourced from the internet, prior datasets, or repeated captures, and are source/subject identifiers available?
5. Which devices, sites, views, lighting, and clinical or consumer capture protocols produced the images?
6. Who annotated each class, what definitions and overlap rules were used, and how was mask quality reviewed?
7. Is there an official patient-aware split, and were duplicates or near-duplicates checked across it?
8. Is a versioned archive manifest with file counts and cryptographic checksums available?

## Ordinary-RGB plaque source audits

Audit date: `2026-08-11`

### Undyed biofilm dataset: 480 images from 160 people

Official records reviewed:

- [2025 full-text study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11972646/), DOI `10.4317/medoral.26937`
- [2023 source study and publisher record](https://onlinelibrary.wiley.com/doi/10.1111/jcpe.13774)

The 2025 ensemble study reuses the 480-image experimental dataset described in the 2023 study. It provides a new evaluation strategy, not an additional independent image cohort.

| Admission field | Verified evidence | Status |
|---|---|---|
| Target relevance | Ordinary-RGB, undisclosed dental-biofilm polygons on full frontal and lateral intraoral photographs | Strong pass for visible-plaque segmentation |
| Cohort and grouping | 480 images from 160 people, three views per person, ages 5-73, with and without orthodontic appliances; folds contain 20 people/60 images | Conditional pass; actual patient/group keys must accompany the release |
| Reference standard | One dentist annotated yellowish areas, loss of shine, and rough/granular appearance; a separate 96-image disclosed/undisclosed set produced ICC `0.93` | Conditional pass; one annotator remains a bias limitation |
| Acquisition | Canon EOS Rebel T3, 100 mm macro lens, ring flash, lip retractor, fixed settings, high-resolution focused JPEG | Pass for a controlled clinical-photo domain; fail as representative consumer/mobile evidence |
| Eligibility | Excluded low light, blur, saliva bubbles, visible caries, and tetracycline staining; limited to buccal frontal/left/right views | Restricted domain; unsuitable as the sole application-facing source |
| Split design | Eight patient-sized groups, seven-fold train/validation rotation, and one fixed 20-person test group | Methodological pass; project partitions must be recreated from supplied subject IDs rather than image-level randomization |
| Ethics and de-identification | Private-clinic retrospective cohort, ethics protocol `4.434.730`, participant letters/random identifiers, no visible patient IDs | Pass at publication level; supplied data-use terms must govern project access |
| Dataset availability and license | No public archive, dataset license, checksum, or explicit data-availability statement was found in the official full text; article CC BY does not automatically license the underlying patient images | Fail closed pending written access terms |

Decision: **highest-priority access candidate, not admitted**. If the authors provide the original images, polygon masks, subject/view identifiers, and written portfolio/research usage terms, this source is suitable for a small plaque-segmentation pilot. It must remain source-stratified and be paired with a separate consumer-style acquisition boundary; it cannot substantiate phone-selfie performance by itself.

Required access clarification:

1. Are the 480 original RGB images and polygon masks available for non-commercial portfolio/research training?
2. What license or data-use agreement governs images, annotations, derived checkpoints, metrics, and screenshots?
3. May derived models and aggregate results be published while raw images remain private?
4. Will stable subject and frontal/left/right view identifiers be supplied for patient-aware splitting?
5. Are the 2023 and 2025 analyses based on exactly the same 480 images and masks, and which annotation version is authoritative?
6. Are original file metadata, archive counts, and cryptographic checksums available?
7. Do any consent, ethics, geographic, age, or redistribution restrictions affect this portfolio use?

### Multi-class plaque/gingiva dataset: 504 images from 168 subjects

Official records reviewed:

- [2026 full-text article](https://pmc.ncbi.nlm.nih.gov/articles/PMC12976221/), DOI `10.1007/s13167-025-00432-5`
- [publisher-version institutional copy](https://kclpure.kcl.ac.uk/ws/portalfiles/portal/367280263/s13167-025-00432-5.pdf)

| Admission field | Verified evidence | Status |
|---|---|---|
| Target relevance | Polygon annotations for anterior/posterior teeth and visible plaque plus healthy/inflamed gingival sites | Conditional; useful spatial supervision but broader than the current task |
| Plaque semantics | Colour, brightness, and texture changes were annotated as biofilm, and mineralized plaque/dental calculus was explicitly included in the same plaque class | Fail for clean plaque-only or calculus-only supervision without separate labels or approved reannotation |
| Cohort and grouping | 504 images from 168 pre-orthodontic subjects, three standardized views each, ages 8-75, 56% female, and 5% of images with fixed appliances | Pass for reported subject identity; limited braces representation |
| Annotation reliability | One calibrated periodontist; prior intra-examiner plaque consistency `r = 0.8` and clinical-photographic plaque correlation `r = 0.6`; 480 annotation hours reported | Conditional pass; single-rater and combined-label limitations remain |
| Acquisition | Retrospective private radiology-clinic images with standardized frontal/left/right views and no exclusion by image quality or oral condition | Conditional; camera/device, resolution, lighting, and capture procedure were not documented in accessible methods |
| Split design | Patient-wise 17-subject/51-image holdout; remaining 151 subjects/453 images used in five-fold development | Pass at publication level; exact manifest and holdout identities must accompany access |
| Ethics and de-identification | Ethics decision `5.681.738`, retrospective anonymized data, consent waiver, randomized subject codes, no identifiable data published | Pass at publication level |
| Dataset availability and license | Available from the corresponding author on reasonable request; article is CC BY 4.0, but the unpublished dataset still requires explicit supplied terms | Fail closed until access and dataset-specific rights are granted |

Decision: **conditional auxiliary candidate, not admitted**. Its broader inclusion policy and tooth/gingiva masks could help ROI and robustness research. Its published plaque mask must not be called plaque-only because calculus is included. Admission requires either separately identifiable plaque and calculus annotations or permission and a documented reannotation policy that preserves the original mask for auditability.

Required access clarification:

1. Are the 504 original images, VIA annotations, randomized subject IDs, view labels, and published split assignments available?
2. What dataset-specific terms permit training, derived checkpoints, metrics, and portfolio screenshots?
3. Can plaque and dental calculus be separated in the existing annotations or source records?
4. If they cannot, may the images be reannotated under separate plaque and calculus definitions while preserving the original combined masks?
5. What camera/device, resolution, lighting, retractors, and capture protocol produced the photographs?
6. Are the supplementary class counts and classifiable/non-classifiable region rules included with the release?
7. Is a versioned archive manifest with file counts and checksums available?

### Complementary role decision

If both sources become available under compatible terms, they must remain separately identifiable:

- use the 480-image source as primary plaque-mask supervision for a controlled ordinary-RGB pilot
- use the 504-image source only after resolving the plaque-plus-calculus label, with its tooth/gingiva masks considered for ROI or auxiliary multi-task learning
- never merge the publishers' holdout cohorts into training or use their published test results to tune this project
- maintain a new, source-stratified validation boundary and a separate rights-safe consumer-style challenge boundary
- report per-source performance so one controlled cohort cannot conceal failure on the other

## Admission gate

Before downloading or merging a source, record and verify:

1. official source page, release/version, retrieval date, and maintainer
2. dataset-level license and whether training, derived artifacts, and redistribution are permitted
3. original image provenance, consent/de-identification statement when relevant, and source chain
4. patient, encounter, clinic, device, and augmentation identifiers available for leakage control
5. exact label semantics, annotation procedure, class balance, and whether absence is truly annotated
6. capture protocol, image resolution, appliance status, views, lighting, and population scope
7. archive hashes, image decodability, duplicate/near-duplicate checks, and annotation geometry
8. intended role in this project and why that role is compatible
9. exclusions, transformations, and mapping decisions in an auditable preparation report
10. a patient- and source-aware split plan created before model evaluation

If provenance, license, group identity, or label meaning is ambiguous, the source fails closed and is not used for training or evaluation.

The AIRC release is the first source registered under this contract. Its record declares the `plaque_supervision` role, the `airc-labden-orthodontic-plaque` source family, patient grouping, standardized orthodontic capture context, source label semantics, and known limitations. Registration does not override artifact integrity: the failed Part 1 nested archive remains excluded.

## Target input envelope

The next application-facing pipeline should accept only a documented, testable image envelope. The proposed contract is:

- JPEG or PNG oral photograph
- teeth and gingival margins visible at a usable scale
- sufficient focus, brightness, and contrast for visual inspection
- mouth/teeth region occupies enough of the image after ROI detection
- supported view and framing, with no assumption that a no-box result means plaque-free

Blurred, very dark, heavily obstructed, non-oral, extreme close-up, or face-dominant images should produce a concise unsupported-input or insufficient-quality result. These checks are proposed work, not current implemented behavior.

## Task-specific compatibility rules

- Keep plaque and calculus annotations as separate classes, datasets, experiments, and claims unless a documented multi-task design explicitly preserves both labels.
- Require spatial condition annotations for localization; image-level disease labels do not become boxes or masks.
- Do not treat missing annotations as confirmed negative labels.
- Do not convert peri-tooth status regions into precise deposit masks.
- Do not merge stained/disclosed plaque with ordinary unstained plaque without measuring the domain difference.
- Keep ROI, image-quality, and OOD data out of condition-positive supervision.

## Next model development plan

The next model is a new experiment boundary, not a resume or continuation of v3 or v4:

1. freeze either the ordinary-RGB plaque task or the ordinary-RGB supragingival-calculus task before preparing data
2. keep ODS/Oralformer deferred unless its maintainers resolve the documented admission blockers
3. define patient-, source-, and derivative-aware development, validation, and locked challenge partitions before training
4. add a separate mouth/teeth ROI and input-quality/OOD stage; do not overload the condition model with unrelated labels
5. require CUDA explicitly and record device, model/tensor placement, timing, throughput, utilization samples, and peak GPU memory
6. run a monitored smoke test, then a small pilot with predeclared comparison and stop criteria
7. initialize a future model from an approved baseline under a new experiment identity; do not resume v3 or v4
8. select thresholds on validation only and evaluate once on the locked challenge boundary
9. report per-source, per-patient, image-quality, and failure-mode results alongside aggregate metrics
10. promote only if representative challenge behavior and the condition-specific baseline improve materially

The current v3 application may remain available as historical engineering evidence, but it must be labeled as a model with known semantic and real-world generalization limitations.

## Training and promotion go/no-go criteria

Training may start only when:

- label semantics are verified and covered by regression tests
- every source passes provenance, license, integrity, and group-identity gates
- split isolation is proven at patient and source/derivative levels
- dataset roles and class mappings are documented
- any image exclusions have explicit reasons and counts
- the target condition, annotation modality, unit of analysis, and success metric are frozen
- the admitted training data adds target-domain information rather than only more derivatives of the same images
- CUDA is required rather than silently falling back to CPU, and the smoke run records telemetry
- the pilot budget and stop criteria are declared before execution
- the config points to a new output directory and cannot overwrite historical artifacts

Promotion may be considered only when:

- validation-selected operating policy is frozen before locked evaluation
- no locked result is used to retune the model or threshold
- challenge images include supported consumer-style variation and explicit unsupported inputs
- failure cases are reviewed, including zero-detection and face-dominant images
- the application can abstain when input quality or scope is unsupported
- all critical ML, backend, frontend, and end-to-end checks pass

## Immediate actions

1. keep ODS deferred unless its maintainers provide the documented license, provenance, grouping, annotation, and integrity clarifications
2. treat any author response to an outstanding dataset access request as evidence for a separate admission decision; request status is tracked outside version control
3. specify and implement the image-quality, oral-ROI, and abstention contract with tests
4. upgrade the trainer preflight to require CUDA and emit auditable runtime telemetry before another experiment
5. prepare a new source-neutral manifest and condition adapter only after the admitted data and split policy are fixed
6. build a small, rights-safe challenge set with recorded source and expected handling; never tune against its final locked portion

## Related documents

- [Dataset card](DATASET_CARD.md)
- [Data acquisition](DATA_ACQUISITION.md)
- [Training and evaluation](TRAINING.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Pipeline](PIPELINE.md)
