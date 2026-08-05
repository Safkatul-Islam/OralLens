# Dataset Card: Orthodontic Plaque Part 2

## Intended use

This dataset supports development and internal evaluation of the OralLens AI plaque-candidate object detector. It is used only for an experimental portfolio and learning project.

It must not be interpreted as sufficient evidence for diagnosis, clinical deployment, population-level claims, or medical-device performance.

## Data in use

Prepared dataset root:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Prepared manifest:

`ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`

Only verified Part 2 material is used. Part 1 remains excluded because its nested 7z archive failed the official integrity check. The source archive was not silently repaired or substituted.

## Structure and label contract

Each manifest row represents one image sample and includes one or more plaque-candidate annotations. The positive-annotation requirement means the current manifest does not represent a clinically sampled plaque-negative population. Annotations use normalized center-width-height coordinates:

- `x_center`
- `y_center`
- `width`
- `height`

The detector contract is binary:

- `0`: background
- `1`: plaque candidate

## Patient-aware splits

Patients do not cross split boundaries.

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |
| **Total** | **5,160** | **74** | **65,238** |

The validation split is used for operating-threshold selection. The test split measured fixed policies for v2 and v3 and was re-run for descriptive trustworthiness evidence. It is not used for threshold tuning, but it has been examined across model generations and is therefore an internal benchmark rather than a pristine external cohort.

## Validation controls

The project loader checks:

- required manifest columns and known split values
- UTF-8/CSV and annotation JSON structure
- positive annotation count
- numeric, finite annotation fields
- individual normalized values in `[0,1]`
- safe relative POSIX image paths and supported extensions
- containment within the configured dataset root
- missing files and disallowed symlinks

Target conversion also validates derived corner coordinates and positive-area boxes.

## Boundary audit and policy

A complete structured audit checked whether derived corners crossed `[0,1]`. It found:

- affected images: 1
- affected annotations: 2
- affected split: test only
- maximum overflow: approximately `5e-7`
- degenerate boxes after clipping: 0

The affected sample was a rotated image. The scale of the crossings supports a floating-point/rotation-edge interpretation rather than major corruption.

The implemented policy allows at most `1e-6` of boundary tolerance, clips tolerated derived corners to the valid image extent with TorchVision, and then rejects non-positive boxes. Larger violations still fail closed. Source manifest values and metadata remain unchanged for auditability; no sample is silently skipped.

## Evaluation use

The latest candidate is v3. Its operating threshold `0.85` was selected on all 468 validation images. Internal fixed-threshold evaluation then covered all 858 test images at IoU `0.5`:

- precision: `0.7671`
- recall: `0.7923`
- F1: `0.7795`
- mean matched IoU: `0.8102`

These values measure overlap with this dataset's annotations. They do not establish clinical sensitivity, specificity, or generalization.

## Internal variation and failure evidence

V3 trustworthiness reporting evaluated threshold `0.85` and the 25-detection cap on complete validation and test splits. No validation image reached the cap. One test image exceeded it by one false positive; no true positive was truncated.

Patient-level v3 F1 ranged from `0.6769` to `0.9245` on validation and from `0.6824` to `0.8889` on the internal test benchmark. These patients are dataset groups, not demographic or clinical subgroups, and the variation must not be interpreted as a fairness result.

Dark brightness-down samples repeatedly appeared among images with the most false negatives on both splits. Blur also appeared in the internal test failure set. Because these conditions are source-dataset augmentations, they are useful hypotheses for controlled robustness testing but do not establish performance under real clinical acquisition conditions.

V3 score-to-match ECE was `0.1940` on validation and `0.1878` on test, worse than v2 despite stronger annotation-matching metrics. This finding is conditional on this dataset, threshold `0.85`, IoU `0.5`, and the project matching policy; it is not probability calibration for disease.

## Known limitations

- specialized orthodontic-plaque context rather than the full range of oral conditions
- one source dataset and its acquisition/annotation conventions
- only 74 patient groups; image and annotation counts overstate the number of independent clinical subjects
- no representative plaque-negative cohort, so patient-level specificity, PPV/NPV, and rule-out claims cannot be established
- generated rotations and related boundary behavior may not represent real capture variation
- no demonstrated cross-clinic, cross-device, demographic, or geographic representativeness
- labels may reflect annotator subjectivity and source-specific definitions
- patient-aware splitting reduces direct leakage but does not remove all dataset bias
- material patient-level performance variation exists within the current splits
- synthetic darkness, blur, and rotation conventions may differ from real acquisition failures
- absence of a returned box does not mean absence of a real oral-health concern
- the internal test cohort has been inspected across model generations and cannot serve as fresh external evidence for future versions

## Data handling

Raw, prepared, and generated dataset artifacts are kept local and excluded from git. The source material is not rewritten during loading or target conversion. Exclusions and policy decisions must be recorded rather than silently applied.

Future external use requires a fresh review of license terms, privacy, consent, provenance, intended use, and distribution restrictions. Future clinical data must follow the governance, reference-standard, negative-case, patient/site isolation, and release gates in [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md).

## Related documents

- [Data acquisition](DATA_ACQUISITION.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
- [ML guide](../ml/README.md)
