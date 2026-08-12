# Dataset Card: Orthodontic Plaque Part 2

## Intended use

This dataset supports restricted development and internal evaluation of plaque-status regions in standardized fixed-orthodontic photographs. It is used only for an experimental portfolio and learning project. Its region boxes are not precise outlines of visible plaque deposits and do not establish performance on consumer-style photographs.

It must not be interpreted as sufficient evidence for diagnosis, clinical deployment, population-level claims, or medical-device performance.

## Data in use

Prepared dataset root:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Historical v3 prepared manifest:

`ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`

Corrected source-aware v4 prepared manifest:

`ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`

Only verified Part 2 material is used. Part 1 remains excluded because its nested 7z archive failed the official integrity check. The source archive was not silently repaired or substituted.

## Structure and label contract

Each manifest row represents one image sample and includes one or more annotated peri-tooth status regions. The boxes identify regions assessed for plaque presence; they are not deposit-segmentation masks or guaranteed tight plaque-localization boxes. Annotations use normalized center-width-height coordinates:

- `x_center`
- `y_center`
- `width`
- `height`

The source label is a plaque-presence attribute for each annotated region:

- `0`: plaque absent in the annotated region
- `1`: plaque present in the annotated region

TorchVision's detector contract is different:

- `0`: implicit background, reserved by TorchVision
- `1`: plaque candidate

Target conversion validates source labels in `{0, 1}` and emits detector objects only for source label `1`. Source label `0` boxes remain in the manifest for auditability but are not converted to plaque objects. All current manifest rows contain at least one source label `1`, so the dataset still does not represent a clinically sampled plaque-negative image or patient population.

## Patient-aware splits

Patients do not cross split boundaries.

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |
| **Total** | **5,160** | **74** | **65,238** |

Source-label audit:

| Split | Plaque absent (`0`) | Plaque present (`1`) | Total annotations |
|---|---:|---:|---:|
| Train | 996 | 47,102 | 48,098 |
| Validation | 30 | 6,040 | 6,070 |
| Test | 144 | 10,926 | 11,070 |
| **Total** | **1,170** | **64,068** | **65,238** |

The validation split is used for operating-threshold selection. The test split measured fixed policies for v2 and v3 and was re-run for descriptive trustworthiness evidence. It is not used for threshold tuning, but it has been examined across model generations and is therefore an internal benchmark rather than a pristine external cohort.

## Validation controls

The project loader checks:

- required manifest columns and known split values
- UTF-8/CSV and annotation JSON structure
- positive annotation count
- integer source labels restricted to `{0, 1}` during target conversion
- numeric, finite annotation fields
- individual normalized values in `[0,1]`
- safe relative POSIX image paths and supported extensions
- containment within the configured dataset root
- missing files and disallowed symlinks

Target conversion validates every source box before filtering, derives bounded corners, rejects invalid or degenerate geometry, and then retains only plaque-present regions as detector targets. Images with no retained positive targets are supported by the TorchVision training contract, although the current manifest has none.

## Boundary audit and policy

A complete structured audit checked whether derived corners crossed `[0,1]`. It found:

- affected images: 1
- affected annotations: 2
- affected split: test only
- maximum overflow: approximately `5e-7`
- degenerate boxes after clipping: 0

The affected sample was a rotated image. The scale of the crossings supports a floating-point/rotation-edge interpretation rather than major corruption.

The implemented policy allows at most `1e-6` of boundary tolerance, clips tolerated derived corners to the valid image extent with TorchVision, and then rejects non-positive boxes. Larger violations still fail closed. Source manifest values and metadata remain unchanged for auditability; no sample is silently skipped.

## Historical target-conversion defect

Historical v1-v3 target conversion assigned detector label `1` to every source annotation, including 1,170 regions whose source label was `0` (plaque absent). The defect is corrected and covered by regression tests, but the existing v1-v3 checkpoints were trained under the old mapping.

Consequences:

- v1-v3 checkpoints must not be resumed for the corrected experiment
- v1-v3 precision, recall, F1, IoU, calibration, and patient-level results are not valid plaque-only evidence
- existing metrics remain local historical artifacts for reproducibility and defect analysis
- the stopped corrected v4 diagnostic does not provide model-quality evidence

The active local application still uses v3 and therefore retains this known limitation.

## Stopped corrected v4 diagnostic

The source-aware v4 manifest was used in a fresh run initialized independently of v1-v3. The user stopped the experiment before epoch 3 after validation loss reversed:

| Completed epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | `0.7966761603586116` | `0.8249539341299962` |
| 2 | `0.6149270393257503` | `1.028658177671779` |

`checkpoint_best.pt` records epoch 1 and `checkpoint_last.pt` records epoch 2. Both checkpoints were inspected read-only with `weights_only=True`; they record `device_type="cuda"` and CUDA RNG state. This proves the completed epochs followed the CUDA code path, but historical utilization and throughput were not logged.

The run was not evaluated, assigned an operating threshold, integrated with the backend, or promoted. Its artifacts remain isolated local evidence. Do not resume it or cite it as corrected plaque-detector performance.

## Historical evaluation record

Under the incorrect target mapping, v3 selected threshold `0.85` on 468 validation images and produced the following internal fixed-threshold measurements on 858 test images at IoU `0.5`:

- precision: `0.7671`
- recall: `0.7923`
- F1: `0.7795`
- mean matched IoU: `0.8102`

These numbers are retained only to reproduce what the old pipeline reported. They mix plaque-present and plaque-absent source regions as positive objects and must not be cited as current detector quality, clinical sensitivity, specificity, or generalization.

## Historical variation and failure evidence

V3 trustworthiness reporting evaluated the same incorrect target contract at threshold `0.85` and the 25-detection cap. No validation image reached the cap. One test image exceeded it by one historical false positive; no historical true positive was truncated.

Patient-level v3 F1 ranged from `0.6769` to `0.9245` on validation and from `0.6824` to `0.8889` on the internal test benchmark. These patients are dataset groups, not demographic or clinical subgroups, and the variation must not be interpreted as a fairness result.

Dark brightness-down samples repeatedly appeared among images with the most false negatives on both splits. Blur also appeared in the internal test failure set. Because these conditions are source-dataset augmentations, they are useful hypotheses for controlled robustness testing but do not establish performance under real clinical acquisition conditions.

V3 score-to-match ECE was `0.1940` on validation and `0.1878` on test. Because the matching targets included source label `0` regions as positives, these values are historical defect evidence rather than valid plaque score calibration. They are not probabilities of plaque or disease.

## Known limitations

- specialized orthodontic-plaque context rather than the full range of oral conditions
- one source dataset and its acquisition/annotation conventions
- peri-tooth plaque-status boxes rather than precise plaque-deposit outlines
- no calculus/tartar annotation contract
- only 74 patient groups; image and annotation counts overstate the number of independent clinical subjects
- v1-v3 checkpoints and metrics use the incorrect historical source-label conversion
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

Future external use requires a fresh review of license terms, privacy, consent, provenance, intended use, and distribution restrictions. Candidate sources and admission rules are recorded in [Data strategy](DATA_STRATEGY.md).

## Related documents

- [Data acquisition](DATA_ACQUISITION.md)
- [Data strategy](DATA_STRATEGY.md)
- [Data acquisition and annotation](DATA_ACQUISITION_AND_ANNOTATION.md)
- [Intended use and claims](INTENDED_USE_AND_CLAIMS.md)
- [Clinical evidence plan](CLINICAL_EVIDENCE_PLAN.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
- [ML guide](../ml/README.md)
