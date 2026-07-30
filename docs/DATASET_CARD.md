# Dataset Card: Orthodontic Plaque Part 2

## Intended use

This dataset supports development and evaluation of the OralLens AI plaque-candidate object detector. It is used only for an experimental portfolio and learning project.

It must not be interpreted as sufficient evidence for diagnosis, clinical deployment, population-level claims, or medical-device performance.

## Data in use

Prepared dataset root:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Prepared manifest:

`ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`

Only verified Part 2 material is used. Part 1 remains excluded because its nested 7z archive failed the official integrity check. The source archive was not silently repaired or substituted.

## Structure and label contract

Each manifest row represents one image sample and includes one or more plaque-candidate annotations. Annotations use normalized center-width-height coordinates:

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

The validation split is used for operating-threshold selection. The test split was evaluated once at the frozen threshold and is not used for tuning.

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

The v2 operating threshold `0.65` was selected on all 468 validation images. Held-out evaluation then covered all 858 test images at that frozen threshold and IoU `0.5`:

- precision: `0.7582`
- recall: `0.7037`
- F1: `0.7299`
- mean matched IoU: `0.7541`

These values measure overlap with this dataset's annotations. They do not establish clinical sensitivity, specificity, or generalization.

## Known limitations

- specialized orthodontic-plaque context rather than the full range of oral conditions
- one source dataset and its acquisition/annotation conventions
- generated rotations and related boundary behavior may not represent real capture variation
- no demonstrated cross-clinic, cross-device, demographic, or geographic representativeness
- labels may reflect annotator subjectivity and source-specific definitions
- patient-aware splitting reduces direct leakage but does not remove all dataset bias
- absence of a returned box does not mean absence of a real oral-health concern

## Data handling

Raw, prepared, and generated dataset artifacts are kept local and excluded from git. The source material is not rewritten during loading or target conversion. Exclusions and policy decisions must be recorded rather than silently applied.

Future external use requires a fresh review of license terms, privacy, consent, provenance, intended use, and distribution restrictions.

## Related documents

- [Data acquisition](DATA_ACQUISITION.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
- [ML guide](../ml/README.md)
