# Dataset Card: AIRC-LABDEN Orthodontic Plaque Part 2

## Purpose and permitted use

OralLens uses the verified Part 2 AIRC-LABDEN orthodontic-plaque dataset for a portfolio-scale object-detection experiment. The task is to localize **plaque-positive peri-tooth regions** in standardized fixed-orthodontic photographs.

The annotations are not precise outlines of visible plaque deposits. Results from this dataset do not establish diagnosis, clinical safety, consumer-photo performance, or medical-device validity.

## Source material in use

Local dataset root:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Prepared manifests:

- historical manifest: `ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`
- source-aware manifest used by the corrected experiments: `ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`

Only verified Part 2 material is used. Part 1 remains excluded because its nested 7z archive failed the publisher-provided integrity check. The failed material was not repaired, substituted, or partially admitted.

Raw data, prepared manifests, and generated run artifacts remain local and are excluded from version control.

## Annotation semantics

Each manifest row represents one image and contains normalized center-width-height boxes around assessed peri-tooth regions:

- `x_center`
- `y_center`
- `width`
- `height`

The source label describes plaque status within the region:

- source label `0`: plaque absent in the annotated region
- source label `1`: plaque present in the annotated region

TorchVision reserves detector class `0` for implicit background. Target conversion therefore validates both source values but emits detector objects only for source label `1`; the application exposes those class-1 objects as plaque-positive peri-tooth region candidates.

The source-aware manifest contains 1,170 source class-0 regions and 64,068 source class-1 regions. Class-0 source regions remain in the manifest for provenance but are not converted into positive detector targets.

## Prepared population

The complete prepared release contains publisher-generated brightness, flip, blur, and rotation derivatives as well as genuine originals. Patients do not cross split boundaries.

| Split | Prepared images | Patients | All source annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |
| **Total** | **5,160** | **74** | **65,238** |

Image count is not treated as independent-sample count because many rows are derivatives of the same photograph.

## Genuine-original experiment population

The final controlled experiment removed pre-generated derivatives as independent samples. Training used only genuine originals with conservative augmentation applied online. Validation and final test evaluation were also originals-only.

| Split | Genuine originals | Patients | Foreground class-1 regions | Source class-0 regions |
|---|---:|---:|---:|---:|
| Train | 480 | 55 | 5,502 | 124 |
| Validation | 58 | 7 | 710 | 4 |
| Test | 107 | 12 | 1,293 | 23 |
| **Total** | **645** | **74** | **7,505** | **151** |

All images from one patient remain in one split. Original/derivative families are kept together, and derivatives are not counted as independent validation or test observations.

The validation originals selected the epoch-9 checkpoint and score threshold `0.80`. The 107-image test population was used only after those choices were frozen for Experiment 1. This is a patient-held-out internal benchmark, not an independent external or consumer-photo cohort.

## Validation and integrity controls

The data boundary validates:

- required columns, UTF-8/CSV structure, and known split values
- JSON annotation structure and annotation counts
- integer source labels restricted to `{0, 1}`
- finite normalized numeric fields
- safe relative POSIX image paths and supported extensions
- resolved containment below the configured dataset root
- missing files and disallowed symbolic links
- derived box bounds and positive area

A full derived-corner audit found two annotations in one rotated test image crossing the normalized boundary by approximately `5e-7`. The shared conversion boundary permits at most `1e-6` tolerance, clips only tolerated crossings with TorchVision, and then revalidates positive area. Larger violations and degenerate boxes fail closed. The source manifest is not modified and no sample is silently skipped.

## Historical label defect

Historical v1-v3 conversion mapped every source box to detector class `1`, including the 1,170 plaque-absent regions. Those checkpoints and their plaque-only metrics are invalid for the corrected task and are retained only as historical engineering evidence.

The corrected baseline and final Experiment 1 checkpoint use the source-aware class mapping. The final model is the originals-only online-augmentation candidate selected on validation at epoch 9; exact methodology and results are in [Training and evaluation](TRAINING.md).

## Limitations

- one source family and only 74 patient groups
- standardized fixed-orthodontic acquisition rather than consumer phone photography
- peri-tooth plaque-status boxes rather than deposit outlines or masks
- no representative image-level clean-mouth or consumer hard-negative population
- no external site, device, demographic, or geographic evaluation
- no calculus/tartar target contract
- the 12-patient internal test is too small for broad clinical generalization or subgroup claims
- the test partition has been inspected across model generations and is not a pristine future benchmark
- a missing prediction cannot be interpreted as plaque-free or disease-free

## Related documents

- [Data acquisition and preparation](DATA_ACQUISITION.md)
- [Training and evaluation](TRAINING.md)
- [Architecture](ARCHITECTURE.md)
- [ML guide](../ml/README.md)

