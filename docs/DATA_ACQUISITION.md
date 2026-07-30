# Data Acquisition and Preparation

## Purpose

This document records how dataset material is admitted into the OralLens AI ML workflow. It focuses on provenance, integrity, extraction, preparation, and audit controls; model results belong in [Training and evaluation](TRAINING.md).

## Admission principles

- use a clearly identified source and version
- preserve source artifacts rather than silently rewriting them
- verify official integrity information when provided
- treat archives and nested paths as untrusted input
- inspect licenses and permitted use before redistribution
- keep raw, prepared, and generated data outside version control
- record exclusions and their reasons
- create patient-aware splits before model evaluation

## Current source decision

The active workflow uses only the verified Part 2 orthodontic-plaque material under:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Part 1 remains excluded. Its nested 7z archive did not pass the official integrity check, so it was not trusted, repaired, or used as partial training data.

## Preparation output

Prepared manifest:

`ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`

The manifest contains 5,160 image rows across 74 patients and 65,238 annotations. Split assignments are patient-aware and fixed:

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |

Test assignments are not modified in response to model performance.

## Loader and filesystem controls

The dataset boundary validates:

- required columns, split values, CSV encoding, and JSON structure
- finite numeric normalized annotation fields
- supported image extensions
- relative POSIX paths without unsafe components
- resolved containment below the configured dataset root
- missing images and disallowed symbolic links
- positive annotation counts

Source paths, prepared paths, datasets, and generated runs are ignored by git. The loader reads source annotations without modifying them.

## Derived-box audit

Center-width-height values can each be within `[0,1]` while a derived corner crosses the boundary. A structured full-manifest audit therefore checked all annotations after corner conversion.

Result:

- one rotated test image contained two affected annotations
- maximum boundary crossing was approximately `5e-7`
- both boxes retained positive area after clipping
- no large violation or broad corruption pattern was found

The target-conversion policy now permits only a `1e-6` tolerance, clips tolerated edges explicitly, and revalidates positive area. Larger or degenerate annotations are rejected. The manifest remains unchanged.

## Reproducibility and audit checklist

For every new source or version:

1. record source, version, retrieval date, license, and expected integrity values
2. store raw artifacts in a versioned local directory
3. validate archives before extraction and nested paths during extraction
4. inspect image decodability and annotation schema
5. identify the patient or grouping key used to prevent leakage
6. prepare deterministic split assignments and record counts
7. audit normalized and derived box geometry
8. record exclusions with a reason and source identity
9. run the complete dataset and ML regression suite
10. update the dataset card before training or evaluation claims

## Current verification status

The complete ML suite reports `132 passed, 1 skipped`. Dataset coverage includes schema failures, numeric validation, path containment, symlink behavior where supported, box boundary cases, and regression paths.

The one skipped case depends on Windows symbolic-link privileges; platform-independent containment coverage remains active.

## Limitations

Acquisition integrity and patient-aware splits do not prove representativeness or clinical suitability. New datasets should not be combined merely to increase sample count; label meaning, acquisition setting, license, patient grouping, and task compatibility must be reviewed first.

See [Dataset card](DATASET_CARD.md), [ML environment](ML_ENVIRONMENT.md), and [ML guide](../ml/README.md).
