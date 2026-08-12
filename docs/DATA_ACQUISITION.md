# Data Acquisition and Preparation

## Purpose

This document records how dataset material is admitted into the OralLens AI ML workflow. It focuses on provenance, integrity, extraction, preparation, and audit controls; model results belong in [Training and evaluation](TRAINING.md).

## Admission principles

- use a clearly identified source and version
- define the source's role before download: plaque supervision, oral ROI, input quality, OOD testing, or locked evaluation
- preserve source artifacts rather than silently rewriting them
- verify official integrity information when provided
- treat archives and nested paths as untrusted input
- inspect licenses and permitted use before redistribution
- verify label semantics from primary source documentation rather than inferring them from filenames or numeric values
- require patient or equivalent group identity before using a source for model evaluation
- keep raw, prepared, and generated data outside version control
- record exclusions and their reasons
- create patient-aware splits before model evaluation

Increasing image count is not an admission reason by itself. A source that lacks plaque labels may support oral-ROI or out-of-distribution testing, but it must not be converted into plaque-negative supervision.

## Current source decision

The active workflow uses only the verified Part 2 orthodontic-plaque material under:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Part 1 remains excluded. Its nested 7z archive did not pass the official integrity check, so it was not trusted, repaired, or used as partial training data.

## Source admission contract

The trusted-source configuration `ml/configs/orthodontic_plaque_v3.toml` uses admission schema version 2. Every dataset CLI operation loads this contract before it can verify, finalize, inspect, extract, audit, or prepare source material.

The contract requires:

- a stable dataset and source-family identity
- publisher, license, reviewed date, and capture context
- one or more allowlisted dataset roles
- grouping keys used to prevent evaluation leakage
- explicit label semantics for supervised sources
- limitations that remain attached to the source decision
- publisher artifact URLs, hashes, filenames, and extraction limits

For the `plaque_supervision` role, the contract fails unless label `0` is explicitly `plaque_absent_region` and label `1` is `plaque_present_region`. Unsupported roles, duplicate or malformed grouping keys, missing limitations, invalid dates, and incomplete label semantics fail before artifact processing.

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

## Source-label semantics

The AIRC-LABDEN source region label means:

- `0`: plaque absent in the annotated region
- `1`: plaque present in the annotated region

TorchVision reserves detector class `0` for implicit background. Target conversion therefore validates both source values and emits object targets only for source label `1`. Source label `0` boxes are retained in the prepared manifest for provenance but are not converted into plaque objects.

Source audit and manifest preparation now also reject class IDs outside `{0,1}`. The prepared-manifest loader repeats the same check, and target conversion remains the final defensive boundary. This layered validation prevents an invalid semantic class from reaching training even if a prepared file is modified independently.

A full prepared-manifest audit counted 1,170 source label `0` annotations and 64,068 source label `1` annotations. Historical v1-v3 conversion incorrectly treated all 65,238 boxes as plaque objects. The correction is implemented and regression-tested, but no corrected model has yet been trained; see [Training and evaluation](TRAINING.md).

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
2. declare the exact project role and verify task/label compatibility
3. store raw artifacts in a versioned local directory
4. validate archives before extraction and nested paths during extraction
5. inspect image decodability and annotation schema
6. verify class meanings, negative-label meaning, and annotation procedure from primary documentation
7. identify the patient or grouping key used to prevent leakage
8. prepare deterministic patient- and source-aware split assignments and record counts
9. audit duplicates, normalized geometry, derived boxes, and derivative leakage
10. record exclusions and transformations with a reason and source identity
11. run the complete dataset and ML regression suite
12. update the dataset card and [Data strategy](DATA_STRATEGY.md) before training or evaluation claims

## Current verification status

The complete ML suite reports `161 passed, 1 skipped`. Dataset coverage includes admission metadata, schema failures, numeric validation, path containment, symlink behavior where supported, box boundary cases, source-label validation/filtering, empty positive targets, and regression paths.

The one skipped case depends on Windows symbolic-link privileges; platform-independent containment coverage remains active.

## Limitations

Acquisition integrity and patient-aware splits do not prove representativeness or clinical suitability. New datasets should not be combined merely to increase sample count; label meaning, acquisition setting, license, patient grouping, and task compatibility must be reviewed first.

See [Data strategy](DATA_STRATEGY.md), [Dataset card](DATASET_CARD.md), [ML environment](ML_ENVIRONMENT.md), and [ML guide](../ml/README.md).
