# Data Acquisition and Preparation

## Purpose

This document records how dataset material enters the OralLens ML workflow. It owns provenance, integrity, extraction, preparation, and leakage controls. Dataset semantics are in the [dataset card](DATASET_CARD.md), while model results are in [training and evaluation](TRAINING.md).

## Admission principles

- use a clearly identified source and version
- verify usage rights and official integrity information before use
- define the source's task role and label semantics before preparation
- preserve source artifacts instead of silently rewriting them
- treat archives, nested paths, filenames, and annotations as untrusted input
- require patient or equivalent group identity for model evaluation
- keep each patient and original/derivative family inside one partition
- keep raw, prepared, and generated data outside version control
- record every exclusion and transformation with a reason
- create protected splits before model selection or evaluation

More files are not automatically more independent evidence. Missing annotations are not confirmed negatives, and incompatible labels must not be merged under one detector class.

## Admitted source

The project uses only the verified AIRC-LABDEN orthodontic-plaque Part 2 material under:

`ml/data/raw/orthodontic_plaque/v3/extracted/part-2/mendeley-dataset-materials_Part_2`

Part 1 remains excluded because its nested 7z archive failed the publisher-provided integrity check. It was not repaired, substituted, or used as partial training data.

The trusted-source configuration is:

`ml/configs/orthodontic_plaque_v3.toml`

Its admission contract records the dataset/source-family identity, publisher, license review, capture context, grouping keys, permitted role, label semantics, limitations, source artifact names, hashes, and extraction limits.

## Prepared outputs

- historical prepared manifest: `ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`
- source-aware prepared manifest: `ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv`

The full release contains 5,160 image rows from 74 patients. The split is patient-aware and fixed:

| Split | Images | Patients | Annotations |
|---|---:|---:|---:|
| Train | 3,834 | 55 | 48,098 |
| Validation | 468 | 7 | 6,070 |
| Test | 858 | 12 | 11,070 |

These image totals include publisher-generated derivatives. The final controlled experiment used 480 genuine training originals, 58 validation originals, and 107 test originals; derivatives were not treated as independent observations.

## Source-label contract

The AIRC region labels mean:

- `0`: plaque absent in the annotated region
- `1`: plaque present in the annotated region

TorchVision class `0` is implicit background. Preparation and loading validate the source vocabulary, and target conversion emits class-1 objects only from source label `1`. Source label `0` boxes remain available for provenance but are not plaque objects.

This corrects the historical v1-v3 all-positive mapping. The corrected baseline and final Experiment 1 model use the source-aware contract; see [training and evaluation](TRAINING.md).

## Filesystem and annotation controls

The pipeline validates:

- declared artifact filenames, hashes, and extraction limits
- archive members and nested extraction paths
- required manifest columns, split values, UTF-8/CSV, and annotation JSON
- finite normalized coordinates, allowed class values, derived box bounds, and positive area
- supported image extensions and decodability where required
- safe relative POSIX paths, root containment, missing files, and symlink policy
- patient and original-family partition isolation

A complete derived-box audit found two rotated-image annotations crossing `[0,1]` by approximately `5e-7`. The converter applies a strict `1e-6` tolerance, clips only tolerated crossings, revalidates area, and rejects larger or degenerate geometry. The source manifest remains unchanged.

## Reproducibility checklist

For any future source or release:

1. record source, version, retrieval date, license, and expected integrity values
2. verify task role, label meaning, annotation method, and negative-label meaning
3. retain raw material under a versioned ignored path
4. validate archives before and during extraction
5. verify image decoding, annotation schema, and coordinate geometry
6. identify patient, source, and original-family grouping keys
7. assign patient- and family-aware partitions before model development
8. audit exact/derived duplicates, paths, labels, exclusions, and transformations
9. run dataset and ML regression tests
10. update this document and the dataset card before making new evidence claims

## Current verification

The most recent complete ML suite after Experiment 1 reports `215 passed, 2 skipped`. Windows symbolic-link cases may skip when the account lacks link privileges; platform-independent containment checks remain active.

## Related documents

- [Dataset card](DATASET_CARD.md)
- [Training and evaluation](TRAINING.md)
- [ML environment](ML_ENVIRONMENT.md)
- [ML guide](../ml/README.md)

