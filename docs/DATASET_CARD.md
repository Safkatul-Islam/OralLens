# Dataset Card: Orthodontic Plaque Severity

## Status

Candidate dataset selected. Both official Mendeley generated ZIP downloads have
been downloaded, verified by SHA-256, finalized locally, and outer-extracted.
Official 7-Zip integrity testing fails inside the Part 1 7z payload, so Part 1
is not trusted for training. Part 2 passes official 7-Zip integrity testing and
its technical listing, source labels, generated manifest, exclusions, and image
decoding have been audited. This card records verified publisher metadata, the
integrity blocker, the usable current scope, and the controls required before
training.

## Dataset

**Title:** A new multi-modal dataset for Dental Plaque Diagnosis of Patients
With Fixed Labial Orthodontic Appliances

Official records:

- [Part 1](https://data.mendeley.com/datasets/g8yhdvgjy2/3)
- [Part 2](https://data.mendeley.com/datasets/xjs4bfgzj5/3)

Verified Version 3 artifacts:

| Part | Publisher file | Generated ZIP SHA-256 |
| --- | --- | --- |
| 1 | `A new multi-modal dataset for Dental Plaque Diagno/mendeley-dataset-materials_Part_1.7z` | `9ab308d919bae0ea6104e9f4c96336be19aa4841c830b8fca1db2f92e3ebe618` |
| 2 | `A new multi-modal dataset for Dental Plaque Diagno/mendeley-dataset-materials_Part_2.7z` | `990691d1c01e8c83be820df22fa38520bc085e3d83efa0a96b09fb8787a49a85` |

These publisher checksums apply to Mendeley's generated ZIP downloads that
contain the listed 7z files. The paths reflect the real member paths observed
after verifying the downloaded ZIPs against the official hashes.

The publisher reports 148 patients, more than 10,000 intraoral images, nine
standardized capture angles per patient, and plaque-severity labels reviewed by
two dentists. The two archives are approximately 18.81 GB combined.

## License

The Mendeley records identify the dataset license as CC BY 4.0. Attribution must
be preserved in derived documentation and published results. The license and
record version must be rechecked when the data is downloaded.

## Intended ML Task

Four-class ordinal image classification:

| Internal label | Published severity group |
| --- | --- |
| `0-1` | Scores 0 through 1 |
| `2` | Score 2 |
| `3` | Score 3 |
| `4` | Score 4 |

The first baseline will screen visible plaque severity. It will not diagnose a
disease, recommend treatment, or replace assessment by a dental professional.

## Population and Scope

The source population consists of patients with fixed labial orthodontic
appliances. Results must not be represented as validated for the general
population, other imaging protocols, or other dental conditions.

## Augmentation and Leakage Risk

The publisher reports noise-added images and six augmentation methods, producing
72 images per patient. Random image-level splitting would allow near-duplicate
views of the same patient or source image into multiple splits and inflate model
performance.

Required controls:

- Split by `patient_id`, never by image path.
- Keep every augmentation and source variant in the same split.
- Preserve a final patient-level test set that is untouched during development.
- Measure duplicate and metadata conflicts before training.
- Report patient-level confidence intervals during final evaluation.

## Current Usable Manifest

The current training scope is Part 2 only. Part 1 remains excluded because its
nested 7z fails integrity testing.

Generated files:

- `ml/data/prepared/orthodontic_plaque/v3/part-2/manifest.csv`
- `ml/data/prepared/orthodontic_plaque/v3/part-2/exclusions.csv`

Part 2 generated manifest summary:

| Metric | Count |
| --- | ---: |
| Source CSV rows | 5,174 |
| Included samples | 5,160 |
| Excluded samples | 14 |
| Retained valid annotations | 65,238 |
| Filtered invalid annotations | 42 |
| Train samples | 3,834 |
| Validation samples | 468 |
| Test samples | 858 |

The exclusion policy filters invalid annotation rows and excludes samples that
have no valid annotations after filtering. It does not clamp, repair, or infer
bounding boxes. Current exclusion reasons are 14 empty label files and 42
non-positive bounding-box rows.

All 5,160 retained images decode successfully with Pillow. All retained images
are RGB; 4,914 images are `6240x4160` and 246 images are `2560x1920`.

## Normalized Manifest Contract

Raw publisher metadata will be adapted into a UTF-8 CSV manifest. Paths use `/`
separators and are relative to a separately configured dataset root.

| Column | Meaning |
| --- | --- |
| `sample_id` | Unique ID for this image record |
| `patient_id` | Group ID used to prevent patient leakage |
| `relative_path` | Dataset-root-relative image path |
| `label` | One of `0-1`, `2`, `3`, or `4` |
| `source_sample_id` | ID shared by an original image and its variants |
| `variant` | Transformation name, such as `original` |

Extra source metadata may be retained as additional columns. The required fields
must never be inferred silently when the publisher metadata is ambiguous.

## Pre-Training Acceptance Checks

- Dataset version and license match the official records.
- Every nested archive passes an official 7-Zip integrity test before extracted
  files are trusted.
- Every manifest path resolves inside the dataset root.
- Every referenced image exists and uses an approved image extension.
- Sample IDs and paths are unique.
- Patient ID casing is normalized and reviewed because Part 1 includes at least
  one casing inconsistency between image and label directories.
- One source image never crosses patients or labels.
- Every patient and source family belongs to exactly one split.
- Class and patient distributions are reviewed before selecting metrics or loss.
- Corrupt and unreadable images are identified before training.

See [`DATA_ACQUISITION.md`](DATA_ACQUISITION.md) for the reviewed download,
checksum, archive-inspection, and extraction workflow.

## Known Limitations

- Part 1 outer ZIP matches the publisher SHA-256, but the inner 7z fails
  official 7-Zip 26.02 integrity testing with a CRC failure at
  `mendeley-dataset-materials_Part_1\data\images\patient0002\patient0002_20251028_bottom-left_rotate-right-15.jpg`.
- Any partial Part 1 extraction must not be used for training, validation,
  testing, manifest generation, or demos.
- The Part 1 listing shows a patient ID casing inconsistency that must be
  handled explicitly if a clean replacement artifact becomes available.
- Part 2 is currently the only integrity-verified usable archive. Its listing
  contains 5,174 images, 5,174 labels, 74 image patients, 74 label patients, no
  image-label pairing gaps, and no patient ID casing conflicts.
- The generated Part 2 manifest excludes 14 samples with empty label files and
  filters 42 invalid annotation rows with non-positive bounding-box sizes.
- Training scope must remain Part 2-only unless Part 1 is replaced by a clean
  archive or the publisher resolves the CRC failure.
- The population is clinically narrow.
- Publisher-generated augmentations may not reflect real-world acquisition shift.
- Label agreement statistics and subgroup coverage require further audit.
- No performance claim is valid until the held-out patient test evaluation.
