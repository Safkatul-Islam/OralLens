# Dataset Card: Orthodontic Plaque Severity

## Status

Candidate dataset selected. The archives have not been downloaded, and their
internal file structure has not yet been audited. This card records verified
publisher metadata and the controls required before training.

## Dataset

**Title:** A new multi-modal dataset for Dental Plaque Diagnosis of Patients
With Fixed Labial Orthodontic Appliances

Official records:

- [Part 1](https://data.mendeley.com/datasets/g8yhdvgjy2/3)
- [Part 2](https://data.mendeley.com/datasets/xjs4bfgzj5/3)

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
- Every manifest path resolves inside the dataset root.
- Every referenced image exists and uses an approved image extension.
- Sample IDs and paths are unique.
- One source image never crosses patients or labels.
- Every patient and source family belongs to exactly one split.
- Class and patient distributions are reviewed before selecting metrics or loss.
- Corrupt and unreadable images are identified once image decoding is added.

## Known Limitations

- Archive layout and detailed metadata schema remain unverified until download.
- The population is clinically narrow.
- Publisher-generated augmentations may not reflect real-world acquisition shift.
- Label agreement statistics and subgroup coverage require further audit.
- No performance claim is valid until the held-out patient test evaluation.
