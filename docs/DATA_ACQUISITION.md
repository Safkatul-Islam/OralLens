# Dataset Acquisition

## Scope

This workflow acquires Version 3 of the two-part Mendeley Data release for
orthodontic plaque-severity screening. It verifies publisher-provided hashes and
inspects each archive layer before extraction. It never commits dataset files.

Acquisition does not imply permission to make diagnostic claims. The dataset is
licensed CC BY 4.0 and represents patients with fixed labial orthodontic
appliances.

## Trusted Sources

| Artifact | Record | Generated ZIP SHA-256 |
| --- | --- | --- |
| Part 1 | `https://data.mendeley.com/datasets/g8yhdvgjy2/3` | `9ab308d919bae0ea6104e9f4c96336be19aa4841c830b8fca1db2f92e3ebe618` |
| Part 2 | `https://data.mendeley.com/datasets/xjs4bfgzj5/3` | `990691d1c01e8c83be820df22fa38520bc085e3d83efa0a96b09fb8787a49a85` |

The hashes apply to Mendeley's generated "Download All" ZIP files. Each ZIP is
expected to contain exactly one publisher 7z file. They do not establish an
independent hash for the nested 7z file.

## Local Layout

```text
ml/data/raw/orthodontic_plaque/v3/
  downloads/
    orthodontic_plaque_v3_part_1.zip
    orthodontic_plaque_v3_part_2.zip
  nested/
    A new multi-modal dataset for Dental Plaque Diagno/
      mendeley-dataset-materials_Part_1.7z
      mendeley-dataset-materials_Part_2.7z
    part-1-inner-listing.txt
    part-2-inner-listing.slt
```

The entire `ml/data/raw/` tree is ignored by git.

Generated Part 2 manifest artifacts are written under the ignored prepared-data
tree:

```text
ml/data/prepared/orthodontic_plaque/v3/part-2/
  manifest.csv
  exclusions.csv
```

## Security Gates

1. Download only the HTTPS URLs in the reviewed TOML configuration.
2. Keep incomplete data under `.zip.part` names.
3. Reject empty, oversized, symlinked, changed-during-read, or hash-mismatched files.
4. Rename a partial file only after SHA-256 verification succeeds.
5. Inspect ZIP members before extraction.
6. Reject absolute paths, traversal, Windows alternate paths, duplicates, links,
   encryption, unsupported compression, unexpected members, and excessive sizes
   or compression ratios.
7. Extract expected members with bounded streaming copies and no overwrite.
8. List and review each nested 7z with official 7-Zip technical-listing output
   before inner extraction.
9. Summarize the saved listing with the project CLI to detect unsafe paths,
   duplicate member names, extension counts, patient coverage, image-label
   pairing gaps, and patient ID casing inconsistencies.
10. Test each nested 7z with official 7-Zip before trusting extracted files.
11. Build generated manifests only through the project CLI. Invalid annotation
    rows are filtered, samples with no valid annotations are excluded, and the
    exclusion report is preserved. Bounding boxes are never silently repaired.
12. Decode every retained image with Pillow before training.

## Resumable Downloads

Run one download at a time from the repository root after command approval.
Manual reruns resume the existing `.part` file. Redirects are limited and may
use HTTPS only.

Part 1:

```powershell
curl.exe --fail --location --max-redirs 5 --max-filesize 11000000000 --proto "=https" --proto-redir "=https" --continue-at - --output "ml\data\raw\orthodontic_plaque\v3\downloads\orthodontic_plaque_v3_part_1.zip.part" "https://data.mendeley.com/public-api/zip/g8yhdvgjy2/download/3"
```

Part 2:

```powershell
curl.exe --fail --location --max-redirs 5 --max-filesize 11000000000 --proto "=https" --proto-redir "=https" --continue-at - --output "ml\data\raw\orthodontic_plaque\v3\downloads\orthodontic_plaque_v3_part_2.zip.part" "https://data.mendeley.com/public-api/zip/xjs4bfgzj5/download/3"
```

Automatic retries are intentionally omitted. If a transfer fails, inspect the
error and rerun the same command to resume safely.

## Verification and Outer Extraction

Verify and finalize one artifact at a time:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" verify part-1 --downloads-dir "ml\data\raw\orthodontic_plaque\v3\downloads"
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" finalize part-1 --downloads-dir "ml\data\raw\orthodontic_plaque\v3\downloads"
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" inspect part-1 --downloads-dir "ml\data\raw\orthodontic_plaque\v3\downloads"
```

Only after inspection reports `safe-to-extract-outer`:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" extract-outer part-1 --downloads-dir "ml\data\raw\orthodontic_plaque\v3\downloads" --destination-dir "ml\data\raw\orthodontic_plaque\v3\nested"
```

Repeat with `part-2` after each previous artifact has either completed all
integrity gates or has been documented as unusable.

## Inner 7z Boundary

The nested archives must first be tested and listed with official 7-Zip. Save
the raw `7z l -slt` output under the ignored raw-data tree, then summarize it
with the project CLI before any inner extraction command is approved:

```powershell
& "C:\Program Files\7-Zip\7z.exe" t "ml\data\raw\orthodontic_plaque\v3\nested\A new multi-modal dataset for Dental Plaque Diagno\mendeley-dataset-materials_Part_2.7z"
& "C:\Program Files\7-Zip\7z.exe" l -slt "ml\data\raw\orthodontic_plaque\v3\nested\A new multi-modal dataset for Dental Plaque Diagno\mendeley-dataset-materials_Part_2.7z" | Set-Content -Encoding utf8 "ml\data\raw\orthodontic_plaque\v3\nested\part-2-inner-listing.slt"
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" summarize-inner-listing part-2 --listing-file "ml\data\raw\orthodontic_plaque\v3\nested\part-2-inner-listing.slt" --format 7z-slt
```

Their member names, directory structure, metadata files, patient IDs, extension
counts, and image-label pairing gaps must be reviewed before any inner
extraction command is approved. The current tooling intentionally does not
automate inner extraction.

## Part 2 Manifest Generation

After Part 2 passes archive and source audits, generate the retained manifest
and exclusion report:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" build-orthodontic-plaque-manifest part-2 --dataset-root "ml\data\raw\orthodontic_plaque\v3\extracted\part-2\mendeley-dataset-materials_Part_2" --manifest-output "ml\data\prepared\orthodontic_plaque\v3\part-2\manifest.csv" --exclusions-output "ml\data\prepared\orthodontic_plaque\v3\part-2\exclusions.csv"
```

Current generated Part 2 output:

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

Exclusion reasons:

| Reason | Count |
| --- | ---: |
| Empty label file | 14 |
| Non-positive bounding-box size | 42 |

Validate retained images before modeling:

```powershell
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" validate-orthodontic-plaque-images part-2 --dataset-root "ml\data\raw\orthodontic_plaque\v3\extracted\part-2\mendeley-dataset-materials_Part_2" --manifest "ml\data\prepared\orthodontic_plaque\v3\part-2\manifest.csv"
```

Current image validation result: all 5,160 retained images decode successfully
with Pillow. All retained images are RGB. Image sizes are 4,914 at `6240x4160`
and 246 at `2560x1920`.

## Recovery

- Interrupted download: retain the `.part` file and rerun the same curl command.
- Hash mismatch: do not rename or extract; remove the bad partial file only after
  explicit review and approval, then restart the download.
- Unexpected ZIP member: stop and compare the current publisher record and
  configuration. Never weaken the policy merely to make extraction proceed.
- Existing extraction target: stop and inspect it. The extractor never overwrites.
- Inner 7z CRC failure: do not use partial extraction output for training.
  Keep the failure documented and move to the next artifact or publisher review.

## Implementation Verification

The acquisition configuration, hashing, finalization, ZIP inspection, bounded
extraction, inner-listing summarization, source audit, manifest generation,
image validation, and CLI behavior are covered by the complete ML test suite.
The latest complete ML suite result is 82 passed tests and 1 skipped
Windows-symlink-permission test using a project-local pytest base temp.
The platform-independent path-containment tests still pass on this workstation.

Part 1 has been downloaded from the official Mendeley URL, verified locally
against SHA-256
`9ab308d919bae0ea6104e9f4c96336be19aa4841c830b8fca1db2f92e3ebe618`, and
finalized as `orthodontic_plaque_v3_part_1.zip`. Initial inspection showed that
Mendeley's generated ZIP wraps the publisher 7z under
`A new multi-modal dataset for Dental Plaque Diagno/`; the acquisition config
pins that exact member path. The Part 1 outer ZIP has been extracted, and the
nested 7z listing completed successfully with 11,089 entries. The listing showed
top-level `clinical_records/`, `data/images/`, `data/labels/`,
`legal_documents/`, and `metadata/` directories, plus a patient ID casing
inconsistency between `Patient0033` and `patient0033` that must be handled if a
clean replacement artifact becomes available.

Part 1 is not trusted for training. Windows `tar.exe` extraction and official
7-Zip 26.02 both failed on the same inner member:
`mendeley-dataset-materials_Part_1\data\images\patient0002\patient0002_20251028_bottom-left_rotate-right-15.jpg`.
The official 7-Zip integrity test reported `ERROR: CRC Failed` for that file.
Any partial extraction under `ml/data/raw/orthodontic_plaque/v3/extracted/part-1`
must be treated as unusable.

Part 2 has been downloaded from the official Mendeley URL, verified locally
against SHA-256
`990691d1c01e8c83be820df22fa38520bc085e3d83efa0a96b09fb8787a49a85`, and
finalized as `orthodontic_plaque_v3_part_2.zip`. Mendeley's generated ZIP uses
the same wrapper directory pattern as Part 1:
`A new multi-modal dataset for Dental Plaque Diagno/mendeley-dataset-materials_Part_2.7z`.
The Part 2 outer ZIP has been inspected and extracted. Official 7-Zip 26.02
reported `Everything is Ok` for the nested Part 2 7z. A raw `7z l -slt`
technical listing was saved as `part-2-inner-listing.slt` and summarized by the
project CLI with `--format 7z-slt`. The summary contains 10,885 entries, 158
directories, 10,727 files, 5,174 `.jpg` images, 5,174 `.txt` labels, 74 image
patients, 74 label patients, no image-label pairing gaps, and no patient ID
casing conflicts.
