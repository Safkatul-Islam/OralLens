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
    orthodontic_plaque_v3_part_2.zip.part
  nested/
    A new multi-modal dataset for Dental Plaque Diagno/
      mendeley-dataset-materials_Part_1.7z
    mendeley-dataset-materials_Part_2.7z
```

The entire `ml/data/raw/` tree is ignored by git.

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
8. List and review each nested 7z with bsdtar before inner extraction.
9. Summarize the saved listing with the project CLI to detect unsafe paths,
   duplicate member names, extension counts, patient coverage, image-label
   pairing gaps, and patient ID casing inconsistencies.
10. Test each nested 7z with official 7-Zip before trusting extracted files.

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

Windows `tar.exe` is bsdtar 3.8.4 backed by libarchive 3.8.4. The official
libarchive format documentation lists 7-Zip read/write support and states that
bsdtar enables libarchive formats by default.

The nested archives must first be listed with `tar.exe -tf`. Save the listing
under the ignored raw-data tree, then summarize it with the project CLI before
any inner extraction command is approved:

```powershell
tar.exe -tf "ml\data\raw\orthodontic_plaque\v3\nested\A new multi-modal dataset for Dental Plaque Diagno\mendeley-dataset-materials_Part_1.7z" > "ml\data\raw\orthodontic_plaque\v3\nested\part-1-inner-listing.txt"
ml\.venv\Scripts\python.exe -B -m orallens_ml.cli.dataset --config "ml\configs\orthodontic_plaque_v3.toml" summarize-inner-listing part-1 --listing-file "ml\data\raw\orthodontic_plaque\v3\nested\part-1-inner-listing.txt"
```

Their member names, directory structure, metadata files, patient IDs, extension
counts, and image-label pairing gaps must be reviewed before any inner
extraction command is approved. The current tooling intentionally does not
automate inner extraction.

Before trusting extracted files, run an official 7-Zip integrity test:

```powershell
& "C:\Program Files\7-Zip\7z.exe" t "ml\data\raw\orthodontic_plaque\v3\nested\A new multi-modal dataset for Dental Plaque Diagno\mendeley-dataset-materials_Part_1.7z"
```

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
extraction, inner-listing summarization, and CLI behavior are covered by the
complete ML test suite. The latest focused archive result is 22 passed tests
using a project-local pytest base temp to avoid Windows temp permission issues.
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
must be treated as unusable. Part 2 has not yet been downloaded.
