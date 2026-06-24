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
    orthodontic_plaque_v3_part_1.zip.part
    orthodontic_plaque_v3_part_2.zip.part
  nested/
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

Repeat with `part-2` only after Part 1 completes successfully.

## Inner 7z Boundary

Windows `tar.exe` is bsdtar 3.8.4 backed by libarchive 3.8.4. The official
libarchive format documentation lists 7-Zip read/write support and states that
bsdtar enables libarchive formats by default.

The nested archives must first be listed with `tar.exe -tf`. Their member names,
reported sizes, directory structure, and metadata files must be reviewed before
any inner extraction command is approved. The current tooling intentionally does
not automate inner extraction.

## Recovery

- Interrupted download: retain the `.part` file and rerun the same curl command.
- Hash mismatch: do not rename or extract; remove the bad partial file only after
  explicit review and approval, then restart the download.
- Unexpected ZIP member: stop and compare the current publisher record and
  configuration. Never weaken the policy merely to make extraction proceed.
- Existing extraction target: stop and inspect it. The extractor never overwrites.

## Implementation Verification

The acquisition configuration, hashing, finalization, ZIP inspection, bounded
extraction, and CLI behavior are covered by the complete ML test suite. The
latest result is 45 passed tests and one capability-gated Windows symlink skip.
The platform-independent path-containment tests still pass on this workstation.

This result verifies the acquisition tooling, not the dataset. No publisher
archive has been downloaded, hashed locally, inspected, or extracted yet.
