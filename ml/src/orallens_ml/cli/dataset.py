"""CLI for verified dataset acquisition and outer-archive handling."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orallens_ml.data.acquisition import (
    AcquisitionError,
    artifact_path,
    finalize_download,
    load_release_config,
    verify_download,
)
from orallens_ml.data.archive import (
    extract_outer_zip,
    inspect_zip,
    summarize_7zip_slt_listing,
    summarize_inner_listing,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify OralLens dataset downloads before extraction."
    )
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("verify", "finalize", "inspect"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("artifact_id")
        command_parser.add_argument("--downloads-dir", type=Path, required=True)

    verify_parser = subparsers.choices["verify"]
    verify_parser.add_argument(
        "--finalized",
        action="store_true",
        help="Verify the final .zip instead of the resumable .zip.part file.",
    )

    extract_parser = subparsers.add_parser("extract-outer")
    extract_parser.add_argument("artifact_id")
    extract_parser.add_argument("--downloads-dir", type=Path, required=True)
    extract_parser.add_argument("--destination-dir", type=Path, required=True)

    inner_parser = subparsers.add_parser("summarize-inner-listing")
    inner_parser.add_argument("artifact_id")
    inner_parser.add_argument("--listing-file", type=Path, required=True)
    inner_parser.add_argument(
        "--format",
        choices=("plain", "7z-slt"),
        default="plain",
        help="Input listing format. Use 7z-slt for saved `7z l -slt` output.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        release = load_release_config(args.config)
        artifact = release.artifact(args.artifact_id)
        if args.command == "verify":
            path = artifact_path(
                args.downloads_dir,
                artifact,
                finalized=args.finalized,
            )
            verified = verify_download(path, artifact)
            payload = {
                "artifact_id": artifact.artifact_id,
                "path": str(verified.path),
                "sha256": verified.sha256,
                "size_bytes": verified.size_bytes,
                "status": "verified",
            }
        elif args.command == "finalize":
            verified = finalize_download(args.downloads_dir, artifact)
            payload = {
                "artifact_id": artifact.artifact_id,
                "path": str(verified.path),
                "sha256": verified.sha256,
                "size_bytes": verified.size_bytes,
                "status": "finalized",
            }
        elif args.command == "inspect":
            path = artifact_path(args.downloads_dir, artifact, finalized=True)
            verify_download(path, artifact)
            report = inspect_zip(path, artifact)
            payload = {
                "artifact_id": artifact.artifact_id,
                "entries": [
                    {
                        "name": entry.name,
                        "compressed_bytes": entry.compressed_bytes,
                        "uncompressed_bytes": entry.uncompressed_bytes,
                        "compression_ratio": entry.compression_ratio,
                    }
                    for entry in report.entries
                ],
                "status": "safe-to-extract-outer",
                "total_uncompressed_bytes": report.total_uncompressed_bytes,
            }
        elif args.command == "extract-outer":
            path = artifact_path(args.downloads_dir, artifact, finalized=True)
            verify_download(path, artifact)
            extracted = extract_outer_zip(path, args.destination_dir, artifact)
            payload = {
                "artifact_id": artifact.artifact_id,
                "paths": [str(path) for path in extracted],
                "status": "outer-extracted",
            }
        else:
            if args.format == "7z-slt":
                report = summarize_7zip_slt_listing(args.listing_file)
            else:
                report = summarize_inner_listing(args.listing_file)
            payload = {
                "directory_count": report.directory_count,
                "entry_count": report.entry_count,
                "extension_counts": report.extension_counts,
                "file_count": report.file_count,
                "image_file_count": report.image_file_count,
                "image_patient_count": report.image_patient_count,
                "image_patients_without_labels": list(
                    report.image_patients_without_labels
                ),
                "images_without_labels_count": report.images_without_labels_count,
                "images_without_labels_examples": list(
                    report.images_without_labels_examples
                ),
                "label_file_count": report.label_file_count,
                "label_patient_count": report.label_patient_count,
                "label_patients_without_images": list(
                    report.label_patients_without_images
                ),
                "labels_without_images_count": report.labels_without_images_count,
                "labels_without_images_examples": list(
                    report.labels_without_images_examples
                ),
                "listing_path": str(report.listing_path),
                "patient_case_conflicts": {
                    key: list(value)
                    for key, value in report.patient_case_conflicts.items()
                },
                "root_entries": list(report.root_entries),
                "status": "inner-listing-summarized",
            }
    except (AcquisitionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
