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
from orallens_ml.data.orthodontic_plaque import (
    audit_orthodontic_plaque_source,
    build_orthodontic_plaque_part2_manifest,
    preflight_part2_output_paths,
    validate_part2_manifest_images,
    write_part2_exclusions_csv,
    write_part2_manifest_csv,
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

    plaque_parser = subparsers.add_parser("audit-orthodontic-plaque")
    plaque_parser.add_argument("artifact_id")
    plaque_parser.add_argument("--dataset-root", type=Path, required=True)

    build_plaque_parser = subparsers.add_parser("build-orthodontic-plaque-manifest")
    build_plaque_parser.add_argument("artifact_id")
    build_plaque_parser.add_argument("--dataset-root", type=Path, required=True)
    build_plaque_parser.add_argument("--manifest-output", type=Path, required=True)
    build_plaque_parser.add_argument("--exclusions-output", type=Path, required=True)

    image_parser = subparsers.add_parser("validate-orthodontic-plaque-images")
    image_parser.add_argument("artifact_id")
    image_parser.add_argument("--dataset-root", type=Path, required=True)
    image_parser.add_argument("--manifest", type=Path, required=True)
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
        elif args.command == "summarize-inner-listing":
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
        elif args.command == "audit-orthodontic-plaque":
            report = audit_orthodontic_plaque_source(args.dataset_root)
            payload = {
                "annotation_count": report.annotation_count,
                "artifact_id": artifact.artifact_id,
                "csv_row_count": report.csv_row_count,
                "dataset_root": str(report.dataset_root),
                "image_count": report.image_count,
                "is_valid": report.is_valid,
                "issue_counts": report.issue_counts,
                "issue_examples": {
                    code: [
                        {
                            "code": issue.code,
                            "csv_line": issue.csv_line,
                            "label_line": issue.label_line,
                            "message": issue.message,
                            "severity": issue.severity,
                        }
                        for issue in examples
                    ]
                    for code, examples in report.issue_examples.items()
                },
                "issues": [
                    {
                        "code": issue.code,
                        "csv_line": issue.csv_line,
                        "label_line": issue.label_line,
                        "message": issue.message,
                        "severity": issue.severity,
                    }
                    for issue in report.issues
                ],
                "label_file_count": report.label_file_count,
                "patient_count": report.patient_count,
                "split_counts": report.split_counts,
                "status": "orthodontic-plaque-source-audited",
                "tooth_position_ids": list(report.tooth_position_ids),
            }
        elif args.command == "build-orthodontic-plaque-manifest":
            samples, report = build_orthodontic_plaque_part2_manifest(args.dataset_root)
            preflight_part2_output_paths(args.manifest_output, args.exclusions_output)
            write_part2_manifest_csv(samples, args.manifest_output)
            write_part2_exclusions_csv(report.exclusions, args.exclusions_output)
            payload = {
                "artifact_id": artifact.artifact_id,
                "excluded_samples": report.excluded_samples,
                "exclusions_output": str(args.exclusions_output),
                "filtered_annotation_count": report.filtered_annotation_count,
                "included_samples": report.included_samples,
                "issue_counts": report.issue_counts,
                "manifest_output": str(args.manifest_output),
                "source_csv_rows": report.source_csv_rows,
                "split_counts": report.split_counts,
                "status": "orthodontic-plaque-manifest-built",
                "valid_annotation_count": report.valid_annotation_count,
            }
        else:
            report = validate_part2_manifest_images(args.dataset_root, args.manifest)
            payload = {
                "artifact_id": artifact.artifact_id,
                "dataset_root": str(report.dataset_root),
                "decoded_images": report.decoded_images,
                "is_valid": report.is_valid,
                "issue_counts": report.issue_counts,
                "issues": [
                    {
                        "code": issue.code,
                        "manifest_line": issue.manifest_line,
                        "message": issue.message,
                        "sample_id": issue.sample_id,
                    }
                    for issue in report.issues
                ],
                "manifest_path": str(report.manifest_path),
                "mode_counts": report.mode_counts,
                "size_counts": report.size_counts,
                "status": "orthodontic-plaque-images-validated",
                "total_records": report.total_records,
            }
    except (AcquisitionError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.command == "audit-orthodontic-plaque" and not payload["is_valid"]:
        return 2
    if args.command == "validate-orthodontic-plaque-images" and not payload["is_valid"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
