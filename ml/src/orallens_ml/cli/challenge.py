"""Read-only CLI for validating real-world challenge manifests."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orallens_ml.data.challenge_manifest import (
    ChallengeManifestError,
    audit_challenge_manifest,
    load_challenge_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an OralLens real-world challenge manifest."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate-manifest")
    validate_parser.add_argument("--manifest", type=Path, required=True)
    validate_parser.add_argument("--dataset-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        records = load_challenge_manifest(args.manifest)
        report = audit_challenge_manifest(records, args.dataset_root)
    except (ChallengeManifestError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = {
        "challenge_group_counts": report.challenge_group_counts,
        "dataset_root": str(args.dataset_root),
        "expected_handling_counts": report.expected_handling_counts,
        "is_valid": report.is_valid,
        "issues": [
            {
                "code": issue.code,
                "message": issue.message,
                "sample_id": issue.sample_id,
            }
            for issue in report.issues
        ],
        "manifest_path": str(args.manifest),
        "partition_counts": report.partition_counts,
        "source_group_count": report.source_group_count,
        "status": "challenge-manifest-validated",
        "total_records": report.total_records,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if report.is_valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
