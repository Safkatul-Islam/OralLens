from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath

from PIL import Image
import pytest

from orallens_ml.data.challenge_manifest import (
    CHALLENGE_MANIFEST_COLUMNS,
    CHALLENGE_MANIFEST_SCHEMA_VERSION,
    ChallengeManifestError,
    ChallengeSample,
    audit_challenge_manifest,
    load_challenge_manifest,
)


def valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "manifest_schema_version": str(CHALLENGE_MANIFEST_SCHEMA_VERSION),
        "sample_id": "sample-001",
        "image_relative_path": "images/sample-001.jpg",
        "source_id": "project-generated",
        "source_group_id": "person-001",
        "derivative_group_id": "capture-001",
        "variant": "original",
        "license_or_permission_id": "consent-001",
        "partition": "development",
        "challenge_group": "consumer_supported_oral",
        "expected_handling": "supported",
        "expected_reason_codes": "[]",
        "view": "frontal",
        "device_context": "phone",
        "quality_flags": "[]",
        "condition_reference": "not_applicable",
        "notes": "Rights record is stored outside version control.",
    }
    row.update(overrides)
    return row


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHALLENGE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(path)


def test_load_and_audit_valid_challenge_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [valid_row()])
    write_image(tmp_path / "images" / "sample-001.jpg")

    records = load_challenge_manifest(manifest)
    report = audit_challenge_manifest(records, tmp_path)

    assert len(records) == 1
    assert records[0].image_relative_path.as_posix() == "images/sample-001.jpg"
    assert records[0].expected_reason_codes == ()
    assert records[0].manifest_line == 2
    assert report.is_valid
    assert report.total_records == 1
    assert report.source_group_count == 1
    assert report.partition_counts == {"development": 1}
    assert report.challenge_group_counts == {"consumer_supported_oral": 1}


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"manifest_schema_version": "2"}, "unsupported challenge manifest schema"),
        ({"license_or_permission_id": ""}, "must not be empty"),
        ({"image_relative_path": "../outside.jpg"}, "unsafe path segment"),
        ({"image_relative_path": "images/sample.webp"}, "unsupported challenge image"),
        ({"challenge_group": "unknown"}, "unsupported challenge_group"),
        ({"expected_handling": "unknown"}, "unsupported expected_handling"),
        (
            {"expected_reason_codes": '["image_too_small"]'},
            "supported samples must not expect reason codes",
        ),
        (
            {"expected_handling": "unsupported"},
            "unsupported samples must expect a reason code",
        ),
        ({"expected_reason_codes": "not-json"}, "must be a JSON array"),
        ({"quality_flags": '["unknown"]'}, "unknown quality_flags"),
    ),
)
def test_load_rejects_invalid_challenge_rows(
    tmp_path: Path,
    overrides: dict[str, str],
    message: str,
) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [valid_row(**overrides)])

    with pytest.raises(ChallengeManifestError, match=message):
        load_challenge_manifest(manifest)


def test_load_rejects_unknown_manifest_column(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    fieldnames = (*CHALLENGE_MANIFEST_COLUMNS, "unexpected")
    row = valid_row()
    row["unexpected"] = "value"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)

    with pytest.raises(ChallengeManifestError, match="unknown columns"):
        load_challenge_manifest(manifest)


def test_load_rejects_surplus_row_values(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    row = valid_row()
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHALLENGE_MANIFEST_COLUMNS)
        writer.writerow([row[column] for column in CHALLENGE_MANIFEST_COLUMNS] + ["extra"])

    with pytest.raises(ChallengeManifestError, match="unexpected values"):
        load_challenge_manifest(manifest)


def test_audit_reports_duplicates_and_partition_leakage(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [valid_row()])
    first = load_challenge_manifest(manifest)[0]
    second = replace(
        first,
        sample_id="sample-002",
        image_relative_path=PurePosixPath("images/sample-002.jpg"),
        partition="locked",
        variant="compressed",
    )

    report = audit_challenge_manifest((first, first, second), tmp_path)
    codes = {issue.code for issue in report.issues}

    assert "duplicate_sample_id" in codes
    assert "duplicate_image_path" in codes
    assert "duplicate_derivative_variant" in codes
    assert "source_group_partition_leakage" in codes
    assert "derivative_group_partition_leakage" in codes


def test_audit_reports_missing_image(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(manifest, [valid_row()])

    report = audit_challenge_manifest(load_challenge_manifest(manifest), tmp_path)

    assert not report.is_valid
    assert {issue.code for issue in report.issues} == {"missing_image"}


def test_audit_rejects_symlinked_image_path(tmp_path: Path) -> None:
    target = tmp_path / "target.jpg"
    write_image(target)
    link = tmp_path / "images" / "sample-001.jpg"
    link.parent.mkdir()
    try:
        try:
            link.symlink_to(target)
        except OSError as exc:
            if exc.winerror == 1314:
                pytest.skip("Windows account cannot create symbolic links")
            raise
        manifest = tmp_path / "manifest.csv"
        write_manifest(manifest, [valid_row()])

        report = audit_challenge_manifest(
            load_challenge_manifest(manifest), tmp_path
        )

        assert {issue.code for issue in report.issues} == {"symlinked_image_path"}
    finally:
        link.unlink(missing_ok=True)


def test_unsupported_record_accepts_structured_reason_codes(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_manifest(
        manifest,
        [
            valid_row(
                expected_handling="unsupported",
                expected_reason_codes=json.dumps(
                    ["image_too_small", "image_low_contrast"]
                ),
                challenge_group="technical_quality_failure",
                quality_flags=json.dumps(["framing"]),
            )
        ],
    )

    record = load_challenge_manifest(manifest)[0]

    assert record.expected_reason_codes == (
        "image_too_small",
        "image_low_contrast",
    )
    assert record.quality_flags == ("framing",)
