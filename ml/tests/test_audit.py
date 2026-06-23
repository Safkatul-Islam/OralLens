from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from orallens_ml.data.audit import audit_dataset
from orallens_ml.data.manifest import SampleRecord


def record(
    sample_id: str,
    patient_id: str,
    relative_path: str,
    *,
    source_id: str | None = None,
    variant: str = "original",
    label: str = "2",
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        patient_id=patient_id,
        relative_path=PurePosixPath(relative_path),
        label=label,
        source_sample_id=source_id or sample_id,
        variant=variant,
        manifest_line=2,
    )


def test_audit_accepts_existing_supported_images(tmp_path: Path) -> None:
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    image.write_bytes(b"synthetic-test-image")

    report = audit_dataset([record("s1", "p1", "images/sample.jpg")], tmp_path)

    assert report.is_valid
    assert report.total_records == 1
    assert report.patient_count == 1
    assert report.label_counts == {"2": 1}
    assert report.issues == ()


def test_audit_reports_duplicate_and_missing_records(tmp_path: Path) -> None:
    records = [
        record("same", "p1", "images/missing.jpg", source_id="source"),
        record("same", "p1", "images/missing.jpg", source_id="source"),
    ]

    report = audit_dataset(records, tmp_path)
    codes = {issue.code for issue in report.issues}

    assert not report.is_valid
    assert "duplicate_sample_id" in codes
    assert "duplicate_relative_path" in codes
    assert "duplicate_source_variant" in codes
    assert "missing_image" in codes


def test_audit_reports_source_metadata_conflicts(tmp_path: Path) -> None:
    records = [
        record("s1", "p1", "one.jpg", source_id="shared", label="2"),
        record("s2", "p2", "two.jpg", source_id="shared", label="3"),
    ]

    report = audit_dataset(records, tmp_path)
    codes = {issue.code for issue in report.issues}

    assert "source_crosses_patients" in codes
    assert "source_label_conflict" in codes


def test_audit_rejects_path_that_escapes_dataset_root(tmp_path: Path) -> None:
    report = audit_dataset(
        [record("s1", "p1", "../outside-audit-image.jpg")],
        tmp_path,
    )

    assert "path_escapes_dataset_root" in {issue.code for issue in report.issues}


def test_audit_rejects_symlink_that_escapes_dataset_root(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-audit-image.jpg"
    outside.write_bytes(b"outside")
    link = tmp_path / "linked.jpg"
    try:
        try:
            link.symlink_to(outside)
        except OSError as exc:
            if exc.winerror == 1314:
                pytest.skip("Windows account cannot create symbolic links")
            raise
        report = audit_dataset([record("s1", "p1", "linked.jpg")], tmp_path)
        assert "path_escapes_dataset_root" in {issue.code for issue in report.issues}
    finally:
        link.unlink(missing_ok=True)
        outside.unlink(missing_ok=True)


def test_audit_reports_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("not an image", encoding="utf-8")

    report = audit_dataset([record("s1", "p1", "sample.txt")], tmp_path)

    assert "unsupported_image_extension" in {issue.code for issue in report.issues}
