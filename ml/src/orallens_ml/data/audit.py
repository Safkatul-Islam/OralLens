"""Read-only dataset auditing for normalized image manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from orallens_ml.data.manifest import SampleRecord

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png", ".webp"})
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """One actionable defect discovered during a dataset audit."""

    code: str
    severity: Severity
    message: str
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Immutable summary of a completed dataset audit."""

    total_records: int
    patient_count: int
    label_counts: dict[str, int]
    issues: tuple[AuditIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)


def audit_dataset(
    records: Iterable[SampleRecord],
    dataset_root: Path,
) -> AuditReport:
    """Audit records without changing the manifest or dataset files."""

    samples = tuple(records)
    root = Path(dataset_root)
    if not root.is_dir():
        raise ValueError(f"Dataset root is not a directory: {root}")
    resolved_root = root.resolve(strict=True)

    issues: list[AuditIssue] = []
    _report_duplicates(samples, issues)
    _report_source_inconsistencies(samples, issues)
    for sample in samples:
        _audit_image_path(sample, resolved_root, issues)

    return AuditReport(
        total_records=len(samples),
        patient_count=len({sample.patient_id for sample in samples}),
        label_counts=dict(sorted(Counter(sample.label for sample in samples).items())),
        issues=tuple(issues),
    )


def _report_duplicates(
    records: tuple[SampleRecord, ...],
    issues: list[AuditIssue],
) -> None:
    checks = (
        ("sample_id", "duplicate_sample_id", lambda record: record.sample_id),
        ("relative path", "duplicate_relative_path", lambda record: record.relative_path),
        (
            "source/variant pair",
            "duplicate_source_variant",
            lambda record: (record.source_sample_id, record.variant),
        ),
    )
    for label, code, key_function in checks:
        grouped: dict[object, list[SampleRecord]] = defaultdict(list)
        for record in records:
            grouped[key_function(record)].append(record)
        for duplicate_records in grouped.values():
            if len(duplicate_records) > 1:
                sample_ids = ", ".join(record.sample_id for record in duplicate_records)
                issues.append(
                    AuditIssue(
                        code=code,
                        severity="error",
                        message=f"Duplicate {label} across samples: {sample_ids}",
                    )
                )


def _report_source_inconsistencies(
    records: tuple[SampleRecord, ...],
    issues: list[AuditIssue],
) -> None:
    source_patients: dict[str, set[str]] = defaultdict(set)
    source_labels: dict[str, set[str]] = defaultdict(set)
    for record in records:
        source_patients[record.source_sample_id].add(record.patient_id)
        source_labels[record.source_sample_id].add(record.label)

    for source_id, patient_ids in source_patients.items():
        if len(patient_ids) > 1:
            issues.append(
                AuditIssue(
                    code="source_crosses_patients",
                    severity="error",
                    message=f"Source {source_id!r} belongs to multiple patients",
                )
            )
    for source_id, labels in source_labels.items():
        if len(labels) > 1:
            issues.append(
                AuditIssue(
                    code="source_label_conflict",
                    severity="error",
                    message=f"Source {source_id!r} has conflicting labels",
                )
            )


def _audit_image_path(
    sample: SampleRecord,
    resolved_root: Path,
    issues: list[AuditIssue],
) -> None:
    candidate = resolved_root.joinpath(*sample.relative_path.parts).resolve(strict=False)
    if not candidate.is_relative_to(resolved_root):
        issues.append(
            AuditIssue(
                code="path_escapes_dataset_root",
                severity="error",
                message="Resolved image path escapes the dataset root",
                sample_id=sample.sample_id,
            )
        )
        return

    if candidate.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        issues.append(
            AuditIssue(
                code="unsupported_image_extension",
                severity="error",
                message=f"Unsupported image extension: {candidate.suffix or '<none>'}",
                sample_id=sample.sample_id,
            )
        )
    if not candidate.is_file():
        issues.append(
            AuditIssue(
                code="missing_image",
                severity="error",
                message="Referenced image does not exist or is not a file",
                sample_id=sample.sample_id,
            )
        )
