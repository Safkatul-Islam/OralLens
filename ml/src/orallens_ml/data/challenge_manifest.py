"""Versioned manifest and read-only audit for real-world challenge images."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Literal, cast

CHALLENGE_MANIFEST_SCHEMA_VERSION = 1
CHALLENGE_MANIFEST_COLUMNS = (
    "manifest_schema_version",
    "sample_id",
    "image_relative_path",
    "source_id",
    "source_group_id",
    "derivative_group_id",
    "variant",
    "license_or_permission_id",
    "partition",
    "challenge_group",
    "expected_handling",
    "expected_reason_codes",
    "view",
    "device_context",
    "quality_flags",
    "condition_reference",
    "notes",
)
CHALLENGE_GROUPS = frozenset(
    {
        "controlled_supported_oral",
        "consumer_supported_oral",
        "wider_facial_context",
        "face_dominant_or_teeth_too_small",
        "technical_quality_failure",
        "blur_glare_occlusion",
        "non_oral_ood",
        "condition_hard_negative",
        "condition_positive",
    }
)
EXPECTED_HANDLING_VALUES = frozenset({"supported", "unsupported", "manual_review"})
PARTITIONS = frozenset({"development", "locked"})
QUALITY_FLAGS = frozenset(
    {
        "blur",
        "compression",
        "darkness",
        "framing",
        "glare",
        "occlusion",
        "overexposure",
    }
)
SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpeg", ".jpg", ".png"})

_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._:-]*")

ChallengePartition = Literal["development", "locked"]
ExpectedHandling = Literal["supported", "unsupported", "manual_review"]


class ChallengeManifestError(ValueError):
    """Raised when a challenge manifest violates its structural contract."""


@dataclass(frozen=True, slots=True)
class ChallengeSample:
    """One immutable real-world challenge sample record."""

    manifest_schema_version: int
    sample_id: str
    image_relative_path: PurePosixPath
    source_id: str
    source_group_id: str
    derivative_group_id: str
    variant: str
    license_or_permission_id: str
    partition: ChallengePartition
    challenge_group: str
    expected_handling: ExpectedHandling
    expected_reason_codes: tuple[str, ...]
    view: str
    device_context: str
    quality_flags: tuple[str, ...]
    condition_reference: str
    notes: str
    manifest_line: int


@dataclass(frozen=True, slots=True)
class ChallengeManifestIssue:
    """One actionable cross-record or filesystem validation failure."""

    code: str
    message: str
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class ChallengeManifestReport:
    """Immutable summary of a challenge-manifest audit."""

    total_records: int
    source_group_count: int
    partition_counts: dict[str, int]
    challenge_group_counts: dict[str, int]
    expected_handling_counts: dict[str, int]
    issues: tuple[ChallengeManifestIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def load_challenge_manifest(manifest_path: Path) -> tuple[ChallengeSample, ...]:
    """Load and structurally validate one UTF-8 challenge CSV."""

    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise ChallengeManifestError(f"Challenge manifest is not a regular file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            _validate_header(reader.fieldnames)
            records = tuple(
                _sample_from_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except UnicodeDecodeError as exc:
        raise ChallengeManifestError("Challenge manifest must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise ChallengeManifestError(
            f"Challenge manifest CSV is malformed: {exc}"
        ) from exc

    if not records:
        raise ChallengeManifestError(
            "Challenge manifest must contain at least one data row"
        )
    return records


def audit_challenge_manifest(
    records: Iterable[ChallengeSample],
    dataset_root: Path,
) -> ChallengeManifestReport:
    """Audit challenge identity, split isolation, and image paths without writes."""

    samples = tuple(records)
    if not samples:
        raise ChallengeManifestError(
            "Challenge manifest must contain at least one sample"
        )

    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise ChallengeManifestError(
            f"Challenge dataset root is not a regular directory: {root}"
        )
    resolved_root = root.resolve(strict=True)

    issues: list[ChallengeManifestIssue] = []
    _report_duplicates(samples, issues)
    _report_partition_leakage(samples, issues)
    for sample in samples:
        _audit_image_path(sample, root, resolved_root, issues)

    return ChallengeManifestReport(
        total_records=len(samples),
        source_group_count=len(
            {(sample.source_id, sample.source_group_id) for sample in samples}
        ),
        partition_counts=_counts(sample.partition for sample in samples),
        challenge_group_counts=_counts(sample.challenge_group for sample in samples),
        expected_handling_counts=_counts(
            sample.expected_handling for sample in samples
        ),
        issues=tuple(issues),
    )


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ChallengeManifestError("Challenge manifest is missing a header row")
    if len(fieldnames) != len(set(fieldnames)):
        raise ChallengeManifestError(
            "Challenge manifest contains duplicate column names"
        )

    missing = [column for column in CHALLENGE_MANIFEST_COLUMNS if column not in fieldnames]
    unknown = [column for column in fieldnames if column not in CHALLENGE_MANIFEST_COLUMNS]
    if missing:
        raise ChallengeManifestError(
            "Challenge manifest is missing required columns: " + ", ".join(missing)
        )
    if unknown:
        raise ChallengeManifestError(
            "Challenge manifest contains unknown columns: " + ", ".join(unknown)
        )


def _sample_from_row(
    row: dict[str, str | None],
    line_number: int,
) -> ChallengeSample:
    if any(column is None for column in row):
        raise ChallengeManifestError(
            f"Line {line_number}: challenge manifest row contains unexpected values"
        )
    schema_version = _parse_schema_version(row.get("manifest_schema_version"), line_number)
    identifiers = {
        column: _required_identifier(row.get(column), column, line_number)
        for column in (
            "sample_id",
            "source_id",
            "source_group_id",
            "derivative_group_id",
            "variant",
            "license_or_permission_id",
            "view",
            "device_context",
            "condition_reference",
        )
    }
    partition = _allowed_value(
        row.get("partition"), "partition", PARTITIONS, line_number
    )
    challenge_group = _allowed_value(
        row.get("challenge_group"),
        "challenge_group",
        CHALLENGE_GROUPS,
        line_number,
    )
    expected_handling = _allowed_value(
        row.get("expected_handling"),
        "expected_handling",
        EXPECTED_HANDLING_VALUES,
        line_number,
    )
    expected_reason_codes = _parse_identifier_array(
        row.get("expected_reason_codes"),
        "expected_reason_codes",
        line_number,
    )
    quality_flags = _parse_identifier_array(
        row.get("quality_flags"), "quality_flags", line_number
    )
    unknown_flags = sorted(set(quality_flags) - QUALITY_FLAGS)
    if unknown_flags:
        raise ChallengeManifestError(
            f"Line {line_number}: unknown quality_flags: {', '.join(unknown_flags)}"
        )
    if expected_handling == "supported" and expected_reason_codes:
        raise ChallengeManifestError(
            f"Line {line_number}: supported samples must not expect reason codes"
        )
    if expected_handling == "unsupported" and not expected_reason_codes:
        raise ChallengeManifestError(
            f"Line {line_number}: unsupported samples must expect a reason code"
        )

    notes_value = row.get("notes")
    notes = notes_value.strip() if notes_value is not None else ""
    if "\x00" in notes:
        raise ChallengeManifestError(f"Line {line_number}: notes contains a null byte")

    return ChallengeSample(
        manifest_schema_version=schema_version,
        sample_id=identifiers["sample_id"],
        image_relative_path=_validate_relative_path(
            row.get("image_relative_path"), line_number
        ),
        source_id=identifiers["source_id"],
        source_group_id=identifiers["source_group_id"],
        derivative_group_id=identifiers["derivative_group_id"],
        variant=identifiers["variant"],
        license_or_permission_id=identifiers["license_or_permission_id"],
        partition=cast(ChallengePartition, partition),
        challenge_group=challenge_group,
        expected_handling=cast(ExpectedHandling, expected_handling),
        expected_reason_codes=expected_reason_codes,
        view=identifiers["view"],
        device_context=identifiers["device_context"],
        quality_flags=quality_flags,
        condition_reference=identifiers["condition_reference"],
        notes=notes,
        manifest_line=line_number,
    )


def _parse_schema_version(value: str | None, line_number: int) -> int:
    normalized = _required_text(value, "manifest_schema_version", line_number)
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise ChallengeManifestError(
            f"Line {line_number}: manifest_schema_version must be an integer"
        ) from exc
    if parsed != CHALLENGE_MANIFEST_SCHEMA_VERSION:
        raise ChallengeManifestError(
            f"Line {line_number}: unsupported challenge manifest schema: {parsed}"
        )
    return parsed


def _required_identifier(
    value: str | None,
    column: str,
    line_number: int,
) -> str:
    normalized = _required_text(value, column, line_number)
    if _IDENTIFIER_PATTERN.fullmatch(normalized) is None:
        raise ChallengeManifestError(
            f"Line {line_number}: {column} must be a normalized identifier"
        )
    return normalized


def _required_text(value: str | None, column: str, line_number: int) -> str:
    normalized = value.strip() if value is not None else ""
    if not normalized:
        raise ChallengeManifestError(f"Line {line_number}: {column} must not be empty")
    if "\x00" in normalized:
        raise ChallengeManifestError(
            f"Line {line_number}: {column} contains a null byte"
        )
    return normalized


def _allowed_value(
    value: str | None,
    column: str,
    allowed: frozenset[str],
    line_number: int,
) -> str:
    normalized = _required_text(value, column, line_number)
    if normalized not in allowed:
        raise ChallengeManifestError(
            f"Line {line_number}: unsupported {column}: {normalized!r}"
        )
    return normalized


def _parse_identifier_array(
    value: str | None,
    column: str,
    line_number: int,
) -> tuple[str, ...]:
    normalized = _required_text(value, column, line_number)
    try:
        parsed = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ChallengeManifestError(
            f"Line {line_number}: {column} must be a JSON array"
        ) from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) for item in parsed):
        raise ChallengeManifestError(
            f"Line {line_number}: {column} must be a JSON array of identifiers"
        )
    if len(parsed) != len(set(parsed)):
        raise ChallengeManifestError(
            f"Line {line_number}: {column} contains duplicate values"
        )
    for item in parsed:
        if _IDENTIFIER_PATTERN.fullmatch(item) is None:
            raise ChallengeManifestError(
                f"Line {line_number}: {column} contains an invalid identifier"
            )
    return tuple(parsed)


def _validate_relative_path(value: str | None, line_number: int) -> PurePosixPath:
    normalized = _required_text(value, "image_relative_path", line_number)
    if "\\" in normalized:
        raise ChallengeManifestError(
            f"Line {line_number}: image_relative_path must use POSIX '/' separators"
        )
    if any(part in {"", ".", ".."} for part in normalized.split("/")):
        raise ChallengeManifestError(
            f"Line {line_number}: image_relative_path contains an unsafe path segment"
        )
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts:
        raise ChallengeManifestError(
            f"Line {line_number}: image_relative_path must be relative"
        )
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ChallengeManifestError(
            f"Line {line_number}: unsupported challenge image extension: {path.suffix or '<none>'}"
        )
    return path


def _report_duplicates(
    records: tuple[ChallengeSample, ...],
    issues: list[ChallengeManifestIssue],
) -> None:
    checks = (
        ("duplicate_sample_id", lambda record: record.sample_id),
        ("duplicate_image_path", lambda record: record.image_relative_path),
        (
            "duplicate_derivative_variant",
            lambda record: (record.source_id, record.derivative_group_id, record.variant),
        ),
    )
    for code, key_function in checks:
        grouped: dict[object, list[ChallengeSample]] = defaultdict(list)
        for record in records:
            grouped[key_function(record)].append(record)
        for duplicate_records in grouped.values():
            if len(duplicate_records) > 1:
                sample_ids = ", ".join(
                    record.sample_id for record in duplicate_records
                )
                issues.append(
                    ChallengeManifestIssue(
                        code=code,
                        message=f"Duplicate challenge identity across: {sample_ids}",
                    )
                )


def _report_partition_leakage(
    records: tuple[ChallengeSample, ...],
    issues: list[ChallengeManifestIssue],
) -> None:
    source_group_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    derivative_memberships: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = defaultdict(set)
    for record in records:
        source_group_partitions[(record.source_id, record.source_group_id)].add(
            record.partition
        )
        derivative_memberships[(record.source_id, record.derivative_group_id)].add(
            (record.source_group_id, record.partition)
        )

    for source_id, source_group_id in sorted(
        key for key, partitions in source_group_partitions.items() if len(partitions) > 1
    ):
        issues.append(
            ChallengeManifestIssue(
                code="source_group_partition_leakage",
                message=(
                    "Source group crosses development and locked partitions: "
                    f"{source_id} {source_group_id}"
                ),
            )
        )
    for source_id, derivative_group_id in sorted(
        key for key, memberships in derivative_memberships.items() if len(memberships) > 1
    ):
        issues.append(
            ChallengeManifestIssue(
                code="derivative_group_partition_leakage",
                message=(
                    "Derivative group crosses source groups or partitions: "
                    f"{source_id} {derivative_group_id}"
                ),
            )
        )


def _audit_image_path(
    sample: ChallengeSample,
    root: Path,
    resolved_root: Path,
    issues: list[ChallengeManifestIssue],
) -> None:
    candidate = root.joinpath(*sample.image_relative_path.parts)
    cursor = root
    for part in sample.image_relative_path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            issues.append(
                ChallengeManifestIssue(
                    code="symlinked_image_path",
                    message="Challenge image path contains a symbolic link",
                    sample_id=sample.sample_id,
                )
            )
            return

    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        issues.append(
            ChallengeManifestIssue(
                code="image_path_escapes_dataset_root",
                message="Resolved challenge image path escapes the dataset root",
                sample_id=sample.sample_id,
            )
        )
        return
    if not candidate.is_file():
        issues.append(
            ChallengeManifestIssue(
                code="missing_image",
                message="Referenced challenge image does not exist or is not a file",
                sample_id=sample.sample_id,
            )
        )


def _counts(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
