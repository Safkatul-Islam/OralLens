"""Normalized manifest contract for OralLens image datasets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

ALLOWED_LABELS = frozenset({"0-1", "2", "3", "4"})
REQUIRED_COLUMNS = (
    "sample_id",
    "patient_id",
    "relative_path",
    "label",
    "source_sample_id",
    "variant",
)


class ManifestError(ValueError):
    """Raised when a manifest violates the normalized data contract."""


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """One immutable image record from the normalized manifest."""

    sample_id: str
    patient_id: str
    relative_path: PurePosixPath
    label: str
    source_sample_id: str
    variant: str
    manifest_line: int


def load_manifest(manifest_path: Path) -> tuple[SampleRecord, ...]:
    """Load and structurally validate a UTF-8 CSV manifest.

    Paths use POSIX separators so the same manifest works on every platform.
    Dataset-wide checks such as duplicate IDs and missing files belong to the
    audit layer.
    """

    path = Path(manifest_path)
    if not path.is_file():
        raise ManifestError(f"Manifest is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_header(reader.fieldnames)
            records = tuple(
                _record_from_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except UnicodeDecodeError as exc:
        raise ManifestError("Manifest must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise ManifestError(f"Manifest CSV is malformed: {exc}") from exc

    if not records:
        raise ManifestError("Manifest must contain at least one data row")
    return records


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise ManifestError("Manifest is missing a header row")
    if len(fieldnames) != len(set(fieldnames)):
        raise ManifestError("Manifest contains duplicate column names")

    missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise ManifestError(f"Manifest is missing required columns: {', '.join(missing)}")


def _record_from_row(row: dict[str, str | None], line_number: int) -> SampleRecord:
    values = {
        column: _required_value(row.get(column), column, line_number)
        for column in REQUIRED_COLUMNS
    }
    label = values["label"]
    if label not in ALLOWED_LABELS:
        allowed = ", ".join(sorted(ALLOWED_LABELS))
        raise ManifestError(
            f"Line {line_number}: label must be one of {allowed}; got {label!r}"
        )

    relative_path = _validate_relative_path(values["relative_path"], line_number)
    return SampleRecord(
        sample_id=values["sample_id"],
        patient_id=values["patient_id"],
        relative_path=relative_path,
        label=label,
        source_sample_id=values["source_sample_id"],
        variant=values["variant"],
        manifest_line=line_number,
    )


def _required_value(value: str | None, column: str, line_number: int) -> str:
    normalized = value.strip() if value is not None else ""
    if not normalized:
        raise ManifestError(f"Line {line_number}: {column} must not be empty")
    if "\x00" in normalized:
        raise ManifestError(f"Line {line_number}: {column} contains a null byte")
    return normalized


def _validate_relative_path(value: str, line_number: int) -> PurePosixPath:
    if "\\" in value:
        raise ManifestError(
            f"Line {line_number}: relative_path must use POSIX '/' separators"
        )
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ManifestError(
            f"Line {line_number}: relative_path contains an unsafe path segment"
        )

    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise ManifestError(f"Line {line_number}: relative_path must be relative")
    return path
