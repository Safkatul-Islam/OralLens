from __future__ import annotations

import csv
from pathlib import Path

import pytest

from orallens_ml.data.manifest import ManifestError, load_manifest

FIELDNAMES = (
    "sample_id",
    "patient_id",
    "relative_path",
    "label",
    "source_sample_id",
    "variant",
)


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def valid_row(**overrides: str) -> dict[str, str]:
    row = {
        "sample_id": "sample-001",
        "patient_id": "patient-001",
        "relative_path": "images/sample-001.jpg",
        "label": "2",
        "source_sample_id": "source-001",
        "variant": "original",
    }
    row.update(overrides)
    return row


def test_load_manifest_returns_immutable_normalized_records(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [valid_row()])

    records = load_manifest(manifest_path)

    assert len(records) == 1
    assert records[0].relative_path.as_posix() == "images/sample-001.jpg"
    assert records[0].manifest_line == 2
    with pytest.raises(AttributeError):
        records[0].label = "3"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"label": "5"}, "label must be one of"),
        ({"patient_id": "  "}, "patient_id must not be empty"),
        ({"relative_path": "../outside.jpg"}, "unsafe path segment"),
        ({"relative_path": "/absolute.jpg"}, "unsafe path segment"),
        ({"relative_path": "images\\sample.jpg"}, "POSIX '/' separators"),
    ],
)
def test_load_manifest_rejects_invalid_rows(
    tmp_path: Path,
    override: dict[str, str],
    message: str,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, [valid_row(**override)])

    with pytest.raises(ManifestError, match=message):
        load_manifest(manifest_path)


def test_load_manifest_rejects_missing_required_column(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("sample_id,patient_id\na,b\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="missing required columns"):
        load_manifest(manifest_path)
