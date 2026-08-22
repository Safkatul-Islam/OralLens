from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

from orallens_ml.cli.challenge import main
from orallens_ml.data.challenge_manifest import CHALLENGE_MANIFEST_COLUMNS


def write_fixture(root: Path, *, image_exists: bool = True) -> Path:
    manifest = root / "manifest.csv"
    row = {
        "manifest_schema_version": "1",
        "sample_id": "sample-001",
        "image_relative_path": "images/sample-001.png",
        "source_id": "synthetic-controls",
        "source_group_id": "control-001",
        "derivative_group_id": "control-001",
        "variant": "original",
        "license_or_permission_id": "project-generated",
        "partition": "locked",
        "challenge_group": "technical_quality_failure",
        "expected_handling": "unsupported",
        "expected_reason_codes": '["image_too_dark"]',
        "view": "not_applicable",
        "device_context": "synthetic",
        "quality_flags": '["darkness"]',
        "condition_reference": "not_applicable",
        "notes": "Synthetic technical control.",
    }
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CHALLENGE_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    if image_exists:
        image = root / "images" / "sample-001.png"
        image.parent.mkdir()
        Image.new("RGB", (8, 8), color=(0, 0, 0)).save(image)
    return manifest


def test_cli_emits_structured_valid_report(tmp_path: Path, capsys: object) -> None:
    manifest = write_fixture(tmp_path)

    exit_code = main(
        [
            "validate-manifest",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "challenge-manifest-validated"
    assert payload["is_valid"] is True
    assert payload["total_records"] == 1
    assert payload["source_group_count"] == 1
    assert payload["partition_counts"] == {"locked": 1}
    assert captured.err == ""


def test_cli_returns_invalid_report_without_traceback(
    tmp_path: Path,
    capsys: object,
) -> None:
    manifest = write_fixture(tmp_path, image_exists=False)

    exit_code = main(
        [
            "validate-manifest",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 2
    assert payload["is_valid"] is False
    assert payload["issues"][0]["code"] == "missing_image"
    assert captured.err == ""
    assert "Traceback" not in captured.out


def test_cli_returns_safe_structural_error_without_traceback(
    tmp_path: Path,
    capsys: object,
) -> None:
    manifest = write_fixture(tmp_path)
    manifest.write_text("not,a,valid,manifest\n", encoding="utf-8")

    exit_code = main(
        [
            "validate-manifest",
            "--manifest",
            str(manifest),
            "--dataset-root",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: Challenge manifest is missing")
    assert "Traceback" not in captured.err
