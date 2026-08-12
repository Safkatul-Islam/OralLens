from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

from orallens_ml.cli.dataset import main


def write_config(path: Path, digest: str) -> None:
    path.write_text(
        f'''schema_version = 2
dataset_id = "dataset"
dataset_version = 3
license = "CC BY 4.0"
publisher = "Mendeley Data"
source_family_id = "airc-labden"
reviewed_on = "2026-08-08"
roles = ["plaque_supervision"]
grouping_keys = ["patient_id"]
capture_context = "Standardized orthodontic photographs."
limitations = ["Not representative of consumer phone photographs."]

[label_semantics]
"0" = "plaque_absent_region"
"1" = "plaque_present_region"

[[artifacts]]
artifact_id = "part-1"
record_url = "https://data.mendeley.com/datasets/example/3"
download_url = "https://data.mendeley.com/public-api/zip/example/download/3"
partial_filename = "part-1.zip.part"
final_filename = "part-1.zip"
sha256 = "{digest}"
max_download_bytes = 1024
max_archive_entries = 4
max_member_bytes = 1024
max_uncompressed_bytes = 2048
max_compression_ratio = 10.0
expected_zip_members = ["part-1.7z"]
''',
        encoding="utf-8",
    )


def test_cli_verify_emits_structured_success(tmp_path: Path, capsys: object) -> None:
    payload = b"verified"
    config = tmp_path / "source.toml"
    write_config(config, hashlib.sha256(payload).hexdigest())
    (tmp_path / "part-1.zip.part").write_bytes(payload)

    exit_code = main(
        [
            "--config",
            str(config),
            "verify",
            "part-1",
            "--downloads-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "verified"' in captured.out
    assert captured.err == ""


def test_cli_returns_safe_error_without_traceback(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    (tmp_path / "part-1.zip.part").write_bytes(b"tampered")

    exit_code = main(
        [
            "--config",
            str(config),
            "verify",
            "part-1",
            "--downloads-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: SHA-256 mismatch")
    assert "Traceback" not in captured.err


def test_cli_summarizes_inner_listing(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    listing = tmp_path / "part-1-listing.txt"
    listing.write_text(
        "\n".join(
            [
                "mendeley-dataset-materials_Part_1/",
                "mendeley-dataset-materials_Part_1/data/images/patient0001/sample.jpg",
                "mendeley-dataset-materials_Part_1/data/labels/patient0001/sample.txt",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "summarize-inner-listing",
            "part-1",
            "--listing-file",
            str(listing),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "inner-listing-summarized"' in captured.out
    assert '"image_file_count": 1' in captured.out
    assert captured.err == ""


def test_cli_summarizes_7zip_slt_inner_listing(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    listing = tmp_path / "part-1.slt"
    listing.write_text(
        "\n".join(
            [
                "Path = C:\\archive\\part.7z",
                "Type = 7z",
                "",
                "Path = mendeley-dataset-materials_Part_1\\data",
                "Folder = +",
                "",
                "Path = mendeley-dataset-materials_Part_1\\data\\images\\patient0001\\sample.jpg",
                "Size = 10",
                "",
                "Path = mendeley-dataset-materials_Part_1\\data\\labels\\patient0001\\sample.txt",
                "Size = 1",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "summarize-inner-listing",
            "part-1",
            "--listing-file",
            str(listing),
            "--format",
            "7z-slt",
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "inner-listing-summarized"' in captured.out
    assert '"directory_count": 1' in captured.out
    assert '"image_file_count": 1' in captured.out
    assert captured.err == ""


def test_cli_audits_orthodontic_plaque_source(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "metadata"
    images = dataset_root / "data" / "images" / "patient0001"
    labels = dataset_root / "data" / "labels" / "patient0001"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (metadata / "teeth_position_mapping.json").write_text(
        json.dumps({"categories": [{"id": 7, "name": "22"}]}),
        encoding="utf-8",
    )
    (dataset_root / "data_splits.csv").write_text(
        "\n".join(
            [
                "patient,Image-Filename,Recommend",
                "patient0001,patient0001_20260101_bottom-left.jpg,train",
            ]
        ),
        encoding="utf-8",
    )
    (images / "patient0001_20260101_bottom-left.jpg").write_bytes(b"image")
    (labels / "patient0001_20260101_bottom-left.txt").write_text(
        "1 0.5 0.5 0.25 0.25 7\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "audit-orthodontic-plaque",
            "part-1",
            "--dataset-root",
            str(dataset_root),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "orthodontic-plaque-source-audited"' in captured.out
    assert '"csv_row_count": 1' in captured.out
    assert '"is_valid": true' in captured.out
    assert captured.err == ""


def test_cli_returns_structured_invalid_orthodontic_plaque_audit(
    tmp_path: Path,
    capsys: object,
) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "metadata"
    images = dataset_root / "data" / "images" / "patient0001"
    labels = dataset_root / "data" / "labels" / "patient0001"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (metadata / "teeth_position_mapping.json").write_text(
        json.dumps({"categories": [{"id": 7, "name": "22"}]}),
        encoding="utf-8",
    )
    (dataset_root / "data_splits.csv").write_text(
        "\n".join(
            [
                "patient,Image-Filename,Recommend",
                "patient0001,patient0001_20260101_bottom-left.jpg,holdout",
            ]
        ),
        encoding="utf-8",
    )
    (images / "patient0001_20260101_bottom-left.jpg").write_bytes(b"image")
    (labels / "patient0001_20260101_bottom-left.txt").write_text(
        "1 0.5 0.5 0.0 0.25 7\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "audit-orthodontic-plaque",
            "part-1",
            "--dataset-root",
            str(dataset_root),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert '"is_valid": false' in captured.out
    assert '"issue_counts"' in captured.out
    assert '"issue_examples"' in captured.out
    assert '"unsupported_split"' in captured.out
    assert '"non_positive_box_size"' in captured.out
    assert captured.err == ""
    assert "Traceback" not in captured.out


def test_cli_builds_orthodontic_plaque_manifest_outputs(
    tmp_path: Path,
    capsys: object,
) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "metadata"
    images = dataset_root / "data" / "images" / "patient0001"
    labels = dataset_root / "data" / "labels" / "patient0001"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (metadata / "teeth_position_mapping.json").write_text(
        json.dumps({"categories": [{"id": 7, "name": "22"}]}),
        encoding="utf-8",
    )
    (dataset_root / "data_splits.csv").write_text(
        "\n".join(
            [
                "patient,Image-Filename,Recommend",
                "patient0001,patient0001_20260101_bottom-left.jpg,train",
            ]
        ),
        encoding="utf-8",
    )
    (images / "patient0001_20260101_bottom-left.jpg").write_bytes(b"image")
    (labels / "patient0001_20260101_bottom-left.txt").write_text(
        "\n".join(
            [
                "1 0.5 0.5 0.0 0.25 7",
                "1 0.5 0.5 0.25 0.25 7",
            ]
        ),
        encoding="utf-8",
    )
    manifest_output = tmp_path / "manifest.csv"
    exclusions_output = tmp_path / "exclusions.csv"

    exit_code = main(
        [
            "--config",
            str(config),
            "build-orthodontic-plaque-manifest",
            "part-1",
            "--dataset-root",
            str(dataset_root),
            "--manifest-output",
            str(manifest_output),
            "--exclusions-output",
            str(exclusions_output),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "orthodontic-plaque-manifest-built"
    assert payload["included_samples"] == 1
    assert payload["excluded_samples"] == 0
    assert payload["filtered_annotation_count"] == 1
    assert payload["issue_counts"] == {"non_positive_box_size": 1}
    assert captured.err == ""
    manifest_rows = list(csv.DictReader(manifest_output.open("r", encoding="utf-8")))
    exclusion_rows = list(csv.DictReader(exclusions_output.open("r", encoding="utf-8")))
    assert manifest_rows[0]["annotation_count"] == "1"
    assert manifest_rows[0]["manifest_schema_version"] == "1"
    assert manifest_rows[0]["dataset_id"] == "dataset"
    assert manifest_rows[0]["dataset_version"] == "3"
    assert manifest_rows[0]["source_family_id"] == "airc-labden"
    assert manifest_rows[0]["source_artifact_id"] == "part-1"
    assert manifest_rows[0]["split_group_id"] == "patient0001"
    assert manifest_rows[0]["derivative_group_id"] == (
        "patient0001_20260101_bottom-left"
    )
    assert manifest_rows[0]["variant"] == "original"
    assert exclusion_rows[0]["code"] == "non_positive_box_size"


def test_cli_build_manifest_preflight_prevents_partial_writes(
    tmp_path: Path,
    capsys: object,
) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    dataset_root = tmp_path / "dataset"
    metadata = dataset_root / "metadata"
    images = dataset_root / "data" / "images" / "patient0001"
    labels = dataset_root / "data" / "labels" / "patient0001"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (metadata / "teeth_position_mapping.json").write_text(
        json.dumps({"categories": [{"id": 7, "name": "22"}]}),
        encoding="utf-8",
    )
    (dataset_root / "data_splits.csv").write_text(
        "\n".join(
            [
                "patient,Image-Filename,Recommend",
                "patient0001,patient0001_20260101_bottom-left.jpg,train",
            ]
        ),
        encoding="utf-8",
    )
    (images / "patient0001_20260101_bottom-left.jpg").write_bytes(b"image")
    (labels / "patient0001_20260101_bottom-left.txt").write_text(
        "1 0.5 0.5 0.25 0.25 7\n",
        encoding="utf-8",
    )
    shared_output = tmp_path / "same.csv"

    exit_code = main(
        [
            "--config",
            str(config),
            "build-orthodontic-plaque-manifest",
            "part-1",
            "--dataset-root",
            str(dataset_root),
            "--manifest-output",
            str(shared_output),
            "--exclusions-output",
            str(shared_output),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: Manifest and exclusions outputs must differ")
    assert "Traceback" not in captured.err
    assert not shared_output.exists()


def test_cli_validates_orthodontic_plaque_images(
    tmp_path: Path,
    capsys: object,
) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    dataset_root = tmp_path / "dataset"
    image = dataset_root / "data" / "images" / "patient0001" / "sample.jpg"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (5, 4), color=(0, 255, 0)).save(image)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "sample_id,image_relative_path",
                "sample-1,data/images/patient0001/sample.jpg",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "validate-orthodontic-plaque-images",
            "part-1",
            "--dataset-root",
            str(dataset_root),
            "--manifest",
            str(manifest),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert payload["status"] == "orthodontic-plaque-images-validated"
    assert payload["is_valid"] is True
    assert payload["decoded_images"] == 1
    assert payload["mode_counts"] == {"RGB": 1}
    assert payload["size_counts"] == {"5x4": 1}
    assert captured.err == ""
