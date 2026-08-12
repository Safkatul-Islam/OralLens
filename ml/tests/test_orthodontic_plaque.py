from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

import pytest
from PIL import Image

from orallens_ml.data.orthodontic_plaque import (
    OrthodonticPlaqueAuditError,
    audit_orthodontic_plaque_source,
    build_orthodontic_plaque_part2_manifest,
    preflight_part2_output_paths,
    validate_part2_manifest_images,
    write_part2_exclusions_csv,
    write_part2_manifest_csv,
)


def write_source_dataset(
    root: Path,
    *,
    split: str = "train",
    image_filename: str = "patient0001_20260101_bottom-left.jpg",
    label_payload: str = "1 0.500000 0.500000 0.250000 0.250000 7\n",
) -> None:
    metadata = root / "metadata"
    images = root / "data" / "images" / "patient0001"
    labels = root / "data" / "labels" / "patient0001"
    metadata.mkdir(parents=True)
    images.mkdir(parents=True)
    labels.mkdir(parents=True)

    (metadata / "teeth_position_mapping.json").write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 7, "name": "22", "supercategory": "objects"},
                    {"id": 21, "name": "brace", "supercategory": "objects"},
                ]
            }
        ),
        encoding="utf-8",
    )
    with (root / "data_splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient", "Image-Filename", "Recommend"))
        writer.writeheader()
        writer.writerow(
            {
                "patient": "patient0001",
                "Image-Filename": image_filename,
                "Recommend": split,
            }
        )
    (images / image_filename).write_bytes(b"synthetic-image")
    (labels / f"{Path(image_filename).stem}.txt").write_text(
        label_payload,
        encoding="utf-8",
    )


def test_audit_orthodontic_plaque_source_accepts_valid_dataset(tmp_path: Path) -> None:
    write_source_dataset(tmp_path)

    report = audit_orthodontic_plaque_source(tmp_path)

    assert report.is_valid
    assert report.csv_row_count == 1
    assert report.patient_count == 1
    assert report.split_counts == {"train": 1}
    assert report.image_count == 1
    assert report.label_file_count == 1
    assert report.annotation_count == 1
    assert report.tooth_position_ids == (7,)
    assert report.issues == ()


def test_audit_orthodontic_plaque_source_rejects_bad_split(tmp_path: Path) -> None:
    write_source_dataset(tmp_path, split="holdout")

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert "unsupported_split" in {issue.code for issue in report.issues}


def test_audit_orthodontic_plaque_source_rejects_missing_label(tmp_path: Path) -> None:
    write_source_dataset(tmp_path)
    (tmp_path / "data" / "labels" / "patient0001" / "patient0001_20260101_bottom-left.txt").unlink()

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert "missing_label" in {issue.code for issue in report.issues}


def test_audit_orthodontic_plaque_source_rejects_bad_label_row(tmp_path: Path) -> None:
    write_source_dataset(tmp_path, label_payload="1 1.500000 0.500000 0.250000 0.250000 7\n")

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert "label_coordinate_out_of_bounds" in {issue.code for issue in report.issues}


def test_audit_orthodontic_plaque_source_rejects_unsupported_class_id(
    tmp_path: Path,
) -> None:
    write_source_dataset(
        tmp_path,
        label_payload="2 0.500000 0.500000 0.250000 0.250000 7\n",
    )

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert report.issue_counts == {"unsupported_class_id": 1}


def test_audit_orthodontic_plaque_source_rejects_unknown_tooth_id(tmp_path: Path) -> None:
    write_source_dataset(tmp_path, label_payload="1 0.500000 0.500000 0.250000 0.250000 99\n")

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert "unknown_tooth_id" in {issue.code for issue in report.issues}


def test_audit_orthodontic_plaque_source_rejects_unsafe_image_filename(
    tmp_path: Path,
) -> None:
    write_source_dataset(tmp_path)
    with (tmp_path / "data_splits.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient", "Image-Filename", "Recommend"))
        writer.writeheader()
        writer.writerow(
            {
                "patient": "patient0001",
                "Image-Filename": "..\\escape.jpg",
                "Recommend": "train",
            }
        )

    report = audit_orthodontic_plaque_source(tmp_path)

    assert not report.is_valid
    assert "unsafe_image_filename" in {issue.code for issue in report.issues}


def test_audit_orthodontic_plaque_source_collects_multiple_issues(tmp_path: Path) -> None:
    write_source_dataset(
        tmp_path,
        split="holdout",
        label_payload="\n".join(
            [
                "1 0.500000 0.500000 0.000000 0.250000 7",
                "1 0.500000 0.500000 0.250000 0.250000 99",
            ]
        ),
    )

    report = audit_orthodontic_plaque_source(tmp_path)
    codes = {issue.code for issue in report.issues}

    assert not report.is_valid
    assert "unsupported_split" in codes
    assert "non_positive_box_size" in codes
    assert "unknown_tooth_id" in codes
    assert report.issue_counts == {
        "non_positive_box_size": 1,
        "unknown_tooth_id": 1,
        "unsupported_split": 1,
    }
    assert report.issue_examples["non_positive_box_size"][0].csv_line == 2


def test_part2_manifest_builder_filters_invalid_annotation_rows(tmp_path: Path) -> None:
    write_source_dataset(
        tmp_path,
        label_payload="\n".join(
            [
                "1 0.500000 0.500000 0.000000 0.250000 7",
                "1 0.500000 0.500000 0.250000 0.250000 7",
            ]
        ),
    )

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert len(samples) == 1
    assert samples[0].derivative_group_id == "patient0001_20260101_bottom-left"
    assert samples[0].variant == "original"
    assert len(samples[0].annotations) == 1
    assert samples[0].annotations[0].width == 0.25
    assert report.filtered_annotation_count == 1
    assert report.issue_counts == {"non_positive_box_size": 1}


def test_part2_manifest_builder_filters_unsupported_class_id(tmp_path: Path) -> None:
    write_source_dataset(
        tmp_path,
        label_payload="\n".join(
            [
                "2 0.500000 0.500000 0.250000 0.250000 7",
                "1 0.500000 0.500000 0.250000 0.250000 7",
            ]
        ),
    )

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert len(samples) == 1
    assert [annotation.class_id for annotation in samples[0].annotations] == [1]
    assert report.filtered_annotation_count == 1
    assert report.issue_counts == {"unsupported_class_id": 1}


def test_part2_manifest_builder_excludes_empty_label_sample(tmp_path: Path) -> None:
    write_source_dataset(tmp_path, label_payload="")

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert samples == ()
    assert report.excluded_samples == 1
    assert report.issue_counts == {"empty_label": 1}
    assert report.exclusions[0].split == "train"


def test_part2_manifest_builder_counts_missing_image_as_excluded_sample(
    tmp_path: Path,
) -> None:
    write_source_dataset(tmp_path)
    (tmp_path / "data" / "images" / "patient0001" / "patient0001_20260101_bottom-left.jpg").unlink()

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert samples == ()
    assert report.excluded_samples == 1
    assert report.issue_counts == {"missing_image": 1}


def test_part2_manifest_builder_excludes_sample_when_all_rows_invalid(tmp_path: Path) -> None:
    write_source_dataset(
        tmp_path,
        label_payload="\n".join(
            [
                "1 0.500000 0.500000 0.000000 0.250000 7",
                "1 0.500000 0.500000 0.250000 0.000000 7",
            ]
        ),
    )

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert samples == ()
    assert report.excluded_samples == 1
    assert report.filtered_annotation_count == 2
    assert report.issue_counts == {"no_valid_annotations": 1, "non_positive_box_size": 2}


def test_part2_manifest_builder_normalizes_validation_split(tmp_path: Path) -> None:
    write_source_dataset(tmp_path, split="val")

    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert samples[0].split == "validation"
    assert report.split_counts == {"validation": 1}


def test_part2_manifest_builder_tracks_publisher_derivative_family(
    tmp_path: Path,
) -> None:
    write_source_dataset(
        tmp_path,
        image_filename="patient0001_20260101_bottom-left_rotate-right-15.jpg",
    )

    samples, _ = build_orthodontic_plaque_part2_manifest(tmp_path)

    assert samples[0].derivative_group_id == "patient0001_20260101_bottom-left"
    assert samples[0].variant == "rotate-right-15"


def test_part2_manifest_builder_rejects_duplicate_retained_sample_ids(
    tmp_path: Path,
) -> None:
    write_source_dataset(tmp_path)
    with (tmp_path / "data_splits.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("patient", "Image-Filename", "Recommend"))
        writer.writerow(
            {
                "patient": "patient0001",
                "Image-Filename": "patient0001_20260101_bottom-left.jpg",
                "Recommend": "train",
            }
        )

    with pytest.raises(OrthodonticPlaqueAuditError, match="Duplicate retained sample IDs"):
        build_orthodontic_plaque_part2_manifest(tmp_path)


def test_part2_manifest_builder_writes_outputs(tmp_path: Path) -> None:
    write_source_dataset(
        tmp_path,
        label_payload="\n".join(
            [
                "1 0.500000 0.500000 0.000000 0.250000 7",
                "1 0.500000 0.500000 0.250000 0.250000 7",
            ]
        ),
    )
    samples, report = build_orthodontic_plaque_part2_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    exclusions_path = tmp_path / "exclusions.csv"

    write_part2_manifest_csv(
        samples,
        manifest_path,
        dataset_id="orthodontic-plaque-fixed-labial",
        dataset_version=3,
        source_family_id="airc-labden-orthodontic-plaque",
        source_artifact_id="part-2",
    )
    write_part2_exclusions_csv(report.exclusions, exclusions_path)

    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_handle:
        manifest_reader = csv.DictReader(manifest_handle)
        assert manifest_reader.fieldnames == [
            "manifest_schema_version",
            "dataset_id",
            "dataset_version",
            "source_family_id",
            "source_artifact_id",
            "split_group_id",
            "derivative_group_id",
            "variant",
            "sample_id",
            "patient_id",
            "split",
            "image_relative_path",
            "source_image_filename",
            "source_csv_line",
            "annotation_count",
            "annotations_json",
        ]
        manifest_rows = list(manifest_reader)
    with exclusions_path.open("r", encoding="utf-8", newline="") as exclusions_handle:
        exclusions_reader = csv.DictReader(exclusions_handle)
        assert exclusions_reader.fieldnames == [
            "sample_id",
            "patient_id",
            "split",
            "source_image_filename",
            "source_csv_line",
            "label_line",
            "code",
            "message",
            "filtered_annotation_count",
        ]
        exclusion_rows = list(exclusions_reader)
    assert manifest_rows[0]["sample_id"] == "patient0001_20260101_bottom-left"
    assert manifest_rows[0]["manifest_schema_version"] == "1"
    assert manifest_rows[0]["dataset_id"] == "orthodontic-plaque-fixed-labial"
    assert manifest_rows[0]["dataset_version"] == "3"
    assert manifest_rows[0]["source_family_id"] == "airc-labden-orthodontic-plaque"
    assert manifest_rows[0]["source_artifact_id"] == "part-2"
    assert manifest_rows[0]["split_group_id"] == "patient0001"
    assert manifest_rows[0]["derivative_group_id"] == (
        "patient0001_20260101_bottom-left"
    )
    assert manifest_rows[0]["variant"] == "original"
    assert manifest_rows[0]["split"] == "train"
    assert manifest_rows[0]["image_relative_path"] == (
        "data/images/patient0001/patient0001_20260101_bottom-left.jpg"
    )
    assert manifest_rows[0]["annotation_count"] == "1"
    assert json.loads(manifest_rows[0]["annotations_json"])[0]["width"] == 0.25
    assert exclusion_rows[0]["code"] == "non_positive_box_size"
    assert exclusion_rows[0]["label_line"] == "1"


def test_part2_output_preflight_rejects_conflicts_and_existing_files(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    exclusions_path = tmp_path / "exclusions.csv"

    with pytest.raises(OrthodonticPlaqueAuditError, match="must differ"):
        preflight_part2_output_paths(manifest_path, manifest_path)

    manifest_path.write_text("existing", encoding="utf-8")
    with pytest.raises(OrthodonticPlaqueAuditError, match="already exists"):
        preflight_part2_output_paths(manifest_path, exclusions_path)


def write_image_manifest_fixture(
    root: Path,
    manifest: Path,
    *,
    image_payload: str = "valid",
    image_path: str = "data/images/patient0001/sample.jpg",
) -> None:
    image_file = root.joinpath(*PurePosixPath(image_path).parts)
    image_file.parent.mkdir(parents=True)
    if image_payload == "valid":
        Image.new("RGB", (4, 3), color=(255, 0, 0)).save(image_file)
    else:
        image_file.write_bytes(b"not an image")
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_relative_path"))
        writer.writeheader()
        writer.writerow({"sample_id": "sample-1", "image_relative_path": image_path})


def test_validate_part2_manifest_images_accepts_decodable_images(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_image_manifest_fixture(tmp_path, manifest)

    report = validate_part2_manifest_images(tmp_path, manifest)

    assert report.is_valid
    assert report.total_records == 1
    assert report.decoded_images == 1
    assert report.mode_counts == {"RGB": 1}
    assert report.size_counts == {"4x3": 1}


def test_validate_part2_manifest_images_reports_corrupt_image(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_image_manifest_fixture(tmp_path, manifest, image_payload="corrupt")

    report = validate_part2_manifest_images(tmp_path, manifest)

    assert not report.is_valid
    assert report.decoded_images == 0
    assert report.issue_counts == {"unreadable_image": 1}


def test_validate_part2_manifest_images_reports_unsafe_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("sample_id", "image_relative_path"))
        writer.writeheader()
        writer.writerow({"sample_id": "sample-1", "image_relative_path": "../escape.jpg"})

    report = validate_part2_manifest_images(tmp_path, manifest)

    assert not report.is_valid
    assert report.issue_counts == {"unsafe_image_path": 1}
