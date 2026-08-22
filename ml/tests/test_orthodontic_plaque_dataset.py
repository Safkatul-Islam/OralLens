from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest

from orallens_ml.data.orthodontic_plaque_dataset import (
    OrthodonticPlaqueDatasetError,
    OrthodonticPlaquePart2Dataset,
)


FIELDNAMES = (
    "sample_id",
    "patient_id",
    "split",
    "image_relative_path",
    "source_csv_line",
    "annotation_count",
    "annotations_json",
)
SOURCE_AWARE_FIELDNAMES = (
    "manifest_schema_version",
    "dataset_id",
    "dataset_version",
    "source_family_id",
    "source_artifact_id",
    "split_group_id",
    "derivative_group_id",
    "variant",
    *FIELDNAMES,
)


def annotation(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "class_id": 1,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.25,
        "height": 0.25,
        "tooth_id": 7,
        "source_label_line": 1,
    }
    row.update(overrides)
    return row


def write_fixture(
    root: Path,
    manifest: Path,
    *,
    split: str = "train",
    image_relative_path: str = "data/images/patient0001/sample.jpg",
    annotations: list[dict[str, object]] | None = None,
    fieldnames: tuple[str, ...] = FIELDNAMES,
) -> None:
    image_path = root.joinpath(*image_relative_path.split("/"))
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (4, 3), color=(20, 40, 60)).save(image_path)
    annotation_rows = [annotation()] if annotations is None else annotations
    values = {
        "manifest_schema_version": "1",
        "dataset_id": "dataset",
        "dataset_version": "3",
        "source_family_id": "source-family",
        "source_artifact_id": "part-2",
        "split_group_id": "patient0001",
        "derivative_group_id": "sample",
        "variant": "original",
        "sample_id": "sample",
        "patient_id": "patient0001",
        "split": split,
        "image_relative_path": image_relative_path,
        "source_csv_line": "2",
        "annotation_count": str(len(annotation_rows)),
        "annotations_json": json.dumps(annotation_rows),
    }
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: values[field] for field in fieldnames})


def test_dataset_loads_valid_sample(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest)

    dataset = OrthodonticPlaquePart2Dataset(dataset_root=tmp_path, manifest_path=manifest)
    image, target = dataset[0]

    assert len(dataset) == 1
    assert tuple(image.shape) == (3, 3, 4)
    assert image.dtype.is_floating_point
    assert target.sample_id == "sample"
    assert target.patient_id == "patient0001"
    assert target.split == "train"
    assert target.boxes.shape == (1, 4)
    assert target.labels.tolist() == [1]
    assert target.tooth_ids.tolist() == [7]
    assert target.source_label_lines.tolist() == [1]


def test_dataset_loads_source_aware_manifest_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(
        tmp_path,
        manifest,
        fieldnames=SOURCE_AWARE_FIELDNAMES,
    )

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=tmp_path,
        manifest_path=manifest,
    )

    target = dataset[0][1]
    assert target.sample_id == "sample"
    assert target.image_relative_path.as_posix() == (
        "data/images/patient0001/sample.jpg"
    )
    assert target.variant == "original"


def test_dataset_filters_source_aware_variant(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest, fieldnames=SOURCE_AWARE_FIELDNAMES)
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    augmented = dict(rows[0])
    augmented.update(
        {
            "sample_id": "sample-brightness-up",
            "image_relative_path": (
                "data/images/patient0001/sample-brightness-up.jpg"
            ),
            "variant": "brightness-up",
        }
    )
    source_dark = dict(rows[0])
    source_dark.update(
        {
            "sample_id": "sample_dark",
            "image_relative_path": "data/images/patient0001/sample_dark.jpg",
            "derivative_group_id": "sample_dark",
            "variant": "original",
        }
    )
    Image.new("RGB", (4, 3), color=(40, 60, 80)).save(
        tmp_path / "data" / "images" / "patient0001" / "sample-brightness-up.jpg"
    )
    Image.new("RGB", (4, 3), color=(5, 5, 5)).save(
        tmp_path / "data" / "images" / "patient0001" / "sample_dark.jpg"
    )
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCE_AWARE_FIELDNAMES)
        writer.writeheader()
        writer.writerows((rows[0], augmented, source_dark))

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=tmp_path,
        manifest_path=manifest,
        split="train",
        variant="original",
        excluded_sample_id_suffixes=("_blur", "_dark", "_light"),
    )

    assert len(dataset) == 1
    assert dataset[0][1].sample_id == "sample"


def test_dataset_rejects_unsafe_excluded_suffix(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest, fieldnames=SOURCE_AWARE_FIELDNAMES)

    with pytest.raises(OrthodonticPlaqueDatasetError, match="normalized suffix"):
        OrthodonticPlaquePart2Dataset(
            dataset_root=tmp_path,
            manifest_path=manifest,
            excluded_sample_id_suffixes=("../dark",),
        )


def test_dataset_rejects_variant_filter_for_legacy_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest)

    with pytest.raises(OrthodonticPlaqueDatasetError, match="variant metadata"):
        OrthodonticPlaquePart2Dataset(
            dataset_root=tmp_path,
            manifest_path=manifest,
            split="train",
            variant="original",
        )


def test_dataset_filters_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest, split="validation")

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=tmp_path,
        manifest_path=manifest,
        split="validation",
    )

    assert len(dataset) == 1
    with pytest.raises(OrthodonticPlaqueDatasetError, match="contains no samples"):
        OrthodonticPlaquePart2Dataset(
            dataset_root=tmp_path,
            manifest_path=manifest,
            split="train",
        )


def test_dataset_rejects_missing_required_columns(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(
        tmp_path,
        manifest,
        fieldnames=tuple(field for field in FIELDNAMES if field != "annotations_json"),
    )

    with pytest.raises(OrthodonticPlaqueDatasetError, match="missing required columns"):
        OrthodonticPlaquePart2Dataset(dataset_root=tmp_path, manifest_path=manifest)


def test_dataset_rejects_unsafe_image_path(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "sample",
                "patient_id": "patient0001",
                "split": "train",
                "image_relative_path": "../escape.jpg",
                "source_csv_line": "2",
                "annotation_count": "1",
                "annotations_json": json.dumps([annotation()]),
            }
        )

    with pytest.raises(OrthodonticPlaqueDatasetError, match="unsafe segment"):
        OrthodonticPlaquePart2Dataset(dataset_root=tmp_path, manifest_path=manifest)


def test_dataset_rejects_invalid_annotation_shape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest, annotations=[annotation(width=0.0)])

    with pytest.raises(OrthodonticPlaqueDatasetError, match="non-positive size"):
        OrthodonticPlaquePart2Dataset(dataset_root=tmp_path, manifest_path=manifest)


def test_dataset_rejects_unsupported_source_class_id(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    write_fixture(tmp_path, manifest, annotations=[annotation(class_id=2)])

    with pytest.raises(OrthodonticPlaqueDatasetError, match="class_id must be 0 or 1"):
        OrthodonticPlaquePart2Dataset(dataset_root=tmp_path, manifest_path=manifest)
