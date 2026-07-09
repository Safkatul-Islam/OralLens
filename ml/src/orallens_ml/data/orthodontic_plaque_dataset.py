"""PyTorch dataset for the generated orthodontic plaque Part 2 manifest."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

_REQUIRED_COLUMNS = (
    "sample_id",
    "patient_id",
    "split",
    "image_relative_path",
    "source_csv_line",
    "annotation_count",
    "annotations_json",
)
_SPLITS = frozenset({"train", "validation", "test"})
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class OrthodonticPlaqueDatasetError(ValueError):
    """Raised when the generated manifest cannot be loaded safely."""


@dataclass(frozen=True, slots=True)
class OrthodonticPlaqueTarget:
    """One target object returned with each image."""

    sample_id: str
    patient_id: str
    split: str
    source_csv_line: int
    boxes: Tensor
    labels: Tensor
    tooth_ids: Tensor
    source_label_lines: Tensor


@dataclass(frozen=True, slots=True)
class _ManifestSample:
    sample_id: str
    patient_id: str
    split: str
    image_relative_path: PurePosixPath
    source_csv_line: int
    annotations: tuple[dict[str, object], ...]


class OrthodonticPlaquePart2Dataset(Dataset[tuple[Tensor, OrthodonticPlaqueTarget]]):
    """Load retained Part 2 samples from the generated manifest CSV."""

    def __init__(
        self,
        *,
        dataset_root: Path,
        manifest_path: Path,
        split: Literal["train", "validation", "test"] | None = None,
    ) -> None:
        root = Path(dataset_root)
        if root.is_symlink() or not root.is_dir():
            raise OrthodonticPlaqueDatasetError(
                f"Dataset root is not a regular directory: {root}"
            )
        manifest = Path(manifest_path)
        if manifest.is_symlink() or not manifest.is_file():
            raise OrthodonticPlaqueDatasetError(
                f"Manifest is not a regular file: {manifest}"
            )
        if split is not None and split not in _SPLITS:
            raise OrthodonticPlaqueDatasetError(f"Unknown split: {split!r}")

        self.dataset_root = root.resolve(strict=True)
        self.manifest_path = manifest
        samples = _load_samples(manifest)
        if split is not None:
            samples = tuple(sample for sample in samples if sample.split == split)
        if not samples:
            raise OrthodonticPlaqueDatasetError("Dataset split contains no samples")
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[Tensor, OrthodonticPlaqueTarget]:
        sample = self._samples[index]
        image_path = self.dataset_root.joinpath(*sample.image_relative_path.parts).resolve(
            strict=False
        )
        if not image_path.is_relative_to(self.dataset_root):
            raise OrthodonticPlaqueDatasetError(
                f"Image path escapes dataset root: {sample.sample_id}"
            )
        if image_path.is_symlink() or not image_path.is_file():
            raise OrthodonticPlaqueDatasetError(f"Image is missing: {sample.sample_id}")

        with Image.open(image_path) as image:
            image_tensor = _pil_to_float_tensor(image.convert("RGB"))
        target = _target_from_sample(sample)
        return image_tensor, target


def _load_samples(manifest_path: Path) -> tuple[_ManifestSample, ...]:
    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_header(reader.fieldnames)
            samples = tuple(
                _sample_from_row(row, line_number)
                for line_number, row in enumerate(reader, start=2)
            )
    except UnicodeDecodeError as exc:
        raise OrthodonticPlaqueDatasetError("Manifest must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise OrthodonticPlaqueDatasetError(f"Manifest CSV is malformed: {exc}") from exc
    if not samples:
        raise OrthodonticPlaqueDatasetError("Manifest must contain at least one row")
    return samples


def _validate_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise OrthodonticPlaqueDatasetError("Manifest is missing a header row")
    missing = [column for column in _REQUIRED_COLUMNS if column not in fieldnames]
    if missing:
        raise OrthodonticPlaqueDatasetError(
            f"Manifest is missing required columns: {', '.join(missing)}"
        )


def _sample_from_row(row: dict[str, str | None], line_number: int) -> _ManifestSample:
    values = {
        column: _required_value(row.get(column), column, line_number)
        for column in _REQUIRED_COLUMNS
    }
    split = values["split"]
    if split not in _SPLITS:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: unknown split value: {split!r}"
        )
    image_relative_path = _validate_image_path(values["image_relative_path"], line_number)
    source_csv_line = _positive_int(values["source_csv_line"], "source_csv_line", line_number)
    annotation_count = _positive_int(
        values["annotation_count"],
        "annotation_count",
        line_number,
    )
    annotations = _parse_annotations(values["annotations_json"], line_number)
    if len(annotations) != annotation_count:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation_count does not match annotations_json"
        )
    return _ManifestSample(
        sample_id=values["sample_id"],
        patient_id=values["patient_id"],
        split=split,
        image_relative_path=image_relative_path,
        source_csv_line=source_csv_line,
        annotations=annotations,
    )


def _required_value(value: str | None, column: str, line_number: int) -> str:
    normalized = value.strip() if value is not None else ""
    if not normalized:
        raise OrthodonticPlaqueDatasetError(f"Line {line_number}: {column} must not be empty")
    if "\x00" in normalized:
        raise OrthodonticPlaqueDatasetError(f"Line {line_number}: {column} contains a null byte")
    return normalized


def _validate_image_path(value: str, line_number: int) -> PurePosixPath:
    if "\\" in value or ":" in value:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: image path must use POSIX separators"
        )
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: image path contains an unsafe segment"
        )
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: image path must be relative"
        )
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: unsupported image extension: {path.suffix or '<none>'}"
        )
    return path


def _positive_int(value: str, column: str, line_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: {column} must be an integer"
        ) from exc
    if parsed <= 0:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: {column} must be positive"
        )
    return parsed


def _parse_annotations(value: str, line_number: int) -> tuple[dict[str, object], ...]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotations_json is malformed"
        ) from exc
    if not isinstance(payload, list) or not payload:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotations_json must be a non-empty list"
        )
    annotations: list[dict[str, object]] = []
    for annotation_index, annotation in enumerate(payload):
        if not isinstance(annotation, dict):
            raise OrthodonticPlaqueDatasetError(
                f"Line {line_number}: annotation {annotation_index} must be an object"
            )
        _validate_annotation(annotation, line_number, annotation_index)
        annotations.append(annotation)
    return tuple(annotations)


def _validate_annotation(
    annotation: dict[str, object],
    line_number: int,
    annotation_index: int,
) -> None:
    required = (
        "class_id",
        "x_center",
        "y_center",
        "width",
        "height",
        "tooth_id",
        "source_label_line",
    )
    missing = [field for field in required if field not in annotation]
    if missing:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation {annotation_index} is missing {', '.join(missing)}"
        )
    class_id = _annotation_int(annotation["class_id"], "class_id", line_number, annotation_index)
    tooth_id = _annotation_int(annotation["tooth_id"], "tooth_id", line_number, annotation_index)
    source_label_line = _annotation_int(
        annotation["source_label_line"],
        "source_label_line",
        line_number,
        annotation_index,
    )
    if class_id < 0 or tooth_id <= 0 or source_label_line <= 0:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation {annotation_index} has invalid integer fields"
        )
    for field in ("x_center", "y_center", "width", "height"):
        value = _annotation_float(annotation[field], field, line_number, annotation_index)
        if not 0.0 <= value <= 1.0:
            raise OrthodonticPlaqueDatasetError(
                f"Line {line_number}: annotation {annotation_index} {field} is out of bounds"
            )
    if float(annotation["width"]) <= 0.0 or float(annotation["height"]) <= 0.0:
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation {annotation_index} has non-positive size"
        )


def _annotation_int(
    value: object,
    field: str,
    line_number: int,
    annotation_index: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation {annotation_index} {field} must be an integer"
        )
    return value


def _annotation_float(
    value: object,
    field: str,
    line_number: int,
    annotation_index: int,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OrthodonticPlaqueDatasetError(
            f"Line {line_number}: annotation {annotation_index} {field} must be numeric"
        )
    return float(value)


def _pil_to_float_tensor(image: Image.Image) -> Tensor:
    width, height = image.size
    data = torch.tensor(list(image.getdata()), dtype=torch.float32)
    return data.view(height, width, 3).permute(2, 0, 1).contiguous().div(255.0)


def _target_from_sample(sample: _ManifestSample) -> OrthodonticPlaqueTarget:
    boxes = []
    labels = []
    tooth_ids = []
    source_label_lines = []
    for annotation in sample.annotations:
        x_center = float(annotation["x_center"])
        y_center = float(annotation["y_center"])
        width = float(annotation["width"])
        height = float(annotation["height"])
        boxes.append(
            [
                x_center - width / 2.0,
                y_center - height / 2.0,
                x_center + width / 2.0,
                y_center + height / 2.0,
            ]
        )
        labels.append(int(annotation["class_id"]))
        tooth_ids.append(int(annotation["tooth_id"]))
        source_label_lines.append(int(annotation["source_label_line"]))

    return OrthodonticPlaqueTarget(
        sample_id=sample.sample_id,
        patient_id=sample.patient_id,
        split=sample.split,
        source_csv_line=sample.source_csv_line,
        boxes=torch.tensor(boxes, dtype=torch.float32),
        labels=torch.tensor(labels, dtype=torch.int64),
        tooth_ids=torch.tensor(tooth_ids, dtype=torch.int64),
        source_label_lines=torch.tensor(source_label_lines, dtype=torch.int64),
    )
