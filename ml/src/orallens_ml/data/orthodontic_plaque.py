"""Read-only audit for the orthodontic plaque source dataset."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from PIL import Image, UnidentifiedImageError

from orallens_ml.data.acquisition import AcquisitionError
from orallens_ml.data.detection_manifest import (
    DETECTION_MANIFEST_IDENTITY_COLUMNS,
    DETECTION_MANIFEST_SCHEMA_VERSION,
    DetectionManifestError,
    DetectionManifestIdentity,
    validate_detection_manifest_identities,
)

_REQUIRED_SPLIT_COLUMNS = ("patient", "Image-Filename", "Recommend")
_ALLOWED_SPLITS = frozenset({"train", "val", "validation", "test"})
_PATIENT_PATTERN = re.compile(r"^patient\d{4}$", re.IGNORECASE)
_SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_LABEL_FIELD_COUNT = 6
_MAX_ISSUE_EXAMPLES = 5
_ANNOTATION_FILTER_CODES = frozenset(
    {
        "invalid_label_field_count",
        "invalid_label_number",
        "unsupported_class_id",
        "label_coordinate_out_of_bounds",
        "non_positive_box_size",
        "unknown_tooth_id",
    }
)
_MANIFEST_IMAGE_COLUMNS = ("sample_id", "image_relative_path")
_DERIVATIVE_SUFFIXES = (
    ("_brightness-down", "brightness-down"),
    ("_brightness-up", "brightness-up"),
    ("_flip_horizontal", "flip-horizontal"),
    ("_rotate-left-15", "rotate-left-15"),
    ("_rotate-right-15", "rotate-right-15"),
)


class OrthodonticPlaqueAuditError(AcquisitionError):
    """Raised when the source orthodontic plaque dataset violates its contract."""


@dataclass(frozen=True, slots=True)
class OrthodonticPlaqueAuditIssue:
    """One source-dataset issue found during row-level validation."""

    code: str
    severity: str
    message: str
    csv_line: int
    label_line: int | None = None


@dataclass(frozen=True, slots=True)
class OrthodonticPlaqueAuditReport:
    """Summary of a read-only source-dataset audit."""

    dataset_root: Path
    csv_row_count: int
    patient_count: int
    split_counts: dict[str, int]
    image_count: int
    label_file_count: int
    annotation_count: int
    tooth_position_ids: tuple[int, ...]
    issues: tuple[OrthodonticPlaqueAuditIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def issue_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.code for issue in self.issues).items()))

    @property
    def issue_examples(self) -> dict[str, tuple[OrthodonticPlaqueAuditIssue, ...]]:
        examples: dict[str, list[OrthodonticPlaqueAuditIssue]] = {}
        for issue in self.issues:
            bucket = examples.setdefault(issue.code, [])
            if len(bucket) < _MAX_ISSUE_EXAMPLES:
                bucket.append(issue)
        return {code: tuple(values) for code, values in sorted(examples.items())}


@dataclass(frozen=True, slots=True)
class Part2Annotation:
    """One validated source annotation retained for training."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float
    tooth_id: int
    source_label_line: int


@dataclass(frozen=True, slots=True)
class Part2Sample:
    """One retained Part 2 image sample with validated annotations."""

    sample_id: str
    patient_id: str
    derivative_group_id: str
    variant: str
    split: Literal["train", "validation", "test"]
    image_relative_path: PurePosixPath
    source_image_filename: str
    source_csv_line: int
    annotations: tuple[Part2Annotation, ...]


@dataclass(frozen=True, slots=True)
class Part2Exclusion:
    """One excluded sample or filtered annotation with its source location."""

    sample_id: str | None
    patient_id: str | None
    split: str | None
    source_image_filename: str | None
    source_csv_line: int
    code: str
    message: str
    label_line: int | None = None
    filtered_annotation_count: int = 0


@dataclass(frozen=True, slots=True)
class Part2ExclusionPolicy:
    """Explicit policy for transforming audited source data into a manifest."""

    filter_annotation_issue_codes: frozenset[str] = _ANNOTATION_FILTER_CODES
    exclude_empty_after_filter: bool = True


@dataclass(frozen=True, slots=True)
class Part2BuildReport:
    """Summary of a Part 2 source-manifest build."""

    source_csv_rows: int
    included_samples: int
    excluded_samples: int
    valid_annotation_count: int
    filtered_annotation_count: int
    split_counts: dict[str, int]
    issue_counts: dict[str, int]
    exclusions: tuple[Part2Exclusion, ...]


@dataclass(frozen=True, slots=True)
class Part2ImageIssue:
    """One image validation issue from a generated Part 2 manifest."""

    code: str
    message: str
    manifest_line: int
    sample_id: str | None = None


@dataclass(frozen=True, slots=True)
class Part2ImageValidationReport:
    """Read-only image decoding summary for a generated Part 2 manifest."""

    manifest_path: Path
    dataset_root: Path
    total_records: int
    decoded_images: int
    mode_counts: dict[str, int]
    size_counts: dict[str, int]
    issues: tuple[Part2ImageIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues

    @property
    def issue_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.code for issue in self.issues).items()))


def audit_orthodontic_plaque_source(dataset_root: Path) -> OrthodonticPlaqueAuditReport:
    """Validate the extracted orthodontic plaque source dataset without writing files."""

    root = _validate_dataset_root(dataset_root)
    allowed_tooth_ids = _load_tooth_position_ids(root / "metadata" / "teeth_position_mapping.json")
    rows = _load_split_rows(root / "data_splits.csv")

    split_counts: Counter[str] = Counter()
    patients: set[str] = set()
    images_seen: set[str] = set()
    labels_seen: set[str] = set()
    annotation_count = 0
    observed_tooth_ids: set[int] = set()
    issues: list[OrthodonticPlaqueAuditIssue] = []

    for line_number, row in rows:
        patient = _required_row_value(row, "patient", line_number, issues)
        image_filename = _required_row_value(row, "Image-Filename", line_number, issues)
        split = _required_row_value(row, "Recommend", line_number, issues)
        if patient is None or image_filename is None or split is None:
            continue
        split = split.casefold()
        if split not in _ALLOWED_SPLITS:
            _add_issue(
                issues,
                code="unsupported_split",
                message=f"Unsupported split value: {split!r}",
                csv_line=line_number,
            )
        if not _PATIENT_PATTERN.fullmatch(patient):
            _add_issue(
                issues,
                code="invalid_patient",
                message=f"Invalid patient identifier: {patient!r}",
                csv_line=line_number,
            )
            continue

        image_relative = _validate_image_filename(image_filename, line_number, issues)
        if image_relative is None:
            continue
        image_stem = PurePosixPath(image_relative).stem
        expected_prefix = f"{patient}_"
        if not image_stem.casefold().startswith(expected_prefix.casefold()):
            _add_issue(
                issues,
                code="image_patient_mismatch",
                message="Image filename does not match patient",
                csv_line=line_number,
            )

        image_path = _resolve_inside(
            root,
            PurePosixPath("data") / "images" / patient / image_relative,
            line_number=line_number,
        )
        label_path = _resolve_inside(
            root,
            PurePosixPath("data") / "labels" / patient / f"{image_stem}.txt",
            line_number=line_number,
        )
        if not image_path.is_file():
            _add_issue(
                issues,
                code="missing_image",
                message=f"Image file is missing: {image_path}",
                csv_line=line_number,
            )
        if not label_path.is_file():
            _add_issue(
                issues,
                code="missing_label",
                message=f"Label file is missing: {label_path}",
                csv_line=line_number,
            )
            continue

        label_annotations, label_tooth_ids = _audit_label_file(
            label_path,
            allowed_tooth_ids,
            line_number=line_number,
            issues=issues,
        )
        annotation_count += label_annotations
        observed_tooth_ids.update(label_tooth_ids)
        if image_path.is_file():
            images_seen.add(str(image_path))
        labels_seen.add(str(label_path))
        patients.add(patient.casefold())
        if split in _ALLOWED_SPLITS:
            split_counts[split] += 1

    return OrthodonticPlaqueAuditReport(
        dataset_root=root,
        csv_row_count=len(rows),
        patient_count=len(patients),
        split_counts=dict(sorted(split_counts.items())),
        image_count=len(images_seen),
        label_file_count=len(labels_seen),
        annotation_count=annotation_count,
        tooth_position_ids=tuple(sorted(observed_tooth_ids)),
        issues=tuple(issues),
    )


def build_orthodontic_plaque_part2_manifest(
    dataset_root: Path,
    *,
    policy: Part2ExclusionPolicy = Part2ExclusionPolicy(),
) -> tuple[tuple[Part2Sample, ...], Part2BuildReport]:
    """Build retained Part 2 samples and exclusions without mutating source data."""

    root = _validate_dataset_root(dataset_root)
    allowed_tooth_ids = _load_tooth_position_ids(root / "metadata" / "teeth_position_mapping.json")
    rows = _load_split_rows(root / "data_splits.csv")

    samples: list[Part2Sample] = []
    exclusions: list[Part2Exclusion] = []
    patient_splits: dict[str, set[str]] = {}

    for line_number, row in rows:
        row_issues: list[OrthodonticPlaqueAuditIssue] = []
        patient = _required_row_value(row, "patient", line_number, row_issues)
        image_filename = _required_row_value(row, "Image-Filename", line_number, row_issues)
        split_value = _required_row_value(row, "Recommend", line_number, row_issues)
        if patient is None or image_filename is None or split_value is None:
            exclusions.extend(
                _exclusions_from_issues(row_issues, None, None, None, None, 0)
            )
            continue

        split = _normalize_split(split_value)
        sample_id = PurePosixPath(image_filename).stem if image_filename else None
        if split is None:
            row_issues.append(
                OrthodonticPlaqueAuditIssue(
                    code="unsupported_split",
                    severity="error",
                    message=f"Unsupported split value: {split_value.casefold()!r}",
                    csv_line=line_number,
                )
            )
        if not _PATIENT_PATTERN.fullmatch(patient):
            row_issues.append(
                OrthodonticPlaqueAuditIssue(
                    code="invalid_patient",
                    severity="error",
                    message=f"Invalid patient identifier: {patient!r}",
                    csv_line=line_number,
                )
            )

        image_relative = _validate_image_filename(image_filename, line_number, row_issues)
        if row_issues or image_relative is None or split is None:
            exclusions.extend(
                _exclusions_from_issues(
                    row_issues,
                    sample_id,
                    patient,
                    split,
                    image_filename,
                    0,
                )
            )
            continue

        image_stem = image_relative.stem
        if not image_stem.casefold().startswith(f"{patient}_".casefold()):
            row_issues.append(
                OrthodonticPlaqueAuditIssue(
                    code="image_patient_mismatch",
                    severity="error",
                    message="Image filename does not match patient",
                    csv_line=line_number,
                )
            )

        image_path = _resolve_inside(
            root,
            PurePosixPath("data") / "images" / patient / image_relative,
            line_number=line_number,
        )
        label_path = _resolve_inside(
            root,
            PurePosixPath("data") / "labels" / patient / f"{image_stem}.txt",
            line_number=line_number,
        )
        if not image_path.is_file():
            row_issues.append(
                OrthodonticPlaqueAuditIssue(
                    code="missing_image",
                    severity="error",
                    message=f"Image file is missing: {image_path}",
                    csv_line=line_number,
                )
            )
        if not label_path.is_file():
            row_issues.append(
                OrthodonticPlaqueAuditIssue(
                    code="missing_label",
                    severity="error",
                    message=f"Label file is missing: {label_path}",
                    csv_line=line_number,
                )
            )
            annotations: list[Part2Annotation] = []
            annotation_issues: list[OrthodonticPlaqueAuditIssue] = []
        else:
            annotations, annotation_issues = _collect_valid_label_annotations(
                label_path,
                allowed_tooth_ids,
                line_number=line_number,
            )
        row_issues.extend(annotation_issues)
        filtered_count = sum(
            1 for issue in annotation_issues if issue.code in policy.filter_annotation_issue_codes
        )
        exclusions.extend(
            _exclusions_from_issues(
                row_issues,
                sample_id,
                patient,
                split,
                image_filename,
                filtered_count,
            )
        )

        sample_level_error = any(
            issue.code not in policy.filter_annotation_issue_codes for issue in row_issues
        )
        if sample_level_error:
            continue
        if policy.exclude_empty_after_filter and not annotations:
            if not any(issue.code == "empty_label" for issue in row_issues):
                exclusions.append(
                    Part2Exclusion(
                        sample_id=sample_id,
                        patient_id=patient,
                        split=split,
                        source_image_filename=image_filename,
                        source_csv_line=line_number,
                        code="no_valid_annotations",
                        message="Sample has no valid annotations after filtering",
                        filtered_annotation_count=filtered_count,
                    )
                )
            continue

        image_manifest_path = PurePosixPath("data") / "images" / patient / image_relative
        derivative_group_id, variant = _derive_part2_variant_identity(image_stem)
        samples.append(
            Part2Sample(
                sample_id=image_stem,
                patient_id=patient.casefold(),
                derivative_group_id=derivative_group_id,
                variant=variant,
                split=split,
                image_relative_path=image_manifest_path,
                source_image_filename=image_filename,
                source_csv_line=line_number,
                annotations=tuple(annotations),
            )
        )
        patient_splits.setdefault(patient.casefold(), set()).add(split)

    leaking_patients = sorted(
        patient for patient, split_names in patient_splits.items() if len(split_names) > 1
    )
    if leaking_patients:
        raise OrthodonticPlaqueAuditError(
            f"Publisher splits leak across {len(leaking_patients)} patient(s)"
        )
    sample_ids = [sample.sample_id for sample in samples]
    duplicate_sample_ids = sorted(
        sample_id for sample_id, count in Counter(sample_ids).items() if count > 1
    )
    if duplicate_sample_ids:
        raise OrthodonticPlaqueAuditError(
            f"Duplicate retained sample IDs: {', '.join(duplicate_sample_ids[:5])}"
        )

    included_sample_ids = set(sample_ids)
    split_counts = Counter(sample.split for sample in samples)
    issue_counts = Counter(exclusion.code for exclusion in exclusions)
    report = Part2BuildReport(
        source_csv_rows=len(rows),
        included_samples=len(samples),
        excluded_samples=len(
            {
                exclusion.sample_id
                for exclusion in exclusions
                if exclusion.sample_id is not None
                and exclusion.sample_id not in included_sample_ids
            }
        ),
        valid_annotation_count=sum(len(sample.annotations) for sample in samples),
        filtered_annotation_count=sum(
            1
            for exclusion in exclusions
            if exclusion.code in policy.filter_annotation_issue_codes
        ),
        split_counts=dict(sorted(split_counts.items())),
        issue_counts=dict(sorted(issue_counts.items())),
        exclusions=tuple(exclusions),
    )
    return tuple(samples), report


def preflight_part2_output_paths(manifest_output: Path, exclusions_output: Path) -> None:
    """Validate both output paths before writing either file."""

    manifest_path = Path(manifest_output)
    exclusions_path = Path(exclusions_output)
    if manifest_path == exclusions_path:
        raise OrthodonticPlaqueAuditError("Manifest and exclusions outputs must differ")
    _prepare_new_output_file(manifest_path)
    _prepare_new_output_file(exclusions_path)


def write_part2_manifest_csv(
    samples: tuple[Part2Sample, ...],
    output_path: Path,
    *,
    dataset_id: str,
    dataset_version: int,
    source_family_id: str,
    source_artifact_id: str,
) -> None:
    """Write retained Part 2 samples as a UTF-8 CSV with embedded annotations JSON."""

    identities = (
        DetectionManifestIdentity(
            manifest_schema_version=DETECTION_MANIFEST_SCHEMA_VERSION,
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            source_family_id=source_family_id,
            source_artifact_id=source_artifact_id,
            sample_id=sample.sample_id,
            split_group_id=sample.patient_id,
            derivative_group_id=sample.derivative_group_id,
            variant=sample.variant,
            split=sample.split,
        )
        for sample in samples
    )
    try:
        validated_identities = validate_detection_manifest_identities(identities)
    except DetectionManifestError as exc:
        raise OrthodonticPlaqueAuditError(str(exc)) from exc

    path = _prepare_new_output_file(output_path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                *DETECTION_MANIFEST_IDENTITY_COLUMNS,
                "sample_id",
                "patient_id",
                "split",
                "image_relative_path",
                "source_image_filename",
                "source_csv_line",
                "annotation_count",
                "annotations_json",
            ),
        )
        writer.writeheader()
        for sample, identity in zip(samples, validated_identities, strict=True):
            writer.writerow(
                {
                    "manifest_schema_version": str(identity.manifest_schema_version),
                    "dataset_id": identity.dataset_id,
                    "dataset_version": str(identity.dataset_version),
                    "source_family_id": identity.source_family_id,
                    "source_artifact_id": identity.source_artifact_id,
                    "split_group_id": identity.split_group_id,
                    "derivative_group_id": identity.derivative_group_id,
                    "variant": identity.variant,
                    "sample_id": sample.sample_id,
                    "patient_id": sample.patient_id,
                    "split": sample.split,
                    "image_relative_path": sample.image_relative_path.as_posix(),
                    "source_image_filename": sample.source_image_filename,
                    "source_csv_line": str(sample.source_csv_line),
                    "annotation_count": str(len(sample.annotations)),
                    "annotations_json": json.dumps(
                        [
                            {
                                "class_id": annotation.class_id,
                                "x_center": annotation.x_center,
                                "y_center": annotation.y_center,
                                "width": annotation.width,
                                "height": annotation.height,
                                "tooth_id": annotation.tooth_id,
                                "source_label_line": annotation.source_label_line,
                            }
                            for annotation in sample.annotations
                        ],
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                }
            )


def _derive_part2_variant_identity(sample_id: str) -> tuple[str, str]:
    """Return the original-image family and exact publisher variant."""

    for suffix, variant in _DERIVATIVE_SUFFIXES:
        if sample_id.endswith(suffix):
            derivative_group_id = sample_id.removesuffix(suffix)
            if not derivative_group_id:
                raise OrthodonticPlaqueAuditError(
                    f"Invalid derivative sample identity: {sample_id!r}"
                )
            return derivative_group_id, variant
    return sample_id, "original"


def write_part2_exclusions_csv(
    exclusions: tuple[Part2Exclusion, ...],
    output_path: Path,
) -> None:
    """Write excluded samples and filtered annotations as a UTF-8 CSV."""

    path = _prepare_new_output_file(output_path)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_id",
                "patient_id",
                "split",
                "source_image_filename",
                "source_csv_line",
                "label_line",
                "code",
                "message",
                "filtered_annotation_count",
            ),
        )
        writer.writeheader()
        for exclusion in exclusions:
            writer.writerow(
                {
                    "sample_id": exclusion.sample_id or "",
                    "patient_id": exclusion.patient_id or "",
                    "split": exclusion.split or "",
                    "source_image_filename": exclusion.source_image_filename or "",
                    "source_csv_line": str(exclusion.source_csv_line),
                    "label_line": "" if exclusion.label_line is None else str(exclusion.label_line),
                    "code": exclusion.code,
                    "message": exclusion.message,
                    "filtered_annotation_count": str(exclusion.filtered_annotation_count),
                }
            )


def validate_part2_manifest_images(
    dataset_root: Path,
    manifest_path: Path,
) -> Part2ImageValidationReport:
    """Open every retained image referenced by a generated Part 2 manifest."""

    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise OrthodonticPlaqueAuditError(f"Dataset root is not a regular directory: {root}")
    resolved_root = root.resolve(strict=True)
    manifest = Path(manifest_path)
    if manifest.is_symlink() or not manifest.is_file():
        raise OrthodonticPlaqueAuditError(f"Manifest is not a regular file: {manifest}")

    rows = _load_manifest_image_rows(manifest)
    decoded_images = 0
    mode_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    issues: list[Part2ImageIssue] = []

    for manifest_line, row in rows:
        sample_id = _manifest_value(row, "sample_id")
        relative_value = _manifest_value(row, "image_relative_path")
        if sample_id is None or relative_value is None:
            issues.append(
                Part2ImageIssue(
                    code="missing_manifest_value",
                    message="Manifest row is missing sample_id or image_relative_path",
                    manifest_line=manifest_line,
                    sample_id=sample_id,
                )
            )
            continue
        relative_path = _validate_manifest_image_path(
            relative_value,
            manifest_line=manifest_line,
            sample_id=sample_id,
            issues=issues,
        )
        if relative_path is None:
            continue
        image_path = resolved_root.joinpath(*relative_path.parts).resolve(strict=False)
        if not image_path.is_relative_to(resolved_root):
            issues.append(
                Part2ImageIssue(
                    code="path_escapes_dataset_root",
                    message="Resolved image path escapes dataset root",
                    manifest_line=manifest_line,
                    sample_id=sample_id,
                )
            )
            continue
        if image_path.is_symlink() or not image_path.is_file():
            issues.append(
                Part2ImageIssue(
                    code="missing_image",
                    message="Referenced image does not exist or is not a regular file",
                    manifest_line=manifest_line,
                    sample_id=sample_id,
                )
            )
            continue

        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height = image.size
                mode = image.mode
        except (OSError, UnidentifiedImageError) as exc:
            issues.append(
                Part2ImageIssue(
                    code="unreadable_image",
                    message=f"Image cannot be decoded: {exc}",
                    manifest_line=manifest_line,
                    sample_id=sample_id,
                )
            )
            continue
        if width <= 0 or height <= 0:
            issues.append(
                Part2ImageIssue(
                    code="invalid_image_dimensions",
                    message="Image has non-positive dimensions",
                    manifest_line=manifest_line,
                    sample_id=sample_id,
                )
            )
            continue
        decoded_images += 1
        mode_counts[mode] += 1
        size_counts[f"{width}x{height}"] += 1

    return Part2ImageValidationReport(
        manifest_path=manifest,
        dataset_root=resolved_root,
        total_records=len(rows),
        decoded_images=decoded_images,
        mode_counts=dict(sorted(mode_counts.items())),
        size_counts=dict(sorted(size_counts.items())),
        issues=tuple(issues),
    )


def _normalize_split(value: str) -> Literal["train", "validation", "test"] | None:
    normalized = value.casefold()
    if normalized == "val":
        return "validation"
    if normalized in {"train", "validation", "test"}:
        return normalized  # type: ignore[return-value]
    return None


def _load_manifest_image_rows(path: Path) -> tuple[tuple[int, dict[str, str | None]], ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_manifest_image_header(reader.fieldnames)
            rows = tuple((line_number, row) for line_number, row in enumerate(reader, start=2))
    except UnicodeDecodeError as exc:
        raise OrthodonticPlaqueAuditError("Manifest must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise OrthodonticPlaqueAuditError(f"Manifest CSV is malformed: {exc}") from exc
    if not rows:
        raise OrthodonticPlaqueAuditError("Manifest must contain at least one row")
    return rows


def _validate_manifest_image_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise OrthodonticPlaqueAuditError("Manifest is missing a header row")
    missing = [column for column in _MANIFEST_IMAGE_COLUMNS if column not in fieldnames]
    if missing:
        raise OrthodonticPlaqueAuditError(
            f"Manifest is missing required columns: {', '.join(missing)}"
        )


def _manifest_value(row: dict[str, str | None], column: str) -> str | None:
    value = row.get(column)
    normalized = value.strip() if value is not None else ""
    if not normalized or "\x00" in normalized:
        return None
    return normalized


def _validate_manifest_image_path(
    value: str,
    *,
    manifest_line: int,
    sample_id: str,
    issues: list[Part2ImageIssue],
) -> PurePosixPath | None:
    if "\\" in value or ":" in value:
        issues.append(
            Part2ImageIssue(
                code="unsafe_image_path",
                message="Image path contains an unsafe Windows path marker",
                manifest_line=manifest_line,
                sample_id=sample_id,
            )
        )
        return None
    if any(part in {"", ".", ".."} for part in value.split("/")):
        issues.append(
            Part2ImageIssue(
                code="unsafe_image_path",
                message="Image path contains an unsafe path segment",
                manifest_line=manifest_line,
                sample_id=sample_id,
            )
        )
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        issues.append(
            Part2ImageIssue(
                code="unsafe_image_path",
                message="Image path must be relative",
                manifest_line=manifest_line,
                sample_id=sample_id,
            )
        )
        return None
    if path.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        issues.append(
            Part2ImageIssue(
                code="unsupported_image_extension",
                message=f"Unsupported image extension: {path.suffix or '<none>'}",
                manifest_line=manifest_line,
                sample_id=sample_id,
            )
        )
        return None
    return path


def _collect_valid_label_annotations(
    label_path: Path,
    allowed_tooth_ids: frozenset[int],
    *,
    line_number: int,
) -> tuple[list[Part2Annotation], list[OrthodonticPlaqueAuditIssue]]:
    annotations: list[Part2Annotation] = []
    issues: list[OrthodonticPlaqueAuditIssue] = []
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        _add_issue(
            issues,
            code="unreadable_label",
            message=f"Label file is unreadable: {label_path}",
            csv_line=line_number,
        )
        return annotations, issues
    if not lines:
        _add_issue(
            issues,
            code="empty_label",
            message="Label file is empty",
            csv_line=line_number,
        )
        return annotations, issues

    for label_line_number, raw_line in enumerate(lines, start=1):
        parsed = _parse_label_row(
            raw_line,
            allowed_tooth_ids,
            csv_line=line_number,
            label_line=label_line_number,
        )
        if isinstance(parsed, OrthodonticPlaqueAuditIssue):
            issues.append(parsed)
        else:
            annotations.append(parsed)
    return annotations, issues


def _parse_label_row(
    raw_line: str,
    allowed_tooth_ids: frozenset[int],
    *,
    csv_line: int,
    label_line: int,
) -> Part2Annotation | OrthodonticPlaqueAuditIssue:
    parts = raw_line.split()
    if len(parts) != _LABEL_FIELD_COUNT:
        return OrthodonticPlaqueAuditIssue(
            code="invalid_label_field_count",
            severity="error",
            message=f"Label row must have {_LABEL_FIELD_COUNT} fields",
            csv_line=csv_line,
            label_line=label_line,
        )
    try:
        class_id = int(parts[0])
        x_center, y_center, width, height = (float(value) for value in parts[1:5])
        tooth_id = int(parts[5])
    except ValueError:
        return OrthodonticPlaqueAuditIssue(
            code="invalid_label_number",
            severity="error",
            message="Label row contains invalid numbers",
            csv_line=csv_line,
            label_line=label_line,
        )

    if class_id not in {0, 1}:
        return OrthodonticPlaqueAuditIssue(
            code="unsupported_class_id",
            severity="error",
            message="Label row class id must be 0 or 1",
            csv_line=csv_line,
            label_line=label_line,
        )
    if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
        return OrthodonticPlaqueAuditIssue(
            code="label_coordinate_out_of_bounds",
            severity="error",
            message="Label row coordinates are out of bounds",
            csv_line=csv_line,
            label_line=label_line,
        )
    if width <= 0.0 or height <= 0.0:
        return OrthodonticPlaqueAuditIssue(
            code="non_positive_box_size",
            severity="error",
            message="Label row has a non-positive box size",
            csv_line=csv_line,
            label_line=label_line,
        )
    if tooth_id not in allowed_tooth_ids:
        return OrthodonticPlaqueAuditIssue(
            code="unknown_tooth_id",
            severity="error",
            message=f"Label row has unknown tooth id {tooth_id}",
            csv_line=csv_line,
            label_line=label_line,
        )
    return Part2Annotation(
        class_id=class_id,
        x_center=x_center,
        y_center=y_center,
        width=width,
        height=height,
        tooth_id=tooth_id,
        source_label_line=label_line,
    )


def _exclusions_from_issues(
    issues: list[OrthodonticPlaqueAuditIssue],
    sample_id: str | None,
    patient_id: str | None,
    split: str | None,
    source_image_filename: str | None,
    filtered_annotation_count: int,
) -> list[Part2Exclusion]:
    return [
        Part2Exclusion(
            sample_id=sample_id,
            patient_id=patient_id,
            split=split,
            source_image_filename=source_image_filename,
            source_csv_line=issue.csv_line,
            code=issue.code,
            message=issue.message,
            label_line=issue.label_line,
            filtered_annotation_count=(
                filtered_annotation_count if issue.code in _ANNOTATION_FILTER_CODES else 0
            ),
        )
        for issue in issues
    ]


def _prepare_new_output_file(output_path: Path) -> Path:
    path = Path(output_path)
    if path.exists() or path.is_symlink():
        raise OrthodonticPlaqueAuditError(f"Output file already exists: {path}")
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise OrthodonticPlaqueAuditError(f"Output parent is not a directory: {path.parent}")
    return path


def _validate_dataset_root(dataset_root: Path) -> Path:
    root = Path(dataset_root)
    if root.is_symlink() or not root.is_dir():
        raise OrthodonticPlaqueAuditError(f"Dataset root is not a regular directory: {root}")
    resolved = root.resolve(strict=True)
    for required in ("data_splits.csv", "metadata/teeth_position_mapping.json"):
        candidate = resolved.joinpath(*PurePosixPath(required).parts)
        _validate_required_child(resolved, candidate, required, expected="file")
    for required in ("data/images", "data/labels"):
        candidate = resolved.joinpath(*PurePosixPath(required).parts)
        _validate_required_child(resolved, candidate, required, expected="directory")
    return resolved


def _validate_required_child(
    root: Path,
    candidate: Path,
    display_name: str,
    *,
    expected: Literal["file", "directory"],
) -> None:
    resolved_candidate = candidate.resolve(strict=False)
    if not resolved_candidate.is_relative_to(root):
        raise OrthodonticPlaqueAuditError(
            f"Dataset root required path escapes root: {display_name}"
        )
    if candidate.is_symlink():
        raise OrthodonticPlaqueAuditError(
            f"Dataset root required path is a symlink: {display_name}"
        )
    if expected == "file" and not candidate.is_file():
        raise OrthodonticPlaqueAuditError(
            f"Dataset root is missing required file: {display_name}"
        )
    if expected == "directory" and not candidate.is_dir():
        raise OrthodonticPlaqueAuditError(
            f"Dataset root is missing required directory: {display_name}"
        )


def _load_tooth_position_ids(path: Path) -> frozenset[int]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OrthodonticPlaqueAuditError("Teeth position mapping is unreadable") from exc

    categories = payload.get("categories")
    if not isinstance(categories, list):
        raise OrthodonticPlaqueAuditError("Teeth position mapping is missing categories")
    ids: set[int] = set()
    for index, category in enumerate(categories):
        if not isinstance(category, dict) or not isinstance(category.get("id"), int):
            raise OrthodonticPlaqueAuditError(
                f"Teeth position category {index} is missing an integer id"
            )
        ids.add(category["id"])
    if not ids:
        raise OrthodonticPlaqueAuditError("Teeth position mapping has no categories")
    return frozenset(ids)


def _load_split_rows(path: Path) -> tuple[tuple[int, dict[str, str | None]], ...]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            _validate_split_header(reader.fieldnames)
            rows = tuple((line_number, row) for line_number, row in enumerate(reader, start=2))
    except UnicodeDecodeError as exc:
        raise OrthodonticPlaqueAuditError("data_splits.csv must be UTF-8 encoded") from exc
    except csv.Error as exc:
        raise OrthodonticPlaqueAuditError(f"data_splits.csv is malformed: {exc}") from exc

    if not rows:
        raise OrthodonticPlaqueAuditError("data_splits.csv must contain at least one row")
    return rows


def _validate_split_header(fieldnames: list[str] | None) -> None:
    if fieldnames is None:
        raise OrthodonticPlaqueAuditError("data_splits.csv is missing a header row")
    missing = [column for column in _REQUIRED_SPLIT_COLUMNS if column not in fieldnames]
    if missing:
        raise OrthodonticPlaqueAuditError(
            f"data_splits.csv is missing required columns: {', '.join(missing)}"
        )


def _required_row_value(
    row: dict[str, str | None],
    column: str,
    line_number: int,
    issues: list[OrthodonticPlaqueAuditIssue],
) -> str | None:
    value = row.get(column)
    normalized = value.strip() if value is not None else ""
    if not normalized:
        _add_issue(
            issues,
            code="missing_required_value",
            message=f"{column} must not be empty",
            csv_line=line_number,
        )
        return None
    if "\x00" in normalized:
        _add_issue(
            issues,
            code="null_byte",
            message=f"{column} contains a null byte",
            csv_line=line_number,
        )
        return None
    return normalized


def _validate_image_filename(
    image_filename: str,
    line_number: int,
    issues: list[OrthodonticPlaqueAuditIssue],
) -> PurePosixPath | None:
    if "\\" in image_filename or ":" in image_filename:
        _add_issue(
            issues,
            code="unsafe_image_filename",
            message="Image filename contains an unsafe path",
            csv_line=line_number,
        )
        return None
    if any(part in {"", ".", ".."} for part in image_filename.split("/")):
        _add_issue(
            issues,
            code="unsafe_image_filename",
            message="Image filename contains an unsafe path segment",
            csv_line=line_number,
        )
        return None
    path = PurePosixPath(image_filename)
    if path.is_absolute() or len(path.parts) != 1:
        _add_issue(
            issues,
            code="unsafe_image_filename",
            message="Image filename must be a plain filename",
            csv_line=line_number,
        )
        return None
    if path.suffix.lower() not in _SUPPORTED_IMAGE_EXTENSIONS:
        _add_issue(
            issues,
            code="unsupported_image_extension",
            message=f"Unsupported image extension: {path.suffix or '<none>'}",
            csv_line=line_number,
        )
        return None
    return path


def _resolve_inside(root: Path, relative_path: PurePosixPath, *, line_number: int) -> Path:
    candidate = root.joinpath(*relative_path.parts).resolve(strict=False)
    if not candidate.is_relative_to(root):
        raise OrthodonticPlaqueAuditError(
            f"Line {line_number}: resolved path escapes dataset root"
        )
    return candidate


def _audit_label_file(
    label_path: Path,
    allowed_tooth_ids: frozenset[int],
    *,
    line_number: int,
    issues: list[OrthodonticPlaqueAuditIssue],
) -> tuple[int, set[int]]:
    observed_tooth_ids: set[int] = set()
    try:
        lines = label_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        _add_issue(
            issues,
            code="unreadable_label",
            message=f"Label file is unreadable: {label_path}",
            csv_line=line_number,
        )
        return 0, observed_tooth_ids
    if not lines:
        _add_issue(
            issues,
            code="empty_label",
            message="Label file is empty",
            csv_line=line_number,
        )
        return 0, observed_tooth_ids

    valid_annotation_count = 0
    for label_line_number, raw_line in enumerate(lines, start=1):
        parts = raw_line.split()
        if len(parts) != _LABEL_FIELD_COUNT:
            _add_issue(
                issues,
                code="invalid_label_field_count",
                message=f"Label row must have {_LABEL_FIELD_COUNT} fields",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue
        try:
            class_id = int(parts[0])
            x_center, y_center, width, height = (float(value) for value in parts[1:5])
            tooth_id = int(parts[5])
        except ValueError as exc:
            _add_issue(
                issues,
                code="invalid_label_number",
                message="Label row contains invalid numbers",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue

        if class_id not in {0, 1}:
            _add_issue(
                issues,
                code="unsupported_class_id",
                message="Label row class id must be 0 or 1",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue
        if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
            _add_issue(
                issues,
                code="label_coordinate_out_of_bounds",
                message="Label row coordinates are out of bounds",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue
        if width <= 0.0 or height <= 0.0:
            _add_issue(
                issues,
                code="non_positive_box_size",
                message="Label row has a non-positive box size",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue
        if tooth_id not in allowed_tooth_ids:
            _add_issue(
                issues,
                code="unknown_tooth_id",
                message=f"Label row has unknown tooth id {tooth_id}",
                csv_line=line_number,
                label_line=label_line_number,
            )
            continue
        observed_tooth_ids.add(tooth_id)
        valid_annotation_count += 1
    return valid_annotation_count, observed_tooth_ids


def _add_issue(
    issues: list[OrthodonticPlaqueAuditIssue],
    *,
    code: str,
    message: str,
    csv_line: int,
    label_line: int | None = None,
) -> None:
    issues.append(
        OrthodonticPlaqueAuditIssue(
            code=code,
            severity="error",
            message=message,
            csv_line=csv_line,
            label_line=label_line,
        )
    )
