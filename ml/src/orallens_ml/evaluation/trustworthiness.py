"""Traceable deployment-parity and score-reliability evaluation for detection."""

from __future__ import annotations

import hashlib
import json
import platform
import tomllib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision

from orallens_ml.data.orthodontic_plaque_dataset import (
    OrthodonticPlaqueDatasetError,
    OrthodonticPlaquePart2Dataset,
    OrthodonticPlaqueTarget,
)
from orallens_ml.evaluation.detection import (
    DetectionEvaluationConfig,
    DetectionEvaluationError,
    DetectionImageEvaluation,
    DetectionPredictionMatch,
    evaluate_detection_image,
    load_detection_evaluation_config,
)
from orallens_ml.inference.detection import (
    DetectionInferenceConfig,
    DetectionInferenceError,
    load_detection_inference_config,
)
from orallens_ml.modeling.detection import (
    DetectionModelConfig,
    DetectionModelError,
    create_fasterrcnn_resnet50_fpn,
    load_detection_checkpoint,
    validate_checkpoint_compatibility,
)
from orallens_ml.training.detection import (
    DetectionTrainingError,
    collate_detection_batch,
    make_detection_target,
)

_REPORT_SCHEMA_VERSION = 1
_REPORT_FILENAME = "trustworthiness_report.json"
_MATCHING_POLICY = "score_ordered_same_class_greedy_one_to_one"


class DetectionTrustworthinessError(ValueError):
    """Raised when a detection trustworthiness report cannot run safely."""


@dataclass(frozen=True, slots=True)
class DetectionTrustworthinessConfig:
    """Validated references and report settings for one trustworthiness run."""

    config_path: Path
    evaluation_config_path: Path
    inference_config_path: Path
    output_dir: Path
    reliability_bins: int
    top_failure_cases: int


@dataclass(frozen=True, slots=True)
class DetectionTrustworthinessResult:
    """Generated report path and concise summary."""

    report_path: Path
    split: str
    deployed_metrics: dict[str, float | int | None]
    uncapped_metrics: dict[str, float | int | None]
    cap_affected_image_count: int
    score_to_match_ece: float | None


@dataclass(slots=True)
class _Aggregate:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    target_count: int = 0
    prediction_count: int = 0
    matched_iou_sum: float = 0.0

    def update(self, image: DetectionImageEvaluation) -> None:
        self.true_positives += image.true_positives
        self.false_positives += image.false_positives
        self.false_negatives += image.false_negatives
        self.target_count += image.target_count
        self.prediction_count += image.prediction_count
        self.matched_iou_sum += sum(image.matched_ious)

    def for_json(self) -> dict[str, float | int | None]:
        precision = _safe_divide(
            self.true_positives,
            self.true_positives + self.false_positives,
        )
        recall = _safe_divide(
            self.true_positives,
            self.true_positives + self.false_negatives,
        )
        return {
            "f1": _safe_divide(2.0 * precision * recall, precision + recall),
            "false_negatives": self.false_negatives,
            "false_positives": self.false_positives,
            "mean_matched_iou": (
                self.matched_iou_sum / self.true_positives
                if self.true_positives
                else None
            ),
            "precision": precision,
            "prediction_count": self.prediction_count,
            "recall": recall,
            "target_count": self.target_count,
            "true_positives": self.true_positives,
        }


def load_detection_trustworthiness_config(
    config_path: Path,
) -> DetectionTrustworthinessConfig:
    """Load a trustworthiness TOML that references frozen evaluation policies."""

    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise DetectionTrustworthinessError(
            f"Trustworthiness config is not a regular file: {path}"
        )
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise DetectionTrustworthinessError(
            f"Trustworthiness config is malformed TOML: {exc}"
        ) from exc

    evaluation = _table(payload, "evaluation")
    inference = _table(payload, "inference")
    reporting = _table(payload, "reporting")
    output = _table(payload, "output")
    return DetectionTrustworthinessConfig(
        config_path=path,
        evaluation_config_path=_path_value(evaluation, "config_path"),
        inference_config_path=_path_value(inference, "config_path"),
        output_dir=_path_value(output, "output_dir"),
        reliability_bins=_bounded_int(
            reporting,
            "reliability_bins",
            minimum=2,
            maximum=100,
        ),
        top_failure_cases=_bounded_int(
            reporting,
            "top_failure_cases",
            minimum=1,
            maximum=1000,
        ),
    )


def run_detection_trustworthiness(
    config: DetectionTrustworthinessConfig,
    *,
    model_factory: Callable[[DetectionEvaluationConfig], Module] | None = None,
) -> DetectionTrustworthinessResult:
    """Evaluate the frozen detector policy and atomically write traceable evidence."""

    try:
        evaluation = load_detection_evaluation_config(config.evaluation_config_path)
        inference = load_detection_inference_config(config.inference_config_path)
    except (DetectionEvaluationError, DetectionInferenceError) as exc:
        raise DetectionTrustworthinessError(str(exc)) from exc
    _validate_config_compatibility(config, evaluation=evaluation, inference=inference)
    _prepare_output_dir(config.output_dir)

    device = _select_device(evaluation.device)
    loader = _build_loader(evaluation)
    model = _build_model(evaluation, model_factory=model_factory).to(device)
    _load_checkpoint(model, evaluation=evaluation, device=device)
    model.eval()

    deployed_total = _Aggregate()
    uncapped_total = _Aggregate()
    patients: dict[str, _Aggregate] = {}
    per_image: list[dict[str, object]] = []
    false_positive_cases: list[dict[str, object]] = []
    reliability_observations: list[DetectionPredictionMatch] = []
    cap_affected_image_count = 0
    truncated_prediction_count = 0
    processed_batches = 0
    sample_index = 0

    with torch.no_grad():
        for batch_index, (images, source_targets) in enumerate(_loader_batches(loader)):
            if evaluation.max_batches is not None and batch_index >= evaluation.max_batches:
                break
            processed_batches += 1
            image_batch = [image.to(device) for image in images]
            try:
                target_batch = [
                    make_detection_target(
                        image,
                        source_target,
                        image_id=sample_index + target_index,
                        num_classes=evaluation.num_classes,
                    )
                    for target_index, (image, source_target) in enumerate(
                        zip(images, source_targets)
                    )
                ]
            except DetectionTrainingError as exc:
                raise DetectionTrustworthinessError(str(exc)) from exc

            predictions = model(image_batch)
            if not isinstance(predictions, list) or len(predictions) != len(images):
                raise DetectionTrustworthinessError(
                    "Model did not return one prediction dictionary per image"
                )

            for image, source_target, target, prediction in zip(
                images,
                source_targets,
                target_batch,
                predictions,
            ):
                try:
                    uncapped = evaluate_detection_image(
                        prediction,
                        target,
                        iou_threshold=evaluation.iou_thresholds[0],
                        score_threshold=inference.score_threshold,
                        num_classes=evaluation.num_classes,
                    )
                    deployed = evaluate_detection_image(
                        prediction,
                        target,
                        iou_threshold=evaluation.iou_thresholds[0],
                        score_threshold=inference.score_threshold,
                        num_classes=evaluation.num_classes,
                        max_detections=inference.max_detections,
                    )
                except DetectionEvaluationError as exc:
                    raise DetectionTrustworthinessError(str(exc)) from exc

                deployed_total.update(deployed)
                uncapped_total.update(uncapped)
                patient = patients.setdefault(source_target.patient_id, _Aggregate())
                patient.update(deployed)
                reliability_observations.extend(deployed.prediction_matches)
                if deployed.truncated_prediction_count:
                    cap_affected_image_count += 1
                    truncated_prediction_count += deployed.truncated_prediction_count

                image_record = _image_record(
                    image=image,
                    source_target=source_target,
                    deployed=deployed,
                    uncapped=uncapped,
                )
                per_image.append(image_record)
                false_positive_cases.extend(
                    _false_positive_cases(
                        source_target=source_target,
                        image_height=int(image.shape[1]),
                        image_width=int(image.shape[2]),
                        deployed=deployed,
                    )
                )
                sample_index += 1

    if processed_batches == 0 or not per_image:
        raise DetectionTrustworthinessError("No evaluation samples were processed")

    reliability = _score_to_match_reliability(
        reliability_observations,
        bin_count=config.reliability_bins,
    )
    deployed_metrics = deployed_total.for_json()
    uncapped_metrics = uncapped_total.for_json()
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "claim_boundary": (
            "Metrics describe plaque-candidate box matching on the configured dataset. "
            "They are not disease probabilities or clinical-performance claims."
        ),
        "scope": {
            "split": evaluation.split,
            "iou_threshold": evaluation.iou_thresholds[0],
            "score_threshold": inference.score_threshold,
            "max_detections": inference.max_detections,
            "matching_policy": _MATCHING_POLICY,
            "prediction_box_coordinate_space": "pixel_xyxy",
            "false_negative_target_coordinate_space": "normalized_xyxy",
            "calibration_interpretation": (
                "Score-to-match reliability is conditional on predictions displayed "
                "at the configured threshold and cap."
            ),
        },
        "provenance": _provenance(
            config=config,
            evaluation=evaluation,
            inference=inference,
            device=device,
        ),
        "dataset": {
            "image_count": len(per_image),
            "patient_count": len(patients),
            "target_count": deployed_total.target_count,
        },
        "deployed_metrics": deployed_metrics,
        "uncapped_metrics": uncapped_metrics,
        "cap_impact": {
            "affected_image_count": cap_affected_image_count,
            "truncated_prediction_count": truncated_prediction_count,
            "metrics_changed": deployed_metrics != uncapped_metrics,
        },
        "score_to_match_reliability": reliability,
        "patients": [
            {"patient_id": patient_id, **aggregate.for_json()}
            for patient_id, aggregate in sorted(patients.items())
        ],
        "failure_summary": {
            "highest_scoring_false_positives": sorted(
                false_positive_cases,
                key=lambda case: (-float(case["score"]), str(case["sample_id"])),
            )[: config.top_failure_cases],
            "images_with_most_false_negatives": sorted(
                (
                    {
                        "sample_id": record["sample_id"],
                        "patient_id": record["patient_id"],
                        "false_negatives": record["deployed"]["false_negatives"],
                        "target_count": record["deployed"]["target_count"],
                    }
                    for record in per_image
                    if int(record["deployed"]["false_negatives"]) > 0
                ),
                key=lambda case: (
                    -int(case["false_negatives"]),
                    str(case["sample_id"]),
                ),
            )[: config.top_failure_cases],
        },
        "images": per_image,
    }
    report_path = _safe_output_file(config.output_dir, _REPORT_FILENAME)
    _write_json_atomic(report_path, report)
    return DetectionTrustworthinessResult(
        report_path=report_path,
        split=evaluation.split,
        deployed_metrics=deployed_metrics,
        uncapped_metrics=uncapped_metrics,
        cap_affected_image_count=cap_affected_image_count,
        score_to_match_ece=reliability["expected_calibration_error"],
    )


def _image_record(
    *,
    image: torch.Tensor,
    source_target: OrthodonticPlaqueTarget,
    deployed: DetectionImageEvaluation,
    uncapped: DetectionImageEvaluation,
) -> dict[str, object]:
    image_height = int(image.shape[1])
    image_width = int(image.shape[2])
    return {
        "sample_id": source_target.sample_id,
        "patient_id": source_target.patient_id,
        "source_csv_line": source_target.source_csv_line,
        "image_height": image_height,
        "image_width": image_width,
        "deployed": _image_metrics(deployed),
        "uncapped": _image_metrics(uncapped),
        "predictions": [
            _prediction_record(
                prediction=prediction,
                image_height=image_height,
                image_width=image_width,
                iou_threshold=deployed.iou_threshold,
            )
            for prediction in deployed.prediction_matches
        ],
        "false_negative_targets": [
            {
                "target_index": target_index,
                "normalized_box_xyxy": [
                    float(value)
                    for value in source_target.boxes[target_index].tolist()
                ],
                "source_label_line": int(source_target.source_label_lines[target_index]),
                "tooth_id": int(source_target.tooth_ids[target_index]),
            }
            for target_index in deployed.unmatched_target_indexes
        ],
    }


def _image_metrics(image: DetectionImageEvaluation) -> dict[str, float | int | None]:
    precision = _safe_divide(
        image.true_positives,
        image.true_positives + image.false_positives,
    )
    recall = _safe_divide(
        image.true_positives,
        image.true_positives + image.false_negatives,
    )
    return {
        "eligible_prediction_count": image.eligible_prediction_count,
        "f1": _safe_divide(2.0 * precision * recall, precision + recall),
        "false_negatives": image.false_negatives,
        "false_positives": image.false_positives,
        "mean_matched_iou": (
            sum(image.matched_ious) / len(image.matched_ious)
            if image.matched_ious
            else None
        ),
        "precision": precision,
        "prediction_count": image.prediction_count,
        "recall": recall,
        "target_count": image.target_count,
        "true_positives": image.true_positives,
        "truncated_prediction_count": image.truncated_prediction_count,
    }


def _prediction_record(
    *,
    prediction: DetectionPredictionMatch,
    image_height: int,
    image_width: int,
    iou_threshold: float,
) -> dict[str, object]:
    x1, y1, x2, y2 = prediction.box_xyxy
    normalized_center = [
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
    ]
    normalized_area = ((x2 - x1) * (y2 - y1)) / (image_width * image_height)
    return {
        "prediction_index": prediction.prediction_index,
        "box_xyxy": list(prediction.box_xyxy),
        "label": prediction.label,
        "score": prediction.score,
        "is_true_positive": prediction.is_true_positive,
        "best_target_index": prediction.best_target_index,
        "best_iou": prediction.best_iou,
        "matched_target_index": prediction.matched_target_index,
        "matched_iou": prediction.matched_iou,
        "false_positive_kind": _false_positive_kind(
            prediction,
            iou_threshold=iou_threshold,
        ),
        "normalized_center_xy": normalized_center,
        "normalized_area": normalized_area,
    }


def _false_positive_kind(
    prediction: DetectionPredictionMatch,
    *,
    iou_threshold: float,
) -> str | None:
    if prediction.is_true_positive:
        return None
    if prediction.best_target_index is None or prediction.best_iou is None:
        return "no_same_class_target"
    if prediction.best_iou >= iou_threshold:
        return "duplicate"
    return "localization_or_background"


def _false_positive_cases(
    *,
    source_target: OrthodonticPlaqueTarget,
    image_height: int,
    image_width: int,
    deployed: DetectionImageEvaluation,
) -> list[dict[str, object]]:
    return [
        {
            "sample_id": source_target.sample_id,
            "patient_id": source_target.patient_id,
            **_prediction_record(
                prediction=prediction,
                image_height=image_height,
                image_width=image_width,
                iou_threshold=deployed.iou_threshold,
            ),
        }
        for prediction in deployed.prediction_matches
        if not prediction.is_true_positive
    ]


def _score_to_match_reliability(
    observations: list[DetectionPredictionMatch],
    *,
    bin_count: int,
) -> dict[str, object]:
    bins: list[list[DetectionPredictionMatch]] = [[] for _ in range(bin_count)]
    for observation in observations:
        bin_index = min(int(observation.score * bin_count), bin_count - 1)
        bins[bin_index].append(observation)

    bin_records: list[dict[str, float | int | None]] = []
    weighted_gap = 0.0
    maximum_gap: float | None = None
    squared_error_sum = 0.0
    observation_count = len(observations)
    for index, values in enumerate(bins):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        if values:
            mean_score = sum(value.score for value in values) / len(values)
            match_rate = sum(value.is_true_positive for value in values) / len(values)
            absolute_gap = abs(mean_score - match_rate)
            weighted_gap += (len(values) / observation_count) * absolute_gap
            maximum_gap = (
                absolute_gap
                if maximum_gap is None
                else max(maximum_gap, absolute_gap)
            )
            squared_error_sum += sum(
                (value.score - float(value.is_true_positive)) ** 2
                for value in values
            )
        else:
            mean_score = None
            match_rate = None
            absolute_gap = None
        bin_records.append(
            {
                "bin_index": index,
                "lower_bound_inclusive": lower,
                "upper_bound_inclusive": upper if index == bin_count - 1 else None,
                "upper_bound_exclusive": upper if index < bin_count - 1 else None,
                "prediction_count": len(values),
                "mean_score": mean_score,
                "empirical_match_rate": match_rate,
                "absolute_gap": absolute_gap,
            }
        )

    return {
        "definition": (
            "For displayed predictions, compares detector score with whether the box "
            "matched one annotation under the configured matching policy and IoU."
        ),
        "observation_count": observation_count,
        "bin_count": bin_count,
        "expected_calibration_error": weighted_gap if observation_count else None,
        "maximum_calibration_error": maximum_gap,
        "brier_score": (
            squared_error_sum / observation_count if observation_count else None
        ),
        "bins": bin_records,
    }


def _provenance(
    *,
    config: DetectionTrustworthinessConfig,
    evaluation: DetectionEvaluationConfig,
    inference: DetectionInferenceConfig,
    device: torch.device,
) -> dict[str, object]:
    return {
        "trustworthiness_config": _file_identity(config.config_path),
        "evaluation_config": _file_identity(config.evaluation_config_path),
        "inference_config": _file_identity(config.inference_config_path),
        "manifest": _file_identity(evaluation.manifest_path),
        "checkpoint": _file_identity(evaluation.checkpoint_path),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "requested_device": evaluation.device,
            "resolved_device": device.type,
            "cuda_device_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "model": {
            "num_classes": evaluation.num_classes,
            "image_min_size": evaluation.image_min_size,
            "image_max_size": evaluation.image_max_size,
            "trainable_backbone_layers": evaluation.trainable_backbone_layers,
            "pretrained_weights": evaluation.pretrained_weights,
        },
    }


def _file_identity(path: Path) -> dict[str, object]:
    file_path = Path(path)
    if file_path.is_symlink() or not file_path.is_file():
        raise DetectionTrustworthinessError(f"Evidence input is not a regular file: {file_path}")
    digest = hashlib.sha256()
    size_bytes = 0
    with file_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size_bytes += len(chunk)
            digest.update(chunk)
    return {
        "path": str(file_path),
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
    }


def _validate_config_compatibility(
    config: DetectionTrustworthinessConfig,
    *,
    evaluation: DetectionEvaluationConfig,
    inference: DetectionInferenceConfig,
) -> None:
    if evaluation.split not in {"validation", "test"}:
        raise DetectionTrustworthinessError(
            "Trustworthiness reporting supports validation or test splits only"
        )
    if evaluation.max_batches is not None:
        raise DetectionTrustworthinessError(
            "Trustworthiness reporting requires the complete configured split"
        )
    if evaluation.num_workers != 0:
        raise DetectionTrustworthinessError(
            "Trustworthiness reporting requires num_workers=0 for deterministic errors"
        )
    if len(evaluation.iou_thresholds) != 1:
        raise DetectionTrustworthinessError(
            "Trustworthiness reporting requires exactly one IoU threshold"
        )
    fields = (
        "checkpoint_path",
        "num_classes",
        "image_min_size",
        "image_max_size",
        "trainable_backbone_layers",
        "pretrained_weights",
    )
    mismatches = [
        field
        for field in fields
        if _comparable_config_value(getattr(evaluation, field))
        != _comparable_config_value(getattr(inference, field))
    ]
    if mismatches:
        raise DetectionTrustworthinessError(
            "Evaluation and inference configs are incompatible: "
            + ", ".join(mismatches)
        )
    if config.output_dir.is_symlink():
        raise DetectionTrustworthinessError(
            f"Output directory is a symlink: {config.output_dir}"
        )
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise DetectionTrustworthinessError(
            f"Output path is not a directory: {config.output_dir}"
        )
    if config.output_dir.resolve(strict=False) == evaluation.output_dir.resolve(
        strict=False
    ):
        raise DetectionTrustworthinessError(
            "Trustworthiness output must not overwrite historical evaluation output"
        )


def _comparable_config_value(value: object) -> object:
    if isinstance(value, Path):
        return value.resolve(strict=False)
    return value


def _build_loader(config: DetectionEvaluationConfig) -> DataLoader:
    try:
        dataset = OrthodonticPlaquePart2Dataset(
            dataset_root=config.dataset_root,
            manifest_path=config.manifest_path,
            split=config.split,
        )
    except OrthodonticPlaqueDatasetError as exc:
        raise DetectionTrustworthinessError(str(exc)) from exc
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_detection_batch,
    )


def _loader_batches(
    loader: DataLoader,
) -> Iterator[tuple[list[torch.Tensor], list[OrthodonticPlaqueTarget]]]:
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            return
        except OrthodonticPlaqueDatasetError as exc:
            raise DetectionTrustworthinessError(str(exc)) from exc


def _build_model(
    config: DetectionEvaluationConfig,
    *,
    model_factory: Callable[[DetectionEvaluationConfig], Module] | None,
) -> Module:
    if model_factory is not None:
        return model_factory(config)
    try:
        return create_fasterrcnn_resnet50_fpn(
            DetectionModelConfig.from_runtime_config(config)
        )
    except DetectionModelError as exc:
        raise DetectionTrustworthinessError(str(exc)) from exc


def _load_checkpoint(
    model: Module,
    *,
    evaluation: DetectionEvaluationConfig,
    device: torch.device,
) -> None:
    try:
        checkpoint = load_detection_checkpoint(evaluation.checkpoint_path, device=device)
        validate_checkpoint_compatibility(
            checkpoint_config=checkpoint.config,
            runtime_config=DetectionModelConfig.from_runtime_config(evaluation),
        )
        model.load_state_dict(checkpoint.model_state_dict)
    except (DetectionModelError, RuntimeError) as exc:
        raise DetectionTrustworthinessError(str(exc)) from exc


def _select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DetectionTrustworthinessError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_output_file(output_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise DetectionTrustworthinessError(f"Unsafe output filename: {filename}")
    root = output_dir.resolve(strict=True)
    path = root / filename
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise DetectionTrustworthinessError(f"Output file escapes output directory: {filename}")
    if path.is_symlink():
        raise DetectionTrustworthinessError(f"Output file is a symlink: {path}")
    return path


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _table(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DetectionTrustworthinessError(
            f"Trustworthiness config is missing [{name}] table"
        )
    return value


def _path_value(payload: dict[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DetectionTrustworthinessError(f"{key} must be a non-empty string path")
    if "\x00" in value:
        raise DetectionTrustworthinessError(f"{key} contains a null byte")
    return Path(value)


def _bounded_int(
    payload: dict[str, object],
    key: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise DetectionTrustworthinessError(
            f"{key} must be between {minimum} and {maximum}"
        )
    return value


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0
