"""Transparent IoU-based evaluation for detection predictions."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset
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

DeviceName = Literal["auto", "cpu", "cuda"]
SplitName = Literal["train", "validation", "test"]


class DetectionEvaluationError(ValueError):
    """Raised when detection evaluation cannot run safely."""


@dataclass(frozen=True, slots=True)
class DetectionEvaluationConfig:
    """Validated settings for one detection evaluation run."""

    dataset_root: Path
    manifest_path: Path
    checkpoint_path: Path
    output_dir: Path
    split: SplitName
    batch_size: int
    num_workers: int
    num_classes: int
    image_min_size: int
    image_max_size: int
    trainable_backbone_layers: int
    pretrained_weights: Literal["none", "default"]
    device: DeviceName
    iou_thresholds: tuple[float, ...]
    score_thresholds: tuple[float, ...]
    variant: str | None = None
    excluded_sample_id_suffixes: tuple[str, ...] = ()
    average_precision_iou_thresholds: tuple[float, ...] = ()
    analysis_score_threshold: float | None = None
    analysis_top_cases: int = 10
    max_batches: int | None = None


@dataclass(frozen=True, slots=True)
class DetectionMetric:
    """Aggregated detection metric for one IoU and score threshold pair."""

    iou_threshold: float
    score_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    target_count: int
    prediction_count: int
    precision: float
    recall: float
    f1: float
    mean_matched_iou: float | None


@dataclass(frozen=True, slots=True)
class DetectionPredictionMatch:
    """Matching evidence for one score-filtered detection prediction."""

    prediction_index: int
    box_xyxy: tuple[float, float, float, float]
    label: int
    score: float
    is_true_positive: bool
    best_target_index: int | None
    best_iou: float | None
    matched_target_index: int | None
    matched_iou: float | None


@dataclass(frozen=True, slots=True)
class DetectionImageEvaluation:
    """Detailed matching result for one image and operating point."""

    iou_threshold: float
    score_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    target_count: int
    prediction_count: int
    eligible_prediction_count: int
    truncated_prediction_count: int
    matched_ious: tuple[float, ...]
    prediction_matches: tuple[DetectionPredictionMatch, ...]
    unmatched_target_indexes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DetectionEvaluationResult:
    """Files and metrics produced by an evaluation run."""

    metrics_path: Path
    metrics: tuple[DetectionMetric, ...]
    dataset_summary: DetectionDatasetSummary
    average_precision: DetectionAveragePrecision | None
    error_analysis: DetectionErrorAnalysis | None


@dataclass(frozen=True, slots=True)
class DetectionDatasetSummary:
    """Population identity for one completed detection evaluation."""

    image_count: int
    patient_count: int
    target_count: int
    variant: str | None
    excluded_sample_id_suffixes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DetectionAveragePrecision:
    """Single-class 101-point interpolated average precision."""

    ap50: float | None
    map50_95: float | None
    by_iou: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class DetectionErrorAnalysis:
    """Aggregate and representative failures at one frozen operating point."""

    iou_threshold: float
    score_threshold: float
    false_positives: int
    false_negatives: int
    duplicate_detections: int
    localization_failures: int
    background_false_positives: int
    low_confidence_matches: int
    unexplained_false_negatives: int
    top_failure_cases: tuple[dict[str, Any], ...]


@dataclass(slots=True)
class _MetricAccumulator:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    target_count: int = 0
    prediction_count: int = 0
    matched_ious: list[float] | None = None

    def __post_init__(self) -> None:
        if self.matched_ious is None:
            self.matched_ious = []


def load_detection_evaluation_config(config_path: Path) -> DetectionEvaluationConfig:
    """Load and validate a detection evaluation TOML file."""

    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise DetectionEvaluationError(f"Evaluation config is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise DetectionEvaluationError(
            f"Evaluation config is malformed TOML: {exc}"
        ) from exc

    data = _table(payload, "data")
    model = _table(payload, "model")
    evaluation = _table(payload, "evaluation")
    output = _table(payload, "output")

    return DetectionEvaluationConfig(
        dataset_root=_path_value(data, "dataset_root"),
        manifest_path=_path_value(data, "manifest_path"),
        checkpoint_path=_path_value(model, "checkpoint_path"),
        output_dir=_path_value(output, "output_dir"),
        split=_split_name(evaluation, "split"),
        batch_size=_positive_int(evaluation, "batch_size"),
        num_workers=_non_negative_int(evaluation, "num_workers"),
        num_classes=_positive_int(model, "num_classes"),
        image_min_size=_positive_int(model, "image_min_size"),
        image_max_size=_positive_int(model, "image_max_size"),
        trainable_backbone_layers=_bounded_int(
            model,
            "trainable_backbone_layers",
            minimum=0,
            maximum=5,
        ),
        pretrained_weights=_weights_name(model, "pretrained_weights"),
        device=_device_name(evaluation, "device"),
        iou_thresholds=_thresholds(evaluation, "iou_thresholds"),
        score_thresholds=_thresholds(evaluation, "score_thresholds"),
        variant=_optional_text(evaluation, "variant"),
        excluded_sample_id_suffixes=_optional_text_list(
            evaluation,
            "excluded_sample_id_suffixes",
        ),
        average_precision_iou_thresholds=_optional_thresholds(
            evaluation,
            "average_precision_iou_thresholds",
        ),
        analysis_score_threshold=_optional_threshold(
            evaluation,
            "analysis_score_threshold",
        ),
        analysis_top_cases=_optional_positive_int_with_default(
            evaluation,
            "analysis_top_cases",
            default=10,
        ),
        max_batches=_optional_positive_int(evaluation, "max_batches"),
    )


def evaluate_detection_predictions(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    iou_thresholds: tuple[float, ...],
    score_thresholds: tuple[float, ...],
    num_classes: int,
) -> tuple[DetectionMetric, ...]:
    """Evaluate prediction dictionaries against TorchVision detection targets."""

    if len(predictions) != len(targets):
        raise DetectionEvaluationError("Prediction and target batch sizes differ")
    accumulators = {
        (iou_threshold, score_threshold): _MetricAccumulator()
        for iou_threshold in iou_thresholds
        for score_threshold in score_thresholds
    }
    for prediction, target in zip(predictions, targets):
        validated_prediction = _validate_prediction(prediction, num_classes=num_classes)
        validated_target = _validate_target(target, num_classes=num_classes)
        for iou_threshold in iou_thresholds:
            for score_threshold in score_thresholds:
                stats = _match_single_image(
                    validated_prediction,
                    validated_target,
                    iou_threshold=iou_threshold,
                    score_threshold=score_threshold,
                )
                accumulator = accumulators[(iou_threshold, score_threshold)]
                accumulator.true_positives += stats.true_positives
                accumulator.false_positives += stats.false_positives
                accumulator.false_negatives += stats.false_negatives
                accumulator.target_count += stats.target_count
                accumulator.prediction_count += stats.prediction_count
                assert accumulator.matched_ious is not None
                accumulator.matched_ious.extend(stats.matched_ious or [])

    return tuple(
        _metric_from_accumulator(
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            accumulator=accumulators[(iou_threshold, score_threshold)],
        )
        for iou_threshold in iou_thresholds
        for score_threshold in score_thresholds
    )


def evaluate_detection_image(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    *,
    iou_threshold: float,
    score_threshold: float,
    num_classes: int,
    max_detections: int | None = None,
) -> DetectionImageEvaluation:
    """Return detailed score-ordered matching evidence for one image."""

    if not 0.0 < iou_threshold < 1.0:
        raise DetectionEvaluationError("iou_threshold must be between 0 and 1")
    if not 0.0 <= score_threshold <= 1.0:
        raise DetectionEvaluationError("score_threshold must be between 0 and 1")
    if max_detections is not None and max_detections <= 0:
        raise DetectionEvaluationError("max_detections must be positive when provided")
    validated_prediction = _validate_prediction(prediction, num_classes=num_classes)
    validated_target = _validate_target(target, num_classes=num_classes)
    return _match_single_image(
        validated_prediction,
        validated_target,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )


def evaluate_detection_average_precision(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    iou_thresholds: tuple[float, ...],
    num_classes: int,
) -> DetectionAveragePrecision:
    """Calculate single-class 101-point interpolated AP across images."""

    if len(predictions) != len(targets):
        raise DetectionEvaluationError("Prediction and target batch sizes differ")
    if not predictions:
        raise DetectionEvaluationError("Average precision requires at least one image")
    if num_classes != 2:
        raise DetectionEvaluationError(
            "Average precision currently supports the binary plaque detector"
        )
    if not iou_thresholds:
        raise DetectionEvaluationError("Average precision requires IoU thresholds")

    validated_predictions = [
        _validate_prediction(prediction, num_classes=num_classes)
        for prediction in predictions
    ]
    validated_targets = [
        _validate_target(target, num_classes=num_classes) for target in targets
    ]
    target_count = sum(int(target["boxes"].shape[0]) for target in validated_targets)
    if target_count == 0:
        raise DetectionEvaluationError(
            "Average precision requires at least one ground-truth object"
        )

    by_iou = tuple(
        (
            iou_threshold,
            _average_precision_at_iou(
                validated_predictions,
                validated_targets,
                iou_threshold=iou_threshold,
                target_count=target_count,
            ),
        )
        for iou_threshold in iou_thresholds
    )
    ap50_values = [value for threshold, value in by_iou if threshold == 0.5]
    complete_coco_range = all(
        any(abs(threshold - expected) < 1e-9 for threshold, _ in by_iou)
        for expected in (0.5 + 0.05 * index for index in range(10))
    )
    return DetectionAveragePrecision(
        ap50=ap50_values[0] if ap50_values else None,
        map50_95=(
            sum(value for _, value in by_iou) / len(by_iou)
            if complete_coco_range
            else None
        ),
        by_iou=by_iou,
    )


def analyze_detection_errors(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    source_targets: list[OrthodonticPlaqueTarget],
    *,
    iou_threshold: float,
    score_threshold: float,
    num_classes: int,
    top_cases: int,
) -> DetectionErrorAnalysis:
    """Quantify threshold, localization, duplicate, and background failures."""

    if not (len(predictions) == len(targets) == len(source_targets)):
        raise DetectionEvaluationError("Error-analysis batch sizes differ")
    if not 0.0 < iou_threshold < 1.0:
        raise DetectionEvaluationError("Analysis IoU threshold must be between 0 and 1")
    if not 0.0 < score_threshold < 1.0:
        raise DetectionEvaluationError(
            "Analysis score threshold must be between 0 and 1"
        )
    if top_cases <= 0:
        raise DetectionEvaluationError("Analysis top_cases must be positive")

    totals = {
        "false_positives": 0,
        "false_negatives": 0,
        "duplicate_detections": 0,
        "localization_failures": 0,
        "background_false_positives": 0,
        "low_confidence_matches": 0,
    }
    cases: list[dict[str, Any]] = []
    for prediction, target, source_target in zip(
        predictions,
        targets,
        source_targets,
    ):
        validated_prediction = _validate_prediction(
            prediction,
            num_classes=num_classes,
        )
        validated_target = _validate_target(target, num_classes=num_classes)
        image_result = _match_single_image(
            validated_prediction,
            validated_target,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        false_positive_records: list[dict[str, Any]] = []
        duplicate_count = 0
        localization_count = 0
        background_count = 0
        for match in image_result.prediction_matches:
            if match.is_true_positive:
                continue
            if match.best_iou is not None and match.best_iou >= iou_threshold:
                kind = "duplicate"
                duplicate_count += 1
            elif match.best_iou is not None and match.best_iou >= 0.1:
                kind = "localization"
                localization_count += 1
            else:
                kind = "background"
                background_count += 1
            false_positive_records.append(
                {
                    "best_iou": match.best_iou,
                    "box_xyxy": list(match.box_xyxy),
                    "kind": kind,
                    "score": match.score,
                }
            )

        low_confidence_records = _low_confidence_target_matches(
            validated_prediction,
            validated_target,
            unmatched_target_indexes=image_result.unmatched_target_indexes,
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
        )
        false_negatives = image_result.false_negatives
        totals["false_positives"] += image_result.false_positives
        totals["false_negatives"] += false_negatives
        totals["duplicate_detections"] += duplicate_count
        totals["localization_failures"] += localization_count
        totals["background_false_positives"] += background_count
        totals["low_confidence_matches"] += len(low_confidence_records)

        if image_result.false_positives or false_negatives:
            cases.append(
                {
                    "sample_id": source_target.sample_id,
                    "patient_id": source_target.patient_id,
                    "image_relative_path": source_target.image_relative_path.as_posix(),
                    "false_positives": image_result.false_positives,
                    "false_negatives": false_negatives,
                    "duplicate_detections": duplicate_count,
                    "localization_failures": localization_count,
                    "background_false_positives": background_count,
                    "low_confidence_matches": len(low_confidence_records),
                    "false_positive_predictions": false_positive_records[:25],
                    "low_confidence_predictions": low_confidence_records[:25],
                    "unmatched_target_boxes_xyxy": [
                        validated_target["boxes"][target_index].tolist()
                        for target_index in image_result.unmatched_target_indexes[:25]
                    ],
                }
            )

    cases.sort(
        key=lambda case: (
            -int(case["false_positives"]) - int(case["false_negatives"]),
            -int(case["localization_failures"]),
            str(case["sample_id"]),
        )
    )
    unexplained_false_negatives = max(
        0,
        totals["false_negatives"] - totals["low_confidence_matches"],
    )
    return DetectionErrorAnalysis(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        false_positives=totals["false_positives"],
        false_negatives=totals["false_negatives"],
        duplicate_detections=totals["duplicate_detections"],
        localization_failures=totals["localization_failures"],
        background_false_positives=totals["background_false_positives"],
        low_confidence_matches=totals["low_confidence_matches"],
        unexplained_false_negatives=unexplained_false_negatives,
        top_failure_cases=tuple(cases[:top_cases]),
    )


def run_detection_evaluation(
    config: DetectionEvaluationConfig,
    *,
    model_factory: Callable[[DetectionEvaluationConfig], Module] | None = None,
) -> DetectionEvaluationResult:
    """Run checkpoint evaluation and write metrics JSON."""

    _validate_evaluation_inputs(config)
    _prepare_output_dir(config.output_dir)
    device = _select_device(config.device)
    loader = _build_loader(config)
    model = _build_model(config, model_factory=model_factory).to(device)
    _load_checkpoint(model, config.checkpoint_path, config=config, device=device)
    model.eval()

    accumulators = {
        (iou_threshold, score_threshold): _MetricAccumulator()
        for iou_threshold in config.iou_thresholds
        for score_threshold in config.score_thresholds
    }
    all_predictions: list[dict[str, Tensor]] = []
    all_targets: list[dict[str, Tensor]] = []
    all_source_targets: list[OrthodonticPlaqueTarget] = []
    with torch.no_grad():
        for batch_index, (images, raw_targets) in enumerate(loader):
            if config.max_batches is not None and batch_index >= config.max_batches:
                break
            image_batch = [image.to(device) for image in images]
            try:
                target_batch = [
                    {
                        key: value.to(device)
                        for key, value in make_detection_target(
                            image,
                            target,
                            image_id=batch_index * config.batch_size + target_index,
                            num_classes=config.num_classes,
                        ).items()
                    }
                    for target_index, (image, target) in enumerate(
                        zip(images, raw_targets)
                    )
                ]
            except DetectionTrainingError as exc:
                raise DetectionEvaluationError(str(exc)) from exc
            predictions = model(image_batch)
            if not isinstance(predictions, list):
                raise DetectionEvaluationError("Model did not return a prediction list")
            validated_predictions = [
                _validate_prediction(prediction, num_classes=config.num_classes)
                for prediction in predictions
            ]
            validated_targets = [
                _validate_target(target, num_classes=config.num_classes)
                for target in target_batch
            ]
            all_predictions.extend(validated_predictions)
            all_targets.extend(validated_targets)
            all_source_targets.extend(raw_targets)
            batch_metrics = evaluate_detection_predictions(
                validated_predictions,
                validated_targets,
                iou_thresholds=config.iou_thresholds,
                score_thresholds=config.score_thresholds,
                num_classes=config.num_classes,
            )
            for metric in batch_metrics:
                accumulator = accumulators[
                    (metric.iou_threshold, metric.score_threshold)
                ]
                accumulator.true_positives += metric.true_positives
                accumulator.false_positives += metric.false_positives
                accumulator.false_negatives += metric.false_negatives
                accumulator.target_count += metric.target_count
                accumulator.prediction_count += metric.prediction_count
                if metric.mean_matched_iou is not None and metric.true_positives > 0:
                    assert accumulator.matched_ious is not None
                    accumulator.matched_ious.extend(
                        [metric.mean_matched_iou] * metric.true_positives
                    )

    metrics = tuple(
        _metric_from_accumulator(
            iou_threshold=iou_threshold,
            score_threshold=score_threshold,
            accumulator=accumulators[(iou_threshold, score_threshold)],
        )
        for iou_threshold in config.iou_thresholds
        for score_threshold in config.score_thresholds
    )
    if not metrics:
        raise DetectionEvaluationError("No evaluation metrics were produced")

    dataset_summary = DetectionDatasetSummary(
        image_count=len(all_source_targets),
        patient_count=len({target.patient_id for target in all_source_targets}),
        target_count=sum(int(target["boxes"].shape[0]) for target in all_targets),
        variant=config.variant,
        excluded_sample_id_suffixes=config.excluded_sample_id_suffixes,
    )
    average_precision = (
        evaluate_detection_average_precision(
            all_predictions,
            all_targets,
            iou_thresholds=config.average_precision_iou_thresholds,
            num_classes=config.num_classes,
        )
        if config.average_precision_iou_thresholds
        else None
    )
    error_analysis = (
        analyze_detection_errors(
            all_predictions,
            all_targets,
            all_source_targets,
            iou_threshold=0.5,
            score_threshold=config.analysis_score_threshold,
            num_classes=config.num_classes,
            top_cases=config.analysis_top_cases,
        )
        if config.analysis_score_threshold is not None
        else None
    )

    metrics_path = _safe_output_file(config.output_dir, "evaluation_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "average_precision": _average_precision_for_json(average_precision),
                "dataset": _dataset_summary_for_json(dataset_summary),
                "error_analysis": _error_analysis_for_json(error_analysis),
                "metrics": [_metric_for_json(metric) for metric in metrics],
                "split": config.split,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return DetectionEvaluationResult(
        metrics_path=metrics_path,
        metrics=metrics,
        dataset_summary=dataset_summary,
        average_precision=average_precision,
        error_analysis=error_analysis,
    )


def _build_model(
    config: DetectionEvaluationConfig,
    *,
    model_factory: Callable[[DetectionEvaluationConfig], Module] | None,
) -> Module:
    if model_factory is not None:
        return model_factory(config)
    return create_fasterrcnn_resnet50_fpn(
        DetectionModelConfig.from_runtime_config(config)
    )


def _load_checkpoint(
    model: Module,
    checkpoint_path: Path,
    *,
    config: DetectionEvaluationConfig,
    device: torch.device,
) -> None:
    try:
        checkpoint = load_detection_checkpoint(checkpoint_path, device=device)
        validate_checkpoint_compatibility(
            checkpoint_config=checkpoint.config,
            runtime_config=DetectionModelConfig.from_runtime_config(config),
        )
    except DetectionModelError as exc:
        raise DetectionEvaluationError(str(exc)) from exc
    model.load_state_dict(checkpoint.model_state_dict)


def _match_single_image(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    *,
    iou_threshold: float,
    score_threshold: float,
    max_detections: int | None = None,
) -> DetectionImageEvaluation:
    scores = prediction["scores"]
    keep = scores >= score_threshold
    eligible_indexes = torch.nonzero(keep, as_tuple=False).flatten()
    eligible_prediction_count = int(eligible_indexes.shape[0])
    order = torch.argsort(scores[eligible_indexes], descending=True)
    ordered_indexes = eligible_indexes[order]
    if max_detections is not None:
        ordered_indexes = ordered_indexes[:max_detections]
    pred_boxes = prediction["boxes"][ordered_indexes]
    pred_labels = prediction["labels"][ordered_indexes]
    pred_scores = scores[ordered_indexes]
    target_boxes = target["boxes"]
    target_labels = target["labels"]

    matched_target_indexes: set[int] = set()
    matched_ious: list[float] = []
    prediction_matches: list[DetectionPredictionMatch] = []
    if pred_boxes.numel() and target_boxes.numel():
        ious = box_iou(pred_boxes, target_boxes)
    else:
        ious = torch.zeros((pred_boxes.shape[0], target_boxes.shape[0]))

    for prediction_index, label in enumerate(pred_labels):
        same_label_indexes = [
            target_index
            for target_index, target_label in enumerate(target_labels)
            if int(target_label) == int(label)
        ]
        best_target_index: int | None = None
        best_iou: float | None = None
        if same_label_indexes:
            same_label_ious = ious[prediction_index, same_label_indexes]
            best_position = int(torch.argmax(same_label_ious))
            best_target_index = same_label_indexes[best_position]
            best_iou = float(same_label_ious[best_position])

        available_indexes = [
            target_index
            for target_index in same_label_indexes
            if target_index not in matched_target_indexes
        ]
        matched_target_index: int | None = None
        matched_iou: float | None = None
        if available_indexes:
            available_ious = ious[prediction_index, available_indexes]
            available_position = int(torch.argmax(available_ious))
            candidate_target_index = available_indexes[available_position]
            candidate_iou = float(available_ious[available_position])
            if candidate_iou >= iou_threshold:
                matched_target_index = candidate_target_index
                matched_iou = candidate_iou
                matched_target_indexes.add(candidate_target_index)
                matched_ious.append(candidate_iou)

        source_prediction_index = int(ordered_indexes[prediction_index])
        prediction_matches.append(
            DetectionPredictionMatch(
                prediction_index=source_prediction_index,
                box_xyxy=tuple(
                    float(value) for value in pred_boxes[prediction_index].tolist()
                ),
                label=int(label),
                score=float(pred_scores[prediction_index]),
                is_true_positive=matched_target_index is not None,
                best_target_index=best_target_index,
                best_iou=best_iou,
                matched_target_index=matched_target_index,
                matched_iou=matched_iou,
            )
        )

    true_positives = len(matched_target_indexes)
    prediction_count = int(pred_boxes.shape[0])
    target_count = int(target_boxes.shape[0])
    false_positives = prediction_count - true_positives
    false_negatives = target_count - true_positives
    return DetectionImageEvaluation(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        target_count=target_count,
        prediction_count=prediction_count,
        eligible_prediction_count=eligible_prediction_count,
        truncated_prediction_count=eligible_prediction_count - prediction_count,
        matched_ious=tuple(matched_ious),
        prediction_matches=tuple(prediction_matches),
        unmatched_target_indexes=tuple(
            target_index
            for target_index in range(target_count)
            if target_index not in matched_target_indexes
        ),
    )


def _average_precision_at_iou(
    predictions: list[dict[str, Tensor]],
    targets: list[dict[str, Tensor]],
    *,
    iou_threshold: float,
    target_count: int,
) -> float:
    ranked_predictions = sorted(
        (
            (float(score), image_index, prediction_index)
            for image_index, prediction in enumerate(predictions)
            for prediction_index, score in enumerate(prediction["scores"])
        ),
        key=lambda item: (-item[0], item[1], item[2]),
    )
    if not ranked_predictions:
        return 0.0

    matched_targets: dict[int, set[int]] = {
        image_index: set() for image_index in range(len(targets))
    }
    true_positive_flags: list[int] = []
    false_positive_flags: list[int] = []
    for _, image_index, prediction_index in ranked_predictions:
        prediction = predictions[image_index]
        target = targets[image_index]
        prediction_label = int(prediction["labels"][prediction_index])
        available_indexes = [
            target_index
            for target_index, target_label in enumerate(target["labels"])
            if int(target_label) == prediction_label
            and target_index not in matched_targets[image_index]
        ]
        matched_target_index: int | None = None
        if available_indexes:
            candidate_ious = box_iou(
                prediction["boxes"][prediction_index].unsqueeze(0),
                target["boxes"][available_indexes],
            ).squeeze(0)
            best_position = int(torch.argmax(candidate_ious))
            if float(candidate_ious[best_position]) >= iou_threshold:
                matched_target_index = available_indexes[best_position]
        if matched_target_index is None:
            true_positive_flags.append(0)
            false_positive_flags.append(1)
        else:
            matched_targets[image_index].add(matched_target_index)
            true_positive_flags.append(1)
            false_positive_flags.append(0)

    cumulative_true_positives: list[int] = []
    cumulative_false_positives: list[int] = []
    running_true_positives = 0
    running_false_positives = 0
    for true_positive, false_positive in zip(
        true_positive_flags,
        false_positive_flags,
    ):
        running_true_positives += true_positive
        running_false_positives += false_positive
        cumulative_true_positives.append(running_true_positives)
        cumulative_false_positives.append(running_false_positives)

    recalls = [value / target_count for value in cumulative_true_positives]
    precisions = [
        _safe_divide(true_positives, true_positives + false_positives)
        for true_positives, false_positives in zip(
            cumulative_true_positives,
            cumulative_false_positives,
        )
    ]
    interpolated_precisions = [
        max(
            (
                precision
                for recall, precision in zip(recalls, precisions)
                if recall >= recall_threshold
            ),
            default=0.0,
        )
        for recall_threshold in (index / 100.0 for index in range(101))
    ]
    return sum(interpolated_precisions) / len(interpolated_precisions)


def _low_confidence_target_matches(
    prediction: dict[str, Tensor],
    target: dict[str, Tensor],
    *,
    unmatched_target_indexes: tuple[int, ...],
    iou_threshold: float,
    score_threshold: float,
) -> list[dict[str, Any]]:
    available_targets = set(unmatched_target_indexes)
    low_confidence_indexes = sorted(
        (
            prediction_index
            for prediction_index, score in enumerate(prediction["scores"])
            if float(score) < score_threshold
        ),
        key=lambda prediction_index: float(prediction["scores"][prediction_index]),
        reverse=True,
    )
    matches: list[dict[str, Any]] = []
    for prediction_index in low_confidence_indexes:
        prediction_label = int(prediction["labels"][prediction_index])
        same_label_targets = [
            target_index
            for target_index in sorted(available_targets)
            if int(target["labels"][target_index]) == prediction_label
        ]
        if not same_label_targets:
            continue
        candidate_ious = box_iou(
            prediction["boxes"][prediction_index].unsqueeze(0),
            target["boxes"][same_label_targets],
        ).squeeze(0)
        best_position = int(torch.argmax(candidate_ious))
        best_iou = float(candidate_ious[best_position])
        if best_iou < iou_threshold:
            continue
        target_index = same_label_targets[best_position]
        available_targets.remove(target_index)
        matches.append(
            {
                "prediction_index": prediction_index,
                "target_index": target_index,
                "score": float(prediction["scores"][prediction_index]),
                "iou": best_iou,
                "box_xyxy": prediction["boxes"][prediction_index].tolist(),
            }
        )
    return matches


def _metric_from_accumulator(
    *,
    iou_threshold: float,
    score_threshold: float,
    accumulator: _MetricAccumulator,
) -> DetectionMetric:
    precision = _safe_divide(
        accumulator.true_positives,
        accumulator.true_positives + accumulator.false_positives,
    )
    recall = _safe_divide(
        accumulator.true_positives,
        accumulator.true_positives + accumulator.false_negatives,
    )
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)
    matched_ious = accumulator.matched_ious or []
    mean_matched_iou = (
        sum(matched_ious) / len(matched_ious) if matched_ious else None
    )
    return DetectionMetric(
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        true_positives=accumulator.true_positives,
        false_positives=accumulator.false_positives,
        false_negatives=accumulator.false_negatives,
        target_count=accumulator.target_count,
        prediction_count=accumulator.prediction_count,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_matched_iou=mean_matched_iou,
    )


def _validate_prediction(
    prediction: dict[str, Tensor],
    *,
    num_classes: int,
) -> dict[str, Tensor]:
    for key in ("boxes", "labels", "scores"):
        if key not in prediction:
            raise DetectionEvaluationError(f"Prediction is missing {key}")
    boxes = prediction["boxes"].detach().to(dtype=torch.float32).cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().to(dtype=torch.float32).cpu()
    _validate_boxes(boxes, "prediction boxes")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise DetectionEvaluationError("Prediction labels do not match boxes")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise DetectionEvaluationError("Prediction scores do not match boxes")
    if labels.dtype != torch.int64:
        raise DetectionEvaluationError("Prediction labels must use torch.int64 dtype")
    if (
        torch.any(~torch.isfinite(scores))
        or torch.any(scores < 0.0)
        or torch.any(scores > 1.0)
    ):
        raise DetectionEvaluationError("Prediction scores must be in [0, 1]")
    _validate_labels(labels, num_classes=num_classes, field="prediction labels")
    return {"boxes": boxes, "labels": labels, "scores": scores}


def _validate_target(target: dict[str, Tensor], *, num_classes: int) -> dict[str, Tensor]:
    for key in ("boxes", "labels"):
        if key not in target:
            raise DetectionEvaluationError(f"Target is missing {key}")
    boxes = target["boxes"].detach().to(dtype=torch.float32).cpu()
    labels = target["labels"].detach().cpu()
    _validate_boxes(boxes, "target boxes")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise DetectionEvaluationError("Target labels do not match boxes")
    if labels.dtype != torch.int64:
        raise DetectionEvaluationError("Target labels must use torch.int64 dtype")
    _validate_labels(labels, num_classes=num_classes, field="target labels")
    return {"boxes": boxes, "labels": labels}


def _validate_boxes(boxes: Tensor, field: str) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise DetectionEvaluationError(f"{field} must have shape [N, 4]")
    if boxes.numel() and (
        torch.any(~torch.isfinite(boxes))
        or torch.any(boxes[:, 0] >= boxes[:, 2])
        or torch.any(boxes[:, 1] >= boxes[:, 3])
    ):
        raise DetectionEvaluationError(f"{field} contain invalid boxes")


def _validate_labels(labels: Tensor, *, num_classes: int, field: str) -> None:
    if labels.numel() and (torch.any(labels <= 0) or torch.any(labels >= num_classes)):
        raise DetectionEvaluationError(f"{field} are outside configured classes")


def _build_loader(config: DetectionEvaluationConfig) -> DataLoader:
    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=config.dataset_root,
        manifest_path=config.manifest_path,
        split=config.split,
        variant=config.variant,
        excluded_sample_id_suffixes=config.excluded_sample_id_suffixes,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_detection_batch,
    )


def _validate_evaluation_inputs(config: DetectionEvaluationConfig) -> None:
    if config.num_classes < 2:
        raise DetectionEvaluationError(
            "num_classes must include background and at least one class"
        )
    if config.image_min_size > config.image_max_size:
        raise DetectionEvaluationError("image_min_size must be <= image_max_size")
    for threshold in config.average_precision_iou_thresholds:
        if not 0.0 < threshold < 1.0:
            raise DetectionEvaluationError(
                "Average-precision IoU thresholds must be between 0 and 1"
            )
    if config.analysis_score_threshold is not None:
        if 0.5 not in config.iou_thresholds:
            raise DetectionEvaluationError(
                "Error analysis requires IoU threshold 0.5"
            )
        if config.analysis_score_threshold not in config.score_thresholds:
            raise DetectionEvaluationError(
                "Analysis score threshold must be included in score_thresholds"
            )
    if config.analysis_top_cases <= 0:
        raise DetectionEvaluationError("analysis_top_cases must be positive")
    if config.checkpoint_path.is_symlink() or not config.checkpoint_path.is_file():
        raise DetectionEvaluationError(
            f"Checkpoint is not a regular file: {config.checkpoint_path}"
        )
    if config.output_dir.is_symlink():
        raise DetectionEvaluationError(f"Output directory is a symlink: {config.output_dir}")
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise DetectionEvaluationError(f"Output path is not a directory: {config.output_dir}")


def _select_device(name: DeviceName) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DetectionEvaluationError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_output_file(output_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise DetectionEvaluationError(f"Unsafe output filename: {filename}")
    root = output_dir.resolve(strict=True)
    path = root / filename
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise DetectionEvaluationError(f"Output file escapes output directory: {filename}")
    if path.is_symlink():
        raise DetectionEvaluationError(f"Output file is a symlink: {path}")
    return path


def _table(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DetectionEvaluationError(f"Evaluation config is missing [{name}] table")
    return value


def _path_value(payload: dict[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DetectionEvaluationError(f"{key} must be a non-empty string path")
    if "\x00" in value:
        raise DetectionEvaluationError(f"{key} contains a null byte")
    return Path(value)


def _positive_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionEvaluationError(f"{key} must be a positive integer")
    return value


def _optional_positive_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionEvaluationError(f"{key} must be a positive integer when provided")
    return value


def _optional_positive_int_with_default(
    payload: dict[str, object],
    key: str,
    *,
    default: int,
) -> int:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionEvaluationError(f"{key} must be a positive integer")
    return value


def _non_negative_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DetectionEvaluationError(f"{key} must be a non-negative integer")
    return value


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
        raise DetectionEvaluationError(f"{key} must be between {minimum} and {maximum}")
    return value


def _thresholds(payload: dict[str, object], key: str) -> tuple[float, ...]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise DetectionEvaluationError(f"{key} must be a non-empty list")
    thresholds: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise DetectionEvaluationError(f"{key} must contain numbers")
        parsed = float(item)
        if parsed <= 0.0 or parsed >= 1.0:
            raise DetectionEvaluationError(f"{key} values must be between 0 and 1")
        thresholds.append(parsed)
    return tuple(thresholds)


def _optional_thresholds(
    payload: dict[str, object],
    key: str,
) -> tuple[float, ...]:
    if payload.get(key) is None:
        return ()
    return _thresholds(payload, key)


def _optional_threshold(payload: dict[str, object], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionEvaluationError(f"{key} must be numeric when provided")
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise DetectionEvaluationError(f"{key} must be between 0 and 1")
    return parsed


def _optional_text(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise DetectionEvaluationError(f"{key} must be a non-empty safe string")
    return value.strip()


def _optional_text_list(payload: dict[str, object], key: str) -> tuple[str, ...]:
    value = payload.get(key)
    if value is None:
        return ()
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or "\x00" in item
        for item in value
    ):
        raise DetectionEvaluationError(f"{key} must be a list of safe strings")
    normalized = tuple(item.strip() for item in value)
    if len(normalized) != len(set(normalized)):
        raise DetectionEvaluationError(f"{key} must not contain duplicates")
    return normalized


def _weights_name(
    payload: dict[str, object],
    key: str,
) -> Literal["none", "default"]:
    value = payload.get(key)
    if value == "none" or value == "default":
        return value
    raise DetectionEvaluationError(f"{key} must be 'none' or 'default'")


def _device_name(payload: dict[str, object], key: str) -> DeviceName:
    value = payload.get(key)
    if value == "auto" or value == "cpu" or value == "cuda":
        return value
    raise DetectionEvaluationError(f"{key} must be 'auto', 'cpu', or 'cuda'")


def _split_name(payload: dict[str, object], key: str) -> SplitName:
    value = payload.get(key)
    if value == "train" or value == "validation" or value == "test":
        return value
    raise DetectionEvaluationError(f"{key} must be 'train', 'validation', or 'test'")


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _metric_for_json(metric: DetectionMetric) -> dict[str, float | int | None]:
    return {
        "f1": metric.f1,
        "false_negatives": metric.false_negatives,
        "false_positives": metric.false_positives,
        "iou_threshold": metric.iou_threshold,
        "mean_matched_iou": metric.mean_matched_iou,
        "precision": metric.precision,
        "prediction_count": metric.prediction_count,
        "recall": metric.recall,
        "score_threshold": metric.score_threshold,
        "target_count": metric.target_count,
        "true_positives": metric.true_positives,
    }


def _dataset_summary_for_json(summary: DetectionDatasetSummary) -> dict[str, object]:
    return {
        "image_count": summary.image_count,
        "patient_count": summary.patient_count,
        "target_count": summary.target_count,
        "variant": summary.variant,
        "excluded_sample_id_suffixes": list(summary.excluded_sample_id_suffixes),
    }


def _average_precision_for_json(
    summary: DetectionAveragePrecision | None,
) -> dict[str, object] | None:
    if summary is None:
        return None
    return {
        "ap50": summary.ap50,
        "map50_95": summary.map50_95,
        "method": "single_class_101_point_interpolated",
        "by_iou": [
            {"iou_threshold": threshold, "average_precision": value}
            for threshold, value in summary.by_iou
        ],
    }


def _error_analysis_for_json(
    analysis: DetectionErrorAnalysis | None,
) -> dict[str, object] | None:
    if analysis is None:
        return None
    return {
        "iou_threshold": analysis.iou_threshold,
        "score_threshold": analysis.score_threshold,
        "false_positives": analysis.false_positives,
        "false_negatives": analysis.false_negatives,
        "duplicate_detections": analysis.duplicate_detections,
        "localization_failures": analysis.localization_failures,
        "background_false_positives": analysis.background_false_positives,
        "low_confidence_matches": analysis.low_confidence_matches,
        "unexplained_false_negatives": analysis.unexplained_false_negatives,
        "top_failure_cases": list(analysis.top_failure_cases),
    }
