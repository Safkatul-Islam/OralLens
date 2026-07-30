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
class DetectionEvaluationResult:
    """Files and metrics produced by an evaluation run."""

    metrics_path: Path
    metrics: tuple[DetectionMetric, ...]


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
            batch_metrics = evaluate_detection_predictions(
                predictions,
                target_batch,
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

    metrics_path = _safe_output_file(config.output_dir, "evaluation_metrics.json")
    metrics_path.write_text(
        json.dumps(
            {
                "metrics": [_metric_for_json(metric) for metric in metrics],
                "split": config.split,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return DetectionEvaluationResult(metrics_path=metrics_path, metrics=metrics)


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
) -> _MetricAccumulator:
    scores = prediction["scores"]
    keep = scores >= score_threshold
    pred_boxes = prediction["boxes"][keep]
    pred_labels = prediction["labels"][keep]
    pred_scores = scores[keep]
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]
    pred_labels = pred_labels[order]
    target_boxes = target["boxes"]
    target_labels = target["labels"]

    matched_target_indexes: set[int] = set()
    matched_ious: list[float] = []
    false_positives = 0
    if pred_boxes.numel() and target_boxes.numel():
        ious = box_iou(pred_boxes, target_boxes)
    else:
        ious = torch.zeros((pred_boxes.shape[0], target_boxes.shape[0]))

    for prediction_index, label in enumerate(pred_labels):
        candidate_indexes = [
            target_index
            for target_index, target_label in enumerate(target_labels)
            if int(target_label) == int(label) and target_index not in matched_target_indexes
        ]
        if not candidate_indexes:
            false_positives += 1
            continue
        candidate_ious = ious[prediction_index, candidate_indexes]
        best_position = int(torch.argmax(candidate_ious))
        best_iou = float(candidate_ious[best_position])
        if best_iou >= iou_threshold:
            matched_target_indexes.add(candidate_indexes[best_position])
            matched_ious.append(best_iou)
        else:
            false_positives += 1

    true_positives = len(matched_target_indexes)
    false_negatives = target_boxes.shape[0] - true_positives
    return _MetricAccumulator(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        target_count=target_boxes.shape[0],
        prediction_count=pred_boxes.shape[0],
        matched_ious=matched_ious,
    )


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
