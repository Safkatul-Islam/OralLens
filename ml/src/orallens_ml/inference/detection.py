"""Checkpoint-backed object-detection inference."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image
import torch
from torch import Tensor
from torch.nn import Module

from orallens_ml.modeling.detection import (
    DetectionModelConfig,
    DetectionModelError,
    create_fasterrcnn_resnet50_fpn,
    load_detection_checkpoint,
    validate_checkpoint_compatibility,
)
from orallens_ml.inference.input_assessment import (
    InputAssessment,
    InputAssessmentError,
    InputAssessmentPolicy,
    load_and_assess_image,
)

DeviceName = Literal["auto", "cpu", "cuda"]
WeightsName = Literal["none", "default"]
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


class DetectionInferenceError(ValueError):
    """Raised when detection inference cannot run safely."""


class InvalidDetectionImageError(DetectionInferenceError):
    """Raised when an input file cannot be decoded as a safe image."""


@dataclass(frozen=True, slots=True)
class DetectionInferenceConfig:
    """Validated settings for one detection inference run."""

    model_name: str
    checkpoint_path: Path
    output_dir: Path
    num_classes: int
    image_min_size: int
    image_max_size: int
    trainable_backbone_layers: int
    pretrained_weights: WeightsName
    device: DeviceName
    score_threshold: float
    max_detections: int
    input_assessment_policy: InputAssessmentPolicy


@dataclass(frozen=True, slots=True)
class DetectionPrediction:
    """One filtered detection prediction."""

    box_xyxy: tuple[float, float, float, float]
    label: int
    score: float


@dataclass(frozen=True, slots=True)
class DetectionInferenceResult:
    """Files and predictions produced by an inference run."""

    model_name: str
    image_path: Path
    output_path: Path
    image_width: int
    image_height: int
    predictions: tuple[DetectionPrediction, ...]
    input_assessment: InputAssessment


def load_detection_inference_config(config_path: Path) -> DetectionInferenceConfig:
    """Load and validate a detection inference TOML file."""

    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise DetectionInferenceError(f"Inference config is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise DetectionInferenceError(f"Inference config is malformed TOML: {exc}") from exc

    model = _table(payload, "model")
    inference = _table(payload, "inference")
    output = _table(payload, "output")
    input_policy = _input_assessment_policy(payload.get("input_policy"))

    return DetectionInferenceConfig(
        model_name=_model_name(model, "model_name"),
        checkpoint_path=_path_value(model, "checkpoint_path"),
        output_dir=_path_value(output, "output_dir"),
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
        device=_device_name(inference, "device"),
        score_threshold=_threshold(inference, "score_threshold"),
        max_detections=_positive_int(inference, "max_detections"),
        input_assessment_policy=input_policy,
    )


def run_detection_inference(
    config: DetectionInferenceConfig,
    *,
    image_path: Path,
    model_factory: Callable[[DetectionInferenceConfig], Module] | None = None,
) -> DetectionInferenceResult:
    """Run one-image detection inference and write predictions JSON."""

    _validate_inference_inputs(config)
    _validate_input_image_path(image_path)
    try:
        assessed_image = load_and_assess_image(
            image_path,
            policy=config.input_assessment_policy,
        )
    except InputAssessmentError as exc:
        raise InvalidDetectionImageError(str(exc)) from exc
    assessment = assessed_image.assessment
    width = assessment.image_width
    height = assessment.image_height
    _prepare_output_dir(config.output_dir)
    predictions: tuple[DetectionPrediction, ...] = ()
    if assessment.is_supported:
        if assessed_image.image is None:
            raise DetectionInferenceError("Supported input is missing decoded image data")
        image = _pil_to_float_tensor(assessed_image.image)
        device = _select_device(config.device)
        model = _build_model(config, model_factory=model_factory).to(device)
        _load_checkpoint(model, config.checkpoint_path, config=config, device=device)
        model.eval()

        with torch.no_grad():
            outputs = model([image.to(device)])
        if not isinstance(outputs, list) or len(outputs) != 1:
            raise DetectionInferenceError("Model did not return one prediction dictionary")
        predictions = _filter_predictions(
            outputs[0],
            score_threshold=config.score_threshold,
            max_detections=config.max_detections,
            num_classes=config.num_classes,
        )
    output_path = _safe_output_file(config.output_dir, f"{Path(image_path).stem}.json")
    output_path.write_text(
        json.dumps(
            {
                "image_height": height,
                "image_path": str(Path(image_path)),
                "image_width": width,
                "model_name": config.model_name,
                "input_assessment": _assessment_payload(assessment),
                "prediction_count": len(predictions),
                "predictions": [
                    {
                        "box_xyxy": list(prediction.box_xyxy),
                        "label": prediction.label,
                        "score": prediction.score,
                    }
                    for prediction in predictions
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return DetectionInferenceResult(
        model_name=config.model_name,
        image_path=Path(image_path),
        output_path=output_path,
        image_width=width,
        image_height=height,
        predictions=predictions,
        input_assessment=assessment,
    )


def _filter_predictions(
    prediction: dict[str, Tensor],
    *,
    score_threshold: float,
    max_detections: int,
    num_classes: int,
) -> tuple[DetectionPrediction, ...]:
    for key in ("boxes", "labels", "scores"):
        if key not in prediction:
            raise DetectionInferenceError(f"Prediction is missing {key}")
    boxes = prediction["boxes"].detach().to(dtype=torch.float32).cpu()
    labels = prediction["labels"].detach().cpu()
    scores = prediction["scores"].detach().to(dtype=torch.float32).cpu()
    _validate_prediction_tensors(boxes, labels, scores, num_classes=num_classes)
    keep = scores >= score_threshold
    boxes = boxes[keep]
    labels = labels[keep]
    scores = scores[keep]
    order = torch.argsort(scores, descending=True)[:max_detections]
    return tuple(
        DetectionPrediction(
            box_xyxy=tuple(float(value) for value in boxes[index].tolist()),
            label=int(labels[index]),
            score=float(scores[index]),
        )
        for index in order
    )


def _validate_prediction_tensors(
    boxes: Tensor,
    labels: Tensor,
    scores: Tensor,
    *,
    num_classes: int,
) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise DetectionInferenceError("Prediction boxes must have shape [N, 4]")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise DetectionInferenceError("Prediction labels do not match boxes")
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        raise DetectionInferenceError("Prediction scores do not match boxes")
    if labels.dtype != torch.int64:
        raise DetectionInferenceError("Prediction labels must use torch.int64 dtype")
    if boxes.numel() and (
        torch.any(~torch.isfinite(boxes))
        or torch.any(boxes[:, 0] >= boxes[:, 2])
        or torch.any(boxes[:, 1] >= boxes[:, 3])
    ):
        raise DetectionInferenceError("Prediction boxes contain invalid boxes")
    if scores.numel() and (
        torch.any(~torch.isfinite(scores))
        or torch.any(scores < 0.0)
        or torch.any(scores > 1.0)
    ):
        raise DetectionInferenceError("Prediction scores must be in [0, 1]")
    if labels.numel() and (torch.any(labels <= 0) or torch.any(labels >= num_classes)):
        raise DetectionInferenceError("Prediction labels are outside configured classes")


def _pil_to_float_tensor(image: Image.Image) -> Tensor:
    width, height = image.size
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return (
        data.view(height, width, 3)
        .permute(2, 0, 1)
        .contiguous()
        .to(torch.float32)
        .div(255.0)
    )


def _assessment_payload(assessment: InputAssessment) -> dict[str, object]:
    return {
        "status": assessment.status,
        "reason_codes": list(assessment.reason_codes),
        "image_width": assessment.image_width,
        "image_height": assessment.image_height,
        "mean_luminance": assessment.mean_luminance,
        "luminance_stddev": assessment.luminance_stddev,
    }


def _build_model(
    config: DetectionInferenceConfig,
    *,
    model_factory: Callable[[DetectionInferenceConfig], Module] | None,
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
    config: DetectionInferenceConfig,
    device: torch.device,
) -> None:
    try:
        checkpoint = load_detection_checkpoint(checkpoint_path, device=device)
        validate_checkpoint_compatibility(
            checkpoint_config=checkpoint.config,
            runtime_config=DetectionModelConfig.from_runtime_config(config),
        )
    except DetectionModelError as exc:
        raise DetectionInferenceError(str(exc)) from exc
    model.load_state_dict(checkpoint.model_state_dict)


def _validate_inference_inputs(config: DetectionInferenceConfig) -> None:
    if config.num_classes < 2:
        raise DetectionInferenceError(
            "num_classes must include background and at least one class"
        )
    if config.image_min_size > config.image_max_size:
        raise DetectionInferenceError("image_min_size must be <= image_max_size")
    if config.checkpoint_path.is_symlink() or not config.checkpoint_path.is_file():
        raise DetectionInferenceError(
            f"Checkpoint is not a regular file: {config.checkpoint_path}"
        )
    if config.output_dir.is_symlink():
        raise DetectionInferenceError(f"Output directory is a symlink: {config.output_dir}")
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise DetectionInferenceError(f"Output path is not a directory: {config.output_dir}")
    _validate_input_assessment_policy(config.input_assessment_policy)


def _validate_input_image_path(image_path: Path) -> None:
    path = Path(image_path)
    if path.suffix.lower() not in _IMAGE_EXTENSIONS:
        raise InvalidDetectionImageError(
            f"Unsupported input image extension: {path.suffix}"
        )


def _validate_input_assessment_policy(policy: InputAssessmentPolicy) -> None:
    if policy.min_short_side <= 0:
        raise DetectionInferenceError("min_short_side must be positive")
    if policy.max_pixels <= 0:
        raise DetectionInferenceError("max_pixels must be positive")
    if not 0.0 <= policy.min_mean_luminance < policy.max_mean_luminance <= 1.0:
        raise DetectionInferenceError("luminance bounds must satisfy 0 <= min < max <= 1")
    if not 0.0 <= policy.min_luminance_stddev <= 1.0:
        raise DetectionInferenceError("min_luminance_stddev must be between 0 and 1")


def _select_device(name: DeviceName) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DetectionInferenceError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_output_file(output_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise DetectionInferenceError(f"Unsafe output filename: {filename}")
    root = output_dir.resolve(strict=True)
    path = root / filename
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise DetectionInferenceError(f"Output file escapes output directory: {filename}")
    if path.is_symlink():
        raise DetectionInferenceError(f"Output file is a symlink: {path}")
    return path


def _table(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DetectionInferenceError(f"Inference config is missing [{name}] table")
    return value


def _path_value(payload: dict[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DetectionInferenceError(f"{key} must be a non-empty string path")
    if "\x00" in value:
        raise DetectionInferenceError(f"{key} contains a null byte")
    return Path(value)


def _model_name(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise DetectionInferenceError(f"{key} must be a string")
    normalized = value.strip()
    if not normalized or len(normalized) > 100:
        raise DetectionInferenceError(
            f"{key} must contain between 1 and 100 characters"
        )
    if any(ord(character) < 32 or ord(character) == 127 for character in normalized):
        raise DetectionInferenceError(f"{key} contains a control character")
    return normalized


def _positive_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionInferenceError(f"{key} must be a positive integer")
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
        raise DetectionInferenceError(f"{key} must be between {minimum} and {maximum}")
    return value


def _threshold(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DetectionInferenceError(f"{key} must be a number")
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise DetectionInferenceError(f"{key} must be between 0 and 1")
    return parsed


def _weights_name(payload: dict[str, object], key: str) -> WeightsName:
    value = payload.get(key)
    if value == "none" or value == "default":
        return value
    raise DetectionInferenceError(f"{key} must be 'none' or 'default'")


def _device_name(payload: dict[str, object], key: str) -> DeviceName:
    value = payload.get(key)
    if value == "auto" or value == "cpu" or value == "cuda":
        return value
    raise DetectionInferenceError(f"{key} must be 'auto', 'cpu', or 'cuda'")


def _input_assessment_policy(value: object) -> InputAssessmentPolicy:
    if value is None:
        return InputAssessmentPolicy(
            enabled=False,
            min_short_side=256,
            max_pixels=30_000_000,
            min_mean_luminance=0.05,
            max_mean_luminance=0.98,
            min_luminance_stddev=0.02,
        )
    if not isinstance(value, dict):
        raise DetectionInferenceError("input_policy must be a TOML table")
    policy = InputAssessmentPolicy(
        enabled=_boolean(value, "enabled"),
        min_short_side=_positive_int(value, "min_short_side"),
        max_pixels=_positive_int(value, "max_pixels"),
        min_mean_luminance=_threshold(value, "min_mean_luminance"),
        max_mean_luminance=_threshold(value, "max_mean_luminance"),
        min_luminance_stddev=_threshold(value, "min_luminance_stddev"),
    )
    _validate_input_assessment_policy(policy)
    return policy


def _boolean(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise DetectionInferenceError(f"{key} must be a boolean")
    return value
