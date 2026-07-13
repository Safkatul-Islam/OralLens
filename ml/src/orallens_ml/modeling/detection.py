"""Shared detector construction and checkpoint handling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import urlparse

import torch
from torch.nn import Module

WeightsName = Literal["none", "default"]

_COMPATIBILITY_FIELDS = (
    "num_classes",
    "image_min_size",
    "image_max_size",
)


class DetectionModelError(ValueError):
    """Raised when detector model construction or checkpoint handling fails."""


@dataclass(frozen=True, slots=True)
class DetectionModelConfig:
    """Architecture settings shared by training, evaluation, and inference."""

    num_classes: int
    image_min_size: int
    image_max_size: int
    trainable_backbone_layers: int
    pretrained_weights: WeightsName

    @classmethod
    def from_runtime_config(cls, config: object) -> "DetectionModelConfig":
        return cls(
            num_classes=int(getattr(config, "num_classes")),
            image_min_size=int(getattr(config, "image_min_size")),
            image_max_size=int(getattr(config, "image_max_size")),
            trainable_backbone_layers=int(getattr(config, "trainable_backbone_layers")),
            pretrained_weights=getattr(config, "pretrained_weights"),
        )


@dataclass(frozen=True, slots=True)
class DetectionCheckpoint:
    """Loaded detection checkpoint payload."""

    model_state_dict: dict[str, object]
    config: dict[str, object]
    metrics: Sequence[Mapping[str, object]]


def create_fasterrcnn_resnet50_fpn(config: DetectionModelConfig) -> Module:
    """Create the TorchVision Faster R-CNN baseline model."""

    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    if config.pretrained_weights == "default":
        require_pretrained_weights_available(config)
        weights = _default_detection_weights()
    elif config.pretrained_weights == "none":
        weights = None
    else:
        raise DetectionModelError(
            "pretrained_weights must be 'none' or 'default'"
        )
    kwargs: dict[str, object] = {
        "weights": weights,
        "weights_backbone": None,
        "min_size": config.image_min_size,
        "max_size": config.image_max_size,
    }
    if weights is not None:
        kwargs["trainable_backbone_layers"] = config.trainable_backbone_layers
    model = fasterrcnn_resnet50_fpn(**kwargs)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, config.num_classes)
    return model


def pretrained_weights_cache_path(config: DetectionModelConfig) -> Path | None:
    """Return the Torch cache path required by the configured pretrained weights."""

    if config.pretrained_weights == "none":
        return None
    if config.pretrained_weights != "default":
        raise DetectionModelError(
            "pretrained_weights must be 'none' or 'default'"
        )
    weights = _default_detection_weights()
    filename = PurePosixPath(urlparse(weights.url).path).name
    if not filename:
        raise DetectionModelError("TorchVision pretrained weights URL is missing a filename")
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def require_pretrained_weights_available(config: DetectionModelConfig) -> None:
    """Fail before TorchVision tries to download weights implicitly."""

    cache_path = pretrained_weights_cache_path(config)
    if cache_path is None or cache_path.is_file():
        return
    raise DetectionModelError(
        "pretrained_weights='default' requires official TorchVision weights to "
        "exist in the local Torch cache before model construction. This project "
        "does not download pretrained weights implicitly. Expected cache file: "
        f"{cache_path}. Approve an official TorchVision weights download before "
        "baseline training, or set pretrained_weights='none' for smoke/offline runs."
    )


def _default_detection_weights() -> Any:
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

    return FasterRCNN_ResNet50_FPN_Weights.DEFAULT


def save_detection_checkpoint(
    *,
    model: Module,
    path: Path,
    config: Mapping[str, object],
    metrics: Sequence[Mapping[str, object]],
) -> None:
    """Save a state-dict-only detection checkpoint."""

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": dict(config),
            "metrics": list(metrics),
        },
        path,
    )


def load_detection_checkpoint(path: Path, *, device: torch.device) -> DetectionCheckpoint:
    """Load a detection checkpoint without falling back to unrestricted pickle."""

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise DetectionModelError("Checkpoint payload must be a dictionary")
    state_dict = checkpoint.get("model_state_dict")
    if not isinstance(state_dict, dict):
        raise DetectionModelError("Checkpoint is missing model_state_dict")
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise DetectionModelError("Checkpoint is missing config")
    raw_metrics = checkpoint.get("metrics", [])
    if not isinstance(raw_metrics, list):
        raise DetectionModelError("Checkpoint metrics are invalid")
    return DetectionCheckpoint(
        model_state_dict=state_dict,
        config=dict(raw_config),
        metrics=raw_metrics,
    )


def validate_checkpoint_compatibility(
    *,
    checkpoint_config: Mapping[str, object],
    runtime_config: DetectionModelConfig,
) -> None:
    """Reject checkpoints whose architecture contract conflicts with runtime config."""

    runtime_values = {
        "num_classes": runtime_config.num_classes,
        "image_min_size": runtime_config.image_min_size,
        "image_max_size": runtime_config.image_max_size,
    }
    mismatches = [
        field
        for field in _COMPATIBILITY_FIELDS
        if checkpoint_config.get(field) != runtime_values[field]
    ]
    if mismatches:
        details = ", ".join(
            f"{field}: checkpoint={checkpoint_config.get(field)!r}, runtime={runtime_values[field]!r}"
            for field in mismatches
        )
        raise DetectionModelError(f"Checkpoint config is incompatible: {details}")
