"""Baseline object-detection training for orthodontic plaque annotations."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from orallens_ml.data.orthodontic_plaque_dataset import (
    OrthodonticPlaquePart2Dataset,
    OrthodonticPlaqueTarget,
)
from orallens_ml.modeling.detection import (
    DetectionModelConfig,
    DetectionModelError,
    create_fasterrcnn_resnet50_fpn as create_detection_model,
    save_detection_checkpoint,
)

DeviceName = Literal["auto", "cpu", "cuda"]
WeightsName = Literal["none", "default"]


class DetectionTrainingError(ValueError):
    """Raised when detection training cannot start safely."""


@dataclass(frozen=True, slots=True)
class DetectionTrainingConfig:
    """Validated settings for one detection training run."""

    dataset_root: Path
    manifest_path: Path
    output_dir: Path
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    num_workers: int
    num_classes: int
    image_min_size: int
    image_max_size: int
    trainable_backbone_layers: int
    pretrained_weights: WeightsName
    device: DeviceName
    seed: int
    max_train_batches: int | None = None
    max_validation_batches: int | None = None


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Loss metrics for a completed epoch."""

    epoch: int
    train_loss: float
    validation_loss: float | None
    learning_rate: float


@dataclass(frozen=True, slots=True)
class TrainingRunResult:
    """Files and metrics produced by a training run."""

    checkpoint_path: Path
    metrics_path: Path
    metrics: tuple[EpochMetrics, ...]


def load_detection_training_config(config_path: Path) -> DetectionTrainingConfig:
    """Load and validate a detection training TOML file."""

    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise DetectionTrainingError(f"Training config is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise DetectionTrainingError(f"Training config is malformed TOML: {exc}") from exc

    training = _table(payload, "training")
    data = _table(payload, "data")
    model = _table(payload, "model")
    output = _table(payload, "output")

    return DetectionTrainingConfig(
        dataset_root=_path_value(data, "dataset_root"),
        manifest_path=_path_value(data, "manifest_path"),
        output_dir=_path_value(output, "output_dir"),
        epochs=_positive_int(training, "epochs"),
        batch_size=_positive_int(training, "batch_size"),
        learning_rate=_positive_float(training, "learning_rate"),
        momentum=_bounded_float(training, "momentum", minimum=0.0, maximum=1.0),
        weight_decay=_non_negative_float(training, "weight_decay"),
        num_workers=_non_negative_int(training, "num_workers"),
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
        device=_device_name(training, "device"),
        seed=_non_negative_int(training, "seed"),
        max_train_batches=_optional_positive_int(training, "max_train_batches"),
        max_validation_batches=_optional_positive_int(training, "max_validation_batches"),
    )


def make_detection_target(
    image: Tensor,
    target: OrthodonticPlaqueTarget,
    *,
    image_id: int,
    num_classes: int,
) -> dict[str, Tensor]:
    """Convert normalized manifest targets into TorchVision detection targets."""

    if image.ndim != 3:
        raise DetectionTrainingError("Detection images must have shape [C, H, W]")
    _, height, width = image.shape
    boxes = target.boxes
    labels = target.labels
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise DetectionTrainingError(f"Invalid box tensor for sample: {target.sample_id}")
    if labels.ndim != 1 or labels.shape[0] != boxes.shape[0]:
        raise DetectionTrainingError(f"Invalid label tensor for sample: {target.sample_id}")
    if boxes.numel() == 0:
        raise DetectionTrainingError(f"Sample has no boxes: {target.sample_id}")
    if torch.any(boxes < 0.0) or torch.any(boxes > 1.0):
        raise DetectionTrainingError(f"Normalized boxes are out of range: {target.sample_id}")
    if torch.any(boxes[:, 0] >= boxes[:, 2]) or torch.any(boxes[:, 1] >= boxes[:, 3]):
        raise DetectionTrainingError(f"Normalized boxes have invalid extents: {target.sample_id}")
    if num_classes != 2:
        raise DetectionTrainingError("The v1 plaque detector requires num_classes=2")
    if torch.any(labels < 0):
        raise DetectionTrainingError(f"Source labels are outside configured classes: {target.sample_id}")

    scale = torch.tensor([width, height, width, height], dtype=torch.float32)
    pixel_boxes = boxes.to(dtype=torch.float32) * scale
    area = (pixel_boxes[:, 2] - pixel_boxes[:, 0]) * (
        pixel_boxes[:, 3] - pixel_boxes[:, 1]
    )
    foreground_labels = torch.ones_like(labels, dtype=torch.int64)
    return {
        "boxes": pixel_boxes,
        "labels": foreground_labels,
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": area.to(dtype=torch.float32),
        "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
    }


def collate_detection_batch(
    batch: Iterable[tuple[Tensor, OrthodonticPlaqueTarget]],
) -> tuple[list[Tensor], list[OrthodonticPlaqueTarget]]:
    """Collate variable-sized detection samples without stacking images."""

    images: list[Tensor] = []
    targets: list[OrthodonticPlaqueTarget] = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


def create_fasterrcnn_resnet50_fpn(config: DetectionTrainingConfig) -> Module:
    """Create the TorchVision Faster R-CNN baseline model."""

    return create_detection_model(DetectionModelConfig.from_runtime_config(config))


def run_detection_training(
    config: DetectionTrainingConfig,
    *,
    model_factory: Callable[[DetectionTrainingConfig], Module] = create_fasterrcnn_resnet50_fpn,
) -> TrainingRunResult:
    """Run a baseline detection training job and write checkpoint artifacts."""

    _validate_training_inputs(config)
    _prepare_output_dir(config.output_dir)
    torch.manual_seed(config.seed)
    device = _select_device(config.device)
    train_loader = _build_loader(config, split="train", shuffle=True)
    validation_loader = _build_loader(config, split="validation", shuffle=False)
    try:
        model = model_factory(config).to(device)
    except DetectionModelError as exc:
        raise DetectionTrainingError(str(exc)) from exc
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    metrics: list[EpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        train_loss = _run_loss_epoch(
            model,
            train_loader,
            device=device,
            optimizer=optimizer,
            num_classes=config.num_classes,
            max_batches=config.max_train_batches,
        )
        validation_loss = _run_loss_epoch(
            model,
            validation_loader,
            device=device,
            optimizer=None,
            num_classes=config.num_classes,
            max_batches=config.max_validation_batches,
        )
        metrics.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
                learning_rate=optimizer.param_groups[0]["lr"],
            )
        )

    checkpoint_path = _safe_output_file(config.output_dir, "checkpoint_last.pt")
    metrics_path = _safe_output_file(config.output_dir, "metrics.json")
    metric_payload = [_metric_for_json(metric) for metric in metrics]
    save_detection_checkpoint(
        model=model,
        path=checkpoint_path,
        config=_config_for_json(config),
        metrics=metric_payload,
    )
    metrics_path.write_text(
        json.dumps(
            {"metrics": metric_payload},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return TrainingRunResult(
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        metrics=tuple(metrics),
    )


def _run_loss_epoch(
    model: Module,
    loader: DataLoader,
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    num_classes: int,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    batches = 0
    grad_enabled = optimizer is not None
    sample_offset = 0
    with _validation_loss_module_modes(model, enabled=optimizer is None):
        for batch_index, (images, raw_targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            image_batch = [image.to(device) for image in images]
            target_batch = [
                {
                    key: value.to(device)
                    for key, value in make_detection_target(
                        image,
                        target,
                        image_id=sample_offset + target_index,
                        num_classes=num_classes,
                    ).items()
                }
                for target_index, (image, target) in enumerate(zip(images, raw_targets))
            ]
            sample_offset += len(images)
            with torch.set_grad_enabled(grad_enabled):
                losses = model(image_batch, target_batch)
                loss = sum(loss_value for loss_value in losses.values())
                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
    if batches == 0:
        raise DetectionTrainingError("No batches were processed")
    return total_loss / batches


@contextmanager
def _validation_loss_module_modes(model: Module, *, enabled: bool) -> Iterable[None]:
    if not enabled:
        yield
        return
    affected_modules = [
        module
        for module in model.modules()
        if isinstance(
            module,
            (
                nn.BatchNorm1d,
                nn.BatchNorm2d,
                nn.BatchNorm3d,
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.SyncBatchNorm,
            ),
        )
    ]
    previous_modes = [module.training for module in affected_modules]
    try:
        for module in affected_modules:
            module.eval()
        yield
    finally:
        for module, mode in zip(affected_modules, previous_modes):
            module.train(mode)


def _build_loader(
    config: DetectionTrainingConfig,
    *,
    split: Literal["train", "validation"],
    shuffle: bool,
) -> DataLoader:
    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=config.dataset_root,
        manifest_path=config.manifest_path,
        split=split,
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_detection_batch,
    )


def _select_device(name: DeviceName) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise DetectionTrainingError("CUDA was requested but is not available")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _validate_training_inputs(config: DetectionTrainingConfig) -> None:
    if config.num_classes < 2:
        raise DetectionTrainingError("num_classes must include background and at least one class")
    if config.image_min_size > config.image_max_size:
        raise DetectionTrainingError("image_min_size must be <= image_max_size")
    if config.output_dir.is_symlink():
        raise DetectionTrainingError(f"Output directory is a symlink: {config.output_dir}")
    if config.output_dir.exists() and not config.output_dir.is_dir():
        raise DetectionTrainingError(f"Output path is not a directory: {config.output_dir}")


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _safe_output_file(output_dir: Path, filename: str) -> Path:
    if Path(filename).name != filename:
        raise DetectionTrainingError(f"Unsafe output filename: {filename}")
    root = output_dir.resolve(strict=True)
    path = root / filename
    resolved = path.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise DetectionTrainingError(f"Output file escapes output directory: {filename}")
    if path.is_symlink():
        raise DetectionTrainingError(f"Output file is a symlink: {path}")
    return path


def _table(payload: dict[str, object], name: str) -> dict[str, object]:
    value = payload.get(name)
    if not isinstance(value, dict):
        raise DetectionTrainingError(f"Training config is missing [{name}] table")
    return value


def _path_value(payload: dict[str, object], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DetectionTrainingError(f"{key} must be a non-empty string path")
    if "\x00" in value:
        raise DetectionTrainingError(f"{key} contains a null byte")
    return Path(value)


def _positive_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionTrainingError(f"{key} must be a positive integer")
    return value


def _optional_positive_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DetectionTrainingError(f"{key} must be a positive integer when provided")
    return value


def _non_negative_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DetectionTrainingError(f"{key} must be a non-negative integer")
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
        raise DetectionTrainingError(f"{key} must be between {minimum} and {maximum}")
    return value


def _positive_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0.0:
        raise DetectionTrainingError(f"{key} must be a positive number")
    return float(value)


def _non_negative_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0.0:
        raise DetectionTrainingError(f"{key} must be a non-negative number")
    return float(value)


def _bounded_float(
    payload: dict[str, object],
    key: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < minimum
        or value > maximum
    ):
        raise DetectionTrainingError(f"{key} must be between {minimum} and {maximum}")
    return float(value)


def _weights_name(payload: dict[str, object], key: str) -> WeightsName:
    value = payload.get(key)
    if value == "none" or value == "default":
        return value
    raise DetectionTrainingError(f"{key} must be 'none' or 'default'")


def _device_name(payload: dict[str, object], key: str) -> DeviceName:
    value = payload.get(key)
    if value == "auto" or value == "cpu" or value == "cuda":
        return value
    raise DetectionTrainingError(f"{key} must be 'auto', 'cpu', or 'cuda'")


def _config_for_json(config: DetectionTrainingConfig) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "dataset_root": str(config.dataset_root),
        "device": config.device,
        "epochs": config.epochs,
        "image_max_size": config.image_max_size,
        "image_min_size": config.image_min_size,
        "learning_rate": config.learning_rate,
        "manifest_path": str(config.manifest_path),
        "max_train_batches": config.max_train_batches,
        "max_validation_batches": config.max_validation_batches,
        "momentum": config.momentum,
        "num_classes": config.num_classes,
        "num_workers": config.num_workers,
        "output_dir": str(config.output_dir),
        "pretrained_weights": config.pretrained_weights,
        "seed": config.seed,
        "trainable_backbone_layers": config.trainable_backbone_layers,
        "weight_decay": config.weight_decay,
    }


def _metric_for_json(metric: EpochMetrics) -> dict[str, float | int | None]:
    return {
        "epoch": metric.epoch,
        "learning_rate": metric.learning_rate,
        "train_loss": metric.train_loss,
        "validation_loss": metric.validation_loss,
    }
