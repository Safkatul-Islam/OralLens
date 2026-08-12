from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from PIL import Image
import pytest
import torch
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

import orallens_ml.training.detection as detection_training
from orallens_ml.modeling.detection import DetectionModelError
from orallens_ml.training.detection import (
    DetectionTrainingConfig,
    DetectionTrainingError,
    collate_detection_batch,
    load_detection_training_config,
    make_detection_target,
    run_detection_training,
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


class TinyDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if targets is None:
            raise AssertionError("training smoke test expects targets")
        image_loss = sum(image.mean() for image in images) * self.weight
        box_anchor = sum(target["boxes"].mean() for target in targets) * 0.0
        return {"loss_classifier": image_loss + box_anchor}


class TinyBackbone(torch.nn.Module):
    out_channels = 4

    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Conv2d(3, self.out_channels, kernel_size=3, padding=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.features(images)


class BatchNormTrackingDetectionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.batch_norm = torch.nn.BatchNorm2d(3)
        self.dropout = torch.nn.Dropout(p=0.5)
        self.batch_norm_training_flags: list[bool] = []
        self.dropout_training_flags: list[bool] = []
        self.root_training_flags: list[bool] = []

    def forward(
        self,
        images: list[torch.Tensor],
        targets: list[dict[str, torch.Tensor]] | None = None,
    ) -> dict[str, torch.Tensor]:
        if targets is None:
            raise AssertionError("training smoke test expects targets")
        self.root_training_flags.append(self.training)
        self.batch_norm_training_flags.append(self.batch_norm.training)
        self.dropout_training_flags.append(self.dropout.training)
        batch = torch.stack(images)
        normalized = self.dropout(self.batch_norm(batch))
        box_anchor = sum(target["boxes"].mean() for target in targets) * 0.0
        return {"loss_classifier": normalized.mean() * self.weight + box_anchor}


def write_manifest_dataset(root: Path) -> tuple[Path, Path]:
    dataset_root = root / "dataset"
    manifest_path = root / "manifest.csv"
    rows = [
        ("train-sample", "patient0001", "train", "data/images/patient0001/train.jpg"),
        (
            "validation-sample",
            "patient0002",
            "validation",
            "data/images/patient0002/validation.jpg",
        ),
    ]
    for index, (_, _, _, relative_path) in enumerate(rows):
        image_path = dataset_root.joinpath(*relative_path.split("/"))
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 6), color=(20 + index, 40, 60)).save(image_path)

    annotation = {
        "class_id": 1,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.25,
        "height": 0.5,
        "tooth_id": 7,
        "source_label_line": 1,
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for line_number, (sample_id, patient_id, split, relative_path) in enumerate(
            rows,
            start=2,
        ):
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "patient_id": patient_id,
                    "split": split,
                    "image_relative_path": relative_path,
                    "source_csv_line": str(line_number),
                    "annotation_count": "1",
                    "annotations_json": json.dumps([annotation]),
                }
            )
    return dataset_root, manifest_path


def test_load_detection_training_config_validates_values(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[data]
dataset_root = "dataset"
manifest_path = "manifest.csv"

[model]
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 1
pretrained_weights = "none"

[training]
epochs = 1
batch_size = 1
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0005
num_workers = 0
device = "cpu"
seed = 1
max_train_batches = 1

[output]
output_dir = "runs/test"
""".strip(),
        encoding="utf-8",
    )

    config = load_detection_training_config(config_path)

    assert config.epochs == 1
    assert config.pretrained_weights == "none"
    assert config.max_train_batches == 1
    assert config.resume_checkpoint_path is None


def test_load_detection_training_config_rejects_invalid_device(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[data]
dataset_root = "dataset"
manifest_path = "manifest.csv"
[model]
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 1
pretrained_weights = "none"
[training]
epochs = 1
batch_size = 1
learning_rate = 0.01
momentum = 0.9
weight_decay = 0.0
num_workers = 0
device = "gpu"
seed = 1
[output]
output_dir = "runs/test"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(DetectionTrainingError, match="device"):
        load_detection_training_config(config_path)


def test_make_detection_target_converts_normalized_boxes_to_pixels(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]

    detection_target = make_detection_target(
        image,
        target,
        image_id=3,
        num_classes=2,
    )

    assert detection_target["boxes"].tolist() == [[3.0, 1.5, 5.0, 4.5]]
    assert detection_target["labels"].tolist() == [1]
    assert detection_target["image_id"].tolist() == [3]
    assert detection_target["iscrowd"].tolist() == [0]


def test_make_detection_target_clips_tiny_boundary_crossing(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(
        target,
        boxes=torch.tensor([[0.25, -5e-7, 0.75, 0.5]], dtype=torch.float32),
    )

    detection_target = make_detection_target(
        image,
        target,
        image_id=1,
        num_classes=2,
    )

    assert detection_target["boxes"].tolist() == [[2.0, 0.0, 6.0, 3.0]]


def test_make_detection_target_rejects_gross_boundary_crossing(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(
        target,
        boxes=torch.tensor([[-1e-3, 0.25, 0.5, 0.75]], dtype=torch.float32),
    )

    with pytest.raises(DetectionTrainingError, match="out of range"):
        make_detection_target(image, target, image_id=1, num_classes=2)


def test_make_detection_target_rejects_box_degenerate_after_clipping(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(
        target,
        boxes=torch.tensor([[-5e-7, 0.25, 0.0, 0.75]], dtype=torch.float32),
    )

    with pytest.raises(DetectionTrainingError, match="invalid extents"):
        make_detection_target(image, target, image_id=1, num_classes=2)


def test_make_detection_target_rejects_non_finite_box(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(
        target,
        boxes=torch.tensor([[0.25, 0.25, float("nan"), 0.75]], dtype=torch.float32),
    )

    with pytest.raises(DetectionTrainingError, match="non-finite"):
        make_detection_target(image, target, image_id=1, num_classes=2)


def test_make_detection_target_excludes_source_class_zero_from_foreground(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.csv"
    image_path = dataset_root / "data" / "images" / "patient0001" / "train.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image_path)
    annotation = {
        "class_id": 0,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.25,
        "height": 0.5,
        "tooth_id": 7,
        "source_label_line": 1,
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "train-sample",
                "patient_id": "patient0001",
                "split": "train",
                "image_relative_path": "data/images/patient0001/train.jpg",
                "source_csv_line": "2",
                "annotation_count": "1",
                "annotations_json": json.dumps([annotation]),
            }
        )
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]

    assert target.labels.tolist() == [0]
    detection_target = make_detection_target(image, target, image_id=1, num_classes=2)

    assert detection_target["boxes"].shape == (0, 4)
    assert detection_target["labels"].shape == (0,)
    assert detection_target["area"].shape == (0,)
    assert detection_target["iscrowd"].shape == (0,)


def test_make_detection_target_retains_only_plaque_boxes(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(
        target,
        boxes=torch.tensor(
            [[0.0, 0.0, 0.25, 0.5], [0.375, 0.25, 0.625, 0.75]],
            dtype=torch.float32,
        ),
        labels=torch.tensor([0, 1], dtype=torch.int64),
    )

    detection_target = make_detection_target(image, target, image_id=1, num_classes=2)

    assert detection_target["boxes"].tolist() == [[3.0, 1.5, 5.0, 4.5]]
    assert detection_target["labels"].tolist() == [1]
    assert detection_target["area"].tolist() == [6.0]
    assert detection_target["iscrowd"].tolist() == [0]


def test_make_detection_target_rejects_unsupported_source_label(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]
    target = replace(target, labels=torch.tensor([2], dtype=torch.int64))

    with pytest.raises(DetectionTrainingError, match="must be 0 or 1"):
        make_detection_target(image, target, image_id=1, num_classes=2)


def test_faster_rcnn_training_accepts_no_plaque_target(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    manifest_path = tmp_path / "manifest.csv"
    image_path = dataset_root / "data" / "images" / "patient0001" / "train.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(20, 40, 60)).save(image_path)
    annotation = {
        "class_id": 0,
        "x_center": 0.5,
        "y_center": 0.5,
        "width": 0.25,
        "height": 0.5,
        "tooth_id": 7,
        "source_label_line": 1,
    }
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "train-sample",
                "patient_id": "patient0001",
                "split": "train",
                "image_relative_path": "data/images/patient0001/train.jpg",
                "source_csv_line": "2",
                "annotation_count": "1",
                "annotations_json": json.dumps([annotation]),
            }
        )
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, source_target = dataset[0]
    detection_target = make_detection_target(
        image,
        source_target,
        image_id=1,
        num_classes=2,
    )
    model = FasterRCNN(
        TinyBackbone(),
        num_classes=2,
        min_size=32,
        max_size=32,
        rpn_anchor_generator=AnchorGenerator(
            sizes=((8,),),
            aspect_ratios=((1.0,),),
        ),
        box_roi_pool=MultiScaleRoIAlign(
            featmap_names=["0"],
            output_size=2,
            sampling_ratio=2,
        ),
        rpn_pre_nms_top_n_train=16,
        rpn_post_nms_top_n_train=8,
        box_detections_per_img=8,
    )
    model.train()

    losses = model([image], [detection_target])

    assert losses
    assert all(torch.isfinite(loss).item() for loss in losses.values())


def test_make_detection_target_rejects_unsupported_class_count(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        split="train",
    )
    image, target = dataset[0]

    with pytest.raises(DetectionTrainingError, match="num_classes=2"):
        make_detection_target(image, target, image_id=1, num_classes=1)


def test_collate_detection_batch_keeps_variable_images_as_lists(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaquePart2Dataset

    dataset = OrthodonticPlaquePart2Dataset(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
    )
    images, targets = collate_detection_batch([dataset[0], dataset[1]])

    assert len(images) == 2
    assert len(targets) == 2
    assert images[0].shape == images[1].shape


def test_run_detection_training_writes_checkpoint_and_metrics(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )

    result = run_detection_training(config, model_factory=lambda _: TinyDetectionModel())

    assert result.checkpoint_path.is_file()
    assert result.best_checkpoint_path.is_file()
    assert result.metrics_path.is_file()
    assert len(result.metrics) == 1
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["metrics"][0]["epoch"] == 1
    assert metrics_payload["best_epoch"] == 1
    checkpoint_payload = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint_payload["training_state"]["completed_epoch"] == 1
    assert isinstance(checkpoint_payload["optimizer_state_dict"], dict)
    assert not (config.output_dir / ".checkpoint_last.pt.tmp").exists()
    assert not (config.output_dir / ".checkpoint_best.pt.tmp").exists()
    assert not (config.output_dir / ".metrics.json.tmp").exists()


def test_run_detection_training_resumes_from_completed_epoch(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    output_dir = tmp_path / "runs"
    initial_config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    initial_result = run_detection_training(
        initial_config,
        model_factory=lambda _: TinyDetectionModel(),
    )
    resumed_config = replace(
        initial_config,
        epochs=2,
        resume_checkpoint_path=initial_result.checkpoint_path,
    )

    resumed_result = run_detection_training(
        resumed_config,
        model_factory=lambda _: TinyDetectionModel(),
    )

    assert [metric.epoch for metric in resumed_result.metrics] == [1, 2]
    checkpoint_payload = torch.load(
        resumed_result.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint_payload["training_state"]["completed_epoch"] == 2


def test_run_detection_training_rejects_incompatible_resume_config(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    output_dir = tmp_path / "runs"
    initial_config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    initial_result = run_detection_training(
        initial_config,
        model_factory=lambda _: TinyDetectionModel(),
    )
    incompatible_config = replace(
        initial_config,
        epochs=2,
        learning_rate=0.02,
        resume_checkpoint_path=initial_result.checkpoint_path,
    )

    with pytest.raises(DetectionTrainingError, match="learning_rate"):
        run_detection_training(
            incompatible_config,
            model_factory=lambda _: TinyDetectionModel(),
        )


def test_run_detection_training_rejects_fresh_overwrite(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    run_detection_training(config, model_factory=lambda _: TinyDetectionModel())

    with pytest.raises(DetectionTrainingError, match="overwrite existing artifacts"):
        run_detection_training(config, model_factory=lambda _: TinyDetectionModel())


def test_run_detection_training_keeps_best_validation_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        epochs=2,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    losses = iter((1.0, 0.5, 0.8, 0.6))
    monkeypatch.setattr(
        detection_training,
        "_run_loss_epoch",
        lambda *args, **kwargs: next(losses),
    )

    result = run_detection_training(config, model_factory=lambda _: TinyDetectionModel())

    last_payload = torch.load(result.checkpoint_path, weights_only=True)
    best_payload = torch.load(result.best_checkpoint_path, weights_only=True)
    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert last_payload["training_state"]["completed_epoch"] == 2
    assert best_payload["training_state"]["completed_epoch"] == 1
    assert metrics_payload["best_epoch"] == 1
    assert metrics_payload["best_validation_loss"] == 0.5


def test_run_detection_training_preserves_completed_epoch_before_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    output_dir = tmp_path / "runs"
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=output_dir,
        epochs=2,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )
    completed_losses = iter((1.0, 0.5))

    def fail_during_second_epoch(*args: object, **kwargs: object) -> float:
        try:
            return next(completed_losses)
        except StopIteration:
            raise RuntimeError("simulated interruption") from None

    monkeypatch.setattr(
        detection_training,
        "_run_loss_epoch",
        fail_during_second_epoch,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_detection_training(config, model_factory=lambda _: TinyDetectionModel())

    checkpoint_payload = torch.load(
        output_dir / "checkpoint_last.pt",
        weights_only=True,
    )
    metrics_payload = json.loads(
        (output_dir / "metrics.json").read_text(encoding="utf-8")
    )
    assert checkpoint_payload["training_state"]["completed_epoch"] == 1
    assert len(metrics_payload["metrics"]) == 1


def test_run_detection_training_wraps_model_construction_errors(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="default",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )

    def raise_model_error(_: DetectionTrainingConfig) -> TinyDetectionModel:
        raise DetectionModelError("missing pretrained weights")

    with pytest.raises(DetectionTrainingError, match="missing pretrained weights"):
        run_detection_training(config, model_factory=raise_model_error)


def test_run_detection_training_freezes_batch_norm_during_validation_loss(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    model = BatchNormTrackingDetectionModel()
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=tmp_path / "runs",
        epochs=2,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )

    run_detection_training(config, model_factory=lambda _: model)

    assert model.root_training_flags == [True, True, True, True]
    assert model.batch_norm_training_flags == [True, False, True, False]
    assert model.dropout_training_flags == [True, False, True, False]


def test_run_detection_training_rejects_file_output_path(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("blocked", encoding="utf-8")
    config = DetectionTrainingConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        output_dir=output_path,
        epochs=1,
        batch_size=1,
        learning_rate=0.01,
        momentum=0.9,
        weight_decay=0.0,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        seed=1,
        max_train_batches=1,
        max_validation_batches=1,
    )

    with pytest.raises(DetectionTrainingError, match="not a directory"):
        run_detection_training(config, model_factory=lambda _: TinyDetectionModel())
