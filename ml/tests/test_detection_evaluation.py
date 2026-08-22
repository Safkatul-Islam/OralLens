from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

from PIL import Image
import pytest
import torch

import orallens_ml.evaluation.detection as detection_evaluation
from orallens_ml.data.orthodontic_plaque_dataset import OrthodonticPlaqueTarget
from orallens_ml.evaluation.detection import (
    DetectionEvaluationConfig,
    DetectionEvaluationError,
    analyze_detection_errors,
    evaluate_detection_average_precision,
    evaluate_detection_image,
    evaluate_detection_predictions,
    load_detection_evaluation_config,
    run_detection_evaluation,
)
from orallens_ml.modeling.detection import save_detection_checkpoint
from orallens_ml.training.detection import DetectionTrainingError

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


class TinyEvalModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor([[3.0, 1.5, 5.0, 4.5]], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.int64),
                "scores": torch.tensor([0.9], dtype=torch.float32),
            }
            for _ in images
        ]


def prediction(
    *,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float],
) -> dict[str, torch.Tensor]:
    box_tensor = (
        torch.empty((0, 4), dtype=torch.float32)
        if not boxes
        else torch.tensor(boxes, dtype=torch.float32)
    )
    return {
        "boxes": box_tensor,
        "labels": torch.tensor(labels, dtype=torch.int64),
        "scores": torch.tensor(scores, dtype=torch.float32),
    }


def target(*, boxes: list[list[float]], labels: list[int]) -> dict[str, torch.Tensor]:
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


def write_manifest_dataset(
    root: Path,
    *,
    annotations: list[dict[str, object]] | None = None,
    source_aware: bool = False,
) -> tuple[Path, Path]:
    dataset_root = root / "dataset"
    manifest_path = root / "manifest.csv"
    image_path = dataset_root / "data" / "images" / "patient0001" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image_path)
    if annotations is None:
        annotations = [
            {
                "class_id": 1,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.25,
                "height": 0.5,
                "tooth_id": 7,
                "source_label_line": 1,
            }
        ]
    row = {
        "manifest_schema_version": "1",
        "dataset_id": "dataset",
        "dataset_version": "4",
        "source_family_id": "source-family",
        "source_artifact_id": "part-2",
        "split_group_id": "patient0001",
        "derivative_group_id": "sample",
        "variant": "original",
        "sample_id": "sample",
        "patient_id": "patient0001",
        "split": "validation",
        "image_relative_path": "data/images/patient0001/sample.jpg",
        "source_csv_line": "2",
        "annotation_count": str(len(annotations)),
        "annotations_json": json.dumps(annotations),
    }
    fieldnames = SOURCE_AWARE_FIELDNAMES if source_aware else FIELDNAMES
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({field: row[field] for field in fieldnames})
    return dataset_root, manifest_path


def checkpoint_config(**overrides: object) -> dict[str, object]:
    config: dict[str, object] = {
        "num_classes": 2,
        "image_min_size": 64,
        "image_max_size": 128,
        "trainable_backbone_layers": 0,
        "pretrained_weights": "none",
    }
    config.update(overrides)
    return config


def test_evaluate_detection_predictions_counts_matches() -> None:
    metrics = evaluate_detection_predictions(
        [
            prediction(
                boxes=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
                labels=[1, 1],
                scores=[0.9, 0.8],
            )
        ],
        [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        num_classes=2,
    )

    metric = metrics[0]
    assert metric.true_positives == 1
    assert metric.false_positives == 1
    assert metric.false_negatives == 0
    assert metric.precision == 0.5
    assert metric.recall == 1.0
    assert metric.f1 == pytest.approx(2 / 3)
    assert metric.mean_matched_iou == 1.0


def test_evaluate_detection_predictions_handles_empty_predictions() -> None:
    metrics = evaluate_detection_predictions(
        [prediction(boxes=[], labels=[], scores=[])],
        [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        num_classes=2,
    )

    assert metrics[0].true_positives == 0
    assert metrics[0].false_positives == 0
    assert metrics[0].false_negatives == 1
    assert metrics[0].precision == 0.0
    assert metrics[0].recall == 0.0


def test_evaluate_detection_predictions_filters_by_score() -> None:
    metrics = evaluate_detection_predictions(
        [prediction(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1], scores=[0.4])],
        [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        num_classes=2,
    )

    assert metrics[0].prediction_count == 0
    assert metrics[0].false_negatives == 1


def test_evaluate_detection_image_preserves_match_evidence_and_applies_cap() -> None:
    result = evaluate_detection_image(
        prediction(
            boxes=[
                [0.0, 0.0, 10.0, 10.0],
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            labels=[1, 1, 1],
            scores=[0.8, 0.9, 0.7],
        ),
        target(
            boxes=[
                [0.0, 0.0, 10.0, 10.0],
                [20.0, 20.0, 30.0, 30.0],
            ],
            labels=[1, 1],
        ),
        iou_threshold=0.5,
        score_threshold=0.5,
        num_classes=2,
        max_detections=2,
    )

    assert result.eligible_prediction_count == 3
    assert result.prediction_count == 2
    assert result.truncated_prediction_count == 1
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.unmatched_target_indexes == (1,)
    assert [match.prediction_index for match in result.prediction_matches] == [1, 0]
    assert result.prediction_matches[0].matched_target_index == 0
    assert result.prediction_matches[1].is_true_positive is False
    assert result.prediction_matches[1].best_target_index == 0
    assert result.prediction_matches[1].best_iou == 1.0


def test_average_precision_uses_global_score_ranking() -> None:
    result = evaluate_detection_average_precision(
        [
            prediction(
                boxes=[[20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]],
                labels=[1, 1],
                scores=[0.9, 0.8],
            )
        ],
        [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
        iou_thresholds=(0.5,),
        num_classes=2,
    )

    assert result.ap50 == pytest.approx(0.5)
    assert result.map50_95 is None
    assert result.by_iou == ((0.5, pytest.approx(0.5)),)


def test_average_precision_reports_complete_coco_iou_range() -> None:
    thresholds = tuple(0.5 + 0.05 * index for index in range(10))

    result = evaluate_detection_average_precision(
        [
            prediction(
                boxes=[[0.0, 0.0, 10.0, 10.0]],
                labels=[1],
                scores=[0.9],
            )
        ],
        [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
        iou_thresholds=thresholds,
        num_classes=2,
    )

    assert result.ap50 == pytest.approx(1.0)
    assert result.map50_95 == pytest.approx(1.0)


def test_error_analysis_separates_dominant_failure_types() -> None:
    prediction_row = prediction(
        boxes=[
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 10.0, 10.0],
            [0.0, 0.0, 5.0, 5.0],
            [40.0, 40.0, 50.0, 50.0],
            [20.0, 20.0, 30.0, 30.0],
        ],
        labels=[1, 1, 1, 1, 1],
        scores=[0.9, 0.8, 0.7, 0.6, 0.4],
    )
    target_row = target(
        boxes=[[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]],
        labels=[1, 1],
    )
    source_target = OrthodonticPlaqueTarget(
        sample_id="sample",
        patient_id="patient",
        split="test",
        image_relative_path=PurePosixPath("images/sample.jpg"),
        variant="original",
        source_csv_line=2,
        boxes=torch.empty((0, 4), dtype=torch.float32),
        labels=torch.empty((0,), dtype=torch.int64),
        tooth_ids=torch.empty((0,), dtype=torch.int64),
        source_label_lines=torch.empty((0,), dtype=torch.int64),
    )

    result = analyze_detection_errors(
        [prediction_row],
        [target_row],
        [source_target],
        iou_threshold=0.5,
        score_threshold=0.5,
        num_classes=2,
        top_cases=5,
    )

    assert result.false_positives == 3
    assert result.false_negatives == 1
    assert result.duplicate_detections == 1
    assert result.localization_failures == 1
    assert result.background_false_positives == 1
    assert result.low_confidence_matches == 1
    assert result.unexplained_false_negatives == 0
    assert result.top_failure_cases[0]["sample_id"] == "sample"


def test_evaluate_detection_image_rejects_invalid_cap() -> None:
    with pytest.raises(DetectionEvaluationError, match="max_detections"):
        evaluate_detection_image(
            prediction(boxes=[], labels=[], scores=[]),
            target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1]),
            iou_threshold=0.5,
            score_threshold=0.5,
            num_classes=2,
            max_detections=0,
        )


def test_evaluate_detection_predictions_rejects_invalid_scores() -> None:
    with pytest.raises(DetectionEvaluationError, match="scores"):
        evaluate_detection_predictions(
            [prediction(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1], scores=[1.2])],
            [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
            iou_thresholds=(0.5,),
            score_thresholds=(0.5,),
            num_classes=2,
        )


def test_evaluate_detection_predictions_rejects_non_finite_scores() -> None:
    with pytest.raises(DetectionEvaluationError, match="scores"):
        evaluate_detection_predictions(
            [
                prediction(
                    boxes=[[0.0, 0.0, 10.0, 10.0]],
                    labels=[1],
                    scores=[float("nan")],
                )
            ],
            [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
            iou_thresholds=(0.5,),
            score_thresholds=(0.5,),
            num_classes=2,
        )


def test_evaluate_detection_predictions_rejects_float_labels() -> None:
    with pytest.raises(DetectionEvaluationError, match="labels"):
        evaluate_detection_predictions(
            [
                {
                    "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]], dtype=torch.float32),
                    "labels": torch.tensor([1.0], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                }
            ],
            [target(boxes=[[0.0, 0.0, 10.0, 10.0]], labels=[1])],
            iou_thresholds=(0.5,),
            score_thresholds=(0.5,),
            num_classes=2,
        )


def test_load_detection_evaluation_config_validates_thresholds(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
[data]
dataset_root = "dataset"
manifest_path = "manifest.csv"
[model]
checkpoint_path = "checkpoint.pt"
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 0
pretrained_weights = "none"
[evaluation]
split = "validation"
batch_size = 1
num_workers = 0
device = "cpu"
iou_thresholds = [0.5]
score_thresholds = [0.5]
variant = "original"
excluded_sample_id_suffixes = ["_blur", "_dark", "_light"]
average_precision_iou_thresholds = [0.5, 0.75]
analysis_score_threshold = 0.5
analysis_top_cases = 5
max_batches = 1
[output]
output_dir = "runs/eval"
""".strip(),
        encoding="utf-8",
    )

    config = load_detection_evaluation_config(config_path)

    assert config.iou_thresholds == (0.5,)
    assert config.score_thresholds == (0.5,)
    assert config.split == "validation"
    assert config.variant == "original"
    assert config.excluded_sample_id_suffixes == ("_blur", "_dark", "_light")
    assert config.average_precision_iou_thresholds == (0.5, 0.75)
    assert config.analysis_score_threshold == 0.5
    assert config.analysis_top_cases == 5


def test_load_detection_evaluation_config_rejects_invalid_split(tmp_path: Path) -> None:
    config_path = tmp_path / "eval.toml"
    config_path.write_text(
        """
[data]
dataset_root = "dataset"
manifest_path = "manifest.csv"
[model]
checkpoint_path = "checkpoint.pt"
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 0
pretrained_weights = "none"
[evaluation]
split = "holdout"
batch_size = 1
num_workers = 0
device = "cpu"
iou_thresholds = [0.5]
score_thresholds = [0.5]
[output]
output_dir = "runs/eval"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(DetectionEvaluationError, match="split"):
        load_detection_evaluation_config(config_path)


def test_run_detection_evaluation_writes_metrics(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(
        tmp_path,
        source_aware=True,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyEvalModel(),
        path=checkpoint_path,
        config=checkpoint_config(),
        metrics=[],
    )
    config = DetectionEvaluationConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "runs",
        split="validation",
        batch_size=1,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        variant="original",
        excluded_sample_id_suffixes=("_blur", "_dark", "_light"),
        average_precision_iou_thresholds=tuple(
            0.5 + 0.05 * index for index in range(10)
        ),
        analysis_score_threshold=0.5,
        max_batches=1,
    )

    result = run_detection_evaluation(config, model_factory=lambda _: TinyEvalModel())

    assert result.metrics_path.is_file()
    assert result.metrics[0].true_positives == 1
    assert result.dataset_summary.image_count == 1
    assert result.dataset_summary.patient_count == 1
    assert result.dataset_summary.variant == "original"
    assert result.dataset_summary.excluded_sample_id_suffixes == (
        "_blur",
        "_dark",
        "_light",
    )
    assert result.average_precision is not None
    assert result.average_precision.ap50 == pytest.approx(1.0)
    assert result.average_precision.map50_95 == pytest.approx(1.0)
    assert result.error_analysis is not None
    assert result.error_analysis.false_positives == 0
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["metrics"][0]["precision"] == 1.0
    assert payload["dataset"] == {
        "image_count": 1,
        "patient_count": 1,
        "target_count": 1,
        "variant": "original",
        "excluded_sample_id_suffixes": ["_blur", "_dark", "_light"],
    }
    assert payload["average_precision"]["map50_95"] == pytest.approx(1.0)


def test_run_detection_evaluation_scores_only_plaque_targets(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(
        tmp_path,
        annotations=[
            {
                "class_id": 0,
                "x_center": 0.25,
                "y_center": 0.5,
                "width": 0.25,
                "height": 0.5,
                "tooth_id": 6,
                "source_label_line": 1,
            },
            {
                "class_id": 1,
                "x_center": 0.5,
                "y_center": 0.5,
                "width": 0.25,
                "height": 0.5,
                "tooth_id": 7,
                "source_label_line": 2,
            },
        ],
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyEvalModel(),
        path=checkpoint_path,
        config=checkpoint_config(),
        metrics=[],
    )
    config = DetectionEvaluationConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "evaluation",
        split="validation",
        batch_size=1,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        max_batches=1,
    )

    result = run_detection_evaluation(config, model_factory=lambda _: TinyEvalModel())

    assert result.metrics[0].target_count == 1
    assert result.metrics[0].true_positives == 1
    assert result.metrics[0].false_negatives == 0


def test_run_detection_evaluation_translates_target_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyEvalModel(),
        path=checkpoint_path,
        config=checkpoint_config(),
        metrics=[],
    )
    config = DetectionEvaluationConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "runs",
        split="validation",
        batch_size=1,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        max_batches=1,
    )

    def reject_target(*args: object, **kwargs: object) -> dict[str, torch.Tensor]:
        raise DetectionTrainingError("invalid normalized target")

    monkeypatch.setattr(detection_evaluation, "make_detection_target", reject_target)

    with pytest.raises(
        DetectionEvaluationError,
        match="invalid normalized target",
    ) as error:
        run_detection_evaluation(config, model_factory=lambda _: TinyEvalModel())

    assert isinstance(error.value.__cause__, DetectionTrainingError)


def test_run_detection_evaluation_rejects_file_output_path(tmp_path: Path) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyEvalModel(),
        path=checkpoint_path,
        config=checkpoint_config(),
        metrics=[],
    )
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("blocked", encoding="utf-8")
    config = DetectionEvaluationConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_path,
        split="validation",
        batch_size=1,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        max_batches=1,
    )

    with pytest.raises(DetectionEvaluationError, match="not a directory"):
        run_detection_evaluation(config, model_factory=lambda _: TinyEvalModel())


def test_run_detection_evaluation_rejects_incompatible_checkpoint_config(
    tmp_path: Path,
) -> None:
    dataset_root, manifest_path = write_manifest_dataset(tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyEvalModel(),
        path=checkpoint_path,
        config=checkpoint_config(num_classes=3),
        metrics=[],
    )
    config = DetectionEvaluationConfig(
        dataset_root=dataset_root,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "runs",
        split="validation",
        batch_size=1,
        num_workers=0,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        iou_thresholds=(0.5,),
        score_thresholds=(0.5,),
        max_batches=1,
    )

    with pytest.raises(DetectionEvaluationError, match="incompatible"):
        run_detection_evaluation(config, model_factory=lambda _: TinyEvalModel())
