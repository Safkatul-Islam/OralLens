from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from orallens_ml.evaluation.trustworthiness import (
    DetectionTrustworthinessError,
    load_detection_trustworthiness_config,
    run_detection_trustworthiness,
)
from orallens_ml.modeling.detection import save_detection_checkpoint

FIELDNAMES = (
    "sample_id",
    "patient_id",
    "split",
    "image_relative_path",
    "source_csv_line",
    "annotation_count",
    "annotations_json",
)


class TinyTrustModel(torch.nn.Module):
    def __init__(self, *, empty: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.empty = empty

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        if self.empty:
            return [
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32),
                    "labels": torch.empty((0,), dtype=torch.int64),
                    "scores": torch.empty((0,), dtype=torch.float32),
                }
                for _ in images
            ]
        return [
            {
                "boxes": torch.tensor(
                    [
                        [3.0, 1.5, 5.0, 4.5],
                        [0.0, 0.0, 1.0, 1.0],
                        [6.0, 4.5, 7.5, 5.5],
                    ],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 1, 1], dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32),
            }
            for _ in images
        ]


def _write_fixture(
    root: Path,
    *,
    max_batches: int | None = None,
    inference_image_max_size: int = 128,
    output_matches_historical: bool = False,
) -> Path:
    dataset_root = root / "dataset"
    manifest_path = root / "manifest.csv"
    image_path = dataset_root / "data" / "images" / "patient0001" / "sample.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(image_path)
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
        writer.writerow(
            {
                "sample_id": "sample",
                "patient_id": "patient0001",
                "split": "validation",
                "image_relative_path": "data/images/patient0001/sample.jpg",
                "source_csv_line": "2",
                "annotation_count": "1",
                "annotations_json": json.dumps([annotation]),
            }
        )

    checkpoint_path = root / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyTrustModel(),
        path=checkpoint_path,
        config={
            "num_classes": 2,
            "image_min_size": 64,
            "image_max_size": 128,
            "trainable_backbone_layers": 0,
            "pretrained_weights": "none",
        },
        metrics=[],
    )
    historical_output = root / "historical-evaluation"
    evaluation_config_path = root / "evaluation.toml"
    max_batches_line = (
        "" if max_batches is None else f"max_batches = {max_batches}\n"
    )
    evaluation_config_path.write_text(
        f"""
[data]
dataset_root = "{dataset_root.as_posix()}"
manifest_path = "{manifest_path.as_posix()}"
[model]
checkpoint_path = "{checkpoint_path.as_posix()}"
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
{max_batches_line}[output]
output_dir = "{historical_output.as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    inference_config_path = root / "inference.toml"
    inference_config_path.write_text(
        f"""
[model]
model_name = "orthodontic-plaque-test-v1"
checkpoint_path = "{checkpoint_path.as_posix()}"
num_classes = 2
image_min_size = 64
image_max_size = {inference_image_max_size}
trainable_backbone_layers = 0
pretrained_weights = "none"
[inference]
device = "cpu"
score_threshold = 0.5
max_detections = 2
[output]
output_dir = "{(root / 'predictions').as_posix()}"
""".strip(),
        encoding="utf-8",
    )

    report_output = historical_output if output_matches_historical else root / "trust-report"
    trust_config_path = root / "trust.toml"
    trust_config_path.write_text(
        f"""
[evaluation]
config_path = "{evaluation_config_path.as_posix()}"
[inference]
config_path = "{inference_config_path.as_posix()}"
[reporting]
reliability_bins = 10
top_failure_cases = 5
[output]
output_dir = "{report_output.as_posix()}"
""".strip(),
        encoding="utf-8",
    )
    return trust_config_path


def test_load_detection_trustworthiness_config_validates_values(
    tmp_path: Path,
) -> None:
    config_path = _write_fixture(tmp_path)

    config = load_detection_trustworthiness_config(config_path)

    assert config.reliability_bins == 10
    assert config.top_failure_cases == 5
    assert config.output_dir == tmp_path / "trust-report"


def test_load_detection_trustworthiness_config_rejects_invalid_bins(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "trust.toml"
    config_path.write_text(
        """
[evaluation]
config_path = "evaluation.toml"
[inference]
config_path = "inference.toml"
[reporting]
reliability_bins = 1
top_failure_cases = 5
[output]
output_dir = "runs/trust"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(DetectionTrustworthinessError, match="reliability_bins"):
        load_detection_trustworthiness_config(config_path)


def test_run_detection_trustworthiness_writes_traceable_cap_comparison(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(_write_fixture(tmp_path))

    result = run_detection_trustworthiness(
        config,
        model_factory=lambda _: TinyTrustModel(),
    )

    assert result.report_path.is_file()
    assert result.split == "validation"
    assert result.deployed_metrics["true_positives"] == 1
    assert result.deployed_metrics["false_positives"] == 1
    assert result.deployed_metrics["prediction_count"] == 2
    assert result.uncapped_metrics["false_positives"] == 2
    assert result.uncapped_metrics["prediction_count"] == 3
    assert result.cap_affected_image_count == 1
    assert result.score_to_match_ece is not None

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["scope"]["score_threshold"] == 0.5
    assert payload["scope"]["max_detections"] == 2
    assert payload["dataset"] == {
        "image_count": 1,
        "patient_count": 1,
        "target_count": 1,
    }
    assert payload["cap_impact"] == {
        "affected_image_count": 1,
        "metrics_changed": True,
        "truncated_prediction_count": 1,
    }
    reliability = payload["score_to_match_reliability"]
    assert reliability["observation_count"] == 2
    assert reliability["expected_calibration_error"] == pytest.approx(0.35)
    assert reliability["brier_score"] == pytest.approx(0.325)
    assert len(reliability["bins"]) == 10
    assert reliability["bins"][8]["prediction_count"] == 2
    assert payload["patients"][0]["patient_id"] == "patient0001"
    assert len(payload["images"]) == 1
    assert len(payload["images"][0]["predictions"]) == 2
    assert (
        payload["failure_summary"]["highest_scoring_false_positives"][0][
            "sample_id"
        ]
        == "sample"
    )
    for identity in (
        "trustworthiness_config",
        "evaluation_config",
        "inference_config",
        "manifest",
        "checkpoint",
    ):
        assert len(payload["provenance"][identity]["sha256"]) == 64
    assert list(result.report_path.parent.iterdir()) == [result.report_path]


def test_run_detection_trustworthiness_handles_empty_predictions(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(_write_fixture(tmp_path))

    result = run_detection_trustworthiness(
        config,
        model_factory=lambda _: TinyTrustModel(empty=True),
    )

    payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    assert result.deployed_metrics["prediction_count"] == 0
    assert result.deployed_metrics["false_negatives"] == 1
    assert result.score_to_match_ece is None
    assert payload["score_to_match_reliability"]["brier_score"] is None
    assert payload["failure_summary"]["highest_scoring_false_positives"] == []
    assert payload["failure_summary"]["images_with_most_false_negatives"] == [
        {
            "false_negatives": 1,
            "patient_id": "patient0001",
            "sample_id": "sample",
            "target_count": 1,
        }
    ]


def test_run_detection_trustworthiness_requires_complete_split(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(
        _write_fixture(tmp_path, max_batches=1)
    )

    with pytest.raises(DetectionTrustworthinessError, match="complete"):
        run_detection_trustworthiness(
            config,
            model_factory=lambda _: TinyTrustModel(),
        )


def test_run_detection_trustworthiness_rejects_incompatible_runtime_configs(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(
        _write_fixture(tmp_path, inference_image_max_size=256)
    )

    with pytest.raises(DetectionTrustworthinessError, match="incompatible"):
        run_detection_trustworthiness(
            config,
            model_factory=lambda _: TinyTrustModel(),
        )


def test_run_detection_trustworthiness_protects_historical_output(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(
        _write_fixture(tmp_path, output_matches_historical=True)
    )

    with pytest.raises(DetectionTrustworthinessError, match="historical"):
        run_detection_trustworthiness(
            config,
            model_factory=lambda _: TinyTrustModel(),
        )


def test_run_detection_trustworthiness_translates_dataset_read_error(
    tmp_path: Path,
) -> None:
    config = load_detection_trustworthiness_config(_write_fixture(tmp_path))
    image_path = (
        tmp_path / "dataset" / "data" / "images" / "patient0001" / "sample.jpg"
    )
    image_path.unlink()

    with pytest.raises(DetectionTrustworthinessError, match="Image is missing"):
        run_detection_trustworthiness(
            config,
            model_factory=lambda _: TinyTrustModel(),
        )
