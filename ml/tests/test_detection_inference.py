from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from orallens_ml.inference.detection import (
    DetectionInferenceConfig,
    DetectionInferenceError,
    load_detection_inference_config,
    run_detection_inference,
)
from orallens_ml.modeling.detection import save_detection_checkpoint


class TinyPredictModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        return [
            {
                "boxes": torch.tensor(
                    [[1.0, 2.0, 5.0, 6.0], [10.0, 10.0, 12.0, 12.0]],
                    dtype=torch.float32,
                ),
                "labels": torch.tensor([1, 1], dtype=torch.int64),
                "scores": torch.tensor([0.9, 0.2], dtype=torch.float32),
            }
            for _ in images
        ]


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 6), color=(20, 40, 60)).save(path)


def config_for(
    tmp_path: Path,
    *,
    checkpoint_path: Path,
    output_dir: Path | None = None,
    score_threshold: float = 0.5,
    max_detections: int = 25,
) -> DetectionInferenceConfig:
    return DetectionInferenceConfig(
        checkpoint_path=checkpoint_path,
        output_dir=tmp_path / "runs" if output_dir is None else output_dir,
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights="none",
        device="cpu",
        score_threshold=score_threshold,
        max_detections=max_detections,
    )


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


def save_test_checkpoint(model: torch.nn.Module, path: Path) -> None:
    save_detection_checkpoint(
        model=model,
        path=path,
        config=checkpoint_config(),
        metrics=[],
    )


def test_load_detection_inference_config_validates_values(tmp_path: Path) -> None:
    config_path = tmp_path / "predict.toml"
    config_path.write_text(
        """
[model]
checkpoint_path = "checkpoint.pt"
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 0
pretrained_weights = "none"
[inference]
device = "cpu"
score_threshold = 0.5
max_detections = 10
[output]
output_dir = "runs/predict"
""".strip(),
        encoding="utf-8",
    )

    config = load_detection_inference_config(config_path)

    assert config.score_threshold == 0.5
    assert config.max_detections == 10
    assert config.device == "cpu"


def test_load_detection_inference_config_rejects_invalid_threshold(tmp_path: Path) -> None:
    config_path = tmp_path / "predict.toml"
    config_path.write_text(
        """
[model]
checkpoint_path = "checkpoint.pt"
num_classes = 2
image_min_size = 64
image_max_size = 128
trainable_backbone_layers = 0
pretrained_weights = "none"
[inference]
device = "cpu"
score_threshold = 1.5
max_detections = 10
[output]
output_dir = "runs/predict"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(DetectionInferenceError, match="score_threshold"):
        load_detection_inference_config(config_path)


def test_run_detection_inference_writes_filtered_predictions(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(TinyPredictModel(), checkpoint_path)
    config = config_for(tmp_path, checkpoint_path=checkpoint_path)

    result = run_detection_inference(
        config,
        image_path=image_path,
        model_factory=lambda _: TinyPredictModel(),
    )

    assert result.output_path.is_file()
    assert len(result.predictions) == 1
    assert result.predictions[0].score == pytest.approx(0.9)
    payload = json.loads(result.output_path.read_text(encoding="utf-8"))
    assert payload["image_width"] == 8
    assert payload["prediction_count"] == 1
    assert len(payload["predictions"]) == 1


def test_run_detection_inference_applies_max_detections(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(TinyPredictModel(), checkpoint_path)
    config = config_for(
        tmp_path,
        checkpoint_path=checkpoint_path,
        score_threshold=0.0,
        max_detections=1,
    )

    result = run_detection_inference(
        config,
        image_path=image_path,
        model_factory=lambda _: TinyPredictModel(),
    )

    assert len(result.predictions) == 1
    assert result.predictions[0].score == pytest.approx(0.9)


def test_run_detection_inference_rejects_unsupported_image_extension(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "images" / "sample.gif"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"not an image")
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(TinyPredictModel(), checkpoint_path)
    config = config_for(tmp_path, checkpoint_path=checkpoint_path)

    with pytest.raises(DetectionInferenceError, match="Unsupported"):
        run_detection_inference(
            config,
            image_path=image_path,
            model_factory=lambda _: TinyPredictModel(),
        )


def test_run_detection_inference_rejects_non_finite_scores(tmp_path: Path) -> None:
    class BadScoreModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            return [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0]], dtype=torch.float32),
                    "labels": torch.tensor([1], dtype=torch.int64),
                    "scores": torch.tensor([float("inf")], dtype=torch.float32),
                }
                for _ in images
            ]

    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(BadScoreModel(), checkpoint_path)
    config = config_for(tmp_path, checkpoint_path=checkpoint_path)

    with pytest.raises(DetectionInferenceError, match="scores"):
        run_detection_inference(
            config,
            image_path=image_path,
            model_factory=lambda _: BadScoreModel(),
        )


def test_run_detection_inference_rejects_float_labels(tmp_path: Path) -> None:
    class BadLabelModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
            return [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 5.0, 6.0]], dtype=torch.float32),
                    "labels": torch.tensor([1.0], dtype=torch.float32),
                    "scores": torch.tensor([0.9], dtype=torch.float32),
                }
                for _ in images
            ]

    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(BadLabelModel(), checkpoint_path)
    config = config_for(tmp_path, checkpoint_path=checkpoint_path)

    with pytest.raises(DetectionInferenceError, match="labels"):
        run_detection_inference(
            config,
            image_path=image_path,
            model_factory=lambda _: BadLabelModel(),
        )


def test_run_detection_inference_rejects_file_output_path(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_test_checkpoint(TinyPredictModel(), checkpoint_path)
    output_path = tmp_path / "not-a-directory"
    output_path.write_text("blocked", encoding="utf-8")
    config = config_for(
        tmp_path,
        checkpoint_path=checkpoint_path,
        output_dir=output_path,
    )

    with pytest.raises(DetectionInferenceError, match="not a directory"):
        run_detection_inference(
            config,
            image_path=image_path,
            model_factory=lambda _: TinyPredictModel(),
        )


def test_run_detection_inference_rejects_incompatible_checkpoint_config(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "images" / "sample.jpg"
    write_image(image_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_detection_checkpoint(
        model=TinyPredictModel(),
        path=checkpoint_path,
        config=checkpoint_config(image_max_size=256),
        metrics=[],
    )
    config = config_for(tmp_path, checkpoint_path=checkpoint_path)

    with pytest.raises(DetectionInferenceError, match="incompatible"):
        run_detection_inference(
            config,
            image_path=image_path,
            model_factory=lambda _: TinyPredictModel(),
        )
