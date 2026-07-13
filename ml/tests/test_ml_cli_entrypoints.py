from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from orallens_ml.cli import evaluate, predict, train
from orallens_ml.evaluation.detection import DetectionEvaluationError
from orallens_ml.inference.detection import DetectionInferenceError
from orallens_ml.training.detection import DetectionTrainingError


def test_train_detection_cli_forwards_config_and_emits_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "train.toml"
    config = object()
    calls: dict[str, object] = {}

    def load_config(path: Path) -> object:
        calls["config_path"] = path
        return config

    def run_training(received_config: object) -> SimpleNamespace:
        calls["config"] = received_config
        return SimpleNamespace(
            checkpoint_path=tmp_path / "runs" / "checkpoint.pt",
            metrics_path=tmp_path / "runs" / "metrics.json",
            metrics=[
                SimpleNamespace(
                    epoch=1,
                    learning_rate=0.01,
                    train_loss=0.25,
                    validation_loss=0.5,
                )
            ],
        )

    monkeypatch.setattr(train, "load_detection_training_config", load_config)
    monkeypatch.setattr(train, "run_detection_training", run_training)

    exit_code = train.main(
        ["detection-baseline", "--config", str(config_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert calls == {"config_path": config_path, "config": config}
    assert payload == {
        "checkpoint_path": str(tmp_path / "runs" / "checkpoint.pt"),
        "metrics": [
            {
                "epoch": 1,
                "learning_rate": 0.01,
                "train_loss": 0.25,
                "validation_loss": 0.5,
            }
        ],
        "metrics_path": str(tmp_path / "runs" / "metrics.json"),
        "status": "detection-baseline-trained",
    }
    assert captured.err == ""


def test_train_detection_cli_returns_safe_error_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_config(path: Path) -> object:
        raise DetectionTrainingError(f"cannot load {path.name}")

    monkeypatch.setattr(train, "load_detection_training_config", load_config)

    exit_code = train.main(
        ["detection-baseline", "--config", str(tmp_path / "missing.toml")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: cannot load missing.toml\n"
    assert "Traceback" not in captured.err


def test_predict_detection_cli_forwards_paths_and_emits_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "predict.toml"
    image_path = tmp_path / "images" / "sample.jpg"
    config = object()
    calls: dict[str, object] = {}

    def load_config(path: Path) -> object:
        calls["config_path"] = path
        return config

    def run_inference(received_config: object, *, image_path: Path) -> SimpleNamespace:
        calls["config"] = received_config
        calls["image_path"] = image_path
        return SimpleNamespace(
            image_height=6,
            image_path=image_path,
            image_width=8,
            output_path=tmp_path / "runs" / "sample.predictions.json",
            predictions=[
                SimpleNamespace(
                    box_xyxy=(1.0, 2.0, 5.0, 6.0),
                    label="plaque",
                    score=0.9,
                )
            ],
        )

    monkeypatch.setattr(predict, "load_detection_inference_config", load_config)
    monkeypatch.setattr(predict, "run_detection_inference", run_inference)

    exit_code = predict.main(
        ["detection", "--config", str(config_path), "--image", str(image_path)]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert calls == {
        "config_path": config_path,
        "config": config,
        "image_path": image_path,
    }
    assert payload == {
        "image_height": 6,
        "image_path": str(image_path),
        "image_width": 8,
        "output_path": str(tmp_path / "runs" / "sample.predictions.json"),
        "prediction_count": 1,
        "predictions": [
            {
                "box_xyxy": [1.0, 2.0, 5.0, 6.0],
                "label": "plaque",
                "score": 0.9,
            }
        ],
        "status": "detection-predicted",
    }
    assert captured.err == ""


def test_predict_detection_cli_returns_safe_error_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_config(path: Path) -> object:
        return object()

    def run_inference(received_config: object, *, image_path: Path) -> SimpleNamespace:
        raise DetectionInferenceError(f"unsupported image {image_path.name}")

    monkeypatch.setattr(predict, "load_detection_inference_config", load_config)
    monkeypatch.setattr(predict, "run_detection_inference", run_inference)

    exit_code = predict.main(
        [
            "detection",
            "--config",
            str(tmp_path / "predict.toml"),
            "--image",
            str(tmp_path / "sample.gif"),
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: unsupported image sample.gif\n"
    assert "Traceback" not in captured.err


def test_evaluate_detection_cli_forwards_config_and_emits_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "eval.toml"
    config = object()
    calls: dict[str, object] = {}

    def load_config(path: Path) -> object:
        calls["config_path"] = path
        return config

    def run_evaluation(received_config: object) -> SimpleNamespace:
        calls["config"] = received_config
        return SimpleNamespace(
            metrics_path=tmp_path / "runs" / "metrics.json",
            metrics=[
                SimpleNamespace(
                    f1=1.0,
                    false_negatives=0,
                    false_positives=0,
                    iou_threshold=0.5,
                    mean_matched_iou=0.75,
                    precision=1.0,
                    prediction_count=1,
                    recall=1.0,
                    score_threshold=0.5,
                    target_count=1,
                    true_positives=1,
                )
            ],
        )

    monkeypatch.setattr(evaluate, "load_detection_evaluation_config", load_config)
    monkeypatch.setattr(evaluate, "run_detection_evaluation", run_evaluation)

    exit_code = evaluate.main(["detection", "--config", str(config_path)])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert exit_code == 0
    assert calls == {"config_path": config_path, "config": config}
    assert payload == {
        "metrics": [
            {
                "f1": 1.0,
                "false_negatives": 0,
                "false_positives": 0,
                "iou_threshold": 0.5,
                "mean_matched_iou": 0.75,
                "precision": 1.0,
                "prediction_count": 1,
                "recall": 1.0,
                "score_threshold": 0.5,
                "target_count": 1,
                "true_positives": 1,
            }
        ],
        "metrics_path": str(tmp_path / "runs" / "metrics.json"),
        "status": "detection-evaluated",
    }
    assert captured.err == ""


def test_evaluate_detection_cli_returns_safe_error_without_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def load_config(path: Path) -> object:
        return object()

    def run_evaluation(received_config: object) -> SimpleNamespace:
        raise DetectionEvaluationError("invalid evaluation config")

    monkeypatch.setattr(evaluate, "load_detection_evaluation_config", load_config)
    monkeypatch.setattr(evaluate, "run_detection_evaluation", run_evaluation)

    exit_code = evaluate.main(
        ["detection", "--config", str(tmp_path / "eval.toml")]
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: invalid evaluation config\n"
    assert "Traceback" not in captured.err
