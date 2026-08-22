from pathlib import Path
import tomllib

import pytest

from app.config import default_ml_detection_config_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINAL_PREDICTION_CONFIG_NAME = (
    "orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml"
)
FINAL_MODEL_NAME = "orthodontic-plaque-mvp-v4-originals-online-aug-epoch9"
FINAL_CHECKPOINT_PATH = (
    "ml/runs/detection/orthodontic_plaque_part2_mvp_v4_originals_online_aug/"
    "checkpoint_best.pt"
)
V3_PREDICTION_CONFIG_NAME = "orthodontic_plaque_detection_mvp_v3_predict.toml"
V2_PREDICTION_CONFIG_NAME = "orthodontic_plaque_detection_mvp_v2_predict.toml"


def test_default_ml_detection_config_selects_final_model() -> None:
    assert default_ml_detection_config_path() == (
        PROJECT_ROOT / "ml" / "configs" / FINAL_PREDICTION_CONFIG_NAME
    )


@pytest.mark.parametrize(
    "script_name",
    ("run-ml-server.ps1", "run-ml-server.cmd"),
)
def test_ml_server_launcher_selects_final_config(script_name: str) -> None:
    script_path = PROJECT_ROOT / "backend" / "scripts" / script_name

    contents = script_path.read_text(encoding="utf-8")

    assert FINAL_PREDICTION_CONFIG_NAME in contents
    assert V3_PREDICTION_CONFIG_NAME not in contents
    assert V2_PREDICTION_CONFIG_NAME not in contents
    assert "orthodontic_plaque_detection_mvp_predict.toml" not in contents


def test_final_inference_config_matches_frozen_model_contract() -> None:
    config_path = PROJECT_ROOT / "ml" / "configs" / FINAL_PREDICTION_CONFIG_NAME

    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    assert config["model"] == {
        "model_name": FINAL_MODEL_NAME,
        "checkpoint_path": FINAL_CHECKPOINT_PATH,
        "num_classes": 2,
        "image_min_size": 512,
        "image_max_size": 768,
        "trainable_backbone_layers": 3,
        "pretrained_weights": "none",
    }
    assert config["inference"]["score_threshold"] == 0.80
    assert config["inference"]["max_detections"] == 100
