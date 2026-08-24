from pathlib import Path
import tomllib

import pytest
from pydantic import ValidationError

from app.config import Settings, default_ml_detection_config_path


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


def test_ml_concurrency_defaults_to_one() -> None:
    assert Settings().ml_max_concurrent_inferences == 1


def test_ml_concurrency_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        Settings(ml_max_concurrent_inferences=0)


def production_settings_values(tmp_path: Path) -> dict[str, object]:
    config_path = tmp_path / "predict.toml"
    config_path.write_text("[model]\n", encoding="utf-8")
    return {
        "runtime_mode": "production",
        "inference_mode": "ml",
        "ml_detection_config_path": config_path,
        "ml_temp_dir": tmp_path / "uploads",
        "ml_runtime_artifact_dir": tmp_path / "runtime-artifacts",
        "cors_allowed_origins": ("https://orallens.example",),
    }


def test_production_runtime_accepts_explicit_safe_settings(tmp_path: Path) -> None:
    settings = Settings(**production_settings_values(tmp_path))

    assert settings.runtime_mode == "production"
    assert settings.inference_mode == "ml"


def test_production_runtime_rejects_mock_inference(tmp_path: Path) -> None:
    values = production_settings_values(tmp_path)
    values["inference_mode"] = "mock"

    with pytest.raises(ValidationError, match="inference_mode='ml'"):
        Settings(**values)


def test_production_runtime_rejects_implicit_runtime_paths_and_origins() -> None:
    with pytest.raises(ValidationError, match="explicit settings"):
        Settings(runtime_mode="production", inference_mode="ml")


def test_production_runtime_rejects_missing_ml_config(tmp_path: Path) -> None:
    values = production_settings_values(tmp_path)
    values["ml_detection_config_path"] = tmp_path / "missing.toml"

    with pytest.raises(ValidationError, match="existing regular file"):
        Settings(**values)


@pytest.mark.parametrize(
    "origin",
    ("*", "https://user:password@orallens.example", "https://orallens.example/path"),
)
def test_production_runtime_rejects_unsafe_origins(
    tmp_path: Path,
    origin: str,
) -> None:
    values = production_settings_values(tmp_path)
    values["cors_allowed_origins"] = (origin,)

    with pytest.raises(ValidationError, match="explicit HTTP"):
        Settings(**values)


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
