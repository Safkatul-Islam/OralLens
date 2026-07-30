from pathlib import Path

import pytest

from app.config import default_ml_detection_config_path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V2_PREDICTION_CONFIG_NAME = "orthodontic_plaque_detection_mvp_v2_predict.toml"


def test_default_ml_detection_config_selects_v2() -> None:
    assert default_ml_detection_config_path() == (
        PROJECT_ROOT / "ml" / "configs" / V2_PREDICTION_CONFIG_NAME
    )


@pytest.mark.parametrize(
    "script_name",
    ("run-ml-server.ps1", "run-ml-server.cmd"),
)
def test_ml_server_launcher_selects_v2_config(script_name: str) -> None:
    script_path = PROJECT_ROOT / "backend" / "scripts" / script_name

    contents = script_path.read_text(encoding="utf-8")

    assert V2_PREDICTION_CONFIG_NAME in contents
    assert "orthodontic_plaque_detection_mvp_predict.toml" not in contents
