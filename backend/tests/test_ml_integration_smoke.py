from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMAGE_PATH = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "raw"
    / "orthodontic_plaque"
    / "v3"
    / "extracted"
    / "part-2"
    / "mendeley-dataset-materials_Part_2"
    / "data"
    / "images"
    / "patient0144"
    / "patient0144_20260118_bottom-left.jpg"
)
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "ml"
    / "configs"
    / "orthodontic_plaque_detection_mvp_v2_predict.toml"
)
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "runs"
    / "detection"
    / "orthodontic_plaque_part2_mvp_v2"
    / "checkpoint_last.pt"
)
DEFAULT_ML_SOURCE_PATH = PROJECT_ROOT / "ml" / "src"


def _integration_enabled() -> bool:
    return os.getenv("ORALLENS_RUN_ML_INTEGRATION") == "1"


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set ORALLENS_RUN_ML_INTEGRATION=1 to run the real backend-to-ML smoke.",
)
def test_backend_ml_mode_returns_model_backed_detections(tmp_path: Path) -> None:
    missing = [
        path
        for path in (
            DEFAULT_IMAGE_PATH,
            DEFAULT_CONFIG_PATH,
            DEFAULT_CHECKPOINT_PATH,
            DEFAULT_ML_SOURCE_PATH,
        )
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"Missing ML integration artifact(s): {missing}")

    temp_dir = tmp_path / "ml-inputs"
    settings = Settings(
        inference_mode="ml",
        storage_path=tmp_path / "scans.json",
        ml_temp_dir=temp_dir,
        ml_source_path=DEFAULT_ML_SOURCE_PATH,
        ml_detection_config_path=DEFAULT_CONFIG_PATH,
    )
    client = TestClient(create_app(settings))

    response = client.post(
        "/scans",
        files={
            "file": (
                DEFAULT_IMAGE_PATH.name,
                DEFAULT_IMAGE_PATH.read_bytes(),
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 201
    payload = response.json()
    prediction = payload["prediction"]
    assert prediction["is_mock"] is False
    assert prediction["model_name"] == "orthodontic-plaque-mvp"
    assert prediction["prediction_count"] > 0
    assert prediction["prediction_count"] == len(prediction["detections"])
    assert not any(temp_dir.iterdir())
