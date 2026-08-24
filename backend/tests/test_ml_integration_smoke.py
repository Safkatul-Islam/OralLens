from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.config import Settings
from app.main import create_app


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "ml"
    / "configs"
    / "orthodontic_plaque_detection_mvp_v4_originals_online_aug_predict.toml"
)
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ml"
    / "runs"
    / "detection"
    / "orthodontic_plaque_part2_mvp_v4_originals_online_aug"
    / "checkpoint_best.pt"
)
DEFAULT_ML_SOURCE_PATH = PROJECT_ROOT / "ml" / "src"


def _integration_enabled() -> bool:
    return os.getenv("ORALLENS_RUN_ML_INTEGRATION") == "1"


def _write_infrastructure_smoke_fixture(path: Path) -> None:
    """Create a deterministic non-clinical image for infrastructure smoke tests."""

    image = Image.new("RGB", (512, 512), color=(70, 90, 110))
    draw = ImageDraw.Draw(image)
    colors = ((70, 90, 110), (185, 165, 135))
    square_size = 64
    for y in range(0, image.height, square_size):
        for x in range(0, image.width, square_size):
            color = colors[((x // square_size) + (y // square_size)) % 2]
            draw.rectangle(
                (x, y, x + square_size - 1, y + square_size - 1),
                fill=color,
            )
    image.save(path, format="PNG")


@pytest.mark.skipif(
    not _integration_enabled(),
    reason="Set ORALLENS_RUN_ML_INTEGRATION=1 to run the real backend-to-ML smoke.",
)
def test_backend_ml_mode_returns_model_backed_detections(tmp_path: Path) -> None:
    missing = [
        path
        for path in (
            DEFAULT_CONFIG_PATH,
            DEFAULT_CHECKPOINT_PATH,
            DEFAULT_ML_SOURCE_PATH,
        )
        if not path.exists()
    ]
    if missing:
        pytest.skip(f"Missing ML integration artifact(s): {missing}")

    image_path = tmp_path / "infrastructure-smoke.png"
    _write_infrastructure_smoke_fixture(image_path)
    temp_dir = tmp_path / "ml-inputs"
    runtime_artifact_dir = tmp_path / "ml-runtime"
    settings = Settings(
        runtime_mode="production",
        inference_mode="ml",
        storage_path=tmp_path / "scans.json",
        ml_temp_dir=temp_dir,
        ml_runtime_artifact_dir=runtime_artifact_dir,
        ml_source_path=DEFAULT_ML_SOURCE_PATH,
        ml_detection_config_path=DEFAULT_CONFIG_PATH,
        cors_allowed_origins=("https://orallens.example",),
    )
    with TestClient(create_app(settings)) as client:
        ready_response = client.get("/health/ready")
        response = client.post(
            "/scans",
            files={
                "file": (
                    image_path.name,
                    image_path.read_bytes(),
                    "image/png",
                )
            },
        )

    assert ready_response.status_code == 200
    assert response.status_code == 201
    payload = response.json()
    prediction = payload["prediction"]
    assert payload["input_assessment"]["status"] == "supported"
    assert prediction is not None
    assert prediction["is_mock"] is False
    assert prediction["model_name"] == (
        "orthodontic-plaque-mvp-v4-originals-online-aug-epoch9"
    )
    assert prediction["prediction_count"] == len(prediction["detections"])
    assert all(detection["label"] == 1 for detection in prediction["detections"])
    assert all(detection["score"] >= 0.80 for detection in prediction["detections"])
    assert not any(temp_dir.iterdir())
    assert not any(runtime_artifact_dir.iterdir())
    assert not settings.storage_path.exists()
