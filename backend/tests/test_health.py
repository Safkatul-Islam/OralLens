from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline.inference import InferenceOutcome


def test_health_returns_service_status(tmp_path):
    settings = Settings(storage_path=tmp_path / "scans.json")
    client = TestClient(create_app(settings))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "OralLens AI Backend",
        "version": "0.1.0",
        "environment": "local",
    }


def test_liveness_does_not_depend_on_model_readiness(tmp_path):
    settings = Settings(
        inference_mode="ml",
        storage_path=tmp_path / "scans.json",
        ml_source_path=tmp_path / "missing-ml-source",
    )
    client = TestClient(create_app(settings))

    response = client.get("/health/live")

    assert response.status_code == 200
    assert response.json()["status"] == "live"


def test_readiness_is_unavailable_before_ml_initialization(tmp_path):
    settings = Settings(
        inference_mode="ml",
        storage_path=tmp_path / "scans.json",
        ml_source_path=tmp_path / "missing-ml-source",
    )
    client = TestClient(create_app(settings))

    response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_lifespan_initializes_pipeline_once_before_readiness(tmp_path):
    class LifecyclePipeline:
        def __init__(self) -> None:
            self.initialize_calls = 0
            self._is_ready = False

        @property
        def is_ready(self) -> bool:
            return self._is_ready

        def initialize(self) -> None:
            self.initialize_calls += 1
            self._is_ready = True

        def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
            raise AssertionError("Health checks must not run inference.")

    pipeline = LifecyclePipeline()
    settings = Settings(storage_path=tmp_path / "scans.json")

    with TestClient(
        create_app(settings, inference_pipeline=pipeline)
    ) as client:
        first = client.get("/health/ready")
        second = client.get("/health/ready")

    assert first.status_code == 200
    assert first.json()["status"] == "ready"
    assert second.status_code == 200
    assert pipeline.initialize_calls == 1
