from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


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

