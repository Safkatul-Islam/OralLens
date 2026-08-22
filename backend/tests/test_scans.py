from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.pipeline.inference import (
    InferenceAssessment,
    InferenceOutcome,
    InvalidInferenceInputError,
)
from app.schemas import ScanRecord


def png_bytes(extra_bytes: int = 32) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + (b"0" * extra_bytes)


def webp_bytes(extra_bytes: int = 32) -> bytes:
    return b"RIFF" + (extra_bytes + 4).to_bytes(4, "little") + b"WEBP" + (b"0" * extra_bytes)


def make_client(tmp_path, max_upload_bytes: int = 5 * 1024 * 1024) -> TestClient:
    settings = Settings(
        max_upload_bytes=max_upload_bytes,
        storage_path=tmp_path / "scans.json",
    )
    return TestClient(create_app(settings))


def make_ml_client(tmp_path) -> TestClient:
    settings = Settings(
        inference_mode="ml",
        storage_path=tmp_path / "scans.json",
        ml_temp_dir=tmp_path / "ml-inputs",
    )
    return TestClient(create_app(settings))


def test_create_scan_accepts_valid_png_upload(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/scans",
        files={"file": ("mouth.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["id"]
    assert payload["original_filename"] == "mouth.png"
    assert payload["content_type"] == "image/png"
    assert payload["size_bytes"] == len(png_bytes())
    assert len(payload["sha256"]) == 64
    assert payload["prediction"]["is_mock"] is True
    assert payload["prediction"]["model_name"] == "deterministic-mock-v1"
    assert payload["prediction"]["prediction_count"] == 0
    assert payload["prediction"]["detections"] == []
    assert payload["input_assessment"]["status"] == "not_assessed"
    assert "mock score of 0." in payload["report"]["summary"]
    assert "% confidence" not in payload["report"]["summary"]
    assert any(
        "not calibrated clinical probabilities" in limitation
        for limitation in payload["report"]["limitations"]
    )
    assert any(
        "displayed score's limitations" in next_step
        for next_step in payload["report"]["recommended_next_steps"]
    )
    assert payload["report"]["disclaimer"]


def test_create_scan_allows_configured_loopback_origin(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/scans",
        headers={"Origin": "http://127.0.0.1:5173"},
        files={"file": ("mouth.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_scans_preflight_allows_configured_loopback_origin(tmp_path):
    client = make_client(tmp_path)

    response = client.options(
        "/scans",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"
    assert "POST" in response.headers["access-control-allow-methods"]


def test_create_scan_does_not_allow_unconfigured_origin(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/scans",
        headers={"Origin": "https://untrusted.example"},
        files={"file": ("mouth.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    assert "access-control-allow-origin" not in response.headers


def test_create_scan_rejects_unsupported_content_type(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/scans",
        files={"file": ("notes.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["detail"]


def test_create_scan_rejects_oversized_upload(tmp_path):
    client = make_client(tmp_path, max_upload_bytes=16)

    response = client.post(
        "/scans",
        files={"file": ("mouth.png", png_bytes(extra_bytes=64), "image/png")},
    )

    assert response.status_code == 413
    assert "larger than the configured limit" in response.json()["detail"]


def test_create_scan_rejects_mismatched_file_signature(tmp_path):
    client = make_client(tmp_path)

    response = client.post(
        "/scans",
        files={"file": ("mouth.png", b"fake png content", "image/png")},
    )

    assert response.status_code == 400
    assert "does not match" in response.json()["detail"]


def test_create_scan_in_ml_mode_rejects_webp_until_ml_supports_it(tmp_path):
    client = make_ml_client(tmp_path)

    response = client.post(
        "/scans",
        files={"file": ("mouth.webp", webp_bytes(), "image/webp")},
    )

    assert response.status_code == 415
    assert "JPEG and PNG" in response.json()["detail"]


def test_scan_history_and_detail_are_persisted(tmp_path):
    client = make_client(tmp_path)
    first = client.post(
        "/scans",
        files={"file": ("first.png", png_bytes(32), "image/png")},
    ).json()
    second = client.post(
        "/scans",
        files={"file": ("second.png", png_bytes(33), "image/png")},
    ).json()

    list_response = client.get("/scans")
    detail_response = client.get(f"/scans/{second['id']}")

    assert list_response.status_code == 200
    assert [scan["id"] for scan in list_response.json()["scans"]] == [
        first["id"],
        second["id"],
    ]
    assert detail_response.status_code == 200
    assert detail_response.json()["id"] == second["id"]


def test_get_scan_returns_404_for_unknown_id(tmp_path):
    client = make_client(tmp_path)

    response = client.get("/scans/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Scan not found."


def test_create_scan_persists_structured_abstention_without_prediction(tmp_path):
    class AbstainingPipeline:
        def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
            return InferenceOutcome(
                assessment=InferenceAssessment(
                    status="unsupported",
                    reason_codes=("image_low_contrast",),
                    image_width=640,
                    image_height=480,
                    mean_luminance=0.5,
                    luminance_stddev=0.01,
                ),
                prediction=None,
            )

    settings = Settings(storage_path=tmp_path / "scans.json")
    app = create_app(settings)
    app.state.scan_service._inference_pipeline = AbstainingPipeline()
    client = TestClient(app)

    response = client.post(
        "/scans",
        files={"file": ("mouth.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["prediction"] is None
    assert payload["input_assessment"]["status"] == "unsupported"
    assert payload["input_assessment"]["reason_codes"] == ["image_low_contrast"]
    assert "did not run" in payload["report"]["summary"]
    assert "not a diagnosis" in payload["report"]["limitations"][2]


def test_historical_scan_without_assessment_remains_readable():
    payload = {
        "id": "historical",
        "created_at": "2026-08-01T00:00:00Z",
        "original_filename": "mouth.png",
        "content_type": "image/png",
        "size_bytes": 8,
        "sha256": "0" * 64,
        "prediction": {
            "label": "no_detection",
            "display_name": "No detection",
            "confidence": 0.0,
            "severity": "low",
            "is_mock": False,
            "model_name": "historical-model",
            "prediction_count": 0,
            "detections": [],
        },
        "evidence": {"kind": "summary", "summary": "Historical result."},
        "report": {
            "title": "Historical report",
            "summary": "Historical report.",
            "limitations": [],
            "recommended_next_steps": [],
            "disclaimer": "Historical record.",
        },
    }

    record = ScanRecord.model_validate(payload)

    assert record.input_assessment.status == "not_assessed"
    assert record.prediction is not None


def test_create_scan_maps_undecodable_ml_input_to_safe_bad_request(tmp_path):
    class InvalidImagePipeline:
        def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
            raise InvalidInferenceInputError("Input image could not be decoded safely.")

    settings = Settings(storage_path=tmp_path / "scans.json")
    app = create_app(settings)
    app.state.scan_service._inference_pipeline = InvalidImagePipeline()
    client = TestClient(app)

    response = client.post(
        "/scans",
        files={"file": ("mouth.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Input image could not be decoded safely."
    assert "Traceback" not in response.text
