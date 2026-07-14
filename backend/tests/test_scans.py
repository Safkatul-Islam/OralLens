from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


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
