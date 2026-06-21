from uuid import UUID

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

TRUSTED_ORIGIN = "https://portfolio.example"


def make_app(tmp_path):
    settings = Settings(
        storage_path=tmp_path / "scans.json",
        cors_allowed_origins=(TRUSTED_ORIGIN,),
    )
    return create_app(settings)


def test_cors_allows_only_configured_origin(tmp_path):
    client = TestClient(make_app(tmp_path))

    trusted_response = client.options(
        "/scans",
        headers={
            "Origin": TRUSTED_ORIGIN,
            "Access-Control-Request-Method": "POST",
        },
    )
    untrusted_response = client.options(
        "/scans",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert trusted_response.status_code == 200
    assert trusted_response.headers["access-control-allow-origin"] == TRUSTED_ORIGIN
    assert "access-control-allow-origin" not in untrusted_response.headers


def test_request_id_is_returned_to_client(tmp_path):
    client = TestClient(make_app(tmp_path))

    response = client.get("/health", headers={"X-Request-ID": "portfolio-request-1"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "portfolio-request-1"


def test_invalid_request_id_is_replaced(tmp_path):
    client = TestClient(make_app(tmp_path))

    response = client.get("/health", headers={"X-Request-ID": "x" * 129})

    generated_request_id = response.headers["X-Request-ID"]
    assert generated_request_id != "x" * 129
    assert str(UUID(generated_request_id)) == generated_request_id


def test_unhandled_error_response_does_not_leak_internal_details(tmp_path):
    app = make_app(tmp_path)

    def fail_to_list_scans():
        raise RuntimeError("secret database path: C:\\private\\scans.json")

    app.state.scan_service.list_scans = fail_to_list_scans
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/scans", headers={"X-Request-ID": "failure-request-1"})

    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "failure-request-1"
    assert response.json() == {
        "code": "internal_error",
        "detail": "An unexpected error occurred.",
        "request_id": "failure-request-1",
    }
    assert "secret database path" not in response.text


def test_validation_error_uses_safe_consistent_schema(tmp_path):
    client = TestClient(make_app(tmp_path))

    response = client.post("/scans")

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert response.json()["detail"] == "Request validation failed."
    assert response.json()["request_id"] == response.headers["X-Request-ID"]

