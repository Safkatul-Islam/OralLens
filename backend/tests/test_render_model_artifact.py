from __future__ import annotations

import hashlib
import importlib.util
import io
from pathlib import Path
import sys
from urllib.request import Request

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "infra" / "render" / "fetch_model.py"
SPEC = importlib.util.spec_from_file_location("orallens_render_fetch_model", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fetch_model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fetch_model
SPEC.loader.exec_module(fetch_model)


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, *, declared_size: int | None = None) -> None:
        super().__init__(payload)
        self.headers: dict[str, str] = {}
        if declared_size is not None:
            self.headers["Content-Length"] = str(declared_size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeOpener:
    def __init__(self, payload: bytes, *, declared_size: int | None = None) -> None:
        self.payload = payload
        self.declared_size = declared_size
        self.request: Request | None = None

    def open(self, request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == 120.0
        self.request = request
        return FakeResponse(self.payload, declared_size=self.declared_size)


def test_download_checkpoint_atomically_publishes_verified_bytes(tmp_path: Path) -> None:
    payload = b"frozen checkpoint bytes"
    target = tmp_path / "checkpoint_best.pt"
    opener = FakeOpener(payload, declared_size=len(payload))

    fetch_model.download_checkpoint(
        url="https://huggingface.co/example/orallens/resolve/"
        + ("a" * 40)
        + "/checkpoint_best.pt",
        token="secret-token",
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_size=len(payload),
        max_bytes=len(payload),
        target=target,
        opener=opener,
    )

    assert target.read_bytes() == payload
    assert not (tmp_path / ".checkpoint_best.pt.downloading").exists()
    assert opener.request is not None
    assert opener.request.get_header("Authorization") == "Bearer secret-token"


def test_download_checkpoint_rejects_hash_mismatch_and_cleans_partial(
    tmp_path: Path,
) -> None:
    payload = b"unexpected"
    target = tmp_path / "checkpoint_best.pt"

    with pytest.raises(fetch_model.ModelArtifactError, match="SHA-256"):
        fetch_model.download_checkpoint(
            url="https://huggingface.co/example/orallens/resolve/"
            + ("a" * 40)
            + "/checkpoint_best.pt",
            token="secret-token",
            expected_sha256=hashlib.sha256(b"expected").hexdigest(),
            expected_size=len(payload),
            max_bytes=len(payload),
            target=target,
            opener=FakeOpener(payload),
        )

    assert not target.exists()
    assert not (tmp_path / ".checkpoint_best.pt.downloading").exists()


def test_download_checkpoint_rejects_oversized_stream(tmp_path: Path) -> None:
    payload = b"12345"
    target = tmp_path / "checkpoint_best.pt"

    with pytest.raises(fetch_model.ModelArtifactError, match="byte limit"):
        fetch_model.download_checkpoint(
            url="https://huggingface.co/example/orallens/resolve/"
            + ("a" * 40)
            + "/checkpoint_best.pt",
            token="secret-token",
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            expected_size=4,
            max_bytes=4,
            target=target,
            opener=FakeOpener(payload),
        )

    assert not target.exists()


@pytest.mark.parametrize(
    "redirect_url",
    [
        "http://huggingface.co/unsafe",
        "https://example.com/unsafe",
        "https://user:password@huggingface.co/unsafe",
    ],
)
def test_redirect_handler_rejects_unsafe_targets(redirect_url: str) -> None:
    handler = fetch_model._SafeHuggingFaceRedirectHandler()
    request = Request("https://huggingface.co/example/orallens")

    with pytest.raises(fetch_model.ModelArtifactError, match="not allowed"):
        handler.redirect_request(
            request,
            io.BytesIO(),
            302,
            "Found",
            {},
            redirect_url,
        )


def test_redirect_handler_strips_token_for_allowed_cdn_host() -> None:
    handler = fetch_model._SafeHuggingFaceRedirectHandler()
    request = Request(
        "https://huggingface.co/example/orallens",
        headers={"Authorization": "Bearer secret-token"},
    )

    redirected = handler.redirect_request(
        request,
        io.BytesIO(),
        302,
        "Found",
        {},
        "https://cdn-lfs.hf.co/signed-artifact",
    )

    assert redirected is not None
    assert redirected.get_header("Authorization") is None


def test_artifact_settings_require_credentials_and_frozen_metadata() -> None:
    with pytest.raises(fetch_model.ModelArtifactError, match="ORALLENS_HF_TOKEN"):
        fetch_model.artifact_settings_from_environment(
            {
                "ORALLENS_HF_REPOSITORY": "example/orallens",
                "ORALLENS_HF_REVISION": "a" * 40,
                "ORALLENS_HF_FILENAME": "checkpoint_best.pt",
                "ORALLENS_MODEL_ARTIFACT_SHA256": "b" * 64,
                "ORALLENS_MODEL_ARTIFACT_SIZE_BYTES": "330069425",
                "ORALLENS_MODEL_ARTIFACT_MAX_BYTES": "350000000",
            }
        )


@pytest.mark.parametrize(
    "filename",
    ["../checkpoint_best.pt", "folder\\checkpoint_best.pt"],
)
def test_artifact_settings_reject_path_like_filename(filename: str) -> None:
    with pytest.raises(fetch_model.ModelArtifactError, match="filename"):
        fetch_model.artifact_settings_from_environment(
            {
                "ORALLENS_HF_REPOSITORY": "example/orallens",
                "ORALLENS_HF_REVISION": "a" * 40,
                "ORALLENS_HF_FILENAME": filename,
                "ORALLENS_HF_TOKEN": "secret-token",
                "ORALLENS_MODEL_ARTIFACT_SHA256": "b" * 64,
                "ORALLENS_MODEL_ARTIFACT_SIZE_BYTES": "330069425",
                "ORALLENS_MODEL_ARTIFACT_MAX_BYTES": "350000000",
            }
        )
