"""Securely materialize the frozen checkpoint on an ephemeral Render instance."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import os
from pathlib import Path
import re
import secrets
import sys
from typing import BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

_TARGET_PATH = Path(
    "/opt/orallens/ml/runs/detection/"
    "orthodontic_plaque_part2_mvp_v4_originals_online_aug/checkpoint_best.pt"
)
_DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_REPOSITORY_PATTERN = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?/"
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,94}[A-Za-z0-9])?$"
)
_REVISION_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class ModelArtifactError(RuntimeError):
    """Raised when the frozen artifact cannot be obtained safely."""


class _Response(Protocol):
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def __enter__(self) -> "_Response": ...

    def __exit__(self, *args: object) -> None: ...


class _Opener(Protocol):
    def open(self, request: Request, *, timeout: float) -> _Response: ...


def _is_allowed_artifact_host(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.rstrip(".").casefold()
    return (
        normalized == "huggingface.co"
        or normalized.endswith(".huggingface.co")
        or normalized.endswith(".hf.co")
    )


class _SafeHuggingFaceRedirectHandler(HTTPRedirectHandler):
    """Permit only HTTPS redirects within Hugging Face-controlled hosts."""

    def redirect_request(
        self,
        request: Request,
        response: BinaryIO,
        code: int,
        message: str,
        headers: Mapping[str, str],
        new_url: str,
    ) -> Request | None:
        parsed = urlsplit(new_url)
        if (
            parsed.scheme != "https"
            or parsed.username is not None
            or parsed.password is not None
            or parsed.port not in {None, 443}
            or not _is_allowed_artifact_host(parsed.hostname)
        ):
            raise ModelArtifactError("Model artifact redirect target is not allowed.")

        redirected = super().redirect_request(
            request,
            response,
            code,
            message,
            headers,
            new_url,
        )
        if redirected is not None:
            original_host = urlsplit(request.full_url).hostname
            if parsed.hostname != original_host:
                redirected.remove_header("Authorization")
        return redirected


def _required_environment(environ: Mapping[str, str], key: str) -> str:
    value = environ.get(key, "").strip()
    if not value:
        raise ModelArtifactError(f"Required model artifact setting is missing: {key}")
    return value


def artifact_settings_from_environment(
    environ: Mapping[str, str],
) -> tuple[str, str, str, int, int]:
    """Validate runtime settings and return URL, token, digest, size, and limit."""

    repository = _required_environment(environ, "ORALLENS_HF_REPOSITORY")
    revision = _required_environment(environ, "ORALLENS_HF_REVISION")
    filename = _required_environment(environ, "ORALLENS_HF_FILENAME")
    token = _required_environment(environ, "ORALLENS_HF_TOKEN")
    expected_sha256 = _required_environment(
        environ, "ORALLENS_MODEL_ARTIFACT_SHA256"
    ).casefold()
    raw_size = _required_environment(
        environ, "ORALLENS_MODEL_ARTIFACT_SIZE_BYTES"
    )
    raw_limit = _required_environment(
        environ, "ORALLENS_MODEL_ARTIFACT_MAX_BYTES"
    )

    if not _REPOSITORY_PATTERN.fullmatch(repository):
        raise ModelArtifactError("Hugging Face repository identifier is invalid.")
    if not _REVISION_PATTERN.fullmatch(revision):
        raise ModelArtifactError("Hugging Face revision must be a full commit SHA.")
    if (
        filename in {"", ".", ".."}
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or filename != Path(filename).name
    ):
        raise ModelArtifactError("Hugging Face artifact filename is invalid.")
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ModelArtifactError("Model artifact SHA-256 is invalid.")
    try:
        expected_size = int(raw_size)
        max_bytes = int(raw_limit)
    except ValueError as exc:
        raise ModelArtifactError("Model artifact byte limits must be integers.") from exc
    if expected_size < 1 or max_bytes < expected_size:
        raise ModelArtifactError("Model artifact byte limits are inconsistent.")

    url = (
        f"https://huggingface.co/{repository}/resolve/"
        f"{revision}/{quote(filename, safe='')}"
    )
    return url, token, expected_sha256, expected_size, max_bytes


def _validated_existing_artifact(
    target: Path,
    *,
    expected_sha256: str,
    expected_size: int,
) -> bool:
    if target.is_symlink():
        raise ModelArtifactError("Model artifact target must not be a symlink.")
    if not target.exists():
        return False
    if not target.is_file() or target.stat().st_size != expected_size:
        raise ModelArtifactError("Existing model artifact does not match frozen metadata.")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        while chunk := handle.read(_DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    if not secrets.compare_digest(digest.hexdigest(), expected_sha256):
        raise ModelArtifactError("Existing model artifact failed SHA-256 verification.")
    return True


def download_checkpoint(
    *,
    url: str,
    token: str,
    expected_sha256: str,
    expected_size: int,
    max_bytes: int,
    target: Path,
    opener: _Opener | None = None,
) -> None:
    """Stream, validate, and atomically publish one frozen checkpoint."""

    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.port not in {None, 443}
        or parsed.hostname != "huggingface.co"
    ):
        raise ModelArtifactError("Model artifact URL is not allowed.")
    if expected_size < 1 or max_bytes < expected_size:
        raise ModelArtifactError("Model artifact byte limits are inconsistent.")
    if not _SHA256_PATTERN.fullmatch(expected_sha256):
        raise ModelArtifactError("Model artifact SHA-256 is invalid.")

    target = Path(target)
    parent = target.parent
    if parent.is_symlink() or not parent.is_dir():
        raise ModelArtifactError("Model artifact directory is not a regular directory.")
    if _validated_existing_artifact(
        target,
        expected_sha256=expected_sha256,
        expected_size=expected_size,
    ):
        return

    temporary = target.with_name(f".{target.name}.downloading")
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ModelArtifactError("Temporary model artifact path is unsafe.")
    temporary.unlink(missing_ok=True)

    request = Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "OralLens-model-bootstrap/1",
        },
        method="GET",
    )
    resolved_opener = opener or build_opener(_SafeHuggingFaceRedirectHandler())
    digest = hashlib.sha256()
    downloaded = 0
    try:
        with resolved_opener.open(request, timeout=120.0) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise ModelArtifactError(
                        "Model artifact Content-Length is invalid."
                    ) from exc
                if declared_size != expected_size or declared_size > max_bytes:
                    raise ModelArtifactError(
                        "Model artifact Content-Length differs from frozen metadata."
                    )

            with temporary.open("xb") as destination:
                while chunk := response.read(_DOWNLOAD_CHUNK_BYTES):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ModelArtifactError("Model artifact exceeds the byte limit.")
                    digest.update(chunk)
                    destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())

        if downloaded != expected_size:
            raise ModelArtifactError("Model artifact size differs from frozen metadata.")
        if not secrets.compare_digest(digest.hexdigest(), expected_sha256):
            raise ModelArtifactError("Model artifact failed SHA-256 verification.")
        temporary.chmod(0o440)
        os.replace(temporary, target)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise ModelArtifactError("Model artifact download failed.") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    try:
        url, token, digest, expected_size, max_bytes = (
            artifact_settings_from_environment(os.environ)
        )
        download_checkpoint(
            url=url,
            token=token,
            expected_sha256=digest,
            expected_size=expected_size,
            max_bytes=max_bytes,
            target=_TARGET_PATH,
        )
    except ModelArtifactError as exc:
        print(f"Model artifact preparation failed: {exc}", file=sys.stderr)
        return 1
    print("Frozen model artifact is present and SHA-256 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
