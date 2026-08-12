"""Trusted-source configuration and downloaded-artifact verification."""

from __future__ import annotations

import hashlib
import math
import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

_ALLOWED_DOWNLOAD_HOST = "data.mendeley.com"
_ALLOWED_DATASET_ROLES = frozenset(
    {
        "input_quality",
        "locked_challenge",
        "ood_abstention",
        "oral_roi",
        "plaque_supervision",
    }
)
_PLAQUE_LABEL_SEMANTICS = {
    "0": "plaque_absent_region",
    "1": "plaque_present_region",
}
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")
_GROUPING_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]*")
_ISO_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")
_DEFAULT_CHUNK_SIZE = 1024 * 1024


class AcquisitionError(ValueError):
    """Raised when acquisition metadata or a downloaded artifact is unsafe."""


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """One publisher artifact and its security limits."""

    artifact_id: str
    record_url: str
    download_url: str
    partial_filename: str
    final_filename: str
    sha256: str
    max_download_bytes: int
    max_archive_entries: int
    max_member_bytes: int
    max_uncompressed_bytes: int
    max_compression_ratio: float
    expected_zip_members: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetRelease:
    """Validated acquisition metadata for one dataset release."""

    schema_version: int
    dataset_id: str
    dataset_version: int
    license: str
    publisher: str
    source_family_id: str
    reviewed_on: str
    roles: tuple[str, ...]
    grouping_keys: tuple[str, ...]
    capture_context: str
    label_semantics: tuple[tuple[str, str], ...]
    limitations: tuple[str, ...]
    artifacts: tuple[DatasetArtifact, ...]

    def artifact(self, artifact_id: str) -> DatasetArtifact:
        for candidate in self.artifacts:
            if candidate.artifact_id == artifact_id:
                return candidate
        raise AcquisitionError(f"Unknown artifact ID: {artifact_id!r}")


@dataclass(frozen=True, slots=True)
class VerifiedDownload:
    """Identity evidence for a verified local artifact."""

    path: Path
    size_bytes: int
    sha256: str


def load_release_config(config_path: Path) -> DatasetRelease:
    """Load and validate a trusted-source TOML configuration."""

    path = Path(config_path)
    if path.is_symlink() or not path.is_file():
        raise AcquisitionError(f"Acquisition config is not a regular file: {path}")

    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise AcquisitionError(f"Acquisition config is invalid TOML: {exc}") from exc

    schema_version = _required_int(data, "schema_version", "config")
    if schema_version != 2:
        raise AcquisitionError(f"Unsupported acquisition schema: {schema_version}")

    roles = _required_unique_string_list(data, "roles", "config")
    unknown_roles = sorted(set(roles) - _ALLOWED_DATASET_ROLES)
    if unknown_roles:
        raise AcquisitionError(
            "Config contains unsupported dataset roles: " + ", ".join(unknown_roles)
        )
    grouping_keys = _required_unique_string_list(data, "grouping_keys", "config")
    invalid_grouping_keys = [
        key for key in grouping_keys if _GROUPING_KEY_PATTERN.fullmatch(key) is None
    ]
    if invalid_grouping_keys:
        raise AcquisitionError(
            "Config grouping_keys must use lowercase identifier names"
        )
    label_semantics = _required_string_mapping(data, "label_semantics", "config")
    if "plaque_supervision" in roles and dict(label_semantics) != _PLAQUE_LABEL_SEMANTICS:
        raise AcquisitionError(
            "Plaque-supervision sources must define label semantics "
            "0=plaque_absent_region and 1=plaque_present_region"
        )

    source_family_id = _required_string(data, "source_family_id", "config")
    if _IDENTIFIER_PATTERN.fullmatch(source_family_id) is None:
        raise AcquisitionError(
            "config source_family_id must use lowercase letters, numbers, '.', '_' or '-'"
        )
    reviewed_on = _validated_reviewed_on(
        _required_string(data, "reviewed_on", "config")
    )

    raw_artifacts = data.get("artifacts")
    if not isinstance(raw_artifacts, list) or not raw_artifacts:
        raise AcquisitionError("Config must define at least one artifact")

    artifacts = tuple(
        _parse_artifact(raw, index)
        for index, raw in enumerate(raw_artifacts, start=1)
    )
    _require_unique(
        (artifact.artifact_id for artifact in artifacts),
        "artifact IDs",
    )
    _require_unique(
        (artifact.partial_filename for artifact in artifacts),
        "partial filenames",
    )
    _require_unique(
        (artifact.final_filename for artifact in artifacts),
        "final filenames",
    )

    return DatasetRelease(
        schema_version=schema_version,
        dataset_id=_required_string(data, "dataset_id", "config"),
        dataset_version=_positive_int(data, "dataset_version", "config"),
        license=_required_string(data, "license", "config"),
        publisher=_required_string(data, "publisher", "config"),
        source_family_id=source_family_id,
        reviewed_on=reviewed_on,
        roles=roles,
        grouping_keys=grouping_keys,
        capture_context=_required_string(data, "capture_context", "config"),
        label_semantics=label_semantics,
        limitations=_required_unique_string_list(data, "limitations", "config"),
        artifacts=artifacts,
    )


def sha256_file(file_path: Path, *, chunk_size: int = _DEFAULT_CHUNK_SIZE) -> tuple[str, int]:
    """Return a streaming SHA-256 digest and reject files changed mid-read."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    path = Path(file_path)
    if path.is_symlink() or not path.is_file():
        raise AcquisitionError(f"Artifact is not a regular file: {path}")

    before = path.stat()
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
            size += len(chunk)
    after = path.stat()

    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or size != after.st_size:
        raise AcquisitionError("Artifact changed while its checksum was calculated")
    return digest.hexdigest(), size


def verify_download(file_path: Path, artifact: DatasetArtifact) -> VerifiedDownload:
    """Verify artifact size and publisher-provided SHA-256 identity."""

    digest, size = sha256_file(file_path)
    if size == 0:
        raise AcquisitionError("Downloaded artifact is empty")
    if size > artifact.max_download_bytes:
        raise AcquisitionError(
            f"Downloaded artifact exceeds {artifact.max_download_bytes} bytes"
        )
    if digest != artifact.sha256:
        raise AcquisitionError(
            f"SHA-256 mismatch for {artifact.artifact_id}: expected "
            f"{artifact.sha256}, got {digest}"
        )
    return VerifiedDownload(path=Path(file_path), size_bytes=size, sha256=digest)


def finalize_download(download_dir: Path, artifact: DatasetArtifact) -> VerifiedDownload:
    """Verify a partial download and atomically promote it to its final name."""

    root = _existing_safe_directory(download_dir)
    partial_path = root / artifact.partial_filename
    final_path = root / artifact.final_filename
    if final_path.exists() or final_path.is_symlink():
        raise AcquisitionError(f"Final artifact already exists: {final_path}")

    verified = verify_download(partial_path, artifact)
    os.replace(partial_path, final_path)
    return VerifiedDownload(
        path=final_path,
        size_bytes=verified.size_bytes,
        sha256=verified.sha256,
    )


def artifact_path(
    download_dir: Path,
    artifact: DatasetArtifact,
    *,
    finalized: bool,
) -> Path:
    """Return a config-derived path inside an existing download directory."""

    root = _existing_safe_directory(download_dir)
    filename = artifact.final_filename if finalized else artifact.partial_filename
    return root / filename


def _parse_artifact(raw: Any, index: int) -> DatasetArtifact:
    context = f"artifact {index}"
    if not isinstance(raw, Mapping):
        raise AcquisitionError(f"{context} must be a table")

    record_url = _validated_url(
        _required_string(raw, "record_url", context),
        required_path_prefix="/datasets/",
        context=f"{context} record_url",
    )
    download_url = _validated_url(
        _required_string(raw, "download_url", context),
        required_path_prefix="/public-api/zip/",
        context=f"{context} download_url",
    )
    partial_filename = _safe_filename(
        _required_string(raw, "partial_filename", context),
        f"{context} partial_filename",
    )
    final_filename = _safe_filename(
        _required_string(raw, "final_filename", context),
        f"{context} final_filename",
    )
    if not partial_filename.endswith(".zip.part"):
        raise AcquisitionError(f"{context} partial_filename must end with .zip.part")
    if not final_filename.endswith(".zip"):
        raise AcquisitionError(f"{context} final_filename must end with .zip")

    sha256 = _required_string(raw, "sha256", context).lower()
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise AcquisitionError(f"{context} sha256 must contain 64 hexadecimal characters")

    expected_members_raw = raw.get("expected_zip_members")
    if not isinstance(expected_members_raw, list) or not expected_members_raw:
        raise AcquisitionError(f"{context} expected_zip_members must be a non-empty list")
    expected_members = tuple(
        _safe_archive_member(member, f"{context} expected_zip_members")
        for member in expected_members_raw
    )
    _require_unique((member.casefold() for member in expected_members), "archive members")

    max_compression_ratio = raw.get("max_compression_ratio")
    if (
        isinstance(max_compression_ratio, bool)
        or not isinstance(max_compression_ratio, (int, float))
        or not math.isfinite(max_compression_ratio)
        or max_compression_ratio < 1.0
    ):
        raise AcquisitionError(
            f"{context} max_compression_ratio must be finite and at least 1.0"
        )

    return DatasetArtifact(
        artifact_id=_required_string(raw, "artifact_id", context),
        record_url=record_url,
        download_url=download_url,
        partial_filename=partial_filename,
        final_filename=final_filename,
        sha256=sha256,
        max_download_bytes=_positive_int(raw, "max_download_bytes", context),
        max_archive_entries=_positive_int(raw, "max_archive_entries", context),
        max_member_bytes=_positive_int(raw, "max_member_bytes", context),
        max_uncompressed_bytes=_positive_int(
            raw, "max_uncompressed_bytes", context
        ),
        max_compression_ratio=float(max_compression_ratio),
        expected_zip_members=expected_members,
    )


def _validated_url(value: str, *, required_path_prefix: str, context: str) -> str:
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise AcquisitionError(f"{context} is not a valid URL") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != _ALLOWED_DOWNLOAD_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or parsed.query
        or parsed.fragment
        or not parsed.path.startswith(required_path_prefix)
    ):
        raise AcquisitionError(
            f"{context} must be an HTTPS {_ALLOWED_DOWNLOAD_HOST} URL under "
            f"{required_path_prefix}"
        )
    return value


def _safe_filename(value: str, context: str) -> str:
    if (
        value in {"", ".", ".."}
        or PurePosixPath(value).name != value
        or PureWindowsPath(value).name != value
        or ":" in value
        or "\x00" in value
    ):
        raise AcquisitionError(f"{context} must be a safe basename")
    return value


def _safe_archive_member(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AcquisitionError(f"{context} contains an invalid member name")
    if "\\" in value or ":" in value:
        raise AcquisitionError(f"{context} must use safe POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise AcquisitionError(f"{context} contains an unsafe member path")
    if path.suffix.lower() != ".7z":
        raise AcquisitionError(f"{context} members must be .7z files")
    return path.as_posix()


def _existing_safe_directory(directory: Path) -> Path:
    path = Path(directory)
    if path.is_symlink() or not path.is_dir():
        raise AcquisitionError(f"Directory is not a regular directory: {path}")
    return path.resolve(strict=True)


def _required_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AcquisitionError(f"{context} {key} must be a non-empty string")
    return value.strip()


def _required_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise AcquisitionError(f"{context} {key} must be an integer")
    return value


def _required_unique_string_list(
    table: Mapping[str, Any],
    key: str,
    context: str,
) -> tuple[str, ...]:
    raw_values = table.get(key)
    if not isinstance(raw_values, list) or not raw_values:
        raise AcquisitionError(f"{context} {key} must be a non-empty list")
    values = tuple(
        _non_empty_string(value, f"{context} {key}") for value in raw_values
    )
    if len(values) != len(set(values)):
        raise AcquisitionError(f"{context} {key} must not contain duplicates")
    return values


def _required_string_mapping(
    table: Mapping[str, Any],
    key: str,
    context: str,
) -> tuple[tuple[str, str], ...]:
    raw_mapping = table.get(key)
    if not isinstance(raw_mapping, Mapping) or not raw_mapping:
        raise AcquisitionError(f"{context} {key} must be a non-empty table")
    values = tuple(
        sorted(
            (
                _non_empty_string(raw_key, f"{context} {key} key"),
                _non_empty_string(raw_value, f"{context} {key} value"),
            )
            for raw_key, raw_value in raw_mapping.items()
        )
    )
    return values


def _non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AcquisitionError(f"{context} must contain non-empty strings")
    return value.strip()


def _validated_reviewed_on(value: str) -> str:
    if _ISO_DATE_PATTERN.fullmatch(value) is None:
        raise AcquisitionError("config reviewed_on must use YYYY-MM-DD format")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise AcquisitionError("config reviewed_on must be a valid calendar date") from exc
    return value


def _positive_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = _required_int(table, key, context)
    if value <= 0:
        raise AcquisitionError(f"{context} {key} must be greater than zero")
    return value


def _require_unique(values: Any, label: str) -> None:
    materialized = tuple(values)
    if len(materialized) != len(set(materialized)):
        raise AcquisitionError(f"Config contains duplicate {label}")
