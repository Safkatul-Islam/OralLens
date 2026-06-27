from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orallens_ml.data.acquisition import (
    AcquisitionError,
    DatasetArtifact,
    finalize_download,
    load_release_config,
    verify_download,
)


def artifact(payload: bytes = b"verified-payload", **overrides: object) -> DatasetArtifact:
    values: dict[str, object] = {
        "artifact_id": "part-1",
        "record_url": "https://data.mendeley.com/datasets/example/3",
        "download_url": "https://data.mendeley.com/public-api/zip/example/download/3",
        "partial_filename": "part-1.zip.part",
        "final_filename": "part-1.zip",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "max_download_bytes": 1024,
        "max_archive_entries": 4,
        "max_member_bytes": 1024,
        "max_uncompressed_bytes": 2048,
        "max_compression_ratio": 10.0,
        "expected_zip_members": ("part-1.7z",),
    }
    values.update(overrides)
    return DatasetArtifact(**values)  # type: ignore[arg-type]


def write_config(path: Path, *, download_url: str | None = None) -> None:
    url = download_url or "https://data.mendeley.com/public-api/zip/example/download/3"
    path.write_text(
        f'''schema_version = 1
dataset_id = "dataset"
dataset_version = 3
license = "CC BY 4.0"
publisher = "Mendeley Data"

[[artifacts]]
artifact_id = "part-1"
record_url = "https://data.mendeley.com/datasets/example/3"
download_url = "{url}"
partial_filename = "part-1.zip.part"
final_filename = "part-1.zip"
sha256 = "{'a' * 64}"
max_download_bytes = 1024
max_archive_entries = 4
max_member_bytes = 1024
max_uncompressed_bytes = 2048
max_compression_ratio = 10.0
expected_zip_members = ["part-1.7z"]
''',
        encoding="utf-8",
    )


def test_load_release_config_accepts_allowlisted_source(tmp_path: Path) -> None:
    config = tmp_path / "source.toml"
    write_config(config)

    release = load_release_config(config)

    assert release.dataset_version == 3
    assert release.artifact("part-1").final_filename == "part-1.zip"


def test_project_config_matches_reviewed_mendeley_release() -> None:
    config = Path(__file__).resolve().parents[1] / "configs" / "orthodontic_plaque_v3.toml"

    release = load_release_config(config)
    artifacts = {candidate.artifact_id: candidate for candidate in release.artifacts}

    assert release.dataset_version == 3
    assert release.license == "CC BY 4.0"
    assert set(artifacts) == {"part-1", "part-2"}
    assert artifacts["part-1"].sha256 == (
        "9ab308d919bae0ea6104e9f4c96336be19aa4841c830b8fca1db2f92e3ebe618"
    )
    assert artifacts["part-2"].sha256 == (
        "990691d1c01e8c83be820df22fa38520bc085e3d83efa0a96b09fb8787a49a85"
    )
    assert artifacts["part-1"].expected_zip_members == (
        "mendeley-dataset-materials_Part_1.7z",
    )
    assert artifacts["part-2"].expected_zip_members == (
        "mendeley-dataset-materials_Part_2.7z",
    )
    assert all(
        candidate.download_url.startswith(
            "https://data.mendeley.com/public-api/zip/"
        )
        for candidate in artifacts.values()
    )
    assert all(
        candidate.max_download_bytes == 11_000_000_000
        for candidate in artifacts.values()
    )


@pytest.mark.parametrize(
    "url",
    [
        "http://data.mendeley.com/public-api/zip/example/download/3",
        "https://evil.example/public-api/zip/example/download/3",
        "https://data.mendeley.com:444/public-api/zip/example/download/3",
        "https://data.mendeley.com/public-api/zip/example/download/3?token=x",
    ],
)
def test_load_release_config_rejects_untrusted_download_urls(
    tmp_path: Path, url: str
) -> None:
    config = tmp_path / "source.toml"
    write_config(config, download_url=url)

    with pytest.raises(AcquisitionError, match="must be an HTTPS"):
        load_release_config(config)


def test_verify_download_checks_streaming_sha256(tmp_path: Path) -> None:
    payload = b"verified-payload"
    path = tmp_path / "part-1.zip.part"
    path.write_bytes(payload)

    verified = verify_download(path, artifact(payload))

    assert verified.size_bytes == len(payload)
    assert verified.sha256 == hashlib.sha256(payload).hexdigest()


def test_verify_download_rejects_hash_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "part-1.zip.part"
    path.write_bytes(b"tampered")

    with pytest.raises(AcquisitionError, match="SHA-256 mismatch"):
        verify_download(path, artifact())


def test_verify_download_rejects_oversized_file(tmp_path: Path) -> None:
    payload = b"large"
    path = tmp_path / "part-1.zip.part"
    path.write_bytes(payload)

    with pytest.raises(AcquisitionError, match="exceeds"):
        verify_download(path, artifact(payload, max_download_bytes=2))


def test_finalize_download_verifies_then_renames_atomically(tmp_path: Path) -> None:
    payload = b"verified-payload"
    partial = tmp_path / "part-1.zip.part"
    partial.write_bytes(payload)

    verified = finalize_download(tmp_path, artifact(payload))

    assert verified.path == tmp_path.resolve() / "part-1.zip"
    assert verified.path.read_bytes() == payload
    assert not partial.exists()


def test_finalize_download_never_overwrites_final_file(tmp_path: Path) -> None:
    payload = b"verified-payload"
    (tmp_path / "part-1.zip.part").write_bytes(payload)
    (tmp_path / "part-1.zip").write_bytes(b"existing")

    with pytest.raises(AcquisitionError, match="already exists"):
        finalize_download(tmp_path, artifact(payload))
