from __future__ import annotations

import hashlib
from pathlib import Path

from orallens_ml.cli.dataset import main


def write_config(path: Path, digest: str) -> None:
    path.write_text(
        f'''schema_version = 1
dataset_id = "dataset"
dataset_version = 3
license = "CC BY 4.0"
publisher = "Mendeley Data"

[[artifacts]]
artifact_id = "part-1"
record_url = "https://data.mendeley.com/datasets/example/3"
download_url = "https://data.mendeley.com/public-api/zip/example/download/3"
partial_filename = "part-1.zip.part"
final_filename = "part-1.zip"
sha256 = "{digest}"
max_download_bytes = 1024
max_archive_entries = 4
max_member_bytes = 1024
max_uncompressed_bytes = 2048
max_compression_ratio = 10.0
expected_zip_members = ["part-1.7z"]
''',
        encoding="utf-8",
    )


def test_cli_verify_emits_structured_success(tmp_path: Path, capsys: object) -> None:
    payload = b"verified"
    config = tmp_path / "source.toml"
    write_config(config, hashlib.sha256(payload).hexdigest())
    (tmp_path / "part-1.zip.part").write_bytes(payload)

    exit_code = main(
        [
            "--config",
            str(config),
            "verify",
            "part-1",
            "--downloads-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "verified"' in captured.out
    assert captured.err == ""


def test_cli_returns_safe_error_without_traceback(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    (tmp_path / "part-1.zip.part").write_bytes(b"tampered")

    exit_code = main(
        [
            "--config",
            str(config),
            "verify",
            "part-1",
            "--downloads-dir",
            str(tmp_path),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err.startswith("error: SHA-256 mismatch")
    assert "Traceback" not in captured.err


def test_cli_summarizes_inner_listing(tmp_path: Path, capsys: object) -> None:
    config = tmp_path / "source.toml"
    write_config(config, "a" * 64)
    listing = tmp_path / "part-1-listing.txt"
    listing.write_text(
        "\n".join(
            [
                "mendeley-dataset-materials_Part_1/",
                "mendeley-dataset-materials_Part_1/data/images/patient0001/sample.jpg",
                "mendeley-dataset-materials_Part_1/data/labels/patient0001/sample.txt",
            ]
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--config",
            str(config),
            "summarize-inner-listing",
            "part-1",
            "--listing-file",
            str(listing),
        ]
    )

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert exit_code == 0
    assert '"status": "inner-listing-summarized"' in captured.out
    assert '"image_file_count": 1' in captured.out
    assert captured.err == ""
