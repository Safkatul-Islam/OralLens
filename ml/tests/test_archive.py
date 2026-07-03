from __future__ import annotations

import hashlib
import stat
import zipfile
from pathlib import Path

import pytest

from orallens_ml.data.acquisition import DatasetArtifact
from orallens_ml.data.archive import (
    ArchiveSecurityError,
    convert_7zip_slt_listing_lines,
    extract_outer_zip,
    inspect_zip,
    summarize_7zip_slt_listing,
    summarize_inner_listing,
    summarize_inner_listing_lines,
)


def artifact(member: str = "nested.7z", **overrides: object) -> DatasetArtifact:
    values: dict[str, object] = {
        "artifact_id": "part-1",
        "record_url": "https://data.mendeley.com/datasets/example/3",
        "download_url": "https://data.mendeley.com/public-api/zip/example/download/3",
        "partial_filename": "part-1.zip.part",
        "final_filename": "part-1.zip",
        "sha256": hashlib.sha256(b"unused").hexdigest(),
        "max_download_bytes": 4096,
        "max_archive_entries": 4,
        "max_member_bytes": 4096,
        "max_uncompressed_bytes": 8192,
        "max_compression_ratio": 20.0,
        "expected_zip_members": (member,),
    }
    values.update(overrides)
    return DatasetArtifact(**values)  # type: ignore[arg-type]


def write_zip(
    path: Path,
    entries: dict[str, bytes],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> None:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, payload in entries.items():
            archive.writestr(name, payload)


def replace_zip_member_name(path: Path, original: str, replacement: str) -> None:
    original_bytes = original.encode("utf-8")
    replacement_bytes = replacement.encode("utf-8")
    assert len(original_bytes) == len(replacement_bytes)

    data = path.read_bytes()
    assert data.count(original_bytes) == 2
    path.write_bytes(data.replace(original_bytes, replacement_bytes))


def test_inspect_zip_accepts_exact_expected_regular_member(tmp_path: Path) -> None:
    path = tmp_path / "part.zip"
    write_zip(path, {"nested.7z": b"nested archive bytes"})

    report = inspect_zip(path, artifact())

    assert [entry.name for entry in report.entries] == ["nested.7z"]
    assert report.total_uncompressed_bytes == len(b"nested archive bytes")


@pytest.mark.parametrize(
    "member",
    [
        "../escape.7z",
        "/absolute.7z",
        "C:/absolute.7z",
        "folder\\escape.7z",
        "folder/../escape.7z",
    ],
)
def test_inspect_zip_rejects_unsafe_member_paths(tmp_path: Path, member: str) -> None:
    path = tmp_path / "unsafe.zip"
    stored_member = member.replace("\\", "/")
    write_zip(path, {stored_member: b"unsafe"})
    if "\\" in member:
        replace_zip_member_name(path, stored_member, member)
        with zipfile.ZipFile(path, "r") as archive:
            info = archive.infolist()[0]
            assert info.orig_filename == member
            assert info.filename == stored_member

    with pytest.raises(ArchiveSecurityError, match="unsafe"):
        inspect_zip(path, artifact(member))


def test_inspect_zip_rejects_case_insensitive_duplicate_paths(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    write_zip(path, {"Nested.7z": b"one", "nested.7z": b"two"})
    policy = artifact(expected_zip_members=("Nested.7z", "nested.7z"))

    with pytest.raises(ArchiveSecurityError, match="duplicate"):
        inspect_zip(path, policy)


def test_inspect_zip_rejects_unexpected_members(tmp_path: Path) -> None:
    path = tmp_path / "unexpected.zip"
    write_zip(path, {"nested.7z": b"expected", "extra.txt": b"unexpected"})

    with pytest.raises(ArchiveSecurityError, match="member set differs"):
        inspect_zip(path, artifact())


def test_inspect_zip_rejects_symbolic_links(tmp_path: Path) -> None:
    path = tmp_path / "link.zip"
    info = zipfile.ZipInfo("nested.7z")
    info.create_system = 3
    info.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(info, "target")

    with pytest.raises(ArchiveSecurityError, match="not a regular file"):
        inspect_zip(path, artifact())


def test_inspect_zip_rejects_encrypted_flag(tmp_path: Path) -> None:
    path = tmp_path / "encrypted.zip"
    write_zip(path, {"nested.7z": b"payload"})
    data = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        position = data.find(signature)
        assert position >= 0
        current = int.from_bytes(data[position + flag_offset : position + flag_offset + 2], "little")
        data[position + flag_offset : position + flag_offset + 2] = (current | 1).to_bytes(2, "little")
    path.write_bytes(data)

    with pytest.raises(ArchiveSecurityError, match="Encrypted"):
        inspect_zip(path, artifact())


def test_inspect_zip_rejects_high_compression_ratio(tmp_path: Path) -> None:
    path = tmp_path / "bomb.zip"
    write_zip(path, {"nested.7z": b"0" * 2048}, compression=zipfile.ZIP_DEFLATED)

    with pytest.raises(ArchiveSecurityError, match="compression ratio"):
        inspect_zip(path, artifact(max_compression_ratio=2.0))


def test_extract_outer_zip_streams_validated_member(tmp_path: Path) -> None:
    path = tmp_path / "part.zip"
    destination = tmp_path / "nested"
    destination.mkdir()
    payload = b"nested archive bytes"
    write_zip(path, {"nested.7z": payload})

    extracted = extract_outer_zip(path, destination, artifact())

    assert extracted == (destination.resolve() / "nested.7z",)
    assert extracted[0].read_bytes() == payload


def test_extract_outer_zip_never_overwrites_existing_member(tmp_path: Path) -> None:
    path = tmp_path / "part.zip"
    destination = tmp_path / "nested"
    destination.mkdir()
    (destination / "nested.7z").write_bytes(b"existing")
    write_zip(path, {"nested.7z": b"new"})

    with pytest.raises(ArchiveSecurityError, match="already exists"):
        extract_outer_zip(path, destination, artifact())


def test_summarize_inner_listing_reports_dataset_shape_and_pairing() -> None:
    report = summarize_inner_listing_lines(
        [
            "mendeley-dataset-materials_Part_1/",
            "mendeley-dataset-materials_Part_1/data/images/patient0001/sample-a.jpg",
            "mendeley-dataset-materials_Part_1/data/images/Patient0002/sample-b.JPG",
            "mendeley-dataset-materials_Part_1/data/labels/patient0001/sample-a.txt",
            "mendeley-dataset-materials_Part_1/data/labels/patient0002/sample-c.txt",
            "mendeley-dataset-materials_Part_1/clinical_records/patient0001.docx",
            "mendeley-dataset-materials_Part_1/metadata/readme.json",
        ]
    )

    assert report.entry_count == 7
    assert report.directory_count == 1
    assert report.file_count == 6
    assert report.root_entries == ("mendeley-dataset-materials_Part_1",)
    assert report.extension_counts == {".docx": 1, ".jpg": 2, ".json": 1, ".txt": 2}
    assert report.image_patient_count == 2
    assert report.label_patient_count == 2
    assert report.images_without_labels_count == 1
    assert report.labels_without_images_count == 1
    assert report.images_without_labels_examples == (
        "mendeley-dataset-materials_Part_1/data/images/Patient0002/sample-b.JPG",
    )
    assert report.labels_without_images_examples == (
        "mendeley-dataset-materials_Part_1/data/labels/patient0002/sample-c.txt",
    )
    assert report.patient_case_conflicts == {
        "patient0002": ("Patient0002", "patient0002")
    }


@pytest.mark.parametrize(
    "member",
    [
        "../escape.jpg",
        "/absolute.jpg",
        "C:/absolute.jpg",
        "folder\\escape.jpg",
        "folder/../escape.jpg",
    ],
)
def test_summarize_inner_listing_rejects_unsafe_paths(member: str) -> None:
    with pytest.raises(ArchiveSecurityError, match="unsafe"):
        summarize_inner_listing_lines(["root/", member])


def test_summarize_inner_listing_rejects_case_insensitive_duplicates() -> None:
    with pytest.raises(ArchiveSecurityError, match="duplicate"):
        summarize_inner_listing_lines(["root/data/file.jpg", "root/data/FILE.jpg"])


def test_summarize_inner_listing_reads_listing_file(tmp_path: Path) -> None:
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

    report = summarize_inner_listing(listing)

    assert report.listing_path == listing
    assert report.image_file_count == 1
    assert report.label_file_count == 1


def test_summarize_inner_listing_strips_bom_from_first_member() -> None:
    report = summarize_inner_listing_lines(
        [
            "\ufeffmendeley-dataset-materials_Part_1/",
            "mendeley-dataset-materials_Part_1/data/images/patient0001/sample.jpg",
            "mendeley-dataset-materials_Part_1/data/labels/patient0001/sample.txt",
        ]
    )

    assert report.root_entries == ("mendeley-dataset-materials_Part_1",)
    assert report.image_file_count == 1
    assert report.label_file_count == 1


def test_convert_7zip_slt_listing_skips_archive_metadata_and_preserves_folders() -> None:
    converted = convert_7zip_slt_listing_lines(
        [
            "Path = ml\\data\\nested\\part.7z",
            "Type = 7z",
            "Physical Size = 123",
            "",
            "Path = mendeley-dataset-materials_Part_2\\clinical_records",
            "Folder = +",
            "",
            "Path = mendeley-dataset-materials_Part_2\\data\\images\\patient0001\\sample.jpg",
            "Folder = -",
            "Size = 10",
            "",
            "Path = mendeley-dataset-materials_Part_2\\data\\labels\\patient0001\\sample.txt",
            "Attributes = A",
            "Size = 1",
        ]
    )

    assert converted == [
        "mendeley-dataset-materials_Part_2/clinical_records/",
        "mendeley-dataset-materials_Part_2/data/images/patient0001/sample.jpg",
        "mendeley-dataset-materials_Part_2/data/labels/patient0001/sample.txt",
    ]


def test_summarize_7zip_slt_listing_uses_existing_shape_validation(tmp_path: Path) -> None:
    listing = tmp_path / "part-2.slt"
    listing.write_text(
        "\n".join(
            [
                "Path = C:\\archive\\part.7z",
                "Type = 7z",
                "",
                "Path = mendeley-dataset-materials_Part_2\\data",
                "Folder = +",
                "",
                "Path = mendeley-dataset-materials_Part_2\\data\\images\\patient0001\\sample.jpg",
                "Size = 10",
                "",
                "Path = mendeley-dataset-materials_Part_2\\data\\labels\\patient0001\\sample.txt",
                "Size = 1",
            ]
        ),
        encoding="utf-8",
    )

    report = summarize_7zip_slt_listing(listing)

    assert report.listing_path == listing
    assert report.directory_count == 1
    assert report.file_count == 2
    assert report.image_file_count == 1
    assert report.label_file_count == 1


def test_summarize_7zip_slt_listing_rejects_unsafe_normalized_paths(
    tmp_path: Path,
) -> None:
    listing = tmp_path / "unsafe.slt"
    listing.write_text(
        "\n".join(
            [
                "Path = C:\\archive\\part.7z",
                "Type = 7z",
                "",
                "Path = root\\..\\escape.jpg",
                "Size = 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ArchiveSecurityError, match="unsafe"):
        summarize_7zip_slt_listing(listing)
