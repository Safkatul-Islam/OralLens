"""Read-only ZIP inspection and bounded outer-archive extraction."""

from __future__ import annotations

import os
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

from orallens_ml.data.acquisition import AcquisitionError, DatasetArtifact

_ALLOWED_COMPRESSION_METHODS = frozenset(
    {
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
        zipfile.ZIP_BZIP2,
        zipfile.ZIP_LZMA,
    }
)
_COPY_CHUNK_SIZE = 1024 * 1024


class ArchiveSecurityError(AcquisitionError):
    """Raised when an archive violates the extraction security policy."""


@dataclass(frozen=True, slots=True)
class ArchiveEntry:
    """Validated metadata for one regular ZIP member."""

    name: str
    compressed_bytes: int
    uncompressed_bytes: int
    compression_ratio: float


@dataclass(frozen=True, slots=True)
class ArchiveReport:
    """Validated ZIP metadata summary."""

    archive_path: Path
    entries: tuple[ArchiveEntry, ...]
    total_compressed_bytes: int
    total_uncompressed_bytes: int


def inspect_zip(zip_path: Path, artifact: DatasetArtifact) -> ArchiveReport:
    """Inspect a ZIP without extracting and enforce the artifact policy."""

    path = Path(zip_path)
    if path.is_symlink() or not path.is_file():
        raise ArchiveSecurityError(f"ZIP is not a regular file: {path}")

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            infos = archive.infolist()
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise ArchiveSecurityError(f"Invalid ZIP archive: {exc}") from exc

    if not infos:
        raise ArchiveSecurityError("ZIP archive is empty")
    if len(infos) > artifact.max_archive_entries:
        raise ArchiveSecurityError(
            f"ZIP contains {len(infos)} entries; limit is {artifact.max_archive_entries}"
        )

    entries: list[ArchiveEntry] = []
    normalized_names: set[str] = set()
    total_compressed = 0
    total_uncompressed = 0
    for info in infos:
        name = _validated_member_name(info)
        normalized = name.casefold()
        if normalized in normalized_names:
            raise ArchiveSecurityError(f"ZIP contains duplicate member path: {name}")
        normalized_names.add(normalized)

        if info.flag_bits & 0x1:
            raise ArchiveSecurityError(f"Encrypted ZIP members are not allowed: {name}")
        if info.compress_type not in _ALLOWED_COMPRESSION_METHODS:
            raise ArchiveSecurityError(
                f"Unsupported ZIP compression method for {name}: {info.compress_type}"
            )
        if info.file_size > artifact.max_member_bytes:
            raise ArchiveSecurityError(
                f"ZIP member exceeds {artifact.max_member_bytes} bytes: {name}"
            )
        if info.compress_size == 0 and info.file_size > 0:
            raise ArchiveSecurityError(f"ZIP member has an invalid compressed size: {name}")

        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > artifact.max_compression_ratio:
            raise ArchiveSecurityError(
                f"ZIP member compression ratio {ratio:.2f} exceeds policy: {name}"
            )
        total_compressed += info.compress_size
        total_uncompressed += info.file_size
        if total_uncompressed > artifact.max_uncompressed_bytes:
            raise ArchiveSecurityError(
                "ZIP total uncompressed size exceeds the configured limit"
            )
        entries.append(
            ArchiveEntry(
                name=name,
                compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,
                compression_ratio=ratio,
            )
        )

    actual_names = {entry.name for entry in entries}
    expected_names = set(artifact.expected_zip_members)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise ArchiveSecurityError(
            f"ZIP member set differs from publisher metadata; "
            f"missing={missing}, unexpected={unexpected}"
        )

    return ArchiveReport(
        archive_path=path,
        entries=tuple(entries),
        total_compressed_bytes=total_compressed,
        total_uncompressed_bytes=total_uncompressed,
    )


def extract_outer_zip(
    zip_path: Path,
    destination_dir: Path,
    artifact: DatasetArtifact,
) -> tuple[Path, ...]:
    """Extract only validated expected members with bounded streaming copies."""

    report = inspect_zip(zip_path, artifact)
    destination = Path(destination_dir)
    if destination.is_symlink() or not destination.is_dir():
        raise ArchiveSecurityError(
            f"Extraction destination is not a regular directory: {destination}"
        )
    resolved_destination = destination.resolve(strict=True)

    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            info_by_name = {info.filename: info for info in archive.infolist()}
            for entry in report.entries:
                target = resolved_destination.joinpath(*PurePosixPath(entry.name).parts)
                resolved_target = target.resolve(strict=False)
                if not resolved_target.is_relative_to(resolved_destination):
                    raise ArchiveSecurityError("Extraction target escapes destination")
                if target.exists() or target.is_symlink():
                    raise ArchiveSecurityError(f"Extraction target already exists: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(f"{target.name}.extracting")
                if temporary.exists() or temporary.is_symlink():
                    raise ArchiveSecurityError(
                        f"Temporary extraction target already exists: {temporary}"
                    )

                copied = 0
                try:
                    with archive.open(info_by_name[entry.name], mode="r") as source:
                        with temporary.open("xb") as destination_handle:
                            while chunk := source.read(_COPY_CHUNK_SIZE):
                                copied += len(chunk)
                                if copied > entry.uncompressed_bytes:
                                    raise ArchiveSecurityError(
                                        f"Extracted data exceeds declared size: {entry.name}"
                                    )
                                destination_handle.write(chunk)
                            destination_handle.flush()
                            os.fsync(destination_handle.fileno())
                    if copied != entry.uncompressed_bytes:
                        raise ArchiveSecurityError(
                            f"Extracted size differs from ZIP metadata: {entry.name}"
                        )
                    os.replace(temporary, target)
                    extracted.append(target)
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
    except (RuntimeError, zipfile.BadZipFile) as exc:
        raise ArchiveSecurityError(f"ZIP extraction failed: {exc}") from exc
    return tuple(extracted)


def _validated_member_name(info: zipfile.ZipInfo) -> str:
    raw_name = info.orig_filename
    if not raw_name or "\x00" in raw_name:
        raise ArchiveSecurityError("ZIP contains an empty or null member name")
    if info.is_dir():
        raise ArchiveSecurityError(f"Directory entries are not allowed: {raw_name}")
    if "\\" in raw_name or ":" in raw_name:
        raise ArchiveSecurityError(f"ZIP member uses an unsafe Windows path: {raw_name}")

    posix_path = PurePosixPath(raw_name)
    windows_path = PureWindowsPath(raw_name)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in raw_name.split("/"))
    ):
        raise ArchiveSecurityError(f"ZIP member path is unsafe: {raw_name}")

    unix_mode = (info.external_attr >> 16) & 0xFFFF
    file_type = stat.S_IFMT(unix_mode)
    if file_type not in {0, stat.S_IFREG}:
        raise ArchiveSecurityError(f"ZIP member is not a regular file: {raw_name}")
    return posix_path.as_posix()
