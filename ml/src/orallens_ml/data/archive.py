"""Read-only archive inspection and bounded outer-archive extraction."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
import os
import re
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
_PATIENT_PATTERN = re.compile(r"^patient\d{4}$", re.IGNORECASE)
_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
_LABEL_EXTENSION = ".txt"
_MAX_EXAMPLES = 25


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


@dataclass(frozen=True, slots=True)
class InnerArchiveListingReport:
    """Read-only summary of a nested archive listing."""

    listing_path: Path
    entry_count: int
    directory_count: int
    file_count: int
    root_entries: tuple[str, ...]
    extension_counts: dict[str, int]
    image_file_count: int
    label_file_count: int
    image_patient_count: int
    label_patient_count: int
    image_patients_without_labels: tuple[str, ...]
    label_patients_without_images: tuple[str, ...]
    images_without_labels_count: int
    labels_without_images_count: int
    images_without_labels_examples: tuple[str, ...]
    labels_without_images_examples: tuple[str, ...]
    patient_case_conflicts: dict[str, tuple[str, ...]]


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


def summarize_inner_listing(listing_path: Path) -> InnerArchiveListingReport:
    """Summarize a saved `tar -tf` style listing without extracting files."""

    path = Path(listing_path)
    if path.is_symlink() or not path.is_file():
        raise ArchiveSecurityError(f"Listing is not a regular file: {path}")

    entries = path.read_text(encoding="utf-8").splitlines()
    return summarize_inner_listing_lines(entries, listing_path=path)


def summarize_inner_listing_lines(
    lines: Iterable[str],
    *,
    listing_path: Path | None = None,
) -> InnerArchiveListingReport:
    """Validate and summarize nested archive member names from listing lines."""

    raw_entries = _normalize_listing_lines(lines)
    if not raw_entries:
        raise ArchiveSecurityError("Inner archive listing is empty")

    seen: set[str] = set()
    directory_count = 0
    file_count = 0
    roots: set[str] = set()
    extensions: Counter[str] = Counter()
    patient_spellings: dict[str, set[str]] = defaultdict(set)
    image_patients: set[str] = set()
    label_patients: set[str] = set()
    image_keys: dict[tuple[str, str], str] = {}
    label_keys: dict[tuple[str, str], str] = {}

    for raw_name in raw_entries:
        member_name = _validated_listing_member_name(raw_name)
        normalized = member_name.casefold()
        if normalized in seen:
            raise ArchiveSecurityError(f"Listing contains duplicate member path: {member_name}")
        seen.add(normalized)

        is_directory = member_name.endswith("/")
        parts = PurePosixPath(member_name.rstrip("/")).parts
        roots.add(parts[0])
        if is_directory:
            directory_count += 1
            continue

        file_count += 1
        suffix = PurePosixPath(member_name).suffix.lower() or "<none>"
        extensions[suffix] += 1
        _collect_inner_dataset_member(
            member_name,
            image_patients=image_patients,
            label_patients=label_patients,
            patient_spellings=patient_spellings,
            image_keys=image_keys,
            label_keys=label_keys,
        )

    image_key_set = set(image_keys)
    label_key_set = set(label_keys)
    missing_labels = sorted(image_key_set - label_key_set)
    missing_images = sorted(label_key_set - image_key_set)
    case_conflicts = {
        patient: tuple(sorted(spellings))
        for patient, spellings in sorted(patient_spellings.items())
        if len(spellings) > 1
    }

    return InnerArchiveListingReport(
        listing_path=Path() if listing_path is None else Path(listing_path),
        entry_count=len(raw_entries),
        directory_count=directory_count,
        file_count=file_count,
        root_entries=tuple(sorted(roots)),
        extension_counts=dict(sorted(extensions.items())),
        image_file_count=len(image_keys),
        label_file_count=len(label_keys),
        image_patient_count=len(image_patients),
        label_patient_count=len(label_patients),
        image_patients_without_labels=tuple(sorted(image_patients - label_patients)),
        label_patients_without_images=tuple(sorted(label_patients - image_patients)),
        images_without_labels_count=len(missing_labels),
        labels_without_images_count=len(missing_images),
        images_without_labels_examples=tuple(
            image_keys[key] for key in missing_labels[:_MAX_EXAMPLES]
        ),
        labels_without_images_examples=tuple(
            label_keys[key] for key in missing_images[:_MAX_EXAMPLES]
        ),
        patient_case_conflicts=case_conflicts,
    )


def _normalize_listing_lines(lines: Iterable[str]) -> list[str]:
    entries: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if not entries:
            stripped = stripped.removeprefix("\ufeff")
            if not stripped:
                continue
        entries.append(stripped)
    return entries


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


def _validated_listing_member_name(raw_name: str) -> str:
    if not raw_name or "\x00" in raw_name:
        raise ArchiveSecurityError("Listing contains an empty or null member name")
    if "\\" in raw_name or ":" in raw_name:
        raise ArchiveSecurityError(f"Inner archive member uses an unsafe Windows path: {raw_name}")

    name_without_trailing_slash = raw_name.rstrip("/")
    posix_path = PurePosixPath(name_without_trailing_slash)
    windows_path = PureWindowsPath(name_without_trailing_slash)
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or any(part in {"", ".", ".."} for part in name_without_trailing_slash.split("/"))
    ):
        raise ArchiveSecurityError(f"Inner archive member path is unsafe: {raw_name}")
    return f"{posix_path.as_posix()}/" if raw_name.endswith("/") else posix_path.as_posix()


def _collect_inner_dataset_member(
    member_name: str,
    *,
    image_patients: set[str],
    label_patients: set[str],
    patient_spellings: dict[str, set[str]],
    image_keys: dict[tuple[str, str], str],
    label_keys: dict[tuple[str, str], str],
) -> None:
    parts = PurePosixPath(member_name).parts
    if len(parts) < 5 or parts[1] != "data":
        return

    section = parts[2]
    patient = parts[3]
    if not _PATIENT_PATTERN.match(patient):
        return

    normalized_patient = patient.casefold()
    patient_spellings[normalized_patient].add(patient)
    stem = PurePosixPath(parts[-1]).stem.casefold()
    key = (normalized_patient, stem)
    if section == "images" and PurePosixPath(member_name).suffix.lower() in _IMAGE_EXTENSIONS:
        image_patients.add(normalized_patient)
        image_keys[key] = member_name
    elif section == "labels" and PurePosixPath(member_name).suffix.lower() == _LABEL_EXTENSION:
        label_patients.add(normalized_patient)
        label_keys[key] = member_name
