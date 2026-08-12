"""Versioned source identity and split-isolation rules for detection manifests."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Literal

DETECTION_MANIFEST_SCHEMA_VERSION = 1
DETECTION_MANIFEST_IDENTITY_COLUMNS = (
    "manifest_schema_version",
    "dataset_id",
    "dataset_version",
    "source_family_id",
    "source_artifact_id",
    "split_group_id",
    "derivative_group_id",
    "variant",
)

_ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
_IDENTIFIER_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*")


class DetectionManifestError(ValueError):
    """Raised when detection-manifest identity or isolation is invalid."""


@dataclass(frozen=True, slots=True)
class DetectionManifestIdentity:
    """Identity fields required for one source-aware detection sample."""

    manifest_schema_version: int
    dataset_id: str
    dataset_version: int
    source_family_id: str
    source_artifact_id: str
    sample_id: str
    split_group_id: str
    derivative_group_id: str
    variant: str
    split: Literal["train", "validation", "test"]


def validate_detection_manifest_identities(
    identities: Iterable[DetectionManifestIdentity],
) -> tuple[DetectionManifestIdentity, ...]:
    """Validate source identity, uniqueness, and split isolation."""

    records = tuple(identities)
    if not records:
        raise DetectionManifestError("Detection manifest must contain at least one sample")

    sample_keys: set[tuple[str, int, str]] = set()
    split_group_splits: dict[tuple[str, str], set[str]] = defaultdict(set)
    derivative_memberships: dict[
        tuple[str, str], set[tuple[str, str]]
    ] = defaultdict(set)
    derivative_variants: dict[tuple[str, str], set[str]] = defaultdict(set)

    for record in records:
        if record.manifest_schema_version != DETECTION_MANIFEST_SCHEMA_VERSION:
            raise DetectionManifestError(
                "Unsupported detection manifest schema: "
                f"{record.manifest_schema_version}"
            )
        if record.dataset_version <= 0:
            raise DetectionManifestError("Dataset version must be positive")
        for field_name in (
            "dataset_id",
            "source_family_id",
            "source_artifact_id",
            "sample_id",
            "split_group_id",
            "derivative_group_id",
            "variant",
        ):
            value = getattr(record, field_name)
            if _IDENTIFIER_PATTERN.fullmatch(value) is None:
                raise DetectionManifestError(
                    f"Invalid detection manifest {field_name}: {value!r}"
                )
        if record.split not in _ALLOWED_SPLITS:
            raise DetectionManifestError(
                f"Unsupported detection manifest split: {record.split!r}"
            )

        sample_key = (record.dataset_id, record.dataset_version, record.sample_id)
        if sample_key in sample_keys:
            raise DetectionManifestError(
                "Duplicate detection sample identity: "
                f"{record.dataset_id} v{record.dataset_version} {record.sample_id}"
            )
        sample_keys.add(sample_key)

        split_group_key = (record.source_family_id, record.split_group_id)
        split_group_splits[split_group_key].add(record.split)

        derivative_key = (record.source_family_id, record.derivative_group_id)
        derivative_memberships[derivative_key].add(
            (record.split_group_id, record.split)
        )
        if record.variant in derivative_variants[derivative_key]:
            raise DetectionManifestError(
                "Duplicate derivative variant within source family: "
                f"{record.source_family_id} {record.derivative_group_id} "
                f"{record.variant}"
            )
        derivative_variants[derivative_key].add(record.variant)

    leaking_groups = [
        key for key, splits in split_group_splits.items() if len(splits) > 1
    ]
    if leaking_groups:
        family, group = sorted(leaking_groups)[0]
        raise DetectionManifestError(
            f"Split group leaks across splits: {family} {group}"
        )

    inconsistent_derivatives = [
        key
        for key, memberships in derivative_memberships.items()
        if len(memberships) > 1
    ]
    if inconsistent_derivatives:
        family, group = sorted(inconsistent_derivatives)[0]
        raise DetectionManifestError(
            f"Derivative group crosses split groups or splits: {family} {group}"
        )
    return records
