from __future__ import annotations

from dataclasses import replace

import pytest

from orallens_ml.data.detection_manifest import (
    DETECTION_MANIFEST_SCHEMA_VERSION,
    DetectionManifestError,
    DetectionManifestIdentity,
    validate_detection_manifest_identities,
)


def _identity(**changes: object) -> DetectionManifestIdentity:
    identity = DetectionManifestIdentity(
        manifest_schema_version=DETECTION_MANIFEST_SCHEMA_VERSION,
        dataset_id="dataset",
        dataset_version=1,
        source_family_id="source-family",
        source_artifact_id="part-2",
        sample_id="sample-original",
        split_group_id="patient0001",
        derivative_group_id="sample",
        variant="original",
        split="train",
    )
    return replace(identity, **changes)


def test_detection_manifest_accepts_isolated_derivative_family() -> None:
    records = validate_detection_manifest_identities(
        (
            _identity(),
            _identity(sample_id="sample-rotated", variant="rotate-left-15"),
        )
    )

    assert len(records) == 2


def test_detection_manifest_rejects_split_group_leakage() -> None:
    with pytest.raises(DetectionManifestError, match="Split group leaks"):
        validate_detection_manifest_identities(
            (
                _identity(),
                _identity(
                    sample_id="other-sample",
                    derivative_group_id="other-sample",
                    split="validation",
                ),
            )
        )


def test_detection_manifest_rejects_derivative_group_leakage() -> None:
    with pytest.raises(DetectionManifestError, match="Derivative group crosses"):
        validate_detection_manifest_identities(
            (
                _identity(),
                _identity(
                    sample_id="sample-rotated",
                    split_group_id="patient0002",
                    split="validation",
                    variant="rotate-left-15",
                ),
            )
        )


def test_detection_manifest_rejects_duplicate_variant() -> None:
    with pytest.raises(DetectionManifestError, match="Duplicate derivative variant"):
        validate_detection_manifest_identities(
            (_identity(), _identity(sample_id="duplicate-sample"))
        )


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    (
        ("manifest_schema_version", 2, "Unsupported detection manifest schema"),
        ("dataset_version", 0, "Dataset version must be positive"),
        ("source_family_id", "Unsafe Family", "source_family_id"),
        ("split", "holdout", "Unsupported detection manifest split"),
    ),
)
def test_detection_manifest_rejects_invalid_identity(
    field_name: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(DetectionManifestError, match=message):
        validate_detection_manifest_identities((_identity(**{field_name: value}),))
