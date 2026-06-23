from __future__ import annotations

from pathlib import PurePosixPath

import pytest

from orallens_ml.data.manifest import SampleRecord
from orallens_ml.data.splits import (
    SplitError,
    SplitRatios,
    assign_patient_splits,
    validate_patient_isolation,
)


def records(patient_count: int = 20, variants: int = 2) -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id=f"p{patient}-v{variant}",
            patient_id=f"p{patient}",
            relative_path=PurePosixPath(f"images/p{patient}-v{variant}.jpg"),
            label=str(2 + patient % 3),
            source_sample_id=f"p{patient}-source",
            variant=f"variant-{variant}",
            manifest_line=patient * variants + variant + 2,
        )
        for patient in range(patient_count)
        for variant in range(variants)
    ]


def test_patient_splits_are_deterministic_and_isolated() -> None:
    samples = records()

    first = assign_patient_splits(samples, seed="reproducible-seed")
    second = assign_patient_splits(reversed(samples), seed="reproducible-seed")

    assert first == second
    assert set(first.values()) == {"train", "validation", "test"}
    validate_patient_isolation(samples, first)
    for patient in {sample.patient_id for sample in samples}:
        patient_assignments = {
            first[sample.sample_id] for sample in samples if sample.patient_id == patient
        }
        assert len(patient_assignments) == 1


def test_different_seeds_change_assignments() -> None:
    samples = records()
    assert assign_patient_splits(samples, seed="one") != assign_patient_splits(
        samples, seed="two"
    )


@pytest.mark.parametrize(
    "ratios",
    [
        (0.7, 0.2, 0.2),
        (1.0, 0.0, 0.0),
        (float("nan"), 0.5, 0.5),
    ],
)
def test_split_ratios_reject_invalid_values(ratios: tuple[float, float, float]) -> None:
    with pytest.raises(SplitError):
        SplitRatios(*ratios)


def test_split_requires_at_least_three_patients() -> None:
    with pytest.raises(SplitError, match="At least three patients"):
        assign_patient_splits(records(patient_count=2))


def test_validation_detects_patient_leakage() -> None:
    samples = records(patient_count=3)
    assignments = {sample.sample_id: "train" for sample in samples}
    assignments[samples[1].sample_id] = "test"

    with pytest.raises(SplitError, match="Patient leakage"):
        validate_patient_isolation(samples, assignments)
