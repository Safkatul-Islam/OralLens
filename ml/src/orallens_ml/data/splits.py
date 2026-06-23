"""Deterministic patient-grouped dataset splitting."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from orallens_ml.data.manifest import SampleRecord

SPLIT_NAMES = ("train", "validation", "test")


class SplitError(ValueError):
    """Raised when split configuration or assignments are unsafe."""


@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Train, validation, and test proportions."""

    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise SplitError("Every split ratio must be finite and greater than zero")
        if not math.isclose(sum(values), 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise SplitError("Split ratios must sum to 1.0")


def assign_patient_splits(
    records: Iterable[SampleRecord],
    ratios: SplitRatios = SplitRatios(),
    *,
    seed: str = "orallens-v1",
) -> dict[str, str]:
    """Return sample-to-split assignments with patients kept together.

    Patient IDs are ordered by a seeded SHA-256 digest. This makes assignments
    reproducible across Python versions and operating systems without depending
    on process-randomized hashes.
    """

    samples = tuple(records)
    if not samples:
        raise SplitError("Cannot split an empty dataset")
    if not seed:
        raise SplitError("Split seed must not be empty")

    patient_ids = sorted(
        {sample.patient_id for sample in samples},
        key=lambda patient_id: _stable_digest(seed, patient_id),
    )
    if len(patient_ids) < len(SPLIT_NAMES):
        raise SplitError("At least three patients are required for three non-empty splits")

    counts = _allocate_counts(len(patient_ids), ratios)
    patient_splits: dict[str, str] = {}
    start = 0
    for split_name, count in zip(SPLIT_NAMES, counts, strict=True):
        for patient_id in patient_ids[start : start + count]:
            patient_splits[patient_id] = split_name
        start += count

    assignments = {
        sample.sample_id: patient_splits[sample.patient_id] for sample in samples
    }
    validate_patient_isolation(samples, assignments)
    return assignments


def validate_patient_isolation(
    records: Iterable[SampleRecord],
    assignments: Mapping[str, str],
) -> None:
    """Reject missing, unknown, invalid, or patient-leaking assignments."""

    samples = tuple(records)
    sample_ids = {sample.sample_id for sample in samples}
    if len(sample_ids) != len(samples):
        raise SplitError("Sample IDs must be unique before creating assignments")

    missing = sample_ids - assignments.keys()
    unknown = assignments.keys() - sample_ids
    if missing:
        raise SplitError(f"Assignments are missing {len(missing)} sample(s)")
    if unknown:
        raise SplitError(f"Assignments contain {len(unknown)} unknown sample(s)")

    patient_splits: dict[str, set[str]] = defaultdict(set)
    for sample in samples:
        split_name = assignments[sample.sample_id]
        if split_name not in SPLIT_NAMES:
            raise SplitError(f"Unknown split name: {split_name!r}")
        patient_splits[sample.patient_id].add(split_name)

    leaking = [patient_id for patient_id, names in patient_splits.items() if len(names) > 1]
    if leaking:
        raise SplitError(f"Patient leakage detected for {len(leaking)} patient(s)")


def _stable_digest(seed: str, patient_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{patient_id}".encode("utf-8")).digest()


def _allocate_counts(patient_count: int, ratios: SplitRatios) -> tuple[int, int, int]:
    raw_counts = (
        patient_count * ratios.train,
        patient_count * ratios.validation,
        patient_count * ratios.test,
    )
    counts = [math.floor(value) for value in raw_counts]
    remainder = patient_count - sum(counts)
    priorities = sorted(
        range(len(counts)),
        key=lambda index: (raw_counts[index] - counts[index], -index),
        reverse=True,
    )
    for index in priorities[:remainder]:
        counts[index] += 1

    for empty_index, count in enumerate(counts):
        if count == 0:
            donor_index = max(range(len(counts)), key=counts.__getitem__)
            if counts[donor_index] <= 1:
                raise SplitError("Not enough patients to create non-empty splits")
            counts[donor_index] -= 1
            counts[empty_index] += 1
    return counts[0], counts[1], counts[2]
