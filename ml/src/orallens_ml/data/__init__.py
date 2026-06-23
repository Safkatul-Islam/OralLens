"""Dataset contracts, auditing, and leakage-resistant splitting."""

from orallens_ml.data.audit import AuditIssue, AuditReport, audit_dataset
from orallens_ml.data.manifest import (
    ALLOWED_LABELS,
    ManifestError,
    SampleRecord,
    load_manifest,
)
from orallens_ml.data.splits import (
    SplitError,
    SplitRatios,
    assign_patient_splits,
    validate_patient_isolation,
)

__all__ = [
    "ALLOWED_LABELS",
    "AuditIssue",
    "AuditReport",
    "ManifestError",
    "SampleRecord",
    "SplitError",
    "SplitRatios",
    "assign_patient_splits",
    "audit_dataset",
    "load_manifest",
    "validate_patient_isolation",
]
