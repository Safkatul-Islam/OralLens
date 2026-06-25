"""Dataset acquisition, auditing, and leakage-resistant splitting."""

from orallens_ml.data.acquisition import (
    AcquisitionError,
    DatasetArtifact,
    DatasetRelease,
    VerifiedDownload,
    finalize_download,
    load_release_config,
    verify_download,
)
from orallens_ml.data.archive import (
    ArchiveEntry,
    ArchiveReport,
    ArchiveSecurityError,
    extract_outer_zip,
    inspect_zip,
)
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
    "AcquisitionError",
    "ArchiveEntry",
    "ArchiveReport",
    "ArchiveSecurityError",
    "AuditIssue",
    "AuditReport",
    "DatasetArtifact",
    "DatasetRelease",
    "ManifestError",
    "SampleRecord",
    "SplitError",
    "SplitRatios",
    "VerifiedDownload",
    "assign_patient_splits",
    "audit_dataset",
    "extract_outer_zip",
    "finalize_download",
    "inspect_zip",
    "load_release_config",
    "load_manifest",
    "validate_patient_isolation",
    "verify_download",
]
