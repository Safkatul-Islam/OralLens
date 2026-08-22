from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    code: str
    detail: str
    request_id: str


class DetectionBoxResponse(BaseModel):
    box_xyxy: tuple[float, float, float, float]
    label: int = Field(ge=1)
    score: float = Field(ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str
    is_mock: bool
    model_name: str
    prediction_count: int = Field(ge=0)
    detections: list[DetectionBoxResponse] = Field(default_factory=list)


class InputAssessmentResponse(BaseModel):
    status: Literal["not_assessed", "supported", "unsupported"]
    reason_codes: list[str] = Field(default_factory=list)
    summary: str
    image_width: int | None = Field(default=None, ge=1)
    image_height: int | None = Field(default=None, ge=1)
    mean_luminance: float | None = Field(default=None, ge=0.0, le=1.0)
    luminance_stddev: float | None = Field(default=None, ge=0.0, le=1.0)


def default_input_assessment() -> InputAssessmentResponse:
    """Preserve readability of scan records created before assessment existed."""

    return InputAssessmentResponse(
        status="not_assessed",
        summary="This historical scan did not record an input assessment.",
    )


class EvidenceResponse(BaseModel):
    kind: str
    summary: str


class ReportResponse(BaseModel):
    title: str
    summary: str
    limitations: list[str]
    recommended_next_steps: list[str]
    disclaimer: str


class ScanRecord(BaseModel):
    id: str
    created_at: datetime
    original_filename: str
    content_type: str
    size_bytes: int = Field(ge=1)
    sha256: str
    prediction: PredictionResponse | None
    input_assessment: InputAssessmentResponse = Field(
        default_factory=default_input_assessment
    )
    evidence: EvidenceResponse
    report: ReportResponse


class ScanListResponse(BaseModel):
    scans: list[ScanRecord]
