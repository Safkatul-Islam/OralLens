from datetime import datetime

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str


class ErrorResponse(BaseModel):
    detail: str


class PredictionResponse(BaseModel):
    label: str
    display_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    severity: str
    is_mock: bool


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
    prediction: PredictionResponse
    evidence: EvidenceResponse
    report: ReportResponse


class ScanListResponse(BaseModel):
    scans: list[ScanRecord]

