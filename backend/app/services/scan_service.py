import hashlib
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile, status

from app.config import Settings
from app.pipeline.inference import (
    InferencePipeline,
    InferenceResult,
    UnsupportedInferenceInputError,
)
from app.schemas import EvidenceResponse, ReportResponse, ScanRecord
from app.storage import JSONScanStore

READ_CHUNK_BYTES = 1024 * 1024


class UploadValidationError(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class ScanService:
    """Coordinates upload validation, inference, report creation, and storage."""

    def __init__(
        self,
        settings: Settings,
        store: JSONScanStore,
        inference_pipeline: InferencePipeline,
    ) -> None:
        self._settings = settings
        self._store = store
        self._inference_pipeline = inference_pipeline

    async def create_scan(self, file: UploadFile) -> ScanRecord:
        content = await self._read_and_validate(file)
        digest = hashlib.sha256(content).hexdigest()
        try:
            prediction = self._inference_pipeline.predict(content, file.content_type or "")
        except UnsupportedInferenceInputError as exc:
            raise UploadValidationError(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, str(exc)) from exc

        evidence = EvidenceResponse(
            kind="summary",
            summary=prediction.evidence_summary,
        )
        report = self._build_report(prediction)

        record = ScanRecord(
            id=str(uuid4()),
            created_at=datetime.now(UTC),
            original_filename=self._safe_display_filename(file.filename),
            content_type=file.content_type or "application/octet-stream",
            size_bytes=len(content),
            sha256=digest,
            prediction=prediction.to_response(),
            evidence=evidence,
            report=report,
        )
        return self._store.save(record)

    def list_scans(self) -> list[ScanRecord]:
        return self._store.list()

    def get_scan(self, scan_id: str) -> ScanRecord | None:
        return self._store.get(scan_id)

    async def _read_and_validate(self, file: UploadFile) -> bytes:
        self._validate_metadata(file)

        chunks: list[bytes] = []
        total_bytes = 0
        while chunk := await file.read(READ_CHUNK_BYTES):
            chunks.append(chunk)
            total_bytes += len(chunk)
            if total_bytes > self._settings.max_upload_bytes:
                raise UploadValidationError(
                    status.HTTP_413_CONTENT_TOO_LARGE,
                    "Uploaded image is larger than the configured limit.",
                )

        content = b"".join(chunks)
        if not content:
            raise UploadValidationError(
                status.HTTP_400_BAD_REQUEST,
                "Uploaded image is empty.",
            )

        self._validate_file_signature(content, file.content_type or "")
        return content

    def _validate_metadata(self, file: UploadFile) -> None:
        content_type = file.content_type or ""
        if content_type not in self._settings.allowed_content_types:
            raise UploadValidationError(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "Unsupported file type. Upload a JPEG, PNG, or WebP image.",
            )

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in self._settings.allowed_extensions:
            raise UploadValidationError(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                "Unsupported file extension. Upload a .jpg, .jpeg, .png, or .webp file.",
            )

    def _validate_file_signature(self, content: bytes, content_type: str) -> None:
        if content_type == "image/jpeg" and content.startswith(b"\xff\xd8\xff"):
            return
        if content_type == "image/png" and content.startswith(b"\x89PNG\r\n\x1a\n"):
            return
        if (
            content_type == "image/webp"
            and len(content) >= 12
            and content[:4] == b"RIFF"
            and content[8:12] == b"WEBP"
        ):
            return

        raise UploadValidationError(
            status.HTTP_400_BAD_REQUEST,
            "File content does not match the declared image type.",
        )

    def _safe_display_filename(self, filename: str | None) -> str:
        safe_name = Path(filename or "uploaded-image").name.strip()
        return safe_name or "uploaded-image"

    def _build_report(self, prediction: InferenceResult) -> ReportResponse:
        score_label = "mock score" if prediction.is_mock else "maximum detector score"
        formatted_score = f"{prediction.confidence:.3f}"
        mode = "mock pipeline" if prediction.is_mock else "model pipeline"
        return ReportResponse(
            title="AI Screening Support Report",
            summary=(
                f"The current {mode} marked this image as "
                f"'{prediction.display_name}' with a {score_label} of {formatted_score}."
            ),
            limitations=[
                "This backend response is screening-support output, not a diagnosis.",
                "The current MVP model is not clinically validated.",
                "Image quality, lighting, angle, and framing can strongly affect screening quality.",
                "Displayed scores are experimental ranking values, not calibrated clinical probabilities.",
                "The result is not a dental diagnosis or treatment recommendation.",
            ],
            recommended_next_steps=[
                "Review the uploaded image and the displayed score's limitations.",
                "Review any returned detection boxes as experimental model evidence.",
                "Consult a licensed dental professional for real symptoms or concerns.",
            ],
            disclaimer=(
                "OralLens AI is a learning project for screening support. "
                "It is not a medical device and does not provide diagnosis."
            ),
        )
