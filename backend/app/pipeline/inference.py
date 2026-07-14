import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from app.schemas import DetectionBoxResponse, PredictionResponse

_ML_CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


class InferencePipelineError(RuntimeError):
    """Raised when the selected inference backend cannot produce a result."""


class UnsupportedInferenceInputError(InferencePipelineError):
    """Raised when the selected inference backend cannot process the upload type."""


@dataclass(frozen=True)
class DetectionCandidate:
    box_xyxy: tuple[float, float, float, float]
    label: int
    score: float

    def to_response(self) -> DetectionBoxResponse:
        return DetectionBoxResponse(
            box_xyxy=self.box_xyxy,
            label=self.label,
            score=self.score,
        )


@dataclass(frozen=True)
class InferenceResult:
    label: str
    display_name: str
    confidence: float
    severity: str
    evidence_summary: str
    model_name: str
    is_mock: bool
    detections: tuple[DetectionCandidate, ...] = ()

    def to_response(self) -> PredictionResponse:
        return PredictionResponse(
            label=self.label,
            display_name=self.display_name,
            confidence=self.confidence,
            severity=self.severity,
            is_mock=self.is_mock,
            model_name=self.model_name,
            prediction_count=len(self.detections),
            detections=[detection.to_response() for detection in self.detections],
        )


class InferencePipeline(Protocol):
    """Backend inference contract used by the scan service."""

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceResult:
        """Return screening-support predictions for validated image bytes."""


class MockInferencePipeline:
    """Deterministic placeholder until the real ML model is available."""

    _CLASSES: tuple[tuple[str, str, str, str], ...] = (
        (
            "no_obvious_issue",
            "No obvious issue",
            "low",
            "No localized visual evidence is available in mock mode.",
        ),
        (
            "possible_tartar_buildup",
            "Possible tartar buildup",
            "medium",
            "Mock mode selected a tartar-like finding from the image hash.",
        ),
        (
            "possible_gum_inflammation",
            "Possible gum inflammation",
            "medium",
            "Mock mode selected an inflammation-like finding from the image hash.",
        ),
    )

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceResult:
        digest = hashlib.sha256(image_bytes + content_type.encode("utf-8")).digest()
        class_index = digest[0] % len(self._CLASSES)
        confidence = 0.62 + ((digest[1] % 31) / 100)
        label, display_name, severity, evidence_summary = self._CLASSES[class_index]
        return InferenceResult(
            label=label,
            display_name=display_name,
            confidence=round(confidence, 2),
            severity=severity,
            evidence_summary=evidence_summary,
            model_name="deterministic-mock-v1",
            is_mock=True,
        )


class MLDetectionInferencePipeline:
    """Optional adapter for the checkpoint-backed orthodontic plaque detector."""

    def __init__(
        self,
        *,
        config_path: Path,
        ml_source_path: Path,
        temp_dir: Path,
    ) -> None:
        self._config_path = Path(config_path)
        self._ml_source_path = Path(ml_source_path)
        self._temp_dir = Path(temp_dir)

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceResult:
        suffix = _ML_CONTENT_TYPE_SUFFIXES.get(content_type)
        if suffix is None:
            raise UnsupportedInferenceInputError(
                "ML inference currently supports JPEG and PNG images."
            )

        input_path = self._write_temp_image(image_bytes, suffix)
        try:
            config = self._load_config()
            result = self._run_inference(config, input_path)
            return self._to_inference_result(result)
        except InferencePipelineError:
            raise
        except Exception as exc:
            raise InferencePipelineError("ML inference failed.") from exc
        finally:
            input_path.unlink(missing_ok=True)

    def _write_temp_image(self, image_bytes: bytes, suffix: str) -> Path:
        if self._temp_dir.is_symlink():
            raise InferencePipelineError("ML temp directory is a symlink.")
        self._temp_dir.mkdir(parents=True, exist_ok=True)
        root = self._temp_dir.resolve(strict=True)
        input_path = root / f"{uuid4()}{suffix}"
        resolved = input_path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise InferencePipelineError("ML temp image path escapes temp directory.")
        input_path.write_bytes(image_bytes)
        return input_path

    def _load_config(self) -> object:
        self._add_ml_source_path()
        from orallens_ml.inference.detection import load_detection_inference_config

        return load_detection_inference_config(self._config_path)

    def _run_inference(self, config: object, image_path: Path) -> object:
        self._add_ml_source_path()
        from orallens_ml.inference.detection import run_detection_inference

        return run_detection_inference(config, image_path=image_path)

    def _add_ml_source_path(self) -> None:
        if self._ml_source_path.is_symlink() or not self._ml_source_path.is_dir():
            raise InferencePipelineError("ML source path is not a regular directory.")
        source_path = str(self._ml_source_path.resolve(strict=True))
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    def _to_inference_result(self, result: object) -> InferenceResult:
        raw_predictions = getattr(result, "predictions", ())
        detections = tuple(
            DetectionCandidate(
                box_xyxy=tuple(float(value) for value in prediction.box_xyxy),
                label=int(prediction.label),
                score=float(prediction.score),
            )
            for prediction in raw_predictions
        )
        confidence = max((detection.score for detection in detections), default=0.0)
        if detections:
            label = "possible_plaque"
            display_name = "Possible plaque candidate"
            severity = "medium" if confidence >= 0.25 else "low"
            evidence_summary = (
                f"Detected {len(detections)} plaque candidate(s) with the MVP detector."
            )
        else:
            label = "no_detection"
            display_name = "No detection"
            severity = "low"
            evidence_summary = "The MVP detector returned no boxes above the configured threshold."

        return InferenceResult(
            label=label,
            display_name=display_name,
            confidence=confidence,
            severity=severity,
            evidence_summary=evidence_summary,
            model_name="orthodontic-plaque-mvp",
            is_mock=False,
            detections=detections,
        )
