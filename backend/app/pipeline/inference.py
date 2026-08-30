import hashlib
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from threading import BoundedSemaphore, Lock
from typing import Literal, Protocol
from uuid import uuid4

from app.schemas import (
    DetectionBoxResponse,
    InputAssessmentResponse,
    PredictionResponse,
)

_ML_CONTENT_TYPE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


class InferencePipelineError(RuntimeError):
    """Raised when the selected inference backend cannot produce a result."""


class UnsupportedInferenceInputError(InferencePipelineError):
    """Raised when the selected inference backend cannot process the upload type."""


class InvalidInferenceInputError(InferencePipelineError):
    """Raised when supported-format bytes cannot be decoded safely."""


AssessmentStatus = Literal["not_assessed", "supported", "unsupported"]
_ASSESSMENT_MESSAGES = {
    "image_too_small": "The image resolution is too small for this screening pipeline.",
    "image_too_large": "The decoded image is too large to process safely.",
    "image_too_dark": "The image is too dark for reliable visual screening.",
    "image_too_bright": "The image is too bright for reliable visual screening.",
    "image_low_contrast": "The image has too little visual contrast for reliable screening.",
}


@dataclass(frozen=True)
class InferenceAssessment:
    status: AssessmentStatus
    reason_codes: tuple[str, ...] = ()
    image_width: int | None = None
    image_height: int | None = None
    mean_luminance: float | None = None
    luminance_stddev: float | None = None

    def __post_init__(self) -> None:
        if any(code not in _ASSESSMENT_MESSAGES for code in self.reason_codes):
            raise InferencePipelineError("Input assessment has an unknown reason code.")
        if self.status == "unsupported" and not self.reason_codes:
            raise InferencePipelineError("Unsupported input requires a rejection reason.")
        if self.status != "unsupported" and self.reason_codes:
            raise InferencePipelineError(
                "Only unsupported input may contain rejection reasons."
            )

    @property
    def summary(self) -> str:
        if self.status == "not_assessed":
            return "This pipeline did not perform an input-quality assessment."
        if self.status == "supported":
            return "The image passed the configured technical-quality checks."
        return " ".join(_ASSESSMENT_MESSAGES[code] for code in self.reason_codes)

    def to_response(self) -> InputAssessmentResponse:
        return InputAssessmentResponse(
            status=self.status,
            reason_codes=list(self.reason_codes),
            summary=self.summary,
            image_width=self.image_width,
            image_height=self.image_height,
            mean_luminance=self.mean_luminance,
            luminance_stddev=self.luminance_stddev,
        )


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


@dataclass(frozen=True)
class InferenceOutcome:
    assessment: InferenceAssessment
    prediction: InferenceResult | None

    def __post_init__(self) -> None:
        if self.assessment.status == "unsupported" and self.prediction is not None:
            raise InferencePipelineError(
                "Unsupported input cannot contain a condition prediction."
            )
        if self.assessment.status != "unsupported" and self.prediction is None:
            raise InferencePipelineError(
                "Supported or unassessed input requires a condition prediction."
            )


class InferencePipeline(Protocol):
    """Backend inference contract used by the scan service."""

    @property
    def is_ready(self) -> bool:
        """Return whether startup initialization completed successfully."""

    def initialize(self) -> None:
        """Initialize reusable resources required for inference."""

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
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

    def __init__(self) -> None:
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def initialize(self) -> None:
        self._is_ready = True

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
        digest = hashlib.sha256(image_bytes + content_type.encode("utf-8")).digest()
        class_index = digest[0] % len(self._CLASSES)
        confidence = 0.62 + ((digest[1] % 31) / 100)
        label, display_name, severity, evidence_summary = self._CLASSES[class_index]
        return InferenceOutcome(
            assessment=InferenceAssessment(status="not_assessed"),
            prediction=InferenceResult(
                label=label,
                display_name=display_name,
                confidence=round(confidence, 2),
                severity=severity,
                evidence_summary=evidence_summary,
                model_name="deterministic-mock-v1",
                is_mock=True,
            ),
        )


class MLDetectionInferencePipeline:
    """Optional adapter for the checkpoint-backed orthodontic plaque detector."""

    def __init__(
        self,
        *,
        config_path: Path,
        ml_source_path: Path,
        temp_dir: Path,
        max_concurrent_inferences: int = 1,
        project_root: Path | None = None,
        runtime_artifact_dir: Path | None = None,
        cleanup_runtime_artifacts: bool = False,
        delete_checkpoint_after_load: bool = False,
    ) -> None:
        if max_concurrent_inferences < 1:
            raise InferencePipelineError(
                "max_concurrent_inferences must be at least one."
            )
        self._config_path = Path(config_path)
        self._ml_source_path = Path(ml_source_path)
        self._temp_dir = Path(temp_dir)
        self._project_root = Path(project_root) if project_root is not None else None
        self._runtime_artifact_dir = (
            Path(runtime_artifact_dir) if runtime_artifact_dir is not None else None
        )
        self._cleanup_runtime_artifacts = cleanup_runtime_artifacts
        self._delete_checkpoint_after_load = delete_checkpoint_after_load
        if cleanup_runtime_artifacts and runtime_artifact_dir is None:
            raise InferencePipelineError(
                "Runtime artifact cleanup requires an explicit artifact directory."
            )
        if delete_checkpoint_after_load and project_root is None:
            raise InferencePipelineError(
                "Checkpoint deletion requires an explicit project root."
            )
        self._temp_root: Path | None = None
        self._runtime_artifact_root: Path | None = None
        self._runtime: object | None = None
        self._initialization_lock = Lock()
        self._inference_slots = BoundedSemaphore(max_concurrent_inferences)

    @property
    def is_ready(self) -> bool:
        return self._runtime is not None

    def initialize(self) -> None:
        """Load and validate the frozen model exactly once."""

        if self._runtime is not None:
            return
        with self._initialization_lock:
            if self._runtime is not None:
                return
            try:
                self._temp_root = self._prepare_runtime_directory(
                    self._temp_dir,
                    label="ML temp",
                )
                if self._runtime_artifact_dir is not None:
                    self._runtime_artifact_root = self._prepare_runtime_directory(
                        self._runtime_artifact_dir,
                        label="ML runtime artifact",
                    )
                config = self._load_config()
                runtime = self._load_runtime(config)
                if self._delete_checkpoint_after_load:
                    self._remove_loaded_checkpoint(config)
            except InferencePipelineError:
                raise
            except Exception as exc:
                raise InferencePipelineError(
                    "ML inference initialization failed."
                ) from exc
            self._runtime = runtime

    def _remove_loaded_checkpoint(self, config: object) -> None:
        checkpoint_value = getattr(config, "checkpoint_path", None)
        if not isinstance(checkpoint_value, Path):
            raise InferencePipelineError(
                "Loaded ML config is missing its checkpoint path."
            )
        project_root = self._project_root
        if project_root is None:
            raise InferencePipelineError(
                "Checkpoint deletion requires an explicit project root."
            )
        if project_root.is_symlink() or not project_root.is_dir():
            raise InferencePipelineError("Project root is not a regular directory.")
        if checkpoint_value.is_symlink() or not checkpoint_value.is_file():
            raise InferencePipelineError(
                "Loaded checkpoint is not a regular file."
            )
        resolved_root = project_root.resolve(strict=True)
        resolved_checkpoint = checkpoint_value.resolve(strict=True)
        if not resolved_checkpoint.is_relative_to(resolved_root):
            raise InferencePipelineError(
                "Loaded checkpoint path escapes the project root."
            )
        resolved_checkpoint.unlink()

    def predict(self, image_bytes: bytes, content_type: str) -> InferenceOutcome:
        suffix = _ML_CONTENT_TYPE_SUFFIXES.get(content_type)
        if suffix is None:
            raise UnsupportedInferenceInputError(
                "ML inference currently supports JPEG and PNG images."
            )
        runtime = self._runtime
        if runtime is None:
            raise InferencePipelineError("ML inference pipeline is not ready.")

        input_path = self._write_temp_image(image_bytes, suffix)
        result: object | None = None
        try:
            with self._inference_slots:
                result = self._run_inference(runtime, input_path)
            return self._to_inference_outcome(result)
        except InferencePipelineError:
            raise
        except Exception as exc:
            raise InferencePipelineError("ML inference failed.") from exc
        finally:
            input_path.unlink(missing_ok=True)
            if self._cleanup_runtime_artifacts and result is not None:
                self._remove_runtime_artifact(result)

    def _write_temp_image(self, image_bytes: bytes, suffix: str) -> Path:
        root = self._temp_root
        if root is None:
            raise InferencePipelineError("ML temp directory is not initialized.")
        input_path = root / f"{uuid4()}{suffix}"
        resolved = input_path.resolve(strict=False)
        if not resolved.is_relative_to(root):
            raise InferencePipelineError("ML temp image path escapes temp directory.")
        input_path.write_bytes(image_bytes)
        return input_path

    def _prepare_runtime_directory(self, path: Path, *, label: str) -> Path:
        if path.is_symlink():
            raise InferencePipelineError(f"{label} directory is a symlink.")
        path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            raise InferencePipelineError(f"{label} path is not a directory.")
        return path.resolve(strict=True)

    def _remove_runtime_artifact(self, result: object) -> None:
        root = self._runtime_artifact_root
        output_value = getattr(result, "output_path", None)
        if root is None or not isinstance(output_value, Path):
            raise InferencePipelineError(
                "ML inference result is missing its runtime artifact path."
            )
        if output_value.is_symlink():
            raise InferencePipelineError("ML runtime artifact is a symlink.")
        resolved_output = output_value.resolve(strict=False)
        if not resolved_output.is_relative_to(root):
            raise InferencePipelineError(
                "ML runtime artifact path escapes its configured directory."
            )
        if resolved_output.exists() and not resolved_output.is_file():
            raise InferencePipelineError("ML runtime artifact is not a regular file.")
        resolved_output.unlink(missing_ok=True)

    def _load_config(self) -> object:
        self._add_ml_source_path()
        from orallens_ml.inference.detection import load_detection_inference_config

        config = load_detection_inference_config(
            self._config_path,
            path_base=self._project_root,
        )
        if self._runtime_artifact_root is not None:
            config = replace(config, output_dir=self._runtime_artifact_root)
        return config

    def _load_runtime(self, config: object) -> object:
        self._add_ml_source_path()
        from orallens_ml.inference.detection import DetectionInferenceRuntime

        return DetectionInferenceRuntime(config)

    def _run_inference(self, runtime: object, image_path: Path) -> object:
        self._add_ml_source_path()
        from orallens_ml.inference.detection import InvalidDetectionImageError

        try:
            return runtime.predict(image_path=image_path)
        except InvalidDetectionImageError as exc:
            raise InvalidInferenceInputError(str(exc)) from exc

    def _add_ml_source_path(self) -> None:
        if self._ml_source_path.is_symlink() or not self._ml_source_path.is_dir():
            raise InferencePipelineError("ML source path is not a regular directory.")
        source_path = str(self._ml_source_path.resolve(strict=True))
        if source_path not in sys.path:
            sys.path.insert(0, source_path)

    def _to_inference_outcome(self, result: object) -> InferenceOutcome:
        model_name = getattr(result, "model_name", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise InferencePipelineError("ML inference result is missing a model name.")
        raw_predictions = getattr(result, "predictions", ())
        assessment = self._to_assessment(getattr(result, "input_assessment", None))
        detections = tuple(
            DetectionCandidate(
                box_xyxy=tuple(float(value) for value in prediction.box_xyxy),
                label=int(prediction.label),
                score=float(prediction.score),
            )
            for prediction in raw_predictions
        )
        if assessment.status == "unsupported":
            if detections:
                raise InferencePipelineError(
                    "Unsupported ML input returned condition detections."
                )
            return InferenceOutcome(assessment=assessment, prediction=None)
        confidence = max((detection.score for detection in detections), default=0.0)
        if detections:
            label = "possible_plaque"
            display_name = "Plaque-positive peri-tooth region candidate"
            severity = "medium" if confidence >= 0.25 else "low"
            evidence_summary = (
                f"Detected {len(detections)} plaque-positive peri-tooth region "
                "candidate(s) with the detector."
            )
        else:
            label = "no_detection"
            display_name = "No candidate regions"
            severity = "low"
            evidence_summary = (
                "The MVP detector returned no candidate regions above the configured "
                "threshold. This does not establish that the image is plaque-free."
            )

        return InferenceOutcome(
            assessment=assessment,
            prediction=InferenceResult(
                label=label,
                display_name=display_name,
                confidence=confidence,
                severity=severity,
                evidence_summary=evidence_summary,
                model_name=model_name,
                is_mock=False,
                detections=detections,
            ),
        )

    def _to_assessment(self, assessment: object) -> InferenceAssessment:
        status = getattr(assessment, "status", None)
        if status not in {"supported", "unsupported"}:
            raise InferencePipelineError("ML inference result has an invalid input assessment.")
        raw_reason_codes = getattr(assessment, "reason_codes", ())
        if not isinstance(raw_reason_codes, tuple):
            raise InferencePipelineError("ML input assessment reasons must be a tuple.")
        if any(code not in _ASSESSMENT_MESSAGES for code in raw_reason_codes):
            raise InferencePipelineError("ML input assessment has an unknown reason code.")
        if status == "supported" and raw_reason_codes:
            raise InferencePipelineError("Supported ML input cannot contain rejection reasons.")
        if status == "unsupported" and not raw_reason_codes:
            raise InferencePipelineError("Unsupported ML input requires a rejection reason.")
        width = getattr(assessment, "image_width", None)
        height = getattr(assessment, "image_height", None)
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            raise InferencePipelineError("ML input assessment has invalid dimensions.")
        mean = getattr(assessment, "mean_luminance", None)
        stddev = getattr(assessment, "luminance_stddev", None)
        for name, value in (("mean luminance", mean), ("luminance standard deviation", stddev)):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise InferencePipelineError(f"ML input assessment has invalid {name}.")
        return InferenceAssessment(
            status=status,
            reason_codes=raw_reason_codes,
            image_width=width,
            image_height=height,
            mean_luminance=None if mean is None else float(mean),
            luminance_stddev=None if stddev is None else float(stddev),
        )
