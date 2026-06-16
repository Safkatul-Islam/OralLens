import hashlib
from dataclasses import dataclass

from app.schemas import PredictionResponse


@dataclass(frozen=True)
class InferenceResult:
    label: str
    display_name: str
    confidence: float
    severity: str
    evidence_summary: str

    def to_response(self) -> PredictionResponse:
        return PredictionResponse(
            label=self.label,
            display_name=self.display_name,
            confidence=self.confidence,
            severity=self.severity,
            is_mock=True,
        )


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
        )

