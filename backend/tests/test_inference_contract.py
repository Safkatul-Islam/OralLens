from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline.inference import (
    DetectionCandidate,
    InferenceAssessment,
    InferencePipelineError,
    InferenceOutcome,
    InferenceResult,
    MLDetectionInferencePipeline,
    MockInferencePipeline,
)


def test_mock_inference_returns_detection_contract_without_boxes() -> None:
    outcome = MockInferencePipeline().predict(b"image-bytes", "image/png")

    assert outcome.assessment.status == "not_assessed"
    assert outcome.prediction is not None
    response = outcome.prediction.to_response()

    assert response.is_mock is True
    assert response.model_name == "deterministic-mock-v1"
    assert response.prediction_count == 0
    assert response.detections == []
    assert 0.0 <= response.confidence <= 1.0


def test_model_backed_inference_result_serializes_detection_boxes() -> None:
    result = InferenceResult(
        label="possible_plaque",
        display_name="Possible plaque",
        confidence=0.73,
        severity="medium",
        evidence_summary="Detected one low-threshold plaque candidate.",
        model_name="orthodontic-plaque-mvp-v4-originals-online-aug-epoch9",
        is_mock=False,
        detections=(
            DetectionCandidate(
                box_xyxy=(1.0, 2.0, 5.0, 6.0),
                label=1,
                score=0.42,
            ),
        ),
    )

    response = result.to_response()

    assert response.is_mock is False
    assert response.model_name == "orthodontic-plaque-mvp-v4-originals-online-aug-epoch9"
    assert response.prediction_count == 1
    assert response.detections[0].box_xyxy == (1.0, 2.0, 5.0, 6.0)
    assert response.detections[0].label == 1
    assert response.detections[0].score == 0.42


def test_inference_outcome_rejects_prediction_for_unsupported_input() -> None:
    prediction = InferenceResult(
        label="possible_plaque",
        display_name="Possible plaque",
        confidence=0.5,
        severity="medium",
        evidence_summary="Candidate.",
        model_name="test-model",
        is_mock=False,
    )

    with pytest.raises(InferencePipelineError, match="cannot contain"):
        InferenceOutcome(
            assessment=InferenceAssessment(
                status="unsupported",
                reason_codes=("image_too_dark",),
            ),
            prediction=prediction,
        )


def test_inference_outcome_rejects_missing_supported_prediction() -> None:
    with pytest.raises(InferencePipelineError, match="requires"):
        InferenceOutcome(
            assessment=InferenceAssessment(status="supported"),
            prediction=None,
        )


class FakeMLDetectionInferencePipeline(MLDetectionInferencePipeline):
    def _load_config(self) -> object:
        return object()

    def _run_inference(self, config: object, image_path: Path) -> object:
        assert image_path.is_file()
        return SimpleNamespace(
            model_name="orthodontic-plaque-mvp-v4-originals-online-aug-epoch9",
            input_assessment=SimpleNamespace(
                status="supported",
                reason_codes=(),
                image_width=100,
                image_height=80,
                mean_luminance=0.5,
                luminance_stddev=0.2,
            ),
            predictions=(
                SimpleNamespace(
                    box_xyxy=(10.0, 20.0, 50.0, 60.0),
                    label=1,
                    score=0.31,
                ),
            )
        )


def test_ml_detection_pipeline_maps_predictions_and_removes_temp_file(
    tmp_path: Path,
) -> None:
    pipeline = FakeMLDetectionInferencePipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")
    assert outcome.assessment.status == "supported"
    assert outcome.prediction is not None
    response = outcome.prediction.to_response()

    assert response.is_mock is False
    assert response.model_name == "orthodontic-plaque-mvp-v4-originals-online-aug-epoch9"
    assert response.label == "possible_plaque"
    assert response.prediction_count == 1
    assert response.detections[0].box_xyxy == (10.0, 20.0, 50.0, 60.0)
    assert list((tmp_path / "ml-inputs").iterdir()) == []


def test_ml_detection_pipeline_rejects_webp_before_ml_runtime(tmp_path: Path) -> None:
    pipeline = FakeMLDetectionInferencePipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    with pytest.raises(InferencePipelineError, match="JPEG and PNG"):
        pipeline.predict(b"RIFF0000WEBP", "image/webp")


def test_ml_detection_pipeline_rejects_missing_model_name(tmp_path: Path) -> None:
    class MissingModelNamePipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, config: object, image_path: Path) -> object:
            assert image_path.is_file()
            return SimpleNamespace(predictions=())

    pipeline = MissingModelNamePipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    with pytest.raises(InferencePipelineError, match="model name"):
        pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert list((tmp_path / "ml-inputs").iterdir()) == []


def test_ml_detection_pipeline_returns_abstention_without_prediction(
    tmp_path: Path,
) -> None:
    class UnsupportedPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, config: object, image_path: Path) -> object:
            return SimpleNamespace(
                model_name="orthodontic-plaque-mvp-v4-originals-online-aug-epoch9",
                input_assessment=SimpleNamespace(
                    status="unsupported",
                    reason_codes=("image_too_dark",),
                    image_width=640,
                    image_height=480,
                    mean_luminance=0.01,
                    luminance_stddev=0.1,
                ),
                predictions=(),
            )

    pipeline = UnsupportedPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outcome.prediction is None
    assert outcome.assessment.status == "unsupported"
    assert outcome.assessment.reason_codes == ("image_too_dark",)
    assert "too dark" in outcome.assessment.summary


def test_ml_detection_pipeline_zero_boxes_is_not_a_plaque_free_finding(
    tmp_path: Path,
) -> None:
    class ZeroBoxPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, config: object, image_path: Path) -> object:
            return SimpleNamespace(
                model_name="orthodontic-plaque-mvp-v4-originals-online-aug-epoch9",
                input_assessment=SimpleNamespace(
                    status="supported",
                    reason_codes=(),
                    image_width=640,
                    image_height=480,
                    mean_luminance=0.5,
                    luminance_stddev=0.2,
                ),
                predictions=(),
            )

    pipeline = ZeroBoxPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outcome.prediction is not None
    assert outcome.prediction.label == "no_detection"
    assert outcome.prediction.display_name == "No candidate regions"
    assert "does not establish" in outcome.prediction.evidence_summary
    assert "plaque-free" in outcome.prediction.evidence_summary
