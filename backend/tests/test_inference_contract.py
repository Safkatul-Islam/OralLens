from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from time import sleep
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

    def _load_runtime(self, config: object) -> object:
        return config

    def _run_inference(self, runtime: object, image_path: Path) -> object:
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
    pipeline.initialize()

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


def test_production_pipeline_removes_runtime_artifact(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "runtime-artifacts"

    class ArtifactPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, runtime: object, image_path: Path) -> object:
            result = super()._run_inference(runtime, image_path)
            output_path = artifact_dir / f"{image_path.stem}.json"
            output_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(**vars(result), output_path=output_path)

    pipeline = ArtifactPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
        runtime_artifact_dir=artifact_dir,
        cleanup_runtime_artifacts=True,
    )
    pipeline.initialize()

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outcome.prediction is not None
    assert list((tmp_path / "ml-inputs").iterdir()) == []
    assert list(artifact_dir.iterdir()) == []


def test_production_pipeline_refuses_to_remove_artifact_outside_root(
    tmp_path: Path,
) -> None:
    artifact_dir = tmp_path / "runtime-artifacts"
    outside_path = tmp_path / "outside.json"

    class EscapingArtifactPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, runtime: object, image_path: Path) -> object:
            result = super()._run_inference(runtime, image_path)
            outside_path.write_text("{}", encoding="utf-8")
            return SimpleNamespace(**vars(result), output_path=outside_path)

    pipeline = EscapingArtifactPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
        runtime_artifact_dir=artifact_dir,
        cleanup_runtime_artifacts=True,
    )
    pipeline.initialize()

    with pytest.raises(InferencePipelineError, match="escapes"):
        pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outside_path.read_text(encoding="utf-8") == "{}"
    assert list((tmp_path / "ml-inputs").iterdir()) == []


def test_ml_detection_pipeline_rejects_webp_before_ml_runtime(tmp_path: Path) -> None:
    pipeline = FakeMLDetectionInferencePipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )
    pipeline.initialize()

    with pytest.raises(InferencePipelineError, match="JPEG and PNG"):
        pipeline.predict(b"RIFF0000WEBP", "image/webp")


def test_ml_detection_pipeline_rejects_missing_model_name(tmp_path: Path) -> None:
    class MissingModelNamePipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, runtime: object, image_path: Path) -> object:
            assert image_path.is_file()
            return SimpleNamespace(predictions=())

    pipeline = MissingModelNamePipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )
    pipeline.initialize()

    with pytest.raises(InferencePipelineError, match="model name"):
        pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert list((tmp_path / "ml-inputs").iterdir()) == []


def test_ml_detection_pipeline_returns_abstention_without_prediction(
    tmp_path: Path,
) -> None:
    class UnsupportedPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, runtime: object, image_path: Path) -> object:
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
    pipeline.initialize()

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outcome.prediction is None
    assert outcome.assessment.status == "unsupported"
    assert outcome.assessment.reason_codes == ("image_too_dark",)
    assert "too dark" in outcome.assessment.summary


def test_ml_detection_pipeline_zero_boxes_is_not_a_plaque_free_finding(
    tmp_path: Path,
) -> None:
    class ZeroBoxPipeline(FakeMLDetectionInferencePipeline):
        def _run_inference(self, runtime: object, image_path: Path) -> object:
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
    pipeline.initialize()

    outcome = pipeline.predict(b"\x89PNG\r\n\x1a\nimage-bytes", "image/png")

    assert outcome.prediction is not None
    assert outcome.prediction.label == "no_detection"
    assert outcome.prediction.display_name == "No candidate regions"
    assert "does not establish" in outcome.prediction.evidence_summary
    assert "plaque-free" in outcome.prediction.evidence_summary


def test_ml_detection_pipeline_initializes_runtime_once(tmp_path: Path) -> None:
    class CountingPipeline(FakeMLDetectionInferencePipeline):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.config_loads = 0
            self.runtime_loads = 0

        def _load_config(self) -> object:
            self.config_loads += 1
            return object()

        def _load_runtime(self, config: object) -> object:
            self.runtime_loads += 1
            return config

    pipeline = CountingPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
    )

    assert pipeline.is_ready is False
    pipeline.initialize()
    pipeline.initialize()
    pipeline.predict(b"\x89PNG\r\n\x1a\nfirst", "image/png")
    pipeline.predict(b"\x89PNG\r\n\x1a\nsecond", "image/png")

    assert pipeline.is_ready is True
    assert pipeline.config_loads == 1
    assert pipeline.runtime_loads == 1


def test_pipeline_deletes_checkpoint_only_after_successful_initialization(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    checkpoint_path.write_bytes(b"frozen checkpoint")

    class DeletingPipeline(FakeMLDetectionInferencePipeline):
        def _load_config(self) -> object:
            return SimpleNamespace(checkpoint_path=checkpoint_path)

        def _load_runtime(self, config: object) -> object:
            assert checkpoint_path.is_file()
            return config

    pipeline = DeletingPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
        project_root=tmp_path,
        delete_checkpoint_after_load=True,
    )

    pipeline.initialize()

    assert pipeline.is_ready is True
    assert not checkpoint_path.exists()


def test_pipeline_preserves_checkpoint_when_initialization_fails(
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    checkpoint_path.write_bytes(b"frozen checkpoint")

    class FailingPipeline(FakeMLDetectionInferencePipeline):
        def _load_config(self) -> object:
            return SimpleNamespace(checkpoint_path=checkpoint_path)

        def _load_runtime(self, config: object) -> object:
            raise RuntimeError("model load failed")

    pipeline = FailingPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
        project_root=tmp_path,
        delete_checkpoint_after_load=True,
    )

    with pytest.raises(InferencePipelineError, match="initialization failed"):
        pipeline.initialize()

    assert pipeline.is_ready is False
    assert checkpoint_path.is_file()


def test_pipeline_refuses_to_delete_checkpoint_outside_project_root(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    checkpoint_path = tmp_path / "checkpoint_best.pt"
    checkpoint_path.write_bytes(b"frozen checkpoint")

    class EscapingCheckpointPipeline(FakeMLDetectionInferencePipeline):
        def _load_config(self) -> object:
            return SimpleNamespace(checkpoint_path=checkpoint_path)

    pipeline = EscapingCheckpointPipeline(
        config_path=project_root / "predict.toml",
        ml_source_path=project_root,
        temp_dir=project_root / "ml-inputs",
        project_root=project_root,
        delete_checkpoint_after_load=True,
    )

    with pytest.raises(InferencePipelineError, match="escapes"):
        pipeline.initialize()

    assert pipeline.is_ready is False
    assert checkpoint_path.is_file()


def test_ml_detection_pipeline_bounds_concurrent_model_calls(tmp_path: Path) -> None:
    class ConcurrencyPipeline(FakeMLDetectionInferencePipeline):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            self.active_calls = 0
            self.maximum_active_calls = 0
            self._counter_lock = Lock()

        def _run_inference(self, runtime: object, image_path: Path) -> object:
            with self._counter_lock:
                self.active_calls += 1
                self.maximum_active_calls = max(
                    self.maximum_active_calls,
                    self.active_calls,
                )
            try:
                sleep(0.05)
                return super()._run_inference(runtime, image_path)
            finally:
                with self._counter_lock:
                    self.active_calls -= 1

    pipeline = ConcurrencyPipeline(
        config_path=tmp_path / "predict.toml",
        ml_source_path=tmp_path,
        temp_dir=tmp_path / "ml-inputs",
        max_concurrent_inferences=1,
    )
    pipeline.initialize()
    barrier = Barrier(2)

    def predict(index: int) -> InferenceOutcome:
        barrier.wait()
        return pipeline.predict(
            b"\x89PNG\r\n\x1a\n" + bytes([index]),
            "image/png",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(predict, range(2)))

    assert len(outcomes) == 2
    assert pipeline.maximum_active_calls == 1
