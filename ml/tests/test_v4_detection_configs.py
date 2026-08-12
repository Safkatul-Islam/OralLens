from __future__ import annotations

from pathlib import Path

from orallens_ml.evaluation.detection import load_detection_evaluation_config
from orallens_ml.training.detection import load_detection_training_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]
V4_MANIFEST = Path(
    "ml/data/prepared/orthodontic_plaque/v4/part-2/manifest.csv"
)


def test_v4_training_config_is_fresh_controlled_full_split_run() -> None:
    config = load_detection_training_config(
        PROJECT_ROOT / "ml/configs/orthodontic_plaque_detection_mvp_v4.toml"
    )

    assert config.manifest_path == V4_MANIFEST
    assert config.output_dir == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4"
    )
    assert config.epochs == 3
    assert config.batch_size == 1
    assert config.image_min_size == 512
    assert config.image_max_size == 768
    assert config.learning_rate == 0.005
    assert config.seed == 20260711
    assert config.pretrained_weights == "default"
    assert config.max_train_batches is None
    assert config.max_validation_batches is None
    assert config.resume_checkpoint_path is None


def test_v4_smoke_config_is_isolated_and_strictly_capped() -> None:
    config = load_detection_training_config(
        PROJECT_ROOT / "ml/configs/orthodontic_plaque_detection_mvp_v4_smoke.toml"
    )

    assert config.manifest_path == V4_MANIFEST
    assert config.output_dir == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4_smoke"
    )
    assert config.epochs == 1
    assert config.pretrained_weights == "none"
    assert config.max_train_batches == 1
    assert config.max_validation_batches == 1
    assert config.resume_checkpoint_path is None


def test_v4_resume_config_is_contained_and_preserves_experiment_contract() -> None:
    config = load_detection_training_config(
        PROJECT_ROOT / "ml/configs/orthodontic_plaque_detection_mvp_v4_resume.toml"
    )

    assert config.manifest_path == V4_MANIFEST
    assert config.output_dir == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4"
    )
    assert config.resume_checkpoint_path == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4/checkpoint_last.pt"
    )
    assert config.epochs == 3
    assert config.pretrained_weights == "default"
    assert config.max_train_batches is None
    assert config.max_validation_batches is None


def test_v4_evaluation_config_is_complete_validation_only_sweep() -> None:
    config = load_detection_evaluation_config(
        PROJECT_ROOT / "ml/configs/orthodontic_plaque_detection_mvp_v4_eval.toml"
    )

    assert config.manifest_path == V4_MANIFEST
    assert config.checkpoint_path == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4/checkpoint_best.pt"
    )
    assert config.output_dir == Path(
        "ml/runs/detection/orthodontic_plaque_part2_mvp_v4_eval"
    )
    assert config.split == "validation"
    assert config.max_batches is None
    assert config.iou_thresholds == (0.5,)
    assert config.score_thresholds[0] == 0.05
    assert config.score_thresholds[-1] == 0.95
