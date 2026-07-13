from __future__ import annotations

from pathlib import Path

import pytest
import torch

from orallens_ml.modeling.detection import (
    DetectionModelConfig,
    DetectionModelError,
    WeightsName,
    pretrained_weights_cache_path,
    require_pretrained_weights_available,
)


def config_for(pretrained_weights: WeightsName) -> DetectionModelConfig:
    return DetectionModelConfig(
        num_classes=2,
        image_min_size=64,
        image_max_size=128,
        trainable_backbone_layers=0,
        pretrained_weights=pretrained_weights,
    )


def test_pretrained_weights_cache_path_is_none_for_uninitialized_model() -> None:
    assert pretrained_weights_cache_path(config_for("none")) is None
    require_pretrained_weights_available(config_for("none"))


def test_pretrained_weights_cache_path_uses_torch_hub_checkpoint_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "torch-hub"))

    cache_path = pretrained_weights_cache_path(config_for("default"))

    assert cache_path is not None
    assert cache_path.parent == tmp_path / "torch-hub" / "checkpoints"
    assert cache_path.name.endswith(".pth")


def test_require_pretrained_weights_available_rejects_missing_default_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "torch-hub"))

    with pytest.raises(DetectionModelError, match="does not download pretrained weights"):
        require_pretrained_weights_available(config_for("default"))


def test_require_pretrained_weights_available_accepts_cached_default_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "torch-hub"))
    cache_path = pretrained_weights_cache_path(config_for("default"))
    assert cache_path is not None
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"placeholder")

    require_pretrained_weights_available(config_for("default"))
