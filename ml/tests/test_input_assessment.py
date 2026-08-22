from __future__ import annotations

from pathlib import Path

from PIL import Image
import pytest

from orallens_ml.inference.input_assessment import (
    InputAssessmentError,
    InputAssessmentPolicy,
    load_and_assess_image,
)


def policy(**changes: object) -> InputAssessmentPolicy:
    values: dict[str, object] = {
        "enabled": True,
        "min_short_side": 256,
        "max_pixels": 25_000_000,
        "min_mean_luminance": 0.05,
        "max_mean_luminance": 0.98,
        "min_luminance_stddev": 0.02,
    }
    values.update(changes)
    return InputAssessmentPolicy(**values)  # type: ignore[arg-type]


def save_split_image(path: Path, size: tuple[int, int] = (300, 300)) -> None:
    image = Image.new("L", size, color=0)
    image.paste(255, (size[0] // 2, 0, size[0], size[1]))
    image.convert("RGB").save(path)


def test_supported_image_is_orientation_normalized(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    image = Image.new("RGB", (300, 400), color=(20, 20, 20))
    image.paste((240, 240, 240), (150, 0, 300, 400))
    exif = Image.Exif()
    exif[274] = 6
    image.save(path, exif=exif)

    result = load_and_assess_image(path, policy=policy())

    assert result.assessment.status == "supported"
    assert (result.assessment.image_width, result.assessment.image_height) == (400, 300)
    assert result.image is not None
    assert result.image.size == (400, 300)


@pytest.mark.parametrize(
    ("color", "reason"),
    (
        (0, "image_too_dark"),
        (255, "image_too_bright"),
        (128, "image_low_contrast"),
    ),
)
def test_luminance_failures_abstain(tmp_path: Path, color: int, reason: str) -> None:
    path = tmp_path / f"{reason}.png"
    Image.new("L", (300, 300), color=color).save(path)

    result = load_and_assess_image(path, policy=policy())

    assert result.assessment.status == "unsupported"
    assert result.assessment.reason_codes == (reason,)
    assert result.image is None


def test_small_image_abstains(tmp_path: Path) -> None:
    path = tmp_path / "small.png"
    save_split_image(path, size=(255, 300))

    result = load_and_assess_image(path, policy=policy())

    assert "image_too_small" in result.assessment.reason_codes


def test_excessive_pixel_count_abstains_before_processing(tmp_path: Path) -> None:
    path = tmp_path / "large.png"
    save_split_image(path, size=(20, 20))

    result = load_and_assess_image(path, policy=policy(max_pixels=399, min_short_side=1))

    assert result.assessment.reason_codes == ("image_too_large",)
    assert result.assessment.mean_luminance is None


def test_corrupt_image_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"not a decodable image")

    with pytest.raises(InputAssessmentError, match="decoded safely"):
        load_and_assess_image(path, policy=policy())


def test_disabled_policy_preserves_safe_decode_without_quality_rejection(
    tmp_path: Path,
) -> None:
    path = tmp_path / "historical.png"
    Image.new("RGB", (8, 6), color=(0, 0, 0)).save(path)

    result = load_and_assess_image(path, policy=policy(enabled=False))

    assert result.assessment.status == "supported"
    assert result.assessment.mean_luminance is None
