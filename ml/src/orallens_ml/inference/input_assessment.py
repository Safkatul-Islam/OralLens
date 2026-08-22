"""Conservative technical-quality assessment for inference images."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

InputAssessmentStatus = Literal["supported", "unsupported"]
InputAssessmentReason = Literal[
    "image_too_small",
    "image_too_large",
    "image_too_dark",
    "image_too_bright",
    "image_low_contrast",
]


class InputAssessmentError(ValueError):
    """Raised when an inference image cannot be decoded safely."""


@dataclass(frozen=True, slots=True)
class InputAssessmentPolicy:
    """Versioned technical thresholds applied before model inference."""

    enabled: bool
    min_short_side: int
    max_pixels: int
    min_mean_luminance: float
    max_mean_luminance: float
    min_luminance_stddev: float


@dataclass(frozen=True, slots=True)
class InputAssessment:
    """Technical-quality result for one safely decoded image."""

    status: InputAssessmentStatus
    reason_codes: tuple[InputAssessmentReason, ...]
    image_width: int
    image_height: int
    mean_luminance: float | None
    luminance_stddev: float | None

    @property
    def is_supported(self) -> bool:
        return self.status == "supported"


@dataclass(frozen=True, slots=True)
class AssessedImage:
    """An orientation-normalized RGB image and its assessment."""

    image: Image.Image | None
    assessment: InputAssessment


def load_and_assess_image(
    image_path: Path,
    *,
    policy: InputAssessmentPolicy,
) -> AssessedImage:
    """Decode one JPEG/PNG safely and apply the configured technical policy."""

    path = Path(image_path)
    if path.is_symlink() or not path.is_file():
        raise InputAssessmentError(f"Input image is not a regular file: {path}")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path, formats=("JPEG", "PNG")) as source:
                source_width, source_height = source.size
                if source_width * source_height > policy.max_pixels:
                    return AssessedImage(
                        image=None,
                        assessment=InputAssessment(
                            status="unsupported",
                            reason_codes=("image_too_large",),
                            image_width=source_width,
                            image_height=source_height,
                            mean_luminance=None,
                            luminance_stddev=None,
                        ),
                    )
                normalized = ImageOps.exif_transpose(source)
                normalized.load()
                rgb_image = normalized.convert("RGB")
                rgb_image.load()
    except (
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise InputAssessmentError("Input image could not be decoded safely.") from exc

    width, height = rgb_image.size
    if not policy.enabled:
        return AssessedImage(
            image=rgb_image,
            assessment=InputAssessment(
                status="supported",
                reason_codes=(),
                image_width=width,
                image_height=height,
                mean_luminance=None,
                luminance_stddev=None,
            ),
        )

    grayscale = rgb_image.convert("L")
    statistics = ImageStat.Stat(grayscale)
    mean_luminance = float(statistics.mean[0]) / 255.0
    luminance_stddev = float(statistics.stddev[0]) / 255.0
    reason_codes: list[InputAssessmentReason] = []

    if min(width, height) < policy.min_short_side:
        reason_codes.append("image_too_small")
    if mean_luminance < policy.min_mean_luminance:
        reason_codes.append("image_too_dark")
    elif mean_luminance > policy.max_mean_luminance:
        reason_codes.append("image_too_bright")
    elif luminance_stddev < policy.min_luminance_stddev:
        reason_codes.append("image_low_contrast")

    status: InputAssessmentStatus = "unsupported" if reason_codes else "supported"
    return AssessedImage(
        image=rgb_image if status == "supported" else None,
        assessment=InputAssessment(
            status=status,
            reason_codes=tuple(reason_codes),
            image_width=width,
            image_height=height,
            mean_luminance=mean_luminance,
            luminance_stddev=luminance_stddev,
        ),
    )
