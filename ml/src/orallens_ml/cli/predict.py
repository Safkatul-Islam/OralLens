"""CLI entry points for ML inference jobs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orallens_ml.inference.detection import (
    DetectionInferenceError,
    load_detection_inference_config,
    run_detection_inference,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OralLens ML inference jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detection_parser = subparsers.add_parser("detection")
    detection_parser.add_argument("--config", type=Path, required=True)
    detection_parser.add_argument("--image", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "detection":
            config = load_detection_inference_config(args.config)
            result = run_detection_inference(config, image_path=args.image)
            payload = {
                "image_height": result.image_height,
                "image_path": str(result.image_path),
                "image_width": result.image_width,
                "output_path": str(result.output_path),
                "input_assessment": {
                    "status": result.input_assessment.status,
                    "reason_codes": list(result.input_assessment.reason_codes),
                    "mean_luminance": result.input_assessment.mean_luminance,
                    "luminance_stddev": result.input_assessment.luminance_stddev,
                },
                "prediction_count": len(result.predictions),
                "predictions": [
                    {
                        "box_xyxy": list(prediction.box_xyxy),
                        "label": prediction.label,
                        "score": prediction.score,
                    }
                    for prediction in result.predictions
                ],
                "status": (
                    "detection-predicted"
                    if result.input_assessment.is_supported
                    else "input-unsupported"
                ),
            }
        else:
            raise DetectionInferenceError(f"Unknown command: {args.command}")
    except (DetectionInferenceError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
