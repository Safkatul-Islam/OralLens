"""CLI entry points for ML evaluation jobs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orallens_ml.evaluation.detection import (
    DetectionEvaluationError,
    load_detection_evaluation_config,
    run_detection_evaluation,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OralLens ML evaluation jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detection_parser = subparsers.add_parser("detection")
    detection_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "detection":
            config = load_detection_evaluation_config(args.config)
            result = run_detection_evaluation(config)
            payload = {
                "metrics": [
                    {
                        "f1": metric.f1,
                        "false_negatives": metric.false_negatives,
                        "false_positives": metric.false_positives,
                        "iou_threshold": metric.iou_threshold,
                        "mean_matched_iou": metric.mean_matched_iou,
                        "precision": metric.precision,
                        "prediction_count": metric.prediction_count,
                        "recall": metric.recall,
                        "score_threshold": metric.score_threshold,
                        "target_count": metric.target_count,
                        "true_positives": metric.true_positives,
                    }
                    for metric in result.metrics
                ],
                "metrics_path": str(result.metrics_path),
                "status": "detection-evaluated",
            }
        else:
            raise DetectionEvaluationError(f"Unknown command: {args.command}")
    except (DetectionEvaluationError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

