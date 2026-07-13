"""CLI entry points for ML training jobs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from orallens_ml.training.detection import (
    DetectionTrainingError,
    load_detection_training_config,
    run_detection_training,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run OralLens ML training jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detection_parser = subparsers.add_parser("detection-baseline")
    detection_parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "detection-baseline":
            config = load_detection_training_config(args.config)
            result = run_detection_training(config)
            payload = {
                "checkpoint_path": str(result.checkpoint_path),
                "metrics_path": str(result.metrics_path),
                "metrics": [
                    {
                        "epoch": metric.epoch,
                        "learning_rate": metric.learning_rate,
                        "train_loss": metric.train_loss,
                        "validation_loss": metric.validation_loss,
                    }
                    for metric in result.metrics
                ],
                "status": "detection-baseline-trained",
            }
        else:
            raise DetectionTrainingError(f"Unknown command: {args.command}")
    except (DetectionTrainingError, OSError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

