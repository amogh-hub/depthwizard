from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthwizard.evaluation.final_campaign import evaluate_final_science_campaign


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen DepthWizard production predictions on the geographically "
            "disjoint four-terrain and cross-sensor final science campaign."
        )
    )
    parser.add_argument("registry", type=Path, help="Frozen DepthWizard dataset registry YAML")
    parser.add_argument(
        "predictions",
        type=Path,
        help="Campaign prediction manifest YAML/JSON",
    )
    parser.add_argument("output_dir", type=Path, help="Evidence output directory")
    parser.add_argument(
        "--min-valid-pixels",
        type=int,
        default=128,
        help="Minimum independent prediction/reference pixels required per scene (default: 128)",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = evaluate_final_science_campaign(
        args.registry,
        args.predictions,
        args.output_dir,
        min_valid_pixels=args.min_valid_pixels,
    )
    summary = {
        "protocol": report["protocol"],
        "model": report["model"],
        "test_overall": report["test_overall"],
        "cross_sensor_overall": report["cross_sensor_overall"],
        "terrain": report["terrain"],
        "artifacts": report["artifacts"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
