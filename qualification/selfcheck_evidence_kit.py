#!/usr/bin/env python3
"""Offline syntax/structure self-check for the DepthWizard operations-only evidence kit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml

REQUIRED = (
    "bootstrap_evidence_kit.sh",
    "run_four_terrain_science_from_frozen_head.sh",
    "run_two_hour_soak_from_frozen_head.sh",
    "stage_manual_evidence_from_frozen_head.sh",
    "download_neon_eval_tiles.py",
    "build_final_science_registry.py",
    "prepare_metric_predictions.py",
    "build_operator_evidence.py",
    "build_rendering_evidence.py",
    "build_clean_machine_evidence.py",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate evidence-kit syntax without network or model execution.")
    parser.add_argument("--kit-root", type=Path, default=Path(__file__).resolve().parent)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.kit_root.resolve(strict=True)
    missing = [name for name in REQUIRED if not (root / name).is_file()]
    if missing:
        raise SystemExit(f"missing required evidence-kit files: {missing}")

    python_files = sorted(root.glob("*.py"))
    for path in python_files:
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")

    shell_files = sorted(root.glob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)

    template_root = root / "templates"
    for path in sorted(template_root.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(template_root.glob("*.yaml")):
        yaml.safe_load(path.read_text(encoding="utf-8"))

    print(
        json.dumps(
            {
                "status": "PASS_EVIDENCE_KIT_OFFLINE_SELFCHECK",
                "python_files": len(python_files),
                "shell_files": len(shell_files),
                "template_json": len(list(template_root.glob("*.json"))),
                "template_yaml": len(list(template_root.glob("*.yaml"))),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
