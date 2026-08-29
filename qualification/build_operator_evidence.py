#!/usr/bin/env python3
"""Convert a real operator-observation checklist into authoritative PASS JSON.

This script does not infer success from feature presence. Every frozen workstation check must be
explicitly marked `observed: true` and carry at least one evidence reference.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

FROZEN_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
REQUIRED_CHECKS = (
    "geo_tiff_metric_dsm",
    "nongeo_png_rdsm",
    "nongeo_jpg_rdsm",
    "optical_source_view",
    "dsm_view",
    "texture_projection_3d",
    "dsm_overlay_3d",
    "slope_overlay_3d",
    "hillshade_3d",
    "contours_3d",
    "orbit_navigation",
    "fly_navigation",
    "first_person_navigation",
    "top_down_navigation",
    "deterministic_flythrough",
    "auto_and_manual_lod",
    "vertical_exaggeration",
    "probe_3d",
    "measure_3d",
    "profile_3d",
    "structure_height_urban",
    "reference_validation_independent",
    "reference_view",
    "residual_view",
    "projection_accuracy_visual",
    "export_bundle",
    "relaunch_reopen",
)


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SystemExit("operator observation root must be an object")
    return payload


def _verify_file_reference(reference: str, base: Path) -> None:
    if not reference.startswith("file:"):
        return
    raw = reference[5:].split("#", 1)[0]
    candidate = Path(raw)
    path = candidate if candidate.is_absolute() else base / candidate
    if not path.exists():
        raise SystemExit(f"operator evidence file does not exist: {path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DepthWizard operator acceptance evidence.")
    parser.add_argument("observation", type=Path, help="YAML/JSON checklist populated during the real session")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve(strict=True)
    head = _git_head(repo)
    if head != FROZEN_HEAD:
        raise SystemExit(f"wrong git head: expected {FROZEN_HEAD}, got {head}")
    observation_path = args.observation.resolve(strict=True)
    payload = _load(observation_path)
    hardware = payload.get("finale_hardware")
    if not isinstance(hardware, str) or not hardware.strip():
        raise SystemExit("finale_hardware must be recorded")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise SystemExit("observation is missing checks")

    unknown = sorted(set(checks) - set(REQUIRED_CHECKS))
    missing = sorted(set(REQUIRED_CHECKS) - set(checks))
    if missing or unknown:
        raise SystemExit(f"operator checklist mismatch: missing={missing}, unknown={unknown}")

    output_checks: dict[str, dict[str, object]] = {}
    for name in REQUIRED_CHECKS:
        item = checks[name]
        if not isinstance(item, dict):
            raise SystemExit(f"{name}: checklist entry must be an object")
        if item.get("observed") is not True:
            raise SystemExit(f"{name}: must be explicitly observed=true after real workstation use")
        evidence = item.get("evidence")
        if not isinstance(evidence, list) or not evidence or not all(
            isinstance(entry, str) and entry.strip() for entry in evidence
        ):
            raise SystemExit(f"{name}: at least one explicit evidence reference is required")
        normalized = [entry.strip() for entry in evidence]
        for reference in normalized:
            _verify_file_reference(reference, observation_path.parent)
        notes = item.get("notes", "")
        if not isinstance(notes, str):
            raise SystemExit(f"{name}: notes must be a string")
        output_checks[name] = {"status": "PASS", "evidence": normalized, "notes": notes}

    report = {
        "schema_version": 1,
        "status": "PASS_SIH26175_OPERATOR_ACCEPTANCE",
        "git_head": FROZEN_HEAD,
        "finale_hardware": hardware.strip(),
        "session_evidence": payload.get("session_evidence", []),
        "checks": output_checks,
        "claim_boundary": (
            "PASS is emitted only from a fully populated observation checklist created during a real "
            "human-visible packaged DepthWizard workstation session."
        ),
    }
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "check_count": len(output_checks), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
