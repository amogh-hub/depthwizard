#!/usr/bin/env python3
"""Build sustained 3D rendering evidence from real one-second telemetry samples.

CSV columns:
  elapsed_seconds,fps,camera_position,rendering_mode,navigating

Accepted camera_position values include at least: aerial, low
Accepted rendering_mode values include at least: texture, analytical_overlay
The script emits PASS only when the corrected-release checker thresholds are truly met.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path

import numpy as np

QUALIFIED_HEAD = "2ec539aea010b974a2781b240fe76c67a99d06a8"


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute DepthWizard sustained-rendering PASS evidence.")
    parser.add_argument("samples", type=Path, help="CSV of real one-second renderer telemetry")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--hardware", required=True, help="Finale hardware description")
    parser.add_argument("--evidence", action="append", default=[], help="Recording/screenshot/log reference; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve(strict=True)
    head = _git_head(repo)
    if head != QUALIFIED_HEAD:
        raise SystemExit(f"wrong git head: expected {QUALIFIED_HEAD}, got {head}")
    path = args.samples.resolve(strict=True)
    samples: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"elapsed_seconds", "fps", "camera_position", "rendering_mode", "navigating"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise SystemExit(f"CSV must contain columns: {sorted(required)}")
        for row_number, row in enumerate(reader, start=2):
            try:
                elapsed = float(row["elapsed_seconds"])
                fps = float(row["fps"])
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"invalid numeric telemetry at CSV row {row_number}") from exc
            if not math.isfinite(elapsed) or not math.isfinite(fps) or elapsed < 0 or fps <= 0:
                raise SystemExit(f"non-finite/non-positive telemetry at CSV row {row_number}")
            samples.append(
                {
                    "elapsed_seconds": elapsed,
                    "fps": fps,
                    "camera_position": str(row["camera_position"]).strip(),
                    "rendering_mode": str(row["rendering_mode"]).strip(),
                    "navigating": _truthy(str(row["navigating"])),
                }
            )
    if len(samples) < 55:
        raise SystemExit(f"need at least 55 samples; got {len(samples)}")
    elapsed_values = [float(sample["elapsed_seconds"]) for sample in samples]
    if any(b <= a for a, b in zip(elapsed_values, elapsed_values[1:])):
        raise SystemExit("elapsed_seconds must be strictly increasing")
    duration = elapsed_values[-1] - elapsed_values[0]
    fps_values = np.asarray([float(sample["fps"]) for sample in samples], dtype=np.float64)
    mean_fps = float(np.mean(fps_values))
    p05_fps = float(np.percentile(fps_values, 5, method="linear"))
    cameras = sorted({str(sample["camera_position"]) for sample in samples if sample["camera_position"]})
    modes = sorted({str(sample["rendering_mode"]) for sample in samples if sample["rendering_mode"]})
    navigation_exercised = any(bool(sample["navigating"]) for sample in samples)

    failures: list[str] = []
    if duration < 60.0:
        failures.append(f"duration {duration:.3f}s < 60s")
    if mean_fps < 30.0:
        failures.append(f"mean FPS {mean_fps:.3f} < 30")
    if p05_fps < 30.0:
        failures.append(f"p05 FPS {p05_fps:.3f} < 30")
    if not navigation_exercised:
        failures.append("navigation was not exercised")
    if not {"aerial", "low"}.issubset(set(cameras)):
        failures.append(f"camera positions missing aerial/low: {cameras}")
    if not {"texture", "analytical_overlay"}.issubset(set(modes)):
        failures.append(f"rendering modes missing texture/analytical_overlay: {modes}")
    if failures:
        raise SystemExit("rendering evidence does not qualify:\n- " + "\n- ".join(failures))

    evidence = [str(path)] + [entry.strip() for entry in args.evidence if entry.strip()]
    report = {
        "schema_version": 1,
        "status": "PASS_SUSTAINED_3D_PERFORMANCE",
        "git_head": QUALIFIED_HEAD,
        "finale_hardware": args.hardware,
        "duration_seconds": duration,
        "sample_count": len(samples),
        "mean_fps": mean_fps,
        "p05_fps": p05_fps,
        "navigation_exercised": navigation_exercised,
        "camera_positions": cameras,
        "rendering_modes": modes,
        "evidence": evidence,
        "samples": samples,
    }
    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "duration_seconds", "sample_count", "mean_fps", "p05_fps")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
