#!/usr/bin/env python3
"""Metadata-only preflight for the frozen SIH26175 four-terrain campaign.

Checks:
- local availability report matches the frozen RELEASE-2026 NEON selection;
- untouched Potsdam 2_14 and 3_14 RGB/reference files exist;
- Potsdam RGB/reference metadata satisfy the official grid contract;
- no reference raster band values are decoded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from depthwizard.evaluation.potsdam import (
    FROZEN_POTSDAM_TILE_IDS,
    inspect_potsdam_reference_contract,
    inspect_potsdam_rgb_contract,
    resolve_potsdam_tile_paths,
)

EXPECTED_NEON = {
    "CPER": ("2024-06", "RELEASE-2026"),
    "NIWO": ("2024-07", "RELEASE-2026"),
    "HARV": ("2024-08", "RELEASE-2026"),
}
FINAL_POTSDAM = ("2_14", "3_14")


def check_neon_availability(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sites = payload.get("sites")
    if not isinstance(sites, list):
        raise RuntimeError("NEON availability report is missing sites")
    by_site = {item.get("site_code"): item for item in sites if isinstance(item, dict)}
    result: dict[str, object] = {}
    for site, (month, release) in EXPECTED_NEON.items():
        item = by_site.get(site)
        if not isinstance(item, dict):
            raise RuntimeError(f"NEON availability report is missing {site}")
        candidates = item.get("candidates_newest_first")
        if not isinstance(candidates, list):
            raise RuntimeError(f"{site}: candidates are missing")
        matched = [
            candidate
            for candidate in candidates
            if isinstance(candidate, dict)
            and candidate.get("month") == month
            and release in (candidate.get("common_releases") or [])
        ]
        if not matched:
            raise RuntimeError(f"{site}: frozen selection {month} {release} is unavailable")
        result[site] = {"month": month, "release": release, "status": "PASS"}
    return result


def check_potsdam(root: Path) -> dict[str, object]:
    consumed = set(FROZEN_POTSDAM_TILE_IDS)
    if consumed.intersection(FINAL_POTSDAM):
        raise RuntimeError("final Potsdam scenes overlap consumed external-v2 evidence")
    output: dict[str, object] = {}
    for tile_id in FINAL_POTSDAM:
        paths = resolve_potsdam_tile_paths(root, tile_id)
        rgb = inspect_potsdam_rgb_contract(paths.rgb)
        reference = inspect_potsdam_reference_contract(paths.rgb, paths.reference_dsm)
        output[tile_id] = {
            "status": "PASS",
            "rgb": str(paths.rgb.resolve()),
            "reference": str(paths.reference_dsm.resolve()),
            "rgb_contract": rgb,
            "reference_metadata_contract": reference,
            "reference_values_decoded": False,
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only preflight for final SIH26175 science inputs")
    parser.add_argument("--availability", type=Path, default=Path("workspace/final-science-data/neon-availability.json"))
    parser.add_argument("--potsdam-root", type=Path, default=Path("data/external/isprs-potsdam"))
    parser.add_argument("--report", type=Path, default=Path("workspace/final-science-data/input-preflight.json"))
    args = parser.parse_args()

    neon = check_neon_availability(args.availability)
    potsdam = check_potsdam(args.potsdam_root)
    report = {
        "schema_version": 1,
        "status": "PASS_FINAL_SCIENCE_INPUT_PREFLIGHT",
        "reference_values_decoded": False,
        "neon": neon,
        "potsdam": potsdam,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
