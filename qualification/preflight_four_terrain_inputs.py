#!/usr/bin/env python3
"""Metadata-only preflight for the frozen SIH26175 four-terrain campaign.

Checks:
- local availability report matches the frozen RELEASE-2026 NEON selection;
- inventories locally available official Potsdam RGB/DSM pairs;
- excludes all Potsdam tiles already consumed by DepthWizard external-v2;
- validates candidate RGB/reference metadata without decoding DSM values;
- deterministically freezes two fresh local tiles when possible.

If fewer than two unused valid local Potsdam pairs exist, the script exits with a compact
inventory explaining that official Potsdam data must be added before science can continue.
"""

from __future__ import annotations

import argparse
import json
import re
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
PREFERRED_POTSDAM = ("2_14", "3_14")
RGB_RE = re.compile(r"^top_potsdam_(\d+)_(\d+)_RGB\.tiff?$", re.IGNORECASE)
DSM_RE = re.compile(r"^dsm_potsdam_(\d+)_(\d+)\.tiff?$", re.IGNORECASE)


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


def _tile_id(row: str, col: str) -> str:
    return f"{int(row)}_{int(col)}"


def _inventory_names(root: Path) -> tuple[set[str], set[str]]:
    rgb: set[str] = set()
    dsm: set[str] = set()
    if not root.is_dir():
        return rgb, dsm
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rgb_match = RGB_RE.fullmatch(path.name)
        if rgb_match:
            rgb.add(_tile_id(*rgb_match.groups()))
            continue
        dsm_match = DSM_RE.fullmatch(path.name)
        if dsm_match:
            dsm.add(_tile_id(*dsm_match.groups()))
    return rgb, dsm


def _tile_sort_key(tile_id: str) -> tuple[int, int]:
    row, col = tile_id.split("_", 1)
    return int(row), int(col)


def check_potsdam(root: Path) -> dict[str, object]:
    consumed = set(FROZEN_POTSDAM_TILE_IDS)
    rgb_ids, dsm_ids = _inventory_names(root)
    paired = sorted(rgb_ids & dsm_ids, key=_tile_sort_key)
    unused = [tile_id for tile_id in paired if tile_id not in consumed]

    valid: dict[str, object] = {}
    invalid: dict[str, str] = {}
    for tile_id in unused:
        try:
            paths = resolve_potsdam_tile_paths(root, tile_id)
            rgb_contract = inspect_potsdam_rgb_contract(paths.rgb)
            reference_contract = inspect_potsdam_reference_contract(paths.rgb, paths.reference_dsm)
        except Exception as exc:  # report all local candidates rather than stop at the first defect
            invalid[tile_id] = f"{type(exc).__name__}: {exc}"
            continue
        valid[tile_id] = {
            "status": "PASS_METADATA_ONLY",
            "rgb": str(paths.rgb.resolve()),
            "reference": str(paths.reference_dsm.resolve()),
            "rgb_contract": rgb_contract,
            "reference_metadata_contract": reference_contract,
            "reference_values_decoded": False,
        }

    valid_ids = sorted(valid, key=_tile_sort_key)
    preferred = [tile_id for tile_id in PREFERRED_POTSDAM if tile_id in valid]
    remainder = [tile_id for tile_id in valid_ids if tile_id not in preferred]
    selected = (preferred + remainder)[:2]

    report = {
        "root": str(root.resolve(strict=False)),
        "consumed_tiles_excluded": sorted(consumed, key=_tile_sort_key),
        "rgb_tile_ids_present": sorted(rgb_ids, key=_tile_sort_key),
        "dsm_tile_ids_present": sorted(dsm_ids, key=_tile_sort_key),
        "paired_tile_ids_present": paired,
        "unused_paired_tile_ids": unused,
        "valid_unused_tiles": valid,
        "invalid_unused_tiles": invalid,
        "selected_tiles": selected,
        "selection_policy": (
            "prefer 2_14 and 3_14 when locally valid; otherwise choose the lexicographically first "
            "two valid unused official RGB/DSM pairs"
        ),
        "reference_values_decoded": False,
    }
    if len(selected) < 2:
        report["status"] = "BLOCKED_NEED_TWO_UNUSED_POTSDAM_PAIRS"
        return report
    report["status"] = "PASS_POTSDAM_INPUT_PREFLIGHT"
    report["urban_test_tile"] = selected[0]
    report["cross_sensor_tile"] = selected[1]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Metadata-only preflight for final SIH26175 science inputs")
    parser.add_argument(
        "--availability",
        type=Path,
        default=Path("workspace/final-science-data/neon-availability.json"),
    )
    parser.add_argument(
        "--potsdam-root",
        type=Path,
        default=Path("data/external/isprs-potsdam"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("workspace/final-science-data/input-preflight.json"),
    )
    args = parser.parse_args()

    neon = check_neon_availability(args.availability)
    potsdam = check_potsdam(args.potsdam_root)
    status = (
        "PASS_FINAL_SCIENCE_INPUT_PREFLIGHT"
        if potsdam.get("status") == "PASS_POTSDAM_INPUT_PREFLIGHT"
        else "BLOCKED_FINAL_SCIENCE_INPUT_PREFLIGHT"
    )
    report = {
        "schema_version": 2,
        "status": status,
        "reference_values_decoded": False,
        "neon": neon,
        "potsdam": potsdam,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if status.startswith("PASS_") else 4


if __name__ == "__main__":
    raise SystemExit(main())
