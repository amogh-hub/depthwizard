#!/usr/bin/env python3
"""Download the frozen NEON final-science acquisitions chosen by metadata preflight.

This is an operations-only wrapper around download_neon_eval_tiles.py. It pins both
release and month so a later NEON catalogue update cannot silently change the final
science scene identity. Reference DSM bytes may be downloaded but are not opened or
hashed before the prediction freeze.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "download_neon_eval_tiles.py"
SELECTION_PATH = HERE / "frozen_neon_selection.json"


def _load_base() -> Any:
    spec = importlib.util.spec_from_file_location("dw_neon_base", BASE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper: {BASE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if payload.get("release") != "RELEASE-2026":
        raise RuntimeError("frozen NEON selection is not RELEASE-2026")
    sites = payload.get("sites")
    if not isinstance(sites, dict) or set(sites) != {"CPER", "NIWO", "HARV"}:
        raise RuntimeError("frozen NEON selection must contain exactly CPER/NIWO/HARV")
    return payload


def acquire_locked(base: Any, *, site: str, terrain: str, month: str, release: str, root: Path, token: str) -> dict[str, Any]:
    metadata = base._site_metadata(site)
    rgb_product = base._product_info(metadata, base.RGB_PRODUCT)
    dsm_product = base._product_info(metadata, base.DSM_PRODUCT)
    if month not in base._released_months(rgb_product, release):
        raise RuntimeError(f"{site} {month}: RGB is no longer present in {release}")
    if month not in base._released_months(dsm_product, release):
        raise RuntimeError(f"{site} {month}: DSM is no longer present in {release}")

    latitude = base._number(metadata.get("siteLatitude"), f"{site}.siteLatitude")
    longitude = base._number(metadata.get("siteLongitude"), f"{site}.siteLongitude")
    rgb_files = base._file_list(base.RGB_PRODUCT, site, month, release, token)
    dsm_files = base._file_list(base.DSM_PRODUCT, site, month, release, token)
    coord, rgb_item, dsm_item, epsg = base._select_pair(
        rgb_files=rgb_files,
        dsm_files=dsm_files,
        latitude=latitude,
        longitude=longitude,
    )
    easting, northing = coord
    scene_dir = root / site / month / f"{easting}_{northing}"
    rgb_name = base._file_name(rgb_item)
    dsm_name = base._file_name(dsm_item)
    if rgb_name is None or dsm_name is None:
        raise RuntimeError(f"{site}: selected NEON entries are missing filenames")

    rgb_download = base._download(rgb_item, scene_dir / "rgb" / rgb_name, token)
    dsm_download = base._download(dsm_item, scene_dir / "reference-sealed" / dsm_name, token)
    rgb_path = Path(str(rgb_download["path"]))
    return {
        "site": site,
        "site_name": metadata.get("siteName"),
        "terrain": terrain,
        "release": release,
        "month": month,
        "selection_policy": "frozen_metadata_preflight_2026-08-29",
        "site_latitude": latitude,
        "site_longitude": longitude,
        "selected_tile": {"easting": easting, "northing": northing, "utm_epsg": epsg},
        "rgb_product": base.RGB_PRODUCT,
        "reference_product": base.DSM_PRODUCT,
        "rgb": {**rgb_download, "filename": rgb_name, "sha256": base._sha256(rgb_path)},
        "reference": {
            **dsm_download,
            "filename": dsm_name,
            "sha256": None,
            "opened_or_hashed": False,
            "claim_boundary": "Reference bytes downloaded only; elevations were not opened and reference SHA-256 was intentionally not computed before prediction freeze."
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download frozen RELEASE-2026 NEON final-science tiles")
    parser.add_argument("--output-root", type=Path, default=Path("workspace/final-science-data/neon"))
    parser.add_argument("--report", type=Path, default=Path("workspace/final-science-data/neon-acquisition.json"))
    args = parser.parse_args()

    token = os.environ.get("NEON_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("NEON_API_TOKEN is required. Keep it only in the shell; never commit it.")

    base = _load_base()
    selection = _selection()
    release = str(selection["release"])
    rows: list[dict[str, Any]] = []
    for site in ("CPER", "NIWO", "HARV"):
        item = selection["sites"][site]
        terrain = str(item["terrain"])
        month = str(item["month"])
        print(f"Acquiring frozen {site} {month} {release} ({terrain}) ...", file=sys.stderr)
        rows.append(acquire_locked(base, site=site, terrain=terrain, month=month, release=release, root=args.output_root, token=token))

    report = {
        "schema_version": 1,
        "status": "PASS_NEON_FINAL_SCIENCE_ACQUISITION",
        "reference_values_opened_or_hashed": False,
        "selection_file": str(SELECTION_PATH),
        "release": release,
        "scenes": rows,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "scene_count": len(rows), "report": str(args.report)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
