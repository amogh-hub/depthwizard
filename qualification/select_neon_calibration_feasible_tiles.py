#!/usr/bin/env python3
"""Pre-reference calibration-feasibility selection for frozen NEON acquisitions.

Why this exists
---------------
The frozen production calibrator intentionally refuses to manufacture metric elevation when
monocular relative geometry is too weakly correlated with independent coarse DEM evidence.
A deterministic site-centre NEON tile can therefore be a valid RGB/reference pair yet still be
unsuitable for *DEM-based* metric calibration.

This operations-only helper preserves the scientific boundary by selecting a replacement tile
without looking at evaluation truth:

* release/month/site remain frozen by ``frozen_neon_selection.json``;
* matched NEON RGB/DSM candidates are ordered deterministically by distance to the site centre;
* only RGB pixels plus independent Copernicus GLO-30 calibration evidence are used to test
  whether the unchanged frozen production ``calibrate-dem`` guard accepts a candidate;
* the FIRST candidate in that deterministic order that passes is selected (not the best score);
* a candidate's NEON reference DSM is downloaded only after selection and remains sealed;
* reference DSM raster values are never opened, decoded or hashed here;
* rejected candidates are retained in the audit report so the fail-closed event is explicit.

The helper may reuse an already completed ``prediction-generation.json`` for the same exact RGB
identity. For a newly passing candidate it writes the normal final-science generation report so
``prepare_metric_predictions.py`` can reuse the work rather than recomputing it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "download_neon_eval_tiles.py"
PREP_PATH = HERE / "prepare_metric_predictions.py"
SELECTION_PATH = HERE / "frozen_neon_selection.json"
FROZEN_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
SITES = ("CPER", "NIWO", "HARV")
CALIBRATION_REJECTION_MARKERS = (
    "relative height is too weakly correlated with frequency-matched DEM evidence",
    "DEM anchors cannot determine relative-height orientation",
    "insufficient independent reliable DEM anchors",
)


class FeasibilityError(RuntimeError):
    """Pre-reference calibration-feasibility selection failed."""


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FeasibilityError(f"cannot load qualification helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_frozen_checkout(root: Path) -> None:
    head = _git(root, "rev-parse", "HEAD")
    if head != FROZEN_HEAD:
        raise FeasibilityError(
            f"NEON feasibility selection must run from frozen {FROZEN_HEAD}; current={head}"
        )
    if _git(root, "status", "--porcelain"):
        raise FeasibilityError("repository must be clean before NEON feasibility selection")


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    if payload.get("release") != "RELEASE-2026":
        raise FeasibilityError("frozen NEON selection is not RELEASE-2026")
    sites = payload.get("sites")
    if not isinstance(sites, dict) or set(sites) != set(SITES):
        raise FeasibilityError("frozen NEON selection must contain exactly CPER/NIWO/HARV")
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _candidate_order(
    base: Any,
    *,
    rgb_files: list[dict[str, Any]],
    dsm_files: list[dict[str, Any]],
    latitude: float,
    longitude: float,
) -> tuple[list[tuple[int, int]], dict[tuple[int, int], dict[str, Any]], dict[tuple[int, int], dict[str, Any]], int]:
    rgb = base._coord_index(rgb_files, base.RGB_RE)
    dsm = base._coord_index(dsm_files, base.DSM_RE)
    common = sorted(set(rgb) & set(dsm))
    if not common:
        raise FeasibilityError("NEON RGB and DSM responses contain no common tile coordinate")
    epsg = base._utm_epsg(latitude, longitude)
    transformer = base.Transformer.from_crs(4326, epsg, always_xy=True)
    site_e, site_n = transformer.transform(longitude, latitude)
    ordered = sorted(
        common,
        key=lambda coord: (
            (coord[0] + 500.0 - site_e) ** 2 + (coord[1] + 500.0 - site_n) ** 2,
            coord[0],
            coord[1],
        ),
    )
    return ordered, rgb, dsm, epsg


def _existing_generation_pass(
    prep: Any,
    *,
    generation_report: Path,
    metric_dsm: Path,
    calibration: Path,
    rgb_sha: str,
) -> dict[str, Any] | None:
    if not generation_report.is_file() or not metric_dsm.is_file() or not calibration.is_file():
        return None
    try:
        prior = json.loads(generation_report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(prior, dict):
        return None
    try:
        valid = (
            prior.get("status") == "PASS_FINAL_SCIENCE_PREDICTION_GENERATION"
            and prior.get("git_head") == FROZEN_HEAD
            and prior.get("source_sha256") == rgb_sha
            and prior.get("model_checkpoint_sha256") == prep.DA3_CHECKPOINT_SHA256
            and prior.get("metric_dsm_sha256") == prep._sha256(metric_dsm)
            and prior.get("calibration_evidence_sha256") == prep._sha256(calibration)
            and prior.get("reference_raster_opened_or_hashed") is False
        )
    except OSError:
        return None
    return prior if valid else None


def _calibration_command(root: Path, rdsm: Path, calibration: Path, metric_dsm: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "-m",
        "depthwizard.cli",
        "calibrate-dem",
        str(rdsm),
        str(calibration),
        str(metric_dsm),
    ]
    print("+ " + " ".join(command), file=sys.stderr)
    return subprocess.run(
        command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )


def _is_calibration_rejection(output: str) -> bool:
    return any(marker in output for marker in CALIBRATION_REJECTION_MARKERS)


def _candidate_prediction(
    prep: Any,
    *,
    root: Path,
    site: str,
    coord: tuple[int, int],
    rgb_path: Path,
    predictions_root: Path,
    copdem_cache: Path,
    checkpoint: Path,
    tile_size: int,
    overlap: int,
) -> tuple[bool, dict[str, Any]]:
    easting, northing = coord
    scene_id = f"neon-{site.lower()}-{easting}-{northing}"
    scene_dir = predictions_root / scene_id
    reconstruction_dir = scene_dir / "reconstruction"
    rdsm = reconstruction_dir / "rdsm.tif"
    calibration = scene_dir / "calibration" / "copernicus-glo30-mosaic.tif"
    metric_dsm = scene_dir / "metric-dsm.tif"
    generation_report = scene_dir / "prediction-generation.json"
    rgb_sha = prep._sha256(rgb_path)

    prior = _existing_generation_pass(
        prep,
        generation_report=generation_report,
        metric_dsm=metric_dsm,
        calibration=calibration,
        rgb_sha=rgb_sha,
    )
    if prior is not None:
        return True, {
            "scene_id": scene_id,
            "coordinate": [easting, northing],
            "status": "PASS_REUSED_EXISTING_GENERATION",
            "source_sha256": rgb_sha,
            "generation_report": str(generation_report.resolve()),
            "reference_raster_opened_or_hashed": False,
        }

    scene_dir.mkdir(parents=True, exist_ok=True)
    if not rdsm.is_file():
        prep._run(
            root,
            "reconstruct-da3",
            str(rgb_path),
            str(reconstruction_dir),
            "--tile-size",
            str(tile_size),
            "--overlap",
            str(overlap),
        )
    if not rdsm.is_file():
        raise FeasibilityError(f"DA3 feasibility reconstruction did not create {rdsm}")

    # Rebuild the calibration mosaic deterministically from independent Copernicus GLO-30.
    # Source tiles are cached, so repeated candidates usually do not redownload shared degrees.
    copdem_sources = prep._copdem_mosaic(rgb_path, copdem_cache, calibration)
    metric_dsm.unlink(missing_ok=True)
    proc = _calibration_command(root, rdsm, calibration, metric_dsm)
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        if not _is_calibration_rejection(combined):
            raise FeasibilityError(
                f"unexpected calibration failure for {scene_id}:\n{combined[-8000:]}"
            )
        metric_dsm.unlink(missing_ok=True)
        return False, {
            "scene_id": scene_id,
            "coordinate": [easting, northing],
            "status": "REJECTED_BY_UNCHANGED_PRODUCTION_CALIBRATION_GUARD",
            "source_sha256": rgb_sha,
            "production_exit_code": proc.returncode,
            "production_output_tail": combined[-4000:],
            "reference_raster_opened_or_hashed": False,
        }

    if not metric_dsm.is_file():
        raise FeasibilityError(f"passing calibration did not create {metric_dsm}")

    report = {
        "schema_version": 1,
        "status": "PASS_FINAL_SCIENCE_PREDICTION_GENERATION",
        "git_head": FROZEN_HEAD,
        "scene_id": scene_id,
        "source_rgb": str(rgb_path.resolve()),
        "source_sha256": rgb_sha,
        "model_id": prep.DA3_MODEL_ID,
        "model_checkpoint": str(checkpoint),
        "model_checkpoint_sha256": prep.DA3_CHECKPOINT_SHA256,
        "relative_dsm": str(rdsm.resolve()),
        "calibration_evidence": str(calibration.resolve()),
        "calibration_evidence_sha256": prep._sha256(calibration),
        "calibration_sources": [str(path) for path in copdem_sources],
        "metric_dsm": str(metric_dsm.resolve()),
        "metric_dsm_sha256": prep._sha256(metric_dsm),
        "reference_raster_opened_or_hashed": False,
        "selection_context": (
            "Generated during pre-reference calibration-feasibility selection. Candidate accepted "
            "only because the unchanged frozen production DEM calibration guard passed; no NEON "
            "reference raster value was opened, decoded or hashed."
        ),
    }
    _write_json_atomic(generation_report, report)
    return True, {
        "scene_id": scene_id,
        "coordinate": [easting, northing],
        "status": "PASS_UNCHANGED_PRODUCTION_CALIBRATION_GUARD",
        "source_sha256": rgb_sha,
        "generation_report": str(generation_report.resolve()),
        "production_stdout": proc.stdout[-4000:] if proc.stdout else "",
        "reference_raster_opened_or_hashed": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Choose the first deterministically ordered NEON tile per frozen site/month whose RGB "
            "can be defensibly calibrated with independent Copernicus evidence, without opening "
            "the NEON reference DSM."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workspace/final-science-data/neon"),
    )
    parser.add_argument(
        "--acquisition-report",
        type=Path,
        default=Path("workspace/final-science-data/neon-acquisition.json"),
    )
    parser.add_argument(
        "--feasibility-report",
        type=Path,
        default=Path("workspace/final-science-data/neon-calibration-feasibility.json"),
    )
    parser.add_argument(
        "--predictions-root",
        type=Path,
        default=Path("workspace/final-science-data/predictions"),
    )
    parser.add_argument(
        "--copdem-cache",
        type=Path,
        default=Path("workspace/final-science-data/calibration/copdem-cache"),
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--max-candidates", type=int, default=10)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_candidates < 1:
        raise SystemExit("--max-candidates must be >= 1")
    root = args.repo.resolve(strict=True)
    _require_frozen_checkout(root)

    token = os.environ.get("NEON_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("NEON_API_TOKEN is required. Keep it only in the shell; never commit it.")

    base = _load_module("dw_neon_feasibility_base", BASE_PATH)
    prep = _load_module("dw_neon_feasibility_prep", PREP_PATH)
    selection = _selection()
    release = str(selection["release"])
    checkpoint = prep._find_da3_checkpoint(root)

    output_root = (root / args.output_root).resolve(strict=False)
    predictions_root = (root / args.predictions_root).resolve(strict=False)
    copdem_cache = (root / args.copdem_cache).resolve(strict=False)
    acquisition_report = (root / args.acquisition_report).resolve(strict=False)
    feasibility_report = (root / args.feasibility_report).resolve(strict=False)

    selected_rows: list[dict[str, Any]] = []
    site_audit: list[dict[str, Any]] = []

    for site in SITES:
        frozen = selection["sites"][site]
        terrain = str(frozen["terrain"])
        month = str(frozen["month"])
        print(
            f"Pre-reference calibration feasibility: {site} {month} {release} ({terrain}) ...",
            file=sys.stderr,
        )

        metadata = base._site_metadata(site)
        rgb_product = base._product_info(metadata, base.RGB_PRODUCT)
        dsm_product = base._product_info(metadata, base.DSM_PRODUCT)
        if month not in base._released_months(rgb_product, release):
            raise FeasibilityError(f"{site} {month}: RGB is no longer present in {release}")
        if month not in base._released_months(dsm_product, release):
            raise FeasibilityError(f"{site} {month}: DSM is no longer present in {release}")

        latitude = base._number(metadata.get("siteLatitude"), f"{site}.siteLatitude")
        longitude = base._number(metadata.get("siteLongitude"), f"{site}.siteLongitude")
        rgb_files = base._file_list(base.RGB_PRODUCT, site, month, release, token)
        dsm_files = base._file_list(base.DSM_PRODUCT, site, month, release, token)
        ordered, rgb_index, dsm_index, epsg = _candidate_order(
            base,
            rgb_files=rgb_files,
            dsm_files=dsm_files,
            latitude=latitude,
            longitude=longitude,
        )

        attempts: list[dict[str, Any]] = []
        selected_coord: tuple[int, int] | None = None
        selected_rgb_item: dict[str, Any] | None = None
        selected_dsm_item: dict[str, Any] | None = None
        selected_rgb_download: dict[str, Any] | None = None

        for rank, coord in enumerate(ordered[: args.max_candidates], start=1):
            easting, northing = coord
            rgb_item = rgb_index[coord]
            dsm_item = dsm_index[coord]
            rgb_name = base._file_name(rgb_item)
            dsm_name = base._file_name(dsm_item)
            if rgb_name is None or dsm_name is None:
                raise FeasibilityError(f"{site} {coord}: selected NEON entries are missing filenames")
            scene_dir = output_root / site / month / f"{easting}_{northing}"
            rgb_download = base._download(rgb_item, scene_dir / "rgb" / rgb_name, token)
            rgb_path = Path(str(rgb_download["path"])).resolve(strict=True)

            passed, audit = _candidate_prediction(
                prep,
                root=root,
                site=site,
                coord=coord,
                rgb_path=rgb_path,
                predictions_root=predictions_root,
                copdem_cache=copdem_cache,
                checkpoint=checkpoint,
                tile_size=args.tile_size,
                overlap=args.overlap,
            )
            audit["rank"] = rank
            audit["rgb"] = str(rgb_path)
            attempts.append(audit)
            if passed:
                selected_coord = coord
                selected_rgb_item = rgb_item
                selected_dsm_item = dsm_item
                selected_rgb_download = rgb_download
                break
            print(
                f"{site} candidate {easting}_{northing} rejected by production calibration guard; "
                "trying the next deterministic matched tile.",
                file=sys.stderr,
            )

        if (
            selected_coord is None
            or selected_rgb_item is None
            or selected_dsm_item is None
            or selected_rgb_download is None
        ):
            blocked = {
                "schema_version": 1,
                "status": "BLOCKED_NEON_CALIBRATION_FEASIBILITY",
                "site": site,
                "month": month,
                "release": release,
                "candidates_attempted": attempts,
                "max_candidates": args.max_candidates,
                "production_calibration_threshold_weakened": False,
                "reference_rasters_opened_or_hashed": False,
            }
            _write_json_atomic(feasibility_report, blocked)
            print(json.dumps(blocked, indent=2, sort_keys=True))
            return 4

        easting, northing = selected_coord
        scene_dir = output_root / site / month / f"{easting}_{northing}"
        rgb_name = base._file_name(selected_rgb_item)
        dsm_name = base._file_name(selected_dsm_item)
        assert rgb_name is not None and dsm_name is not None
        # Reference transfer occurs only after calibration feasibility has selected the candidate.
        # The file is kept sealed: no raster open and no SHA computation here.
        dsm_download = base._download(
            selected_dsm_item,
            scene_dir / "reference-sealed" / dsm_name,
            token,
        )
        rgb_path = Path(str(selected_rgb_download["path"])).resolve(strict=True)

        row = {
            "site": site,
            "site_name": metadata.get("siteName"),
            "terrain": terrain,
            "release": release,
            "month": month,
            "selection_policy": (
                "frozen_release_month_site__distance_order__first_candidate_passing_"
                "unchanged_production_copdem_calibration_guard__reference_sealed"
            ),
            "site_latitude": latitude,
            "site_longitude": longitude,
            "selected_tile": {
                "easting": easting,
                "northing": northing,
                "utm_epsg": epsg,
            },
            "rgb_product": base.RGB_PRODUCT,
            "reference_product": base.DSM_PRODUCT,
            "rgb": {
                **selected_rgb_download,
                "filename": rgb_name,
                "sha256": base._sha256(rgb_path),
            },
            "reference": {
                **dsm_download,
                "filename": dsm_name,
                "sha256": None,
                "opened_or_hashed": False,
                "claim_boundary": (
                    "Reference bytes downloaded only after RGB/Copernicus-only calibration feasibility "
                    "selection; elevations were not opened and reference SHA-256 was intentionally not "
                    "computed before prediction freeze."
                ),
            },
            "calibration_feasibility": {
                "status": "PASS_FIRST_DETERMINISTIC_CANDIDATE_ACCEPTED_BY_PRODUCTION_GUARD",
                "selected_rank": attempts[-1]["rank"],
                "candidates_attempted": len(attempts),
                "production_calibration_threshold_weakened": False,
                "reference_raster_opened_or_hashed": False,
            },
        }
        selected_rows.append(row)
        site_audit.append(
            {
                "site": site,
                "terrain": terrain,
                "month": month,
                "release": release,
                "selected_coordinate": [easting, northing],
                "attempts": attempts,
                "production_calibration_threshold_weakened": False,
                "reference_raster_opened_or_hashed": False,
            }
        )

    acquisition = {
        "schema_version": 2,
        "status": "PASS_NEON_FINAL_SCIENCE_ACQUISITION",
        "calibration_feasibility_status": "PASS_NEON_PRE_REFERENCE_CALIBRATION_FEASIBILITY",
        "reference_values_opened_or_hashed": False,
        "selection_file": str(SELECTION_PATH),
        "release": release,
        "selection_policy": (
            "Frozen site/release/month; candidate coordinates ordered by site-centre distance; "
            "select first matched RGB/DSM coordinate whose RGB-derived DA3 geometry passes the "
            "unchanged frozen production Copernicus GLO-30 calibration guard. No reference DSM "
            "raster values are opened or hashed during selection."
        ),
        "scenes": selected_rows,
    }
    feasibility = {
        "schema_version": 1,
        "status": "PASS_NEON_PRE_REFERENCE_CALIBRATION_FEASIBILITY",
        "git_head": FROZEN_HEAD,
        "release": release,
        "production_calibration_threshold_weakened": False,
        "selection_uses_reference_values_or_scores": False,
        "reference_rasters_opened_or_hashed": False,
        "sites": site_audit,
    }
    _write_json_atomic(acquisition_report, acquisition)
    _write_json_atomic(feasibility_report, feasibility)
    print(
        json.dumps(
            {
                "status": feasibility["status"],
                "acquisition_report": str(acquisition_report),
                "feasibility_report": str(feasibility_report),
                "selected": {
                    row["site"]: row["selected_tile"] for row in selected_rows
                },
                "production_calibration_threshold_weakened": False,
                "reference_rasters_opened_or_hashed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
