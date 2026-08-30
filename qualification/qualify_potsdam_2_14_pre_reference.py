#!/usr/bin/env python3
"""Qualify the scene-global DA3 mosaic correction before opening evaluator truth.

This helper deliberately uses only:
- the exact corrective DepthWizard source checkout;
- official Potsdam RGB tile 2_14 as model input; and
- independent Copernicus GLO-30 as metric calibration evidence.

The official Potsdam DSM is recorded only as an opaque future path string. This stage never opens,
stats, hashes, raster-inspects, or otherwise consumes the evaluator reference.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from depthwizard.tiling.grid import generate_tiles

CORRECTIVE_HEAD = "5e876709ad5a98cafb03a5c60ab7d84b135c6ffe"
DA3_CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
DA3_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
TILE_SIZE = 1024
OVERLAP = 128
COPDEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"

RGB_RELATIVE = Path("data/external/isprs-potsdam/2_Ortho_RGB/top_potsdam_2_14_RGB.tif")
# IMPORTANT: keep this a plain string until the post-freeze evaluator stage.
REFERENCE_FOR_LATER = (
    "data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif"
)
OUTPUT_RELATIVE = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14")
COPDEM_CACHE_RELATIVE = Path("workspace/final-science-data/calibration/copdem-cache")


class QualificationError(RuntimeError):
    """Corrective urban qualification failed closed."""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _require_exact_clean_checkout(root: Path) -> None:
    head = _git(root, "rev-parse", "HEAD")
    if head != CORRECTIVE_HEAD:
        raise QualificationError(
            f"qualification requires exact corrective head {CORRECTIVE_HEAD}; current={head}"
        )
    dirty = _git(root, "status", "--porcelain")
    if dirty:
        raise QualificationError("repository must be clean before corrective qualification")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_da3_checkpoint(root: Path) -> Path:
    explicit = os.environ.get("DEPTHWIZARD_DA3_CHECKPOINT", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--depth-anything--DA3MONO-LARGE"
        / "snapshots"
        / DA3_REVISION
        / "model.safetensors"
    )
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == DA3_CHECKPOINT_SHA256:
            return candidate.resolve()
    raise QualificationError(
        "pinned DA3 checkpoint was not found with the required SHA-256; "
        "set DEPTHWIZARD_DA3_CHECKPOINT to the exact model.safetensors if needed"
    )


def _run(root: Path, *args: str) -> None:
    command = [sys.executable, "-m", "depthwizard.cli", *args]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def _copdem_tile_name(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}_00"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM"


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DepthWizard-SIH26175-urban-mosaic-qualification/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise QualificationError(f"Copernicus DEM HTTP {exc.code}: {url}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise QualificationError(f"Copernicus DEM download was empty: {url}")
    temporary.replace(destination)


def _required_copdem_tiles(rgb_path: Path) -> list[tuple[int, int]]:
    with rasterio.open(rgb_path) as src:
        if src.crs is None or src.transform.is_identity:
            raise QualificationError("Potsdam qualification RGB must be georeferenced")
        left, bottom, right, top = transform_bounds(
            src.crs,
            CRS.from_epsg(4326),
            *src.bounds,
            densify_pts=21,
        )
    return [
        (lat, lon)
        for lat in range(math.floor(bottom), math.ceil(top))
        for lon in range(math.floor(left), math.ceil(right))
    ]


def _build_copdem_mosaic(rgb_path: Path, cache_root: Path, output: Path) -> list[Path]:
    source_paths: list[Path] = []
    for latitude, longitude in _required_copdem_tiles(rgb_path):
        tile = _copdem_tile_name(latitude, longitude)
        filename = f"{tile}.tif"
        destination = cache_root / tile / filename
        _download(f"{COPDEM_BASE}/{tile}/{filename}", destination)
        source_paths.append(destination.resolve())
    if not source_paths:
        raise QualificationError("no Copernicus GLO-30 tile coverage resolved for Potsdam 2_14")

    datasets = [rasterio.open(path) for path in source_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            count=mosaic.shape[0],
            transform=transform,
            compress="deflate",
            tiled=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.tif")
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(mosaic)
        temporary.replace(output)
    finally:
        for dataset in datasets:
            dataset.close()
    return source_paths


def _read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    if int(valid.sum()) < 1024:
        raise QualificationError(f"surface has insufficient valid pixels: {path}")
    return values, valid


def _internal_grid_coordinates(length: int, axis: str) -> list[int]:
    tiles = generate_tiles(6000, 6000, tile_size=TILE_SIZE, overlap=OVERLAP)
    if axis == "x":
        coordinates = {tile.x for tile in tiles}
        coordinates.update(tile.x + tile.width for tile in tiles)
    elif axis == "y":
        coordinates = {tile.y for tile in tiles}
        coordinates.update(tile.y + tile.height for tile in tiles)
    else:
        raise ValueError("axis must be x or y")
    return sorted(value for value in coordinates if 0 < value < length)


def _stripe_mask(
    height: int,
    width: int,
    *,
    xs: list[int],
    ys: list[int],
    half_width: int,
) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for x in xs:
        mask[:, max(0, x - half_width) : min(width, x + half_width + 1)] = True
    for y in ys:
        mask[max(0, y - half_width) : min(height, y + half_width + 1), :] = True
    return mask


def _control_coordinates(coordinates: list[int], length: int, forbidden: list[int]) -> list[int]:
    stride = TILE_SIZE - OVERLAP
    candidates: list[int] = []
    for coordinate in coordinates:
        candidate = coordinate + stride // 2
        if candidate >= length - OVERLAP:
            candidate = coordinate - stride // 2
        if candidate <= OVERLAP or candidate >= length - OVERLAP:
            continue
        if min(abs(candidate - item) for item in forbidden) <= OVERLAP // 2:
            continue
        candidates.append(candidate)
    return sorted(set(candidates))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 1e-12:
        return float("inf") if numerator > 1e-12 else 1.0
    return numerator / denominator


def _tile_grid_diagnostic(path: Path) -> dict[str, Any]:
    values, valid = _read_surface(path)
    fill = float(np.median(values[valid]))
    filled = np.where(valid, values, fill).astype(np.float32, copy=False)
    grad_y, grad_x = np.gradient(filled)
    gradient = np.hypot(grad_x, grad_y).astype(np.float32, copy=False)

    height, width = values.shape
    xs = _internal_grid_coordinates(width, "x")
    ys = _internal_grid_coordinates(height, "y")
    control_xs = _control_coordinates(xs, width, xs)
    control_ys = _control_coordinates(ys, height, ys)
    band_half_width = max(8, OVERLAP // 4)

    seam_mask = _stripe_mask(
        height,
        width,
        xs=xs,
        ys=ys,
        half_width=band_half_width,
    ) & valid
    control_mask = _stripe_mask(
        height,
        width,
        xs=control_xs,
        ys=control_ys,
        half_width=band_half_width,
    ) & valid
    control_mask &= ~seam_mask

    seam_values = gradient[seam_mask]
    control_values = gradient[control_mask]
    if seam_values.size < 1024 or control_values.size < 1024:
        raise QualificationError("tile-grid diagnostic masks have insufficient support")

    seam_median = float(np.median(seam_values))
    control_median = float(np.median(control_values))
    seam_p75 = float(np.percentile(seam_values, 75.0))
    control_p75 = float(np.percentile(control_values, 75.0))
    seam_p90 = float(np.percentile(seam_values, 90.0))
    control_p90 = float(np.percentile(control_values, 90.0))

    median_ratio = _safe_ratio(seam_median, control_median)
    p75_ratio = _safe_ratio(seam_p75, control_p75)
    p90_ratio = _safe_ratio(seam_p90, control_p90)

    # These are predeclared engineering guardrails, not reference-tuned science thresholds. A severe
    # inference-grid imprint must not concentrate gradients in expected tile bands by >50% versus
    # phase-shifted control bands across the same scene.
    pass_threshold = 1.50
    passed = max(median_ratio, p75_ratio, p90_ratio) <= pass_threshold

    finite_values = values[valid].astype(np.float64)
    p01, p50, p99 = np.percentile(finite_values, [1.0, 50.0, 99.0])
    return {
        "surface": str(path),
        "shape": [height, width],
        "valid_pixels": int(valid.sum()),
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": TILE_SIZE - OVERLAP,
        "internal_grid_x": xs,
        "internal_grid_y": ys,
        "control_grid_x": control_xs,
        "control_grid_y": control_ys,
        "band_half_width_px": band_half_width,
        "seam_gradient_median": seam_median,
        "control_gradient_median": control_median,
        "seam_to_control_median_ratio": median_ratio,
        "seam_gradient_p75": seam_p75,
        "control_gradient_p75": control_p75,
        "seam_to_control_p75_ratio": p75_ratio,
        "seam_gradient_p90": seam_p90,
        "control_gradient_p90": control_p90,
        "seam_to_control_p90_ratio": p90_ratio,
        "guardrail_ratio_max": pass_threshold,
        "guardrail_pass": passed,
        "surface_p01": float(p01),
        "surface_p50": float(p50),
        "surface_p99": float(p99),
    }


def _write_preview(surface: Path, output: Path, *, draw_grid: bool) -> None:
    values, valid = _read_surface(surface)
    finite = values[valid].astype(np.float64)
    low, high = np.percentile(finite, [2.0, 98.0])
    span = float(high - low)
    if span <= 1e-12:
        normalized = np.zeros(values.shape, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / span, 0.0, 1.0)
    image = Image.fromarray((np.nan_to_num(normalized) * 255.0).astype(np.uint8), mode="L")
    if draw_grid:
        image = image.convert("RGB")
        drawing = ImageDraw.Draw(image)
        for x in _internal_grid_coordinates(values.shape[1], "x"):
            drawing.line((x, 0, x, values.shape[0] - 1), fill=(255, 0, 0), width=2)
        for y in _internal_grid_coordinates(values.shape[0], "y"):
            drawing.line((0, y, values.shape[1] - 1, y), fill=(255, 0, 0), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _find_old_defect_rdsm(root: Path) -> Path | None:
    prediction_root = root / "workspace" / "final-science-data" / "predictions"
    if not prediction_root.is_dir():
        return None
    for report_path in sorted(prediction_root.rglob("prediction-generation.json")):
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get("source_rgb")
        relative = payload.get("relative_dsm")
        if not isinstance(source, str) or not isinstance(relative, str):
            continue
        if Path(source).name != RGB_RELATIVE.name:
            continue
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()
    return None


def _validate_existing_freeze(root: Path, output_dir: Path) -> bool:
    freeze_path = output_dir / "pre-reference-freeze.json"
    if not freeze_path.is_file():
        return False
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("git_head") != CORRECTIVE_HEAD:
        raise QualificationError("existing corrective freeze belongs to a different source head")
    for key in ("relative_dsm", "metric_dsm", "calibration_evidence"):
        record = payload.get(key)
        if not isinstance(record, dict):
            raise QualificationError(f"existing freeze is missing {key}")
        path_value = record.get("path")
        expected_sha = record.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected_sha, str):
            raise QualificationError(f"existing freeze has malformed {key} identity")
        path = Path(path_value)
        if not path.is_file() or _sha256(path) != expected_sha:
            raise QualificationError(f"existing frozen {key} identity no longer matches disk")
    print("PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE_REUSED")
    print(f"freeze={freeze_path}")
    return True


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    _require_exact_clean_checkout(root)
    checkpoint = _find_da3_checkpoint(root)
    rgb = (root / RGB_RELATIVE).resolve(strict=True)
    output_dir = (root / OUTPUT_RELATIVE).resolve(strict=False)

    if output_dir.exists():
        if _validate_existing_freeze(root, output_dir):
            return 0
        raise QualificationError(
            "corrective output directory exists without a valid completed freeze; remove only that "
            f"qualification directory after diagnosis before retrying: {output_dir}"
        )

    working = output_dir.with_name(output_dir.name + ".working")
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True, exist_ok=False)

    reconstruction = working / "reconstruction"
    rdsm = reconstruction / "rdsm.tif"
    calibration = working / "calibration" / "copernicus-glo30-mosaic.tif"
    metric_dsm = working / "metric-dsm.tif"
    diagnostic_path = working / "pre-reference-tile-grid-diagnostic.json"

    print("=== CORRECTIVE POTSDAM 2_14 PRE-REFERENCE QUALIFICATION ===")
    print(f"git_head={CORRECTIVE_HEAD}")
    print(f"rgb={rgb}")
    print("reference_values_consumed=false")

    _run(
        root,
        "reconstruct-da3",
        str(rgb),
        str(reconstruction),
        "--tile-size",
        str(TILE_SIZE),
        "--overlap",
        str(OVERLAP),
    )
    if not rdsm.is_file():
        raise QualificationError("corrective DA3 reconstruction did not create rDSM")

    copdem_sources = _build_copdem_mosaic(
        rgb,
        root / COPDEM_CACHE_RELATIVE,
        calibration,
    )
    _run(root, "calibrate-dem", str(rdsm), str(calibration), str(metric_dsm))
    if not metric_dsm.is_file():
        raise QualificationError("corrective metric calibration did not create DSM")

    rdsm_diagnostic = _tile_grid_diagnostic(rdsm)
    metric_diagnostic = _tile_grid_diagnostic(metric_dsm)
    baseline_path = _find_old_defect_rdsm(root)
    baseline_diagnostic = (
        _tile_grid_diagnostic(baseline_path) if baseline_path is not None else None
    )

    diagnostic = {
        "schema_version": 1,
        "status": (
            "PASS_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC"
            if rdsm_diagnostic["guardrail_pass"] and metric_diagnostic["guardrail_pass"]
            else "REVIEW_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC"
        ),
        "git_head": CORRECTIVE_HEAD,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "guardrail_is_reference_tuned": False,
        "rdsm": rdsm_diagnostic,
        "metric_dsm": metric_diagnostic,
        "old_defect_baseline_rdsm": baseline_diagnostic,
    }
    diagnostic_path.write_text(
        json.dumps(diagnostic, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_preview(rdsm, working / "rdsm-preview.png", draw_grid=False)
    _write_preview(rdsm, working / "rdsm-grid-preview.png", draw_grid=True)
    _write_preview(metric_dsm, working / "metric-dsm-preview.png", draw_grid=False)
    if baseline_path is not None:
        _write_preview(
            baseline_path,
            working / "old-defect-rdsm-grid-preview.png",
            draw_grid=True,
        )

    freeze = {
        "schema_version": 1,
        "status": "PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE",
        "git_head": CORRECTIVE_HEAD,
        "geometry_pipeline_revision": "scene-global-affine-mosaic-v2",
        "source_rgb": {"path": str(rgb), "sha256": _sha256(rgb)},
        "model_checkpoint": {
            "path": str(checkpoint),
            "sha256": DA3_CHECKPOINT_SHA256,
        },
        "relative_dsm": {"path": str(rdsm), "sha256": _sha256(rdsm)},
        "calibration_evidence": {
            "path": str(calibration),
            "sha256": _sha256(calibration),
            "sources": [str(path) for path in copdem_sources],
        },
        "metric_dsm": {"path": str(metric_dsm), "sha256": _sha256(metric_dsm)},
        "pre_reference_diagnostic": {
            "path": str(diagnostic_path),
            "sha256": _sha256(diagnostic_path),
            "status": diagnostic["status"],
        },
        "reference_for_later_string_only": REFERENCE_FOR_LATER,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
    }
    freeze_path = working / "pre-reference-freeze.json"
    freeze_path.write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # No evaluator reference has been touched. Atomically promote the complete qualification bundle.
    working.replace(output_dir)
    promoted_freeze = output_dir / "pre-reference-freeze.json"
    promoted_diagnostic = output_dir / "pre-reference-tile-grid-diagnostic.json"

    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    print("PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE")
    print(f"freeze={promoted_freeze}")
    print(f"diagnostic={promoted_diagnostic}")
    print(f"rdsm_preview={output_dir / 'rdsm-grid-preview.png'}")
    print("reference_values_consumed=false")
    if diagnostic["status"] == "PASS_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC":
        print("PASS_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC")
        return 0
    print("REVIEW_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC")
    return 5


if __name__ == "__main__":
    raise SystemExit(main())
