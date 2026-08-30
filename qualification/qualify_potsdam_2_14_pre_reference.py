#!/usr/bin/env python3
"""Pre-reference qualification for the scene-global DA3 mosaic correction.

Uses only Potsdam RGB 2_14 + Copernicus GLO-30 calibration evidence. The official Potsdam DSM is
kept as an opaque string and is never opened, stat'ed, raster-inspected, or hashed by this stage.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
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

HEAD = "5e876709ad5a98cafb03a5c60ab7d84b135c6ffe"
CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
CHECKPOINT_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
TILE_SIZE = 1024
OVERLAP = 128
STRIDE = TILE_SIZE - OVERLAP
RGB_REL = Path("data/external/isprs-potsdam/2_Ortho_RGB/top_potsdam_2_14_RGB.tif")
OUTPUT_REL = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14")
COPDEM_CACHE_REL = Path("workspace/final-science-data/calibration/copdem-cache")
REFERENCE_STRING_ONLY = "data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif"
COPDEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


class QualificationError(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkout(root: Path) -> None:
    current = git(root, "rev-parse", "HEAD")
    if current != HEAD:
        raise QualificationError(f"requires exact head {HEAD}; current={current}")
    if git(root, "status", "--porcelain"):
        raise QualificationError("repository must be clean")


def find_checkpoint() -> Path:
    explicit = os.environ.get("DEPTHWIZARD_DA3_CHECKPOINT", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(
        Path.home()
        / ".cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots"
        / CHECKPOINT_REVISION
        / "model.safetensors"
    )
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == CHECKPOINT_SHA256:
            return candidate.resolve()
    raise QualificationError("pinned DA3 checkpoint not found with required SHA-256")


def run_cli(root: Path, *args: str) -> None:
    command = [sys.executable, "-m", "depthwizard.cli", *args]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def copdem_name(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}_00"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM"


def required_copdem(rgb: Path) -> list[tuple[int, int]]:
    with rasterio.open(rgb) as src:
        if src.crs is None or src.transform.is_identity:
            raise QualificationError("Potsdam RGB must be georeferenced")
        left, bottom, right, top = transform_bounds(
            src.crs, CRS.from_epsg(4326), *src.bounds, densify_pts=21
        )
    return [
        (lat, lon)
        for lat in range(math.floor(bottom), math.ceil(top))
        for lon in range(math.floor(left), math.ceil(right))
    ]


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "DepthWizard-SIH26175-urban-mosaic-qualification/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise QualificationError(f"empty Copernicus DEM download: {url}")
    temporary.replace(destination)


def build_copdem(rgb: Path, cache: Path, output: Path) -> list[Path]:
    sources: list[Path] = []
    for latitude, longitude in required_copdem(rgb):
        tile = copdem_name(latitude, longitude)
        filename = f"{tile}.tif"
        path = cache / tile / filename
        download(f"{COPDEM_BASE}/{tile}/{filename}", path)
        sources.append(path.resolve())
    if not sources:
        raise QualificationError("no Copernicus GLO-30 coverage resolved")
    datasets = [rasterio.open(path) for path in sources]
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
    return sources


def read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    if int(valid.sum()) < 1024:
        raise QualificationError(f"insufficient valid surface pixels: {path}")
    return values, valid


def grid_coordinates(height: int, width: int) -> tuple[list[int], list[int]]:
    tiles = generate_tiles(height, width, tile_size=TILE_SIZE, overlap=OVERLAP)
    xs = {tile.x for tile in tiles} | {tile.x + tile.width for tile in tiles}
    ys = {tile.y for tile in tiles} | {tile.y + tile.height for tile in tiles}
    return (
        sorted(x for x in xs if 0 < x < width),
        sorted(y for y in ys if 0 < y < height),
    )


def shifted_controls(lines: list[int], length: int) -> list[int]:
    result: list[int] = []
    for line in lines:
        candidate = line + STRIDE // 2
        if candidate >= length - OVERLAP:
            candidate = line - STRIDE // 2
        if not OVERLAP < candidate < length - OVERLAP:
            continue
        if min(abs(candidate - existing) for existing in lines) <= OVERLAP // 2:
            continue
        result.append(candidate)
    return sorted(set(result))


def stripe_mask(
    height: int,
    width: int,
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


def ratio(a: float, b: float) -> float:
    if b <= 1e-12:
        return float("inf") if a > 1e-12 else 1.0
    return a / b


def grid_diagnostic(read_path: Path, report_path: Path) -> dict[str, Any]:
    values, valid = read_surface(read_path)
    fill = float(np.median(values[valid]))
    filled = np.where(valid, values, fill).astype(np.float32, copy=False)
    gy, gx = np.gradient(filled)
    gradient = np.hypot(gx, gy).astype(np.float32, copy=False)
    height, width = values.shape
    xs, ys = grid_coordinates(height, width)
    control_xs = shifted_controls(xs, width)
    control_ys = shifted_controls(ys, height)
    half_width = OVERLAP // 4
    seam = stripe_mask(height, width, xs, ys, half_width) & valid
    control = stripe_mask(height, width, control_xs, control_ys, half_width) & valid
    control &= ~seam
    seam_values = gradient[seam]
    control_values = gradient[control]
    if seam_values.size < 1024 or control_values.size < 1024:
        raise QualificationError("tile-grid diagnostic lacks support")
    statistics: dict[str, float] = {}
    ratios: list[float] = []
    for percentile in (50.0, 75.0, 90.0):
        seam_value = float(np.percentile(seam_values, percentile))
        control_value = float(np.percentile(control_values, percentile))
        current_ratio = ratio(seam_value, control_value)
        key = f"p{int(percentile)}"
        statistics[f"seam_gradient_{key}"] = seam_value
        statistics[f"control_gradient_{key}"] = control_value
        statistics[f"seam_to_control_{key}_ratio"] = current_ratio
        ratios.append(current_ratio)
    p01, p50, p99 = np.percentile(values[valid].astype(np.float64), [1.0, 50.0, 99.0])
    threshold = 1.50
    return {
        "surface": str(report_path),
        "shape": [height, width],
        "valid_pixels": int(valid.sum()),
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": STRIDE,
        "internal_grid_x": xs,
        "internal_grid_y": ys,
        "control_grid_x": control_xs,
        "control_grid_y": control_ys,
        "band_half_width_px": half_width,
        **statistics,
        "guardrail_ratio_max": threshold,
        "guardrail_pass": max(ratios) <= threshold,
        "surface_p01": float(p01),
        "surface_p50": float(p50),
        "surface_p99": float(p99),
    }


def write_preview(read_path: Path, output: Path, draw_grid: bool) -> None:
    values, valid = read_surface(read_path)
    low, high = np.percentile(values[valid].astype(np.float64), [2.0, 98.0])
    span = float(high - low)
    normalized = np.zeros(values.shape, dtype=np.float32)
    if span > 1e-12:
        normalized[valid] = np.clip((values[valid] - low) / span, 0.0, 1.0)
    image = Image.fromarray((normalized * 255.0).astype(np.uint8), mode="L")
    if draw_grid:
        image = image.convert("RGB")
        drawing = ImageDraw.Draw(image)
        xs, ys = grid_coordinates(values.shape[0], values.shape[1])
        for x in xs:
            drawing.line((x, 0, x, values.shape[0] - 1), fill=(255, 0, 0), width=2)
        for y in ys:
            drawing.line((0, y, values.shape[1] - 1, y), fill=(255, 0, 0), width=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def old_rdsm(root: Path) -> Path | None:
    base = root / "workspace/final-science-data/predictions"
    if not base.is_dir():
        return None
    for report in sorted(base.rglob("prediction-generation.json")):
        try:
            payload = json.loads(report.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source = payload.get("source_rgb")
        relative = payload.get("relative_dsm")
        if not isinstance(source, str) or not isinstance(relative, str):
            continue
        if Path(source).name != RGB_REL.name:
            continue
        candidate = Path(relative)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()
    return None


def validate_existing(output: Path) -> bool:
    freeze_path = output / "pre-reference-freeze.json"
    if not freeze_path.is_file():
        return False
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("git_head") != HEAD:
        raise QualificationError("existing freeze belongs to another source head")
    for key in ("relative_dsm", "metric_dsm", "calibration_evidence"):
        record = payload.get(key)
        if not isinstance(record, dict):
            raise QualificationError(f"existing freeze missing {key}")
        path_text = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path_text, str) or not isinstance(digest, str):
            raise QualificationError(f"malformed frozen identity for {key}")
        path = Path(path_text)
        if not path.is_file() or sha256(path) != digest:
            raise QualificationError(f"frozen {key} identity no longer matches disk")
    print("PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE_REUSED")
    print(f"freeze={freeze_path}")
    return True


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    checkpoint = find_checkpoint()
    rgb = (root / RGB_REL).resolve(strict=True)
    output = (root / OUTPUT_REL).resolve(strict=False)
    if output.exists():
        if validate_existing(output):
            return 0
        raise QualificationError(f"incomplete prior qualification directory exists: {output}")

    working = output.with_name(output.name + ".working")
    if working.exists():
        shutil.rmtree(working)
    working.mkdir(parents=True, exist_ok=False)

    rdsm_work = working / "reconstruction/rdsm.tif"
    calibration_work = working / "calibration/copernicus-glo30-mosaic.tif"
    metric_work = working / "metric-dsm.tif"
    rdsm_final = output / "reconstruction/rdsm.tif"
    calibration_final = output / "calibration/copernicus-glo30-mosaic.tif"
    metric_final = output / "metric-dsm.tif"
    diagnostic_final = output / "pre-reference-tile-grid-diagnostic.json"

    print("=== POTSDAM 2_14 CORRECTIVE PRE-REFERENCE QUALIFICATION ===")
    print(f"git_head={HEAD}")
    print(f"source_rgb={rgb}")
    print("reference_values_consumed=false")

    run_cli(
        root,
        "reconstruct-da3",
        str(rgb),
        str(working / "reconstruction"),
        "--tile-size",
        str(TILE_SIZE),
        "--overlap",
        str(OVERLAP),
    )
    if not rdsm_work.is_file():
        raise QualificationError("reconstruction did not create rDSM")
    sources = build_copdem(rgb, root / COPDEM_CACHE_REL, calibration_work)
    run_cli(root, "calibrate-dem", str(rdsm_work), str(calibration_work), str(metric_work))
    if not metric_work.is_file():
        raise QualificationError("metric calibration did not create DSM")

    rdsm_diag = grid_diagnostic(rdsm_work, rdsm_final)
    metric_diag = grid_diagnostic(metric_work, metric_final)
    baseline = old_rdsm(root)
    baseline_diag = grid_diagnostic(baseline, baseline) if baseline is not None else None
    status = (
        "PASS_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC"
        if rdsm_diag["guardrail_pass"] and metric_diag["guardrail_pass"]
        else "REVIEW_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC"
    )
    diagnostic = {
        "schema_version": 1,
        "status": status,
        "git_head": HEAD,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "guardrail_is_reference_tuned": False,
        "rdsm": rdsm_diag,
        "metric_dsm": metric_diag,
        "old_defect_baseline_rdsm": baseline_diag,
    }
    diagnostic_work = working / diagnostic_final.name
    diagnostic_work.write_text(json.dumps(diagnostic, indent=2, sort_keys=True) + "\n")

    write_preview(rdsm_work, working / "rdsm-preview.png", False)
    write_preview(rdsm_work, working / "rdsm-grid-preview.png", True)
    write_preview(metric_work, working / "metric-dsm-preview.png", False)
    if baseline is not None:
        write_preview(baseline, working / "old-defect-rdsm-grid-preview.png", True)

    freeze = {
        "schema_version": 1,
        "status": "PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE",
        "git_head": HEAD,
        "geometry_pipeline_revision": "scene-global-affine-mosaic-v2",
        "source_rgb": {"path": str(rgb), "sha256": sha256(rgb)},
        "model_checkpoint": {"path": str(checkpoint), "sha256": CHECKPOINT_SHA256},
        "relative_dsm": {"path": str(rdsm_final), "sha256": sha256(rdsm_work)},
        "calibration_evidence": {
            "path": str(calibration_final),
            "sha256": sha256(calibration_work),
            "sources": [str(path) for path in sources],
        },
        "metric_dsm": {"path": str(metric_final), "sha256": sha256(metric_work)},
        "pre_reference_diagnostic": {
            "path": str(diagnostic_final),
            "sha256": sha256(diagnostic_work),
            "status": status,
        },
        "reference_for_later_string_only": REFERENCE_STRING_ONLY,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
    }
    (working / "pre-reference-freeze.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n"
    )
    working.replace(output)

    print(json.dumps(diagnostic, indent=2, sort_keys=True))
    print("PASS_POTSDAM_MOSAIC_PRE_REFERENCE_FREEZE")
    print(f"freeze={output / 'pre-reference-freeze.json'}")
    print(f"diagnostic={diagnostic_final}")
    print(f"grid_preview={output / 'rdsm-grid-preview.png'}")
    print("reference_values_consumed=false")
    print(status)
    return 0 if status == "PASS_PRE_REFERENCE_TILE_GRID_DIAGNOSTIC" else 5


if __name__ == "__main__":
    raise SystemExit(main())
