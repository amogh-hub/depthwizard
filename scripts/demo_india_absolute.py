from __future__ import annotations

import io
import json
import math
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.raster import read_rgb, reproject_to_match, write_float_geotiff
from depthwizard.mesh.terrain import export_lod_pyramid
from depthwizard.pipeline.geometry import infer_geometry_scene

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo" / "india"
RDSM_DIR = ROOT / "artifacts" / "demo" / "india-rdsm"
ABS_DIR = ROOT / "artifacts" / "demo" / "india-absolute"
MESH_DIR = ROOT / "artifacts" / "demo" / "india-absolute-mesh"

# Joshimath, Uttarakhand: mountainous Indian terrain with strong disaster-management relevance.
CENTER_LAT = 30.555
CENTER_LON = 79.565
RGB_ZOOM = 12
DEM_ZOOM = 10
TILE_SIZE = 256
WORLD_HALF = 20037508.342789244
S2_TEMPLATE = (
    "https://tiles.maps.eox.at/wmts/1.0.0/"
    "s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg"
)
TERRARIUM_TEMPLATE = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
USER_AGENT = "DepthWizard-SIH26175/0.2 educational-demo"


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    lat = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(lat)
    x = math.floor((lon + 180.0) / 360.0 * n)
    y = math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile_bounds(
    x: int,
    y: int,
    zoom: int,
    width_tiles: int = 1,
    height_tiles: int = 1,
) -> tuple[float, float, float, float]:
    n = 2**zoom
    span = 2.0 * WORLD_HALF / n
    min_x = -WORLD_HALF + x * span
    max_x = -WORLD_HALF + (x + width_tiles) * span
    max_y = WORLD_HALF - y * span
    min_y = WORLD_HALF - (y + height_tiles) * span
    return min_x, min_y, max_x, max_y


def download(url: str, cache_path: Path) -> bytes:
    if cache_path.exists():
        return cache_path.read_bytes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty response from {url}")
    cache_path.write_bytes(payload)
    return payload


def build_rgb_scene() -> Path:
    # One DEM z10 tile corresponds exactly to a 4x4 block of z12 imagery tiles.
    dem_x, dem_y = lonlat_to_tile(CENTER_LON, CENTER_LAT, DEM_ZOOM)
    rgb_x0 = dem_x * 4
    rgb_y0 = dem_y * 4
    mosaic = np.zeros((TILE_SIZE * 4, TILE_SIZE * 4, 3), dtype=np.uint8)

    for row in range(4):
        for col in range(4):
            x = rgb_x0 + col
            y = rgb_y0 + row
            url = S2_TEMPLATE.format(z=RGB_ZOOM, x=x, y=y)
            cache = DATA_DIR / "tiles" / f"s2_z{RGB_ZOOM}_{x}_{y}.jpg"
            payload = download(url, cache)
            tile = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.uint8)
            if tile.shape != (TILE_SIZE, TILE_SIZE, 3):
                raise RuntimeError(f"unexpected Sentinel-2 tile shape {tile.shape} from {url}")
            y0 = row * TILE_SIZE
            x0 = col * TILE_SIZE
            mosaic[y0 : y0 + TILE_SIZE, x0 : x0 + TILE_SIZE] = tile

    bounds = tile_bounds(rgb_x0, rgb_y0, RGB_ZOOM, width_tiles=4, height_tiles=4)
    transform = from_bounds(*bounds, width=mosaic.shape[1], height=mosaic.shape[0])
    output = DATA_DIR / "joshimath_s2cloudless_2024.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=transform,
        compress="deflate",
        tiled=True,
        blockxsize=512,
        blockysize=512,
    ) as dst:
        dst.write(np.moveaxis(mosaic, -1, 0))
        dst.update_tags(
            DEPTHWIZARD_DEMO_SCENE="Joshimath_Uttarakhand_India",
            IMAGERY_SOURCE="EOX Sentinel-2 cloudless 2024",
            IMAGERY_ATTRIBUTION=(
                "EOxCloudless by EOX IT Services GmbH; contains modified "
                "Copernicus Sentinel data 2024"
            ),
        )
    return output


def build_low_resolution_dem() -> Path:
    x, y = lonlat_to_tile(CENTER_LON, CENTER_LAT, DEM_ZOOM)
    url = TERRARIUM_TEMPLATE.format(z=DEM_ZOOM, x=x, y=y)
    cache = DATA_DIR / "tiles" / f"terrarium_z{DEM_ZOOM}_{x}_{y}.png"
    payload = download(url, cache)
    rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32)
    if rgb.shape != (TILE_SIZE, TILE_SIZE, 3):
        raise RuntimeError(f"unexpected Terrarium tile shape {rgb.shape} from {url}")
    elevation = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0

    bounds = tile_bounds(x, y, DEM_ZOOM)
    transform = from_bounds(*bounds, width=TILE_SIZE, height=TILE_SIZE)
    output = DATA_DIR / "joshimath_terrarium_z10_dem.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=TILE_SIZE,
        width=TILE_SIZE,
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
    ) as dst:
        dst.write(elevation.astype(np.float32), 1)
        dst.set_band_description(1, "Low-resolution terrain DEM in metres")
        dst.update_tags(
            DEM_SOURCE="AWS Terrain Tiles / Mapzen Terrarium",
            DEM_ENCODING="Terrarium",
            DEM_ZOOM=str(DEM_ZOOM),
        )
    return output


def dem_consistency_diagnostics(
    dsm: np.ndarray,
    dem: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int | None]:
    mask = np.asarray(valid, dtype=bool) & np.isfinite(dsm) & np.isfinite(dem)
    count = int(mask.sum())
    if count == 0:
        return {"valid_pixels": 0, "rmse_m": None, "mae_m": None, "pearson_r": None}
    predicted = np.asarray(dsm[mask], dtype=np.float64)
    reference = np.asarray(dem[mask], dtype=np.float64)
    error = predicted - reference
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    if np.ptp(predicted) <= 1e-9 or np.ptp(reference) <= 1e-9:
        correlation: float | None = None
    else:
        correlation = float(np.corrcoef(predicted, reference)[0, 1])
    return {
        "valid_pixels": count,
        "rmse_m": rmse,
        "mae_m": mae,
        "pearson_r": correlation,
    }


def run_demo() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RDSM_DIR.mkdir(parents=True, exist_ok=True)
    ABS_DIR.mkdir(parents=True, exist_ok=True)
    MESH_DIR.mkdir(parents=True, exist_ok=True)

    print("Preparing India-specific Sentinel-2 / low-resolution DEM scene...")
    rgb_path = build_rgb_scene()
    dem_path = build_low_resolution_dem()

    print("Running DA3MONO-LARGE tiled reconstruction...")
    prior = DA3MonocularPrior(device="auto")
    reconstruction_started = time.perf_counter()
    scene = infer_geometry_scene(
        rgb_path,
        prior,
        tile_size=768,
        overlap=128,
        harmonize_overlaps=True,
    )
    reconstruction_seconds = time.perf_counter() - reconstruction_started
    rdsm_path = RDSM_DIR / "rdsm.tif"
    write_float_geotiff(
        rdsm_path,
        scene.relative_height,
        template_path=rgb_path,
        description="DepthWizard relative DSM (dimensionless)",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": scene.model_id,
        },
    )

    print(
        "Calibrating relative surface height to metric elevation using "
        "lower-resolution DEM evidence..."
    )
    aligned_dem, valid_dem = reproject_to_match(dem_path, rdsm_path)
    rel = scene.relative_height.astype(np.float32, copy=False)
    valid = np.isfinite(rel) & valid_dem & np.isfinite(aligned_dem)
    if int(valid.sum()) < 32:
        raise RuntimeError("insufficient DEM overlap for absolute calibration")
    dem_dynamic_range = float(
        np.percentile(aligned_dem[valid], 99) - np.percentile(aligned_dem[valid], 1)
    )
    if dem_dynamic_range < 50.0:
        raise RuntimeError(
            f"DEM terrain range is only {dem_dynamic_range:.1f} m; "
            "scene is underdetermined for this demo"
        )

    calibration_started = time.perf_counter()
    calibrated = calibrate_relative_height_with_dem(
        rel,
        aligned_dem,
        dem_valid=valid,
        low_frequency_sigma_px=48.0,
    )
    calibration_seconds = time.perf_counter() - calibration_started
    evidence_consistency = dem_consistency_diagnostics(calibrated.dsm, aligned_dem, valid)

    dsm_path = ABS_DIR / "dsm.tif"
    write_float_geotiff(
        dsm_path,
        calibrated.dsm,
        template_path=rdsm_path,
        description="DepthWizard DEM-calibrated Digital Surface Model (metres)",
        tags={
            "DEPTHWIZARD_PRODUCT": "DEM_CALIBRATED_DSM_METRES",
            "CALIBRATION_METHOD": calibrated.calibration.method,
            "CALIBRATION_EVIDENCE": "lower_resolution_terrarium_dem",
            "VALIDATION_STATUS": "engineering_demo_not_independent_accuracy_benchmark",
        },
    )

    print("Exporting metric-coordinate textured 3D terrain LODs...")
    rgb = read_rgb(rgb_path)
    with rasterio.open(dsm_path) as src:
        dsm = src.read(1).astype(np.float32)
        valid_dsm = np.isfinite(dsm)
        if src.nodata is not None:
            valid_dsm &= dsm != src.nodata
        gsd_x = abs(float(src.transform.a))
        gsd_y = abs(float(src.transform.e))
        crs = src.crs.to_string() if src.crs is not None else None
    mesh_started = time.perf_counter()
    lods = export_lod_pyramid(
        MESH_DIR,
        np.where(valid_dsm, dsm, 0.0).astype(np.float32),
        rgb,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        strides=(2, 4, 8, 16),
        valid_mask=valid_dsm,
    )
    mesh_seconds = time.perf_counter() - mesh_started

    calibration_payload = calibrated.calibration.model_dump()
    calibration_payload.update(
        {
            "dem_source": str(dem_path.resolve()),
            "dem_dynamic_range_p01_p99_m": dem_dynamic_range,
            "low_frequency_sigma_px": 48.0,
            "orientation_flipped": calibrated.orientation_flipped,
            "anchor_correlation_before": calibrated.anchor_correlation_before,
            "anchor_correlation_after": calibrated.anchor_correlation_after,
            "dem_consistency_after_low_frequency_correction": evidence_consistency,
            "diagnostic_semantics": (
                "same DEM used for calibration; consistency diagnostics are not independent accuracy metrics"
            ),
        }
    )
    calibration_path = ABS_DIR / "calibration.json"
    calibration_path.write_text(json.dumps(calibration_payload, indent=2), encoding="utf-8")

    report = {
        "status": "PASS_ENGINEERING_PATH",
        "scene": "Joshimath, Uttarakhand, India",
        "purpose": (
            "engineering demonstration of the SIH georeferenced absolute-elevation path; "
            "not an accuracy benchmark"
        ),
        "imagery": {
            "source": "EOX Sentinel-2 cloudless 2024",
            "attribution": (
                "EOxCloudless by EOX IT Services GmbH; contains modified "
                "Copernicus Sentinel data 2024"
            ),
            "path": str(rgb_path.resolve()),
            "zoom": RGB_ZOOM,
        },
        "dem_evidence": {
            "source": "AWS Terrain Tiles / Mapzen Terrarium",
            "path": str(dem_path.resolve()),
            "zoom": DEM_ZOOM,
            "dynamic_range_p01_p99_m": dem_dynamic_range,
        },
        "model": scene.model_id,
        "device": prior._resolved_device or "unknown",
        "shape": list(scene.relative_height.shape),
        "tile_count": scene.tile_count,
        "harmonized_tiles": scene.harmonized_tiles,
        "reconstruction_seconds": reconstruction_seconds,
        "calibration_seconds": calibration_seconds,
        "calibration": calibration_payload,
        "rdsm": str(rdsm_path.resolve()),
        "dsm": str(dsm_path.resolve()),
        "crs": crs,
        "gsd_x_m": gsd_x,
        "gsd_y_m": gsd_y,
        "mesh_seconds": mesh_seconds,
        "mesh_lods": [
            {
                "path": str(item.path.resolve()),
                "stride": item.stride,
                "vertices": item.vertices,
                "faces": item.faces,
            }
            for item in lods
        ],
    }
    report_path = ABS_DIR / "demo_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard India absolute-elevation engineering path: PASS")
    print("Scene: Joshimath, Uttarakhand, India")
    print(f"Model/device: {scene.model_id} / {prior._resolved_device or 'unknown'}")
    print(f"Tiles: {scene.tile_count} ({scene.harmonized_tiles} overlap-harmonized)")
    print(f"DEM range (p01-p99): {dem_dynamic_range:.1f} m")
    print(
        "Anchor polarity/correlation: "
        f"flipped={calibrated.orientation_flipped}, "
        f"r={calibrated.anchor_correlation_before:.3f} -> "
        f"{calibrated.anchor_correlation_after:.3f}"
    )
    print(
        "Calibration scale/offset: "
        f"{calibrated.calibration.scale:.3f} / {calibrated.calibration.offset:.3f} m"
    )
    print(f"Affine anchor RMSE (diagnostic, not accuracy): {calibrated.calibration.rmse_anchor:.2f} m")
    consistency_rmse = evidence_consistency["rmse_m"]
    if isinstance(consistency_rmse, float):
        print(f"Post-correction DEM consistency RMSE (not independent): {consistency_rmse:.2f} m")
    print(f"Metric DSM: {dsm_path}")
    print(f"3D LODs: {MESH_DIR}")
    print(f"Report: {report_path}")
    print("Accuracy claim: NONE — independent LiDAR/reference validation is still required.")


if __name__ == "__main__":
    run_demo()
