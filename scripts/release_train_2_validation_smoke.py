from __future__ import annotations

import io
import json
import math
import os
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds

from depthwizard.contracts import (
    ProcessingRequest,
    ProjectRunStatus,
    ReferenceValidationRequest,
)
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "acceptance" / "release-train-2-ortholoc"
PROJECT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-2-ortholoc"
ORTHOLOC_BASE_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/demo"
DOP_URL = f"{ORTHOLOC_BASE_URL}/urban_residential_DOP.tif"
DSM_URL = f"{ORTHOLOC_BASE_URL}/urban_residential_DSM.tif"
TERRARIUM_TEMPLATE = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
TERRARIUM_ZOOM = 10
TILE_SIZE = 256
WORLD_HALF = 20037508.342789244
USER_AGENT = "DepthWizard-SIH26175/0.2 release-train-2-acceptance"


def _download(url: str, path: Path) -> bytes:
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty response from {url}")
    path.write_bytes(payload)
    return payload


def _lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    latitude = max(min(lat, 85.05112878), -85.05112878)
    lat_rad = math.radians(latitude)
    x = math.floor((lon + 180.0) / 360.0 * n)
    y = math.floor((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _tile_bounds(
    x: int,
    y: int,
    zoom: int,
    *,
    width_tiles: int,
    height_tiles: int,
) -> tuple[float, float, float, float]:
    n = 2**zoom
    span = 2.0 * WORLD_HALF / n
    min_x = -WORLD_HALF + x * span
    max_x = -WORLD_HALF + (x + width_tiles) * span
    max_y = WORLD_HALF - y * span
    min_y = WORLD_HALF - (y + height_tiles) * span
    return min_x, min_y, max_x, max_y


def _prepare_ortholoc_pair() -> tuple[Path, Path]:
    dop_path = DATA_DIR / "urban_residential_DOP.tif"
    dsm_path = DATA_DIR / "urban_residential_DSM.tif"
    _download(DOP_URL, dop_path)
    _download(DSM_URL, dsm_path)
    return dop_path, dsm_path


def _terrarium_dem_for_source(source_path: Path) -> Path:
    """Build independent low-resolution calibration evidence covering the source footprint."""
    with rasterio.open(source_path) as src:
        if src.crs is None:
            raise ValueError("Release Train 2 acceptance requires georeferenced OrthoLoC imagery")
        west, south, east, north = transform_bounds(
            src.crs,
            "EPSG:4326",
            *src.bounds,
            densify_pts=21,
        )

    x_min, y_top = _lonlat_to_tile(west, north, TERRARIUM_ZOOM)
    x_max, y_bottom = _lonlat_to_tile(east, south, TERRARIUM_ZOOM)
    x0, x1 = sorted((x_min, x_max))
    y0, y1 = sorted((y_top, y_bottom))
    width_tiles = x1 - x0 + 1
    height_tiles = y1 - y0 + 1
    tile_count = width_tiles * height_tiles
    if tile_count > 16:
        raise RuntimeError(
            f"OrthoLoC acceptance footprint unexpectedly requires {tile_count} Terrarium tiles"
        )

    mosaic = np.zeros(
        (height_tiles * TILE_SIZE, width_tiles * TILE_SIZE),
        dtype=np.float32,
    )
    for row, tile_y in enumerate(range(y0, y1 + 1)):
        for col, tile_x in enumerate(range(x0, x1 + 1)):
            url = TERRARIUM_TEMPLATE.format(z=TERRARIUM_ZOOM, x=tile_x, y=tile_y)
            cache = DATA_DIR / "terrarium" / f"z{TERRARIUM_ZOOM}_{tile_x}_{tile_y}.png"
            payload = _download(url, cache)
            rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"), dtype=np.float32)
            if rgb.shape != (TILE_SIZE, TILE_SIZE, 3):
                raise RuntimeError(f"unexpected Terrarium tile shape {rgb.shape} from {url}")
            elevation = rgb[..., 0] * 256.0 + rgb[..., 1] + rgb[..., 2] / 256.0 - 32768.0
            row0 = row * TILE_SIZE
            col0 = col * TILE_SIZE
            mosaic[row0 : row0 + TILE_SIZE, col0 : col0 + TILE_SIZE] = elevation

    bounds = _tile_bounds(
        x0,
        y0,
        TERRARIUM_ZOOM,
        width_tiles=width_tiles,
        height_tiles=height_tiles,
    )
    transform = from_bounds(*bounds, width=mosaic.shape[1], height=mosaic.shape[0])
    output = DATA_DIR / f"terrarium_z{TERRARIUM_ZOOM}_calibration_dem.tif"
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=mosaic.shape[0],
        width=mosaic.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:3857",
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
    ) as dst:
        dst.write(mosaic, 1)
        dst.set_band_description(1, "Independent low-resolution Terrarium calibration DEM")
        dst.update_tags(
            DEPTHWIZARD_ROLE="CALIBRATION_ONLY",
            DEM_SOURCE="AWS Terrain Tiles / Mapzen Terrarium",
            DEM_ENCODING="Terrarium",
            DEM_ZOOM=str(TERRARIUM_ZOOM),
        )
    return output


def _artifact_evidence(manifest: ProjectManifest) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for name in ("rdsm", "dsm", "slope", "reference", "residual", "metrics", "validation_report"):
        artifact = manifest.artifacts.get(name)
        if artifact is None:
            continue
        evidence[name] = {
            "path": artifact.get("path"),
            "sha256": artifact.get("sha256"),
            "semantics": artifact.get("semantics"),
            "units": artifact.get("units"),
        }
    return evidence


def main() -> None:
    """Run the final RT2 local integration gate with separated calibration/evaluation lineage.

    The OrthoLoC scene is not treated as unseen model-promotion evidence here. It is intentionally
    reused only as a product-integration acceptance scene. Metric scale comes from an independent
    Terrarium DEM; the TUM OrthoLoC DSM is loaded only after the production DSM is complete and is
    used exclusively by the downstream validation subsystem.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print("DepthWizard Release Train 2 acceptance")
    print("Purpose: integrated production validation smoke; NOT model promotion or unseen Gate B evidence")
    print("Preparing/caching OrthoLoC optical + DSM evaluation pair...")
    source_path, reference_path = _prepare_ortholoc_pair()
    print("Preparing independent AWS Terrain Tiles / Terrarium calibration DEM...")
    dem_path = _terrarium_dem_for_source(source_path)

    source_sha = sha256_file(source_path)
    dem_sha = sha256_file(dem_path)
    reference_sha = sha256_file(reference_path)
    if dem_sha == reference_sha:
        raise RuntimeError("calibration DEM and evaluation DSM unexpectedly have identical SHA-256")

    print("Running real production elevation runtime with independent low-resolution DEM evidence...")
    runtime_result = ProductionElevationRuntime().run(
        ProcessingRequest(
            source=source_path,
            output_dir=PROJECT_DIR,
            dem_path=dem_path,
            requested_output="dsm",
            tile_size=768,
            overlap=128,
        ),
        job_id="release-train-2-validation-smoke",
    )
    if runtime_result.status is not ProjectRunStatus.COMPLETE:
        raise RuntimeError(f"production runtime did not complete: {runtime_result.status.value}")

    print("Production DSM complete. Loading TUM OrthoLoC DSM as downstream evaluation-only evidence...")
    validation = validate_project_reference(
        ReferenceValidationRequest(
            project_dir=PROJECT_DIR,
            reference_path=reference_path,
            reference_label="TUM OrthoLoC urban_residential DSM / evaluation-only",
            min_valid_pixels=1000,
        )
    )
    manifest = ProjectManifest.load(PROJECT_DIR)

    acceptance = {
        "schema_version": 1,
        "status": "PASS_INTEGRATED_VALIDATION_PATH",
        "purpose": (
            "Release Train 2 production integration acceptance; not an unseen benchmark and not "
            "model-promotion evidence"
        ),
        "scientific_separation": {
            "calibration": "AWS Terrain Tiles / Mapzen Terrarium low-resolution DEM only",
            "evaluation": "TUM OrthoLoC urban_residential DSM only after production DSM completion",
            "calibration_dem_sha256": dem_sha,
            "evaluation_reference_sha256": reference_sha,
            "distinct_sha256": dem_sha != reference_sha,
            "note": (
                "OrthoLoC urban_residential has prior DepthWizard research use. This run validates "
                "the integrated product path and must not be presented as unseen Gate B evidence."
            ),
        },
        "source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha,
            "dataset": "TUM OrthoLoC demo / urban_residential",
            "dataset_license": "CC BY-NC-SA 4.0",
        },
        "calibration_evidence": {
            "path": str(dem_path.resolve()),
            "sha256": dem_sha,
            "source": "AWS Terrain Tiles / Mapzen Terrarium",
            "zoom": TERRARIUM_ZOOM,
        },
        "evaluation_reference": {
            "path": str(reference_path.resolve()),
            "sha256": reference_sha,
            "source": "TUM OrthoLoC DSM",
        },
        "runtime": runtime_result.as_dict(),
        "validation": validation.model_dump(mode="json"),
        "artifacts": _artifact_evidence(manifest),
        "manifest_warnings": manifest.warnings,
        "manifest_errors": manifest.errors,
    }
    report_path = PROJECT_DIR / "release-train-2-acceptance.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)

    print("DepthWizard Release Train 2 integrated validation path: PASS")
    print(f"Project: {PROJECT_DIR}")
    print(f"Calibration DEM SHA-256: {dem_sha}")
    print(f"Evaluation DSM SHA-256: {reference_sha}")
    print(f"Valid evaluation pixels: {validation.valid_pixels:,}")
    print(
        f"Elevation: RMSE {validation.elevation.rmse_m:.3f} m | "
        f"MAE {validation.elevation.mae_m:.3f} m | "
        f"P95 {validation.elevation.p95_abs_error_m:.3f} m | "
        f"r {validation.elevation.pearson_r}"
    )
    print(
        f"Slope: RMSE {validation.slope.rmse_degrees:.3f}° | "
        f"MAE {validation.slope.mae_degrees:.3f}°"
    )
    print("Claim boundary: engineering/product acceptance only; no model-promotion claim.")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
