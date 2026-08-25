from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
import typer
from rich import print

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.evaluation.report import validate_geospatial_dsm
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.raster import (
    inspect_raster,
    read_rgb,
    reproject_to_match,
    write_float_geotiff,
    write_relative_tiff,
)
from depthwizard.mesh.terrain import export_lod_pyramid
from depthwizard.pipeline.geometry import infer_geometry_scene

app = typer.Typer(no_args_is_help=True, help="DepthWizard engineering CLI")


@app.command()
def inspect(path: Path) -> None:
    """Inspect an input raster and classify its geospatial metadata state."""
    meta = inspect_raster(path)
    print(meta.model_dump_json(indent=2))


@app.command("reconstruct-da3")
def reconstruct_da3(
    source: Path,
    output_dir: Path,
    tile_size: int = typer.Option(1024, min=256, help="Inference tile edge in source pixels"),
    overlap: int = typer.Option(128, min=0, help="Tile overlap in source pixels"),
    harmonize_overlaps: bool = typer.Option(
        True,
        help="Robustly scale/offset harmonize overlapping monocular tiles",
    ),
) -> None:
    """Generate a production rDSM from RGB imagery with the pinned DA3MONO-LARGE prior."""
    if not source.exists():
        raise typer.BadParameter(f"source does not exist: {source}")
    if source.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
        raise typer.BadParameter("source must be PNG, JPG/JPEG, TIFF, or GeoTIFF")
    if overlap >= tile_size:
        raise typer.BadParameter("overlap must be smaller than tile_size")

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    output_dir.mkdir(parents=True, exist_ok=True)

    meta = inspect_raster(source)
    prior = DA3MonocularPrior(device="auto")

    started = time.perf_counter()
    scene = infer_geometry_scene(
        source,
        prior,
        tile_size=tile_size,
        overlap=overlap,
        harmonize_overlaps=harmonize_overlaps,
    )
    elapsed = time.perf_counter() - started

    rdsm_path = output_dir / "rdsm.tif"
    georeferenced = meta.crs is not None and meta.transform is not None
    if georeferenced:
        write_float_geotiff(
            rdsm_path,
            scene.relative_height,
            template_path=source,
            description="DepthWizard relative DSM (dimensionless)",
            tags={
                "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
                "ELEVATION_UNITS": "relative",
                "MODEL_ID": scene.model_id,
            },
        )
    else:
        write_relative_tiff(rdsm_path, scene.relative_height)

    confidence_path: Path | None = None
    if scene.confidence is not None:
        confidence_path = output_dir / "confidence.npy"
        np.save(confidence_path, scene.confidence.astype(np.float32, copy=False))

    report = {
        "status": "PASS",
        "source": str(source.resolve()),
        "product": "rDSM",
        "units": "dimensionless_relative_elevation",
        "georeferenced": georeferenced,
        "model_id": scene.model_id,
        "device": prior._resolved_device or "unknown",
        "shape": list(scene.relative_height.shape),
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": scene.tile_count,
        "harmonized_tiles": scene.harmonized_tiles,
        "normalization": asdict(scene.normalization),
        "wall_time_seconds": elapsed,
        "rdsm": str(rdsm_path.resolve()),
        "confidence": str(confidence_path.resolve()) if confidence_path is not None else None,
    }
    report_path = output_dir / "reconstruction_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[bold green]DepthWizard DA3 reconstruction: PASS[/bold green]")
    print(f"Model: {scene.model_id}")
    print(f"Device: {report['device']}")
    print(f"Product: {rdsm_path}")
    print(f"Tiles: {scene.tile_count} ({scene.harmonized_tiles} harmonized)")
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Report: {report_path}")


@app.command("mesh-rdsm")
def mesh_rdsm(
    rdsm: Path,
    texture: Path,
    output_dir: Path,
    vertical_scale: float = typer.Option(
        1.0,
        min=1e-6,
        help="Visualization-only multiplier for dimensionless rDSM elevation",
    ),
) -> None:
    """Export textured GLB LOD terrain from a DepthWizard rDSM and matching RGB source."""
    if not rdsm.exists():
        raise typer.BadParameter(f"rDSM does not exist: {rdsm}")
    if not texture.exists():
        raise typer.BadParameter(f"texture does not exist: {texture}")

    with rasterio.open(rdsm) as src:
        elevation = src.read(1).astype(np.float32)
        valid = np.isfinite(elevation)
        if src.nodata is not None:
            valid &= elevation != src.nodata
        if not np.all(valid):
            raise typer.BadParameter("mesh-rdsm currently requires a fully valid rDSM raster")
        gsd_x = abs(float(src.transform.a)) if not src.transform.is_identity else 1.0
        gsd_y = abs(float(src.transform.e)) if not src.transform.is_identity else 1.0
        crs = src.crs.to_string() if src.crs is not None else None

    rgb = read_rgb(texture)
    if rgb.shape[:2] != elevation.shape:
        raise typer.BadParameter(
            f"texture shape {rgb.shape[:2]} does not match rDSM shape {elevation.shape}"
        )

    visual_elevation = elevation * np.float32(vertical_scale)
    started = time.perf_counter()
    results = export_lod_pyramid(
        output_dir,
        visual_elevation,
        rgb,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        strides=(1, 2, 4, 8),
    )
    elapsed = time.perf_counter() - started

    report = {
        "status": "PASS",
        "product": "textured_terrain_lod",
        "source_rdsm": str(rdsm.resolve()),
        "texture": str(texture.resolve()),
        "crs": crs,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "vertical_scale": vertical_scale,
        "vertical_scale_semantics": "visualization_only_for_dimensionless_rdsm",
        "wall_time_seconds": elapsed,
        "lods": [
            {
                "path": str(result.path.resolve()),
                "stride": result.stride,
                "vertices": result.vertices,
                "faces": result.faces,
                "width_samples": result.width_samples,
                "height_samples": result.height_samples,
            }
            for result in results
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "mesh_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[bold green]DepthWizard terrain mesh export: PASS[/bold green]")
    print(f"CRS: {crs or 'none'}")
    print(f"Ground spacing: {gsd_x:.3f} x {gsd_y:.3f}")
    print(f"Visualization vertical scale: {vertical_scale:g}x")
    for index, result in enumerate(results):
        print(
            f"LOD{index}: {result.path.name} — {result.vertices:,} vertices / "
            f"{result.faces:,} faces"
        )
    print(f"Wall time: {elapsed:.2f} s")
    print(f"Report: {report_path}")


@app.command()
def evaluate(prediction: Path, reference: Path) -> None:
    """Compute official DSM metrics when rasters already share an identical grid."""
    with rasterio.open(prediction) as psrc, rasterio.open(reference) as rsrc:
        pred = psrc.read(1).astype(np.float64)
        ref = rsrc.read(1).astype(np.float64)
        valid = np.ones_like(pred, dtype=bool)
        if psrc.nodata is not None:
            valid &= pred != psrc.nodata
        if rsrc.nodata is not None:
            valid &= ref != rsrc.nodata
    metrics = compute_elevation_metrics(pred, ref, valid_mask=valid)
    print(json.dumps(metrics.model_dump(), indent=2))


@app.command("validate-dsm")
def validate_dsm(prediction: Path, reference: Path, output_dir: Path) -> None:
    """Align reference data to the prediction grid and emit SIH evidence artifacts."""
    payload = validate_geospatial_dsm(prediction, reference, output_dir)
    print(json.dumps(payload, indent=2))


@app.command("calibrate-dem")
def calibrate_dem(
    relative_height: Path,
    dem: Path,
    output: Path,
    sigma_px: float = typer.Option(24.0, help="Low-frequency residual smoothing scale in pixels"),
) -> None:
    """Calibrate a georeferenced relative-height raster against a DEM and emit metric DSM."""
    with rasterio.open(relative_height) as src:
        if src.crs is None:
            raise typer.BadParameter("relative-height input must be georeferenced for DEM calibration")
        rel = src.read(1).astype(np.float32)
        valid_rel = np.isfinite(rel)
        if src.nodata is not None:
            valid_rel &= rel != src.nodata

    aligned_dem, valid_dem = reproject_to_match(dem, relative_height)
    result = calibrate_relative_height_with_dem(
        rel,
        aligned_dem,
        dem_valid=valid_rel & valid_dem,
        low_frequency_sigma_px=sigma_px,
    )
    write_float_geotiff(
        output,
        result.dsm,
        template_path=relative_height,
        description="DepthWizard absolute Digital Surface Model (metres)",
        tags={
            "DEPTHWIZARD_PRODUCT": "ABSOLUTE_DSM_METRES",
            "CALIBRATION_METHOD": result.calibration.method,
        },
    )
    print(result.calibration.model_dump_json(indent=2))


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Loopback host only"),
    port: int = typer.Option(8765, min=1, max=65535),
) -> None:
    """Run the local DepthWizard core service for the desktop application."""
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise typer.BadParameter("DepthWizard core service may bind only to loopback")
    import uvicorn

    uvicorn.run("depthwizard.service:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
