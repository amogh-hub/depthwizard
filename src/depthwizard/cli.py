from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
import typer
from rich import print

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.evaluation.report import validate_geospatial_dsm
from depthwizard.io.raster import inspect_raster, reproject_to_match, write_float_geotiff

app = typer.Typer(no_args_is_help=True, help="DepthWizard engineering CLI")


@app.command()
def inspect(path: Path) -> None:
    """Inspect an input raster and classify its geospatial metadata state."""
    meta = inspect_raster(path)
    print(meta.model_dump_json(indent=2))


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
