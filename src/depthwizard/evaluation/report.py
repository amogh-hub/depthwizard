from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

from depthwizard.evaluation.metrics import compute_elevation_metrics, compute_slope_metrics
from depthwizard.io.raster import reproject_to_match, write_float_geotiff


def validate_geospatial_dsm(
    prediction_path: str | Path,
    reference_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Run official metrics on an exactly aligned reference and emit reusable evidence artifacts."""
    prediction_path = Path(prediction_path)
    reference_path = Path(reference_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with rasterio.open(prediction_path) as pred_src:
        if pred_src.crs is None:
            raise ValueError("metric DSM validation requires a georeferenced prediction")
        pred = pred_src.read(1).astype(np.float32)
        valid = np.isfinite(pred)
        if pred_src.nodata is not None:
            valid &= pred != pred_src.nodata
        gsd_x = abs(pred_src.transform.a)
        gsd_y = abs(pred_src.transform.e)

    ref, ref_valid = reproject_to_match(reference_path, prediction_path)
    valid &= ref_valid
    metrics = compute_elevation_metrics(pred, ref, valid_mask=valid)
    slope_metrics = compute_slope_metrics(
        pred,
        ref,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        valid_mask=valid,
    )

    residual = np.full(pred.shape, np.nan, dtype=np.float32)
    residual[valid] = pred[valid] - ref[valid]
    write_float_geotiff(
        out / "residual.tif",
        residual,
        template_path=prediction_path,
        description="DepthWizard prediction minus reference elevation (metres)",
        tags={"DEPTHWIZARD_PRODUCT": "DSM_RESIDUAL_METRES"},
    )

    payload = {
        "official": metrics.model_dump(),
        "diagnostics": {"slope": slope_metrics.model_dump()},
    }
    (out / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
