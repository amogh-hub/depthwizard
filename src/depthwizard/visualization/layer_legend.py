from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.visualization.raster_preview import PreviewLayer


def _surface_path(manifest: ProjectManifest) -> Path:
    for name in ("dsm", "rdsm"):
        path = manifest.artifact_path(name)
        if path is not None and path.is_file():
            return path
    raise FileNotFoundError("project has no persisted DSM/rDSM surface for layer legend")


def _layer_path(manifest: ProjectManifest, layer: PreviewLayer) -> Path:
    if layer in {"hillshade", "contours"}:
        return _surface_path(manifest)
    path = manifest.artifact_path(layer)
    if path is None or not path.is_file():
        raise FileNotFoundError(f"project layer '{layer}' is not available")
    return path


def _sample_values(path: Path, max_side: int = 1024) -> np.ndarray:
    with rasterio.open(path) as src:
        scale = min(1.0, float(max_side) / max(src.height, src.width))
        out_h = max(1, round(src.height * scale))
        out_w = max(1, round(src.width * scale))
        sample = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)
        values = np.asarray(sample.filled(np.nan), dtype=np.float32)
        valid = np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    finite = values[valid]
    if finite.size == 0:
        raise ValueError("project layer contains no finite values for legend")
    return finite.astype(np.float64, copy=False)


def _percentile_range(values: np.ndarray) -> tuple[float, float, float]:
    low, median, high = np.percentile(values, [2.0, 50.0, 98.0])
    if not all(np.isfinite(item) for item in (low, median, high)):
        raise ValueError("unable to derive finite layer legend range")
    if high <= low:
        high = low + 1.0
    return float(low), float(median), float(high)


def project_layer_legend(project_dir: str | Path, layer: PreviewLayer) -> dict[str, Any]:
    """Return display-only quantitative legend semantics for one persisted project layer.

    The ranges intentionally mirror the preview normalization contract. They describe the rendered
    workstation view and never replace the persisted numerical raster as the measurement source.
    """
    manifest = ProjectManifest.load(project_dir)
    if layer == "optical":
        return {
            "available": False,
            "layer": layer,
            "title": "Optical RGB",
            "units": None,
            "semantics": "source_rgb_no_scalar_legend",
            "ramp": "optical",
        }

    if layer == "hillshade":
        return {
            "available": True,
            "layer": layer,
            "title": "Hillshade illumination",
            "units": "relative illumination",
            "minimum": 0.0,
            "midpoint": 0.5,
            "maximum": 1.0,
            "semantics": "display_derivative_not_measurement_surface",
            "ramp": "grayscale",
        }

    path = _layer_path(manifest, layer)
    values = _sample_values(path)
    if layer == "residual":
        limit = max(float(np.percentile(np.abs(values), 95.0)), 1e-6)
        low, median, high = -limit, 0.0, limit
        units = "m"
        ramp = "diverging"
        semantics = "prediction_minus_reference_display_p95_symmetric_range"
    else:
        low, median, high = _percentile_range(values)
        units = {
            "rdsm": "relative",
            "dsm": "m",
            "slope": "°",
            "reference": "m",
            "confidence": "model-native",
            "contours": "m" if manifest.artifact_path("dsm") else "relative",
        }.get(layer)
        ramp = {
            "slope": "slope",
            "confidence": "grayscale",
            "contours": "contours",
        }.get(layer, "elevation")
        semantics = "display_p02_p98_range_from_persisted_project_raster"

    return {
        "available": True,
        "layer": layer,
        "title": {
            "rdsm": "Relative elevation",
            "dsm": "Elevation",
            "slope": "Surface slope",
            "reference": "Reference elevation",
            "residual": "Prediction − reference",
            "confidence": "Model-native confidence",
            "contours": "Contour elevation",
        }.get(layer, layer),
        "units": units,
        "minimum": low,
        "midpoint": median,
        "maximum": high,
        "semantics": semantics,
        "ramp": ramp,
        "sampled_values": int(values.size),
    }
