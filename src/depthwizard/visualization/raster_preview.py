from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from depthwizard.pipeline.project import ProjectManifest

PreviewLayer = Literal["optical", "rdsm", "dsm", "slope", "reference", "residual", "confidence"]


def _preview_shape(height: int, width: int, max_side: int) -> tuple[int, int]:
    if max_side < 64:
        raise ValueError("max_side must be at least 64 pixels")
    scale = min(1.0, float(max_side) / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _finite_percentiles(values: np.ndarray, low: float, high: float) -> tuple[float, float]:
    finite = np.asarray(values, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("raster preview contains no finite pixels")
    lo, hi = np.percentile(finite, [low, high])
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("unable to derive finite raster preview range")
    if hi <= lo:
        hi = lo + 1.0
    return float(lo), float(hi)


def _normalize_scalar(values: np.ndarray) -> np.ndarray:
    lo, hi = _finite_percentiles(values, 2.0, 98.0)
    normalized = (values - lo) / (hi - lo)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)


def _scalar_rgb(values: np.ndarray, *, layer: PreviewLayer, valid: np.ndarray) -> np.ndarray:
    if layer == "residual":
        finite = np.abs(values[np.isfinite(values)])
        if finite.size == 0:
            raise ValueError("residual preview contains no finite pixels")
        limit = max(float(np.percentile(finite, 95.0)), 1e-6)
        signed = np.clip(values / limit, -1.0, 1.0)
        rgb = np.empty((*values.shape, 3), dtype=np.float32)
        positive = signed >= 0
        magnitude = np.abs(signed)
        # Neutral centre with restrained red/blue scientific divergence.
        rgb[..., 0] = np.where(positive, 1.0, 1.0 - 0.72 * magnitude)
        rgb[..., 1] = 1.0 - 0.80 * magnitude
        rgb[..., 2] = np.where(positive, 1.0 - 0.72 * magnitude, 1.0)
    else:
        normalized = _normalize_scalar(values)
        if layer == "confidence":
            # Confidence remains model-native; preview intensity must not imply calibrated probability.
            rgb = np.stack((normalized, normalized, normalized), axis=-1)
        elif layer == "slope":
            # Warm neutral ramp improves relief readability while preserving a monotonic scalar map.
            rgb = np.stack(
                (
                    0.22 + 0.68 * normalized,
                    0.24 + 0.56 * normalized,
                    0.27 + 0.38 * normalized,
                ),
                axis=-1,
            )
        else:
            # Elevation products use a restrained terrain-like monotonic ramp.
            rgb = np.stack(
                (
                    0.16 + 0.64 * normalized,
                    0.25 + 0.62 * normalized,
                    0.34 + 0.48 * normalized,
                ),
                axis=-1,
            )
    rgb[~valid] = np.array([0.94, 0.94, 0.94], dtype=np.float32)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _read_optical(path: Path, *, max_side: int) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError("optical preview requires at least three raster bands")
        out_h, out_w = _preview_shape(src.height, src.width, max_side)
        sample = src.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)
    channels: list[np.ndarray] = []
    for band in range(3):
        channel = np.asarray(sample[band].filled(np.nan), dtype=np.float32)
        lo, hi = _finite_percentiles(channel, 1.0, 99.0)
        normalized = np.clip((channel - lo) / (hi - lo), 0.0, 1.0)
        normalized[~np.isfinite(normalized)] = 0.0
        channels.append(normalized)
    return np.clip(np.rint(np.stack(channels, axis=-1) * 255.0), 0, 255).astype(np.uint8)


def _read_scalar(path: Path, *, layer: PreviewLayer, max_side: int) -> np.ndarray:
    with rasterio.open(path) as src:
        out_h, out_w = _preview_shape(src.height, src.width, max_side)
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
    values[~valid] = np.nan
    return _scalar_rgb(values, layer=layer, valid=valid)


def render_project_layer_preview(
    project_dir: str | Path,
    layer: PreviewLayer,
    *,
    max_side: int = 1600,
) -> bytes:
    manifest = ProjectManifest.load(project_dir)
    if layer == "optical":
        path = manifest.source_path
        pixels = _read_optical(path, max_side=max_side)
    else:
        path = manifest.artifact_path(layer)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"project layer '{layer}' is not available")
        pixels = _read_scalar(path, layer=layer, max_side=max_side)

    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
