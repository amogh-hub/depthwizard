from __future__ import annotations

import io
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import Resampling

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.visualization.relative_scale import relative_display_vertical_scale

PreviewLayer = Literal[
    "optical",
    "rdsm",
    "dsm",
    "slope",
    "reference",
    "residual",
    "confidence",
    "hillshade",
    "contours",
]


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


def _read_scalar_values(path: Path, *, max_side: int) -> tuple[np.ndarray, np.ndarray]:
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
    return values, valid


def _read_scalar(path: Path, *, layer: PreviewLayer, max_side: int) -> np.ndarray:
    values, valid = _read_scalar_values(path, max_side=max_side)
    return _scalar_rgb(values, layer=layer, valid=valid)


def _relative_hillshade_contrast(intensity: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Stretch relative-only illumination for readable display without changing terrain semantics."""
    usable = valid & np.isfinite(intensity)
    finite = np.asarray(intensity[usable], dtype=np.float64)
    if finite.size < 4:
        return intensity.astype(np.float32)
    low, high = np.percentile(finite, [2.0, 98.0])
    if not np.isfinite(low) or not np.isfinite(high) or high <= low + 1e-8:
        return intensity.astype(np.float32)
    normalized = np.clip((intensity - low) / (high - low), 0.0, 1.0)
    # Keep a small white/black margin so clipped percentile tails remain visually distinguishable.
    stretched = 0.08 + 0.88 * normalized
    return stretched.astype(np.float32)


def _hillshade_rgb(path: Path, *, max_side: int, relative_surface: bool = False) -> np.ndarray:
    values, valid = _read_scalar_values(path, max_side=max_side)
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("hillshade preview requires at least a 2x2 surface")
    fill_value = float(np.nanmedian(values[valid]))
    fill = np.where(valid, values, fill_value).astype(np.float64)
    if relative_surface:
        # A dimensionless rDSM has no physical dz/dx angle. Use the same deterministic display-only
        # normalization policy as the 3D mesh so relief illumination remains visible without ever
        # claiming degrees or changing the persisted rDSM values.
        display_scale = relative_display_vertical_scale(
            fill,
            valid,
            span_x=max(values.shape[1] - 1, 1),
            span_y=max(values.shape[0] - 1, 1),
        )
        fill = (fill - fill_value) * display_scale
    grad_y, grad_x = np.gradient(fill)
    slope = np.arctan(np.hypot(grad_x, grad_y))
    aspect = np.arctan2(-grad_x, grad_y)
    azimuth = np.deg2rad(315.0)
    altitude = np.deg2rad(45.0)
    shaded = (
        np.sin(altitude) * np.cos(slope)
        + np.cos(altitude) * np.sin(slope) * np.cos(azimuth - aspect)
    )
    intensity = np.clip(0.5 + 0.5 * shaded, 0.0, 1.0).astype(np.float32)
    if relative_surface:
        # Relative display scaling makes gradients meaningful but the conventional illumination
        # equation still clusters ordinary terrain near ~0.85 white. Robust display-only stretching
        # restores readable shadow/highlight separation without altering metric DSM hillshade.
        intensity = _relative_hillshade_contrast(intensity, valid)
    rgb = np.stack((intensity, intensity, intensity), axis=-1)
    rgb[~valid] = np.array([0.94, 0.94, 0.94], dtype=np.float32)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _contour_rgb(path: Path, *, max_side: int) -> np.ndarray:
    values, valid = _read_scalar_values(path, max_side=max_side)
    lo, hi = _finite_percentiles(values, 2.0, 98.0)
    normalized = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
    normalized[~valid] = 0.0
    bins = np.floor(normalized * 12.0).astype(np.int16)
    edges = np.zeros(values.shape, dtype=bool)
    edges[:, 1:] |= bins[:, 1:] != bins[:, :-1]
    edges[1:, :] |= bins[1:, :] != bins[:-1, :]
    edges &= valid
    base = 0.94 - 0.24 * normalized
    rgb = np.stack((base, base, base), axis=-1)
    rgb[edges] = np.array([0.12, 0.18, 0.24], dtype=np.float32)
    rgb[~valid] = np.array([0.97, 0.97, 0.97], dtype=np.float32)
    return np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)


def _surface_path(manifest: ProjectManifest) -> tuple[Path, bool]:
    dsm = manifest.artifact_path("dsm")
    if dsm is not None and dsm.is_file():
        return dsm, False
    rdsm = manifest.artifact_path("rdsm")
    if rdsm is not None and rdsm.is_file():
        return rdsm, True
    raise FileNotFoundError("project has no persisted DSM/rDSM surface for derived visualization")


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
    elif layer == "hillshade":
        path, relative_surface = _surface_path(manifest)
        pixels = _hillshade_rgb(
            path,
            max_side=max_side,
            relative_surface=relative_surface,
        )
    elif layer == "contours":
        path, _relative_surface = _surface_path(manifest)
        pixels = _contour_rgb(path, max_side=max_side)
    else:
        path = manifest.artifact_path(layer)
        if path is None or not path.is_file():
            raise FileNotFoundError(f"project layer '{layer}' is not available")
        pixels = _read_scalar(path, layer=layer, max_side=max_side)

    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGB").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()
