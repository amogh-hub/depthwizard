from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from depthwizard.calibration.robust import robust_affine_calibration

_MIN_ALIGNMENT_CORRELATION = 0.15
_MIN_ALIGNMENT_SCALE = 0.25
_MAX_ALIGNMENT_SCALE = 4.0


def resize_field_to_shape(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly resize a 2D scientific field while preserving invalid support."""
    field = np.asarray(values, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("field must be a 2D raster")
    target_height, target_width = shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target shape must be positive")
    if field.shape == shape:
        return field.copy()

    valid = np.isfinite(field)
    if not np.any(valid):
        return np.full(shape, np.nan, dtype=np.float32)
    fill = float(np.median(field[valid]))
    filled = np.where(valid, field, fill).astype(np.float32, copy=False)
    factors = (target_height / field.shape[0], target_width / field.shape[1])
    resized = np.asarray(
        zoom(filled, factors, order=1, mode="nearest", prefilter=False),
        dtype=np.float32,
    )
    valid_resized = np.asarray(
        zoom(valid.astype(np.float32), factors, order=0, mode="nearest", prefilter=False),
        dtype=np.float32,
    )

    # scipy.ndimage.zoom normally lands exactly on the requested shape for these factors. Keep a
    # deterministic crop/pad fallback so a future SciPy rounding change cannot alter contracts.
    out = np.full(shape, fill, dtype=np.float32)
    h = min(target_height, resized.shape[0])
    w = min(target_width, resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    valid_out = np.zeros(shape, dtype=bool)
    vh = min(target_height, valid_resized.shape[0])
    vw = min(target_width, valid_resized.shape[1])
    valid_out[:vh, :vw] = valid_resized[:vh, :vw] >= 0.5
    out[~valid_out] = np.nan
    return out


def normalized_gaussian_filter(values: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian low-pass that does not let NaNs bleed into valid terrain."""
    field = np.asarray(values, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("field must be a 2D raster")
    if sigma_px <= 0:
        raise ValueError("sigma_px must be positive")
    valid = np.isfinite(field)
    if not np.any(valid):
        return np.full(field.shape, np.nan, dtype=np.float32)

    numerator = gaussian_filter(
        np.where(valid, field, 0.0).astype(np.float32),
        sigma=sigma_px,
        mode="nearest",
    )
    denominator = gaussian_filter(
        valid.astype(np.float32),
        sigma=sigma_px,
        mode="nearest",
    )
    out = np.full(field.shape, np.nan, dtype=np.float32)
    supported = denominator > 1e-6
    out[supported] = (numerator[supported] / denominator[supported]).astype(np.float32)
    return out


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.size < 2 or y64.size != x64.size:
        return None
    x64 = x64 - float(np.mean(x64))
    y64 = y64 - float(np.mean(y64))
    denominator = float(np.linalg.norm(x64) * np.linalg.norm(y64))
    if denominator <= 1e-12:
        return None
    value = float(np.dot(x64, y64) / denominator)
    return value if np.isfinite(value) else None


def align_tile_to_scaffold(
    tile: np.ndarray,
    scaffold: np.ndarray,
    *,
    lowpass_sigma_px: float,
    min_pixels: int = 256,
) -> tuple[np.ndarray, bool]:
    """Align a local monocular tile to the scene-wide scaffold in low-frequency space.

    A robust affine fit is accepted only when the smoothed fields are positively correlated and the
    scale is bounded. Otherwise only a robust additive baseline is applied. This keeps high-frequency
    roof/tree evidence from driving a scene-scale alignment fit.
    """
    local = np.asarray(tile, dtype=np.float32)
    base = np.asarray(scaffold, dtype=np.float32)
    if local.shape != base.shape or local.ndim != 2:
        raise ValueError("tile and scaffold must be same-shaped 2D rasters")

    local_low = normalized_gaussian_filter(local, lowpass_sigma_px)
    base_low = normalized_gaussian_filter(base, lowpass_sigma_px)
    valid = np.isfinite(local_low) & np.isfinite(base_low)
    if int(valid.sum()) < min_pixels:
        return local, False

    step = max(1, min(local.shape) // 256)
    sampled = valid[::step, ::step]
    local_values = local_low[::step, ::step][sampled].astype(np.float64)
    base_values = base_low[::step, ::step][sampled].astype(np.float64)
    if local_values.size < min_pixels:
        local_values = local_low[valid].astype(np.float64)
        base_values = base_low[valid].astype(np.float64)

    offset = float(np.median(base_values - local_values))
    offset_aligned = (local + offset).astype(np.float32)
    correlation = _pearson(local_values, base_values)
    if correlation is None or correlation < _MIN_ALIGNMENT_CORRELATION:
        return offset_aligned, abs(offset) > 1e-8

    try:
        fit = robust_affine_calibration(
            local_values,
            base_values,
            require_positive_scale=True,
        )
    except ValueError:
        return offset_aligned, abs(offset) > 1e-8
    if not (_MIN_ALIGNMENT_SCALE <= fit.scale <= _MAX_ALIGNMENT_SCALE):
        return offset_aligned, abs(offset) > 1e-8

    return (fit.scale * local + fit.offset).astype(np.float32), True


def high_frequency_residual(
    aligned_tile: np.ndarray,
    scaffold: np.ndarray,
    *,
    sigma_px: float,
) -> np.ndarray:
    """Keep local geometry while removing broad tile-specific disagreement.

    The residual is formed relative to the global scaffold, then its low-frequency component is
    removed. A tile therefore cannot contribute a broad 1024-pixel plateau, but roofs, vegetation,
    edges and other local geometry survive for the final mosaic and metric calibration.
    """
    local = np.asarray(aligned_tile, dtype=np.float32)
    base = np.asarray(scaffold, dtype=np.float32)
    if local.shape != base.shape or local.ndim != 2:
        raise ValueError("aligned tile and scaffold must be same-shaped 2D rasters")
    difference = local - base
    low_frequency = normalized_gaussian_filter(difference, sigma_px)
    residual = difference - low_frequency
    residual[~(np.isfinite(local) & np.isfinite(base))] = np.nan
    return residual.astype(np.float32, copy=False)
