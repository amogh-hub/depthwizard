from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling


@dataclass(frozen=True)
class RGBNormalizationStats:
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    low_percentile: float
    high_percentile: float


def estimate_rgb_normalization_stats(
    path: str | Path,
    *,
    band_indices: tuple[int, int, int] = (1, 2, 3),
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    max_sample_side: int = 2048,
) -> RGBNormalizationStats:
    """Estimate scene-level radiometric normalization from a bounded downsampled read."""
    with rasterio.open(path) as src:
        if any(index < 1 or index > src.count for index in band_indices):
            raise ValueError("RGB band selection exceeds source band count")
        scale = min(1.0, max_sample_side / max(src.width, src.height))
        out_h = max(1, round(src.height * scale))
        out_w = max(1, round(src.width * scale))
        sample = src.read(
            list(band_indices),
            out_shape=(3, out_h, out_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)

    low: list[float] = []
    high: list[float] = []
    for band in range(3):
        values = sample[band].compressed() if np.ma.isMaskedArray(sample[band]) else sample[band].reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"RGB band {band_indices[band]} contains no finite sample pixels")
        lo, hi = np.percentile(values, [low_percentile, high_percentile])
        low.append(float(lo))
        high.append(float(hi))
    return RGBNormalizationStats(
        low=(low[0], low[1], low[2]),
        high=(high[0], high[1], high[2]),
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )


def normalize_with_stats(rgb: np.ndarray, stats: RGBNormalizationStats) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape HxWx3")
    low = np.asarray(stats.low, dtype=np.float32).reshape(1, 1, 3)
    high = np.asarray(stats.high, dtype=np.float32).reshape(1, 1, 3)
    normalized = (image - low) / np.maximum(high - low, 1e-6)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype(np.float32)
