from __future__ import annotations

import numpy as np


def cosine_window(height: int, width: int, floor: float = 1e-3) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    wy = np.hanning(height) if height > 1 else np.ones(1)
    wx = np.hanning(width) if width > 1 else np.ones(1)
    window = np.outer(wy, wx)
    return np.maximum(window, floor).astype(np.float32)


class WeightedTileAccumulator:
    """Seam-resistant raster accumulator for overlapping model inference tiles."""

    def __init__(self, height: int, width: int) -> None:
        self._sum = np.zeros((height, width), dtype=np.float64)
        self._weight = np.zeros((height, width), dtype=np.float64)

    @property
    def shape(self) -> tuple[int, int]:
        return self._sum.shape

    def add(self, tile: np.ndarray, y: int, x: int, weight: np.ndarray | None = None) -> None:
        t = np.asarray(tile, dtype=np.float64)
        if t.ndim != 2:
            raise ValueError("tile must be a 2D raster")
        h, w = t.shape
        if y < 0 or x < 0 or y + h > self._sum.shape[0] or x + w > self._sum.shape[1]:
            raise ValueError("tile placement exceeds accumulator bounds")
        ww = cosine_window(h, w) if weight is None else np.asarray(weight, dtype=np.float64)
        if ww.shape != t.shape:
            raise ValueError("weight must match tile shape")
        valid = np.isfinite(t)
        self._sum[y:y+h, x:x+w][valid] += t[valid] * ww[valid]
        self._weight[y:y+h, x:x+w][valid] += ww[valid]

    def current_region(self, y: int, x: int, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
        if y < 0 or x < 0 or y + height > self._sum.shape[0] or x + width > self._sum.shape[1]:
            raise ValueError("region exceeds accumulator bounds")
        sums = self._sum[y:y+height, x:x+width]
        weights = self._weight[y:y+height, x:x+width]
        valid = weights > 0
        estimate = np.full((height, width), np.nan, dtype=np.float64)
        estimate[valid] = sums[valid] / weights[valid]
        return estimate, valid

    def finalize(self, nodata: float = np.nan) -> np.ndarray:
        out = np.full(self._sum.shape, nodata, dtype=np.float32)
        valid = self._weight > 0
        out[valid] = (self._sum[valid] / self._weight[valid]).astype(np.float32)
        return out
