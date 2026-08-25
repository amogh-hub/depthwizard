from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates


@dataclass(frozen=True)
class ElevationProfile:
    distance_m: np.ndarray
    elevation: np.ndarray
    start_pixel: tuple[float, float]
    end_pixel: tuple[float, float]


def sample_elevation_profile(
    elevation: np.ndarray,
    *,
    start_pixel: tuple[float, float],
    end_pixel: tuple[float, float],
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    samples: int = 256,
) -> ElevationProfile:
    """Sample a bilinear elevation transect between two (row, col) pixel coordinates."""
    raster = np.asarray(elevation, dtype=np.float64)
    if raster.ndim != 2:
        raise ValueError("elevation must be a 2D raster")
    if samples < 2:
        raise ValueError("samples must be >= 2")
    if gsd_x <= 0 or gsd_y <= 0:
        raise ValueError("gsd_x and gsd_y must be positive")

    r0, c0 = start_pixel
    r1, c1 = end_pixel
    rows = np.linspace(r0, r1, samples)
    cols = np.linspace(c0, c1, samples)
    values = map_coordinates(raster, [rows, cols], order=1, mode="constant", cval=np.nan)
    total_distance = float(np.hypot((c1 - c0) * gsd_x, (r1 - r0) * gsd_y))
    distance = np.linspace(0.0, total_distance, samples, dtype=np.float32)
    return ElevationProfile(
        distance_m=distance,
        elevation=values.astype(np.float32),
        start_pixel=start_pixel,
        end_pixel=end_pixel,
    )
