from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class SeamConsistencyMetrics:
    tile_size: int
    overlap: int
    stride: int
    vertical_seams: int
    horizontal_seams: int
    seam_samples: int
    interior_samples: int
    seam_mean_abs_jump: float
    seam_p95_abs_jump: float
    interior_mean_abs_gradient: float
    interior_p95_abs_gradient: float
    p95_seam_to_interior_ratio: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return flattened[np.isfinite(flattened)]


def evaluate_tiling_seams(
    surface: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
    valid_mask: np.ndarray | None = None,
) -> SeamConsistencyMetrics:
    """Measure discontinuities at deterministic tile-reconstruction boundaries.

    This evaluator is deliberately model-agnostic. It does not declare a surface accurate; it asks
    whether the assembled raster has anomalous jumps specifically at the boundaries implied by the
    tiling configuration. Seam jumps are compared with ordinary one-pixel gradients elsewhere in
    the same surface, which makes the metric usable for both relative and metric elevation.
    """
    values = np.asarray(surface, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("surface must be a 2D raster")
    if tile_size < 2:
        raise ValueError("tile_size must be >= 2")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap
    if stride < 1:
        raise ValueError("tiling stride must be positive")

    valid = np.isfinite(values)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != values.shape:
            raise ValueError("valid_mask must match surface shape")
        valid &= supplied

    height, width = values.shape
    vertical_boundaries = list(range(stride, width, stride))
    horizontal_boundaries = list(range(stride, height, stride))
    if not vertical_boundaries and not horizontal_boundaries:
        raise ValueError("surface is too small to contain a tile boundary for this configuration")

    dx = np.abs(np.diff(values, axis=1))
    dy = np.abs(np.diff(values, axis=0))
    dx_valid = valid[:, 1:] & valid[:, :-1]
    dy_valid = valid[1:, :] & valid[:-1, :]

    seam_dx = np.zeros(dx.shape, dtype=bool)
    for boundary in vertical_boundaries:
        seam_dx[:, boundary - 1] = True
    seam_dy = np.zeros(dy.shape, dtype=bool)
    for boundary in horizontal_boundaries:
        seam_dy[boundary - 1, :] = True

    seam_values = np.concatenate(
        [
            _finite(np.where(dx_valid & seam_dx, dx, np.nan)),
            _finite(np.where(dy_valid & seam_dy, dy, np.nan)),
        ]
    )
    interior_values = np.concatenate(
        [
            _finite(np.where(dx_valid & ~seam_dx, dx, np.nan)),
            _finite(np.where(dy_valid & ~seam_dy, dy, np.nan)),
        ]
    )
    if seam_values.size == 0:
        raise ValueError("no finite seam samples are available")
    if interior_values.size == 0:
        raise ValueError("no finite interior-gradient samples are available")

    seam_mean = float(np.mean(seam_values))
    seam_p95 = float(np.percentile(seam_values, 95))
    interior_mean = float(np.mean(interior_values))
    interior_p95 = float(np.percentile(interior_values, 95))
    ratio = None if interior_p95 <= 1e-12 else float(seam_p95 / interior_p95)
    return SeamConsistencyMetrics(
        tile_size=tile_size,
        overlap=overlap,
        stride=stride,
        vertical_seams=len(vertical_boundaries),
        horizontal_seams=len(horizontal_boundaries),
        seam_samples=int(seam_values.size),
        interior_samples=int(interior_values.size),
        seam_mean_abs_jump=seam_mean,
        seam_p95_abs_jump=seam_p95,
        interior_mean_abs_gradient=interior_mean,
        interior_p95_abs_gradient=interior_p95,
        p95_seam_to_interior_ratio=ratio,
    )
