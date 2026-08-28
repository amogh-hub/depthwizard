from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation


@dataclass(frozen=True)
class StructureHeightEstimate:
    top_elevation_m: float
    ground_elevation_m: float
    structure_height_m: float
    structure_pixels: int
    ground_pixels: int


def _robust_ground_plane(
    elevation: np.ndarray,
    ring: np.ndarray,
    *,
    iterations: int = 8,
) -> np.ndarray:
    """Fit a robust local z=a*x+b*y+c ground plane from surrounding ring pixels.

    A plane is preferable to a single ring median on sloping terrain. Iteratively reweighted least
    squares suppresses trees, neighbouring structures and isolated DSM spikes without pretending the
    selected footprint itself is ground evidence.
    """
    rows, cols = np.nonzero(ring)
    z = elevation[rows, cols].astype(np.float64, copy=False)
    if z.size < 3:
        raise ValueError("not enough surrounding valid ground candidates")

    x_center = float(np.mean(cols))
    y_center = float(np.mean(rows))
    x_scale = max(float(np.std(cols)), 1.0)
    y_scale = max(float(np.std(rows)), 1.0)
    x = (cols.astype(np.float64) - x_center) / x_scale
    y = (rows.astype(np.float64) - y_center) / y_scale
    design = np.column_stack([x, y, np.ones_like(x)])
    weights = np.ones(z.shape, dtype=np.float64)
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]

    for _ in range(iterations):
        predicted = design @ coefficients
        residual = z - predicted
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        sigma = max(1.4826 * mad, 1e-6)
        normalized = np.abs(residual - center) / (1.5 * sigma)
        weights = np.ones_like(normalized)
        large = normalized > 1.0
        weights[large] = 1.0 / normalized[large]
        weights[residual - center > 2.5 * sigma] *= 0.25
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_z = z * np.sqrt(weights)
        next_coefficients = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)[0]
        if np.max(np.abs(next_coefficients - coefficients)) < 1e-8:
            coefficients = next_coefficients
            break
        coefficients = next_coefficients

    all_rows, all_cols = np.indices(elevation.shape, dtype=np.float64)
    all_x = (all_cols - x_center) / x_scale
    all_y = (all_rows - y_center) / y_scale
    return coefficients[0] * all_x + coefficients[1] * all_y + coefficients[2]


def estimate_structure_height(
    dsm: np.ndarray,
    structure_mask: np.ndarray,
    *,
    ring_pixels: int = 8,
    min_structure_pixels: int = 4,
    min_ground_pixels: int = 8,
) -> StructureHeightEstimate:
    """Estimate object height above a robust local ground surface.

    The structure mask is explicit analyst/semantic evidence. DepthWizard never pretends a raw DSM
    can identify a building footprint by itself. Ground is estimated only from a surrounding ring;
    a robust local plane is extrapolated beneath the selected footprint so hillside structures are
    not biased by a single flat ring median.
    """
    elevation = np.asarray(dsm, dtype=np.float64)
    mask = np.asarray(structure_mask, dtype=bool)
    if elevation.shape != mask.shape:
        raise ValueError("dsm and structure_mask must have identical shape")
    valid_structure = mask & np.isfinite(elevation)
    if int(valid_structure.sum()) < min_structure_pixels:
        raise ValueError("not enough valid structure pixels")
    if ring_pixels <= 0:
        raise ValueError("ring_pixels must be positive")

    expanded = np.asarray(binary_dilation(mask, iterations=ring_pixels), dtype=bool)
    ring = expanded & ~mask & np.isfinite(elevation)
    if int(ring.sum()) < min_ground_pixels:
        raise ValueError("not enough surrounding valid ground candidates")

    ground_surface = _robust_ground_plane(elevation, ring)
    top = float(np.median(elevation[valid_structure]))
    ground_under_structure = ground_surface[valid_structure]
    ground = float(np.median(ground_under_structure))
    local_heights = elevation[valid_structure] - ground_under_structure
    height = float(np.median(local_heights))
    return StructureHeightEstimate(
        top_elevation_m=top,
        ground_elevation_m=ground,
        structure_height_m=height,
        structure_pixels=int(valid_structure.sum()),
        ground_pixels=int(ring.sum()),
    )
