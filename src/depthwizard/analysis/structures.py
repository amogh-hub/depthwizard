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


def estimate_structure_height(
    dsm: np.ndarray,
    structure_mask: np.ndarray,
    *,
    ring_pixels: int = 8,
    min_structure_pixels: int = 4,
    min_ground_pixels: int = 8,
) -> StructureHeightEstimate:
    """Estimate object height above a robust local-ground ring.

    The structure mask is expected to come from a semantic/interactive selection. The routine does
    not pretend a raw DSM can infer object footprint semantics by itself.
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

    expanded = binary_dilation(mask, iterations=ring_pixels)
    ring = expanded & ~mask & np.isfinite(elevation)
    if int(ring.sum()) < min_ground_pixels:
        raise ValueError("not enough surrounding valid ground candidates")

    # Median is intentionally robust to a few nearby objects contaminating the selection ring.
    top = float(np.median(elevation[valid_structure]))
    ground = float(np.median(elevation[ring]))
    return StructureHeightEstimate(
        top_elevation_m=top,
        ground_elevation_m=ground,
        structure_height_m=top - ground,
        structure_pixels=int(valid_structure.sum()),
        ground_pixels=int(ring.sum()),
    )
