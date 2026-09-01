from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass(frozen=True)
class StructureHeightEstimate:
    top_elevation_m: float
    ground_elevation_m: float
    structure_height_m: float
    structure_pixels: int
    ground_pixels: int
    ground_inlier_pixels: int
    ground_inlier_fraction: float
    ground_sector_coverage: float
    roof_inset_m: float
    ground_inner_buffer_m: float
    ground_outer_buffer_m: float
    roof_dispersion_m: float
    ground_residual_sigma_m: float
    local_height_dispersion_m: float


@dataclass(frozen=True)
class _GroundPlaneFit:
    surface: np.ndarray
    inliers: np.ndarray
    residual_sigma_m: float


def _robust_sigma(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    return max(1.4826 * mad, 1e-6)


def _robust_ground_plane(
    elevation: np.ndarray,
    ring: np.ndarray,
    *,
    iterations: int = 10,
) -> _GroundPlaneFit:
    """Fit a robust local z=a*x+b*y+c ground plane from surrounding candidates.

    High positive residuals are aggressively downweighted because they are more likely to be
    trees, neighbouring structures, vehicles, or DSM edge artefacts than valid bare ground.
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
    coefficients = np.linalg.lstsq(design, z, rcond=None)[0]

    for _ in range(iterations):
        predicted = design @ coefficients
        residual = z - predicted
        center = float(np.median(residual))
        sigma = _robust_sigma(residual - center)
        scaled = np.abs(residual - center) / (1.5 * sigma)
        weights = np.ones_like(scaled)
        large = scaled > 1.0
        weights[large] = 1.0 / scaled[large]
        high_objects = residual - center > 1.5 * sigma
        weights[high_objects] *= 0.10
        extreme_objects = residual - center > 3.0 * sigma
        weights[extreme_objects] *= 0.05

        if int(np.count_nonzero(weights > 1e-4)) < 3:
            break
        root_weights = np.sqrt(weights)
        weighted_design = design * root_weights[:, None]
        weighted_z = z * root_weights
        next_coefficients = np.linalg.lstsq(weighted_design, weighted_z, rcond=None)[0]
        if np.max(np.abs(next_coefficients - coefficients)) < 1e-8:
            coefficients = next_coefficients
            break
        coefficients = next_coefficients

    residual = z - design @ coefficients
    center = float(np.median(residual))
    sigma = _robust_sigma(residual - center)
    inliers_1d = (residual - center >= -3.0 * sigma) & (residual - center <= 2.0 * sigma)
    if int(np.count_nonzero(inliers_1d)) >= 3:
        coefficients = np.linalg.lstsq(design[inliers_1d], z[inliers_1d], rcond=None)[0]
        residual = z - design @ coefficients
        center = float(np.median(residual[inliers_1d]))
        sigma = _robust_sigma(residual[inliers_1d] - center)
        inliers_1d = (residual - center >= -3.0 * sigma) & (
            residual - center <= 2.0 * sigma
        )

    all_rows, all_cols = np.indices(elevation.shape, dtype=np.float64)
    all_x = (all_cols - x_center) / x_scale
    all_y = (all_rows - y_center) / y_scale
    surface = coefficients[0] * all_x + coefficients[1] * all_y + coefficients[2]

    inliers = np.zeros(ring.shape, dtype=bool)
    inliers[rows[inliers_1d], cols[inliers_1d]] = True
    final_residual = elevation[inliers] - surface[inliers]
    return _GroundPlaneFit(
        surface=surface,
        inliers=inliers,
        residual_sigma_m=float(_robust_sigma(final_residual)),
    )


def _metric_roof_core(
    mask: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
    requested_inset_m: float,
    min_pixels: int,
) -> tuple[np.ndarray, float]:
    if requested_inset_m <= 0.0:
        return mask.copy(), 0.0

    distance_inside_m = distance_transform_edt(
        mask,
        sampling=(pixel_size_y_m, pixel_size_x_m),
    )
    for factor in (1.0, 0.75, 0.5, 0.25):
        inset = requested_inset_m * factor
        core = mask & (distance_inside_m >= inset)
        if int(np.count_nonzero(core)) >= min_pixels:
            return core, float(inset)
    if int(np.count_nonzero(mask)) >= min_pixels:
        return mask.copy(), 0.0
    raise ValueError("not enough valid structure pixels after roof-edge exclusion")


def _ground_sector_coverage(
    structure_mask: np.ndarray,
    ground_inliers: np.ndarray,
    *,
    pixel_size_x_m: float,
    pixel_size_y_m: float,
) -> float:
    structure_rows, structure_cols = np.nonzero(structure_mask)
    ground_rows, ground_cols = np.nonzero(ground_inliers)
    if structure_rows.size == 0 or ground_rows.size == 0:
        return 0.0
    center_row = float(np.mean(structure_rows))
    center_col = float(np.mean(structure_cols))
    dy = (ground_rows.astype(np.float64) - center_row) * pixel_size_y_m
    dx = (ground_cols.astype(np.float64) - center_col) * pixel_size_x_m
    angles = np.arctan2(dy, dx)
    sectors = np.floor((angles + np.pi) / (np.pi / 2.0)).astype(np.int64) % 4
    return float(np.unique(sectors).size / 4.0)


def estimate_structure_height(
    dsm: np.ndarray,
    structure_mask: np.ndarray,
    *,
    ring_pixels: int | None = 8,
    min_structure_pixels: int = 4,
    min_ground_pixels: int = 8,
    pixel_size_x_m: float | None = None,
    pixel_size_y_m: float | None = None,
    roof_inset_m: float = 0.0,
    ground_inner_buffer_m: float = 0.0,
    ground_outer_buffer_m: float | None = None,
    ground_candidate_mask: np.ndarray | None = None,
) -> StructureHeightEstimate:
    """Estimate structure height above a robust local ground surface.

    Metric mode is selected when physical pixel sizes and ``ground_outer_buffer_m`` are supplied.
    It uses Euclidean distance in metres, an inner ground-exclusion buffer, and roof-edge inset.
    ``ground_candidate_mask`` is optional explicit semantic evidence; when supplied, only those
    pixels may support the local terrain plane. Legacy pixel mode is retained for low-level
    compatibility and synthetic tests.
    """
    elevation = np.asarray(dsm, dtype=np.float64)
    mask = np.asarray(structure_mask, dtype=bool)
    if elevation.shape != mask.shape:
        raise ValueError("dsm and structure_mask must have identical shape")
    if min_structure_pixels < 1 or min_ground_pixels < 1:
        raise ValueError("minimum support counts must be positive")

    valid = np.isfinite(elevation)
    candidate_mask: np.ndarray | None = None
    if ground_candidate_mask is not None:
        candidate_mask = np.asarray(ground_candidate_mask, dtype=bool)
        if candidate_mask.shape != elevation.shape:
            raise ValueError("ground_candidate_mask must match dsm shape")

    valid_structure = mask & valid
    if int(np.count_nonzero(valid_structure)) < min_structure_pixels:
        raise ValueError("not enough valid structure pixels")

    metric_mode = (
        pixel_size_x_m is not None
        or pixel_size_y_m is not None
        or ground_outer_buffer_m is not None
        or roof_inset_m > 0.0
        or ground_inner_buffer_m > 0.0
    )

    if metric_mode:
        if pixel_size_x_m is None or pixel_size_y_m is None:
            raise ValueError("metric structure sampling requires both physical pixel sizes")
        if not np.isfinite(pixel_size_x_m) or pixel_size_x_m <= 0.0:
            raise ValueError("pixel_size_x_m must be a positive finite value")
        if not np.isfinite(pixel_size_y_m) or pixel_size_y_m <= 0.0:
            raise ValueError("pixel_size_y_m must be a positive finite value")
        if ground_outer_buffer_m is None:
            raise ValueError("metric structure sampling requires ground_outer_buffer_m")
        if ground_inner_buffer_m < 0.0 or ground_outer_buffer_m <= ground_inner_buffer_m:
            raise ValueError("ground buffers must satisfy 0 <= inner < outer")
        if roof_inset_m < 0.0:
            raise ValueError("roof_inset_m cannot be negative")

        roof_core, effective_roof_inset_m = _metric_roof_core(
            valid_structure,
            pixel_size_x_m=pixel_size_x_m,
            pixel_size_y_m=pixel_size_y_m,
            requested_inset_m=roof_inset_m,
            min_pixels=min_structure_pixels,
        )
        distance_from_structure_m = distance_transform_edt(
            ~mask,
            sampling=(pixel_size_y_m, pixel_size_x_m),
        )
        ring = (
            ~mask
            & valid
            & (distance_from_structure_m >= ground_inner_buffer_m)
            & (distance_from_structure_m <= ground_outer_buffer_m)
        )
        effective_inner_m = float(ground_inner_buffer_m)
        effective_outer_m = float(ground_outer_buffer_m)
        sector_pixel_x = float(pixel_size_x_m)
        sector_pixel_y = float(pixel_size_y_m)
    else:
        if ring_pixels is None or ring_pixels <= 0:
            raise ValueError("ring_pixels must be positive in legacy pixel mode")
        expanded = np.asarray(binary_dilation(mask, iterations=ring_pixels), dtype=bool)
        ring = expanded & ~mask & valid
        roof_core = valid_structure
        effective_roof_inset_m = 0.0
        effective_inner_m = 0.0
        effective_outer_m = float(ring_pixels)
        sector_pixel_x = 1.0
        sector_pixel_y = 1.0

    if candidate_mask is not None:
        ring &= candidate_mask

    ground_pixels = int(np.count_nonzero(ring))
    if ground_pixels < min_ground_pixels:
        raise ValueError("not enough surrounding valid ground candidates")

    ground_fit = _robust_ground_plane(elevation, ring)
    ground_inlier_pixels = int(np.count_nonzero(ground_fit.inliers))
    if ground_inlier_pixels < min_ground_pixels:
        raise ValueError("not enough robust local-ground inliers")

    roof_values = elevation[roof_core]
    ground_under_roof = ground_fit.surface[roof_core]
    local_heights = roof_values - ground_under_roof

    sector_coverage = _ground_sector_coverage(
        mask,
        ground_fit.inliers,
        pixel_size_x_m=sector_pixel_x,
        pixel_size_y_m=sector_pixel_y,
    )
    return StructureHeightEstimate(
        top_elevation_m=float(np.median(roof_values)),
        ground_elevation_m=float(np.median(ground_under_roof)),
        structure_height_m=float(np.median(local_heights)),
        structure_pixels=int(np.count_nonzero(roof_core)),
        ground_pixels=ground_pixels,
        ground_inlier_pixels=ground_inlier_pixels,
        ground_inlier_fraction=float(ground_inlier_pixels / max(ground_pixels, 1)),
        ground_sector_coverage=sector_coverage,
        roof_inset_m=effective_roof_inset_m,
        ground_inner_buffer_m=effective_inner_m,
        ground_outer_buffer_m=effective_outer_m,
        roof_dispersion_m=float(_robust_sigma(roof_values)),
        ground_residual_sigma_m=float(ground_fit.residual_sigma_m),
        local_height_dispersion_m=float(_robust_sigma(local_heights)),
    )
