from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter
from scipy.spatial import ConvexHull, QhullError

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.contracts import CalibrationResult, GroundControlPoint

MINIMUM_GCP_COUNT = 4
RECOMMENDED_GCP_COUNT = 6


@dataclass(frozen=True)
class GCPCalibrationOutput:
    dsm: np.ndarray
    calibration: CalibrationResult
    gcp_residuals_m: np.ndarray
    sampled_relative_height: np.ndarray
    low_frequency_bias: np.ndarray
    orientation_flipped: bool
    anchor_correlation_before: float
    cross_validation_rmse_m: float
    spatial_coverage_fraction: float
    spatial_rank_ratio: float


@dataclass(frozen=True)
class GCPMetricValidationOutput:
    """GCP validation/correction result for a DSM already placed in metric elevation space."""

    dsm: np.ndarray
    gcp_residuals_before_m: np.ndarray
    gcp_residuals_after_m: np.ndarray
    offset_applied_m: float
    rmse_before_m: float
    rmse_after_m: float
    cross_validation_rmse_m: float
    spatial_coverage_fraction: float
    spatial_rank_ratio: float


def _bilinear_sample(array: np.ndarray, row: float, col: float) -> float:
    h, w = array.shape
    if row < 0 or col < 0 or row > h - 1 or col > w - 1:
        return float("nan")
    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)
    dy = row - r0
    dx = col - c0
    values = np.array([array[r0, c0], array[r0, c1], array[r1, c0], array[r1, c1]])
    if not np.all(np.isfinite(values)):
        return float("nan")
    top = values[0] * (1 - dx) + values[1] * dx
    bottom = values[2] * (1 - dx) + values[3] * dx
    return float(top * (1 - dy) + bottom * dy)


def _anchor_correlation(samples: np.ndarray, elevations: np.ndarray) -> float:
    if samples.size < 2 or np.ptp(samples) <= 1e-12 or np.ptp(elevations) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(samples, elevations)[0, 1])


def _sample_gcps(
    array: np.ndarray,
    *,
    transform: Affine,
    gcps: list[GroundControlPoint],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    inv = ~transform
    samples: list[float] = []
    elevations: list[float] = []
    weights: list[float] = []
    pixel_rows: list[float] = []
    pixel_cols: list[float] = []
    for gcp in gcps:
        col_corner, row_corner = inv * (gcp.x, gcp.y)
        col = col_corner - 0.5
        row = row_corner - 0.5
        value = _bilinear_sample(array, row, col)
        if np.isfinite(value):
            samples.append(value)
            elevations.append(gcp.elevation_m)
            weights.append(gcp.weight)
            pixel_rows.append(row)
            pixel_cols.append(col)
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (samples, elevations, weights, pixel_rows, pixel_cols)
    )  # type: ignore[return-value]


def _validate_spatial_distribution(
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    shape: tuple[int, int],
    min_gcps: int,
    min_axis_span_fraction: float,
    min_spatial_coverage_fraction: float,
    min_rank_ratio: float,
) -> tuple[float, float]:
    if rows.size < min_gcps:
        raise ValueError(
            f"fewer than {min_gcps} valid GCPs overlap finite raster pixels; "
            "insufficient evidence for defensible metric calibration"
        )
    points = np.column_stack([cols, rows])
    if np.unique(np.round(points, decimals=6), axis=0).shape[0] != points.shape[0]:
        raise ValueError("GCP evidence contains duplicate raster locations")

    height, width = shape
    col_span = float(np.ptp(cols) / max(width - 1, 1))
    row_span = float(np.ptp(rows) / max(height - 1, 1))
    if col_span < min_axis_span_fraction or row_span < min_axis_span_fraction:
        raise ValueError(
            "GCPs are too spatially clustered; distribute control across both raster axes"
        )

    centered = points - np.mean(points, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_ratio = (
        float(singular_values[1] / singular_values[0])
        if singular_values.size >= 2 and singular_values[0] > 1e-12
        else 0.0
    )
    if rank_ratio < min_rank_ratio:
        raise ValueError("GCPs are nearly collinear and cannot validate spatial calibration")
    try:
        hull_area = float(ConvexHull(points).volume)
    except QhullError as exc:
        raise ValueError("GCPs do not form a stable two-dimensional control footprint") from exc
    coverage = hull_area / max(float((width - 1) * (height - 1)), 1.0)
    if coverage < min_spatial_coverage_fraction:
        raise ValueError(
            "GCP convex-hull coverage is too small for a scene-level absolute-height claim"
        )
    return coverage, rank_ratio


def _weighted_rmse(residuals: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.average(np.asarray(residuals) ** 2, weights=weights)))


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, sorted_values.size - 1)])


def _affine_leave_one_out_rmse(
    samples: np.ndarray,
    elevations: np.ndarray,
    weights: np.ndarray,
) -> float:
    predictions = np.empty(samples.size, dtype=np.float64)
    for held_out in range(samples.size):
        keep = np.arange(samples.size) != held_out
        fit = robust_affine_calibration(
            samples[keep],
            elevations[keep],
            weights=weights[keep],
            require_positive_scale=True,
        )
        predictions[held_out] = fit.scale * samples[held_out] + fit.offset
    return _weighted_rmse(elevations - predictions, weights)


def calibrate_relative_height_with_gcps(
    relative_height: np.ndarray,
    *,
    transform: Affine,
    gcps: list[GroundControlPoint],
    low_frequency_sigma_px: float | None = None,
    resolve_orientation: bool = True,
    min_gcps: int = RECOMMENDED_GCP_COUNT,
    min_abs_anchor_correlation: float = 0.35,
    max_anchor_rmse_m: float | None = 10.0,
    max_cross_validation_rmse_m: float | None = 15.0,
    min_elevation_span_m: float = 1.0,
    min_axis_span_fraction: float = 0.20,
    min_spatial_coverage_fraction: float = 0.02,
    min_rank_ratio: float = 0.05,
) -> GCPCalibrationOutput:
    """Calibrate relative height with sparse metric Ground Control Points.

    GCP coordinates are interpreted in the same CRS as the source raster. Six or more reliable,
    spatially distributed points are required by default so scale/offset fitting retains independent
    quality evidence; four- or five-point fits require an explicit lower ``min_gcps`` override and
    must be reported as low-confidence by the caller. Two points merely determine the two affine
    parameters. When
    ``resolve_orientation`` is enabled, a negative GCP/relative-height correlation is recorded and
    the relative-height polarity is inverted before the physically constrained positive-scale fit.
    This mirrors the DEM calibration contract and prevents domain-shift polarity from being hidden
    inside an invalid negative metric scale.
    """
    rel = np.asarray(relative_height, dtype=np.float64)
    if rel.ndim != 2:
        raise ValueError("relative_height must be a 2D raster")
    if min_gcps < MINIMUM_GCP_COUNT:
        raise ValueError(
            f"min_gcps must be >= {MINIMUM_GCP_COUNT} to preserve independent quality evidence"
        )
    if len(gcps) < min_gcps:
        raise ValueError(f"at least {min_gcps} GCPs are required for metric calibration")
    if low_frequency_sigma_px is not None and low_frequency_sigma_px < 0:
        raise ValueError("low_frequency_sigma_px must be non-negative when supplied")
    if not 0.0 <= min_abs_anchor_correlation <= 1.0:
        raise ValueError("min_abs_anchor_correlation must be in [0, 1]")

    samples_array, elevations_array, weights_array, rows_array, cols_array = _sample_gcps(
        rel,
        transform=transform,
        gcps=gcps,
    )
    spatial_coverage, rank_ratio = _validate_spatial_distribution(
        rows_array,
        cols_array,
        shape=(rel.shape[0], rel.shape[1]),
        min_gcps=min_gcps,
        min_axis_span_fraction=min_axis_span_fraction,
        min_spatial_coverage_fraction=min_spatial_coverage_fraction,
        min_rank_ratio=min_rank_ratio,
    )
    if np.ptp(samples_array) <= 1e-8:
        raise ValueError("GCPs do not span enough relative-height variation to determine scale")
    if np.ptp(elevations_array) < min_elevation_span_m:
        raise ValueError(
            "GCP elevations do not span enough metric relief to determine a stable scale"
        )

    correlation_before = _anchor_correlation(samples_array, elevations_array)
    if not np.isfinite(correlation_before) or abs(correlation_before) < min_abs_anchor_correlation:
        raise ValueError(
            "relative height is too weakly correlated with GCP elevations for defensible metric "
            f"calibration (|r|={abs(correlation_before):.3f} < {min_abs_anchor_correlation:.3f})"
        )
    orientation_flipped = bool(
        resolve_orientation and np.isfinite(correlation_before) and correlation_before < 0
    )
    if not resolve_orientation and np.isfinite(correlation_before) and correlation_before < 0:
        raise ValueError(
            "GCP evidence implies inverted height orientation while orientation resolution is disabled"
        )

    oriented_rel = -rel if orientation_flipped else rel
    oriented_samples = -samples_array if orientation_flipped else samples_array
    fit = robust_affine_calibration(
        oriented_samples,
        elevations_array,
        weights=weights_array,
    )
    if not fit.converged:
        raise ValueError("GCP calibration fit did not converge")
    global_dsm = fit.scale * oriented_rel + fit.offset
    residuals = elevations_array - (fit.scale * oriented_samples + fit.offset)
    cross_validation_rmse = _affine_leave_one_out_rmse(
        oriented_samples,
        elevations_array,
        weights_array,
    )
    if max_anchor_rmse_m is not None and fit.rmse_anchor > max_anchor_rmse_m:
        raise ValueError(
            f"GCP anchor RMSE {fit.rmse_anchor:.3f} m exceeds the configured "
            f"{max_anchor_rmse_m:.3f} m quality limit"
        )
    if (
        max_cross_validation_rmse_m is not None
        and cross_validation_rmse > max_cross_validation_rmse_m
    ):
        raise ValueError(
            f"GCP leave-one-out RMSE {cross_validation_rmse:.3f} m exceeds the configured "
            f"{max_cross_validation_rmse_m:.3f} m quality limit"
        )

    # Sparse residual impulses are smoothed into a low-frequency correction field. Weight
    # normalization prevents regions far from any GCP from being forced toward zero residual.
    impulses = np.zeros(rel.shape, dtype=np.float64)
    support = np.zeros(rel.shape, dtype=np.float64)
    for row, col, residual, weight in zip(
        rows_array, cols_array, residuals, weights_array, strict=True
    ):
        rr = int(np.clip(round(row), 0, rel.shape[0] - 1))
        cc = int(np.clip(round(col), 0, rel.shape[1] - 1))
        impulses[rr, cc] += residual * weight
        support[rr, cc] += weight

    if (
        low_frequency_sigma_px is not None
        and low_frequency_sigma_px > 0
        and np.count_nonzero(support) >= 6
    ):
        numerator = gaussian_filter(impulses, sigma=low_frequency_sigma_px)
        denominator = gaussian_filter(support, sigma=low_frequency_sigma_px)
        bias = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-10,
        )
    else:
        bias = np.zeros(rel.shape, dtype=np.float64)

    dsm = global_dsm + bias
    dsm[~np.isfinite(rel)] = np.nan
    return GCPCalibrationOutput(
        dsm=dsm.astype(np.float32),
        calibration=fit,
        gcp_residuals_m=residuals.astype(np.float32),
        sampled_relative_height=samples_array.astype(np.float32),
        low_frequency_bias=bias.astype(np.float32),
        orientation_flipped=orientation_flipped,
        anchor_correlation_before=correlation_before,
        cross_validation_rmse_m=cross_validation_rmse,
        spatial_coverage_fraction=spatial_coverage,
        spatial_rank_ratio=rank_ratio,
    )


def validate_metric_dsm_with_gcps(
    metric_dsm: np.ndarray,
    *,
    transform: Affine,
    gcps: list[GroundControlPoint],
    min_gcps: int = RECOMMENDED_GCP_COUNT,
    max_rmse_m: float | None = 10.0,
    max_cross_validation_rmse_m: float | None = 15.0,
    max_abs_offset_correction_m: float = 50.0,
    min_axis_span_fraction: float = 0.20,
    min_spatial_coverage_fraction: float = 0.02,
    min_rank_ratio: float = 0.05,
) -> GCPMetricValidationOutput:
    """Validate an already metric DSM and apply at most one robust global datum offset.

    DEM calibration already establishes relief scale. Re-fitting both scale and offset to a small
    GCP set can distort every roof, tree, and slope in the scene, so DEM+GCP fusion deliberately
    limits GCP influence to a global vertical-datum offset and held-out validation.
    """
    dsm = np.asarray(metric_dsm, dtype=np.float64)
    if dsm.ndim != 2:
        raise ValueError("metric_dsm must be a 2D raster")
    samples, elevations, weights, rows, cols = _sample_gcps(
        dsm,
        transform=transform,
        gcps=gcps,
    )
    coverage, rank_ratio = _validate_spatial_distribution(
        rows,
        cols,
        shape=(dsm.shape[0], dsm.shape[1]),
        min_gcps=min_gcps,
        min_axis_span_fraction=min_axis_span_fraction,
        min_spatial_coverage_fraction=min_spatial_coverage_fraction,
        min_rank_ratio=min_rank_ratio,
    )
    residuals_before = elevations - samples
    offset = _weighted_median(residuals_before, weights)
    if abs(offset) > max_abs_offset_correction_m:
        raise ValueError(
            f"GCP datum offset {offset:.3f} m exceeds the configured "
            f"{max_abs_offset_correction_m:.3f} m safety limit"
        )
    residuals_after = elevations - (samples + offset)
    rmse_before = _weighted_rmse(residuals_before, weights)
    rmse_after = _weighted_rmse(residuals_after, weights)

    held_out_predictions = np.empty(samples.size, dtype=np.float64)
    for held_out in range(samples.size):
        keep = np.arange(samples.size) != held_out
        fold_offset = _weighted_median(residuals_before[keep], weights[keep])
        held_out_predictions[held_out] = samples[held_out] + fold_offset
    cross_validation_rmse = _weighted_rmse(elevations - held_out_predictions, weights)
    if max_rmse_m is not None and rmse_after > max_rmse_m:
        raise ValueError(
            f"DEM+GCP residual RMSE {rmse_after:.3f} m exceeds the configured "
            f"{max_rmse_m:.3f} m quality limit"
        )
    if (
        max_cross_validation_rmse_m is not None
        and cross_validation_rmse > max_cross_validation_rmse_m
    ):
        raise ValueError(
            f"DEM+GCP leave-one-out RMSE {cross_validation_rmse:.3f} m exceeds the configured "
            f"{max_cross_validation_rmse_m:.3f} m quality limit"
        )

    corrected = dsm + offset
    corrected[~np.isfinite(dsm)] = np.nan
    return GCPMetricValidationOutput(
        dsm=corrected.astype(np.float32),
        gcp_residuals_before_m=residuals_before.astype(np.float32),
        gcp_residuals_after_m=residuals_after.astype(np.float32),
        offset_applied_m=offset,
        rmse_before_m=rmse_before,
        rmse_after_m=rmse_after,
        cross_validation_rmse_m=cross_validation_rmse,
        spatial_coverage_fraction=coverage,
        spatial_rank_ratio=rank_ratio,
    )
