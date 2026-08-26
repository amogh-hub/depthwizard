from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rasterio.transform import Affine
from scipy.ndimage import gaussian_filter

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.contracts import CalibrationResult, GroundControlPoint


@dataclass(frozen=True)
class GCPCalibrationOutput:
    dsm: np.ndarray
    calibration: CalibrationResult
    gcp_residuals_m: np.ndarray
    sampled_relative_height: np.ndarray
    low_frequency_bias: np.ndarray
    orientation_flipped: bool
    anchor_correlation_before: float


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


def calibrate_relative_height_with_gcps(
    relative_height: np.ndarray,
    *,
    transform: Affine,
    gcps: list[GroundControlPoint],
    low_frequency_sigma_px: float = 32.0,
    resolve_orientation: bool = True,
) -> GCPCalibrationOutput:
    """Calibrate relative height with sparse metric Ground Control Points.

    GCP coordinates are interpreted in the same CRS as the source raster. At least two reliable
    points with distinguishable relative height are required to solve scale and offset. When
    ``resolve_orientation`` is enabled, a negative GCP/relative-height correlation is recorded and
    the relative-height polarity is inverted before the physically constrained positive-scale fit.
    This mirrors the DEM calibration contract and prevents domain-shift polarity from being hidden
    inside an invalid negative metric scale.
    """
    rel = np.asarray(relative_height, dtype=np.float64)
    if rel.ndim != 2:
        raise ValueError("relative_height must be a 2D raster")
    if len(gcps) < 2:
        raise ValueError("at least two GCPs are required for scale/offset calibration")

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
        value = _bilinear_sample(rel, row, col)
        if np.isfinite(value):
            samples.append(value)
            elevations.append(gcp.elevation_m)
            weights.append(gcp.weight)
            pixel_rows.append(row)
            pixel_cols.append(col)

    if len(samples) < 2:
        raise ValueError("fewer than two valid GCPs overlap finite relative-height pixels")
    samples_array = np.asarray(samples, dtype=np.float64)
    elevations_array = np.asarray(elevations, dtype=np.float64)
    if np.ptp(samples_array) <= 1e-8:
        raise ValueError("GCPs do not span enough relative-height variation to determine scale")

    correlation_before = _anchor_correlation(samples_array, elevations_array)
    orientation_flipped = bool(resolve_orientation and np.isfinite(correlation_before) and correlation_before < 0)
    if not resolve_orientation and np.isfinite(correlation_before) and correlation_before < 0:
        raise ValueError(
            "GCP evidence implies inverted height orientation while orientation resolution is disabled"
        )

    oriented_rel = -rel if orientation_flipped else rel
    oriented_samples = -samples_array if orientation_flipped else samples_array
    fit = robust_affine_calibration(
        oriented_samples,
        elevations_array,
        weights=np.asarray(weights),
    )
    global_dsm = fit.scale * oriented_rel + fit.offset
    residuals = elevations_array - (fit.scale * oriented_samples + fit.offset)

    # Sparse residual impulses are smoothed into a low-frequency correction field. Weight
    # normalization prevents regions far from any GCP from being forced toward zero residual.
    impulses = np.zeros(rel.shape, dtype=np.float64)
    support = np.zeros(rel.shape, dtype=np.float64)
    for row, col, residual, weight in zip(pixel_rows, pixel_cols, residuals, weights, strict=True):
        rr = int(np.clip(round(row), 0, rel.shape[0] - 1))
        cc = int(np.clip(round(col), 0, rel.shape[1] - 1))
        impulses[rr, cc] += residual * weight
        support[rr, cc] += weight

    if low_frequency_sigma_px > 0 and np.count_nonzero(support) >= 3:
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
    )
