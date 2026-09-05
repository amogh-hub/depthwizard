from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from depthwizard.contracts import EvaluationMetrics, SlopeMetrics


def compute_elevation_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> EvaluationMetrics:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, reference={ref.shape}")

    mask = np.isfinite(pred) & np.isfinite(ref)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != pred.shape:
            raise ValueError("valid_mask must match prediction/reference shape")
        mask &= vm

    p = pred[mask]
    r = ref[mask]
    if p.size == 0:
        raise ValueError("no valid pixels available for evaluation")

    error = p - r
    abs_error = np.abs(error)
    mae = float(abs_error.mean())
    rmse = float(np.sqrt(np.mean(error**2)))
    bias = float(error.mean())
    medae = float(np.median(abs_error))
    p90 = float(np.percentile(abs_error, 90))
    p95 = float(np.percentile(abs_error, 95))

    pearson: float | None
    spearman: float | None
    if p.size < 2 or np.std(p) == 0 or np.std(r) == 0:
        pearson = None
        spearman = None
    else:
        pearson = float(np.corrcoef(p, r)[0, 1])
        ranked_prediction = rankdata(p)
        ranked_reference = rankdata(r)
        spearman = float(np.corrcoef(ranked_prediction, ranked_reference)[0, 1])

    median_error = float(np.median(error))
    nmad = float(1.4826 * np.median(np.abs(error - median_error)))

    return EvaluationMetrics(
        valid_pixels=int(p.size),
        mae_m=mae,
        rmse_m=rmse,
        pearson_r=pearson,
        spearman_r=spearman,
        mean_bias_m=bias,
        median_abs_error_m=medae,
        nmad_m=nmad,
        p90_abs_error_m=p90,
        p95_abs_error_m=p95,
    )


def slope_degrees(
    elevation: np.ndarray,
    *,
    gsd_x: float | None = None,
    gsd_y: float | None = None,
    ground_jacobian_m: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate surface slope using either orthogonal GSD or a full local ground Jacobian."""
    z = np.asarray(elevation, dtype=np.float64)
    dz_drow, dz_dcol = np.gradient(z)
    if ground_jacobian_m is not None:
        jacobian = np.asarray(ground_jacobian_m, dtype=np.float64)
        if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
            raise ValueError("ground_jacobian_m must be a finite 2x2 matrix")
        determinant = float(np.linalg.det(jacobian))
        if abs(determinant) <= 1e-12:
            raise ValueError("ground_jacobian_m must be invertible")
        pixel_to_ground_gradient = np.linalg.inv(jacobian.T)
        gradient_east = (
            pixel_to_ground_gradient[0, 0] * dz_dcol
            + pixel_to_ground_gradient[0, 1] * dz_drow
        )
        gradient_north = (
            pixel_to_ground_gradient[1, 0] * dz_dcol
            + pixel_to_ground_gradient[1, 1] * dz_drow
        )
    else:
        if gsd_x is None or gsd_y is None or gsd_x <= 0 or gsd_y <= 0:
            raise ValueError(
                "positive gsd_x/gsd_y or an invertible ground_jacobian_m is required"
            )
        gradient_east = dz_dcol / gsd_x
        gradient_north = dz_drow / gsd_y
    return np.degrees(np.arctan(np.hypot(gradient_east, gradient_north))).astype(np.float32)


def compute_slope_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    gsd_x: float,
    gsd_y: float,
    ground_jacobian_m: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> SlopeMetrics:
    pred_slope = slope_degrees(
        prediction,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        ground_jacobian_m=ground_jacobian_m,
    )
    ref_slope = slope_degrees(
        reference,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        ground_jacobian_m=ground_jacobian_m,
    )
    mask = np.isfinite(pred_slope) & np.isfinite(ref_slope)
    if valid_mask is not None:
        mask &= np.asarray(valid_mask, dtype=bool)
    error = pred_slope[mask] - ref_slope[mask]
    if error.size == 0:
        raise ValueError("no valid pixels available for slope evaluation")
    abs_error = np.abs(error)
    return SlopeMetrics(
        valid_pixels=int(error.size),
        mae_degrees=float(abs_error.mean()),
        rmse_degrees=float(np.sqrt(np.mean(error**2))),
        p95_abs_error_degrees=float(np.percentile(abs_error, 95)),
    )
