from __future__ import annotations

import numpy as np

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
    if p.size < 2 or np.std(p) == 0 or np.std(r) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(p, r)[0, 1])

    return EvaluationMetrics(
        valid_pixels=int(p.size),
        mae_m=mae,
        rmse_m=rmse,
        pearson_r=pearson,
        mean_bias_m=bias,
        median_abs_error_m=medae,
        p90_abs_error_m=p90,
        p95_abs_error_m=p95,
    )


def slope_degrees(elevation: np.ndarray, *, gsd_x: float, gsd_y: float) -> np.ndarray:
    if gsd_x <= 0 or gsd_y <= 0:
        raise ValueError("gsd_x and gsd_y must be positive")
    z = np.asarray(elevation, dtype=np.float64)
    dy, dx = np.gradient(z, gsd_y, gsd_x)
    return np.degrees(np.arctan(np.hypot(dx, dy))).astype(np.float32)


def compute_slope_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    gsd_x: float,
    gsd_y: float,
    valid_mask: np.ndarray | None = None,
) -> SlopeMetrics:
    pred_slope = slope_degrees(prediction, gsd_x=gsd_x, gsd_y=gsd_y)
    ref_slope = slope_degrees(reference, gsd_x=gsd_x, gsd_y=gsd_y)
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
