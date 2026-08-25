from __future__ import annotations

import numpy as np

from depthwizard.contracts import CalibrationResult


def robust_affine_calibration(
    relative_values: np.ndarray,
    metric_anchors: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    huber_delta: float = 1.5,
    max_iter: int = 50,
    tol: float = 1e-8,
    require_positive_scale: bool = True,
) -> CalibrationResult:
    """Robustly fit metric ~= scale * relative + offset with IRLS Huber weighting.

    A positive scale is physically required for the normal DepthWizard height convention.
    If the supplied relative prior is inverse-depth-like, it must be converted before calibration.
    """
    x = np.asarray(relative_values, dtype=np.float64).reshape(-1)
    y = np.asarray(metric_anchors, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("relative_values and metric_anchors need matching length >= 2")

    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 2:
        raise ValueError("fewer than two finite calibration anchors")

    if weights is None:
        base_w = np.ones_like(x)
    else:
        all_w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if all_w.shape[0] != finite.shape[0]:
            raise ValueError("weights must have the same flattened length as input values")
        base_w = all_w[finite]
    if np.any(~np.isfinite(base_w)) or np.any(base_w <= 0):
        raise ValueError("weights must be finite and positive")

    X = np.column_stack([x, np.ones_like(x)])
    w = base_w.copy()
    beta = np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
    converged = False
    iteration = 0

    for iteration in range(1, max_iter + 1):
        residual = y - X @ beta
        mad = np.median(np.abs(residual - np.median(residual)))
        sigma = max(1.4826 * mad, 1e-9)
        normalized = np.abs(residual) / sigma
        huber_w = np.ones_like(normalized)
        outlier = normalized > huber_delta
        huber_w[outlier] = huber_delta / normalized[outlier]
        w = base_w * huber_w

        new_beta = np.linalg.lstsq(X * np.sqrt(w[:, None]), y * np.sqrt(w), rcond=None)[0]
        if require_positive_scale and new_beta[0] <= 0:
            raise ValueError(
                "calibration produced non-positive scale; convert inverse depth to relative height "
                "or inspect anchor quality before metric calibration"
            )
        if np.linalg.norm(new_beta - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = new_beta
            converged = True
            break
        beta = new_beta

    residual = y - X @ beta
    rmse = float(np.sqrt(np.average(residual**2, weights=base_w)))
    medae = float(np.median(np.abs(residual)))
    return CalibrationResult(
        scale=float(beta[0]),
        offset=float(beta[1]),
        rmse_anchor=rmse,
        median_abs_residual=medae,
        anchors_used=int(x.size),
        iterations=iteration,
        converged=converged,
    )
