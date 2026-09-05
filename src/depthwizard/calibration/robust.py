from __future__ import annotations

import numpy as np

from depthwizard.contracts import CalibrationResult


def robust_affine_calibration(
    relative_values: np.ndarray,
    metric_anchors: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    huber_delta: float = 1.5,
    max_iter: int = 100,
    tol: float = 1e-6,
    require_positive_scale: bool = True,
    require_convergence: bool = True,
    max_condition_number: float = 1e6,
) -> CalibrationResult:
    """Robustly fit metric ~= scale * relative + offset with IRLS Huber weighting.

    A positive scale is physically required for the normal DepthWizard height convention.
    If the supplied relative prior is inverse-depth-like, it must be converted before calibration.
    """
    x = np.asarray(relative_values, dtype=np.float64).reshape(-1)
    y = np.asarray(metric_anchors, dtype=np.float64).reshape(-1)
    if x.shape != y.shape or x.size < 2:
        raise ValueError("relative_values and metric_anchors need matching length >= 2")
    if max_iter < 1:
        raise ValueError("max_iter must be >= 1")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1.0:
        raise ValueError("max_condition_number must be finite and greater than one")

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

    # Centre and scale the explanatory variable before solving. Remote-sensing calibration often
    # combines a dimensionless prior with a large elevation offset; solving directly in those
    # disparate numeric ranges can hide an ill-conditioned anchor set. Coefficients are converted
    # back to the caller's original convention before being returned.
    total_base_weight = float(np.sum(base_w))
    x_mean = float(np.sum(base_w * x) / total_base_weight)
    x_variance = float(np.sum(base_w * (x - x_mean) ** 2) / total_base_weight)
    x_scale = float(np.sqrt(max(x_variance, 0.0)))
    robust_x_span = float(np.percentile(x, 95.0) - np.percentile(x, 5.0))
    if x_scale <= 1e-12 or robust_x_span <= 1e-10:
        raise ValueError("calibration anchors do not span enough relative-height variation")

    normalized_x = (x - x_mean) / x_scale
    X = np.column_stack([normalized_x, np.ones_like(normalized_x)])
    w = base_w.copy()
    condition_number = float(np.linalg.cond(X * np.sqrt(w[:, None])))
    if not np.isfinite(condition_number) or condition_number > max_condition_number:
        raise ValueError(
            "calibration anchor design is ill-conditioned; improve anchor distribution or weights"
        )
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
        original_scale = float(new_beta[0] / x_scale)
        if require_positive_scale and original_scale <= 0:
            raise ValueError(
                "calibration produced non-positive scale; convert inverse depth to relative height "
                "or inspect anchor quality before metric calibration"
            )
        if np.linalg.norm(new_beta - beta) <= tol * (1.0 + np.linalg.norm(beta)):
            beta = new_beta
            converged = True
            break
        beta = new_beta

    if require_convergence and not converged:
        raise ValueError(
            f"robust affine calibration did not converge within {max_iter} iterations"
        )

    residual = y - X @ beta
    rmse = float(np.sqrt(np.average(residual**2, weights=base_w)))
    mae = float(np.average(np.abs(residual), weights=base_w))
    medae = float(np.median(np.abs(residual)))
    metric_span = float(np.percentile(y, 95.0) - np.percentile(y, 5.0))
    normalized_rmse = rmse / metric_span if metric_span > 1e-9 else None
    scale = float(beta[0] / x_scale)
    offset = float(beta[1] - scale * x_mean)
    return CalibrationResult(
        scale=scale,
        offset=offset,
        rmse_anchor=rmse,
        median_abs_residual=medae,
        mae_anchor=mae,
        normalized_rmse=normalized_rmse,
        condition_number=condition_number,
        anchors_used=int(x.size),
        iterations=iteration,
        converged=converged,
        quality_passed=converged,
        quality_notes=["fit_converged"] if converged else ["fit_did_not_converge"],
    )
