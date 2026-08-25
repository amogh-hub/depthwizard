from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.contracts import CalibrationResult, EvaluationMetrics
from depthwizard.evaluation.metrics import compute_elevation_metrics


@dataclass(frozen=True)
class SparseAnchorHoldoutResult:
    prediction: np.ndarray
    calibration: CalibrationResult
    metrics: EvaluationMetrics
    anchor_mask: np.ndarray
    evaluation_mask: np.ndarray
    orientation_flipped: bool
    anchor_correlation_before: float
    anchor_correlation_after: float


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    valid = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[valid], yv[valid]
    if xv.size < 2 or float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def sparse_anchor_holdout_benchmark(
    relative_height: np.ndarray,
    reference_dsm: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    anchor_count: int = 64,
    seed: int = 26175,
    exclusion_radius_px: int = 2,
    min_abs_anchor_correlation: float = 0.05,
) -> SparseAnchorHoldoutResult:
    """Evaluate metric DSM on pixels disjoint from sparse calibration anchors.

    This protocol models the SIH allowance for limited GCP/height anchors without evaluating on
    the same pixels used to recover global metric scale. Only a global robust scale/offset fit is
    learned from the anchors; no spatial residual field is estimated from reference data.

    The resulting RMSE/MAE/correlation are therefore held-out metrics conditional on the declared
    anchor budget. They are not zero-shot metric-depth scores and must be reported together with
    ``anchor_count`` and the split seed.
    """
    rel = np.asarray(relative_height, dtype=np.float64)
    ref = np.asarray(reference_dsm, dtype=np.float64)
    if rel.shape != ref.shape or rel.ndim != 2:
        raise ValueError("relative_height and reference_dsm must be matching 2D arrays")
    if anchor_count < 2:
        raise ValueError("anchor_count must be at least 2")
    if exclusion_radius_px < 0:
        raise ValueError("exclusion_radius_px must be non-negative")
    if not (0.0 <= min_abs_anchor_correlation <= 1.0):
        raise ValueError("min_abs_anchor_correlation must be in [0, 1]")

    valid = np.isfinite(rel) & np.isfinite(ref)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != rel.shape:
            raise ValueError("valid_mask must match relative_height")
        valid &= supplied

    valid_flat = np.flatnonzero(valid)
    if valid_flat.size <= anchor_count:
        raise ValueError("not enough valid pixels for disjoint calibration and evaluation")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_flat, size=anchor_count, replace=False)
    anchor_mask = np.zeros(rel.shape, dtype=bool)
    anchor_mask.flat[chosen] = True

    correlation_before = _correlation(rel[anchor_mask], ref[anchor_mask])
    if not np.isfinite(correlation_before):
        raise ValueError("sparse anchors cannot determine relative-height orientation")
    if abs(correlation_before) < min_abs_anchor_correlation:
        raise ValueError(
            "sparse anchors are too weakly correlated with reference DSM for defensible metric "
            f"calibration (|r|={abs(correlation_before):.3f})"
        )

    orientation_flipped = correlation_before < 0.0
    oriented = -rel if orientation_flipped else rel
    correlation_after = -correlation_before if orientation_flipped else correlation_before

    calibration = robust_affine_calibration(
        oriented[anchor_mask],
        ref[anchor_mask],
        require_positive_scale=True,
    )
    prediction = calibration.scale * oriented + calibration.offset
    prediction[~np.isfinite(rel)] = np.nan

    if exclusion_radius_px > 0:
        size = 2 * exclusion_radius_px + 1
        excluded = binary_dilation(anchor_mask, structure=np.ones((size, size), dtype=bool))
    else:
        excluded = anchor_mask
    evaluation_mask = valid & ~excluded
    if int(evaluation_mask.sum()) == 0:
        raise ValueError("anchor exclusion removed every evaluation pixel")

    metrics = compute_elevation_metrics(prediction, ref, valid_mask=evaluation_mask)
    return SparseAnchorHoldoutResult(
        prediction=prediction.astype(np.float32),
        calibration=calibration,
        metrics=metrics,
        anchor_mask=anchor_mask,
        evaluation_mask=evaluation_mask,
        orientation_flipped=orientation_flipped,
        anchor_correlation_before=float(correlation_before),
        anchor_correlation_after=float(correlation_after),
    )
