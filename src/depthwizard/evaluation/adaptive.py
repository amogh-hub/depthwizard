from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.evaluation.holdout import (
    SparseAnchorHoldoutResult,
    select_sparse_anchor_mask,
    sparse_anchor_holdout_benchmark,
)


@dataclass(frozen=True)
class BlendCandidateScore:
    weight: float
    cv_rmse_m: float | None
    calibration_valid: bool
    rejection_reason: str | None


@dataclass(frozen=True)
class AdaptiveBlendSelection:
    selected_weight: float
    baseline_cv_rmse_m: float | None
    selected_cv_rmse_m: float | None
    relative_cv_improvement: float
    safety_margin_fraction: float
    near_best_fraction: float
    selection_evidence_valid: bool
    fallback_reason: str | None
    candidate_scores: tuple[BlendCandidateScore, ...]


@dataclass(frozen=True)
class AdaptiveSparseAnchorHoldoutResult:
    baseline: SparseAnchorHoldoutResult
    adaptive: SparseAnchorHoldoutResult
    selection: AdaptiveBlendSelection


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[finite], yv[finite]
    if xv.size < 2 or float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(xv, yv)[0, 1])


def _validate_candidate_weights(candidate_weights: tuple[float, ...]) -> tuple[float, ...]:
    if not candidate_weights:
        raise ValueError("candidate_weights must not be empty")
    weights = tuple(float(value) for value in candidate_weights)
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in weights):
        raise ValueError("candidate_weights must be finite values in [0, 1]")
    if len(set(weights)) != len(weights):
        raise ValueError("candidate_weights must be unique")
    if 0.0 not in weights:
        raise ValueError("candidate_weights must include the exact DA3 fallback weight 0.0")
    return tuple(sorted(weights))


def _cross_validated_anchor_rmse(
    relative_height: np.ndarray,
    reference_dsm: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    folds: int,
    seed: int,
    min_abs_anchor_correlation: float,
) -> float:
    rel = np.asarray(relative_height, dtype=np.float64)
    ref = np.asarray(reference_dsm, dtype=np.float64)
    anchors = np.flatnonzero(np.asarray(anchor_mask, dtype=bool))
    if rel.shape != ref.shape or rel.shape != anchor_mask.shape:
        raise ValueError("relative, reference and anchor_mask must share shape")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if anchors.size < 2 * folds:
        raise ValueError("anchor budget is too small for the requested cross-validation folds")

    rng = np.random.default_rng(seed)
    shuffled = anchors.copy()
    rng.shuffle(shuffled)
    fold_indices = [part for part in np.array_split(shuffled, folds) if part.size > 0]
    squared_errors: list[np.ndarray] = []

    for heldout in fold_indices:
        training = np.setdiff1d(shuffled, heldout, assume_unique=True)
        train_rel = rel.flat[training]
        train_ref = ref.flat[training]
        correlation = _correlation(train_rel, train_ref)
        if not np.isfinite(correlation):
            raise ValueError("anchor CV fold cannot determine relative-height orientation")
        if abs(correlation) < min_abs_anchor_correlation:
            raise ValueError(
                "anchor CV fold is too weakly correlated for defensible metric calibration "
                f"(|r|={abs(correlation):.3f})"
            )

        oriented_train = -train_rel if correlation < 0.0 else train_rel
        calibration = robust_affine_calibration(
            oriented_train,
            train_ref,
            require_positive_scale=True,
        )
        heldout_rel = rel.flat[heldout]
        oriented_heldout = -heldout_rel if correlation < 0.0 else heldout_rel
        prediction = calibration.scale * oriented_heldout + calibration.offset
        error = prediction - ref.flat[heldout]
        if np.any(~np.isfinite(error)):
            raise ValueError("anchor CV produced non-finite metric errors")
        squared_errors.append(np.square(error, dtype=np.float64))

    if not squared_errors:
        raise RuntimeError("anchor cross-validation produced no held-out predictions")
    all_squared = np.concatenate(squared_errors)
    return float(np.sqrt(np.mean(all_squared)))


def _fallback_selection(
    *,
    scores: list[BlendCandidateScore],
    safety_margin_fraction: float,
    near_best_fraction: float,
    reason: str,
    baseline_cv_rmse_m: float | None = None,
) -> AdaptiveBlendSelection:
    """Return exact DA3 when anchor evidence cannot defensibly rank learned refinement.

    The operational invariant is fail-closed: an inability to validate the refinement policy must
    never make the baseline unavailable. Metric calibration of DA3 itself is still performed later
    by ``sparse_anchor_holdout_benchmark`` and remains free to reject the scene if the *full* anchor
    set is genuinely insufficient. This helper only handles failure of the nested policy-selection
    cross-validation.
    """
    return AdaptiveBlendSelection(
        selected_weight=0.0,
        baseline_cv_rmse_m=baseline_cv_rmse_m,
        selected_cv_rmse_m=baseline_cv_rmse_m,
        relative_cv_improvement=0.0,
        safety_margin_fraction=float(safety_margin_fraction),
        near_best_fraction=float(near_best_fraction),
        selection_evidence_valid=False,
        fallback_reason=reason,
        candidate_scores=tuple(scores),
    )


def select_evidence_adaptive_blend(
    geometry_prior: np.ndarray,
    refined_relative_height: np.ndarray,
    reference_dsm: np.ndarray,
    anchor_mask: np.ndarray,
    *,
    candidate_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    folds: int = 4,
    cv_seed: int = 104729,
    min_abs_anchor_correlation: float = 0.05,
    safety_margin_fraction: float = 0.01,
    near_best_fraction: float = 0.005,
) -> AdaptiveBlendSelection:
    """Choose how much learned refinement to trust using calibration anchors only.

    The decision never sees raster evaluation pixels. The declared sparse anchors are internally
    cross-validated: each candidate blend is calibrated on a subset of anchors and scored on the
    held-out anchor fold. A non-zero refinement is accepted only when its cross-validated RMSE
    improves over exact DA3 by at least ``safety_margin_fraction``. Among candidates within
    ``near_best_fraction`` of the best valid score, the smallest correction weight is preferred.

    If nested cross-validation cannot produce a defensible DA3 reference score, the policy fails
    closed to exact DA3. That is different from final metric calibration: the complete declared
    anchor set is still used afterwards and may independently accept or reject DA3 itself.

    This is an operational calibration policy, not a replacement for untouched scientific
    validation. It is designed for the SIH setting where coarse DEM/GCP evidence is available and
    should actively prevent a learned residual from overriding a stronger scene-specific prior.
    """
    geometry = np.asarray(geometry_prior, dtype=np.float64)
    refined = np.asarray(refined_relative_height, dtype=np.float64)
    reference = np.asarray(reference_dsm, dtype=np.float64)
    anchors = np.asarray(anchor_mask, dtype=bool)
    if geometry.shape != refined.shape or geometry.shape != reference.shape or geometry.ndim != 2:
        raise ValueError("geometry, refined and reference must be matching 2D arrays")
    if anchors.shape != geometry.shape:
        raise ValueError("anchor_mask must match geometry shape")
    if not (0.0 <= min_abs_anchor_correlation <= 1.0):
        raise ValueError("min_abs_anchor_correlation must be in [0, 1]")
    if not (0.0 <= safety_margin_fraction < 1.0):
        raise ValueError("safety_margin_fraction must be in [0, 1)")
    if not (0.0 <= near_best_fraction < 1.0):
        raise ValueError("near_best_fraction must be in [0, 1)")

    weights = _validate_candidate_weights(candidate_weights)
    delta = refined - geometry
    scores: list[BlendCandidateScore] = []
    valid_scores: dict[float, float] = {}

    for weight in weights:
        candidate = geometry + weight * delta
        try:
            score = _cross_validated_anchor_rmse(
                candidate,
                reference,
                anchors,
                folds=folds,
                seed=cv_seed,
                min_abs_anchor_correlation=min_abs_anchor_correlation,
            )
        except ValueError as exc:
            scores.append(
                BlendCandidateScore(
                    weight=weight,
                    cv_rmse_m=None,
                    calibration_valid=False,
                    rejection_reason=str(exc),
                )
            )
            continue
        valid_scores[weight] = score
        scores.append(
            BlendCandidateScore(
                weight=weight,
                cv_rmse_m=score,
                calibration_valid=True,
                rejection_reason=None,
            )
        )

    if 0.0 not in valid_scores:
        baseline_rejection = next(
            (
                score.rejection_reason
                for score in scores
                if score.weight == 0.0 and score.rejection_reason is not None
            ),
            "exact DA3 anchor cross-validation was unavailable",
        )
        return _fallback_selection(
            scores=scores,
            safety_margin_fraction=safety_margin_fraction,
            near_best_fraction=near_best_fraction,
            reason=(
                "adaptive policy evidence was insufficient; preserved exact DA3 fallback: "
                f"{baseline_rejection}"
            ),
        )

    baseline_score = valid_scores[0.0]
    if not np.isfinite(baseline_score) or baseline_score < 0.0:
        return _fallback_selection(
            scores=scores,
            safety_margin_fraction=safety_margin_fraction,
            near_best_fraction=near_best_fraction,
            reason="adaptive policy produced a non-finite/negative DA3 CV score",
        )
    if baseline_score == 0.0:
        return AdaptiveBlendSelection(
            selected_weight=0.0,
            baseline_cv_rmse_m=0.0,
            selected_cv_rmse_m=0.0,
            relative_cv_improvement=0.0,
            safety_margin_fraction=float(safety_margin_fraction),
            near_best_fraction=float(near_best_fraction),
            selection_evidence_valid=True,
            fallback_reason="exact DA3 anchor-CV RMSE is zero; refinement cannot improve it",
            candidate_scores=tuple(scores),
        )

    best_weight, best_score = min(valid_scores.items(), key=lambda item: (item[1], item[0]))
    improvement = (baseline_score - best_score) / baseline_score
    if best_weight == 0.0 or improvement < safety_margin_fraction:
        selected_weight = 0.0
        selected_score = baseline_score
        selected_improvement = 0.0
    else:
        near_best_limit = best_score * (1.0 + near_best_fraction)
        conservative = [
            (weight, score)
            for weight, score in valid_scores.items()
            if weight > 0.0 and score <= near_best_limit
        ]
        selected_weight, selected_score = min(conservative, key=lambda item: item[0])
        selected_improvement = (baseline_score - selected_score) / baseline_score

    return AdaptiveBlendSelection(
        selected_weight=float(selected_weight),
        baseline_cv_rmse_m=float(baseline_score),
        selected_cv_rmse_m=float(selected_score),
        relative_cv_improvement=float(selected_improvement),
        safety_margin_fraction=float(safety_margin_fraction),
        near_best_fraction=float(near_best_fraction),
        selection_evidence_valid=True,
        fallback_reason=None,
        candidate_scores=tuple(scores),
    )


def adaptive_sparse_anchor_holdout_benchmark(
    geometry_prior: np.ndarray,
    refined_relative_height: np.ndarray,
    reference_dsm: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    anchor_count: int = 64,
    seed: int = 26175,
    exclusion_radius_px: int = 4,
    candidate_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    cv_folds: int = 4,
    cv_seed: int = 104729,
    min_abs_anchor_correlation: float = 0.05,
    safety_margin_fraction: float = 0.01,
    near_best_fraction: float = 0.005,
) -> AdaptiveSparseAnchorHoldoutResult:
    """Evaluate evidence-adaptive refinement with anchor-only blend selection.

    The same deterministic anchor mask is used for DA3 calibration, adaptive blend selection and
    final metric calibration. Blend selection itself uses only cross-validated anchor residuals;
    raster evaluation pixels remain excluded throughout policy selection. If nested CV is too weak
    to rank candidates, exact DA3 is preserved and full-anchor metric calibration proceeds normally.
    """
    geometry = np.asarray(geometry_prior, dtype=np.float64)
    refined = np.asarray(refined_relative_height, dtype=np.float64)
    reference = np.asarray(reference_dsm, dtype=np.float64)
    if geometry.shape != refined.shape or geometry.shape != reference.shape or geometry.ndim != 2:
        raise ValueError("geometry, refined and reference must be matching 2D arrays")

    valid = np.isfinite(geometry) & np.isfinite(refined) & np.isfinite(reference)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != geometry.shape:
            raise ValueError("valid_mask must match geometry")
        valid &= supplied

    anchor_mask = select_sparse_anchor_mask(valid, anchor_count=anchor_count, seed=seed)
    selection = select_evidence_adaptive_blend(
        geometry,
        refined,
        reference,
        anchor_mask,
        candidate_weights=candidate_weights,
        folds=cv_folds,
        cv_seed=cv_seed,
        min_abs_anchor_correlation=min_abs_anchor_correlation,
        safety_margin_fraction=safety_margin_fraction,
        near_best_fraction=near_best_fraction,
    )
    blended = geometry + selection.selected_weight * (refined - geometry)

    baseline = sparse_anchor_holdout_benchmark(
        geometry,
        reference,
        valid_mask=valid,
        anchor_count=anchor_count,
        seed=seed,
        exclusion_radius_px=exclusion_radius_px,
        min_abs_anchor_correlation=min_abs_anchor_correlation,
    )
    adaptive = sparse_anchor_holdout_benchmark(
        blended,
        reference,
        valid_mask=valid,
        anchor_count=anchor_count,
        seed=seed,
        exclusion_radius_px=exclusion_radius_px,
        min_abs_anchor_correlation=min_abs_anchor_correlation,
    )
    if not np.array_equal(anchor_mask, baseline.anchor_mask) or not np.array_equal(
        anchor_mask, adaptive.anchor_mask
    ):
        raise RuntimeError("adaptive benchmark anchor masks diverged from the declared protocol")
    if not np.array_equal(baseline.evaluation_mask, adaptive.evaluation_mask):
        raise RuntimeError("adaptive benchmark evaluation masks diverged between candidates")

    return AdaptiveSparseAnchorHoldoutResult(
        baseline=baseline,
        adaptive=adaptive,
        selection=selection,
    )
