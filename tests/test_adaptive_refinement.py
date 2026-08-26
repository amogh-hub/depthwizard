from __future__ import annotations

import numpy as np
import pytest

from depthwizard.evaluation import adaptive as adaptive_module
from depthwizard.evaluation.adaptive import (
    adaptive_sparse_anchor_holdout_benchmark,
    select_evidence_adaptive_blend,
)
from depthwizard.evaluation.holdout import select_sparse_anchor_mask


def _synthetic_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows, cols = np.mgrid[0:64, 0:80]
    geometry = 0.004 * rows + 0.006 * cols + 0.05 * np.sin(cols / 7.0)
    local_structure = 0.16 * np.exp(-((rows - 30.0) ** 2 + (cols - 42.0) ** 2) / 85.0)
    reference = 120.0 * (geometry + local_structure) + 430.0
    refined = geometry + 0.90 * local_structure
    valid = np.ones_like(geometry, dtype=bool)
    return (
        geometry.astype(np.float32),
        refined.astype(np.float32),
        reference.astype(np.float32),
        valid,
    )


def test_adaptive_policy_selects_refinement_when_anchor_cv_has_clear_gain() -> None:
    geometry, refined, reference, valid = _synthetic_scene()
    anchor_mask = select_sparse_anchor_mask(valid, anchor_count=64, seed=26175)

    selection = select_evidence_adaptive_blend(
        geometry,
        refined,
        reference,
        anchor_mask,
        safety_margin_fraction=0.01,
    )

    assert selection.selected_weight > 0.0
    assert selection.selection_evidence_valid is True
    assert selection.fallback_reason is None
    assert selection.baseline_cv_rmse_m is not None
    assert selection.selected_cv_rmse_m is not None
    assert selection.selected_cv_rmse_m < selection.baseline_cv_rmse_m
    assert selection.relative_cv_improvement >= 0.01


def test_adaptive_policy_falls_back_to_exact_da3_when_candidate_is_worse() -> None:
    geometry, _, reference, valid = _synthetic_scene()
    harmful_refined = geometry - 0.20 * np.sin(np.linspace(0.0, 8.0, geometry.shape[1]))[None, :]
    harmful_refined = harmful_refined.astype(np.float32)
    anchor_mask = select_sparse_anchor_mask(valid, anchor_count=64, seed=26175)

    selection = select_evidence_adaptive_blend(
        geometry,
        harmful_refined,
        reference,
        anchor_mask,
        safety_margin_fraction=0.01,
    )

    assert selection.selected_weight == 0.0
    assert selection.selection_evidence_valid is True
    assert selection.baseline_cv_rmse_m is not None
    assert selection.selected_cv_rmse_m == selection.baseline_cv_rmse_m


def test_adaptive_policy_preserves_exact_da3_when_nested_cv_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry, refined, reference, valid = _synthetic_scene()

    def reject_nested_cv(*args: object, **kwargs: object) -> float:
        del args, kwargs
        raise ValueError("synthetic weak anchor CV fold")

    monkeypatch.setattr(adaptive_module, "_cross_validated_anchor_rmse", reject_nested_cv)

    result = adaptive_sparse_anchor_holdout_benchmark(
        geometry,
        refined,
        reference,
        valid_mask=valid,
        anchor_count=64,
        seed=26175,
        exclusion_radius_px=4,
    )

    assert result.selection.selected_weight == 0.0
    assert result.selection.selection_evidence_valid is False
    assert result.selection.baseline_cv_rmse_m is None
    assert result.selection.selected_cv_rmse_m is None
    assert result.selection.relative_cv_improvement == 0.0
    assert result.selection.fallback_reason is not None
    assert "preserved exact DA3 fallback" in result.selection.fallback_reason
    assert np.array_equal(result.baseline.anchor_mask, result.adaptive.anchor_mask)
    assert np.array_equal(result.baseline.evaluation_mask, result.adaptive.evaluation_mask)
    assert np.allclose(
        result.baseline.prediction[result.baseline.evaluation_mask],
        result.adaptive.prediction[result.adaptive.evaluation_mask],
    )


def test_adaptive_holdout_never_uses_raster_evaluation_pixels_for_policy_selection() -> None:
    geometry, refined, reference, valid = _synthetic_scene()
    anchor_mask = select_sparse_anchor_mask(valid, anchor_count=64, seed=26175)
    changed_reference = reference.copy()
    changed_reference[~anchor_mask] += 5000.0

    original = select_evidence_adaptive_blend(
        geometry,
        refined,
        reference,
        anchor_mask,
    )
    changed = select_evidence_adaptive_blend(
        geometry,
        refined,
        changed_reference,
        anchor_mask,
    )

    assert changed.selected_weight == original.selected_weight
    assert changed.baseline_cv_rmse_m == original.baseline_cv_rmse_m
    assert changed.selected_cv_rmse_m == original.selected_cv_rmse_m
    assert changed.selection_evidence_valid == original.selection_evidence_valid


def test_adaptive_benchmark_uses_identical_anchors_and_evaluation_mask_for_both_candidates() -> None:
    geometry, refined, reference, valid = _synthetic_scene()

    result = adaptive_sparse_anchor_holdout_benchmark(
        geometry,
        refined,
        reference,
        valid_mask=valid,
        anchor_count=64,
        seed=26175,
        exclusion_radius_px=4,
    )

    assert np.array_equal(result.baseline.anchor_mask, result.adaptive.anchor_mask)
    assert np.array_equal(result.baseline.evaluation_mask, result.adaptive.evaluation_mask)
    assert result.adaptive.metrics.rmse_m <= result.baseline.metrics.rmse_m


def test_adaptive_policy_is_deterministic() -> None:
    geometry, refined, reference, valid = _synthetic_scene()
    anchor_mask = select_sparse_anchor_mask(valid, anchor_count=64, seed=26175)

    first = select_evidence_adaptive_blend(
        geometry,
        refined,
        reference,
        anchor_mask,
    )
    second = select_evidence_adaptive_blend(
        geometry,
        refined,
        reference,
        anchor_mask,
    )

    assert first == second
