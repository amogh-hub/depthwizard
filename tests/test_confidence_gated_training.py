from __future__ import annotations

import torch

from depthwizard.height_model.confidence import (
    ConfidenceGatedLossWeights,
    compute_confidence_gated_refinement_loss,
)
from depthwizard.height_model.model import HeightModelOutput


def _output(
    geometry: torch.Tensor,
    raw_correction: torch.Tensor,
    gate: torch.Tensor,
) -> HeightModelOutput:
    applied = raw_correction * gate
    relative_height = geometry + applied
    batch, _, height, width = geometry.shape
    return HeightModelOutput(
        relative_height=relative_height,
        relative_correction=applied,
        log_variance=torch.zeros_like(geometry),
        uncertainty=torch.ones_like(geometry),
        semantic_logits=torch.zeros(batch, 5, height, width, dtype=geometry.dtype),
        height_bin_logits=torch.zeros(batch, 8, height, width, dtype=geometry.dtype),
        normals=torch.nn.functional.normalize(
            torch.ones(batch, 3, height, width, dtype=geometry.dtype), dim=1
        ),
        boundary_probability=torch.full_like(geometry, 0.5),
        raw_relative_correction=raw_correction,
        correction_gate=gate,
    )


def _gate_only_weights() -> ConfidenceGatedLossWeights:
    return ConfidenceGatedLossWeights(
        normalized_regression=0.0,
        candidate_regression=0.0,
        normalized_gradient=0.0,
        non_degradation=0.0,
        residual_correlation=0.0,
        correction_budget=0.0,
        gate_calibration=1.0,
        scene_bias=0.0,
        uncertainty_nll=0.0,
        boundary=0.0,
    )


def test_gate_supervision_prefers_applying_a_beneficial_candidate() -> None:
    geometry = torch.full((1, 1, 8, 8), 0.4)
    target = torch.zeros_like(geometry)
    valid = torch.ones_like(geometry, dtype=torch.bool)
    raw = torch.full_like(geometry, -0.3)

    high_gate = compute_confidence_gated_refinement_loss(
        _output(geometry, raw, torch.full_like(geometry, 0.95)),
        geometry,
        target,
        valid,
        torch.tensor([10.0]),
        torch.tensor([4.0]),
        weights=_gate_only_weights(),
    )
    low_gate = compute_confidence_gated_refinement_loss(
        _output(geometry, raw, torch.full_like(geometry, 0.05)),
        geometry,
        target,
        valid,
        torch.tensor([10.0]),
        torch.tensor([4.0]),
        weights=_gate_only_weights(),
    )

    assert high_gate.oracle_gate_mean > 0.9
    assert high_gate.gate_calibration < low_gate.gate_calibration


def test_gate_supervision_suppresses_a_harmful_candidate() -> None:
    geometry = torch.full((1, 1, 8, 8), 0.2)
    target = torch.zeros_like(geometry)
    valid = torch.ones_like(geometry, dtype=torch.bool)
    raw = torch.full_like(geometry, 0.2)

    low_gate = compute_confidence_gated_refinement_loss(
        _output(geometry, raw, torch.full_like(geometry, 0.05)),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        torch.tensor([1.0]),
        weights=_gate_only_weights(),
    )
    high_gate = compute_confidence_gated_refinement_loss(
        _output(geometry, raw, torch.full_like(geometry, 0.95)),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        torch.tensor([1.0]),
        weights=_gate_only_weights(),
    )

    assert low_gate.oracle_gate_mean.item() == 0.0
    assert low_gate.gate_calibration < high_gate.gate_calibration


def test_scene_error_normalization_equalizes_proportional_metric_errors() -> None:
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    target = torch.zeros(1, 1, 8, 8)
    weights = ConfidenceGatedLossWeights(
        normalized_regression=1.0,
        candidate_regression=0.0,
        normalized_gradient=0.0,
        non_degradation=0.0,
        residual_correlation=0.0,
        correction_budget=0.0,
        gate_calibration=0.0,
        scene_bias=0.0,
        uncertainty_nll=0.0,
        boundary=0.0,
    )

    geometry_a = torch.full_like(target, 0.2)
    loss_a = compute_confidence_gated_refinement_loss(
        _output(geometry_a, torch.zeros_like(target), torch.full_like(target, 0.1)),
        geometry_a,
        target,
        valid,
        torch.tensor([10.0]),
        torch.tensor([2.0]),
        weights=weights,
    )

    geometry_b = torch.full_like(target, 0.1)
    loss_b = compute_confidence_gated_refinement_loss(
        _output(geometry_b, torch.zeros_like(target), torch.full_like(target, 0.1)),
        geometry_b,
        target,
        valid,
        torch.tensor([40.0]),
        torch.tensor([4.0]),
        weights=weights,
    )

    assert torch.allclose(
        loss_a.normalized_regression,
        loss_b.normalized_regression,
        atol=1e-7,
    )


def test_non_degradation_penalty_detects_harmful_applied_correction() -> None:
    geometry = torch.full((1, 1, 8, 8), 0.2)
    target = torch.zeros_like(geometry)
    valid = torch.ones_like(geometry, dtype=torch.bool)
    weights = ConfidenceGatedLossWeights(
        normalized_regression=0.0,
        candidate_regression=0.0,
        normalized_gradient=0.0,
        non_degradation=1.0,
        residual_correlation=0.0,
        correction_budget=0.0,
        gate_calibration=0.0,
        scene_bias=0.0,
        uncertainty_nll=0.0,
        boundary=0.0,
    )

    helpful = compute_confidence_gated_refinement_loss(
        _output(geometry, torch.full_like(geometry, -0.1), torch.ones_like(geometry)),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        torch.tensor([1.0]),
        weights=weights,
    )
    harmful = compute_confidence_gated_refinement_loss(
        _output(geometry, torch.full_like(geometry, 0.1), torch.ones_like(geometry)),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        torch.tensor([1.0]),
        weights=weights,
    )

    assert helpful.non_degradation.item() == 0.0
    assert harmful.non_degradation.item() > 0.0


def test_confidence_gated_loss_is_finite_and_backpropagates_to_raw_and_gate() -> None:
    torch.manual_seed(26175)
    geometry = torch.rand(1, 1, 12, 12)
    target = geometry + 0.05 * torch.sin(
        torch.linspace(0.0, 5.0, 12)
    ).view(1, 1, 1, 12)
    valid = torch.ones_like(geometry, dtype=torch.bool)
    raw = torch.zeros_like(geometry, requires_grad=True)
    gate_logits = torch.full_like(geometry, -2.0, requires_grad=True)
    gate = torch.sigmoid(gate_logits)

    loss = compute_confidence_gated_refinement_loss(
        _output(geometry, raw, gate),
        geometry,
        target,
        valid,
        torch.tensor([20.0]),
        torch.tensor([2.0]),
    )
    assert torch.isfinite(loss.total)
    loss.total.backward()

    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert gate_logits.grad is not None and torch.isfinite(gate_logits.grad).all()
    assert torch.count_nonzero(raw.grad) > 0
    assert torch.count_nonzero(gate_logits.grad) > 0
