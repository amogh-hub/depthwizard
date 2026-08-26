from __future__ import annotations

import numpy as np
import torch

from depthwizard.height_model.model import HeightModelOutput
from depthwizard.height_model.multiscene import (
    MultisceneLossWeights,
    balanced_group_order,
    compute_multiscene_refinement_loss,
)


def _output(relative_height: torch.Tensor, correction: torch.Tensor) -> HeightModelOutput:
    batch, _, height, width = relative_height.shape
    return HeightModelOutput(
        relative_height=relative_height,
        relative_correction=correction,
        log_variance=torch.zeros_like(relative_height),
        uncertainty=torch.ones_like(relative_height),
        semantic_logits=torch.zeros(batch, 5, height, width, dtype=relative_height.dtype),
        height_bin_logits=torch.zeros(batch, 8, height, width, dtype=relative_height.dtype),
        normals=torch.nn.functional.normalize(
            torch.ones(batch, 3, height, width, dtype=relative_height.dtype), dim=1
        ),
        boundary_probability=torch.full_like(relative_height, 0.5),
    )


def test_metric_loss_equalizes_equivalent_errors_across_prior_scales() -> None:
    valid = torch.ones(1, 1, 8, 8, dtype=torch.bool)
    target = torch.zeros(1, 1, 8, 8)
    geometry = target.clone()
    weights = MultisceneLossWeights(
        metric_regression=1.0,
        metric_gradient=0.0,
        non_degradation=0.0,
        correction_regularization=0.0,
        auxiliary=0.0,
    )

    prediction_a = torch.full_like(target, 0.5)
    loss_a = compute_multiscene_refinement_loss(
        _output(prediction_a, prediction_a),
        geometry,
        target,
        valid,
        torch.tensor([2.0]),
        weights=weights,
    )

    prediction_b = torch.full_like(target, 0.1)
    loss_b = compute_multiscene_refinement_loss(
        _output(prediction_b, prediction_b),
        geometry,
        target,
        valid,
        torch.tensor([10.0]),
        weights=weights,
    )

    assert torch.allclose(loss_a.metric_regression, loss_b.metric_regression, atol=1e-7)


def test_non_degradation_penalty_activates_only_when_refinement_is_worse() -> None:
    valid = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    target = torch.zeros(1, 1, 4, 4)
    geometry = torch.full_like(target, 0.2)
    weights = MultisceneLossWeights(
        metric_regression=0.0,
        metric_gradient=0.0,
        non_degradation=1.0,
        correction_regularization=0.0,
        auxiliary=0.0,
    )

    improved = torch.full_like(target, 0.1)
    improved_loss = compute_multiscene_refinement_loss(
        _output(improved, improved - geometry),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        weights=weights,
    )
    worsened = torch.full_like(target, 0.3)
    worsened_loss = compute_multiscene_refinement_loss(
        _output(worsened, worsened - geometry),
        geometry,
        target,
        valid,
        torch.tensor([5.0]),
        weights=weights,
    )

    assert improved_loss.non_degradation.item() == 0.0
    assert worsened_loss.non_degradation.item() > 0.0


def test_balanced_group_order_equalizes_geographic_contribution() -> None:
    groups = [0, 0, 0, 1, 2, 2]
    order = balanced_group_order(groups, np.random.default_rng(26175))
    sampled_groups = np.asarray(groups, dtype=np.int64)[order]

    counts = [int(np.sum(sampled_groups == group)) for group in (0, 1, 2)]
    assert counts == [3, 3, 3]
    assert len(order) == 9
