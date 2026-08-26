from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from depthwizard.height_model.losses import HeightLossWeights, compute_height_losses
from depthwizard.height_model.model import HeightModelOutput


@dataclass(frozen=True)
class MultisceneLossWeights:
    """Weights for geographically diverse relative-height refinement.

    Primary supervision is expressed in metres through each scene's training-only positive
    DA3-to-reference scale. This prevents a scene with a numerically small DA3 scale from
    dominating optimization simply because its canonical target spans many prior units.
    """

    metric_regression: float = 1.0
    metric_gradient: float = 0.25
    non_degradation: float = 0.20
    correction_regularization: float = 0.02
    auxiliary: float = 1.0


@dataclass(frozen=True)
class MultisceneLossResult:
    total: torch.Tensor
    metric_regression: torch.Tensor
    metric_gradient: torch.Tensor
    non_degradation: torch.Tensor
    correction_regularization: torch.Tensor
    auxiliary: torch.Tensor


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = values[valid]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def _metric_gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    scale_m_per_prior_unit: torch.Tensor,
) -> torch.Tensor:
    scale = scale_m_per_prior_unit.reshape(-1, 1, 1, 1)
    prediction_m = prediction * scale
    target_m = target * scale

    pred_dx = prediction_m[..., :, 1:] - prediction_m[..., :, :-1]
    target_dx = target_m[..., :, 1:] - target_m[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]

    pred_dy = prediction_m[..., 1:, :] - prediction_m[..., :-1, :]
    target_dy = target_m[..., 1:, :] - target_m[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]

    loss_x = _masked_mean(torch.abs(pred_dx - target_dx), valid_dx)
    loss_y = _masked_mean(torch.abs(pred_dy - target_dy), valid_dy)
    return 0.5 * (loss_x + loss_y)


def compute_multiscene_refinement_loss(
    output: HeightModelOutput,
    geometry_prior: torch.Tensor,
    target_prior: torch.Tensor,
    valid_mask: torch.Tensor,
    scale_m_per_prior_unit: torch.Tensor,
    *,
    weights: MultisceneLossWeights | None = None,
) -> MultisceneLossResult:
    """Train a conservative refiner in metric-equivalent units across scenes.

    ``target_prior`` is the reference DSM expressed in the scene's DA3 coordinate system and
    ``scale_m_per_prior_unit`` is the training-only positive affine scale used for that
    canonicalization. Multiplying residuals by the scale is algebraically equivalent to measuring
    error in metres (the affine offset cancels), while the model itself remains a relative-height
    predictor at inference time.

    A non-degradation hinge explicitly penalizes corrections that make a supervised pixel worse
    than the frozen DA3 prior. Together with a small correction penalty this keeps the refinement
    conservative when evidence is weak instead of rewarding large scene-specific changes.
    """
    w = weights or MultisceneLossWeights()
    valid = valid_mask.to(dtype=torch.bool)
    if output.relative_height.shape != target_prior.shape:
        raise ValueError("output and target_prior must have identical shapes")
    if geometry_prior.shape != target_prior.shape or valid.shape != target_prior.shape:
        raise ValueError("geometry_prior, target_prior and valid_mask must have identical shapes")
    if target_prior.ndim != 4 or target_prior.shape[1] != 1:
        raise ValueError("multiscene targets must have shape N x 1 x H x W")

    scale = scale_m_per_prior_unit.reshape(-1)
    if scale.numel() != target_prior.shape[0]:
        raise ValueError("one metric scale is required per batch element")
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("metric scales must be finite and positive")
    scale_map = scale.to(dtype=target_prior.dtype, device=target_prior.device).view(-1, 1, 1, 1)

    refined_error_m = (output.relative_height - target_prior) * scale_map
    prior_error_m = (geometry_prior - target_prior) * scale_map
    metric_regression = _masked_mean(
        F.smooth_l1_loss(
            refined_error_m,
            torch.zeros_like(refined_error_m),
            reduction="none",
            beta=1.0,
        ),
        valid,
    )
    metric_gradient = _metric_gradient_loss(
        output.relative_height,
        target_prior,
        valid,
        scale_map.reshape(-1),
    )
    non_degradation = _masked_mean(
        F.relu(torch.abs(refined_error_m) - torch.abs(prior_error_m)),
        valid,
    )
    correction_regularization = _masked_mean(output.relative_correction.square(), valid)

    # Keep only scale-invariant/uncertainty auxiliary signals in this cross-scene stage. Absolute
    # regression and gradient terms are handled above in metres; ordinal bins are intentionally
    # disabled because canonical targets can legitimately extend beyond DA3's nominal [0, 1].
    auxiliary_weights = HeightLossWeights(
        regression=0.0,
        heteroscedastic=0.02,
        gradient=0.0,
        correlation=0.25,
        normals=0.0,
        ordinal=0.0,
        semantics=0.0,
        boundary=0.05,
    )
    auxiliary = compute_height_losses(
        output,
        target_prior,
        valid,
        weights=auxiliary_weights,
    ).total

    total = (
        w.metric_regression * metric_regression
        + w.metric_gradient * metric_gradient
        + w.non_degradation * non_degradation
        + w.correction_regularization * correction_regularization
        + w.auxiliary * auxiliary
    )
    return MultisceneLossResult(
        total=total,
        metric_regression=metric_regression,
        metric_gradient=metric_gradient,
        non_degradation=non_degradation,
        correction_regularization=correction_regularization,
        auxiliary=auxiliary,
    )


def balanced_group_order(
    group_ids: list[int] | np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a shuffled epoch order with equal sample count from every geographic group.

    Smaller groups are sampled with replacement. This prevents locations with more valid windows
    from silently dominating a multiscene epoch while retaining all available diversity from the
    largest group.
    """
    groups = np.asarray(group_ids, dtype=np.int64)
    if groups.ndim != 1 or groups.size == 0:
        raise ValueError("group_ids must be a non-empty 1D sequence")
    unique = np.unique(groups)
    buckets = [np.flatnonzero(groups == group) for group in unique]
    if any(bucket.size == 0 for bucket in buckets):
        raise RuntimeError("balanced sampling encountered an empty group")
    target = max(int(bucket.size) for bucket in buckets)
    sampled: list[np.ndarray] = []
    for bucket in buckets:
        sampled.append(rng.choice(bucket, size=target, replace=bucket.size < target))
    order = np.concatenate(sampled)
    rng.shuffle(order)
    return order.astype(np.int64, copy=False)
