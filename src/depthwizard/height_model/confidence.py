from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from depthwizard.height_model.losses import boundary_target_from_height
from depthwizard.height_model.model import HeightModelOutput


@dataclass(frozen=True)
class ConfidenceGatedLossWeights:
    """Weights for conservative cross-scene residual learning.

    V3 showed that the bidirectional refiner can improve two geographically disjoint validation
    locations while still slightly degrading an already-inspected out-of-place aggregate. V4
    therefore separates *candidate correction learning* from *correction application*. The raw
    candidate learns the reference residual, while a second head learns whether applying that
    candidate is beneficial. All principal terms are normalized by each training scene's DA3
    baseline RMSE so difficult/noisy locations do not dominate simply because their errors are
    larger in metres.
    """

    normalized_regression: float = 1.0
    candidate_regression: float = 0.30
    normalized_gradient: float = 0.30
    non_degradation: float = 0.75
    residual_correlation: float = 0.20
    correction_budget: float = 0.08
    gate_calibration: float = 0.25
    scene_bias: float = 0.05
    uncertainty_nll: float = 0.05
    boundary: float = 0.03


@dataclass(frozen=True)
class ConfidenceGatedLossResult:
    total: torch.Tensor
    normalized_regression: torch.Tensor
    candidate_regression: torch.Tensor
    normalized_gradient: torch.Tensor
    non_degradation: torch.Tensor
    residual_correlation: torch.Tensor
    correction_budget: torch.Tensor
    gate_calibration: torch.Tensor
    scene_bias: torch.Tensor
    uncertainty_nll: torch.Tensor
    boundary: torch.Tensor
    oracle_gate_mean: torch.Tensor
    predicted_gate_mean: torch.Tensor


def _masked_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    selected = values[valid]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def _correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = valid.to(dtype=prediction.dtype)
    count = mask.sum(dim=(-2, -1), keepdim=True)
    safe_count = count.clamp_min(1.0)
    prediction_mean = (prediction * mask).sum(dim=(-2, -1), keepdim=True) / safe_count
    target_mean = (target * mask).sum(dim=(-2, -1), keepdim=True) / safe_count
    prediction_centered = (prediction - prediction_mean) * mask
    target_centered = (target - target_mean) * mask
    covariance = (prediction_centered * target_centered).sum(dim=(-2, -1), keepdim=True)
    prediction_energy = prediction_centered.square().sum(dim=(-2, -1), keepdim=True)
    target_energy = target_centered.square().sum(dim=(-2, -1), keepdim=True)
    denominator = torch.sqrt(prediction_energy * target_energy + eps)
    correlation = covariance / denominator
    usable = (count >= 2.0) & (prediction_energy > eps) & (target_energy > eps)
    selected = correlation[usable]
    if selected.numel() == 0:
        return prediction.sum() * 0.0
    return (1.0 - torch.clamp(selected, -1.0, 1.0)).mean()


def _normalized_gradient_loss(
    final_correction_m: torch.Tensor,
    target_correction_m: torch.Tensor,
    valid: torch.Tensor,
    error_scale_m: torch.Tensor,
) -> torch.Tensor:
    predicted = final_correction_m / error_scale_m
    target = target_correction_m / error_scale_m

    pred_dx = predicted[..., :, 1:] - predicted[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]

    pred_dy = predicted[..., 1:, :] - predicted[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]

    loss_x = _masked_mean(torch.abs(pred_dx - target_dx), valid_dx)
    loss_y = _masked_mean(torch.abs(pred_dy - target_dy), valid_dy)
    return 0.5 * (loss_x + loss_y)


def _scene_bias_loss(
    normalized_correction: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    mask = valid.to(dtype=normalized_correction.dtype)
    count = mask.sum(dim=(-2, -1), keepdim=True)
    safe_count = count.clamp_min(1.0)
    mean = (normalized_correction * mask).sum(dim=(-2, -1), keepdim=True) / safe_count
    usable = count > 0
    selected = torch.abs(mean[usable])
    if selected.numel() == 0:
        return normalized_correction.sum() * 0.0
    return selected.mean()


def compute_confidence_gated_refinement_loss(
    output: HeightModelOutput,
    geometry_prior: torch.Tensor,
    target_prior: torch.Tensor,
    valid_mask: torch.Tensor,
    scale_m_per_prior_unit: torch.Tensor,
    baseline_rmse_m: torch.Tensor,
    *,
    weights: ConfidenceGatedLossWeights | None = None,
    error_scale_floor_m: float = 0.5,
    gate_full_benefit_fraction: float = 0.25,
) -> ConfidenceGatedLossResult:
    """Train a candidate residual and a learned safety gate without weakening DA3 fallback.

    The scene-level positive affine canonicalization has already removed global scale and offset
    from ``target_prior``. The remaining target residual is therefore the local structure that the
    RGB/geometry refiner should learn. V4 trains the raw candidate directly on that residual, then
    supervises the gate with an oracle benefit signal computed *only from training reference data*:
    a candidate receives gate support where it actually reduces absolute DA3 error.

    Final correction losses are divided by each scene's training-only DA3 baseline RMSE. This
    equalizes optimization pressure across easy and hard scenes while keeping a separate gate and
    non-degradation penalty to protect the frozen geometry prior.
    """
    w = weights or ConfidenceGatedLossWeights()
    valid = valid_mask.to(dtype=torch.bool)
    if output.relative_height.shape != target_prior.shape:
        raise ValueError("output and target_prior must have identical shapes")
    if geometry_prior.shape != target_prior.shape or valid.shape != target_prior.shape:
        raise ValueError("geometry_prior, target_prior and valid_mask must have identical shapes")
    if target_prior.ndim != 4 or target_prior.shape[1] != 1:
        raise ValueError("confidence-gated targets must have shape N x 1 x H x W")
    if output.raw_relative_correction is None or output.correction_gate is None:
        raise ValueError("confidence-gated loss requires raw correction and correction gate outputs")
    if output.raw_relative_correction.shape != target_prior.shape:
        raise ValueError("raw correction must match target shape")
    if output.correction_gate.shape != target_prior.shape:
        raise ValueError("correction gate must match target shape")

    batch = target_prior.shape[0]
    scale = scale_m_per_prior_unit.reshape(-1)
    baseline = baseline_rmse_m.reshape(-1)
    if scale.numel() != batch or baseline.numel() != batch:
        raise ValueError("one metric scale and baseline RMSE are required per batch element")
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("metric scales must be finite and positive")
    if not torch.all(torch.isfinite(baseline) & (baseline > 0)):
        raise ValueError("baseline RMSE values must be finite and positive")
    if error_scale_floor_m <= 0:
        raise ValueError("error_scale_floor_m must be positive")
    if gate_full_benefit_fraction <= 0:
        raise ValueError("gate_full_benefit_fraction must be positive")

    dtype = target_prior.dtype
    device = target_prior.device
    scale_map = scale.to(dtype=dtype, device=device).view(-1, 1, 1, 1)
    error_scale = torch.clamp(
        baseline.to(dtype=dtype, device=device),
        min=error_scale_floor_m,
    ).view(-1, 1, 1, 1)

    target_correction_m = (target_prior - geometry_prior) * scale_map
    candidate_correction_m = output.raw_relative_correction * scale_map
    final_correction_m = output.relative_correction * scale_map
    prior_error_m = (geometry_prior - target_prior) * scale_map
    candidate_error_m = prior_error_m + candidate_correction_m
    final_error_m = prior_error_m + final_correction_m

    normalized_final_error = final_error_m / error_scale
    normalized_regression = _masked_mean(
        F.smooth_l1_loss(
            normalized_final_error,
            torch.zeros_like(normalized_final_error),
            reduction="none",
            beta=0.5,
        ),
        valid,
    )

    normalized_candidate_error = (candidate_correction_m - target_correction_m) / error_scale
    candidate_regression = _masked_mean(
        F.smooth_l1_loss(
            normalized_candidate_error,
            torch.zeros_like(normalized_candidate_error),
            reduction="none",
            beta=0.5,
        ),
        valid,
    )

    normalized_gradient = _normalized_gradient_loss(
        final_correction_m,
        target_correction_m,
        valid,
        error_scale,
    )
    non_degradation = _masked_mean(
        F.relu(torch.abs(final_error_m) - torch.abs(prior_error_m)) / error_scale,
        valid,
    )
    residual_correlation = _correlation_loss(
        candidate_correction_m / error_scale,
        target_correction_m / error_scale,
        valid,
    )
    correction_budget = _masked_mean(
        F.smooth_l1_loss(
            final_correction_m / error_scale,
            torch.zeros_like(final_correction_m),
            reduction="none",
            beta=0.5,
        ),
        valid,
    )

    candidate_benefit_m = torch.abs(prior_error_m) - torch.abs(candidate_error_m)
    full_benefit_scale = (gate_full_benefit_fraction * error_scale).clamp_min(1e-4)
    oracle_gate = torch.clamp(candidate_benefit_m / full_benefit_scale, 0.0, 1.0).detach()
    gate = torch.clamp(output.correction_gate, min=1e-5, max=1.0 - 1e-5)
    gate_calibration = _masked_mean(
        F.binary_cross_entropy(gate, oracle_gate, reduction="none"),
        valid,
    )
    scene_bias = _scene_bias_loss(final_correction_m / error_scale, valid)

    normalized_nll = 0.5 * (
        torch.exp(-output.log_variance) * normalized_final_error.square()
        + output.log_variance
    )
    uncertainty_nll = _masked_mean(normalized_nll, valid)

    boundary_target = boundary_target_from_height(target_prior)
    boundary = _masked_mean(
        F.binary_cross_entropy(
            output.boundary_probability,
            boundary_target,
            reduction="none",
        ),
        valid,
    )

    total = (
        w.normalized_regression * normalized_regression
        + w.candidate_regression * candidate_regression
        + w.normalized_gradient * normalized_gradient
        + w.non_degradation * non_degradation
        + w.residual_correlation * residual_correlation
        + w.correction_budget * correction_budget
        + w.gate_calibration * gate_calibration
        + w.scene_bias * scene_bias
        + w.uncertainty_nll * uncertainty_nll
        + w.boundary * boundary
    )
    return ConfidenceGatedLossResult(
        total=total,
        normalized_regression=normalized_regression,
        candidate_regression=candidate_regression,
        normalized_gradient=normalized_gradient,
        non_degradation=non_degradation,
        residual_correlation=residual_correlation,
        correction_budget=correction_budget,
        gate_calibration=gate_calibration,
        scene_bias=scene_bias,
        uncertainty_nll=uncertainty_nll,
        boundary=boundary,
        oracle_gate_mean=_masked_mean(oracle_gate, valid),
        predicted_gate_mean=_masked_mean(output.correction_gate, valid),
    )