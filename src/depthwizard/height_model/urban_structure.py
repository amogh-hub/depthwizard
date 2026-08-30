from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from depthwizard.height_model.model import HeightModelOutput
from depthwizard.height_model.structure_band import (
    _masked_weighted_mean,
    _validate_gsd,
    physical_gaussian_blur_torch,
    physical_highpass_torch,
)


@dataclass(frozen=True)
class UrbanStructureConfig:
    fine_scale_m: float = 2.0
    mid_scale_m: float = 4.0
    coarse_scale_m: float = 8.0
    safety_scale_m: float = 16.0
    active_threshold_m: float = 0.75
    max_structure_weight: float = 6.0

    def __post_init__(self) -> None:
        if not 0 < self.fine_scale_m < self.mid_scale_m < self.coarse_scale_m:
            raise ValueError("structure scales must satisfy 0 < fine < mid < coarse")
        if self.safety_scale_m <= self.coarse_scale_m:
            raise ValueError("safety_scale_m must exceed coarse_scale_m")
        if self.active_threshold_m <= 0:
            raise ValueError("active_threshold_m must be positive")
        if self.max_structure_weight < 1.0:
            raise ValueError("max_structure_weight must be at least 1")


@dataclass(frozen=True)
class UrbanStructureLossWeights:
    fine_band: float = 0.15
    mid_band: float = 0.35
    coarse_band: float = 0.50
    spatial_alignment: float = 0.35
    structural_gradient: float = 0.25
    low_frequency_drift: float = 1.50
    quiet_region_non_degradation: float = 0.50


@dataclass(frozen=True)
class UrbanStructureLossResult:
    total: torch.Tensor
    fine_band: torch.Tensor
    mid_band: torch.Tensor
    coarse_band: torch.Tensor
    spatial_alignment: torch.Tensor
    structural_gradient: torch.Tensor
    low_frequency_drift: torch.Tensor
    quiet_region_non_degradation: torch.Tensor
    active_fraction: torch.Tensor


def multiscale_structure_bands_torch(
    correction: torch.Tensor,
    gsd_m: torch.Tensor,
    *,
    config: UrbanStructureConfig | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = config or UrbanStructureConfig()
    hp_fine = physical_highpass_torch(
        correction,
        gsd_m,
        characteristic_scale_m=cfg.fine_scale_m,
    )
    hp_mid = physical_highpass_torch(
        correction,
        gsd_m,
        characteristic_scale_m=cfg.mid_scale_m,
    )
    hp_coarse = physical_highpass_torch(
        correction,
        gsd_m,
        characteristic_scale_m=cfg.coarse_scale_m,
    )
    return hp_fine, hp_mid - hp_fine, hp_coarse - hp_mid, hp_coarse


def _normalized_band_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    error_scale: torch.Tensor,
) -> torch.Tensor:
    loss = F.smooth_l1_loss(
        prediction / error_scale,
        target / error_scale,
        reduction="none",
        beta=0.35,
    )
    return _masked_weighted_mean(loss, valid, weights)


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    error_scale: torch.Tensor,
) -> torch.Tensor:
    pred = prediction / error_scale
    truth = target / error_scale
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    truth_dx = truth[..., :, 1:] - truth[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    truth_dy = truth[..., 1:, :] - truth[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    x_loss = _masked_weighted_mean(torch.abs(pred_dx - truth_dx), valid_dx)
    y_loss = _masked_weighted_mean(torch.abs(pred_dy - truth_dy), valid_dy)
    return 0.5 * (x_loss + y_loss)


def _spatial_alignment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    threshold_m: float,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for index in range(prediction.shape[0]):
        active = valid[index] & (torch.abs(target[index]) >= threshold_m)
        if int(active.sum().detach().cpu()) < 8:
            continue
        pred = prediction[index][active]
        truth = target[index][active]
        pred = pred - pred.mean()
        truth = truth - truth.mean()
        denominator = torch.linalg.vector_norm(pred) * torch.linalg.vector_norm(truth)
        eps = torch.finfo(pred.dtype).eps
        if bool((denominator <= eps).detach().cpu()):
            losses.append(pred.abs().mean() * 0.0 + 1.0)
            continue
        cosine = torch.clamp(torch.dot(pred, truth) / denominator, -1.0, 1.0)
        losses.append(1.0 - cosine)
    if not losses:
        return prediction.sum() * 0.0
    return torch.stack(losses).mean()


def compute_urban_structure_loss(
    output: HeightModelOutput,
    geometry_prior: torch.Tensor,
    target_prior: torch.Tensor,
    valid_mask: torch.Tensor,
    gsd_m: torch.Tensor,
    scale_m_per_prior_unit: torch.Tensor,
    baseline_rmse_m: torch.Tensor,
    *,
    config: UrbanStructureConfig | None = None,
    weights: UrbanStructureLossWeights | None = None,
) -> UrbanStructureLossResult:
    cfg = config or UrbanStructureConfig()
    loss_weights = weights or UrbanStructureLossWeights()
    valid = valid_mask.to(dtype=torch.bool)
    if geometry_prior.shape != target_prior.shape or valid.shape != target_prior.shape:
        raise ValueError("geometry_prior, target_prior and valid_mask must have identical shapes")
    if output.relative_correction.shape != target_prior.shape:
        raise ValueError("model correction and target must have identical shapes")
    if target_prior.ndim != 4 or target_prior.shape[1] != 1:
        raise ValueError("urban-structure tensors must have shape N x 1 x H x W")

    batch = target_prior.shape[0]
    gsd = _validate_gsd(gsd_m, batch)
    scale = scale_m_per_prior_unit.reshape(-1)
    baseline = baseline_rmse_m.reshape(-1)
    if scale.numel() != batch or baseline.numel() != batch:
        raise ValueError("one metric scale and baseline RMSE are required per batch element")
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("metric scales must be finite and positive")
    if not torch.all(torch.isfinite(baseline) & (baseline > 0)):
        raise ValueError("baseline RMSE values must be finite and positive")

    dtype = target_prior.dtype
    device = target_prior.device
    scale_map = scale.to(dtype=dtype, device=device).view(-1, 1, 1, 1)
    error_scale = baseline.to(dtype=dtype, device=device).clamp_min(0.5).view(-1, 1, 1, 1)
    target_correction_m = (target_prior - geometry_prior) * scale_map
    predicted_correction_m = output.relative_correction * scale_map

    target_fine, target_mid, target_coarse, target_hp = multiscale_structure_bands_torch(
        target_correction_m,
        gsd,
        config=cfg,
    )
    pred_fine, pred_mid, pred_coarse, pred_hp = multiscale_structure_bands_torch(
        predicted_correction_m,
        gsd,
        config=cfg,
    )
    activity = torch.clamp(
        torch.abs(target_hp) / cfg.active_threshold_m,
        min=0.0,
        max=cfg.max_structure_weight - 1.0,
    )
    structure_weights = 1.0 + activity

    fine_band = _normalized_band_loss(
        pred_fine,
        target_fine,
        valid,
        structure_weights,
        error_scale,
    )
    mid_band = _normalized_band_loss(
        pred_mid,
        target_mid,
        valid,
        structure_weights,
        error_scale,
    )
    coarse_band = _normalized_band_loss(
        pred_coarse,
        target_coarse,
        valid,
        structure_weights,
        error_scale,
    )
    spatial_alignment = _spatial_alignment_loss(
        pred_hp,
        target_hp,
        valid,
        threshold_m=cfg.active_threshold_m,
    )
    structural_gradient = _gradient_loss(pred_hp, target_hp, valid, error_scale)

    predicted_low_frequency_m = physical_gaussian_blur_torch(
        predicted_correction_m,
        gsd,
        characteristic_scale_m=cfg.safety_scale_m,
    )
    low_frequency_drift = _masked_weighted_mean(
        F.smooth_l1_loss(
            predicted_low_frequency_m / error_scale,
            torch.zeros_like(predicted_low_frequency_m),
            reduction="none",
            beta=0.25,
        ),
        valid,
    )

    quiet = valid & (torch.abs(target_hp) < cfg.active_threshold_m)
    prior_error_m = (geometry_prior - target_prior) * scale_map
    refined_error_m = prior_error_m + predicted_correction_m
    quiet_region_non_degradation = _masked_weighted_mean(
        F.relu(torch.abs(refined_error_m) - torch.abs(prior_error_m)) / error_scale,
        quiet,
    )

    total = (
        loss_weights.fine_band * fine_band
        + loss_weights.mid_band * mid_band
        + loss_weights.coarse_band * coarse_band
        + loss_weights.spatial_alignment * spatial_alignment
        + loss_weights.structural_gradient * structural_gradient
        + loss_weights.low_frequency_drift * low_frequency_drift
        + loss_weights.quiet_region_non_degradation * quiet_region_non_degradation
    )
    active_fraction = _masked_weighted_mean(
        (torch.abs(target_hp) >= cfg.active_threshold_m).to(dtype=dtype),
        valid,
    )
    return UrbanStructureLossResult(
        total=total,
        fine_band=fine_band,
        mid_band=mid_band,
        coarse_band=coarse_band,
        spatial_alignment=spatial_alignment,
        structural_gradient=structural_gradient,
        low_frequency_drift=low_frequency_drift,
        quiet_region_non_degradation=quiet_region_non_degradation,
        active_fraction=active_fraction,
    )
