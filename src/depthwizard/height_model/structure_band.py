from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import functional as F

from depthwizard.height_model.model import HeightModelOutput


@dataclass(frozen=True)
class StructureBandConfig:
    """Physical-scale contract for structure-only height refinement.

    ``target_outer_scale_m`` defines the broadest structure scale used as a supervised residual
    target. ``safety_outer_scale_m`` is intentionally larger: after patch predictions are mosaicked,
    a final full-scene high-pass at this scale removes any remaining broad terrain drift without
    repeatedly suppressing the 2--8 m object-scale signal learned by the model.
    """

    target_outer_scale_m: float = 8.0
    safety_outer_scale_m: float = 16.0
    structure_threshold_m: float = 1.0
    max_structure_weight: float = 5.0

    def __post_init__(self) -> None:
        if self.target_outer_scale_m <= 0:
            raise ValueError("target_outer_scale_m must be positive")
        if self.safety_outer_scale_m <= self.target_outer_scale_m:
            raise ValueError("safety_outer_scale_m must exceed target_outer_scale_m")
        if self.structure_threshold_m <= 0:
            raise ValueError("structure_threshold_m must be positive")
        if self.max_structure_weight < 1.0:
            raise ValueError("max_structure_weight must be at least 1")


@dataclass(frozen=True)
class StructureBandLossWeights:
    structural_regression: float = 1.0
    structural_gradient: float = 0.45
    low_frequency_drift: float = 1.25
    quiet_region_budget: float = 0.20
    quiet_region_non_degradation: float = 0.50


@dataclass(frozen=True)
class StructureBandLossResult:
    total: torch.Tensor
    structural_regression: torch.Tensor
    structural_gradient: torch.Tensor
    low_frequency_drift: torch.Tensor
    quiet_region_budget: torch.Tensor
    quiet_region_non_degradation: torch.Tensor
    structure_weight_mean: torch.Tensor


def _validate_gsd(gsd_m: torch.Tensor, batch: int) -> torch.Tensor:
    gsd = gsd_m.reshape(-1)
    if gsd.numel() != batch:
        raise ValueError("one metric GSD value is required per batch element")
    if not torch.all(torch.isfinite(gsd) & (gsd > 0)):
        raise ValueError("metric GSD values must be finite and positive")
    return gsd


def _gaussian_sigma_px(characteristic_scale_m: float, gsd_m: float) -> float:
    if characteristic_scale_m <= 0 or gsd_m <= 0:
        raise ValueError("physical scale and GSD must be positive")
    # Match the exposed Potsdam structure diagnostic convention: characteristic scale = 2 sigma.
    return characteristic_scale_m / (2.0 * gsd_m)


def _gaussian_kernel_1d(
    sigma_px: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    max_radius: int,
) -> torch.Tensor:
    if not math.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError("sigma_px must be finite and positive")
    radius = min(max(1, math.ceil(3.0 * sigma_px)), max_radius)
    coordinates = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (coordinates / sigma_px).square())
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def _blur_one(sample: torch.Tensor, *, sigma_px: float) -> torch.Tensor:
    if sample.ndim != 4 or sample.shape[0] != 1 or sample.shape[1] != 1:
        raise ValueError("sample must have shape 1 x 1 x H x W")
    height, width = sample.shape[-2:]
    if height < 3 or width < 3:
        raise ValueError("spatial dimensions must be at least 3 pixels")
    max_radius = max(1, min(height, width) // 2)
    kernel = _gaussian_kernel_1d(
        sigma_px,
        dtype=sample.dtype,
        device=sample.device,
        max_radius=max_radius,
    )
    radius = kernel.numel() // 2
    padding_mode = "reflect" if radius < min(height, width) else "replicate"

    horizontal = kernel.view(1, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1)
    blurred = F.conv2d(
        F.pad(sample, (radius, radius, 0, 0), mode=padding_mode),
        horizontal,
    )
    return F.conv2d(
        F.pad(blurred, (0, 0, radius, radius), mode=padding_mode),
        vertical,
    )


def physical_gaussian_blur_torch(
    values: torch.Tensor,
    gsd_m: torch.Tensor,
    *,
    characteristic_scale_m: float,
) -> torch.Tensor:
    """Differentiable per-sample Gaussian low-pass defined in physical metres."""
    if values.ndim != 4 or values.shape[1] != 1:
        raise ValueError("values must have shape N x 1 x H x W")
    gsd = _validate_gsd(gsd_m, values.shape[0])
    outputs: list[torch.Tensor] = []
    for index in range(values.shape[0]):
        sigma = _gaussian_sigma_px(characteristic_scale_m, float(gsd[index].detach().cpu()))
        outputs.append(_blur_one(values[index : index + 1], sigma_px=sigma))
    return torch.cat(outputs, dim=0)


def physical_highpass_torch(
    values: torch.Tensor,
    gsd_m: torch.Tensor,
    *,
    characteristic_scale_m: float,
) -> torch.Tensor:
    """Return local physical-scale content while removing broader terrain drift."""
    return values - physical_gaussian_blur_torch(
        values,
        gsd_m,
        characteristic_scale_m=characteristic_scale_m,
    )


def physical_highpass_numpy(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gsd_m: float,
    characteristic_scale_m: float,
) -> np.ndarray:
    """NaN-safe full-scene physical high-pass for scoring and safety projection."""
    data = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool) & np.isfinite(data)
    if data.ndim != 2 or valid.shape != data.shape:
        raise ValueError("values and valid_mask must be matching 2D arrays")
    if not valid.any():
        raise ValueError("physical high-pass requires at least one valid pixel")
    sigma = _gaussian_sigma_px(characteristic_scale_m, gsd_m)
    numerator = gaussian_filter(np.where(valid, data, 0.0), sigma=sigma, mode="nearest")
    denominator = gaussian_filter(valid.astype(np.float64), sigma=sigma, mode="nearest")
    smooth = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )
    result = np.full(data.shape, np.nan, dtype=np.float32)
    result[valid] = (data[valid] - smooth[valid]).astype(np.float32)
    return result


def project_structure_correction_numpy(
    correction: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gsd_m: float,
    config: StructureBandConfig | None = None,
) -> np.ndarray:
    """Hard full-scene safety projection applied after learned patch corrections are mosaicked."""
    cfg = config or StructureBandConfig()
    return physical_highpass_numpy(
        correction,
        valid_mask,
        gsd_m=gsd_m,
        characteristic_scale_m=cfg.safety_outer_scale_m,
    )


def structure_activity_score(
    target_correction_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    gsd_m: float,
    config: StructureBandConfig | None = None,
) -> float:
    """Score one scene/patch by robust structure-scale residual energy for oversampling."""
    cfg = config or StructureBandConfig()
    highpass = physical_highpass_numpy(
        target_correction_m,
        valid_mask,
        gsd_m=gsd_m,
        characteristic_scale_m=cfg.target_outer_scale_m,
    )
    selected = np.abs(highpass[np.isfinite(highpass)])
    if selected.size == 0:
        return 0.0
    # P90 rewards genuine object-rich patches without allowing one extreme pixel to dominate.
    return float(np.percentile(selected, 90.0))


def _masked_weighted_mean(
    values: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    selected = valid.to(dtype=torch.bool)
    if not torch.any(selected):
        return values.sum() * 0.0
    if weights is None:
        return values[selected].mean()
    if weights.shape != values.shape:
        raise ValueError("weights must match values")
    numerator = (values * weights)[selected].sum()
    denominator = weights[selected].sum().clamp_min(torch.finfo(values.dtype).eps)
    return numerator / denominator


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]
    weight_dx = 0.5 * (weights[..., :, 1:] + weights[..., :, :-1])

    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]
    weight_dy = 0.5 * (weights[..., 1:, :] + weights[..., :-1, :])

    x_loss = _masked_weighted_mean(torch.abs(pred_dx - target_dx), valid_dx, weight_dx)
    y_loss = _masked_weighted_mean(torch.abs(pred_dy - target_dy), valid_dy, weight_dy)
    return 0.5 * (x_loss + y_loss)


def compute_structure_band_loss(
    output: HeightModelOutput,
    geometry_prior: torch.Tensor,
    target_prior: torch.Tensor,
    valid_mask: torch.Tensor,
    gsd_m: torch.Tensor,
    scale_m_per_prior_unit: torch.Tensor,
    baseline_rmse_m: torch.Tensor,
    *,
    config: StructureBandConfig | None = None,
    weights: StructureBandLossWeights | None = None,
) -> StructureBandLossResult:
    """Train object-scale residuals while strongly discouraging broad terrain correction.

    The supervised target is the 2--8 m high-frequency part of the scene-canonicalized
    reference-minus-DA3 residual. The network correction itself is trained directly against that
    band-limited target. A separate low-pass penalty makes broad correction expensive. Production
    inference adds a second, broader 16 m full-scene safety projection after patch mosaicking.
    """
    cfg = config or StructureBandConfig()
    loss_weights = weights or StructureBandLossWeights()
    valid = valid_mask.to(dtype=torch.bool)
    if geometry_prior.shape != target_prior.shape or valid.shape != target_prior.shape:
        raise ValueError("geometry_prior, target_prior and valid_mask must have identical shapes")
    if output.relative_correction.shape != target_prior.shape:
        raise ValueError("model correction and target must have identical shapes")
    if target_prior.ndim != 4 or target_prior.shape[1] != 1:
        raise ValueError("structure-band tensors must have shape N x 1 x H x W")

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

    target_correction = target_prior - geometry_prior
    predicted_correction = output.relative_correction
    target_highpass = physical_highpass_torch(
        target_correction,
        gsd,
        characteristic_scale_m=cfg.target_outer_scale_m,
    )
    predicted_lowpass = physical_gaussian_blur_torch(
        predicted_correction,
        gsd,
        characteristic_scale_m=cfg.target_outer_scale_m,
    )

    target_highpass_m = target_highpass * scale_map
    predicted_correction_m = predicted_correction * scale_map
    predicted_lowpass_m = predicted_lowpass * scale_map

    normalized_target = target_highpass_m / error_scale
    normalized_prediction = predicted_correction_m / error_scale
    activity = torch.clamp(
        torch.abs(target_highpass_m) / cfg.structure_threshold_m,
        min=0.0,
        max=cfg.max_structure_weight - 1.0,
    )
    structure_weights = 1.0 + activity

    regression_map = F.smooth_l1_loss(
        normalized_prediction,
        normalized_target,
        reduction="none",
        beta=0.5,
    )
    structural_regression = _masked_weighted_mean(
        regression_map,
        valid,
        structure_weights,
    )
    structural_gradient = _gradient_loss(
        normalized_prediction,
        normalized_target,
        valid,
        structure_weights,
    )

    low_frequency_drift = _masked_weighted_mean(
        F.smooth_l1_loss(
            predicted_lowpass_m / error_scale,
            torch.zeros_like(predicted_lowpass_m),
            reduction="none",
            beta=0.25,
        ),
        valid,
    )

    quiet = valid & (torch.abs(target_highpass_m) < cfg.structure_threshold_m)
    quiet_region_budget = _masked_weighted_mean(
        F.smooth_l1_loss(
            predicted_correction_m / error_scale,
            torch.zeros_like(predicted_correction_m),
            reduction="none",
            beta=0.25,
        ),
        quiet,
    )
    prior_error_m = (geometry_prior - target_prior) * scale_map
    refined_error_m = prior_error_m + predicted_correction_m
    quiet_region_non_degradation = _masked_weighted_mean(
        F.relu(torch.abs(refined_error_m) - torch.abs(prior_error_m)) / error_scale,
        quiet,
    )

    total = (
        loss_weights.structural_regression * structural_regression
        + loss_weights.structural_gradient * structural_gradient
        + loss_weights.low_frequency_drift * low_frequency_drift
        + loss_weights.quiet_region_budget * quiet_region_budget
        + loss_weights.quiet_region_non_degradation * quiet_region_non_degradation
    )
    return StructureBandLossResult(
        total=total,
        structural_regression=structural_regression,
        structural_gradient=structural_gradient,
        low_frequency_drift=low_frequency_drift,
        quiet_region_budget=quiet_region_budget,
        quiet_region_non_degradation=quiet_region_non_degradation,
        structure_weight_mean=_masked_weighted_mean(structure_weights, valid),
    )
