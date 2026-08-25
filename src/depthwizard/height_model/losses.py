from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from depthwizard.height_model.model import HeightModelOutput


@dataclass(frozen=True)
class HeightLossWeights:
    regression: float = 1.0
    heteroscedastic: float = 0.35
    gradient: float = 0.35
    normals: float = 0.20
    ordinal: float = 0.20
    semantics: float = 0.15
    boundary: float = 0.15


@dataclass(frozen=True)
class HeightLossResult:
    total: torch.Tensor
    regression: torch.Tensor
    heteroscedastic: torch.Tensor
    gradient: torch.Tensor
    normals: torch.Tensor
    ordinal: torch.Tensor
    semantics: torch.Tensor
    boundary: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def surface_normals_from_height(height: torch.Tensor) -> torch.Tensor:
    """Construct normalized pseudo-surface normals from a dense relative-height field."""
    if height.ndim != 4 or height.shape[1] != 1:
        raise ValueError("height must have shape N x 1 x H x W")
    dx = F.pad(height[..., :, 1:] - height[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(height[..., 1:, :] - height[..., :-1, :], (0, 0, 0, 1))
    ones = torch.ones_like(height)
    normal = torch.cat([-dx, -dy, ones], dim=1)
    return F.normalize(normal, dim=1, eps=1e-6)


def boundary_target_from_height(height: torch.Tensor) -> torch.Tensor:
    """Derive a soft, accelerator-safe boundary target from relative-height gradients."""
    dx = F.pad(torch.abs(height[..., :, 1:] - height[..., :, :-1]), (0, 1, 0, 0))
    dy = F.pad(torch.abs(height[..., 1:, :] - height[..., :-1, :]), (0, 0, 0, 1))
    magnitude = torch.sqrt(dx.square() + dy.square() + 1e-12)
    # Use an amax-based per-image normalization rather than quantile so the permanent training
    # objective remains reproducible on CUDA, CPU and Apple MPS without backend-specific fallbacks.
    scale = magnitude.flatten(1).amax(dim=1).clamp_min(1e-4).view(-1, 1, 1, 1)
    return torch.clamp(magnitude / scale, 0.0, 1.0)


def _gradient_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    pred_dx = prediction[..., :, 1:] - prediction[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    valid_dx = valid[..., :, 1:] & valid[..., :, :-1]

    pred_dy = prediction[..., 1:, :] - prediction[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    valid_dy = valid[..., 1:, :] & valid[..., :-1, :]

    loss_x = _masked_mean(torch.abs(pred_dx - target_dx), valid_dx)
    loss_y = _masked_mean(torch.abs(pred_dy - target_dy), valid_dy)
    return 0.5 * (loss_x + loss_y)


def compute_height_losses(
    output: HeightModelOutput,
    target_relative_height: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    semantic_target: torch.Tensor | None = None,
    weights: HeightLossWeights | None = None,
) -> HeightLossResult:
    """Compute the permanent multi-task objective for DepthWizard height refinement.

    The primary target is a normalized remote-sensing relative surface-height field. Metric
    elevation supervision is introduced separately through the geodetic calibration/evidence lane.
    """
    w = weights or HeightLossWeights()
    target = target_relative_height
    valid = valid_mask.to(dtype=torch.bool)
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError("target_relative_height must have shape N x 1 x H x W")
    if valid.shape != target.shape:
        raise ValueError("valid_mask must have the same shape as target_relative_height")
    if output.relative_height.shape != target.shape:
        raise ValueError("model output and target relative height must have identical shapes")

    residual = output.relative_height - target
    regression_map = F.smooth_l1_loss(output.relative_height, target, reduction="none")
    regression = _masked_mean(regression_map, valid)

    nll_map = 0.5 * (torch.exp(-output.log_variance) * residual.square() + output.log_variance)
    heteroscedastic = _masked_mean(nll_map, valid)
    gradient = _gradient_loss(output.relative_height, target, valid)

    target_normals = surface_normals_from_height(target)
    cosine = 1.0 - torch.sum(output.normals * target_normals, dim=1, keepdim=True)
    normals = _masked_mean(cosine, valid)

    bins = output.height_bin_logits.shape[1]
    ordinal_target = torch.clamp((target[:, 0] * bins).long(), min=0, max=bins - 1)
    ordinal_map = F.cross_entropy(output.height_bin_logits, ordinal_target, reduction="none")
    ordinal = _masked_mean(ordinal_map[:, None, :, :], valid)

    if semantic_target is None:
        semantics = output.semantic_logits.sum() * 0.0
    else:
        if semantic_target.ndim != 3 or semantic_target.shape != target.shape[:1] + target.shape[2:]:
            raise ValueError("semantic_target must have shape N x H x W")
        semantic_valid = (semantic_target >= 0) & valid[:, 0]
        semantic_labels = torch.clamp(semantic_target, min=0)
        semantic_map = F.cross_entropy(
            output.semantic_logits,
            semantic_labels,
            reduction="none",
        )
        semantics = _masked_mean(semantic_map[:, None, :, :], semantic_valid[:, None, :, :])

    boundary_target = boundary_target_from_height(target)
    boundary_map = F.binary_cross_entropy(
        output.boundary_probability,
        boundary_target,
        reduction="none",
    )
    boundary = _masked_mean(boundary_map, valid)

    total = (
        w.regression * regression
        + w.heteroscedastic * heteroscedastic
        + w.gradient * gradient
        + w.normals * normals
        + w.ordinal * ordinal
        + w.semantics * semantics
        + w.boundary * boundary
    )
    return HeightLossResult(
        total=total,
        regression=regression,
        heteroscedastic=heteroscedastic,
        gradient=gradient,
        normals=normals,
        ordinal=ordinal,
        semantics=semantics,
        boundary=boundary,
    )
