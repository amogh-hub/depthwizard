from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from depthwizard.height_model.terrain_structure import (
    TerrainStructureConfig,
    TerrainStructureOutput,
)


@dataclass(frozen=True)
class TerrainStructureTargets:
    """Supervision required by the terrain/structure research candidate.

    ``terrain_relative`` is a bare-earth terrain target expressed in the same relative coordinate
    system as the geometry prior. It must be produced by an explicit, auditable target-preparation
    step; the loss never invents terrain beneath buildings. ``above_ground_relative`` is nonnegative
    height above that terrain. ``relative_dsm`` must equal terrain + above-ground on valid pixels.
    """

    relative_dsm: torch.Tensor
    terrain_relative: torch.Tensor
    above_ground_relative: torch.Tensor
    valid_mask: torch.Tensor
    building_mask: torch.Tensor
    ground_mask: torch.Tensor
    boundary_mask: torch.Tensor


@dataclass(frozen=True)
class TerrainStructureLossWeights:
    recomposition: float = 0.75
    terrain_surface: float = 1.25
    terrain_slope: float = 0.75
    structure_agl: float = 2.00
    height_stratified_agl: float = 1.00
    roof_surface: float = 1.50
    ground_surface: float = 1.50
    roof_bias: float = 0.75
    ground_bias: float = 0.75
    structure_support: float = 0.50
    height_ordinal: float = 0.50
    boundary: float = 0.75
    roof_curvature: float = 0.50

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} weight must be finite and nonnegative")
        if sum(self.__dict__.values()) <= 0:
            raise ValueError("at least one terrain-structure loss weight must be positive")


@dataclass(frozen=True)
class TerrainStructureLossResult:
    total: torch.Tensor
    recomposition: torch.Tensor
    terrain_surface: torch.Tensor
    terrain_slope: torch.Tensor
    structure_agl: torch.Tensor
    height_stratified_agl: torch.Tensor
    roof_surface: torch.Tensor
    ground_surface: torch.Tensor
    roof_bias: torch.Tensor
    ground_bias: torch.Tensor
    structure_support: torch.Tensor
    height_ordinal: torch.Tensor
    boundary: torch.Tensor
    roof_curvature: torch.Tensor
    valid_fraction: torch.Tensor
    building_fraction: torch.Tensor
    ground_fraction: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    boolean = mask.to(dtype=torch.bool)
    if values.shape != boolean.shape:
        raise ValueError("masked values and mask must have identical shapes")
    count = boolean.sum()
    if int(count.detach().cpu()) == 0:
        return values.sum() * 0.0
    return values[boolean].mean()


def _validate_shape(name: str, tensor: torch.Tensor, expected: torch.Size) -> None:
    if tensor.shape != expected:
        raise ValueError(f"{name} must have shape {tuple(expected)}")


def _validate_targets(output: TerrainStructureOutput, targets: TerrainStructureTargets) -> None:
    expected = output.relative_height.shape
    if len(expected) != 4 or expected[1] != 1:
        raise ValueError("terrain-structure fields must have shape N x 1 x H x W")
    for name, tensor in (
        ("relative_dsm", targets.relative_dsm),
        ("terrain_relative", targets.terrain_relative),
        ("above_ground_relative", targets.above_ground_relative),
        ("valid_mask", targets.valid_mask),
        ("building_mask", targets.building_mask),
        ("ground_mask", targets.ground_mask),
        ("boundary_mask", targets.boundary_mask),
    ):
        _validate_shape(name, tensor, expected)
    for name, tensor in (
        ("terrain_relative output", output.terrain_relative),
        ("above_ground_amplitude_relative output", output.above_ground_amplitude_relative),
        ("above_ground_relative output", output.above_ground_relative),
        ("structure_logits", output.structure_logits),
        ("structure_probability", output.structure_probability),
        ("boundary_probability", output.boundary_probability),
    ):
        _validate_shape(name, tensor, expected)

    valid = targets.valid_mask.to(dtype=torch.bool)
    building = targets.building_mask.to(dtype=torch.bool)
    ground = targets.ground_mask.to(dtype=torch.bool)
    boundary = targets.boundary_mask.to(dtype=torch.bool)
    if torch.any(building & ground):
        raise ValueError("building_mask and ground_mask must be disjoint")
    if torch.any((building | ground | boundary) & ~valid):
        raise ValueError("semantic supervision masks must be subsets of valid_mask")
    for name, tensor in (
        ("relative_dsm", targets.relative_dsm),
        ("terrain_relative", targets.terrain_relative),
        ("above_ground_relative", targets.above_ground_relative),
    ):
        if not torch.all(torch.isfinite(tensor[valid])):
            raise ValueError(f"{name} must be finite on valid pixels")
    if torch.any(targets.above_ground_relative[valid] < 0):
        raise ValueError("above_ground_relative target must be nonnegative")
    recomposed = targets.terrain_relative + targets.above_ground_relative
    tolerance = 2e-4
    if torch.any(torch.abs(recomposed[valid] - targets.relative_dsm[valid]) > tolerance):
        raise ValueError("relative_dsm target must equal terrain + above-ground on valid pixels")


def _metric_scale_map(
    scale_m_per_prior_unit: torch.Tensor,
    batch: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    scale = scale_m_per_prior_unit.reshape(-1).to(dtype=dtype, device=device)
    if scale.numel() != batch:
        raise ValueError("one metric scale is required per batch element")
    if not torch.all(torch.isfinite(scale) & (scale > 0)):
        raise ValueError("metric scales must be finite and positive")
    return scale.view(batch, 1, 1, 1)


def _gsd_map(
    gsd_m: torch.Tensor,
    batch: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    gsd = gsd_m.reshape(-1).to(dtype=dtype, device=device)
    if gsd.numel() != batch:
        raise ValueError("one GSD value is required per batch element")
    if not torch.all(torch.isfinite(gsd) & (gsd > 0)):
        raise ValueError("GSD values must be finite and positive")
    return gsd.view(batch, 1, 1, 1)


def _smooth_l1_metric(
    prediction_relative: torch.Tensor,
    target_relative: torch.Tensor,
    scale_map: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta_m: float = 0.5,
) -> torch.Tensor:
    error_m = (prediction_relative - target_relative) * scale_map
    zeros = torch.zeros_like(error_m)
    loss = F.smooth_l1_loss(error_m, zeros, reduction="none", beta=beta_m)
    return _masked_mean(loss, mask)


def _signed_bias_loss(
    prediction_relative: torch.Tensor,
    target_relative: torch.Tensor,
    scale_map: torch.Tensor,
    mask: torch.Tensor,
    *,
    beta_m: float = 0.5,
) -> torch.Tensor:
    """Penalize systematic signed surface displacement per scene.

    Pixelwise robust losses can still tolerate a nonzero common-mode shift when positive and negative
    local errors coexist. This term computes one signed metric bias per batch item on the declared
    support and drives that bias toward zero without using any exposed-scene-specific target value.
    """

    boolean = mask.to(dtype=torch.bool)
    error_m = (prediction_relative - target_relative) * scale_map
    flat_error = error_m.flatten(start_dim=1)
    flat_mask = boolean.flatten(start_dim=1)
    counts = flat_mask.sum(dim=1)
    weighted_sum = (flat_error * flat_mask.to(dtype=flat_error.dtype)).sum(dim=1)
    biases = weighted_sum / counts.clamp_min(1).to(dtype=flat_error.dtype)
    losses = F.smooth_l1_loss(biases, torch.zeros_like(biases), reduction="none", beta=beta_m)
    valid_batches = counts > 0
    if int(valid_batches.sum().detach().cpu()) == 0:
        return error_m.sum() * 0.0
    return losses[valid_batches].mean()


def _slope_degrees(
    elevation_m: torch.Tensor,
    gsd_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = (elevation_m[..., :, 1:] - elevation_m[..., :, :-1]) / gsd_map
    dy = (elevation_m[..., 1:, :] - elevation_m[..., :-1, :]) / gsd_map
    dx_center = dx[..., :-1, :]
    dy_center = dy[..., :, :-1]
    gradient = torch.sqrt(dx_center.square() + dy_center.square() + 1e-12)
    slope = torch.rad2deg(torch.atan(gradient))
    return slope, gradient


def _terrain_slope_loss(
    prediction_relative: torch.Tensor,
    target_relative: torch.Tensor,
    scale_map: torch.Tensor,
    gsd_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    prediction_m = prediction_relative * scale_map
    target_m = target_relative * scale_map
    pred_slope, _ = _slope_degrees(prediction_m, gsd_map)
    target_slope, _ = _slope_degrees(target_m, gsd_map)
    slope_mask = (
        mask[..., :-1, :-1]
        & mask[..., 1:, :-1]
        & mask[..., :-1, 1:]
        & mask[..., 1:, 1:]
    )
    return _masked_mean(torch.abs(pred_slope - target_slope), slope_mask)


def _roof_curvature_loss(
    prediction_relative: torch.Tensor,
    target_relative: torch.Tensor,
    scale_map: torch.Tensor,
    building_mask: torch.Tensor,
) -> torch.Tensor:
    prediction_m = prediction_relative * scale_map
    target_m = target_relative * scale_map
    pred_dxx = prediction_m[..., :, 2:] - 2.0 * prediction_m[..., :, 1:-1] + prediction_m[..., :, :-2]
    target_dxx = target_m[..., :, 2:] - 2.0 * target_m[..., :, 1:-1] + target_m[..., :, :-2]
    pred_dyy = prediction_m[..., 2:, :] - 2.0 * prediction_m[..., 1:-1, :] + prediction_m[..., :-2, :]
    target_dyy = target_m[..., 2:, :] - 2.0 * target_m[..., 1:-1, :] + target_m[..., :-2, :]

    mask_x = building_mask[..., :, 2:] & building_mask[..., :, 1:-1] & building_mask[..., :, :-2]
    mask_y = building_mask[..., 2:, :] & building_mask[..., 1:-1, :] & building_mask[..., :-2, :]
    x_loss = _masked_mean(torch.abs(pred_dxx - target_dxx), mask_x)
    y_loss = _masked_mean(torch.abs(pred_dyy - target_dyy), mask_y)
    return 0.5 * (x_loss + y_loss)


def _height_stratified_agl_loss(
    prediction_relative: torch.Tensor,
    target_relative: torch.Tensor,
    building_mask: torch.Tensor,
    scale_map: torch.Tensor,
    config: TerrainStructureConfig,
) -> torch.Tensor:
    """Give each occupied target-height stratum equal authority over continuous AGL error.

    This prevents the dominant low/mid-rise pixel population from numerically drowning rare tall
    structures. The strata are the predeclared metric bin edges in the model configuration; no exposed
    evaluation scene is used to choose a special tall-building cutoff.
    """

    prediction_m = prediction_relative * scale_map
    target_m = target_relative * scale_map
    per_pixel = F.smooth_l1_loss(
        prediction_m,
        target_m,
        reduction="none",
        beta=0.5,
    )
    edges = torch.tensor(
        config.height_bin_edges_m,
        dtype=target_m.dtype,
        device=target_m.device,
    )
    classes = torch.bucketize(target_m[:, 0], edges).unsqueeze(1)
    occupied: list[torch.Tensor] = []
    for class_index in range(config.height_bins):
        stratum_mask = building_mask & (classes == class_index)
        if int(stratum_mask.sum().detach().cpu()) > 0:
            occupied.append(_masked_mean(per_pixel, stratum_mask))
    if not occupied:
        return per_pixel.sum() * 0.0
    return torch.stack(occupied).mean()


def _height_ordinal_loss(
    logits: torch.Tensor,
    target_above_ground_relative: torch.Tensor,
    building_mask: torch.Tensor,
    scale_map: torch.Tensor,
    config: TerrainStructureConfig,
) -> torch.Tensor:
    """Cumulative ordinal regression over metric height thresholds.

    For threshold ``e_k`` channel ``k`` learns ``P(height > e_k)``. This preserves height ordering,
    unlike a nominal categorical cross-entropy over disjoint bins.
    """

    if logits.ndim != 4 or logits.shape[1] != config.height_ordinal_channels:
        raise ValueError("height_ordinal_logits channel count must match configured height edges")
    if (
        logits.shape[0] != target_above_ground_relative.shape[0]
        or logits.shape[-2:] != target_above_ground_relative.shape[-2:]
    ):
        raise ValueError("height_ordinal_logits must share batch and spatial dimensions with targets")
    metric_height = target_above_ground_relative * scale_map
    edges = torch.tensor(
        config.height_bin_edges_m,
        dtype=metric_height.dtype,
        device=metric_height.device,
    ).view(1, -1, 1, 1)
    ordinal_target = (metric_height > edges).to(dtype=logits.dtype)
    per_threshold = F.binary_cross_entropy_with_logits(
        logits,
        ordinal_target,
        reduction="none",
    )
    ordinal_mask = building_mask.expand(-1, config.height_ordinal_channels, -1, -1)
    return _masked_mean(per_threshold, ordinal_mask)


def compute_terrain_structure_loss(
    output: TerrainStructureOutput,
    targets: TerrainStructureTargets,
    scale_m_per_prior_unit: torch.Tensor,
    gsd_m: torch.Tensor,
    *,
    config: TerrainStructureConfig | None = None,
    weights: TerrainStructureLossWeights | None = None,
) -> TerrainStructureLossResult:
    """Compute compensation- and compression-resistant supervision for explicit T + H_AGL.

    Roof and ground surface losses are separate by design. Signed bias terms additionally reject
    common-mode surface displacement, while continuous AGL is supervised on the ungated conditional
    amplitude so uncertain support probability cannot suppress height learning. Equal-weight metric
    height strata keep rare tall structures from disappearing inside the dominant low-rise population.
    """

    cfg = config or TerrainStructureConfig()
    loss_weights = weights or TerrainStructureLossWeights()
    _validate_targets(output, targets)

    valid = targets.valid_mask.to(dtype=torch.bool)
    building = targets.building_mask.to(dtype=torch.bool)
    ground = targets.ground_mask.to(dtype=torch.bool)
    boundary = targets.boundary_mask.to(dtype=torch.bool)
    batch = output.relative_height.shape[0]
    dtype = output.relative_height.dtype
    device = output.relative_height.device
    scale_map = _metric_scale_map(
        scale_m_per_prior_unit,
        batch,
        dtype=dtype,
        device=device,
    )
    gsd_map = _gsd_map(gsd_m, batch, dtype=dtype, device=device)

    recomposition = _smooth_l1_metric(
        output.relative_height,
        targets.relative_dsm,
        scale_map,
        valid,
    )
    terrain_surface = _smooth_l1_metric(
        output.terrain_relative,
        targets.terrain_relative,
        scale_map,
        valid,
    )
    ground_surface = _smooth_l1_metric(
        output.terrain_relative,
        targets.terrain_relative,
        scale_map,
        ground,
    )
    roof_surface = _smooth_l1_metric(
        output.relative_height,
        targets.relative_dsm,
        scale_map,
        building,
    )
    structure_agl = _smooth_l1_metric(
        output.above_ground_amplitude_relative,
        targets.above_ground_relative,
        scale_map,
        building,
    )
    height_stratified_agl = _height_stratified_agl_loss(
        output.above_ground_amplitude_relative,
        targets.above_ground_relative,
        building,
        scale_map,
        cfg,
    )
    roof_bias = _signed_bias_loss(
        output.relative_height,
        targets.relative_dsm,
        scale_map,
        building,
    )
    ground_bias = _signed_bias_loss(
        output.terrain_relative,
        targets.terrain_relative,
        scale_map,
        ground,
    )
    terrain_slope = _terrain_slope_loss(
        output.terrain_relative,
        targets.terrain_relative,
        scale_map,
        gsd_map,
        ground,
    )

    target_structure = building.to(dtype=dtype)
    structure_support_map = F.binary_cross_entropy_with_logits(
        output.structure_logits,
        target_structure,
        reduction="none",
    )
    structure_support = _masked_mean(structure_support_map, valid)

    height_ordinal = _height_ordinal_loss(
        output.height_ordinal_logits,
        targets.above_ground_relative,
        building,
        scale_map,
        cfg,
    )

    target_boundary = boundary.to(dtype=dtype)
    boundary_probability = output.boundary_probability.clamp(1e-6, 1.0 - 1e-6)
    boundary_map = F.binary_cross_entropy(
        boundary_probability,
        target_boundary,
        reduction="none",
    )
    boundary_loss = _masked_mean(boundary_map, valid)

    roof_curvature = _roof_curvature_loss(
        output.relative_height,
        targets.relative_dsm,
        scale_map,
        building,
    )

    total = (
        loss_weights.recomposition * recomposition
        + loss_weights.terrain_surface * terrain_surface
        + loss_weights.terrain_slope * terrain_slope
        + loss_weights.structure_agl * structure_agl
        + loss_weights.height_stratified_agl * height_stratified_agl
        + loss_weights.roof_surface * roof_surface
        + loss_weights.ground_surface * ground_surface
        + loss_weights.roof_bias * roof_bias
        + loss_weights.ground_bias * ground_bias
        + loss_weights.structure_support * structure_support
        + loss_weights.height_ordinal * height_ordinal
        + loss_weights.boundary * boundary_loss
        + loss_weights.roof_curvature * roof_curvature
    )

    denominator = valid.to(dtype=dtype).sum().clamp_min(1.0)
    valid_fraction = valid.to(dtype=dtype).mean()
    building_fraction = building.to(dtype=dtype).sum() / denominator
    ground_fraction = ground.to(dtype=dtype).sum() / denominator

    return TerrainStructureLossResult(
        total=total,
        recomposition=recomposition,
        terrain_surface=terrain_surface,
        terrain_slope=terrain_slope,
        structure_agl=structure_agl,
        height_stratified_agl=height_stratified_agl,
        roof_surface=roof_surface,
        ground_surface=ground_surface,
        roof_bias=roof_bias,
        ground_bias=ground_bias,
        structure_support=structure_support,
        height_ordinal=height_ordinal,
        boundary=boundary_loss,
        roof_curvature=roof_curvature,
        valid_fraction=valid_fraction,
        building_fraction=building_fraction,
        ground_fraction=ground_fraction,
    )
