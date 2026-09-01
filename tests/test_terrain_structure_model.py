from __future__ import annotations

import math

import pytest
import torch

from depthwizard.height_model.terrain_structure import (
    TerrainStructureConfig,
    TerrainStructureModel,
    TerrainStructureOutput,
)
from depthwizard.height_model.terrain_structure_loss import (
    TerrainStructureTargets,
    compute_terrain_structure_loss,
)


def test_terrain_structure_model_starts_from_exact_geometry_identity() -> None:
    torch.manual_seed(4)
    config = TerrainStructureConfig(
        rgb_channels=(8, 16, 24, 32),
        terrain_channels=(4, 8, 12, 16),
        semantic_classes=3,
        height_bin_edges_m=(2.0, 5.0, 10.0),
        dropout=0.0,
    )
    model = TerrainStructureModel(config).eval()
    rgb = torch.rand((2, 3, 32, 40), dtype=torch.float32)
    geometry = torch.rand((2, 1, 32, 40), dtype=torch.float32)

    with torch.no_grad():
        output = model(rgb, geometry, gsd_m=torch.tensor([0.05, 0.25]))

    assert output.relative_height.shape == (2, 1, 32, 40)
    assert output.terrain_relative.shape == output.relative_height.shape
    assert output.above_ground_relative.shape == output.relative_height.shape
    assert output.structure_logits.shape == output.relative_height.shape
    assert output.semantic_logits.shape == (2, 3, 32, 40)
    assert output.height_bin_logits.shape == (2, 4, 32, 40)
    assert output.normals.shape == (2, 3, 32, 40)
    assert torch.equal(output.terrain_residual, torch.zeros_like(output.terrain_residual))
    assert torch.equal(
        output.above_ground_relative,
        torch.zeros_like(output.above_ground_relative),
    )
    assert torch.equal(output.relative_height, geometry)
    assert torch.allclose(
        output.structure_probability,
        torch.full_like(output.structure_probability, 0.10),
        atol=1e-7,
        rtol=0.0,
    )


def test_optional_coarse_terrain_evidence_accepts_partial_missing_values() -> None:
    config = TerrainStructureConfig(
        rgb_channels=(8, 16, 24, 32),
        terrain_channels=(4, 8, 12, 16),
        semantic_classes=3,
        height_bin_edges_m=(2.0, 5.0),
        dropout=0.0,
    )
    model = TerrainStructureModel(config).eval()
    rgb = torch.rand((1, 3, 24, 24))
    geometry = torch.rand((1, 1, 24, 24))
    coarse = torch.rand((1, 1, 24, 24))
    coarse[..., :6, :6] = torch.nan

    with torch.no_grad():
        output = model(rgb, geometry, coarse_terrain_prior=coarse, gsd_m=torch.tensor([0.05]))

    assert torch.all(torch.isfinite(output.relative_height))
    assert torch.equal(output.relative_height, geometry)


def test_configuration_rejects_invalid_height_bins() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        TerrainStructureConfig(height_bin_edges_m=(2.0, 2.0, 5.0))
    with pytest.raises(ValueError, match="finite and positive"):
        TerrainStructureConfig(height_bin_edges_m=(2.0, float("nan")))


def _synthetic_output(
    terrain_relative: torch.Tensor,
    above_ground_relative: torch.Tensor,
    *,
    bins: int,
) -> TerrainStructureOutput:
    batch, _, height, width = terrain_relative.shape
    relative_height = terrain_relative + above_ground_relative
    structure_logits = torch.zeros_like(terrain_relative)
    structure_probability = torch.sigmoid(structure_logits)
    semantic_logits = torch.zeros((batch, 3, height, width), dtype=terrain_relative.dtype)
    height_bin_logits = torch.zeros((batch, bins, height, width), dtype=terrain_relative.dtype)
    boundary_probability = torch.full_like(terrain_relative, 0.5)
    normals = torch.zeros((batch, 3, height, width), dtype=terrain_relative.dtype)
    log_variance = torch.zeros_like(terrain_relative)
    return TerrainStructureOutput(
        terrain_relative=terrain_relative,
        terrain_residual=torch.zeros_like(terrain_relative),
        above_ground_relative=above_ground_relative,
        relative_height=relative_height,
        structure_logits=structure_logits,
        structure_probability=structure_probability,
        semantic_logits=semantic_logits,
        height_bin_logits=height_bin_logits,
        boundary_probability=boundary_probability,
        normals=normals,
        terrain_log_variance=log_variance,
        structure_log_variance=log_variance,
        terrain_uncertainty=torch.ones_like(terrain_relative),
        structure_uncertainty=torch.ones_like(terrain_relative),
    )


def test_equal_roof_and_ground_shift_cannot_hide_behind_correct_agl() -> None:
    config = TerrainStructureConfig(
        semantic_classes=3,
        height_bin_edges_m=(2.0, 5.0, 10.0),
    )
    shape = (1, 1, 16, 16)
    terrain_target = torch.ones(shape) * 10.0
    building = torch.zeros(shape, dtype=torch.bool)
    building[..., 4:12, 4:12] = True
    ground = ~building
    valid = torch.ones(shape, dtype=torch.bool)
    boundary = torch.zeros(shape, dtype=torch.bool)
    boundary[..., 4, 4:12] = True
    boundary[..., 11, 4:12] = True
    boundary[..., 4:12, 4] = True
    boundary[..., 4:12, 11] = True

    agl_target = torch.zeros(shape)
    agl_target[building] = 5.0
    dsm_target = terrain_target + agl_target
    targets = TerrainStructureTargets(
        relative_dsm=dsm_target,
        terrain_relative=terrain_target,
        above_ground_relative=agl_target,
        valid_mask=valid,
        building_mask=building,
        ground_mask=ground,
        boundary_mask=boundary,
    )

    perfect = _synthetic_output(terrain_target, agl_target, bins=config.height_bins)
    # Shift terrain, roof and ground together by +2 relative units while leaving AGL exactly right.
    # This reproduces the compensation failure that a height-only objective can miss.
    shifted = _synthetic_output(terrain_target + 2.0, agl_target, bins=config.height_bins)

    scale = torch.tensor([1.0])
    gsd = torch.tensor([0.5])
    perfect_loss = compute_terrain_structure_loss(
        perfect,
        targets,
        scale,
        gsd,
        config=config,
    )
    shifted_loss = compute_terrain_structure_loss(
        shifted,
        targets,
        scale,
        gsd,
        config=config,
    )

    assert perfect_loss.structure_agl.item() == pytest.approx(0.0, abs=1e-8)
    assert shifted_loss.structure_agl.item() == pytest.approx(0.0, abs=1e-8)
    assert perfect_loss.roof_surface.item() == pytest.approx(0.0, abs=1e-8)
    assert perfect_loss.ground_surface.item() == pytest.approx(0.0, abs=1e-8)
    assert shifted_loss.roof_surface.item() > 1.0
    assert shifted_loss.ground_surface.item() > 1.0
    assert shifted_loss.terrain_surface.item() > 1.0
    assert shifted_loss.total.item() > perfect_loss.total.item() + 3.0
    assert math.isfinite(shifted_loss.total.item())


def test_loss_rejects_inconsistent_decomposition_targets() -> None:
    config = TerrainStructureConfig(semantic_classes=3, height_bin_edges_m=(2.0, 5.0))
    shape = (1, 1, 8, 8)
    terrain = torch.zeros(shape)
    agl = torch.zeros(shape)
    valid = torch.ones(shape, dtype=torch.bool)
    building = torch.zeros(shape, dtype=torch.bool)
    ground = torch.ones(shape, dtype=torch.bool)
    boundary = torch.zeros(shape, dtype=torch.bool)
    output = _synthetic_output(terrain, agl, bins=config.height_bins)
    targets = TerrainStructureTargets(
        relative_dsm=torch.ones(shape),
        terrain_relative=terrain,
        above_ground_relative=agl,
        valid_mask=valid,
        building_mask=building,
        ground_mask=ground,
        boundary_mask=boundary,
    )

    with pytest.raises(ValueError, match="must equal terrain"):
        compute_terrain_structure_loss(
            output,
            targets,
            torch.tensor([1.0]),
            torch.tensor([1.0]),
            config=config,
        )
