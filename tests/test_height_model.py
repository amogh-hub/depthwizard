from __future__ import annotations

import torch
from torch import nn

from depthwizard.height_model.losses import align_scale_shift, compute_height_losses
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig


def small_model() -> DepthWizardHeightModel:
    return DepthWizardHeightModel(
        HeightModelConfig(
            rgb_channels=(16, 24, 32, 48),
            geometry_channels=(8, 12, 16, 24),
            semantic_classes=5,
            height_bins=8,
            dropout=0.0,
        )
    )


def test_height_model_emits_all_dense_heads() -> None:
    torch.manual_seed(26175)
    model = small_model().eval()
    rgb = torch.rand(2, 3, 64, 80)
    geometry = torch.rand(2, 1, 64, 80)
    gsd = torch.tensor([0.5, float("nan")])

    with torch.inference_mode():
        output = model(rgb, geometry, gsd_m=gsd)

    assert output.relative_height.shape == (2, 1, 64, 80)
    assert output.relative_correction.shape == (2, 1, 64, 80)
    assert output.uncertainty.shape == (2, 1, 64, 80)
    assert output.semantic_logits.shape == (2, 5, 64, 80)
    assert output.height_bin_logits.shape == (2, 8, 64, 80)
    assert output.normals.shape == (2, 3, 64, 80)
    assert output.boundary_probability.shape == (2, 1, 64, 80)
    assert torch.isfinite(output.relative_height).all()
    assert torch.max(torch.abs(output.relative_correction)) <= model.config.max_relative_correction
    assert torch.all(output.uncertainty > 0)
    normal_lengths = torch.linalg.vector_norm(output.normals, dim=1)
    assert torch.allclose(normal_lengths, torch.ones_like(normal_lengths), atol=1e-4)


def test_untrained_height_refiner_is_exact_identity_on_geometry_prior() -> None:
    torch.manual_seed(26175)
    model = small_model().eval()
    rgb = torch.rand(1, 3, 48, 64)
    geometry = torch.rand(1, 1, 48, 64)

    with torch.inference_mode():
        output = model(rgb, geometry, gsd_m=torch.tensor([0.5]))

    assert torch.count_nonzero(output.relative_correction) == 0
    assert torch.equal(output.relative_height, geometry)


def test_group_norm_is_valid_for_non_multiple_of_eight_channel_widths() -> None:
    model = DepthWizardHeightModel(
        HeightModelConfig(
            rgb_channels=(10, 14, 22, 30),
            geometry_channels=(7, 11, 13, 17),
            semantic_classes=5,
            height_bins=8,
            dropout=0.0,
        )
    )

    group_norms = [module for module in model.modules() if isinstance(module, nn.GroupNorm)]
    assert group_norms
    assert all(module.num_channels % module.num_groups == 0 for module in group_norms)


def test_affine_alignment_removes_relative_scale_and_offset_ambiguity() -> None:
    prediction = torch.linspace(0.05, 0.95, steps=64, dtype=torch.float32).reshape(1, 1, 8, 8)
    target = 2.75 * prediction + 1.4
    valid = torch.ones_like(prediction, dtype=torch.bool)
    valid[..., :2, :2] = False

    aligned = align_scale_shift(prediction, target, valid)

    assert torch.allclose(aligned[valid], target[valid], atol=2e-5, rtol=2e-5)


def test_height_losses_reward_scene_consistent_geometry() -> None:
    torch.manual_seed(26175)
    model = small_model().eval()
    rgb = torch.rand(1, 3, 48, 48)
    geometry = torch.rand(1, 1, 48, 48)
    valid = torch.ones_like(geometry, dtype=torch.bool)

    with torch.inference_mode():
        output = model(rgb, geometry)

    identical = compute_height_losses(output, geometry, valid)
    shifted_target = geometry + 0.15 * torch.sin(
        torch.linspace(0.0, 6.0, steps=48, dtype=geometry.dtype)
    ).view(1, 1, 1, 48)
    distorted = compute_height_losses(output, shifted_target, valid)

    assert identical.regression < distorted.regression
    assert identical.correlation <= distorted.correlation


def test_height_losses_are_finite_and_differentiable() -> None:
    torch.manual_seed(26175)
    model = small_model().train()
    rgb = torch.rand(1, 3, 48, 48)
    geometry = torch.rand(1, 1, 48, 48)
    target = geometry + 0.08 * torch.rand(1, 1, 48, 48)
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[..., :4, :4] = False
    semantic = torch.randint(0, 5, (1, 48, 48))
    semantic[:, :4, :4] = -1

    output = model(rgb, geometry, gsd_m=torch.tensor([1.2]))
    losses = compute_height_losses(
        output,
        target,
        valid,
        semantic_target=semantic,
    )
    assert torch.isfinite(losses.total)
    assert losses.total.item() > 0

    losses.total.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    residual_gradient = model.height_residual_head.weight.grad
    assert residual_gradient is not None
    assert torch.isfinite(residual_gradient).all()
    assert torch.count_nonzero(residual_gradient) > 0


def test_height_model_rejects_misaligned_geometry() -> None:
    model = small_model()
    rgb = torch.rand(1, 3, 64, 64)
    geometry = torch.rand(1, 1, 32, 32)

    try:
        model(rgb, geometry)
    except ValueError as exc:
        assert "share batch and spatial dimensions" in str(exc)
    else:
        raise AssertionError("misaligned geometry should be rejected")