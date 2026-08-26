from __future__ import annotations

import torch
from torch import nn

from depthwizard.height_model.losses import align_scale_shift, compute_height_losses
from depthwizard.height_model.model import (
    BidirectionalGatedFusion,
    CrossScaleContext,
    DepthWizardHeightModel,
    HeightModelConfig,
)


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


def test_bidirectional_fusion_propagates_gradients_into_both_evidence_streams() -> None:
    torch.manual_seed(26175)
    fusion = BidirectionalGatedFusion(16, 8, dropout=0.0)
    rgb = torch.rand(2, 16, 24, 20, requires_grad=True)
    geometry = torch.rand(2, 8, 24, 20, requires_grad=True)

    output = fusion(rgb, geometry)
    assert output.rgb.shape == rgb.shape
    assert output.geometry.shape == geometry.shape
    assert output.joint.shape == rgb.shape

    loss = output.joint.square().mean() + output.rgb.mean() + output.geometry.mean()
    loss.backward()

    assert rgb.grad is not None and torch.isfinite(rgb.grad).all()
    assert geometry.grad is not None and torch.isfinite(geometry.grad).all()
    assert torch.count_nonzero(rgb.grad) > 0
    assert torch.count_nonzero(geometry.grad) > 0


def test_cross_scale_context_preserves_shallow_grid_and_uses_deeper_features() -> None:
    torch.manual_seed(26175)
    module = CrossScaleContext(16, 32, dropout=0.0).eval()
    shallow = torch.rand(1, 16, 32, 40)
    deep = torch.rand(1, 32, 16, 20)

    with torch.inference_mode():
        with_context = module(shallow, deep)
        without_context = module(shallow, torch.zeros_like(deep))

    assert with_context.shape == shallow.shape
    assert torch.isfinite(with_context).all()
    assert not torch.allclose(with_context, without_context)


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


def test_architecture_version_is_explicit_and_rejects_unknown_variants() -> None:
    assert HeightModelConfig().architecture_version == "bidirectional-cross-scale-v1"
    try:
        HeightModelConfig(architecture_version="unknown")
    except ValueError as exc:
        assert "architecture_version" in str(exc)
    else:
        raise AssertionError("unknown height-model architecture should be rejected")


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
