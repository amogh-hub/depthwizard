from __future__ import annotations

import torch

from depthwizard.height_model.model import HeightModelOutput
from depthwizard.height_model.urban_structure import (
    UrbanStructureConfig,
    compute_urban_structure_loss,
    multiscale_structure_bands_torch,
)


def _output(correction: torch.Tensor) -> HeightModelOutput:
    zeros = torch.zeros_like(correction)
    batch, _, height, width = correction.shape
    return HeightModelOutput(
        relative_height=correction,
        relative_correction=correction,
        log_variance=zeros,
        uncertainty=torch.ones_like(correction),
        semantic_logits=torch.zeros((batch, 5, height, width), dtype=correction.dtype),
        height_bin_logits=torch.zeros((batch, 16, height, width), dtype=correction.dtype),
        normals=torch.zeros((batch, 3, height, width), dtype=correction.dtype),
        boundary_probability=zeros,
        raw_relative_correction=correction,
        correction_gate=torch.ones_like(correction),
    )


def _roof_target() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    geometry = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    target = geometry.clone()
    target[..., 20:44, 20:44] = 3.0
    valid = torch.ones_like(target, dtype=torch.bool)
    return geometry, target, valid


def _loss(correction: torch.Tensor) -> float:
    geometry, target, valid = _roof_target()
    result = compute_urban_structure_loss(
        _output(correction),
        geometry,
        target,
        valid,
        torch.tensor([0.25]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
    )
    return float(result.total)


def test_multiscale_bands_reconstruct_coarse_highpass() -> None:
    correction = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    correction[..., 20:44, 20:44] = 2.0
    fine, mid, coarse, highpass = multiscale_structure_bands_torch(
        correction,
        torch.tensor([0.25]),
        config=UrbanStructureConfig(),
    )
    assert torch.allclose(fine + mid + coarse, highpass, atol=1e-5)


def test_correctly_aligned_roof_beats_shifted_roof() -> None:
    correct = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    correct[..., 20:44, 20:44] = 3.0
    shifted = torch.zeros_like(correct)
    shifted[..., 20:44, 28:52] = 3.0
    assert _loss(correct) < _loss(shifted)


def test_correct_sign_beats_opposite_sign() -> None:
    correct = torch.zeros((1, 1, 64, 64), dtype=torch.float32)
    correct[..., 20:44, 20:44] = 3.0
    opposite = -correct
    assert _loss(correct) < _loss(opposite)


def test_broad_scene_offset_is_penalized() -> None:
    geometry, target, valid = _roof_target()
    correction = target.clone()
    broad = correction + 2.0
    good = compute_urban_structure_loss(
        _output(correction),
        geometry,
        target,
        valid,
        torch.tensor([0.25]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
    )
    bad = compute_urban_structure_loss(
        _output(broad),
        geometry,
        target,
        valid,
        torch.tensor([0.25]),
        torch.tensor([1.0]),
        torch.tensor([3.0]),
    )
    assert float(good.low_frequency_drift) < float(bad.low_frequency_drift)
    assert float(good.total) < float(bad.total)
