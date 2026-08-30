from __future__ import annotations

import numpy as np
import torch

from depthwizard.height_model.model import HeightModelOutput
from depthwizard.height_model.structure_band import (
    StructureBandConfig,
    compute_structure_band_loss,
    physical_highpass_numpy,
    physical_highpass_torch,
    structure_activity_score,
)


def _output(correction: torch.Tensor) -> HeightModelOutput:
    batch, _, height, width = correction.shape
    zeros = torch.zeros_like(correction)
    return HeightModelOutput(
        relative_height=correction,
        relative_correction=correction,
        log_variance=zeros,
        uncertainty=torch.ones_like(correction),
        semantic_logits=torch.zeros((batch, 5, height, width), dtype=correction.dtype),
        height_bin_logits=torch.zeros((batch, 16, height, width), dtype=correction.dtype),
        normals=torch.zeros((batch, 3, height, width), dtype=correction.dtype),
        boundary_probability=torch.full_like(correction, 0.5),
        raw_relative_correction=correction,
        correction_gate=torch.ones_like(correction),
    )


def _roof(size: int = 65) -> torch.Tensor:
    values = torch.zeros((1, 1, size, size), dtype=torch.float32)
    center = size // 2
    values[..., center - 5 : center + 6, center - 7 : center + 8] = 1.0
    return values


def test_physical_highpass_removes_constant_broad_offset() -> None:
    values = torch.full((1, 1, 65, 65), 3.25, dtype=torch.float32)
    highpass = physical_highpass_torch(
        values,
        torch.tensor([0.25]),
        characteristic_scale_m=8.0,
    )
    assert torch.max(torch.abs(highpass)).item() < 1e-5


def test_physical_highpass_preserves_compact_roof_relief() -> None:
    roof = _roof()
    highpass = physical_highpass_torch(
        roof,
        torch.tensor([0.25]),
        characteristic_scale_m=8.0,
    )
    center = roof.shape[-1] // 2
    assert highpass[0, 0, center, center].item() > 0.70
    assert torch.max(highpass).item() > 0.70


def test_numpy_structure_activity_scores_roof_above_flat_scene() -> None:
    size = 129
    valid = np.ones((size, size), dtype=bool)
    flat = np.zeros((size, size), dtype=np.float32)
    roof = flat.copy()
    center = size // 2
    roof[center - 12 : center + 13, center - 12 : center + 13] = 4.0
    flat_score = structure_activity_score(flat, valid, gsd_m=0.25)
    roof_score = structure_activity_score(roof, valid, gsd_m=0.25)
    assert flat_score < 1e-6
    assert roof_score > flat_score + 0.05


def test_numpy_highpass_keeps_invalid_pixels_nan() -> None:
    values = np.zeros((33, 33), dtype=np.float32)
    values[10:20, 10:20] = 2.0
    valid = np.ones_like(values, dtype=bool)
    valid[:3, :] = False
    result = physical_highpass_numpy(
        values,
        valid,
        gsd_m=0.5,
        characteristic_scale_m=8.0,
    )
    assert np.isnan(result[:3, :]).all()
    assert np.isfinite(result[3:, :]).all()


def test_structure_band_loss_prefers_correct_local_structure_to_zero() -> None:
    geometry = torch.zeros((1, 1, 65, 65), dtype=torch.float32)
    target_correction = _roof()
    target = geometry + target_correction
    valid = torch.ones_like(geometry, dtype=torch.bool)
    gsd = torch.tensor([0.25])
    scale = torch.tensor([5.0])
    baseline = torch.tensor([3.0])
    band_target = physical_highpass_torch(
        target_correction,
        gsd,
        characteristic_scale_m=8.0,
    )

    zero_loss = compute_structure_band_loss(
        _output(torch.zeros_like(geometry)),
        geometry,
        target,
        valid,
        gsd,
        scale,
        baseline,
    )
    matching_loss = compute_structure_band_loss(
        _output(band_target),
        geometry,
        target,
        valid,
        gsd,
        scale,
        baseline,
    )
    assert matching_loss.total.item() < zero_loss.total.item() * 0.35


def test_structure_band_loss_penalizes_broad_drift_on_top_of_correct_roof() -> None:
    geometry = torch.zeros((1, 1, 65, 65), dtype=torch.float32)
    target_correction = _roof()
    target = geometry + target_correction
    valid = torch.ones_like(geometry, dtype=torch.bool)
    gsd = torch.tensor([0.25])
    scale = torch.tensor([5.0])
    baseline = torch.tensor([3.0])
    band_target = physical_highpass_torch(
        target_correction,
        gsd,
        characteristic_scale_m=8.0,
    )

    correct = compute_structure_band_loss(
        _output(band_target),
        geometry,
        target,
        valid,
        gsd,
        scale,
        baseline,
    )
    drifted = compute_structure_band_loss(
        _output(band_target + 0.35),
        geometry,
        target,
        valid,
        gsd,
        scale,
        baseline,
    )
    assert drifted.low_frequency_drift.item() > correct.low_frequency_drift.item() + 0.05
    assert drifted.total.item() > correct.total.item() + 0.05


def test_structure_weight_is_higher_when_target_contains_metric_structure() -> None:
    geometry = torch.zeros((1, 1, 65, 65), dtype=torch.float32)
    target = geometry + _roof()
    valid = torch.ones_like(geometry, dtype=torch.bool)
    result = compute_structure_band_loss(
        _output(torch.zeros_like(geometry)),
        geometry,
        target,
        valid,
        torch.tensor([0.25]),
        torch.tensor([5.0]),
        torch.tensor([3.0]),
        config=StructureBandConfig(structure_threshold_m=1.0),
    )
    assert result.structure_weight_mean.item() > 1.0
