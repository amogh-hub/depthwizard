from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.geometry import (
    _harmonize_tile,
    infer_geometry_scene,
    normalize_relative_height_scene,
)


class ChangingScalePrior(GeometryPrior):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        self.calls += 1
        base = rgb_normalized[..., 0]
        scale = 1.0 if self.calls % 2 else 1.8
        offset = 0.0 if self.calls % 3 else 0.2
        height = (scale * base + offset).astype(np.float32)
        return GeometryPriorOutput(height, None, "changing-test", {})


class SceneNormalizedChangingScalePrior(ChangingScalePrior):
    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        output = super().infer(rgb_normalized)
        return GeometryPriorOutput(
            relative_height=output.relative_height,
            confidence=output.confidence,
            model_id=output.model_id,
            metadata={"scene_normalize_relative_height": True},
        )


def _write_gradient_scene(path: Path) -> None:
    y, x = np.mgrid[:64, :96]
    base = (x + y).astype(np.float32)
    rgb = np.stack([base, base, base], axis=0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=96,
        count=3,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(rgb)


def test_overlap_harmonization_aligns_changing_tile_scale(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    _write_gradient_scene(path)

    result = infer_geometry_scene(
        path,
        ChangingScalePrior(),
        tile_size=48,
        overlap=24,
        min_harmonization_pixels=64,
    )
    assert result.harmonized_tiles >= 1
    # A globally smooth input should remain smooth through tile boundaries after harmonization.
    seam_jump = np.abs(np.diff(result.relative_height, axis=1))
    assert np.percentile(seam_jump, 99) < 0.05


def test_scene_global_normalization_runs_after_harmonization(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    _write_gradient_scene(path)

    result = infer_geometry_scene(
        path,
        SceneNormalizedChangingScalePrior(),
        tile_size=48,
        overlap=24,
        min_harmonization_pixels=64,
    )

    finite = result.relative_height[np.isfinite(result.relative_height)]
    assert float(np.min(finite)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.max(finite)) == pytest.approx(1.0, abs=1e-6)
    seam_jump = np.abs(np.diff(result.relative_height, axis=1))
    assert np.percentile(seam_jump, 99) < 0.05


def test_scene_normalization_does_not_invent_relief_in_flat_regions() -> None:
    x = np.linspace(0.0, 2.0 * np.pi, 128, dtype=np.float32)
    low_relief = 10.0 + 0.01 * np.sin(x)
    high_relief = 10.0 + 1.0 * np.sin(x)
    field = np.vstack(
        [
            np.repeat(low_relief[None, :], 32, axis=0),
            np.repeat(high_relief[None, :], 32, axis=0),
        ]
    )

    normalized = normalize_relative_height_scene(field, low_percentile=0, high_percentile=100)
    low_span = float(np.ptp(normalized[:32]))
    high_span = float(np.ptp(normalized[32:]))
    assert low_span < high_span * 0.02


def test_low_information_overlap_cannot_rescale_an_entire_tile() -> None:
    rng = np.random.default_rng(20260830)
    tile = (10.0 + rng.normal(0.0, 1e-3, size=(32, 32))).astype(np.float32)
    existing = (20.0 + rng.normal(0.0, 1e-3, size=(32, 32))).astype(np.float32)
    overlap_mask = np.ones((32, 32), dtype=bool)

    aligned, changed = _harmonize_tile(
        tile,
        existing,
        overlap_mask,
        min_overlap_pixels=256,
    )

    assert changed is True
    # Weakly correlated near-flat evidence may receive an additive baseline alignment, but its
    # within-tile relief must not be amplified by an unstable affine scale fit.
    assert float(np.ptp(aligned)) == pytest.approx(float(np.ptp(tile)), rel=1e-4, abs=1e-7)
