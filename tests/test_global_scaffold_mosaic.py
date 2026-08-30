from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.pipeline.scaffold import (
    align_tile_to_scaffold,
    high_frequency_residual,
)


def test_scaffold_residual_removes_broad_bias_without_flattening_local_structure() -> None:
    y, x = np.mgrid[:256, :256]
    scaffold = (0.002 * x + 0.001 * y).astype(np.float32)
    broad_bias = (0.8 + 0.0015 * x - 0.001 * y).astype(np.float32)
    structure = np.zeros_like(scaffold)
    structure[112:144, 112:144] = 2.0
    local = scaffold + broad_bias + structure

    residual = high_frequency_residual(local, scaffold, sigma_px=48.0)

    background = residual[64:192, 64:192].copy()
    background[48:80, 48:80] = np.nan
    assert float(np.nanstd(background)) < 0.08
    assert float(np.nanmedian(residual[120:136, 120:136])) > 1.0


def test_low_frequency_alignment_ignores_compact_structure_when_matching_scaffold() -> None:
    y, x = np.mgrid[:192, :192]
    scaffold = (0.004 * x + 0.003 * y).astype(np.float32)
    tile = (1.8 * scaffold + 4.0).astype(np.float32)
    tile[80:112, 80:112] += 3.0

    aligned, changed = align_tile_to_scaffold(
        tile,
        scaffold,
        lowpass_sigma_px=32.0,
        min_pixels=256,
    )

    mask = np.ones(tile.shape, dtype=bool)
    mask[64:128, 64:128] = False
    assert changed is True
    assert float(np.median(np.abs(aligned[mask] - scaffold[mask]))) < 0.08
    assert float(np.median(aligned[88:104, 88:104] - scaffold[88:104, 88:104])) > 1.0


class ScaffoldBiasPrior(GeometryPrior):
    def __init__(self, full_shape: tuple[int, int]) -> None:
        self.full_shape = full_shape
        self.shapes: list[tuple[int, int]] = []

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        shape = (int(rgb_normalized.shape[0]), int(rgb_normalized.shape[1]))
        self.shapes.append(shape)
        base = rgb_normalized[..., 0].astype(np.float32)
        if shape != self.full_shape:
            yy, xx = np.mgrid[: shape[0], : shape[1]]
            bowl = 0.4 * np.cos(np.pi * xx / max(shape[1] - 1, 1))
            bowl += 0.25 * np.cos(np.pi * yy / max(shape[0] - 1, 1))
            base = base + bowl.astype(np.float32)
        return GeometryPriorOutput(
            relative_height=base,
            confidence=None,
            model_id="scaffold-test",
            metadata={
                "scene_normalize_relative_height": True,
                "scene_global_scaffold": True,
            },
        )


def test_scene_scaffold_path_runs_one_overview_and_suppresses_tile_context_bias(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rgb.tif"
    y, x = np.mgrid[:64, :96]
    base = (x + 0.5 * y).astype(np.float32)
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

    prior = ScaffoldBiasPrior((64, 96))
    result = infer_geometry_scene(
        path,
        prior,
        tile_size=48,
        overlap=24,
        min_harmonization_pixels=64,
    )

    assert prior.shapes.count((64, 96)) == 1
    assert result.tile_count > 1
    assert result.harmonized_tiles >= 1
    seam_jump = np.abs(np.diff(result.relative_height, axis=1))
    assert float(np.percentile(seam_jump, 99)) < 0.08
    finite = result.relative_height[np.isfinite(result.relative_height)]
    assert float(np.percentile(finite, 1.0)) < 1e-4
    assert abs(float(np.percentile(finite, 99.0)) - 1.0) < 1e-4
