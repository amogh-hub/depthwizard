from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.geometry import infer_geometry_scene, normalize_relative_height_scene
from depthwizard.preprocess.stats import estimate_rgb_normalization_stats, normalize_with_stats
from depthwizard.tiling.grid import generate_shifted_tiles, generate_tiles


class StrideLockedBiasPrior(GeometryPrior):
    def __init__(self, *, dual_lattice: bool, stride: int) -> None:
        self.dual_lattice = dual_lattice
        self.stride = stride
        self.calls = 0

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        self.calls += 1
        height, width = rgb_normalized.shape[:2]
        yy, xx = np.mgrid[:height, :width]
        phase = 2.0 * np.pi / float(self.stride)
        artifact = 0.20 * (np.cos(phase * xx) + np.cos(phase * yy))
        relative = rgb_normalized[..., 0].astype(np.float32) + artifact.astype(np.float32)
        return GeometryPriorOutput(
            relative_height=relative,
            confidence=None,
            model_id="stride-bias-test",
            metadata={
                "scene_normalize_relative_height": True,
                "dual_lattice_mosaic": self.dual_lattice,
            },
        )


def _write_scene(path: Path, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[:size, :size]
    base = 0.15 + 0.65 * (xx / float(size - 1)) + 0.10 * (yy / float(size - 1))
    rgb = np.stack([base, base, base], axis=0).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=3,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(0, size, 1, 1),
    ) as dst:
        dst.write(rgb)
    return np.moveaxis(rgb, 0, -1)


def test_shifted_lattice_has_distinct_phase_and_full_coverage() -> None:
    primary = generate_tiles(6000, 6000, tile_size=1024, overlap=128)
    shifted = generate_shifted_tiles(6000, 6000, tile_size=1024, overlap=128)

    primary_x = sorted({tile.x for tile in primary})
    shifted_x = sorted({tile.x for tile in shifted})
    primary_y = sorted({tile.y for tile in primary})
    shifted_y = sorted({tile.y for tile in shifted})

    assert primary_x[0] == shifted_x[0] == 0
    assert primary_y[0] == shifted_y[0] == 0
    assert primary_x[-1] == shifted_x[-1] == 4976
    assert primary_y[-1] == shifted_y[-1] == 4976
    assert 896 in primary_x and 448 in shifted_x
    assert 896 in primary_y and 448 in shifted_y
    assert 4928 not in shifted_x
    assert 4928 not in shifted_y
    assert len(primary) == 49
    assert len(shifted) == 49


def test_dual_lattice_reduces_stride_locked_tile_bias(tmp_path: Path) -> None:
    path = tmp_path / "scene.tif"
    rgb = _write_scene(path)
    tile_size = 64
    overlap = 16
    stride = tile_size - overlap

    single_prior = StrideLockedBiasPrior(dual_lattice=False, stride=stride)
    dual_prior = StrideLockedBiasPrior(dual_lattice=True, stride=stride)

    single = infer_geometry_scene(
        path,
        single_prior,
        tile_size=tile_size,
        overlap=overlap,
        min_harmonization_pixels=64,
    )
    dual = infer_geometry_scene(
        path,
        dual_prior,
        tile_size=tile_size,
        overlap=overlap,
        min_harmonization_pixels=64,
    )

    stats = estimate_rgb_normalization_stats(path)
    expected_affine = normalize_with_stats(rgb, stats)[..., 0]
    expected = normalize_relative_height_scene(expected_affine)

    # Ignore the outer tile-size margin where both lattices are necessarily anchored to scene edges.
    core = np.s_[tile_size:-tile_size, tile_size:-tile_size]
    single_rmse = float(np.sqrt(np.mean((single.relative_height[core] - expected[core]) ** 2)))
    dual_rmse = float(np.sqrt(np.mean((dual.relative_height[core] - expected[core]) ** 2)))

    assert single.tile_count == 25
    assert dual.tile_count == 61
    assert dual_prior.calls == dual.tile_count
    assert dual_rmse < single_rmse * 0.80
