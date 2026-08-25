from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.geometry import infer_geometry_scene


class DeterministicTestPrior(GeometryPrior):
    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        height = rgb_normalized.mean(axis=-1).astype(np.float32)
        return GeometryPriorOutput(
            relative_height=height,
            confidence=np.ones_like(height, dtype=np.float32),
            model_id="test-prior",
            metadata={},
        )


def test_tiled_geometry_pipeline_has_no_shape_or_seam_failure(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    y, x = np.mgrid[:70, :90]
    rgb = np.stack([x, y, x + y], axis=0).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=70,
        width=90,
        count=3,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(0, 70, 1, 1),
    ) as dst:
        dst.write(rgb)

    result = infer_geometry_scene(path, DeterministicTestPrior(), tile_size=32, overlap=8)
    assert result.relative_height.shape == (70, 90)
    assert result.confidence is not None
    assert np.allclose(result.confidence, 1.0, atol=1e-5)
    assert result.tile_count > 1
    assert np.all(np.isfinite(result.relative_height))
