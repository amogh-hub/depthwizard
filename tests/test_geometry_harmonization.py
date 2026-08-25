from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.geometry import infer_geometry_scene


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


def test_overlap_harmonization_aligns_changing_tile_scale(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    y, x = np.mgrid[:64, :96]
    base = (x + y).astype(np.float32)
    rgb = np.stack([base, base, base], axis=0)
    with rasterio.open(
        path, "w", driver="GTiff", height=64, width=96, count=3, dtype="float32",
        crs="EPSG:32643", transform=from_origin(0, 64, 1, 1),
    ) as dst:
        dst.write(rgb)

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
