from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.io.raster import (
    ground_sample_distance_m,
    inspect_raster,
    reproject_to_match,
    write_float_geotiff,
)


def _write(path: Path, data: np.ndarray, *, transform, crs="EPSG:32643") -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def test_reproject_to_match_and_export(tmp_path: Path) -> None:
    reference = tmp_path / "reference.tif"
    target = tmp_path / "target.tif"
    _write(reference, np.arange(16).reshape(4, 4), transform=from_origin(0, 4, 1, 1))
    _write(target, np.ones((8, 8)), transform=from_origin(0, 4, 0.5, 0.5))

    aligned, valid = reproject_to_match(reference, target)
    assert aligned.shape == (8, 8)
    assert valid.all()

    output = tmp_path / "output.tif"
    write_float_geotiff(output, aligned, template_path=target, description="test")
    with rasterio.open(output) as src:
        assert src.crs.to_string() == "EPSG:32643"
        assert src.transform == from_origin(0, 4, 0.5, 0.5)
        assert src.dtypes[0] == "float32"
        assert src.descriptions[0] == "test"


def test_ground_sample_distance_is_metric_for_projected_crs(tmp_path: Path) -> None:
    path = tmp_path / "projected.tif"
    _write(path, np.ones((8, 8)), transform=from_origin(500000, 1500000, 10, 10))

    gsd = ground_sample_distance_m(path)
    assert gsd is not None
    assert abs(gsd[0] - 10.0) < 0.05
    assert abs(gsd[1] - 10.0) < 0.05

    metadata = inspect_raster(path)
    assert metadata.ground_sample_distance_x is not None
    assert abs(metadata.ground_sample_distance_x - 10.0) < 0.05


def test_ground_sample_distance_converts_geographic_degrees_to_metres(tmp_path: Path) -> None:
    path = tmp_path / "geographic.tif"
    _write(
        path,
        np.ones((8, 8)),
        transform=from_origin(77.0, 13.0, 0.0001, 0.0001),
        crs="EPSG:4326",
    )

    gsd = ground_sample_distance_m(path)
    assert gsd is not None
    assert 10.0 < gsd[0] < 11.5
    assert 10.5 < gsd[1] < 11.5
