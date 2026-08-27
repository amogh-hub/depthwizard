from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.io.raster import (
    ground_sample_distance_m,
    inspect_raster,
    reproject_to_match,
    write_float_geotiff,
)


def _write(
    path: Path,
    data: np.ndarray,
    *,
    transform,
    crs: str | None = "EPSG:32643",
) -> None:
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


def test_crs_free_exact_grid_alignment_is_allowed_without_guessing(tmp_path: Path) -> None:
    reference = tmp_path / "reference_no_crs.tif"
    target = tmp_path / "target_no_crs.tif"
    transform = from_origin(1000.0, 2000.0, 0.25, 0.25)
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    _write(reference, values, transform=transform, crs=None)
    _write(target, np.ones((8, 8), dtype=np.float32), transform=transform, crs=None)

    aligned, valid = reproject_to_match(reference, target)

    assert valid.all()
    np.testing.assert_allclose(aligned, values)


def test_crs_free_alignment_rejects_transform_mismatch(tmp_path: Path) -> None:
    reference = tmp_path / "reference_no_crs.tif"
    target = tmp_path / "target_no_crs.tif"
    _write(
        reference,
        np.ones((8, 8), dtype=np.float32),
        transform=from_origin(1000.0, 2000.0, 0.25, 0.25),
        crs=None,
    )
    _write(
        target,
        np.ones((8, 8), dtype=np.float32),
        transform=from_origin(1000.5, 2000.0, 0.25, 0.25),
        crs=None,
    )

    with pytest.raises(ValueError, match="dimensions and affine transforms"):
        reproject_to_match(reference, target)


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


def test_ground_sample_distance_rejects_projected_extent_outside_crs_area(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset_local_but_epsg_tagged.tif"
    _write(
        path,
        np.ones((8, 8), dtype=np.float32),
        transform=from_origin(100_000_000.0, 100_000_000.0, 1.0, 1.0),
        crs="EPSG:32632",
    )

    assert ground_sample_distance_m(path) is None
    metadata = inspect_raster(path)
    assert metadata.crs == "EPSG:32632"
    assert metadata.ground_sample_distance_x is None
    assert metadata.ground_sample_distance_y is None


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


def test_crs_free_metric_affine_requires_explicit_ortholoc_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ortholoc_unpacked.tif"
    _write(
        path,
        np.ones((8, 8), dtype=np.float32),
        transform=from_origin(500.0, 600.0, 0.2, 0.3),
        crs=None,
    )

    monkeypatch.delenv("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", raising=False)
    assert ground_sample_distance_m(path) is None

    monkeypatch.setenv("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    gsd = ground_sample_distance_m(path)
    assert gsd is not None
    assert abs(gsd[0] - 0.2) < 1e-6
    assert abs(gsd[1] - 0.3) < 1e-6


def test_crs_bearing_ortholoc_override_precedes_global_crs_interpretation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "ortholoc_raw_crs_tagged.tif"
    _write(
        path,
        np.ones((8, 8), dtype=np.float32),
        transform=from_origin(11.0, 48.0, 0.18, 0.22),
        crs="EPSG:4326",
    )

    monkeypatch.delenv("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", raising=False)
    ordinary = ground_sample_distance_m(path)
    assert ordinary is not None
    assert ordinary[0] > 10_000.0
    assert ordinary[1] > 10_000.0

    monkeypatch.setenv("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    local_metric = ground_sample_distance_m(path)
    assert local_metric is not None
    assert abs(local_metric[0] - 0.18) < 1e-6
    assert abs(local_metric[1] - 0.22) < 1e-6

    metadata = inspect_raster(path)
    assert metadata.crs == "EPSG:4326"
    assert metadata.ground_sample_distance_x is not None
    assert abs(metadata.ground_sample_distance_x - 0.18) < 1e-6
