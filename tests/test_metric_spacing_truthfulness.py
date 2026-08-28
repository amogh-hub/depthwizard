from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.cli import mesh_rdsm
from depthwizard.evaluation.report import validate_geospatial_dsm
from depthwizard.io.raster import ground_sample_distance_m
from depthwizard.pipeline.runtime import ProductionElevationRuntime


def _write_high_latitude_web_mercator(path: Path, data: np.ndarray, *, count: int = 1) -> None:
    # EPSG:3857 northing near 60 degrees latitude. Ten projected map metres are only about five
    # metres on the ground here, making raw affine spacing observably wrong for metric analytics.
    transform = from_origin(0.0, 8_399_737.89, 10.0, 10.0)
    height, width = data.shape[-2:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=str(data.dtype),
        crs="EPSG:3857",
        transform=transform,
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(data)


def test_legacy_validation_uses_local_ground_geodesic_spacing(tmp_path: Path) -> None:
    prediction = tmp_path / "prediction.tif"
    reference = tmp_path / "reference.tif"
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    _write_high_latitude_web_mercator(prediction, values)
    _write_high_latitude_web_mercator(reference, values)

    gsd = ground_sample_distance_m(prediction)
    assert gsd is not None
    assert 4.0 < gsd[0] < 6.0
    assert 4.0 < gsd[1] < 6.0

    payload = validate_geospatial_dsm(prediction, reference, tmp_path / "evidence")
    diagnostics = payload["diagnostics"]
    assert isinstance(diagnostics, dict)
    recorded = diagnostics["ground_sample_distance_m"]
    assert isinstance(recorded, dict)
    assert recorded["semantics"] == "local_ground_geodesic_spacing"
    assert np.isclose(float(recorded["x"]), gsd[0])
    assert np.isclose(float(recorded["y"]), gsd[1])


def test_cli_mesh_uses_ground_spacing_not_projected_affine_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rdsm = tmp_path / "rdsm.tif"
    texture = tmp_path / "texture.tif"
    relative = np.arange(64, dtype=np.float32).reshape(8, 8)
    rgb = np.stack(
        [
            np.full((8, 8), 80, dtype=np.uint8),
            np.full((8, 8), 120, dtype=np.uint8),
            np.full((8, 8), 160, dtype=np.uint8),
        ]
    )
    _write_high_latitude_web_mercator(rdsm, relative)
    _write_high_latitude_web_mercator(texture, rgb, count=3)

    captured: dict[str, float] = {}

    def fake_export(*args: object, **kwargs: Any) -> list[object]:
        del args
        captured["x"] = float(kwargs["gsd_x"])
        captured["y"] = float(kwargs["gsd_y"])
        return []

    monkeypatch.setattr("depthwizard.cli.export_lod_pyramid", fake_export)
    mesh_rdsm(rdsm, texture, tmp_path / "mesh", vertical_scale=1.0)

    assert 4.0 < captured["x"] < 6.0
    assert 4.0 < captured["y"] < 6.0
    assert not np.isclose(captured["x"], 10.0)


def test_production_dem_calibration_fails_when_physical_support_is_unknown(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.tif"
    dem = tmp_path / "dem.tif"
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    transform = from_origin(0.0, 8.0, 1.0, 1.0)
    for path in (source, dem):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=8,
            width=8,
            count=1,
            dtype="float32",
            transform=transform,
        ) as dst:
            dst.write(values, 1)

    runtime = ProductionElevationRuntime.__new__(ProductionElevationRuntime)
    request = SimpleNamespace(source=source, low_frequency_sigma_px=8.0)
    geometry = SimpleNamespace(relative_height=values, confidence=None)

    try:
        runtime._dem_calibration(SimpleNamespace(), request, geometry, dem)
    except ValueError as exc:
        assert "trustworthy physical ground sample distance" in str(exc)
    else:
        raise AssertionError("metric DEM calibration must fail closed when physical support is unknown")


def test_metric_calibration_rejects_crs_tagged_but_implausible_source_grid(
    tmp_path: Path,
) -> None:
    source = tmp_path / "implausible-utm.tif"
    values = np.arange(64, dtype=np.float32).reshape(8, 8)
    # EPSG:32643 is valid, but easting/northing near zero is outside its registered area of use.
    # A syntactic CRS tag is therefore insufficient evidence for an absolute geospatial DSM.
    with rasterio.open(
        source,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(0.0, 8.0, 1.0, 1.0),
    ) as dst:
        dst.write(values, 1)

    assert ground_sample_distance_m(source) is None
    runtime = ProductionElevationRuntime.__new__(ProductionElevationRuntime)
    request = SimpleNamespace(source=source)

    try:
        runtime._calibrate(SimpleNamespace(), request, SimpleNamespace())
    except ValueError as exc:
        assert "trustworthy physical/geospatial source grid" in str(exc)
    else:
        raise AssertionError("metric calibration must reject an implausible CRS-tagged source grid")
