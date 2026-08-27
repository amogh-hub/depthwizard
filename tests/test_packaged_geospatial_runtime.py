from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts.pyinstaller_geospatial_runtime_hook import configure_geospatial_data


def test_packaged_geospatial_runtime_hook_configures_bundle_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rasterio_gdal = tmp_path / "rasterio" / "gdal_data"
    rasterio_proj = tmp_path / "rasterio" / "proj_data"
    rasterio_gdal.mkdir(parents=True)
    rasterio_proj.mkdir(parents=True)

    for key in ("GDAL_DATA", "PROJ_DATA", "PROJ_LIB"):
        monkeypatch.delenv(key, raising=False)

    configured = configure_geospatial_data(tmp_path)

    assert configured["GDAL_DATA"] == str(rasterio_gdal)
    assert configured["PROJ_DATA"] == str(rasterio_proj)
    assert configured["PROJ_LIB"] == str(rasterio_proj)
    assert os.environ["GDAL_DATA"] == str(rasterio_gdal)


def test_packaged_geospatial_runtime_hook_preserves_explicit_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "rasterio" / "gdal_data").mkdir(parents=True)
    (tmp_path / "pyproj" / "proj_dir" / "share" / "proj").mkdir(parents=True)
    monkeypatch.setenv("GDAL_DATA", "/explicit/gdal")
    monkeypatch.setenv("PROJ_DATA", "/explicit/proj")
    monkeypatch.setenv("PROJ_LIB", "/explicit/proj")

    configured = configure_geospatial_data(tmp_path)

    assert configured == {
        "GDAL_DATA": "/explicit/gdal",
        "PROJ_DATA": "/explicit/proj",
        "PROJ_LIB": "/explicit/proj",
    }
