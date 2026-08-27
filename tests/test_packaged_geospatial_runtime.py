from __future__ import annotations

from pathlib import Path

from scripts.pyinstaller_geospatial_runtime_hook import configure_geospatial_data


def test_packaged_geospatial_runtime_hook_configures_bundle_data(tmp_path: Path) -> None:
    rasterio_gdal = tmp_path / "rasterio" / "gdal_data"
    rasterio_proj = tmp_path / "rasterio" / "proj_data"
    rasterio_gdal.mkdir(parents=True)
    rasterio_proj.mkdir(parents=True)
    environment: dict[str, str] = {}

    configured = configure_geospatial_data(tmp_path, environment)

    assert configured["GDAL_DATA"] == str(rasterio_gdal)
    assert configured["PROJ_DATA"] == str(rasterio_proj)
    assert configured["PROJ_LIB"] == str(rasterio_proj)
    assert environment["GDAL_DATA"] == str(rasterio_gdal)


def test_packaged_geospatial_runtime_hook_preserves_explicit_environment(tmp_path: Path) -> None:
    (tmp_path / "rasterio" / "gdal_data").mkdir(parents=True)
    (tmp_path / "pyproj" / "proj_dir" / "share" / "proj").mkdir(parents=True)
    environment = {
        "GDAL_DATA": "/explicit/gdal",
        "PROJ_DATA": "/explicit/proj",
        "PROJ_LIB": "/explicit/proj",
    }

    configured = configure_geospatial_data(tmp_path, environment)

    assert configured == environment
