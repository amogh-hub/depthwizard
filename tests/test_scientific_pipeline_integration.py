from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem
from depthwizard.evaluation.report import validate_geospatial_dsm
from depthwizard.io.raster import reproject_to_match, write_float_geotiff
from depthwizard.mesh.terrain import export_terrain_glb


def _write(path: Path, data: np.ndarray, *, gsd: float, crs: str = "EPSG:32643") -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(500000, 1400000, gsd, gsd),
        nodata=-9999.0,
    ) as dst:
        dst.write(data.astype(np.float32), 1)


def test_geospatial_calibrate_validate_mesh_pipeline(tmp_path: Path) -> None:
    y, x = np.mgrid[:32, :32]
    relative = (x / 31.0 + y / 62.0).astype(np.float32)
    truth = (180.0 + 12.0 * relative).astype(np.float32)

    relative_path = tmp_path / "relative.tif"
    coarse_dem_path = tmp_path / "coarse_dem.tif"
    reference_path = tmp_path / "reference.tif"
    dsm_path = tmp_path / "dsm.tif"

    _write(relative_path, relative, gsd=1.0)
    _write(reference_path, truth, gsd=1.0)
    # A lower-resolution metric anchor representing SRTM-like support for the integration path.
    coarse = truth.reshape(8, 4, 8, 4).mean(axis=(1, 3))
    _write(coarse_dem_path, coarse, gsd=4.0)

    dem, dem_valid = reproject_to_match(coarse_dem_path, relative_path)
    calibrated = calibrate_relative_height_with_dem(
        relative,
        dem,
        dem_valid=dem_valid,
        low_frequency_sigma_px=4,
        min_anchors=16,
    )
    write_float_geotiff(dsm_path, calibrated.dsm, template_path=relative_path)

    evidence = validate_geospatial_dsm(dsm_path, reference_path, tmp_path / "evidence")
    assert evidence["official"]["rmse_m"] < 0.3
    assert (tmp_path / "evidence" / "residual.tif").exists()
    assert (tmp_path / "evidence" / "metrics.json").exists()

    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    rgb[..., 0] = 85
    rgb[..., 1] = 125
    rgb[..., 2] = 90
    mesh = export_terrain_glb(tmp_path / "terrain.glb", calibrated.dsm, rgb, stride=2)
    assert mesh.path.exists()
    assert mesh.vertices > 0 and mesh.faces > 0
