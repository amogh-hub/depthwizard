from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.release_train_2_validation_smoke import (
    CALIBRATION_DOWNSAMPLE_FACTOR,
    _build_coarse_calibration_dem,
)


def _write_calibration_source(path: Path) -> None:
    data = np.linspace(100.0, 180.0, 128 * 128, dtype=np.float32).reshape(128, 128)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=128,
        width=128,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000.0, 5400000.0, 0.25, 0.25),
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)


def test_coarse_calibration_surrogate_preserves_extent_and_claim_boundary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source_xdsm.tif"
    _write_calibration_source(source)
    monkeypatch.setattr(
        "scripts.release_train_2_validation_smoke.DATA_DIR",
        tmp_path,
    )

    output = _build_coarse_calibration_dem(source)

    with rasterio.open(source) as src, rasterio.open(output) as coarse:
        assert coarse.width == 128 // CALIBRATION_DOWNSAMPLE_FACTOR
        assert coarse.height == 128 // CALIBRATION_DOWNSAMPLE_FACTOR
        assert coarse.crs == src.crs
        assert np.allclose(coarse.bounds, src.bounds)
        tags = coarse.tags()
        assert tags["DEPTHWIZARD_ROLE"] == "CALIBRATION_ONLY_INTEGRATION_SURROGATE"
        assert tags["CALIBRATION_LINEAGE_INDEPENDENT"] == "false"
        assert tags["DOWNSAMPLE_FACTOR"] == str(CALIBRATION_DOWNSAMPLE_FACTOR)
