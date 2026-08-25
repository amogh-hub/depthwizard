from pathlib import Path

import numpy as np
import rasterio

from depthwizard.io.raster import write_relative_tiff


def test_relative_export_does_not_invent_crs(tmp_path: Path) -> None:
    output = tmp_path / "rdsm.tif"
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    write_relative_tiff(output, data)
    with rasterio.open(output) as src:
        assert src.crs is None
        assert src.tags()["ELEVATION_UNITS"] == "relative"
        assert src.tags()["CRS_STATUS"] == "none_intentionally"
