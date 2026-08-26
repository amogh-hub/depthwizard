from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


def write_unreferenced_float_tiff(
    path: str | Path,
    array: np.ndarray,
    *,
    nodata: float = -9999.0,
    description: str,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a float32 TIFF while intentionally omitting CRS/geotransform metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("unreferenced float product must be a 2D raster")
    profile: dict[str, object] = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "nodata": nodata,
        "compress": "deflate",
        "predictor": 3,
        "BIGTIFF": "IF_SAFER",
    }
    if arr.shape[0] >= 16 and arr.shape[1] >= 16:
        profile.update(
            tiled=True,
            blockxsize=min(512, (arr.shape[1] // 16) * 16),
            blockysize=min(512, (arr.shape[0] // 16) * 16),
        )
    encoded = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
    with rasterio.open(output, "w", **profile) as dst:
        dst.write(encoded, 1)
        dst.set_band_description(1, description)
        dst.update_tags(CRS_STATUS="none_intentionally", **(tags or {}))
    return output
