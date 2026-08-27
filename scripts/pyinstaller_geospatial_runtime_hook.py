from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_geospatial_data(bundle_root: Path) -> dict[str, str]:
    """Point frozen GDAL/PROJ consumers at data packaged inside the PyInstaller bundle."""
    configured: dict[str, str] = {}

    gdal_data = bundle_root / "rasterio" / "gdal_data"
    if gdal_data.is_dir():
        os.environ.setdefault("GDAL_DATA", str(gdal_data))
        configured["GDAL_DATA"] = os.environ["GDAL_DATA"]

    proj_candidates = (
        bundle_root / "rasterio" / "proj_data",
        bundle_root / "pyproj" / "proj_dir" / "share" / "proj",
    )
    for proj_data in proj_candidates:
        if proj_data.is_dir():
            os.environ.setdefault("PROJ_DATA", str(proj_data))
            os.environ.setdefault("PROJ_LIB", str(proj_data))
            configured["PROJ_DATA"] = os.environ["PROJ_DATA"]
            configured["PROJ_LIB"] = os.environ["PROJ_LIB"]
            break

    return configured


def _bundle_root() -> Path | None:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if not isinstance(frozen_root, str) or not frozen_root:
        return None
    return Path(frozen_root)


root = _bundle_root()
if root is not None:
    configure_geospatial_data(root)
