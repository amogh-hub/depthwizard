from __future__ import annotations

import os
import sys
from collections.abc import MutableMapping
from pathlib import Path


def configure_geospatial_data(
    bundle_root: Path,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Point frozen GDAL/PROJ consumers at data packaged inside the PyInstaller bundle."""
    target = os.environ if environ is None else environ
    configured: dict[str, str] = {}

    gdal_data = bundle_root / "rasterio" / "gdal_data"
    if gdal_data.is_dir():
        target.setdefault("GDAL_DATA", str(gdal_data))
        configured["GDAL_DATA"] = target["GDAL_DATA"]

    proj_candidates = (
        bundle_root / "rasterio" / "proj_data",
        bundle_root / "pyproj" / "proj_dir" / "share" / "proj",
    )
    for proj_data in proj_candidates:
        if proj_data.is_dir():
            target.setdefault("PROJ_DATA", str(proj_data))
            target.setdefault("PROJ_LIB", str(proj_data))
            configured["PROJ_DATA"] = target["PROJ_DATA"]
            configured["PROJ_LIB"] = target["PROJ_LIB"]
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
