from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from depthwizard.contracts import RasterMetadata


def inspect_raster(path: str | Path) -> RasterMetadata:
    p = Path(path)
    with rasterio.open(p) as src:
        transform = src.transform
        has_meaningful_transform = not transform.is_identity
        transform_tuple = (
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ) if has_meaningful_transform else None
        crs = src.crs.to_string() if src.crs is not None else None
        gsd_x = abs(transform.a) if has_meaningful_transform else None
        gsd_y = abs(transform.e) if has_meaningful_transform else None
        return RasterMetadata(
            path=p,
            width=src.width,
            height=src.height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=crs,
            transform=transform_tuple,
            nodata=src.nodata,
            ground_sample_distance_x=gsd_x,
            ground_sample_distance_y=gsd_y,
        )


def read_rgb(path: str | Path, *, band_indices: tuple[int, int, int] = (1, 2, 3)) -> np.ndarray:
    """Read RGB into HxWx3 without silently reordering bands.

    Remote-sensing products can have non-RGB band orders. The caller must supply explicit
    1-based band indices when the first three bands are not RGB.
    """
    with rasterio.open(path) as src:
        if src.count < 3:
            raise ValueError(f"RGB input requires at least 3 bands; got {src.count}")
        if any(index < 1 or index > src.count for index in band_indices):
            raise ValueError("band_indices reference bands outside the source raster")
        rgb = src.read(list(band_indices))
    return np.moveaxis(rgb, 0, -1)


def read_single_band(path: str | Path, band: int = 1) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        profile = src.profile.copy()
    return arr, profile


def reproject_to_match(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproject one source band exactly onto the target raster grid.

    Returns (data, valid_mask). This is the canonical path for SRTM/reference alignment,
    avoiding silent shape-only comparisons.
    """
    with rasterio.open(source_path) as src, rasterio.open(target_path) as dst_ref:
        if src.crs is None or dst_ref.crs is None:
            raise ValueError("both source and target require CRS for reprojection")
        destination = np.full((dst_ref.height, dst_ref.width), dst_nodata, dtype=np.float32)
        source = src.read(source_band).astype(np.float32)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_ref.transform,
            dst_crs=dst_ref.crs,
            dst_nodata=dst_nodata,
            resampling=resampling,
        )
        valid = np.isfinite(destination)
        if np.isfinite(dst_nodata):
            valid &= destination != dst_nodata
        return destination, valid


def write_float_geotiff(
    path: str | Path,
    array: np.ndarray,
    *,
    template_path: str | Path,
    nodata: float = -9999.0,
    description: str | None = None,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a float32 geospatial raster preserving the target grid exactly."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    with rasterio.open(template_path) as template:
        if arr.shape != (template.height, template.width):
            raise ValueError(
                f"array shape {arr.shape} does not match template grid "
                f"{(template.height, template.width)}"
            )
        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            dtype="float32",
            count=1,
            nodata=nodata,
            compress="deflate",
            predictor=3,
            BIGTIFF="IF_SAFER",
        )
        if template.width >= 16 and template.height >= 16:
            profile.update(tiled=True, blockxsize=min(512, (template.width // 16) * 16), blockysize=min(512, (template.height // 16) * 16))
        else:
            profile.pop("tiled", None)
            profile.pop("blockxsize", None)
            profile.pop("blockysize", None)
        encoded = np.where(np.isfinite(arr), arr, nodata).astype(np.float32)
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(encoded, 1)
            if description:
                dst.set_band_description(1, description)
            if tags:
                dst.update_tags(**tags)
    return output


def write_relative_tiff(
    path: str | Path,
    array: np.ndarray,
    *,
    nodata: float = -9999.0,
    description: str = "DepthWizard relative Digital Surface Model (dimensionless)",
) -> Path:
    """Write an rDSM without inventing CRS/geotransform metadata."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("relative DSM must be a 2D raster")
    profile = {
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
        dst.update_tags(
            DEPTHWIZARD_PRODUCT="RELATIVE_DSM_DIMENSIONLESS",
            ELEVATION_UNITS="relative",
            CRS_STATUS="none_intentionally",
        )
    return output
