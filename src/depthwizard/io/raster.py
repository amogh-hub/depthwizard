from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from pyproj.exceptions import CRSError, ProjError
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject

from depthwizard.contracts import RasterMetadata


def _crs_with_authority_metadata(crs: CRS) -> CRS:
    """Recover registry metadata lost when GDAL/Rasterio exposes an authority CRS as WKT."""
    if crs.area_of_use is not None:
        return crs
    authority = crs.to_authority()
    if authority is None:
        return crs
    try:
        return CRS.from_authority(authority[0], authority[1])
    except CRSError:
        return crs


def _longitude_within_bounds(longitude: float, west: float, east: float) -> bool:
    tolerance = 1e-7
    if west <= east:
        return west - tolerance <= longitude <= east + tolerance
    return longitude >= west - tolerance or longitude <= east + tolerance


def _trusted_wgs84_coordinate(crs: CRS, x: float, y: float) -> tuple[float, float] | None:
    """Transform one map coordinate to WGS84 and enforce the CRS area-of-use contract."""
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(x, y, errcheck=True)
    except (CRSError, ProjError):
        return None
    longitude = float(longitude)
    latitude = float(latitude)
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        return None
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        return None

    metadata_crs = _crs_with_authority_metadata(crs)
    area = metadata_crs.area_of_use
    if area is not None:
        if not _longitude_within_bounds(longitude, float(area.west), float(area.east)):
            return None
        tolerance = 1e-7
        if not float(area.south) - tolerance <= latitude <= float(area.north) + tolerance:
            return None
    return longitude, latitude


def _projected_axis_factors_m(crs: CRS) -> tuple[float, float] | None:
    if not crs.is_projected or len(crs.axis_info) < 2:
        return None
    x_factor = crs.axis_info[0].unit_conversion_factor
    y_factor = crs.axis_info[1].unit_conversion_factor
    if x_factor is None or y_factor is None:
        return None
    factors = float(x_factor), float(y_factor)
    if not all(np.isfinite(value) and value > 0 for value in factors):
        return None
    return factors


def _projected_area_bounds(crs: CRS) -> tuple[float, float, float, float] | None:
    """Project the registered CRS area-of-use envelope for map-coordinate plausibility checks."""
    metadata_crs = _crs_with_authority_metadata(crs)
    area = metadata_crs.area_of_use
    if area is None or area.west > area.east:
        return None
    try:
        transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        left, bottom, right, top = transformer.transform_bounds(
            float(area.west),
            float(area.south),
            float(area.east),
            float(area.north),
            densify_pts=21,
        )
    except (CRSError, ProjError):
        return None
    bounds = float(left), float(bottom), float(right), float(top)
    if not all(np.isfinite(value) for value in bounds):
        return None
    if bounds[0] > bounds[2] or bounds[1] > bounds[3]:
        return None
    return bounds


def _projected_coordinate_within_area(
    x: float,
    y: float,
    bounds: tuple[float, float, float, float],
) -> bool:
    left, bottom, right, top = bounds
    span = max(right - left, top - bottom, 1.0)
    tolerance = span * 1e-9
    return (
        left - tolerance <= x <= right + tolerance
        and bottom - tolerance <= y <= top + tolerance
    )


def ground_sample_distance_m(path: str | Path) -> tuple[float, float] | None:
    """Return trustworthy pixel ground spacing in metres when the raster supports it.

    Projected rasters use the affine pixel basis converted from declared CRS linear units, but only
    when representative scene coordinates are consistent with the registered CRS area of use.
    Geographic rasters use WGS84 geodesic neighbour distances. This prevents syntactically present
    but dataset-local or otherwise inconsistent CRS metadata from manufacturing absurd metric scale.

    The OrthoLoC unpacked benchmark is a special, explicit exception: its public dataset contract
    defines the DOP/DSM pixel scale in metres even though some unpacked TIFFs omit a formal CRS.
    The dedicated multiscene acceptance target opts into that interpretation with
    ``DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE=1``. No other CRS-free raster is treated as metric.
    """
    with rasterio.open(path) as src:
        if src.crs is None:
            allow_ortholoc_metric_affine = (
                os.environ.get("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE") == "1"
            )
            if not allow_ortholoc_metric_affine or src.transform.is_identity:
                return None
            gsd_x = float(np.hypot(src.transform.a, src.transform.d))
            gsd_y = float(np.hypot(src.transform.b, src.transform.e))
            if (
                not np.isfinite(gsd_x)
                or not np.isfinite(gsd_y)
                or gsd_x <= 0
                or gsd_y <= 0
            ):
                raise ValueError("CRS-free OrthoLoC affine grid has invalid metric pixel spacing")
            return gsd_x, gsd_y
        if src.transform.is_identity:
            return None

        crs = CRS.from_user_input(src.crs)
        metadata_crs = _crs_with_authority_metadata(crs)
        transform = src.transform
        representative_pixels = {
            (0, 0),
            (max(src.width - 1, 0), 0),
            (0, max(src.height - 1, 0)),
            (max(src.width - 1, 0), max(src.height - 1, 0)),
            (max((src.width - 1) // 2, 0), max((src.height - 1) // 2, 0)),
        }

        if crs.is_projected:
            factors = _projected_axis_factors_m(crs)
            if factors is None:
                return None
            area_bounds = _projected_area_bounds(crs)
            if metadata_crs.area_of_use is not None:
                if area_bounds is None:
                    return None
                for col, row in representative_pixels:
                    x, y = src.xy(row, col)
                    if not _projected_coordinate_within_area(float(x), float(y), area_bounds):
                        return None
            x_factor, y_factor = factors
            gsd_x = float(np.hypot(transform.a * x_factor, transform.d * y_factor))
            gsd_y = float(np.hypot(transform.b * x_factor, transform.e * y_factor))
            if not np.isfinite(gsd_x) or not np.isfinite(gsd_y) or gsd_x <= 0 or gsd_y <= 0:
                return None
            return gsd_x, gsd_y

        if not crs.is_geographic:
            return None

        col = (src.width - 1) / 2.0
        row = (src.height - 1) / 2.0
        x0, y0 = transform * (col + 0.5, row + 0.5)
        x1, y1 = transform * (col + 1.5, row + 0.5)
        x2, y2 = transform * (col + 0.5, row + 1.5)
        point0 = _trusted_wgs84_coordinate(crs, float(x0), float(y0))
        point1 = _trusted_wgs84_coordinate(crs, float(x1), float(y1))
        point2 = _trusted_wgs84_coordinate(crs, float(x2), float(y2))
        if point0 is None or point1 is None or point2 is None:
            return None

    geod = Geod(ellps="WGS84")
    _, _, gsd_x = geod.inv(point0[0], point0[1], point1[0], point1[1])
    _, _, gsd_y = geod.inv(point0[0], point0[1], point2[0], point2[1])
    if not np.isfinite(gsd_x) or not np.isfinite(gsd_y) or gsd_x <= 0 or gsd_y <= 0:
        return None
    return float(abs(gsd_x)), float(abs(gsd_y))


def inspect_raster(path: str | Path) -> RasterMetadata:
    p = Path(path)
    metric_gsd = ground_sample_distance_m(p)
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
        gsd_x = metric_gsd[0] if metric_gsd is not None else None
        gsd_y = metric_gsd[1] if metric_gsd is not None else None
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


def _direct_read_exact_grid(
    src: DatasetReader,
    dst_ref: DatasetReader,
    *,
    source_band: int,
    dst_nodata: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read directly only when two rasters are provably on the exact same pixel grid.

    Some authoritative paired datasets intentionally omit CRS metadata while preserving identical
    DOP/DSM affine grids. Reprojection is undefined without a CRS, but no reprojection is needed
    when width, height and affine transform are identical. This path is deliberately strict: a
    missing CRS never permits shape-only alignment or transform mismatch.
    """
    same_shape = src.width == dst_ref.width and src.height == dst_ref.height
    same_transform = src.transform.almost_equals(dst_ref.transform)
    if not (same_shape and same_transform):
        return None

    source = src.read(source_band).astype(np.float32)
    valid = src.read_masks(source_band) > 0
    valid &= np.isfinite(source)
    if src.nodata is not None and np.isfinite(src.nodata):
        valid &= source != np.float32(src.nodata)
    destination = np.where(valid, source, np.float32(dst_nodata)).astype(np.float32)
    return destination, valid


def reproject_to_match(
    source_path: str | Path,
    target_path: str | Path,
    *,
    source_band: int = 1,
    resampling: Resampling = Resampling.bilinear,
    dst_nodata: float = np.nan,
) -> tuple[np.ndarray, np.ndarray]:
    """Align one source band exactly onto the target raster grid.

    Normal geospatial inputs are reprojected using their CRS. If CRS metadata is absent on both
    rasters, direct reading is allowed only when the two rasters already have identical dimensions
    and affine transforms. Missing CRS plus any grid mismatch is rejected rather than guessed.

    Returns ``(data, valid_mask)``.
    """
    with rasterio.open(source_path) as src, rasterio.open(target_path) as dst_ref:
        if src.crs is None or dst_ref.crs is None:
            if src.crs is not None or dst_ref.crs is not None:
                raise ValueError(
                    "source/target CRS mismatch: one raster has CRS metadata and the other does not"
                )
            direct = _direct_read_exact_grid(
                src,
                dst_ref,
                source_band=source_band,
                dst_nodata=dst_nodata,
            )
            if direct is None:
                raise ValueError(
                    "CRS-free rasters can only be aligned when dimensions and affine transforms "
                    "match exactly"
                )
            return direct

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
            profile.update(
                tiled=True,
                blockxsize=min(512, (template.width // 16) * 16),
                blockysize=min(512, (template.height // 16) * 16),
            )
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
