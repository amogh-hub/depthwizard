from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer
from pyproj.exceptions import CRSError, ProjError
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.warp import reproject

from depthwizard.contracts import RasterMetadata, RasterQualityAssessment

ORTHOLOC_METRIC_AFFINE_ENV = "DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE"

_OFF_NADIR_TAGS = (
    "OFF_NADIR",
    "OFF_NADIR_ANGLE",
    "VIEW_ANGLE",
    "VIEW_ZENITH",
    "VIEW_ZENITH_ANGLE",
)


def _tagged_off_nadir_degrees(src: DatasetReader) -> float | None:
    tags = {str(key).upper(): str(value).strip() for key, value in src.tags().items()}
    for name in _OFF_NADIR_TAGS:
        raw = tags.get(name)
        if not raw:
            continue
        try:
            angle = abs(float(raw.removesuffix("°").strip()))
        except ValueError:
            continue
        if np.isfinite(angle) and angle <= 90.0:
            return angle
    return None


def _sampled_raster_quality(src: DatasetReader) -> RasterQualityAssessment:
    """Return conservative diagnostics from a bounded RGB overview.

    Bright/low-chroma and dark fractions are candidates, not semantic cloud/shadow masks. Their
    ambiguity is recorded explicitly. Off-nadir risk is reported only from source metadata; view
    angle is never guessed from image appearance.
    """
    if src.count < 3:
        return RasterQualityAssessment(
            status="not_assessed",
            flags=["rgb_quality_not_assessed_fewer_than_three_bands"],
            assessment_limitations=["RGB radiometric diagnostics require at least three bands"],
        )

    scale = min(1.0, 1024.0 / max(src.width, src.height))
    out_h = max(1, round(src.height * scale))
    out_w = max(1, round(src.width * scale))
    sampled = src.read(
        [1, 2, 3],
        out_shape=(3, out_h, out_w),
        masked=True,
        resampling=Resampling.average,
    ).astype(np.float64)
    values = np.asarray(sampled.filled(np.nan), dtype=np.float64)
    valid = ~np.any(np.ma.getmaskarray(sampled), axis=0)
    valid &= np.all(np.isfinite(values), axis=0)
    if not np.any(valid):
        return RasterQualityAssessment(
            status="warning",
            flags=["no_valid_rgb_pixels_for_quality_assessment"],
            assessment_limitations=["No valid sampled RGB support was available"],
        )

    dtype = np.dtype(src.dtypes[0])
    saturation_fraction: float | None = None
    if np.issubdtype(dtype, np.integer):
        limits = np.iinfo(dtype)
        saturated = np.any((values <= limits.min) | (values >= limits.max), axis=0)
        saturation_fraction = float(np.mean(saturated[valid]))

    # Use one robust range for all RGB channels. Normalizing each band independently would erase
    # real channel differences and make the bright/low-chroma diagnostic overconfident.
    radiometric_values = values[:, valid]
    low, high = np.percentile(radiometric_values, (2.0, 98.0))
    usable_radiometry = bool(
        np.isfinite(low) and np.isfinite(high) and high - low > 1e-12
    )
    normalized = (
        np.clip((values - low) / (high - low), 0.0, 1.0)
        if usable_radiometry
        else np.zeros_like(values, dtype=np.float64)
    )

    flags: list[str] = []
    limitations = [
        "Cloud and shadow fractions are radiometric candidates, not semantic masks",
        "Building lean cannot be inferred reliably without sensor geometry or stereo evidence",
    ]
    if saturation_fraction is not None and saturation_fraction >= 0.20:
        flags.append("high_saturation_fraction")

    shadow_fraction: float | None = None
    bright_fraction: float | None = None
    texture_score: float | None = None
    if usable_radiometry:
        brightness = np.mean(normalized, axis=0)
        chroma = np.max(normalized, axis=0) - np.min(normalized, axis=0)
        shadow_fraction = float(np.mean(brightness[valid] <= 0.06))
        bright_fraction = float(np.mean((brightness[valid] >= 0.94) & (chroma[valid] <= 0.08)))
        grayscale = np.where(valid, brightness, np.nan)
        horizontal = np.abs(np.diff(grayscale, axis=1))
        vertical = np.abs(np.diff(grayscale, axis=0))
        gradients = np.concatenate(
            [horizontal[np.isfinite(horizontal)], vertical[np.isfinite(vertical)]]
        )
        texture_score = float(np.median(gradients)) if gradients.size else 0.0
        if shadow_fraction >= 0.35:
            flags.append("large_deep_shadow_candidate_fraction")
        if bright_fraction >= 0.35:
            flags.append("large_bright_low_chroma_candidate_fraction")
        if texture_score <= 0.002:
            flags.append("insufficient_texture_risk")
    else:
        flags.append("insufficient_dynamic_range_for_radiometric_quality_assessment")

    off_nadir = _tagged_off_nadir_degrees(src)
    if off_nadir is not None and off_nadir >= 20.0:
        flags.append("off_nadir_building_lean_risk")

    return RasterQualityAssessment(
        status="warning" if flags else "pass",
        flags=flags,
        saturation_fraction=saturation_fraction,
        deep_shadow_candidate_fraction=shadow_fraction,
        bright_low_chroma_candidate_fraction=bright_fraction,
        texture_gradient_score=texture_score,
        off_nadir_degrees=off_nadir,
        assessment_limitations=limitations,
    )


def ortholoc_metric_affine_override_enabled() -> bool:
    """Return whether the dedicated OrthoLoC local-metric affine contract is enabled.

    OrthoLoC publishes DOP/DSM pixel scale in metres. Some raw/unpacked TIFF representations carry
    affine coordinates that are dataset-local even when a syntactic CRS tag is present. This opt-in
    is intentionally dataset-specific and must never become a generic escape hatch for arbitrary
    geospatial rasters.
    """
    return os.environ.get(ORTHOLOC_METRIC_AFFINE_ENV) == "1"


def _affine_metric_spacing(transform) -> tuple[float, float] | None:
    gsd_x = float(np.hypot(transform.a, transform.d))
    gsd_y = float(np.hypot(transform.b, transform.e))
    if not np.isfinite(gsd_x) or not np.isfinite(gsd_y) or gsd_x <= 0 or gsd_y <= 0:
        return None
    return gsd_x, gsd_y


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
    """Return declared map-axis unit conversion factors.

    This helper is retained for metadata diagnostics only. It must not be used as a substitute for
    true local ground distance because projected metres can include projection scale distortion.
    """
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
    return left - tolerance <= x <= right + tolerance and bottom - tolerance <= y <= top + tolerance


def ground_sample_distance_m(path: str | Path) -> tuple[float, float] | None:
    """Return trustworthy local ground pixel spacing in metres.

    The returned values are *ground* distances near the raster centre, not merely coordinate-axis
    increments. For ordinary projected and geographic CRSs, adjacent pixel centres are transformed
    to WGS84 and measured geodesically. This is essential for distorted map projections such as
    EPSG:3857, where one projected map metre is not one ground metre away from the equator.

    The dedicated OrthoLoC acceptance/benchmark opt-in is evaluated before CRS interpretation. Its
    published dataset-local affine scale is explicitly metric and deliberately carries no global
    longitude/latitude claim. No arbitrary raster receives that treatment by default.
    """
    with rasterio.open(path) as src:
        if src.transform.is_identity:
            return None

        if ortholoc_metric_affine_override_enabled():
            metric_affine = _affine_metric_spacing(src.transform)
            if metric_affine is None:
                raise ValueError("OrthoLoC affine grid has invalid metric pixel spacing")
            return metric_affine

        if src.crs is None:
            return None

        crs = CRS.from_user_input(src.crs)
        if not (crs.is_projected or crs.is_geographic):
            return None

        transform = src.transform
        representative_pixels = {
            (0, 0),
            (max(src.width - 1, 0), 0),
            (0, max(src.height - 1, 0)),
            (max(src.width - 1, 0), max(src.height - 1, 0)),
            (max((src.width - 1) // 2, 0), max((src.height - 1) // 2, 0)),
        }
        # Reject syntactically valid but geographically implausible CRS-tagged rasters before any
        # metric claim. This preserves the fail-closed behavior used by dataset-local fixtures.
        for col_i, row_i in representative_pixels:
            x_i, y_i = src.xy(row_i, col_i)
            if _trusted_wgs84_coordinate(crs, float(x_i), float(y_i)) is None:
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


def ground_pixel_jacobian_m(path: str | Path) -> np.ndarray | None:
    """Return local east/north ground vectors for one pixel-column and pixel-row step.

    The 2x2 matrix maps ``[delta_col, delta_row]`` to local ``[east_m, north_m]``. Unlike two
    scalar GSD values, it preserves raster rotation, skew, and non-orthogonal axes for slope
    analysis. Geographic/projected grids are evaluated geodesically at the scene centre.
    """
    trusted_spacing = ground_sample_distance_m(path)
    if trusted_spacing is None:
        return None

    with rasterio.open(path) as src:
        if ortholoc_metric_affine_override_enabled():
            jacobian = np.array(
                [[src.transform.a, src.transform.b], [src.transform.d, src.transform.e]],
                dtype=np.float64,
            )
        else:
            if src.crs is None:
                return None
            crs = CRS.from_user_input(src.crs)
            col = (src.width - 1) / 2.0
            row = (src.height - 1) / 2.0
            x0, y0 = src.transform * (col + 0.5, row + 0.5)
            x1, y1 = src.transform * (col + 1.5, row + 0.5)
            x2, y2 = src.transform * (col + 0.5, row + 1.5)
            point0 = _trusted_wgs84_coordinate(crs, float(x0), float(y0))
            point1 = _trusted_wgs84_coordinate(crs, float(x1), float(y1))
            point2 = _trusted_wgs84_coordinate(crs, float(x2), float(y2))
            if point0 is None or point1 is None or point2 is None:
                return None
            geod = Geod(ellps="WGS84")
            azimuth_col, _, distance_col = geod.inv(point0[0], point0[1], point1[0], point1[1])
            azimuth_row, _, distance_row = geod.inv(point0[0], point0[1], point2[0], point2[1])

            def components(azimuth_degrees: float, distance_m: float) -> tuple[float, float]:
                radians = np.deg2rad(azimuth_degrees)
                return float(np.sin(radians) * distance_m), float(np.cos(radians) * distance_m)

            col_east, col_north = components(float(azimuth_col), float(distance_col))
            row_east, row_north = components(float(azimuth_row), float(distance_row))
            jacobian = np.array(
                [[col_east, row_east], [col_north, row_north]],
                dtype=np.float64,
            )

    determinant = float(np.linalg.det(jacobian))
    scale = max(float(np.linalg.norm(jacobian, ord=2)), 1e-12)
    if not np.all(np.isfinite(jacobian)) or abs(determinant) <= 1e-10 * scale * scale:
        return None
    return jacobian


def _vertical_metadata(
    src: DatasetReader,
) -> tuple[
    str | None,
    str | None,
    Literal["orthometric", "ellipsoidal", "local", "unknown"],
]:
    """Extract explicit vertical-reference metadata without guessing from horizontal CRS."""
    tags = {str(key).upper(): str(value).strip() for key, value in src.tags().items()}
    placeholders = {"unknown", "unspecified", "none", "null", "n/a", "na", "tbd"}

    def explicit_tag(*names: str) -> str | None:
        value = next((tags[name] for name in names if tags.get(name)), None)
        if value is None or value.casefold() in placeholders:
            return None
        return value

    vertical_crs = explicit_tag("VERTICAL_CRS", "VERTICAL_CRS_WKT")
    vertical_datum = explicit_tag("VERTICAL_DATUM", "VERT_DATUM")
    reference_raw = tags.get("ELEVATION_REFERENCE", "").strip().lower()
    elevation_reference: Literal["orthometric", "ellipsoidal", "local", "unknown"]
    if reference_raw == "orthometric":
        elevation_reference = "orthometric"
    elif reference_raw == "ellipsoidal":
        elevation_reference = "ellipsoidal"
    elif reference_raw == "local":
        elevation_reference = "local"
    else:
        elevation_reference = "unknown"

    if src.crs is not None:
        try:
            parsed = CRS.from_user_input(src.crs)
            candidates = [parsed] if parsed.is_vertical else list(parsed.sub_crs_list)
            vertical = next((candidate for candidate in candidates if candidate.is_vertical), None)
            if vertical is not None:
                vertical_crs = vertical.to_string()
                if vertical_datum is None and vertical.datum is not None:
                    vertical_datum = vertical.datum.name
                descriptor = f"{vertical.name} {vertical_datum or ''}".lower()
                if elevation_reference == "unknown":
                    if "ellipsoid" in descriptor:
                        elevation_reference = "ellipsoidal"
                    elif any(word in descriptor for word in ("orthometric", "geoid", "gravity")):
                        elevation_reference = "orthometric"
        except (CRSError, AttributeError):
            pass
    return vertical_crs, vertical_datum, elevation_reference


def inspect_raster(path: str | Path) -> RasterMetadata:
    p = Path(path)
    metric_gsd = ground_sample_distance_m(p)
    with rasterio.open(p) as src:
        transform = src.transform
        has_meaningful_transform = not transform.is_identity
        transform_tuple = (
            (
                transform.a,
                transform.b,
                transform.c,
                transform.d,
                transform.e,
                transform.f,
            )
            if has_meaningful_transform
            else None
        )
        crs = src.crs.to_string() if src.crs is not None else None
        gsd_x = metric_gsd[0] if metric_gsd is not None else None
        gsd_y = metric_gsd[1] if metric_gsd is not None else None
        vertical_crs, vertical_datum, elevation_reference = _vertical_metadata(src)
        scale = min(1.0, 2048.0 / max(src.width, src.height))
        out_h = max(1, round(src.height * scale))
        out_w = max(1, round(src.width * scale))
        mask_bands = min(3, src.count)
        sampled_masks = src.read_masks(
            list(range(1, mask_bands + 1)),
            out_shape=(mask_bands, out_h, out_w),
            resampling=Resampling.nearest,
        )
        valid_data_fraction = float(np.mean(np.all(sampled_masks > 0, axis=0)))
        quality = _sampled_raster_quality(src)
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
            valid_data_fraction=valid_data_fraction,
            vertical_crs=vertical_crs,
            vertical_datum=vertical_datum,
            elevation_reference=elevation_reference,
            quality=quality,
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
