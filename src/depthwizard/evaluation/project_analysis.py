from __future__ import annotations

from itertools import pairwise
from math import ceil, floor
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pyproj import Geod, Transformer
from pyproj.exceptions import CRSError, ProjError
from rasterio.windows import Window

from depthwizard.contracts import (
    NormalizedPoint,
    ProfileSample,
    ProjectProbeRequest,
    ProjectProbeResult,
    ProjectProfileRequest,
    ProjectProfileResult,
    RasterSample,
)
from depthwizard.io.raster import (
    ground_sample_distance_m,
    ortholoc_metric_affine_override_enabled,
)
from depthwizard.pipeline.project import ProjectManifest

_GEOD = Geod(ellps="WGS84")
SurfaceProduct = Literal["dsm", "rdsm"]


def _load_project(project_dir: Path) -> ProjectManifest:
    if not (project_dir / "project-manifest.json").is_file():
        raise FileNotFoundError("DepthWizard project manifest does not exist")
    return ProjectManifest.load(project_dir)


def _primary_surface(manifest: ProjectManifest) -> tuple[SurfaceProduct, Path]:
    for name in ("dsm", "rdsm"):
        path = manifest.artifact_path(name)
        if path is not None and path.is_file():
            return name, path
    raise ValueError("analytical sampling requires a completed DSM or rDSM artifact")


def _floating_pixel(point: NormalizedPoint, *, width: int, height: int) -> tuple[float, float]:
    col = float(point.x * max(width - 1, 0))
    row = float(point.y * max(height - 1, 0))
    return col, row


def _pixel_index(point: NormalizedPoint, *, width: int, height: int) -> tuple[int, int]:
    col_f, row_f = _floating_pixel(point, width=width, height=height)
    col = round(col_f)
    row = round(row_f)
    return min(max(col, 0), width - 1), min(max(row, 0), height - 1)


def _read_cell(path: Path, point: NormalizedPoint) -> float | None:
    with rasterio.open(path) as src:
        col, row = _pixel_index(point, width=src.width, height=src.height)
        window = ((row, row + 1), (col, col + 1))
        value = src.read(1, window=window, masked=True)
        if value.size != 1 or bool(np.ma.getmaskarray(value)[0, 0]):
            return None
        scalar = float(value[0, 0])
        if not np.isfinite(scalar):
            return None
        if src.nodata is not None and np.isfinite(src.nodata) and scalar == float(src.nodata):
            return None
        return scalar


def _artifact_sample(manifest: ProjectManifest, name: str, point: NormalizedPoint) -> RasterSample:
    payload = manifest.artifacts.get(name)
    if not payload:
        return RasterSample(available=False, semantics=f"unavailable_{name}_artifact")
    raw_path = payload.get("path")
    if not isinstance(raw_path, str):
        raise TypeError(f"{name} artifact path is malformed in project manifest")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"persisted {name} artifact does not exist")
    semantics = str(payload.get("semantics") or name)
    units = payload.get("units")
    value = _read_cell(path, point)
    if value is None:
        return RasterSample(
            available=False,
            units=str(units) if units is not None else None,
            semantics=f"{semantics}; nodata_at_point",
        )
    return RasterSample(
        available=True,
        value=value,
        units=str(units) if units is not None else None,
        semantics=semantics,
    )


def _bilinear_series(path: Path, points: list[NormalizedPoint]) -> list[float | None]:
    """Sample one raster along a normalized transect with nodata-aware bilinear interpolation.

    Only the bounding window around the requested samples is read, avoiding hundreds of independent
    file opens while preserving subpixel profile geometry.
    """
    with rasterio.open(path) as src:
        cols = np.asarray([point.x * max(src.width - 1, 0) for point in points], dtype=np.float64)
        rows = np.asarray([point.y * max(src.height - 1, 0) for point in points], dtype=np.float64)
        col0 = max(0, floor(float(np.min(cols))) - 1)
        row0 = max(0, floor(float(np.min(rows))) - 1)
        col1 = min(src.width, ceil(float(np.max(cols))) + 2)
        row1 = min(src.height, ceil(float(np.max(rows))) + 2)
        window = Window.from_slices((row0, row1), (col0, col1))
        sample = src.read(1, window=window, masked=True).astype(np.float64)
        values = np.asarray(sample.filled(np.nan), dtype=np.float64)
        valid = ~np.ma.getmaskarray(sample) & np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != float(src.nodata)

    local_cols = cols - col0
    local_rows = rows - row0
    c0 = np.floor(local_cols).astype(np.int64)
    r0 = np.floor(local_rows).astype(np.int64)
    c1 = np.minimum(c0 + 1, values.shape[1] - 1)
    r1 = np.minimum(r0 + 1, values.shape[0] - 1)
    fx = local_cols - c0
    fy = local_rows - r0

    weights = (
        (1.0 - fx) * (1.0 - fy),
        fx * (1.0 - fy),
        (1.0 - fx) * fy,
        fx * fy,
    )
    corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    numerator = np.zeros(len(points), dtype=np.float64)
    denominator = np.zeros(len(points), dtype=np.float64)
    for weight, (rr, cc) in zip(weights, corners, strict=True):
        corner_valid = valid[rr, cc]
        effective = np.where(corner_valid, weight, 0.0)
        numerator += np.where(corner_valid, values[rr, cc], 0.0) * effective
        denominator += effective

    result: list[float | None] = []
    for value, support in zip(numerator, denominator, strict=True):
        if support <= 1e-8 or not np.isfinite(value):
            result.append(None)
        else:
            result.append(float(value / support))
    return result


def _artifact_series(
    manifest: ProjectManifest,
    name: str,
    points: list[NormalizedPoint],
) -> list[RasterSample]:
    payload = manifest.artifacts.get(name)
    if not payload:
        return [RasterSample(available=False, semantics=f"unavailable_{name}_artifact") for _ in points]
    raw_path = payload.get("path")
    if not isinstance(raw_path, str):
        raise TypeError(f"{name} artifact path is malformed in project manifest")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"persisted {name} artifact does not exist")
    semantics = str(payload.get("semantics") or name)
    units = str(payload.get("units")) if payload.get("units") is not None else None
    values = _bilinear_series(path, points)
    return [
        RasterSample(
            available=value is not None,
            value=value,
            units=units,
            semantics=semantics if value is not None else f"{semantics}; nodata_at_point",
        )
        for value in values
    ]


def _safe_geographic_coordinates(
    crs: object,
    x: float,
    y: float,
) -> tuple[float | None, float | None]:
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(x, y, errcheck=True)
    except (CRSError, ProjError):
        return None, None
    if not np.isfinite(longitude) or not np.isfinite(latitude):
        return None, None
    if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
        return None, None
    return float(longitude), float(latitude)


def _spatial_coordinates(
    path: Path,
    point: NormalizedPoint,
) -> tuple[int, int, float | None, float | None, float | None, float | None]:
    metric_gsd = ground_sample_distance_m(path)
    with rasterio.open(path) as src:
        col, row = _pixel_index(point, width=src.width, height=src.height)
        if src.transform.is_identity:
            return col, row, None, None, None, None
        x, y = src.xy(row, col)
        if ortholoc_metric_affine_override_enabled():
            return col, row, float(x), float(y), None, None
        if src.crs is None:
            return col, row, None, None, None, None
        if metric_gsd is None:
            return col, row, float(x), float(y), None, None
        longitude, latitude = _safe_geographic_coordinates(src.crs, float(x), float(y))
        return col, row, float(x), float(y), longitude, latitude


def probe_project(request: ProjectProbeRequest) -> ProjectProbeResult:
    """Sample persisted analytical products at one normalized image position."""
    manifest = _load_project(request.project_dir)
    surface_name, surface_path = _primary_surface(manifest)
    col, row, map_x, map_y, longitude, latitude = _spatial_coordinates(
        surface_path,
        request.point,
    )
    return ProjectProbeResult(
        project_id=manifest.project_id,
        point=request.point,
        pixel_col=col,
        pixel_row=row,
        map_x=map_x,
        map_y=map_y,
        longitude=longitude,
        latitude=latitude,
        surface_product=surface_name,
        surface=_artifact_sample(manifest, surface_name, request.point),
        slope=_artifact_sample(manifest, "slope", request.point),
        reference=_artifact_sample(manifest, "reference", request.point),
        residual=_artifact_sample(manifest, "residual", request.point),
        confidence=_artifact_sample(manifest, "confidence", request.point),
    )


def _cumulative_map_distance_m(
    map_coordinates: list[tuple[float, float]],
) -> list[float | None]:
    metric_distance: list[float | None] = [0.0]
    cumulative = 0.0
    for (x0, y0), (x1, y1) in pairwise(map_coordinates):
        segment = float(np.hypot(x1 - x0, y1 - y0))
        if not np.isfinite(segment):
            return [None for _ in map_coordinates]
        cumulative += segment
        metric_distance.append(cumulative)
    return metric_distance


def _profile_distances(
    surface_path: Path,
    points: list[NormalizedPoint],
) -> tuple[list[float], list[float | None]]:
    metric_gsd = ground_sample_distance_m(surface_path)
    with rasterio.open(surface_path) as src:
        pixels = [
            _floating_pixel(point, width=src.width, height=src.height)
            for point in points
        ]
        pixel_distance = [0.0]
        for (previous_col, previous_row), (col, row) in pairwise(pixels):
            pixel_distance.append(
                pixel_distance[-1] + float(np.hypot(col - previous_col, row - previous_row))
            )

        if src.transform.is_identity or metric_gsd is None:
            return pixel_distance, [None for _ in points]

        map_coordinates: list[tuple[float, float]] = []
        for col, row in pixels:
            x, y = src.transform * (col + 0.5, row + 0.5)
            map_coordinates.append((float(x), float(y)))

        if ortholoc_metric_affine_override_enabled():
            return pixel_distance, _cumulative_map_distance_m(map_coordinates)

        if src.crs is None:
            return pixel_distance, [None for _ in points]

        geographic: list[tuple[float, float]] = []
        for x, y in map_coordinates:
            longitude, latitude = _safe_geographic_coordinates(src.crs, x, y)
            if longitude is None or latitude is None:
                return pixel_distance, [None for _ in points]
            geographic.append((longitude, latitude))

    metric_distance: list[float | None] = [0.0]
    cumulative = 0.0
    for (lon0, lat0), (lon1, lat1) in pairwise(geographic):
        _, _, segment = _GEOD.inv(lon0, lat0, lon1, lat1)
        if not np.isfinite(segment):
            return pixel_distance, [None for _ in points]
        cumulative += float(abs(segment))
        metric_distance.append(cumulative)
    return pixel_distance, metric_distance


def _surface_statistics(samples: list[ProfileSample]) -> tuple[
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
]:
    values = [sample.surface.value if sample.surface.available else None for sample in samples]
    finite_values = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not finite_values:
        return None, None, None, None, None

    start_value = values[0]
    end_value = values[-1]
    vertical_delta = (
        float(end_value - start_value)
        if start_value is not None and end_value is not None
        else None
    )
    gain = 0.0
    loss = 0.0
    for previous, current in pairwise(values):
        if previous is None or current is None:
            continue
        delta = float(current - previous)
        if delta > 0:
            gain += delta
        elif delta < 0:
            loss += abs(delta)
    return vertical_delta, min(finite_values), max(finite_values), gain, loss


def sample_project_profile(request: ProjectProfileRequest) -> ProjectProfileResult:
    """Sample a deterministic subpixel analyst transect across persisted project products."""
    manifest = _load_project(request.project_dir)
    surface_name, surface_path = _primary_surface(manifest)
    fractions = np.linspace(0.0, 1.0, request.samples)
    points = [
        NormalizedPoint(
            x=float(request.start.x + fraction * (request.end.x - request.start.x)),
            y=float(request.start.y + fraction * (request.end.y - request.start.y)),
        )
        for fraction in fractions
    ]
    pixel_distances, metric_distances = _profile_distances(surface_path, points)

    surfaces = _artifact_series(manifest, surface_name, points)
    slopes = _artifact_series(manifest, "slope", points)
    references = _artifact_series(manifest, "reference", points)
    residuals = _artifact_series(manifest, "residual", points)
    confidences = _artifact_series(manifest, "confidence", points)

    samples: list[ProfileSample] = []
    for index, (fraction, point) in enumerate(zip(fractions, points, strict=True)):
        samples.append(
            ProfileSample(
                fraction=float(fraction),
                point=point,
                distance_pixels=pixel_distances[index],
                distance_m=metric_distances[index],
                surface=surfaces[index],
                slope=slopes[index],
                reference=references[index],
                residual=residuals[index],
                confidence=confidences[index],
            )
        )

    vertical_delta, minimum, maximum, gain, loss = _surface_statistics(samples)
    horizontal_distance_m = metric_distances[-1] if metric_distances else None
    return ProjectProfileResult(
        project_id=manifest.project_id,
        surface_product=surface_name,
        start=request.start,
        end=request.end,
        sample_count=len(samples),
        horizontal_distance_pixels=pixel_distances[-1],
        horizontal_distance_m=horizontal_distance_m,
        vertical_delta=vertical_delta,
        vertical_units=samples[0].surface.units if samples else None,
        minimum_surface=minimum,
        maximum_surface=maximum,
        elevation_gain=gain,
        elevation_loss=loss,
        samples=samples,
        semantics=(
            "Analyst-selected persisted-surface transect. Profile values use nodata-aware bilinear "
            "subpixel sampling. Vertical delta is endpoint surface elevation difference; it is not "
            "automatically a building-height classification. Standard georeferenced rasters use "
            "WGS84 geodesic ground distance so projected-coordinate distortion is not reported as "
            "physical length. Metric horizontal distance is omitted when the spatial contract is "
            "not trustworthy; the explicit OrthoLoC local-metric affine contract remains separate."
        ),
    )
