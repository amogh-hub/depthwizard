from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from pyproj import CRS, Geod, Transformer

from depthwizard.contracts import (
    NormalizedPoint,
    ProfileSample,
    ProjectProbeRequest,
    ProjectProbeResult,
    ProjectProfileRequest,
    ProjectProfileResult,
    RasterSample,
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


def _pixel_index(point: NormalizedPoint, *, width: int, height: int) -> tuple[int, int]:
    col = round(point.x * max(width - 1, 0))
    row = round(point.y * max(height - 1, 0))
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


def _safe_geographic_coordinates(
    crs: object,
    x: float,
    y: float,
) -> tuple[float | None, float | None]:
    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        longitude, latitude = transformer.transform(x, y)
    except Exception:
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
    with rasterio.open(path) as src:
        col, row = _pixel_index(point, width=src.width, height=src.height)
        if src.crs is None or src.transform.is_identity:
            return col, row, None, None, None, None
        x, y = src.xy(row, col)
        longitude, latitude = _safe_geographic_coordinates(src.crs, float(x), float(y))
        return col, row, float(x), float(y), longitude, latitude


def probe_project(request: ProjectProbeRequest) -> ProjectProbeResult:
    """Sample persisted analytical products at one normalized image position.

    Coordinates are normalized to the raster extent so desktop previews may be downsampled without
    changing scientific sampling. The endpoint is read-only and never writes analyst clicks into the
    project evidence record.
    """
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


def _projected_distance_factors(crs: CRS) -> tuple[float, float] | None:
    """Return projected-coordinate conversion factors to metres when the CRS declares them."""
    if not crs.is_projected or len(crs.axis_info) < 2:
        return None
    x_factor = crs.axis_info[0].unit_conversion_factor
    y_factor = crs.axis_info[1].unit_conversion_factor
    if x_factor is None or y_factor is None:
        return None
    factors = (float(x_factor), float(y_factor))
    if not all(np.isfinite(value) and value > 0 for value in factors):
        return None
    return factors


def _profile_distances(
    surface_path: Path,
    points: list[NormalizedPoint],
) -> tuple[list[float], list[float | None]]:
    with rasterio.open(surface_path) as src:
        pixels = [
            _pixel_index(point, width=src.width, height=src.height)
            for point in points
        ]
        pixel_distance = [0.0]
        for (previous_col, previous_row), (col, row) in pairwise(pixels):
            pixel_distance.append(
                pixel_distance[-1] + float(np.hypot(col - previous_col, row - previous_row))
            )

        if src.crs is None or src.transform.is_identity:
            return pixel_distance, [None for _ in points]

        crs = CRS.from_user_input(src.crs)
        map_coordinates = [
            tuple(float(value) for value in src.xy(row, col))
            for col, row in pixels
        ]

        projected_factors = _projected_distance_factors(crs)
        if projected_factors is not None:
            x_factor, y_factor = projected_factors
            metric_distance: list[float | None] = [0.0]
            cumulative = 0.0
            for (x0, y0), (x1, y1) in pairwise(map_coordinates):
                segment = float(
                    np.hypot((x1 - x0) * x_factor, (y1 - y0) * y_factor)
                )
                if not np.isfinite(segment):
                    return pixel_distance, [None for _ in points]
                cumulative += segment
                metric_distance.append(cumulative)
            return pixel_distance, metric_distance

        if not crs.is_geographic:
            return pixel_distance, [None for _ in points]

        geographic: list[tuple[float, float]] = []
        for x, y in map_coordinates:
            longitude, latitude = _safe_geographic_coordinates(crs, x, y)
            if longitude is None or latitude is None:
                return pixel_distance, [None for _ in points]
            geographic.append((longitude, latitude))

    metric_distance = [0.0]
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
    """Sample a deterministic analyst transect across the persisted project products."""
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

    samples: list[ProfileSample] = []
    for index, (fraction, point) in enumerate(zip(fractions, points, strict=True)):
        samples.append(
            ProfileSample(
                fraction=float(fraction),
                point=point,
                distance_pixels=pixel_distances[index],
                distance_m=metric_distances[index],
                surface=_artifact_sample(manifest, surface_name, point),
                slope=_artifact_sample(manifest, "slope", point),
                reference=_artifact_sample(manifest, "reference", point),
                residual=_artifact_sample(manifest, "residual", point),
                confidence=_artifact_sample(manifest, "confidence", point),
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
            "Analyst-selected surface transect. Vertical delta is endpoint surface elevation "
            "difference; it is not automatically a building-height classification."
        ),
    )
