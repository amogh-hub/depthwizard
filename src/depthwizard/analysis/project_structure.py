from __future__ import annotations

from math import ceil, floor

import numpy as np
import rasterio
from rasterio.windows import Window

from depthwizard.analysis.structures import estimate_structure_height
from depthwizard.contracts import (
    NormalizedPoint,
    ProjectStructureHeightRequest,
    ProjectStructureHeightResult,
)
from depthwizard.pipeline.project import ProjectManifest


def _polygon_mask(
    rows: np.ndarray,
    cols: np.ndarray,
    polygon_xy: list[tuple[float, float]],
) -> np.ndarray:
    """Vectorized even/odd polygon fill over pixel-centre coordinates."""
    inside = np.zeros(rows.shape, dtype=bool)
    x = cols.astype(np.float64, copy=False)
    y = rows.astype(np.float64, copy=False)
    count = len(polygon_xy)
    for index in range(count):
        x0, y0 = polygon_xy[index]
        x1, y1 = polygon_xy[(index + 1) % count]
        crosses = (y0 > y) != (y1 > y)
        denominator = y1 - y0
        if abs(denominator) < 1e-12:
            continue
        intersection_x = (x1 - x0) * (y - y0) / denominator + x0
        inside ^= crosses & (x < intersection_x)
    return inside


def _pixel_polygon(
    polygon: list[NormalizedPoint],
    *,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    return [
        (
            point.x * max(width - 1, 0),
            point.y * max(height - 1, 0),
        )
        for point in polygon
    ]


def estimate_project_structure_height(
    request: ProjectStructureHeightRequest,
) -> ProjectStructureHeightResult:
    """Measure a user-selected structure against robust surrounding local ground.

    Structural height is available only from an absolute metric DSM. The polygon is explicit analyst
    evidence; DepthWizard never pretends that a DSM alone identifies an object footprint.
    """
    manifest_path = request.project_dir / "project-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("project manifest does not exist")
    manifest = ProjectManifest.load(request.project_dir)
    dsm_path = manifest.artifact_path("dsm")
    if dsm_path is None or not dsm_path.is_file():
        raise ValueError("structural-height measurement requires an absolute metric DSM")
    dsm_artifact = manifest.artifacts.get("dsm") or {}
    if dsm_artifact.get("units") != "m":
        raise ValueError("structural-height measurement requires DSM units in metres")

    with rasterio.open(dsm_path) as src:
        if src.width < 2 or src.height < 2:
            raise ValueError("DSM is too small for structural-height measurement")
        polygon_xy = _pixel_polygon(request.polygon, width=src.width, height=src.height)
        xs = [item[0] for item in polygon_xy]
        ys = [item[1] for item in polygon_xy]
        margin = request.ring_pixels + 2
        col0 = max(0, floor(min(xs)) - margin)
        row0 = max(0, floor(min(ys)) - margin)
        col1 = min(src.width, ceil(max(xs)) + margin + 1)
        row1 = min(src.height, ceil(max(ys)) + margin + 1)
        if col1 - col0 < 2 or row1 - row0 < 2:
            raise ValueError("selected structure polygon has insufficient raster coverage")

        window = Window(col0, row0, col1 - col0, row1 - row0)
        sample = src.read(1, window=window, masked=True).astype(np.float64)
        values = np.asarray(sample.filled(np.nan), dtype=np.float64)
        invalid = np.ma.getmaskarray(sample) | ~np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            invalid |= values == float(src.nodata)
        values[invalid] = np.nan

    local_rows, local_cols = np.mgrid[row0:row1, col0:col1]
    structure_mask = _polygon_mask(local_rows + 0.5, local_cols + 0.5, polygon_xy)
    estimate = estimate_structure_height(
        values,
        structure_mask,
        ring_pixels=request.ring_pixels,
        min_structure_pixels=request.min_structure_pixels,
        min_ground_pixels=request.min_ground_pixels,
    )
    warnings: list[str] = []
    if estimate.structure_height_m <= 0.0:
        warnings.append(
            "Selected footprint does not rise above the robust local-ground estimate; verify the structure selection."
        )

    return ProjectStructureHeightResult(
        project_id=manifest.project_id,
        polygon=request.polygon,
        ring_pixels=request.ring_pixels,
        top_elevation_m=estimate.top_elevation_m,
        ground_elevation_m=estimate.ground_elevation_m,
        structure_height_m=estimate.structure_height_m,
        structure_pixels=estimate.structure_pixels,
        ground_pixels=estimate.ground_pixels,
        warnings=warnings,
        semantics=(
            "Analyst-selected structural height = robust median DSM elevation inside the explicit "
            "footprint minus robust median elevation in the surrounding local-ground ring. The "
            "selection is not an automatic building classification."
        ),
    )
