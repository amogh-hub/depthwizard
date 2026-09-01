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
from depthwizard.io.raster import ground_sample_distance_m
from depthwizard.pipeline.project import ProjectManifest

DEFAULT_ROOF_INSET_M = 0.50
DEFAULT_GROUND_INNER_BUFFER_M = 1.50
DEFAULT_GROUND_OUTER_BUFFER_M = 8.00


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
    """Measure a selected structure against physically scaled surrounding local ground.

    Structural height is available only from an absolute metric DSM with trustworthy physical GSD.
    The historical ``ring_pixels`` request field is retained for client compatibility, but project
    measurements no longer use a fixed pixel radius because that changes physical support with GSD.
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

    gsd = ground_sample_distance_m(dsm_path)
    if gsd is None:
        raise ValueError(
            "structural-height measurement requires trustworthy physical GSD; "
            "DepthWizard will not convert an arbitrary pixel ring into metres"
        )
    gsd_x_m, gsd_y_m = gsd
    margin_cols = ceil(DEFAULT_GROUND_OUTER_BUFFER_M / gsd_x_m) + 2
    margin_rows = ceil(DEFAULT_GROUND_OUTER_BUFFER_M / gsd_y_m) + 2
    effective_outer_ring_pixels = max(margin_cols - 2, margin_rows - 2)

    with rasterio.open(dsm_path) as src:
        if src.width < 2 or src.height < 2:
            raise ValueError("DSM is too small for structural-height measurement")
        polygon_xy = _pixel_polygon(request.polygon, width=src.width, height=src.height)
        xs = [item[0] for item in polygon_xy]
        ys = [item[1] for item in polygon_xy]
        col0 = max(0, floor(min(xs)) - margin_cols)
        row0 = max(0, floor(min(ys)) - margin_rows)
        col1 = min(src.width, ceil(max(xs)) + margin_cols + 1)
        row1 = min(src.height, ceil(max(ys)) + margin_rows + 1)
        if col1 - col0 < 2 or row1 - row0 < 2:
            raise ValueError("selected structure polygon has insufficient raster coverage")

        window = Window.from_slices((row0, row1), (col0, col1))
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
        ring_pixels=None,
        min_structure_pixels=request.min_structure_pixels,
        min_ground_pixels=request.min_ground_pixels,
        pixel_size_x_m=gsd_x_m,
        pixel_size_y_m=gsd_y_m,
        roof_inset_m=DEFAULT_ROOF_INSET_M,
        ground_inner_buffer_m=DEFAULT_GROUND_INNER_BUFFER_M,
        ground_outer_buffer_m=DEFAULT_GROUND_OUTER_BUFFER_M,
    )

    warnings: list[str] = []
    if estimate.structure_height_m <= 0.0:
        warnings.append(
            "Selected footprint does not rise above the robust local-ground estimate; "
            "verify the structure selection and DSM fidelity."
        )
    if estimate.structure_pixels < 9:
        warnings.append(
            "Selected roof core has fewer than nine valid DSM pixels; structural height has "
            "limited independent spatial support."
        )
    if estimate.roof_inset_m < DEFAULT_ROOF_INSET_M - 1e-9:
        warnings.append(
            "Roof-edge exclusion was reduced because the selected footprint was too small for the "
            "default 0.50 m inset."
        )
    if estimate.ground_inlier_fraction < 0.60:
        warnings.append(
            "Fewer than 60% of surrounding ground candidates survived robust plane fitting; "
            "nearby objects or DSM artefacts may contaminate local-ground evidence."
        )
    if estimate.ground_sector_coverage < 0.75:
        warnings.append(
            "Robust ground support does not surround the structure on at least three of four sides; "
            "the fitted local terrain plane has limited spatial support."
        )
    dispersion_limit = max(1.0, 0.25 * abs(estimate.structure_height_m))
    if estimate.local_height_dispersion_m > dispersion_limit:
        warnings.append(
            "Roof-to-ground height varies strongly inside the selected footprint; inspect the DSM "
            "for mixed roof levels, vegetation, or local reconstruction error."
        )

    return ProjectStructureHeightResult(
        project_id=manifest.project_id,
        polygon=request.polygon,
        ring_pixels=effective_outer_ring_pixels,
        top_elevation_m=estimate.top_elevation_m,
        ground_elevation_m=estimate.ground_elevation_m,
        structure_height_m=estimate.structure_height_m,
        structure_pixels=estimate.structure_pixels,
        ground_pixels=estimate.ground_inlier_pixels,
        warnings=warnings,
        semantics=(
            "Analyst-selected structural height = robust median DSM roof elevation from a "
            f"{estimate.roof_inset_m:.2f} m edge-inset roof core minus a robust local ground plane. "
            f"Ground candidates are sampled in a physical {estimate.ground_inner_buffer_m:.2f}-"
            f"{estimate.ground_outer_buffer_m:.2f} m annulus using trustworthy raster GSD, then "
            "high-object contaminants are rejected before the plane is extrapolated beneath the "
            "roof. The legacy request ring_pixels field does not control project-level scientific "
            "support. The footprint is explicit analyst evidence and is not an automatic building "
            "classification."
        ),
    )
