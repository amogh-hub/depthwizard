from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio

from depthwizard.contracts import (
    ProjectMeshBuildRequest,
    ProjectMeshReport,
    TerrainLodArtifact,
)
from depthwizard.io.raster import ground_sample_distance_m, read_rgb
from depthwizard.mesh.terrain import export_lod_pyramid
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.relative_scale import (
    RELATIVE_DISPLAY_SCALE_POLICY,
    relative_display_vertical_scale,
)

SurfaceProduct = Literal["dsm", "rdsm"]
HorizontalUnits = Literal["m", "px"]
VerticalUnits = Literal["m", "relative"]


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _surface_artifact(manifest: ProjectManifest) -> tuple[SurfaceProduct, Path, str]:
    for name in ("dsm", "rdsm"):
        path = manifest.artifact_path(name)
        artifact = manifest.artifacts.get(name)
        if path is None or artifact is None:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"project {name} artifact does not exist: {path}")
        recorded_sha = artifact.get("sha256")
        actual_sha = sha256_file(path)
        if isinstance(recorded_sha, str) and recorded_sha != actual_sha:
            raise RuntimeError(f"project {name} artifact SHA-256 no longer matches the manifest")
        product: SurfaceProduct = "dsm" if name == "dsm" else "rdsm"
        return product, path, actual_sha
    raise FileNotFoundError("project has no persisted DSM or rDSM available for 3D terrain")


def _read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        elevation = src.read(1).astype(np.float32)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(elevation)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= elevation != np.float32(src.nodata)
    valid_count = int(valid.sum())
    if valid_count < 4:
        raise ValueError("terrain mesh requires at least four valid elevation pixels")
    fill = float(np.median(elevation[valid]))
    sanitized = np.where(valid, elevation, fill).astype(np.float32)
    return sanitized, valid


def _lod_strides(shape: tuple[int, int], max_finest_samples: int, lod_levels: int) -> tuple[int, ...]:
    height, width = shape
    max_dimension = max(height, width)
    base_stride = max(1, math.ceil(max_dimension / max_finest_samples))
    strides: list[int] = []
    for level in range(lod_levels):
        stride = base_stride * (2**level)
        if stride not in strides:
            strides.append(stride)
    return tuple(strides)


def _existing_report_if_valid(
    report_path: Path,
    *,
    project_id: str,
    build_config_sha256: str,
    surface_sha256: str,
    texture_sha256: str,
) -> ProjectMeshReport | None:
    if not report_path.is_file():
        return None
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        report = ProjectMeshReport.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if (
        report.project_id != project_id
        or report.build_config_sha256 != build_config_sha256
        or report.surface_sha256 != surface_sha256
        or report.texture_sha256 != texture_sha256
    ):
        return None
    for lod in report.lods:
        if not lod.path.is_file() or sha256_file(lod.path) != lod.sha256:
            return None
    return report


def load_project_mesh(project_dir: str | Path) -> ProjectMeshReport:
    path = Path(project_dir) / "mesh" / "mesh-manifest.json"
    if not path.is_file():
        raise FileNotFoundError("project 3D mesh has not been built")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        report = ProjectMeshReport.model_validate(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"unable to read project mesh manifest: {exc}") from exc
    for lod in report.lods:
        if not lod.path.is_file():
            raise FileNotFoundError(f"terrain LOD is missing: {lod.path}")
        if sha256_file(lod.path) != lod.sha256:
            raise RuntimeError(f"terrain LOD SHA-256 mismatch: {lod.path.name}")
    return report


def _surface_semantics(
    surface_product: SurfaceProduct,
    horizontal_units: HorizontalUnits,
) -> str:
    if surface_product == "dsm" and horizontal_units == "m":
        return "textured_metric_dsm_terrain_metric_xy"
    if surface_product == "dsm":
        return "textured_metric_dsm_terrain_pixel_xy_untrusted_spatial_scale"
    if horizontal_units == "m":
        return "textured_relative_surface_terrain_metric_xy_display_normalized_z"
    return "textured_relative_surface_terrain_pixel_xy_display_normalized_z"


def build_project_mesh(request: ProjectMeshBuildRequest) -> ProjectMeshReport:
    """Build and persist a deterministic textured terrain LOD pyramid for one project.

    This is a derivative visualization stage. It consumes only the already-persisted project
    surface and source RGB, never reference DSM values or validation residuals, so generating a 3D
    view cannot feed evaluation evidence back into reconstruction or calibration. Horizontal metric
    scale is used only when the persisted raster passes the shared CRS/GSD trust contract; otherwise
    the mesh falls back to a pixel XY grid instead of manufacturing metres.

    Metric DSM meshes preserve physical Z at 1x. Dimensionless rDSM meshes cannot have a physical
    XY:Z aspect ratio, so their *display geometry only* receives a deterministic robust Z scale that
    maps P02-P98 relief to 12% of the shorter horizontal scene span. Persisted rDSM values and every
    probe/profile/calibration/evaluation path remain unchanged and in raw relative units.
    """
    project_dir = request.project_dir
    if not (project_dir / "project-manifest.json").is_file():
        raise FileNotFoundError("project manifest does not exist")
    manifest = ProjectManifest.load(project_dir)
    surface_product, surface_path, surface_sha = _surface_artifact(manifest)
    source_path = manifest.source_path
    if not source_path.is_file():
        raise FileNotFoundError(f"project source raster does not exist: {source_path}")
    texture_sha = sha256_file(source_path)
    if manifest.source_sha256 is not None and texture_sha != manifest.source_sha256:
        raise RuntimeError("project source raster SHA-256 no longer matches the manifest")

    metric_gsd = ground_sample_distance_m(surface_path)
    horizontal_units: HorizontalUnits
    if metric_gsd is None:
        gsd_x = 1.0
        gsd_y = 1.0
        horizontal_units = "px"
    else:
        gsd_x, gsd_y = metric_gsd
        horizontal_units = "m"
    metric_horizontal_scale_trusted = horizontal_units == "m"

    elevation, valid = _read_surface(surface_path)
    raster_shape = (int(elevation.shape[0]), int(elevation.shape[1]))
    horizontal_span_x = max(raster_shape[1] - 1, 1) * gsd_x
    horizontal_span_y = max(raster_shape[0] - 1, 1) * gsd_y
    if surface_product == "rdsm":
        display_vertical_scale = relative_display_vertical_scale(
            elevation,
            valid,
            span_x=horizontal_span_x,
            span_y=horizontal_span_y,
        )
        display_vertical_center = float(np.median(elevation[valid].astype(np.float64)))
        display_scale_policy = RELATIVE_DISPLAY_SCALE_POLICY
    else:
        display_vertical_scale = 1.0
        display_vertical_center = 0.0
        display_scale_policy = "metric_identity_v1"

    build_config_sha = _canonical_sha256(
        {
            "schema_version": 3,
            "max_finest_samples": request.max_finest_samples,
            "lod_levels": request.lod_levels,
            "surface_product": surface_product,
            "surface_sha256": surface_sha,
            "texture_sha256": texture_sha,
            "horizontal_units": horizontal_units,
            "gsd_x": gsd_x,
            "gsd_y": gsd_y,
            "metric_horizontal_scale_trusted": metric_horizontal_scale_trusted,
            "display_vertical_scale_policy": display_scale_policy,
            "display_vertical_scale": display_vertical_scale,
        }
    )
    mesh_dir = project_dir / "mesh"
    report_path = mesh_dir / "mesh-manifest.json"
    existing = _existing_report_if_valid(
        report_path,
        project_id=manifest.project_id,
        build_config_sha256=build_config_sha,
        surface_sha256=surface_sha,
        texture_sha256=texture_sha,
    )
    if existing is not None:
        return existing

    started = time.perf_counter()
    manifest.record_stage(
        ProcessingStage.MESH,
        status="running",
        details={
            "surface_product": surface_product,
            "surface_sha256": surface_sha,
            "texture_sha256": texture_sha,
            "build_config_sha256": build_config_sha,
            "reference_data_used": False,
            "horizontal_units": horizontal_units,
            "gsd_x": gsd_x,
            "gsd_y": gsd_y,
            "metric_horizontal_scale_trusted": metric_horizontal_scale_trusted,
            "display_vertical_scale": display_vertical_scale,
            "display_vertical_center_raw": display_vertical_center,
            "display_vertical_scale_policy": display_scale_policy,
            "display_only_normalization": surface_product == "rdsm",
        },
    )
    try:
        rgb = read_rgb(source_path)
        if rgb.shape[:2] != elevation.shape:
            raise ValueError(
                "project source RGB and persisted elevation grid differ; refusing texture/grid guess"
            )

        vertical_units: VerticalUnits = "m" if surface_product == "dsm" else "relative"
        strides = _lod_strides(raster_shape, request.max_finest_samples, request.lod_levels)
        if surface_product == "rdsm":
            mesh_elevation = (
                (elevation.astype(np.float64) - display_vertical_center) * display_vertical_scale
            ).astype(np.float32)
        else:
            mesh_elevation = elevation

        mesh_dir.mkdir(parents=True, exist_ok=True)
        exports = export_lod_pyramid(
            mesh_dir,
            mesh_elevation,
            rgb,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            strides=strides,
            valid_mask=valid,
        )
        lods: list[TerrainLodArtifact] = []
        for level, item in enumerate(exports):
            lods.append(
                TerrainLodArtifact(
                    level=level,
                    stride=item.stride,
                    path=item.path.resolve(strict=False),
                    sha256=sha256_file(item.path),
                    vertices=item.vertices,
                    faces=item.faces,
                    width_samples=item.width_samples,
                    height_samples=item.height_samples,
                )
            )

        valid_values = elevation[valid].astype(np.float64)
        report = ProjectMeshReport(
            project_id=manifest.project_id,
            surface_product=surface_product,
            surface_sha256=surface_sha,
            texture_sha256=texture_sha,
            build_config_sha256=build_config_sha,
            horizontal_units=horizontal_units,
            vertical_units=vertical_units,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            raster_width=raster_shape[1],
            raster_height=raster_shape[0],
            valid_pixels=int(valid.sum()),
            minimum_elevation=float(np.min(valid_values)),
            maximum_elevation=float(np.max(valid_values)),
            relief=float(np.max(valid_values) - np.min(valid_values)),
            lods=lods,
            mesh_manifest_path=report_path.resolve(strict=False),
            semantics=_surface_semantics(surface_product, horizontal_units),
        )
        temporary = report_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(report_path)

        for lod in lods:
            manifest.register_artifact(
                f"terrain_lod{lod.level}",
                lod.path,
                semantics=f"textured_terrain_glb_lod_{lod.level}",
                units=None,
                sha256=lod.sha256,
            )
        manifest.register_artifact(
            "mesh_manifest",
            report_path,
            semantics="terrain_lod_manifest",
            units=None,
            sha256=sha256_file(report_path),
        )
        manifest.record_stage(
            ProcessingStage.MESH,
            status="completed",
            artifacts={
                "mesh_manifest": str(report_path.resolve(strict=False)),
                **{f"terrain_lod{lod.level}": str(lod.path) for lod in lods},
            },
            details={
                "surface_product": surface_product,
                "surface_sha256": surface_sha,
                "texture_sha256": texture_sha,
                "build_config_sha256": build_config_sha,
                "reference_data_used": False,
                "horizontal_units": horizontal_units,
                "vertical_units": vertical_units,
                "gsd_x": gsd_x,
                "gsd_y": gsd_y,
                "metric_horizontal_scale_trusted": metric_horizontal_scale_trusted,
                "display_vertical_scale": display_vertical_scale,
                "display_vertical_center_raw": display_vertical_center,
                "display_vertical_scale_policy": display_scale_policy,
                "display_only_normalization": surface_product == "rdsm",
                "lod_strides": list(strides),
                "valid_pixels": int(valid.sum()),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        return report
    except Exception as exc:
        manifest.add_error(str(exc), stage=ProcessingStage.MESH)
        manifest.record_stage(
            ProcessingStage.MESH,
            status="failed",
            details={
                "surface_product": surface_product,
                "surface_sha256": surface_sha,
                "texture_sha256": texture_sha,
                "build_config_sha256": build_config_sha,
                "reference_data_used": False,
                "horizontal_units": horizontal_units,
                "gsd_x": gsd_x,
                "gsd_y": gsd_y,
                "metric_horizontal_scale_trusted": metric_horizontal_scale_trusted,
                "display_vertical_scale": display_vertical_scale,
                "display_vertical_center_raw": display_vertical_center,
                "display_vertical_scale_policy": display_scale_policy,
                "display_only_normalization": surface_product == "rdsm",
                "error": str(exc),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
