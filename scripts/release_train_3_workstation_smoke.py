from __future__ import annotations

import json
import os
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import rasterio
import trimesh

from depthwizard.contracts import NormalizedPoint, ProjectProbeRequest, ProjectProfileRequest
from depthwizard.evaluation.project_analysis import probe_project, sample_project_profile
from depthwizard.export.project_package import load_project_export
from depthwizard.mesh.project_mesh import load_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.raster_preview import PreviewLayer, render_project_layer_preview
from scripts.release_train_3_spatial_foundation_smoke import (
    PROJECT_DIR,
    _direct_affine_spacing_m,
)
from scripts.release_train_3_spatial_foundation_smoke import (
    REPORT_PATH as SPATIAL_FOUNDATION_REPORT,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _uv_contract(path: Path) -> dict[str, object]:
    loaded = cast(Any, trimesh.load(path, force="scene", process=False))
    geometry = getattr(loaded, "geometry", None)
    if not isinstance(geometry, dict) or not geometry:
        raise RuntimeError(f"terrain LOD contains no loadable geometry: {path.name}")

    uv_vertices = 0
    minimum = np.array([np.inf, np.inf], dtype=np.float64)
    maximum = np.array([-np.inf, -np.inf], dtype=np.float64)
    for mesh in geometry.values():
        visual = getattr(mesh, "visual", None)
        raw_uv = getattr(visual, "uv", None)
        if raw_uv is None:
            continue
        uv = np.asarray(raw_uv, dtype=np.float64)
        if uv.ndim != 2 or uv.shape[1] != 2 or uv.shape[0] == 0:
            raise RuntimeError(f"terrain LOD has malformed UV coordinates: {path.name}")
        if not np.all(np.isfinite(uv)):
            raise RuntimeError(f"terrain LOD has non-finite UV coordinates: {path.name}")
        uv_vertices += int(uv.shape[0])
        minimum = np.minimum(minimum, np.min(uv, axis=0))
        maximum = np.maximum(maximum, np.max(uv, axis=0))

    if uv_vertices == 0:
        raise RuntimeError(f"terrain LOD has no texture UV coordinates: {path.name}")
    tolerance = 1e-4
    if np.any(minimum < -tolerance) or np.any(maximum > 1.0 + tolerance):
        raise RuntimeError(
            f"terrain LOD UV coordinates escape normalized texture space: {path.name} "
            f"min={minimum.tolist()} max={maximum.tolist()}"
        )
    return {
        "uv_vertices": uv_vertices,
        "minimum": [float(value) for value in minimum],
        "maximum": [float(value) for value in maximum],
    }


def _preview_contract(layer: PreviewLayer) -> dict[str, object]:
    payload = render_project_layer_preview(PROJECT_DIR, layer, max_side=512)
    if not payload.startswith(PNG_SIGNATURE):
        raise RuntimeError(f"project preview layer '{layer}' did not render as PNG")
    return {
        "layer": layer,
        "bytes": len(payload),
        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
    }


def _pixel_index(point: NormalizedPoint, *, width: int, height: int) -> tuple[int, int]:
    col = round(point.x * max(width - 1, 0))
    row = round(point.y * max(height - 1, 0))
    return min(max(col, 0), width - 1), min(max(row, 0), height - 1)


def _direct_affine_profile_distance_m(
    surface_path: Path,
    points: list[NormalizedPoint],
) -> float:
    """Independently derive the accepted profile distance from raster affine coordinates."""
    with rasterio.open(surface_path) as src:
        map_coordinates: list[tuple[float, float]] = []
        for point in points:
            col, row = _pixel_index(point, width=src.width, height=src.height)
            x, y = src.xy(row, col)
            map_coordinates.append((float(x), float(y)))
    cumulative = 0.0
    for (x0, y0), (x1, y1) in pairwise(map_coordinates):
        cumulative += float(np.hypot(x1 - x0, y1 - y0))
    return cumulative


def _direct_affine_scene_diagonal_m(surface_path: Path) -> float:
    """Return the larger opposite-corner diagonal directly in local metric affine coordinates."""
    with rasterio.open(surface_path) as src:
        corners = [
            tuple(float(value) for value in src.xy(0, 0)),
            tuple(float(value) for value in src.xy(0, max(src.width - 1, 0))),
            tuple(float(value) for value in src.xy(max(src.height - 1, 0), 0)),
            tuple(
                float(value)
                for value in src.xy(max(src.height - 1, 0), max(src.width - 1, 0))
            ),
        ]
    diagonal_a = float(np.hypot(corners[3][0] - corners[0][0], corners[3][1] - corners[0][1]))
    diagonal_b = float(np.hypot(corners[2][0] - corners[1][0], corners[2][1] - corners[1][1]))
    return max(diagonal_a, diagonal_b)


def main() -> None:
    """Close RT3 workstation integration without rerunning consumed scientific evidence."""
    os.environ["DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE"] = "1"
    manifest_path = PROJECT_DIR / "project-manifest.json"
    if not manifest_path.is_file() or not SPATIAL_FOUNDATION_REPORT.is_file():
        raise RuntimeError(
            "RT3 workstation acceptance requires the corrected OrthoLoC spatial-foundation project at "
            f"{PROJECT_DIR}. Run make release-train-3-spatial-foundation-smoke first."
        )

    print("DepthWizard Release Train 3 workstation acceptance")
    print("Purpose: integrated 3D analyst/runtime closure; no consumed benchmark/model-promotion rerun")

    manifest = ProjectManifest.load(PROJECT_DIR)
    mesh = load_project_mesh(PROJECT_DIR)
    export = load_project_export(PROJECT_DIR)
    mesh_stage = manifest.stages.get("mesh", {})
    mesh_details = mesh_stage.get("details", {}) if isinstance(mesh_stage, dict) else {}
    if not isinstance(mesh_details, dict) or mesh_details.get("reference_data_used") is not False:
        raise RuntimeError("mesh stage no longer preserves reference_data_used=false")
    if mesh.surface_product != "dsm" or mesh.vertical_units != "m":
        raise RuntimeError("accepted workstation project is no longer backed by a metric DSM surface")
    if mesh.horizontal_units != "m" or mesh_details.get("metric_horizontal_scale_trusted") is not True:
        raise RuntimeError("corrected OrthoLoC workstation did not preserve trusted local metric-affine XY")

    surface_path = manifest.artifact_path("dsm")
    if surface_path is None or not surface_path.is_file():
        raise RuntimeError("accepted workstation project has no persisted DSM")
    direct_gsd = _direct_affine_spacing_m(surface_path)
    if not np.allclose((mesh.gsd_x, mesh.gsd_y), direct_gsd, rtol=1e-7, atol=1e-9):
        raise RuntimeError("mesh GSD no longer agrees with the persisted DSM affine metric contract")

    lod_contracts: list[dict[str, object]] = []
    for lod in mesh.lods:
        if sha256_file(lod.path) != lod.sha256:
            raise RuntimeError(f"terrain LOD hash mismatch: {lod.path.name}")
        lod_contracts.append(
            {
                "level": lod.level,
                "stride": lod.stride,
                "vertices": lod.vertices,
                "faces": lod.faces,
                "sha256": lod.sha256,
                "uv": _uv_contract(lod.path),
            }
        )

    preview_layers: list[PreviewLayer] = ["optical", "dsm", "slope", "reference", "residual"]
    preview_contracts = [_preview_contract(layer) for layer in preview_layers]
    if manifest.artifact_path("confidence") is not None:
        preview_contracts.append(_preview_contract("confidence"))

    centre = probe_project(
        ProjectProbeRequest(project_dir=PROJECT_DIR, point=NormalizedPoint(x=0.5, y=0.5))
    )
    if not centre.surface.available or centre.surface.units != "m" or centre.surface.value is None:
        raise RuntimeError("centre terrain probe is not backed by an available metric DSM value")
    if centre.longitude is not None or centre.latitude is not None:
        raise RuntimeError(
            "OrthoLoC local-metric affine acceptance must not manufacture global longitude/latitude"
        )

    profile = sample_project_profile(
        ProjectProfileRequest(
            project_dir=PROJECT_DIR,
            start=NormalizedPoint(x=0.18, y=0.22),
            end=NormalizedPoint(x=0.82, y=0.78),
            samples=96,
        )
    )
    available_surface_samples = sum(1 for sample in profile.samples if sample.surface.available)
    if available_surface_samples < 2:
        raise RuntimeError("profile has insufficient raster-backed surface samples")
    if profile.horizontal_distance_m is None or profile.horizontal_distance_m <= 0:
        raise RuntimeError("corrected local metric-affine profile did not preserve physical distance")

    direct_profile_m = _direct_affine_profile_distance_m(
        surface_path,
        [sample.point for sample in profile.samples],
    )
    if not np.isclose(profile.horizontal_distance_m, direct_profile_m, rtol=1e-7, atol=1e-7):
        raise RuntimeError(
            "profile metric distance disagrees with an independent direct-affine recomputation: "
            f"profile={profile.horizontal_distance_m} direct_affine={direct_profile_m}"
        )
    direct_scene_diagonal_m = _direct_affine_scene_diagonal_m(surface_path)
    if not np.isfinite(direct_scene_diagonal_m) or direct_scene_diagonal_m <= 0:
        raise RuntimeError("persisted DSM affine does not preserve a finite positive metric scene extent")

    scene_diagonal_pixels = float(
        np.hypot(max(mesh.raster_width - 1, 0), max(mesh.raster_height - 1, 0))
    )
    if profile.horizontal_distance_pixels > scene_diagonal_pixels * 1.01:
        raise RuntimeError("profile pixel distance exceeds the full scene pixel-grid diagonal")

    if not export.bundle_path.is_file() or sha256_file(export.bundle_path) != export.bundle_sha256:
        raise RuntimeError("previously accepted project export bundle no longer matches its recorded hash")
    if export.include_source:
        raise RuntimeError("default accepted export unexpectedly includes original source imagery bytes")

    acceptance = {
        "schema_version": 3,
        "status": "PASS_RT3_WORKSTATION_PATH",
        "purpose": (
            "Release Train 3 integrated analyst workstation acceptance over the corrected OrthoLoC "
            "spatial-foundation project; visualization/runtime closure only"
        ),
        "project_id": manifest.project_id,
        "project_manifest_sha256": sha256_file(manifest_path),
        "surface_product": mesh.surface_product,
        "surface_sha256": mesh.surface_sha256,
        "horizontal_units": mesh.horizontal_units,
        "vertical_units": mesh.vertical_units,
        "metric_horizontal_scale_trusted": True,
        "spatial_scale_policy": "ortholoc_dataset_local_metric_affine",
        "global_lon_lat_claim": False,
        "direct_affine_gsd_x_m": direct_gsd[0],
        "direct_affine_gsd_y_m": direct_gsd[1],
        "reference_data_used_for_mesh_geometry": False,
        "mesh_lods": lod_contracts,
        "preview_layers": preview_contracts,
        "probe": {
            "point": centre.point.model_dump(mode="json"),
            "pixel_col": centre.pixel_col,
            "pixel_row": centre.pixel_row,
            "map_x": centre.map_x,
            "map_y": centre.map_y,
            "longitude": centre.longitude,
            "latitude": centre.latitude,
            "surface_value": centre.surface.value,
            "surface_units": centre.surface.units,
            "slope_available": centre.slope.available,
            "reference_available": centre.reference.available,
            "residual_available": centre.residual.available,
            "confidence_available": centre.confidence.available,
        },
        "profile": {
            "samples": profile.sample_count,
            "available_surface_samples": available_surface_samples,
            "horizontal_distance_pixels": profile.horizontal_distance_pixels,
            "scene_diagonal_pixels": scene_diagonal_pixels,
            "horizontal_distance_m": profile.horizontal_distance_m,
            "direct_affine_horizontal_distance_m": direct_profile_m,
            "direct_affine_scene_diagonal_m": direct_scene_diagonal_m,
            "distance_matches_direct_affine": True,
            "vertical_delta": profile.vertical_delta,
            "vertical_units": profile.vertical_units,
        },
        "export": {
            "bundle_path": str(export.bundle_path),
            "bundle_sha256": export.bundle_sha256,
            "bundle_bytes": export.bundle_bytes,
            "registered_files": len(export.files),
            "include_source": export.include_source,
        },
        "scientific_boundary": (
            "Terrain LODs, UV analytical overlays, camera flythroughs, renderer LOD selection and 3D "
            "analysis graphics are display derivatives. Numeric probe/profile values remain raster-backed. "
            "For this official OrthoLoC acceptance scene, horizontal metres come only from the explicit "
            "dataset-local affine pixel-scale contract and are cross-checked by a direct affine profile "
            "recomputation; the demo TIFF CRS is not used to claim global lon/lat. Export remains a "
            "transport derivative. No consumed benchmark or model-promotion protocol was rerun."
        ),
    }
    output_dir = PROJECT_DIR / "workstation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "release-train-3-workstation-acceptance.json"
    output.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("DepthWizard Release Train 3 integrated workstation path: PASS")
    print(f"Project: {manifest.project_id}")
    print(f"Terrain LODs with normalized UV contract: {len(lod_contracts)}")
    print(f"Rendered persisted analytical preview layers: {len(preview_contracts)}")
    print(f"Centre raster-backed surface probe: {centre.surface.value:.3f} m")
    print(f"Affine metric scale: {direct_gsd[0]:.6f} m/px × {direct_gsd[1]:.6f} m/px")
    print(
        f"Metric profile: {profile.sample_count} samples · "
        f"{profile.horizontal_distance_m:.3f} m · {available_surface_samples} available surface samples"
    )
    print(f"Direct-affine profile cross-check: {direct_profile_m:.3f} m")
    print(f"Direct-affine scene diagonal: {direct_scene_diagonal_m:.3f} m")
    print("Global lon/lat claim from demo TIFF CRS: NO")
    print(f"Accepted export SHA-256 preserved: {export.bundle_sha256}")
    print("Reference data used for mesh geometry: NO")
    print("Consumed benchmark/model-promotion protocols rerun: NO")
    print(f"Acceptance report: {output}")


if __name__ == "__main__":
    main()
