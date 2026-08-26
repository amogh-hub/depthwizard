from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import trimesh

from depthwizard.contracts import NormalizedPoint, ProjectProbeRequest, ProjectProfileRequest
from depthwizard.evaluation.project_analysis import probe_project, sample_project_profile
from depthwizard.export.project_package import load_project_export
from depthwizard.mesh.project_mesh import load_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.raster_preview import PreviewLayer, render_project_layer_preview

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-2-ortholoc"
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
    return {"layer": layer, "bytes": len(payload), "sha256": __import__("hashlib").sha256(payload).hexdigest()}


def main() -> None:
    """Close RT3 workstation integration without rerunning reconstruction or benchmark evidence."""
    manifest_path = PROJECT_DIR / "project-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "RT3 workstation acceptance requires the accepted RT2/RT3 local project at "
            f"{PROJECT_DIR}. Run the earlier acceptance paths instead of fabricating substitute data."
        )

    print("DepthWizard Release Train 3 workstation acceptance")
    print("Purpose: integrated 3D analyst/runtime closure; no reconstruction or model-promotion rerun")

    manifest = ProjectManifest.load(PROJECT_DIR)
    mesh = load_project_mesh(PROJECT_DIR)
    export = load_project_export(PROJECT_DIR)
    mesh_stage = manifest.stages.get("mesh", {})
    mesh_details = mesh_stage.get("details", {}) if isinstance(mesh_stage, dict) else {}
    if not isinstance(mesh_details, dict) or mesh_details.get("reference_data_used") is not False:
        raise RuntimeError("mesh stage no longer preserves reference_data_used=false")
    if mesh.surface_product != "dsm" or mesh.vertical_units != "m" or mesh.horizontal_units != "m":
        raise RuntimeError("accepted workstation project is no longer a metric DSM terrain")

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

    profile = sample_project_profile(
        ProjectProfileRequest(
            project_dir=PROJECT_DIR,
            start=NormalizedPoint(x=0.18, y=0.22),
            end=NormalizedPoint(x=0.82, y=0.78),
            samples=96,
        )
    )
    available_surface_samples = sum(1 for sample in profile.samples if sample.surface.available)
    if profile.horizontal_distance_m is None or profile.horizontal_distance_m <= 0:
        raise RuntimeError("metric profile did not preserve a physical horizontal distance")
    if available_surface_samples < 2:
        raise RuntimeError("profile has insufficient raster-backed surface samples")

    if not export.bundle_path.is_file() or sha256_file(export.bundle_path) != export.bundle_sha256:
        raise RuntimeError("previously accepted project export bundle no longer matches its recorded hash")
    if export.include_source:
        raise RuntimeError("default accepted export unexpectedly includes original source imagery bytes")

    acceptance = {
        "schema_version": 1,
        "status": "PASS_RT3_WORKSTATION_PATH",
        "purpose": (
            "Release Train 3 integrated analyst workstation acceptance; visualization/runtime closure "
            "only and not new elevation-accuracy or model-promotion evidence"
        ),
        "project_id": manifest.project_id,
        "project_manifest_sha256": sha256_file(manifest_path),
        "surface_product": mesh.surface_product,
        "surface_sha256": mesh.surface_sha256,
        "reference_data_used_for_mesh_geometry": False,
        "mesh_lods": lod_contracts,
        "preview_layers": preview_contracts,
        "probe": {
            "point": centre.point.model_dump(mode="json"),
            "pixel_col": centre.pixel_col,
            "pixel_row": centre.pixel_row,
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
            "horizontal_distance_m": profile.horizontal_distance_m,
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
            "Terrain LODs, UV analytical overlays, camera flythroughs, renderer LOD selection and "
            "3D analysis graphics are display derivatives. Numeric probe/profile values remain "
            "raster-backed. Export remains a transport derivative. No sealed benchmark was rerun."
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
    print(
        f"Metric profile: {profile.sample_count} samples · "
        f"{profile.horizontal_distance_m:.2f} m · {available_surface_samples} available surface samples"
    )
    print(f"Accepted export SHA-256 preserved: {export.bundle_sha256}")
    print("Reference data used for mesh geometry: NO")
    print("Consumed benchmark/model-promotion protocols rerun: NO")
    print(f"Acceptance report: {output}")


if __name__ == "__main__":
    main()
