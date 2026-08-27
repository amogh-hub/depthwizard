from __future__ import annotations

import json
import os

import numpy as np

from depthwizard.contracts import ProjectMeshBuildRequest
from depthwizard.mesh.project_mesh import build_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from scripts.release_train_3_spatial_foundation_smoke import (
    PROJECT_DIR,
    REPORT_PATH as SPATIAL_FOUNDATION_REPORT,
    _direct_affine_spacing_m,
)


def main() -> None:
    """Exercise RT3 mesh generation on the corrected OrthoLoC spatial-foundation project."""
    os.environ["DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE"] = "1"
    manifest_path = PROJECT_DIR / "project-manifest.json"
    if not manifest_path.is_file() or not SPATIAL_FOUNDATION_REPORT.is_file():
        raise RuntimeError(
            "RT3 mesh smoke requires the corrected spatial-foundation acceptance at "
            f"{PROJECT_DIR}. Run make release-train-3-spatial-foundation-smoke first."
        )

    print("DepthWizard Release Train 3 mesh acceptance")
    print("Purpose: persistent project-derived 3D product acceptance; no reconstruction rerun")
    before = ProjectManifest.load(PROJECT_DIR)
    surface_path = before.artifact_path("dsm") or before.artifact_path("rdsm")
    if surface_path is None or not surface_path.is_file():
        raise RuntimeError("RT3 spatial-foundation project has no persisted surface for terrain generation")
    direct_gsd = _direct_affine_spacing_m(surface_path)

    report = build_project_mesh(
        ProjectMeshBuildRequest(
            project_dir=PROJECT_DIR,
            max_finest_samples=512,
            lod_levels=4,
        )
    )
    after = ProjectManifest.load(PROJECT_DIR)
    mesh_stage = after.stages.get("mesh", {})
    if mesh_stage.get("status") != "completed":
        raise RuntimeError("project manifest did not persist a completed mesh stage")
    details = mesh_stage.get("details", {})
    if details.get("reference_data_used") is not False:
        raise RuntimeError("mesh stage did not preserve the no-reference-data scientific boundary")
    if details.get("metric_horizontal_scale_trusted") is not True:
        raise RuntimeError("OrthoLoC spatial-foundation mesh did not preserve trusted metric-affine XY")
    if report.horizontal_units != "m" or report.vertical_units != "m":
        raise RuntimeError("corrected OrthoLoC DSM terrain must preserve metre XY and metre Z units")
    if not np.allclose((report.gsd_x, report.gsd_y), direct_gsd, rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            "mesh GSD disagrees with direct OrthoLoC affine metric spacing: "
            f"mesh={(report.gsd_x, report.gsd_y)} affine={direct_gsd}"
        )
    if after.artifact_path("mesh_manifest") != report.mesh_manifest_path:
        raise RuntimeError("project manifest mesh-manifest path does not match the mesh report")

    lod_evidence: list[dict[str, object]] = []
    for lod in report.lods:
        if not lod.path.is_file():
            raise RuntimeError(f"terrain LOD is missing: {lod.path}")
        actual_sha = sha256_file(lod.path)
        if actual_sha != lod.sha256:
            raise RuntimeError(f"terrain LOD SHA-256 mismatch: {lod.path.name}")
        if lod.path.read_bytes()[:4] != b"glTF":
            raise RuntimeError(f"terrain LOD is not a binary GLB payload: {lod.path.name}")
        artifact = after.artifacts.get(f"terrain_lod{lod.level}")
        if artifact is None or artifact.get("sha256") != lod.sha256:
            raise RuntimeError(f"terrain LOD {lod.level} is not correctly registered in the project manifest")
        lod_evidence.append(
            {
                "level": lod.level,
                "stride": lod.stride,
                "sha256": lod.sha256,
                "vertices": lod.vertices,
                "faces": lod.faces,
                "width_samples": lod.width_samples,
                "height_samples": lod.height_samples,
                "bytes": lod.path.stat().st_size,
            }
        )

    acceptance = {
        "schema_version": 2,
        "status": "PASS_PROJECT_MESH_PATH",
        "purpose": (
            "Release Train 3 project-derived mesh integration acceptance; visualization derivative "
            "only and not a new elevation-accuracy benchmark"
        ),
        "project_id": report.project_id,
        "surface_product": report.surface_product,
        "surface_sha256": report.surface_sha256,
        "texture_sha256": report.texture_sha256,
        "build_config_sha256": report.build_config_sha256,
        "horizontal_units": report.horizontal_units,
        "vertical_units": report.vertical_units,
        "gsd_x": report.gsd_x,
        "gsd_y": report.gsd_y,
        "direct_affine_gsd_x_m": direct_gsd[0],
        "direct_affine_gsd_y_m": direct_gsd[1],
        "spatial_scale_policy": "ortholoc_dataset_local_metric_affine",
        "valid_pixels": report.valid_pixels,
        "minimum_elevation": report.minimum_elevation,
        "maximum_elevation": report.maximum_elevation,
        "relief": report.relief,
        "reference_data_used": details.get("reference_data_used"),
        "lods": lod_evidence,
        "mesh_manifest_sha256": sha256_file(report.mesh_manifest_path),
        "scientific_boundary": (
            "GLB geometry derives from the persisted DSM and source RGB only. Horizontal geometry "
            "uses the explicit OrthoLoC dataset-local metric affine contract and is checked directly "
            "against the persisted raster transform. Validation reference products are not geometry inputs."
        ),
    }
    output = PROJECT_DIR / "mesh" / "release-train-3-acceptance.json"
    output.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")

    print("DepthWizard Release Train 3 persistent mesh path: PASS")
    print(f"Surface: {report.surface_product} · {report.horizontal_units} XY · {report.vertical_units} Z")
    print(f"Affine GSD: {report.gsd_x:.6f} m/px × {report.gsd_y:.6f} m/px")
    print(f"Valid pixels: {report.valid_pixels:,}")
    print(f"Relief: {report.relief:.3f} {report.vertical_units}")
    for lod in report.lods:
        print(
            f"LOD {lod.level}: stride {lod.stride} · {lod.vertices:,} vertices · "
            f"{lod.faces:,} faces · {lod.path.stat().st_size / (1024 * 1024):.2f} MiB"
        )
    print("Reference data used for mesh geometry: NO")
    print(f"Acceptance report: {output}")


if __name__ == "__main__":
    main()
