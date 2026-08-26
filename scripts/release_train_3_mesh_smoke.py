from __future__ import annotations

import json
from pathlib import Path

from depthwizard.contracts import ProjectMeshBuildRequest
from depthwizard.mesh.project_mesh import build_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-2-ortholoc"


def main() -> None:
    """Exercise RT3 against the already-accepted RT2 project without rerunning model inference."""
    manifest_path = PROJECT_DIR / "project-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "RT3 mesh smoke requires the local RT2 acceptance project at "
            f"{PROJECT_DIR}. Run the RT2 acceptance first rather than fabricating substitute evidence."
        )

    print("DepthWizard Release Train 3 mesh acceptance")
    print("Purpose: persistent project-derived 3D product acceptance; no reconstruction rerun")
    before = ProjectManifest.load(PROJECT_DIR)
    if before.artifact_path("dsm") is None and before.artifact_path("rdsm") is None:
        raise RuntimeError("RT2 project has no persisted surface for 3D terrain generation")

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
        "schema_version": 1,
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
        "valid_pixels": report.valid_pixels,
        "minimum_elevation": report.minimum_elevation,
        "maximum_elevation": report.maximum_elevation,
        "relief": report.relief,
        "reference_data_used": details.get("reference_data_used"),
        "lods": lod_evidence,
        "mesh_manifest_sha256": sha256_file(report.mesh_manifest_path),
        "scientific_boundary": (
            "GLB geometry derives from the persisted DSM/rDSM and source RGB only. Analyst values "
            "remain raster-backed; validation reference products are not geometry inputs."
        ),
    }
    output = PROJECT_DIR / "mesh" / "release-train-3-acceptance.json"
    output.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")

    print("DepthWizard Release Train 3 persistent mesh path: PASS")
    print(f"Surface: {report.surface_product} · {report.horizontal_units} XY · {report.vertical_units} Z")
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
