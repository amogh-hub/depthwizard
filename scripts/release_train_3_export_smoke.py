from __future__ import annotations

import json
import zipfile

from depthwizard.contracts import ProjectExportRequest
from depthwizard.export.project_package import build_project_export, load_project_export
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from scripts.release_train_3_spatial_foundation_smoke import PROJECT_DIR
from scripts.release_train_3_spatial_foundation_smoke import (
    REPORT_PATH as SPATIAL_FOUNDATION_REPORT,
)


def main() -> None:
    """Accept RT3 export packaging against the corrected spatial-foundation project without inference."""
    manifest_path = PROJECT_DIR / "project-manifest.json"
    if not manifest_path.is_file() or not SPATIAL_FOUNDATION_REPORT.is_file():
        raise RuntimeError(
            "RT3 export smoke requires the corrected spatial-foundation acceptance at "
            f"{PROJECT_DIR}. Run make release-train-3-spatial-foundation-smoke first."
        )
    manifest = ProjectManifest.load(PROJECT_DIR)
    if manifest.artifact_path("dsm") is None:
        raise RuntimeError("corrected RT3 acceptance project has no persisted metric DSM")
    if manifest.artifact_path("mesh_manifest") is None:
        raise RuntimeError("run the RT3 mesh acceptance before the export acceptance")

    print("DepthWizard Release Train 3 export acceptance")
    print("Purpose: deterministic transport/export packaging; no reconstruction or validation rerun")

    request = ProjectExportRequest(
        project_dir=PROJECT_DIR,
        include_source=False,
        include_mesh=True,
        include_validation=True,
    )
    first = build_project_export(request)
    second = build_project_export(request)
    if first.bundle_sha256 != second.bundle_sha256:
        raise RuntimeError("repeated unchanged project export is not byte-deterministic")
    if sha256_file(first.bundle_path) != first.bundle_sha256:
        raise RuntimeError("export ZIP SHA-256 does not match the returned report")

    loaded = load_project_export(PROJECT_DIR)
    if loaded.bundle_sha256 != first.bundle_sha256:
        raise RuntimeError("persisted export report does not reproduce the accepted bundle identity")

    with zipfile.ZipFile(first.bundle_path) as archive:
        names = set(archive.namelist())
        if "export-manifest.json" not in names or "project-manifest.json" not in names:
            raise RuntimeError("export ZIP is missing its audit manifests")
        if "products/dsm.tif" not in names:
            raise RuntimeError("export ZIP is missing the corrected primary metric DSM")
        if "mesh/mesh-manifest.json" not in names or not any(
            name.startswith("mesh/terrain-lod") and name.endswith(".glb") for name in names
        ):
            raise RuntimeError("export ZIP is missing persistent terrain products")
        if any(name.startswith("source/") for name in names):
            raise RuntimeError("source imagery was included despite explicit include_source=False")
        packaged_manifest = json.loads(archive.read("export-manifest.json"))
        if packaged_manifest.get("project_manifest_sha256") != first.project_manifest_sha256:
            raise RuntimeError("packaged export manifest does not preserve project-manifest identity")
        if packaged_manifest.get("include_source") is not False:
            raise RuntimeError("packaged export manifest lost source-omission semantics")

    acceptance = {
        "schema_version": 2,
        "status": "PASS_PROJECT_EXPORT_PATH",
        "purpose": (
            "Release Train 3 deterministic scientific export integration acceptance over the "
            "corrected OrthoLoC spatial-foundation project; transport derivative only"
        ),
        "project_id": first.project_id,
        "bundle_path": str(first.bundle_path),
        "bundle_sha256": first.bundle_sha256,
        "bundle_bytes": first.bundle_bytes,
        "project_manifest_sha256": first.project_manifest_sha256,
        "spatial_scale_policy": "ortholoc_dataset_local_metric_affine",
        "include_source": first.include_source,
        "include_mesh": first.include_mesh,
        "include_validation": first.include_validation,
        "member_count": len(first.files) + 2,
        "files": [
            {
                "arcname": item.arcname,
                "sha256": item.sha256,
                "bytes": item.bytes,
                "semantics": item.semantics,
                "units": item.units,
            }
            for item in first.files
        ],
        "scientific_boundary": (
            "The export copies hash-verified persisted artifacts from the corrected spatial-foundation "
            "project. It does not rerun or alter reconstruction, calibration, validation, mesh geometry, "
            "or analytical values."
        ),
    }
    output = PROJECT_DIR / "exports" / "release-train-3-export-acceptance.json"
    output.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")

    print("DepthWizard Release Train 3 deterministic export path: PASS")
    print(f"Project: {first.project_id}")
    print(f"Bundle: {first.bundle_bytes / (1024 * 1024):.2f} MiB")
    print(f"Bundle SHA-256: {first.bundle_sha256}")
    print(f"Packaged registered artifacts: {len(first.files)}")
    print("Source imagery included: NO")
    print(f"Acceptance report: {output}")


if __name__ == "__main__":
    main()
