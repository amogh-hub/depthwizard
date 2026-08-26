from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from depthwizard.contracts import ProjectExportFile, ProjectExportReport, ProjectExportRequest
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file

_VALIDATION_ARTIFACTS = {
    "reference",
    "residual",
    "metrics",
    "validation_report",
}
_MESH_ARTIFACT_PREFIXES = ("terrain_lod", "mesh_manifest")
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class _ExportMember:
    arcname: str
    path: Path
    sha256: str
    semantics: str
    units: str | None


def _is_mesh_artifact(name: str) -> bool:
    return name == "mesh_manifest" or name.startswith("terrain_lod")


def _artifact_arcname(name: str, path: Path) -> str:
    suffix = path.suffix.lower()
    if name == "dsm":
        return "products/dsm.tif"
    if name == "rdsm":
        return "products/rdsm.tif"
    if name == "slope":
        return "products/slope.tif"
    if name == "confidence":
        return "products/confidence.tif"
    if name == "calibration":
        return "evidence/calibration.json"
    if name == "provenance":
        return "evidence/provenance.json"
    if name == "reference":
        return "validation/reference-aligned.tif"
    if name == "residual":
        return "validation/residual.tif"
    if name == "metrics":
        return "validation/metrics.json"
    if name == "validation_report":
        return "validation/validation-report.md"
    if name == "mesh_manifest":
        return "mesh/mesh-manifest.json"
    if name.startswith("terrain_lod"):
        return f"mesh/{path.name}"
    safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in name)
    return f"artifacts/{safe_name}{suffix}"


def _member_from_artifact(name: str, payload: dict[str, object]) -> _ExportMember:
    raw_path = payload.get("path")
    recorded_sha = payload.get("sha256")
    semantics = payload.get("semantics")
    units = payload.get("units")
    if not isinstance(raw_path, str) or not isinstance(recorded_sha, str):
        raise ValueError(f"project artifact '{name}' has incomplete path/hash metadata")
    if not isinstance(semantics, str):
        raise ValueError(f"project artifact '{name}' has no semantics")
    if units is not None and not isinstance(units, str):
        raise ValueError(f"project artifact '{name}' has invalid units metadata")
    path = Path(raw_path)
    if not path.is_file():
        raise FileNotFoundError(f"project artifact '{name}' is missing: {path}")
    actual_sha = sha256_file(path)
    if actual_sha != recorded_sha:
        raise RuntimeError(f"project artifact '{name}' SHA-256 no longer matches the manifest")
    return _ExportMember(
        arcname=_artifact_arcname(name, path),
        path=path,
        sha256=actual_sha,
        semantics=semantics,
        units=units,
    )


def _collect_members(request: ProjectExportRequest, manifest: ProjectManifest) -> list[_ExportMember]:
    members: list[_ExportMember] = []
    for name, payload in sorted(manifest.artifacts.items()):
        if not request.include_mesh and _is_mesh_artifact(name):
            continue
        if not request.include_validation and name in _VALIDATION_ARTIFACTS:
            continue
        members.append(_member_from_artifact(name, payload))

    if request.include_source:
        source = manifest.source_path
        if not source.is_file():
            raise FileNotFoundError(f"project source raster is missing: {source}")
        source_sha = sha256_file(source)
        if manifest.source_sha256 is not None and source_sha != manifest.source_sha256:
            raise RuntimeError("project source raster SHA-256 no longer matches the manifest")
        members.append(
            _ExportMember(
                arcname=f"source/{source.name}",
                path=source,
                sha256=source_sha,
                semantics="original_project_source_imagery",
                units=None,
            )
        )

    arcnames = [member.arcname for member in members]
    if len(arcnames) != len(set(arcnames)):
        raise RuntimeError("project export produced duplicate archive member names")
    return sorted(members, key=lambda item: item.arcname)


def _zip_info(arcname: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def _write_member(archive: zipfile.ZipFile, member: _ExportMember) -> None:
    info = _zip_info(member.arcname)
    with member.path.open("rb") as source, archive.open(info, "w") as destination:
        shutil.copyfileobj(source, destination, length=1024 * 1024)


def _export_manifest_payload(
    *,
    manifest: ProjectManifest,
    project_manifest_sha256: str,
    request: ProjectExportRequest,
    members: list[_ExportMember],
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "project_id": manifest.project_id,
        "project_manifest_sha256": project_manifest_sha256,
        "source_sha256": manifest.source_sha256,
        "include_source": request.include_source,
        "include_mesh": request.include_mesh,
        "include_validation": request.include_validation,
        "scientific_boundary": (
            "Export packaging copies already-persisted project products and evidence only; it does "
            "not recompute reconstruction, calibration, validation, or analytical measurements."
        ),
        "files": [
            {
                "arcname": member.arcname,
                "sha256": member.sha256,
                "bytes": member.path.stat().st_size,
                "semantics": member.semantics,
                "units": member.units,
            }
            for member in members
        ],
    }


def build_project_export(request: ProjectExportRequest) -> ProjectExportReport:
    """Create a deterministic, hash-audited ZIP of persisted project products.

    The bundle is a transport derivative. Every registered artifact is re-hashed before inclusion,
    and packaging never mutates scientific project evidence. Source imagery is opt-in because it can
    be large or redistribution-restricted; its SHA-256 identity remains in the project manifest when
    the bytes are omitted.
    """
    project_dir = request.project_dir
    manifest_path = project_dir / "project-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("project manifest does not exist")
    manifest = ProjectManifest.load(project_dir)
    project_manifest_sha = sha256_file(manifest_path)
    members = _collect_members(request, manifest)
    if not members:
        raise ValueError("project export has no persisted artifacts to package")

    export_dir = project_dir / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = export_dir / f"depthwizard-{manifest.project_id}.zip"
    export_manifest_path = export_dir / "export-manifest.json"
    export_manifest = _export_manifest_payload(
        manifest=manifest,
        project_manifest_sha256=project_manifest_sha,
        request=request,
        members=members,
    )
    export_manifest_text = json.dumps(export_manifest, indent=2, sort_keys=True) + "\n"
    export_manifest_path.write_text(export_manifest_text, encoding="utf-8")

    temporary = bundle_path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(_zip_info("export-manifest.json"), export_manifest_text.encode("utf-8"))
        archive.writestr(_zip_info("project-manifest.json"), manifest_path.read_bytes())
        for member in members:
            _write_member(archive, member)
    temporary.replace(bundle_path)

    files = [
        ProjectExportFile(
            arcname=member.arcname,
            source_path=member.path.resolve(strict=False),
            sha256=member.sha256,
            bytes=member.path.stat().st_size,
            semantics=member.semantics,
            units=member.units,
        )
        for member in members
    ]
    return ProjectExportReport(
        project_id=manifest.project_id,
        bundle_path=bundle_path.resolve(strict=False),
        bundle_sha256=sha256_file(bundle_path),
        bundle_bytes=bundle_path.stat().st_size,
        project_manifest_sha256=project_manifest_sha,
        export_manifest_path=export_manifest_path.resolve(strict=False),
        include_source=request.include_source,
        include_mesh=request.include_mesh,
        include_validation=request.include_validation,
        files=files,
        semantics="deterministic_hash_audited_depthwizard_project_export",
    )


def load_project_export(project_dir: str | Path) -> ProjectExportReport:
    directory = Path(project_dir)
    export_manifest_path = directory / "exports" / "export-manifest.json"
    if not export_manifest_path.is_file():
        raise FileNotFoundError("project export has not been built")
    try:
        payload = json.loads(export_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to read export manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("export manifest root must be an object")

    manifest = ProjectManifest.load(directory)
    bundle_path = directory / "exports" / f"depthwizard-{manifest.project_id}.zip"
    if not bundle_path.is_file():
        raise FileNotFoundError("project export ZIP is missing")
    manifest_sha = sha256_file(manifest.path)
    recorded_manifest_sha = payload.get("project_manifest_sha256")
    if recorded_manifest_sha != manifest_sha:
        raise RuntimeError("project manifest changed after the export was built; rebuild the export")

    request = ProjectExportRequest(
        project_dir=directory,
        include_source=bool(payload.get("include_source", False)),
        include_mesh=bool(payload.get("include_mesh", True)),
        include_validation=bool(payload.get("include_validation", True)),
    )
    members = _collect_members(request, manifest)
    expected = {member.arcname: member for member in members}
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("export manifest files must be an array")
    for item in raw_files:
        if not isinstance(item, dict) or not isinstance(item.get("arcname"), str):
            raise ValueError("export manifest contains an invalid file entry")
        arcname = item["arcname"]
        member = expected.get(arcname)
        if member is None or item.get("sha256") != member.sha256:
            raise RuntimeError(f"export member identity changed after packaging: {arcname}")

    files = [
        ProjectExportFile(
            arcname=member.arcname,
            source_path=member.path.resolve(strict=False),
            sha256=member.sha256,
            bytes=member.path.stat().st_size,
            semantics=member.semantics,
            units=member.units,
        )
        for member in members
    ]
    return ProjectExportReport(
        project_id=manifest.project_id,
        bundle_path=bundle_path.resolve(strict=False),
        bundle_sha256=sha256_file(bundle_path),
        bundle_bytes=bundle_path.stat().st_size,
        project_manifest_sha256=manifest_sha,
        export_manifest_path=export_manifest_path.resolve(strict=False),
        include_source=request.include_source,
        include_mesh=request.include_mesh,
        include_validation=request.include_validation,
        files=files,
        semantics="deterministic_hash_audited_depthwizard_project_export",
    )
