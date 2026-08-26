from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.contracts import ProjectExportRequest, ProjectMeshBuildRequest
from depthwizard.export.project_package import build_project_export, load_project_export
from depthwizard.mesh.project_mesh import build_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.service import app


def _write_source(path: Path) -> None:
    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            (30 + xx * 3) % 255,
            (60 + yy * 4) % 255,
            (90 + xx + yy) % 255,
        ],
        axis=0,
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="uint8",
        crs="EPSG:32632",
        transform=from_origin(500000.0, 5400000.0, 1.0, 1.0),
    ) as dst:
        dst.write(rgb)


def _write_dsm(path: Path) -> None:
    height, width = 32, 40
    yy, xx = np.mgrid[0:height, 0:width]
    dsm = (420.0 + xx * 0.15 + yy * 0.11 + np.sin(xx / 4.0)).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs="EPSG:32632",
        transform=from_origin(500000.0, 5400000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(dsm, 1)


def _project(tmp_path: Path, *, with_mesh: bool = True) -> Path:
    source = tmp_path / "source.tif"
    project_dir = tmp_path / "project"
    products = project_dir / "products"
    products.mkdir(parents=True)
    dsm = products / "dsm.tif"
    _write_source(source)
    _write_dsm(dsm)

    manifest = ProjectManifest.create_or_load(project_dir, source)
    manifest.set_identity(
        source_sha256=sha256_file(source),
        input_kind="georeferenced",
        geometry_config_sha256="geometry-export-test",
        run_config_sha256="run-export-test",
    )
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )
    provenance = project_dir / "provenance.json"
    provenance.write_text(json.dumps({"project_id": manifest.project_id}, sort_keys=True), encoding="utf-8")
    manifest.register_artifact(
        "provenance",
        provenance,
        semantics="project_processing_provenance",
        units=None,
        sha256=sha256_file(provenance),
    )
    if with_mesh:
        build_project_mesh(
            ProjectMeshBuildRequest(project_dir=project_dir, max_finest_samples=128, lod_levels=2)
        )
    return project_dir


def test_project_export_is_deterministic_and_hash_audited(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    request = ProjectExportRequest(project_dir=project_dir)

    first = build_project_export(request)
    second = build_project_export(request)

    assert first.bundle_sha256 == second.bundle_sha256
    assert first.bundle_bytes == second.bundle_bytes
    assert first.include_source is False
    assert first.include_mesh is True
    assert first.include_validation is True
    assert first.bundle_path.is_file()
    assert sha256_file(first.bundle_path) == first.bundle_sha256

    with zipfile.ZipFile(first.bundle_path) as archive:
        names = set(archive.namelist())
        assert "export-manifest.json" in names
        assert "project-manifest.json" in names
        assert "products/dsm.tif" in names
        assert "evidence/provenance.json" in names
        assert "mesh/mesh-manifest.json" in names
        assert "mesh/terrain-lod0.glb" in names
        assert not any(name.startswith("source/") for name in names)
        export_manifest = json.loads(archive.read("export-manifest.json"))
        assert export_manifest["project_id"] == first.project_id
        assert export_manifest["scientific_boundary"].startswith("Export packaging copies")

    assert load_project_export(project_dir).bundle_sha256 == first.bundle_sha256


def test_project_export_source_is_explicit_opt_in(tmp_path: Path) -> None:
    project_dir = _project(tmp_path, with_mesh=False)
    report = build_project_export(
        ProjectExportRequest(
            project_dir=project_dir,
            include_source=True,
            include_mesh=False,
            include_validation=False,
        )
    )
    with zipfile.ZipFile(report.bundle_path) as archive:
        names = set(archive.namelist())
        assert "source/source.tif" in names
        assert not any(name.startswith("mesh/") for name in names)


def test_project_export_rejects_tampered_registered_artifact(tmp_path: Path) -> None:
    project_dir = _project(tmp_path, with_mesh=False)
    manifest = ProjectManifest.load(project_dir)
    dsm = manifest.artifact_path("dsm")
    assert dsm is not None
    with dsm.open("ab") as stream:
        stream.write(b"tamper")

    try:
        build_project_export(ProjectExportRequest(project_dir=project_dir))
    except RuntimeError as exc:
        assert "SHA-256" in str(exc)
    else:
        raise AssertionError("tampered artifact should have failed export packaging")


def test_project_export_service_build_report_and_archive(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    client = TestClient(app)

    built = client.post(
        "/v1/projects/export",
        json={"project_dir": str(project_dir), "include_source": False},
    )
    assert built.status_code == 200
    payload = built.json()
    assert payload["project_id"] == ProjectManifest.load(project_dir).project_id
    assert payload["bundle_sha256"]

    report = client.get("/v1/projects/export", params={"project_dir": str(project_dir)})
    assert report.status_code == 200
    assert report.json()["bundle_sha256"] == payload["bundle_sha256"]

    archive = client.get("/v1/projects/export/archive", params={"project_dir": str(project_dir)})
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/zip")
    assert archive.headers["x-depthwizard-export-sha256"] == payload["bundle_sha256"]
    assert archive.content[:2] == b"PK"
