from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.contracts import ProjectMeshBuildRequest
from depthwizard.mesh.project_mesh import build_project_mesh, load_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.service import app


def _write_source(path: Path) -> None:
    height, width = 48, 64
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            (40 + xx * 2) % 255,
            (70 + yy * 3) % 255,
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
    height, width = 48, 64
    yy, xx = np.mgrid[0:height, 0:width]
    dsm = (510.0 + xx * 0.12 + yy * 0.08 + np.sin(xx / 5.0)).astype(np.float32)
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


def _project(tmp_path: Path) -> Path:
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
        geometry_config_sha256="geometry-test",
        run_config_sha256="run-test",
    )
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )
    return project_dir


def test_project_mesh_build_is_persisted_hashed_and_resumable(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    request = ProjectMeshBuildRequest(project_dir=project_dir, max_finest_samples=128, lod_levels=3)

    first = build_project_mesh(request)
    second = build_project_mesh(request)

    assert first == second
    assert first.surface_product == "dsm"
    assert first.horizontal_units == "m"
    assert first.vertical_units == "m"
    assert first.valid_pixels == 48 * 64
    assert len(first.lods) == 3
    assert [lod.stride for lod in first.lods] == [1, 2, 4]
    assert all(lod.path.is_file() for lod in first.lods)
    assert all(sha256_file(lod.path) == lod.sha256 for lod in first.lods)

    persisted = load_project_mesh(project_dir)
    assert persisted == first
    manifest = ProjectManifest.load(project_dir)
    assert manifest.stage_completed.__self__ is manifest
    assert manifest.stages["mesh"]["status"] == "completed"
    assert manifest.stages["mesh"]["details"]["reference_data_used"] is False
    assert manifest.artifacts["mesh_manifest"]["sha256"] == sha256_file(first.mesh_manifest_path)
    assert manifest.artifacts["terrain_lod0"]["sha256"] == first.lods[0].sha256


def test_project_mesh_service_build_report_and_glb(tmp_path: Path) -> None:
    project_dir = _project(tmp_path)
    client = TestClient(app)

    built = client.post(
        "/v1/projects/mesh",
        json={"project_dir": str(project_dir), "max_finest_samples": 128, "lod_levels": 2},
    )
    assert built.status_code == 200
    payload = built.json()
    assert payload["surface_product"] == "dsm"
    assert len(payload["lods"]) == 2

    report = client.get("/v1/projects/mesh", params={"project_dir": str(project_dir)})
    assert report.status_code == 200
    assert report.json()["build_config_sha256"] == payload["build_config_sha256"]

    glb = client.get("/v1/projects/mesh/lod/0", params={"project_dir": str(project_dir)})
    assert glb.status_code == 200
    assert glb.headers["content-type"].startswith("model/gltf-binary")
    assert glb.headers["x-depthwizard-mesh-sha256"] == payload["lods"][0]["sha256"]
    assert glb.content[:4] == b"glTF"
