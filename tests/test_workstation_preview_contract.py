from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.service import app


def _write_rgb(path: Path) -> None:
    y, x = np.mgrid[:32, :32]
    data = np.stack(
        [
            30 + 3 * x,
            45 + 2 * y,
            65 + x + y,
        ],
        axis=0,
    ).clip(0, 255).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def _write_surface(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:32, :32]
    values = (100.0 + 0.5 * x + 0.25 * y).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=32,
        width=32,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)


def test_workstation_preview_routes_exist_with_deterministic_missing_project_detail(tmp_path: Path) -> None:
    client = TestClient(app)
    missing = tmp_path / "missing-project"
    for route, params in (
        ("/v1/projects/preview", {"project_dir": str(missing), "layer": "optical", "max_side": 64}),
        ("/v1/projects/preview/legend", {"project_dir": str(missing), "layer": "dsm"}),
    ):
        response = client.get(route, params=params)
        assert response.status_code == 404
        assert response.json()["detail"] == "project manifest does not exist"


def test_reopened_project_can_render_optical_dsm_and_derived_previews(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    _write_rgb(source)
    _write_surface(dsm)

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )

    client = TestClient(app)
    for layer in ("optical", "dsm", "hillshade", "contours"):
        response = client.get(
            "/v1/projects/preview",
            params={"project_dir": str(project), "layer": layer, "max_side": 256},
        )
        assert response.status_code == 200, (layer, response.text)
        assert response.headers["content-type"] == "image/png"
        assert response.content.startswith(b"\x89PNG\r\n\x1a\n")

    legend = client.get(
        "/v1/projects/preview/legend",
        params={"project_dir": str(project), "layer": "dsm"},
    )
    assert legend.status_code == 200
    payload = legend.json()
    assert payload["available"] is True
    assert payload["units"] == "m"
    assert payload["minimum"] < payload["midpoint"] < payload["maximum"]


def test_missing_reopened_source_fails_truthfully_without_fabricating_optical_preview(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    _write_rgb(source)
    _write_surface(dsm)

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )
    source.unlink()

    client = TestClient(app)
    optical = client.get(
        "/v1/projects/preview",
        params={"project_dir": str(project), "layer": "optical", "max_side": 256},
    )
    assert optical.status_code == 404
    assert "source" in optical.json()["detail"].lower() or "no such file" in optical.json()["detail"].lower()

    dsm_response = client.get(
        "/v1/projects/preview",
        params={"project_dir": str(project), "layer": "dsm", "max_side": 256},
    )
    assert dsm_response.status_code == 200
