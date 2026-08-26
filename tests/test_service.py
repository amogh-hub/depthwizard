from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.service import app


def test_inspect_endpoint_reports_georeferenced_raster(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)

    client = TestClient(app)
    response = client.post("/v1/inspect", json={"path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["crs"] == "EPSG:32643"
    # Reported GSD is physical ground distance, so projected CRS scale factor can differ slightly
    # from the affine coordinate-unit spacing.
    assert abs(payload["ground_sample_distance_x"] - 1.0) < 0.01


def test_project_submission_rejects_missing_source_before_background_execution(
    tmp_path: Path,
) -> None:
    client = TestClient(app)
    response = client.post(
        "/v1/projects",
        json={
            "source": str(tmp_path / "missing.tif"),
            "output_dir": str(tmp_path / "project"),
        },
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "source raster does not exist"


def test_project_manifest_endpoint_returns_exact_durable_state(tmp_path: Path) -> None:
    source = tmp_path / "source.tif"
    source.touch()
    project = tmp_path / "project"
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.add_warning("test-warning")

    client = TestClient(app)
    response = client.get("/v1/projects/manifest", params={"project_dir": str(project)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["schema_version"] == 2
    assert payload["project_id"] == manifest.project_id
    assert payload["warnings"] == ["test-warning"]


def test_unknown_project_job_returns_404() -> None:
    client = TestClient(app)
    response = client.get("/v1/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["detail"] == "unknown DepthWizard job id"
