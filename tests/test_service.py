from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.contracts import ProjectRunStatus
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file
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


def test_gcp_inspect_endpoint_returns_typed_metric_evidence(tmp_path: Path) -> None:
    path = tmp_path / "gcps.csv"
    path.write_text(
        "x,y,elevation_m,weight\n"
        "500000,1400000,101,1\n"
        "500010,1400010,109,2\n",
        encoding="utf-8",
    )
    client = TestClient(app)
    response = client.post("/v1/calibration/gcps/inspect", json={"path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["point_count"] == 2
    assert payload["points"][1]["weight"] == 2.0
    assert len(payload["sha256"]) == 64


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


def _write_rgb(path: Path) -> None:
    data = np.zeros((3, 32, 32), dtype=np.uint8)
    y, x = np.mgrid[:32, :32]
    data[0] = (20 + x).astype(np.uint8)
    data[1] = (40 + y).astype(np.uint8)
    data[2] = (60 + (x + y) // 2).astype(np.uint8)
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


def _write_surface(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def test_validation_and_preview_endpoints_use_persisted_project_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    reference = tmp_path / "reference.tif"
    _write_rgb(source)
    y, x = np.mgrid[:32, :32]
    truth = (120.0 + 0.5 * x + 0.25 * y).astype(np.float32)
    _write_surface(dsm, truth + 1.5)
    _write_surface(reference, truth)

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.mark_status(ProjectRunStatus.COMPLETE)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )
    manifest.record_stage(
        ProcessingStage.CALIBRATION,
        status="completed",
        details={"evidence": {"dem": {"sha256": "different-calibration-file"}}},
    )

    client = TestClient(app)
    validated = client.post(
        "/v1/projects/validate",
        json={"project_dir": str(project), "reference_path": str(reference)},
    )
    assert validated.status_code == 200
    payload = validated.json()
    assert abs(payload["elevation"]["rmse_m"] - 1.5) < 1e-4
    assert payload["independence_check"] == "different_sha_from_calibration_dem"

    reloaded = client.get("/v1/projects/validation", params={"project_dir": str(project)})
    assert reloaded.status_code == 200
    assert reloaded.json()["reference_sha256"] == payload["reference_sha256"]

    for layer in ("residual", "hillshade", "contours"):
        preview = client.get(
            "/v1/projects/preview",
            params={"project_dir": str(project), "layer": layer, "max_side": 256},
        )
        assert preview.status_code == 200
        assert preview.headers["content-type"] == "image/png"
        assert preview.content.startswith(b"\x89PNG\r\n\x1a\n")

    dsm_legend = client.get(
        "/v1/projects/preview/legend",
        params={"project_dir": str(project), "layer": "dsm"},
    )
    assert dsm_legend.status_code == 200
    dsm_payload = dsm_legend.json()
    assert dsm_payload["available"] is True
    assert dsm_payload["units"] == "m"
    assert dsm_payload["minimum"] < dsm_payload["midpoint"] < dsm_payload["maximum"]
    assert dsm_payload["semantics"] == "display_p02_p98_range_from_persisted_project_raster"

    residual_legend = client.get(
        "/v1/projects/preview/legend",
        params={"project_dir": str(project), "layer": "residual"},
    )
    assert residual_legend.status_code == 200
    residual_payload = residual_legend.json()
    assert residual_payload["ramp"] == "diverging"
    assert residual_payload["minimum"] == -residual_payload["maximum"]
    assert residual_payload["midpoint"] == 0.0

    optical_legend = client.get(
        "/v1/projects/preview/legend",
        params={"project_dir": str(project), "layer": "optical"},
    )
    assert optical_legend.status_code == 200
    assert optical_legend.json()["available"] is False


def test_structure_height_endpoint_is_metric_and_raster_backed(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    _write_rgb(source)
    values = np.full((32, 32), 100.0, dtype=np.float32)
    values[10:22, 10:22] = 114.0
    _write_surface(dsm, values)
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )

    client = TestClient(app)
    response = client.post(
        "/v1/projects/structure-height",
        json={
            "project_dir": str(project),
            "polygon": [
                {"x": 10 / 31, "y": 10 / 31},
                {"x": 21 / 31, "y": 10 / 31},
                {"x": 21 / 31, "y": 21 / 31},
                {"x": 10 / 31, "y": 21 / 31},
            ],
            "ring_pixels": 5,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert abs(payload["structure_height_m"] - 14.0) < 1e-6
    assert payload["structure_pixels"] > 100
    assert payload["ground_pixels"] >= 8
