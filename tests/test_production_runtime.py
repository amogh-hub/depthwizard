from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.cancellation import CancellationRequested
from depthwizard.contracts import (
    GroundControlPoint,
    GroundControlPointEvidence,
    ProcessingRequest,
    ProjectRunStatus,
)
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.policy import EstimatorPath, current_production_estimator_decision
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file


def _relative(shape: tuple[int, int]) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    return ((x + 0.5 * y) / max(shape[0] + shape[1], 1)).astype(np.float32)


class FakePrior(GeometryPrior):
    def __init__(self) -> None:
        self.calls = 0

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        self.calls += 1
        shape = (int(rgb_normalized.shape[0]), int(rgb_normalized.shape[1]))
        return GeometryPriorOutput(
            relative_height=_relative(shape),
            confidence=np.full(shape, 0.8, dtype=np.float32),
            model_id="FAKE-PRIOR",
            metadata={"purpose": "unit-test"},
        )


class AlternateFakePrior(FakePrior):
    pass


def _write_rgb(path: Path, *, georeferenced: bool) -> None:
    data = np.zeros((3, 32, 32), dtype=np.uint8)
    y, x = np.mgrid[:32, :32]
    data[0] = (40 + x).astype(np.uint8)
    data[1] = (60 + y).astype(np.uint8)
    data[2] = (80 + (x + y) // 2).astype(np.uint8)
    profile: dict[str, Any] = {
        "driver": "GTiff",
        "height": 32,
        "width": 32,
        "count": 3,
        "dtype": "uint8",
    }
    if georeferenced:
        profile.update(
            crs="EPSG:32643",
            transform=from_origin(500000, 1400000, 1.0, 1.0),
        )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)


def _write_dem(path: Path) -> None:
    truth = 180.0 + 12.0 * _relative((32, 32))
    coarse = truth.reshape(8, 4, 8, 4).mean(axis=(1, 3)).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 4.0, 4.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(coarse, 1)


def _read_float(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            data[data == np.float32(src.nodata)] = np.nan
        return data


def _metric_gcps() -> list[GroundControlPoint]:
    relative = _relative((32, 32))
    transform = from_origin(500000, 1400000, 1.0, 1.0)

    def gcp(row: int, col: int) -> GroundControlPoint:
        x, y = transform * (col + 0.5, row + 0.5)
        return GroundControlPoint(
            x=x,
            y=y,
            elevation_m=75.0 + 20.0 * float(relative[row, col]),
        )

    return [gcp(2, 2), gcp(4, 25), gcp(18, 8), gcp(27, 28)]


def test_current_production_policy_keeps_external_safe_da3() -> None:
    decision = current_production_estimator_decision()
    assert decision.selected_path is EstimatorPath.CALIBRATED_DA3
    assert decision.selected_model_id == "DA3MONO-LARGE"
    assert len(decision.evidence) == 1
    assert decision.evidence[0].independently_evaluated is True
    assert decision.evidence[0].promotion_passed is False
    assert "potsdam-external-v2" in decision.evidence[0].evidence_id


def test_non_georeferenced_project_is_truthful_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=False)
    prior = FakePrior()
    runtime = ProductionElevationRuntime(prior=prior)
    request = ProcessingRequest(source=source, output_dir=project, requested_output="rdsm")

    first = runtime.run(request, job_id="job-a")
    second = runtime.run(request, job_id="job-b")

    assert first.status is ProjectRunStatus.COMPLETE
    assert second.status is ProjectRunStatus.COMPLETE
    assert second.resumed is True
    assert prior.calls == 1
    rdsm = Path(first.artifacts["rdsm"])
    confidence = Path(first.artifacts["confidence"])
    with rasterio.open(rdsm) as src:
        assert src.crs is None
        assert src.transform.is_identity
        assert src.tags()["ELEVATION_UNITS"] == "relative"
    with rasterio.open(confidence) as src:
        assert src.crs is None
        assert src.tags()["CONFIDENCE_SEMANTICS"] == "model_native_not_probability_calibrated"


def test_georeferenced_project_waits_for_evidence_then_resumes_geometry(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    dem = tmp_path / "dem.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    _write_dem(dem)
    prior = FakePrior()
    runtime = ProductionElevationRuntime(prior=prior)

    waiting = runtime.run(ProcessingRequest(source=source, output_dir=project), job_id="job-wait")
    assert waiting.status is ProjectRunStatus.WAITING_FOR_CALIBRATION
    assert prior.calls == 1
    assert "rdsm" in waiting.artifacts
    assert "dsm" not in waiting.artifacts

    completed = runtime.run(
        ProcessingRequest(source=source, output_dir=project, dem_path=dem),
        job_id="job-calibrate",
    )
    assert completed.status is ProjectRunStatus.COMPLETE
    assert completed.resumed is True
    assert prior.calls == 1
    assert {"dsm", "slope", "calibration", "provenance"}.issubset(completed.artifacts)
    prediction = _read_float(Path(completed.artifacts["dsm"]))
    truth = 180.0 + 12.0 * _relative((32, 32))
    assert float(np.nanmean(np.abs(prediction - truth))) < 0.5
    with rasterio.open(completed.artifacts["dsm"]) as src:
        assert src.tags()["ABSOLUTE_ELEVATION_STATUS"] == "vertical_datum_unspecified"
        assert src.tags()["DEPTHWIZARD_PRODUCT"] == "METRIC_DSM_VERTICAL_DATUM_UNSPECIFIED"
    calibration = json.loads(Path(completed.artifacts["calibration"]).read_text(encoding="utf-8"))
    assert calibration["metric_claim"] is True
    assert calibration["absolute_elevation_claim"] is False


def test_waiting_project_rejects_geometry_prior_identity_drift(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    dem = tmp_path / "dem.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    _write_dem(dem)
    ProductionElevationRuntime(prior=FakePrior()).run(
        ProcessingRequest(source=source, output_dir=project)
    )

    with pytest.raises(RuntimeError, match="geometry-affecting configuration"):
        ProductionElevationRuntime(prior=AlternateFakePrior()).run(
            ProcessingRequest(source=source, output_dir=project, dem_path=dem)
        )


def test_gcp_only_metric_project_recovers_absolute_height(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    runtime = ProductionElevationRuntime(prior=FakePrior())

    result = runtime.run(
        ProcessingRequest(
            source=source,
            output_dir=project,
            gcps=_metric_gcps(),
            requested_output="dsm",
        )
    )

    assert result.status is ProjectRunStatus.COMPLETE
    prediction = _read_float(Path(result.artifacts["dsm"]))
    truth = 75.0 + 20.0 * _relative((32, 32))
    assert float(np.nanmean(np.abs(prediction - truth))) < 1e-3
    calibration = json.loads(Path(result.artifacts["calibration"]).read_text(encoding="utf-8"))
    assert calibration["evidence"]["gcp"]["source_evidence"] == {
        "identity_verified": False,
        "kind": "inline_points",
        "sha256": None,
        "source": None,
    }


def test_declared_vertical_reference_produces_datum_resolved_absolute_dsm(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)

    result = ProductionElevationRuntime(prior=FakePrior()).run(
        ProcessingRequest(
            source=source,
            output_dir=project,
            gcps=_metric_gcps(),
            requested_output="dsm",
            vertical_crs="EPSG:5773",
            vertical_datum="EGM96 geoid",
            elevation_reference="orthometric",
        )
    )

    with rasterio.open(result.artifacts["dsm"]) as src:
        tags = src.tags()
        assert tags["ABSOLUTE_ELEVATION_STATUS"] == "datum_resolved"
        assert tags["VERTICAL_CRS"] == "EPSG:5773"
        assert tags["VERTICAL_DATUM"] == "EGM96 geoid"
        assert tags["ELEVATION_REFERENCE"] == "orthometric"
    calibration = json.loads(Path(result.artifacts["calibration"]).read_text(encoding="utf-8"))
    assert calibration["absolute_elevation_claim"] is True
    assert calibration["vertical_reference"]["metadata_source"] == "processing_request"


def test_conflicting_dem_vertical_metadata_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    dem = tmp_path / "dem.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    _write_dem(dem)
    with rasterio.open(dem, "r+") as dataset:
        dataset.update_tags(
            VERTICAL_CRS="EPSG:5773",
            VERTICAL_DATUM="EGM96 geoid",
            ELEVATION_REFERENCE="orthometric",
        )

    with pytest.raises(ValueError, match="vertical datum conflicts"):
        ProductionElevationRuntime(prior=FakePrior()).run(
            ProcessingRequest(
                source=source,
                output_dir=project,
                dem_path=dem,
                requested_output="dsm",
                vertical_crs="EPSG:5773",
                vertical_datum="EGM2008 geoid",
                elevation_reference="orthometric",
            )
        )


def test_runtime_cancellation_is_durable_and_cooperative(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=False)
    calls = 0

    def cancellation_probe() -> bool:
        nonlocal calls
        calls += 1
        return calls >= 3

    with pytest.raises(CancellationRequested):
        ProductionElevationRuntime(prior=FakePrior()).run(
            ProcessingRequest(source=source, output_dir=project, requested_output="rdsm"),
            cancellation_probe=cancellation_probe,
        )

    manifest = json.loads((project / "project-manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert any(stage["status"] == "cancelled" for stage in manifest["stages"].values())


def test_gcp_file_identity_is_verified_and_persisted(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    gcp_file = tmp_path / "control.csv"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    gcp_file.write_text("x,y,elevation_m\n500002.5,1399997.5,76.0\n", encoding="utf-8")
    evidence_sha = sha256_file(gcp_file)

    result = ProductionElevationRuntime(prior=FakePrior()).run(
        ProcessingRequest(
            source=source,
            output_dir=project,
            gcps=_metric_gcps(),
            gcp_evidence=GroundControlPointEvidence(
                source_path=gcp_file,
                sha256=evidence_sha,
            ),
            requested_output="dsm",
        )
    )

    calibration = json.loads(Path(result.artifacts["calibration"]).read_text(encoding="utf-8"))
    source_evidence = calibration["evidence"]["gcp"]["source_evidence"]
    assert source_evidence["kind"] == "csv_file"
    assert source_evidence["identity_verified"] is True
    assert source_evidence["sha256"] == evidence_sha
    assert Path(source_evidence["source"]) == gcp_file.resolve()


def test_gcp_file_mutation_after_inspection_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    gcp_file = tmp_path / "control.csv"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    gcp_file.write_text("x,y,elevation_m\n500002.5,1399997.5,76.0\n", encoding="utf-8")
    inspected_sha = sha256_file(gcp_file)
    gcp_file.write_text("x,y,elevation_m\n500002.5,1399997.5,999.0\n", encoding="utf-8")

    request = ProcessingRequest(
        source=source,
        output_dir=project,
        gcps=_metric_gcps(),
        gcp_evidence=GroundControlPointEvidence(
            source_path=gcp_file,
            sha256=inspected_sha,
        ),
        requested_output="dsm",
    )
    with pytest.raises(RuntimeError, match="GCP evidence file bytes changed after inspection"):
        ProductionElevationRuntime(prior=FakePrior()).run(request)


def test_completed_project_rejects_geometry_configuration_drift(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=False)
    runtime = ProductionElevationRuntime(prior=FakePrior())
    runtime.run(ProcessingRequest(source=source, output_dir=project, tile_size=256))

    with pytest.raises(
        RuntimeError,
        match="geometry-affecting configuration or estimator policy changed",
    ):
        runtime.run(ProcessingRequest(source=source, output_dir=project, tile_size=512))
