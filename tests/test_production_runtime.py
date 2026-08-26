from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.contracts import GroundControlPoint, ProcessingRequest, ProjectRunStatus
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.policy import EstimatorPath, current_production_estimator_decision
from depthwizard.pipeline.runtime import ProductionElevationRuntime


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


def test_gcp_only_metric_project_recovers_absolute_height(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source, georeferenced=True)
    prior = FakePrior()
    runtime = ProductionElevationRuntime(prior=prior)
    relative = _relative((32, 32))
    transform = from_origin(500000, 1400000, 1.0, 1.0)

    def gcp(row: int, col: int) -> GroundControlPoint:
        x, y = transform * (col + 0.5, row + 0.5)
        return GroundControlPoint(
            x=x,
            y=y,
            elevation_m=75.0 + 20.0 * float(relative[row, col]),
        )

    gcps = [gcp(2, 2), gcp(4, 25), gcp(18, 8), gcp(27, 28)]
    result = runtime.run(
        ProcessingRequest(source=source, output_dir=project, gcps=gcps, requested_output="dsm")
    )

    assert result.status is ProjectRunStatus.COMPLETE
    prediction = _read_float(Path(result.artifacts["dsm"]))
    truth = 75.0 + 20.0 * relative
    assert float(np.nanmean(np.abs(prediction - truth))) < 1e-3


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
