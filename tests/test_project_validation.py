from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.contracts import ProjectRunStatus, ReferenceValidationRequest
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file


def _write_surface(path: Path, values: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _project_with_dsm(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    project = tmp_path / "project"
    source = tmp_path / "rgb.tif"
    dsm = project / "products" / "dsm.tif"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.touch()
    dsm.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:32, :32]
    truth = 100.0 + 0.5 * x + 0.25 * y
    _write_surface(dsm, truth + 2.0)

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
        details={"evidence": {"dem": {"sha256": "different-calibration-evidence"}}},
    )
    return project, dsm, truth.astype(np.float32)


def test_project_reference_validation_emits_durable_products(tmp_path: Path) -> None:
    project, _dsm, truth = _project_with_dsm(tmp_path)
    reference = tmp_path / "reference.tif"
    _write_surface(reference, truth)

    report = validate_project_reference(
        ReferenceValidationRequest(project_dir=project, reference_path=reference)
    )

    assert report.valid_pixels == 32 * 32
    assert report.coverage_fraction == pytest.approx(1.0)
    assert report.elevation.rmse_m == pytest.approx(2.0, abs=1e-5)
    assert report.elevation.mae_m == pytest.approx(2.0, abs=1e-5)
    assert report.elevation.mean_bias_m == pytest.approx(2.0, abs=1e-5)
    assert report.elevation.pearson_r == pytest.approx(1.0, abs=1e-6)
    assert report.slope.rmse_degrees == pytest.approx(0.0, abs=1e-4)
    assert report.reliability.available is False
    assert report.independence_check == "different_sha_from_calibration_dem"

    manifest = ProjectManifest.load(project)
    assert manifest.stage_completed(ProcessingStage.VALIDATION)
    assert {"reference", "residual", "metrics", "validation_report"}.issubset(manifest.artifacts)
    assert not manifest.errors
    residual_path = manifest.artifact_path("residual")
    assert residual_path is not None
    with rasterio.open(residual_path) as src:
        residual = src.read(1)
        assert np.allclose(residual, 2.0)
        assert src.tags()["RESIDUAL_SIGN"] == "prediction_minus_reference"

    # Repeating the same reference is resumable and returns the persisted report instead of
    # rewriting scientific evidence.
    repeated = validate_project_reference(
        ReferenceValidationRequest(project_dir=project, reference_path=reference)
    )
    assert repeated.reference_sha256 == report.reference_sha256
    assert repeated.elevation.rmse_m == report.elevation.rmse_m


def test_project_reference_validation_rejects_calibration_dem_reuse(tmp_path: Path) -> None:
    project, _dsm, truth = _project_with_dsm(tmp_path)
    reference = tmp_path / "same-as-calibration.tif"
    _write_surface(reference, truth)
    manifest = ProjectManifest.load(project)
    manifest.record_stage(
        ProcessingStage.CALIBRATION,
        status="completed",
        details={"evidence": {"dem": {"sha256": sha256_file(reference)}}},
    )

    with pytest.raises(ValueError, match="byte-identical to the calibration DEM"):
        validate_project_reference(
            ReferenceValidationRequest(project_dir=project, reference_path=reference)
        )


def test_project_reference_validation_preserves_first_reference(tmp_path: Path) -> None:
    project, _dsm, truth = _project_with_dsm(tmp_path)
    reference_a = tmp_path / "reference-a.tif"
    reference_b = tmp_path / "reference-b.tif"
    _write_surface(reference_a, truth)
    _write_surface(reference_b, truth + 1.0)
    validate_project_reference(
        ReferenceValidationRequest(project_dir=project, reference_path=reference_a)
    )

    with pytest.raises(RuntimeError, match="preserve the existing evidence"):
        validate_project_reference(
            ReferenceValidationRequest(project_dir=project, reference_path=reference_b)
        )
