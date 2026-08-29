from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from depthwizard.contracts import ProcessingRequest, ProjectRunStatus
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from scripts.verify_problem_statement_inputs import run_input_format_qualification


class _DeterministicFormatPrior(GeometryPrior):
    """Fast format-contract surrogate; production DA3 is covered by packaged acceptance."""

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        height, width = int(rgb_normalized.shape[0]), int(rgb_normalized.shape[1])
        rows, cols = np.mgrid[:height, :width]
        denominator = max(height + width - 2, 1)
        surface = ((rows + cols) / denominator).astype(np.float32)
        return GeometryPriorOutput(
            relative_height=surface,
            confidence=None,
            model_id="FORMAT-CONTRACT-SURROGATE-NOT-PRODUCTION",
            metadata={"purpose": "literal_png_jpg_rdsm_contract"},
        )


def test_problem_statement_input_formats_are_literal_and_truthful(tmp_path: Path) -> None:
    report = run_input_format_qualification(tmp_path / "input-contract")

    assert report["status"] == "PASS_SIH26175_INPUT_FORMAT_CONTRACT"
    formats = report["formats"]
    assert isinstance(formats, dict)

    png = formats["png"]
    jpg = formats["jpg"]
    tiff = formats["tiff"]
    assert isinstance(png, dict)
    assert isinstance(jpg, dict)
    assert isinstance(tiff, dict)

    for nongeo in (png, jpg):
        assert nongeo["status"] == "PASS"
        assert nongeo["crs"] is None
        assert nongeo["ground_sample_distance_x"] is None
        assert nongeo["ground_sample_distance_y"] is None
        assert nongeo["elevation_contract"] == "relative_only_no_metric_claim"

    assert tiff["status"] == "PASS"
    assert tiff["crs"] == "EPSG:32643"
    assert tiff["ground_sample_distance_x"] == pytest.approx(2.0, abs=0.01)
    assert tiff["ground_sample_distance_y"] == pytest.approx(2.0, abs=0.01)
    assert tiff["elevation_contract"] == "georeferenced_metric_calibration_eligible"


@pytest.mark.parametrize("format_key", ["png", "jpg"])
def test_nongeoreferenced_png_and_jpg_complete_truthful_rdsm_pipeline(
    tmp_path: Path,
    format_key: str,
) -> None:
    report = run_input_format_qualification(tmp_path / f"{format_key}-input-contract")
    formats = report["formats"]
    assert isinstance(formats, dict)
    entry = formats[format_key]
    assert isinstance(entry, dict)
    source_value = entry["path"]
    assert isinstance(source_value, str)
    source = Path(source_value)

    project_dir = tmp_path / f"{format_key}-project"
    result = ProductionElevationRuntime(prior=_DeterministicFormatPrior()).run(
        ProcessingRequest(
            source=source,
            output_dir=project_dir,
            requested_output="rdsm",
            tile_size=256,
            overlap=32,
        ),
        job_id=f"format-{format_key}",
    )

    assert result.status is ProjectRunStatus.COMPLETE
    manifest = ProjectManifest.load(project_dir)
    assert manifest.status == ProjectRunStatus.COMPLETE.value
    assert manifest.input_kind == "non_georeferenced"
    rdsm = manifest.artifact_path("rdsm")
    assert rdsm is not None and rdsm.is_file()
    assert manifest.artifact_path("dsm") is None
    assert manifest.artifacts["rdsm"]["units"] == "relative"
