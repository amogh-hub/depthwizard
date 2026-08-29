from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_problem_statement_inputs import run_input_format_qualification


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
