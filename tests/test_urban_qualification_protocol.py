from dataclasses import replace

import numpy as np
import pytest

from depthwizard.evaluation.building_height import evaluate_building_height_instances
from qualification.compare_urban_building_height import _validate_report_integrity
from qualification.evaluate_urban_building_height import _prepare_evaluation_inputs


def _two_building_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (160, 160)
    reference = np.full(shape, 100.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    first = (slice(25, 45), slice(25, 45))
    second = (slice(95, 115), slice(95, 115))
    buildings[first] = True
    buildings[second] = True
    reference[first] += 8.0
    reference[second] += 12.0
    return reference, buildings, ~buildings


def _evaluate_valid_report():
    reference, buildings, strict_ground = _two_building_scene()
    return evaluate_building_height_instances(
        reference.copy(),
        reference,
        buildings,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        ground_candidate_mask=strict_ground,
        min_building_area_m2=20.0,
        min_reference_height_m=2.0,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=32,
    )


def test_candidate_nodata_cannot_remove_reference_eligible_building() -> None:
    reference, buildings, strict_ground = _two_building_scene()
    prediction = reference.copy()
    second = (slice(95, 115), slice(95, 115))

    reference_valid = np.ones(reference.shape, dtype=bool)
    prediction_valid = np.ones(reference.shape, dtype=bool)
    prediction_valid[second] = False

    prepared_prediction, prepared_reference, prepared_buildings, prepared_ground = (
        _prepare_evaluation_inputs(
            prediction,
            prediction_valid,
            reference,
            reference_valid,
            buildings,
            strict_ground,
        )
    )

    assert np.all(np.isfinite(prepared_reference[second]))
    assert np.all(~np.isfinite(prepared_prediction[second]))
    assert np.all(prepared_buildings[second])
    assert prepared_ground is not None

    report = evaluate_building_height_instances(
        prepared_prediction,
        prepared_reference,
        prepared_buildings,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        ground_candidate_mask=prepared_ground,
        min_building_area_m2=20.0,
        min_reference_height_m=2.0,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=32,
    )

    assert len(report.eligible_instance_ids) == 2
    assert len(report.evaluated_instance_ids) == 1
    assert len(report.prediction_failure_ids) == 1
    assert report.prediction_failure_ids[0] in report.eligible_instance_ids


def test_persisted_promotion_evidence_rejects_nonfinite_metric() -> None:
    report = _evaluate_valid_report()
    corrupted = replace(report, height_mae_m=float("nan"))

    with pytest.raises(ValueError, match="non-finite height_mae_m"):
        _validate_report_integrity(corrupted, role="candidate")


def test_persisted_promotion_evidence_rejects_inconsistent_instance_support() -> None:
    report = _evaluate_valid_report()
    corrupted = replace(report, prediction_failure_ids=(999,))

    with pytest.raises(ValueError, match="eligibility is inconsistent"):
        _validate_report_integrity(corrupted, role="candidate")
