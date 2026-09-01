import numpy as np

from depthwizard.evaluation.building_height import evaluate_building_height_instances
from qualification.evaluate_urban_building_height import _prepare_evaluation_inputs


def test_candidate_nodata_cannot_remove_reference_eligible_building() -> None:
    shape = (160, 160)
    reference = np.full(shape, 100.0, dtype=np.float64)
    prediction = reference.copy()
    buildings = np.zeros(shape, dtype=bool)
    first = (slice(25, 45), slice(25, 45))
    second = (slice(95, 115), slice(95, 115))
    buildings[first] = True
    buildings[second] = True
    reference[first] += 8.0
    reference[second] += 12.0
    prediction[first] += 8.0
    prediction[second] += 12.0

    reference_valid = np.ones(shape, dtype=bool)
    prediction_valid = np.ones(shape, dtype=bool)
    prediction_valid[second] = False
    strict_ground = ~buildings

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
