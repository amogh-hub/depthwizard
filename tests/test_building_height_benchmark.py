import numpy as np

from depthwizard.evaluation.building_height import (
    BuildingHeightPromotionThresholds,
    building_height_promotion_gate,
    evaluate_building_height_instances,
)


def _urban_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[slice, slice]], list[float]]:
    shape = (160, 160)
    ground = np.full(shape, 100.0, dtype=np.float64)
    building_mask = np.zeros(shape, dtype=bool)
    footprints = [
        (slice(20, 36), slice(20, 36)),
        (slice(20, 36), slice(70, 86)),
        (slice(80, 96), slice(25, 41)),
        (slice(90, 106), slice(95, 111)),
    ]
    heights = [5.0, 8.0, 11.0, 14.0]
    reference = ground.copy()
    for footprint, height in zip(footprints, heights, strict=True):
        building_mask[footprint] = True
        reference[footprint] += height
    return ground, reference, building_mask, footprints, heights


def _prediction_with_height_errors(errors_m: list[float]) -> np.ndarray:
    ground, reference, _, footprints, heights = _urban_scene()
    prediction = ground.copy()
    for footprint, height, error in zip(footprints, heights, errors_m, strict=True):
        prediction[footprint] += height + error
    return prediction


def _evaluate(prediction: np.ndarray):
    _, reference, buildings, _, _ = _urban_scene()
    return evaluate_building_height_instances(
        prediction,
        reference,
        buildings,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        min_building_area_m2=20.0,
        min_reference_height_m=2.0,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=32,
    )


def test_building_height_benchmark_measures_instances_not_scene_average() -> None:
    report = _evaluate(_prediction_with_height_errors([-3.0, -3.0, -3.0, -3.0]))

    assert len(report.eligible_instance_ids) == 4
    assert report.prediction_failure_ids == ()
    assert report.height_mae_m == 3.0
    assert report.height_rmse_m == 3.0
    assert report.height_bias_m == -3.0
    assert report.within_2m_fraction == 0.0
    assert report.ground_mae_m == 0.0


def test_material_building_height_improvement_passes_promotion_gate() -> None:
    baseline = _evaluate(_prediction_with_height_errors([-3.0, -3.0, -3.0, -3.0]))
    candidate = _evaluate(_prediction_with_height_errors([-0.5, -0.5, -0.5, -0.5]))

    decision = building_height_promotion_gate(
        baseline,
        candidate,
        thresholds=BuildingHeightPromotionThresholds(min_instances=4),
    )

    assert decision.passed
    assert decision.reasons == ()
    assert decision.mae_reduction_fraction > 0.80
    assert decision.rmse_reduction_fraction > 0.80
    assert decision.p90_reduction_fraction > 0.80


def test_millimetre_scale_improvement_cannot_pass_urban_promotion() -> None:
    baseline = _evaluate(_prediction_with_height_errors([-3.0, -3.0, -3.0, -3.0]))
    candidate = _evaluate(_prediction_with_height_errors([-2.999, -2.999, -2.999, -2.999]))

    decision = building_height_promotion_gate(
        baseline,
        candidate,
        thresholds=BuildingHeightPromotionThresholds(min_instances=4),
    )

    assert not decision.passed
    assert decision.mae_reduction_fraction < 0.001
    assert any("MAE reduction" in reason for reason in decision.reasons)
    assert any("RMSE reduction" in reason for reason in decision.reasons)
    assert any("P90 reduction" in reason for reason in decision.reasons)


def test_prediction_failure_on_reference_eligible_building_fails_closed() -> None:
    baseline = _evaluate(_prediction_with_height_errors([-3.0, -3.0, -3.0, -3.0]))
    candidate_prediction = _prediction_with_height_errors([-0.5, -0.5, -0.5, -0.5])
    _, _, _, footprints, _ = _urban_scene()
    candidate_prediction[footprints[0]] = np.nan
    candidate = _evaluate(candidate_prediction)

    decision = building_height_promotion_gate(
        baseline,
        candidate,
        thresholds=BuildingHeightPromotionThresholds(min_instances=3),
    )

    assert candidate.prediction_failure_ids
    assert not decision.passed
    assert any("prediction failures" in reason for reason in decision.reasons)
