import numpy as np

from depthwizard.evaluation.topography import (
    TerrainPromotionThresholds,
    compute_stratified_terrain_metrics,
    terrain_promotion_gate,
)


def _piecewise_slope_reference() -> np.ndarray:
    rows = 120
    cols = 200
    segment_width = 50
    angles = (5.0, 20.0, 35.0, 50.0)
    profile = np.zeros(cols, dtype=np.float64)
    for col in range(1, cols):
        segment = min(col // segment_width, len(angles) - 1)
        profile[col] = profile[col - 1] + np.tan(np.deg2rad(angles[segment]))
    return np.repeat(profile[None, :], rows, axis=0)


def _prediction(reference: np.ndarray, *, steep_scale: float) -> np.ndarray:
    cols = reference.shape[1]
    x = np.arange(cols, dtype=np.float64)
    error = 0.4 * np.sin(x / 3.0)
    error[x >= 100] = steep_scale * np.sin(x[x >= 100] / 2.0)
    return reference + error[None, :]


def test_stratified_terrain_report_contains_all_reference_slope_bands() -> None:
    reference = _piecewise_slope_reference()
    report = compute_stratified_terrain_metrics(
        _prediction(reference, steep_scale=2.0),
        reference,
        gsd_x_m=1.0,
        gsd_y_m=1.0,
    )

    names = {item.name for item in report.strata}
    assert names == {"gentle_lt15", "moderate_15_30", "steep_30_45", "very_steep_ge45"}
    assert report.steep_valid_pixels > 10_000
    assert report.steep_elevation_rmse_m > 0.5
    assert report.steep_slope_rmse_degrees > 0.0


def test_material_steep_terrain_improvement_passes_without_easy_terrain_regression() -> None:
    reference = _piecewise_slope_reference()
    baseline = compute_stratified_terrain_metrics(
        _prediction(reference, steep_scale=2.0),
        reference,
        gsd_x_m=1.0,
        gsd_y_m=1.0,
    )
    candidate = compute_stratified_terrain_metrics(
        _prediction(reference, steep_scale=0.25),
        reference,
        gsd_x_m=1.0,
        gsd_y_m=1.0,
    )

    decision = terrain_promotion_gate(
        baseline,
        candidate,
        thresholds=TerrainPromotionThresholds(min_steep_pixels=100),
    )

    assert decision.passed
    assert decision.steep_elevation_rmse_reduction_fraction > 0.5
    assert decision.steep_slope_rmse_reduction_fraction > 0.5


def test_tiny_steep_terrain_gain_cannot_pass_promotion() -> None:
    reference = _piecewise_slope_reference()
    baseline = compute_stratified_terrain_metrics(
        _prediction(reference, steep_scale=2.0),
        reference,
        gsd_x_m=1.0,
        gsd_y_m=1.0,
    )
    candidate = compute_stratified_terrain_metrics(
        _prediction(reference, steep_scale=1.999),
        reference,
        gsd_x_m=1.0,
        gsd_y_m=1.0,
    )

    decision = terrain_promotion_gate(
        baseline,
        candidate,
        thresholds=TerrainPromotionThresholds(min_steep_pixels=100),
    )

    assert not decision.passed
    assert decision.steep_elevation_rmse_reduction_fraction < 0.01
    assert any("steep elevation RMSE reduction" in reason for reason in decision.reasons)
