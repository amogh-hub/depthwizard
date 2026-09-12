import numpy as np
from scipy.ndimage import gaussian_filter

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem


def test_dem_calibration_recovers_metric_scale_and_offset() -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.1 * x + 0.2 * y).astype(np.float32)
    metric = 3.0 * relative + 120.0
    metric += (0.5 * np.sin(x / 20)).astype(np.float32)
    valid = np.ones_like(relative, dtype=bool)

    result = calibrate_relative_height_with_dem(
        relative,
        metric,
        dem_valid=valid,
        low_frequency_sigma_px=8,
    )
    assert not result.orientation_flipped
    assert result.anchor_correlation_before > 0.99
    assert abs(result.calibration.scale - 3.0) < 0.1
    assert abs(result.calibration.offset - 120.0) < 0.5
    assert result.frequency_match_sigma_px == 0.0
    assert result.anchor_stride_px == 1
    assert np.nanmean(np.abs(result.dsm - metric)) < 0.2


def test_dem_calibration_resolves_inverted_monocular_orientation() -> None:
    y, x = np.mgrid[:64, :64]
    physical_height = (0.15 * x + 0.25 * y).astype(np.float32)
    inverted_prior = -physical_height
    metric = 4.0 * physical_height + 250.0
    valid = np.ones_like(physical_height, dtype=bool)

    result = calibrate_relative_height_with_dem(
        inverted_prior,
        metric,
        dem_valid=valid,
        low_frequency_sigma_px=0,
    )

    assert result.orientation_flipped
    assert result.anchor_correlation_before < -0.99
    assert result.anchor_correlation_after > 0.99
    assert result.calibration.scale > 0
    assert np.nanmean(np.abs(result.dsm - metric)) < 1e-3


def test_dem_calibration_rejects_uncorrelated_evidence() -> None:
    rng = np.random.default_rng(7)
    relative = rng.normal(size=(64, 64)).astype(np.float32)
    dem = rng.normal(size=(64, 64)).astype(np.float32)
    valid = np.ones_like(relative, dtype=bool)

    try:
        calibrate_relative_height_with_dem(
            relative,
            dem,
            dem_valid=valid,
            low_frequency_sigma_px=0,
            min_abs_anchor_correlation=0.2,
        )
    except ValueError as exc:
        assert "weakly correlated" in str(exc)
    else:
        raise AssertionError("uncorrelated DEM evidence should be rejected")


def test_coarse_dem_frequency_matching_recovers_scale_without_erasing_detail() -> None:
    y, x = np.mgrid[:256, :256]
    low_frequency_surface = (0.003 * x + 0.005 * y).astype(np.float64)
    high_frequency_structure = (0.18 * ((x + y) % 2 * 2 - 1)).astype(np.float64)
    relative = (low_frequency_surface + high_frequency_structure).astype(np.float32)
    coarse_dem = (5.0 * low_frequency_surface + 100.0).astype(np.float32)
    valid = np.ones_like(relative, dtype=bool)

    result = calibrate_relative_height_with_dem(
        relative,
        coarse_dem,
        dem_valid=valid,
        target_gsd_m=0.5,
        dem_effective_gsd_m=8.0,
        low_frequency_sigma_px=24.0,
        min_anchors=32,
    )

    assert result.anchor_stride_px == 16
    assert 6.0 < result.frequency_match_sigma_px < 7.0
    assert 200 <= int(result.anchor_mask.sum()) <= 300
    assert result.anchor_correlation_after > 0.99
    assert abs(result.calibration.scale - 5.0) < 0.2

    input_high_frequency = relative - gaussian_filter(relative.astype(np.float64), sigma=2.0)
    output_high_frequency = result.dsm - gaussian_filter(result.dsm.astype(np.float64), sigma=2.0)
    valid_inner = np.s_[16:-16, 16:-16]
    input_energy = float(np.std(input_high_frequency[valid_inner]))
    output_energy = float(np.std(output_high_frequency[valid_inner]))
    assert output_energy > 4.0 * input_energy


def test_coarse_dem_frequency_matching_does_not_count_upsampled_pixels_as_anchors() -> None:
    y, x = np.mgrid[:32, :32]
    relative = (0.1 * x + 0.2 * y).astype(np.float32)
    dem = (3.0 * relative + 50.0).astype(np.float32)

    try:
        calibrate_relative_height_with_dem(
            relative,
            dem,
            target_gsd_m=1.0,
            dem_effective_gsd_m=32.0,
            min_anchors=2,
        )
    except ValueError as exc:
        assert "independent reliable DEM anchors" in str(exc)
    else:
        raise AssertionError("one effective DEM support cell must not become many fake anchors")


def test_coarse_dem_frequency_contract_requires_both_resolutions() -> None:
    relative = np.arange(64, dtype=np.float32).reshape(8, 8)
    dem = 2.0 * relative + 10.0

    try:
        calibrate_relative_height_with_dem(relative, dem, target_gsd_m=1.0)
    except ValueError as exc:
        assert "must either both be supplied or both be omitted" in str(exc)
    else:
        raise AssertionError("missing DEM effective GSD must be rejected")

    try:
        calibrate_relative_height_with_dem(relative, dem, dem_effective_gsd_m=30.0)
    except ValueError as exc:
        assert "must either both be supplied or both be omitted" in str(exc)
    else:
        raise AssertionError("missing target GSD must be rejected")


def test_default_bias_scale_tracks_dem_support_not_fixed_image_pixels() -> None:
    y, x = np.mgrid[:256, :256]
    relative = (0.003 * x + 0.005 * y).astype(np.float32)
    dem = (5.0 * relative + 100.0).astype(np.float32)

    result = calibrate_relative_height_with_dem(
        relative,
        dem,
        target_gsd_m=0.5,
        dem_effective_gsd_m=8.0,
        min_anchors=32,
    )

    assert result.anchor_stride_px == 16
    assert result.bias_sigma_px == 16.0


def test_unknown_dem_support_uses_conservative_bias_scale() -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.1 * x + 0.2 * y).astype(np.float32)
    dem = (3.0 * relative + 120.0).astype(np.float32)

    result = calibrate_relative_height_with_dem(relative, dem)

    assert result.bias_sigma_px == 24.0


def test_dem_calibration_rejects_high_residual_metric_claim() -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.1 * x + 0.2 * y).astype(np.float32)
    structured_error = np.where((x + y) % 2 == 0, 8.0, -8.0)
    dem = (3.0 * relative + 120.0 + structured_error).astype(np.float32)

    try:
        calibrate_relative_height_with_dem(
            relative,
            dem,
            low_frequency_sigma_px=0,
            max_anchor_rmse_m=2.0,
        )
    except ValueError as exc:
        assert "anchor RMSE" in str(exc)
    else:
        raise AssertionError("high residual DEM evidence must not produce a metric DSM claim")


def test_dem_rmse_gate_assesses_final_frequency_matched_bias_corrected_surface() -> None:
    y, x = np.mgrid[:256, :256]
    relative = (0.003 * x + 0.005 * y).astype(np.float32)
    smooth_terrain_bias = 8.0 * np.sin(x / 48.0) * np.cos(y / 56.0)
    dem = (4.0 * relative + 250.0 + smooth_terrain_bias).astype(np.float32)

    result = calibrate_relative_height_with_dem(
        relative,
        dem,
        low_frequency_sigma_px=6.0,
        max_anchor_rmse_m=2.0,
    )

    assert result.affine_anchor_rmse_m > 2.0
    assert result.post_bias_frequency_matched_anchor_rmse_m < 2.0
    assert result.post_bias_frequency_matched_anchor_mae_m < 2.0
    assert "post_bias_frequency_matched_anchor_rmse_passed" in (
        result.calibration.quality_notes
    )


def test_dem_calibration_rejects_spatially_clustered_anchor_mask() -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.1 * x + 0.2 * y).astype(np.float32)
    dem = (3.0 * relative + 120.0).astype(np.float32)
    valid = np.zeros_like(relative, dtype=bool)
    valid[:16, :16] = True

    try:
        calibrate_relative_height_with_dem(relative, dem, dem_valid=valid)
    except ValueError as exc:
        assert "spatially clustered" in str(exc)
    else:
        raise AssertionError("clustered DEM evidence must not support a scene-level metric claim")
