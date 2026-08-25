import numpy as np

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
