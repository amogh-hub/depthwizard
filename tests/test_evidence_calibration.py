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
    assert abs(result.calibration.scale - 3.0) < 0.1
    assert abs(result.calibration.offset - 120.0) < 0.5
    assert np.nanmean(np.abs(result.dsm - metric)) < 0.2
