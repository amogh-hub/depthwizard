import numpy as np

from depthwizard.calibration.robust import robust_affine_calibration


def test_recovers_affine_mapping_with_outlier() -> None:
    x = np.arange(20, dtype=float)
    y = 3.0 * x + 7.0
    y[-1] += 100.0
    result = robust_affine_calibration(x, y)
    assert abs(result.scale - 3.0) < 0.2
    assert abs(result.offset - 7.0) < 1.0
