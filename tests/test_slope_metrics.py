import numpy as np

from depthwizard.evaluation.metrics import compute_slope_metrics, slope_degrees


def test_planar_slope_is_correct() -> None:
    y, x = np.mgrid[:20, :20]
    z = x.astype(np.float32)
    slope = slope_degrees(z, gsd_x=1.0, gsd_y=1.0)
    assert np.allclose(slope[2:-2, 2:-2], 45.0, atol=1e-4)
    metrics = compute_slope_metrics(z, z, gsd_x=1.0, gsd_y=1.0)
    assert metrics.mae_degrees == 0.0
