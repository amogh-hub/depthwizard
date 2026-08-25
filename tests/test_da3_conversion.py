import numpy as np

from depthwizard.geometry_prior.da3 import depth_to_relative_height


def test_depth_to_relative_height_reverses_depth_order_without_metric_claim() -> None:
    depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    height = depth_to_relative_height(depth, low_percentile=0, high_percentile=100)
    assert height[0, 0] > height[1, 1]
    assert np.isclose(height.min(), 0.0)
    assert np.isclose(height.max(), 1.0)
