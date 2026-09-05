import numpy as np

from depthwizard.evaluation.metrics import compute_slope_metrics, slope_degrees


def test_planar_slope_is_correct() -> None:
    _, x = np.mgrid[:20, :20]
    z = x.astype(np.float32)
    slope = slope_degrees(z, gsd_x=1.0, gsd_y=1.0)
    assert np.allclose(slope[2:-2, 2:-2], 45.0, atol=1e-4)
    metrics = compute_slope_metrics(z, z, gsd_x=1.0, gsd_y=1.0)
    assert metrics.mae_degrees == 0.0


def test_planar_slope_uses_full_rotated_sheared_ground_jacobian() -> None:
    row, col = np.mgrid[:20, :20]
    jacobian = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=np.float64)
    ground_gradient = np.array([0.5, 0.25], dtype=np.float64)
    pixel_gradient = jacobian.T @ ground_gradient
    z = (pixel_gradient[0] * col + pixel_gradient[1] * row).astype(np.float32)

    slope = slope_degrees(z, ground_jacobian_m=jacobian)
    expected = np.degrees(np.arctan(np.linalg.norm(ground_gradient)))

    assert np.allclose(slope[2:-2, 2:-2], expected, atol=1e-4)
