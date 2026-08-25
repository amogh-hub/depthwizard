import numpy as np

from depthwizard.evaluation.metrics import compute_elevation_metrics


def test_metrics_exact_match() -> None:
    ref = np.array([[1.0, 2.0], [3.0, 4.0]])
    metrics = compute_elevation_metrics(ref.copy(), ref)
    assert metrics.mae_m == 0.0
    assert metrics.rmse_m == 0.0
    assert metrics.pearson_r == 1.0


def test_metrics_known_error() -> None:
    ref = np.array([0.0, 1.0, 2.0])
    pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_elevation_metrics(pred, ref)
    assert metrics.mae_m == 1.0
    assert metrics.rmse_m == 1.0
    assert metrics.mean_bias_m == 1.0
