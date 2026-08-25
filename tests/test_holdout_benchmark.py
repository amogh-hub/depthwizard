import numpy as np

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark


def test_sparse_anchor_holdout_recovers_affine_metric_scale() -> None:
    y, x = np.mgrid[:96, :112]
    relative = (0.02 * x + 0.03 * y + 0.2 * np.sin(x / 13)).astype(np.float32)
    reference = (18.0 * relative + 420.0).astype(np.float32)

    result = sparse_anchor_holdout_benchmark(
        relative,
        reference,
        anchor_count=48,
        seed=26175,
        exclusion_radius_px=2,
    )

    assert not result.orientation_flipped
    assert abs(result.calibration.scale - 18.0) < 1e-4
    assert abs(result.calibration.offset - 420.0) < 1e-3
    assert result.metrics.rmse_m < 1e-3
    assert result.metrics.mae_m < 1e-3
    assert not np.any(result.anchor_mask & result.evaluation_mask)


def test_sparse_anchor_holdout_resolves_inverted_orientation() -> None:
    y, x = np.mgrid[:80, :90]
    height = (0.04 * x + 0.01 * y).astype(np.float32)
    relative_inverted = -height
    reference = (7.5 * height + 1200.0).astype(np.float32)

    result = sparse_anchor_holdout_benchmark(
        relative_inverted,
        reference,
        anchor_count=32,
        seed=8,
    )

    assert result.orientation_flipped
    assert result.anchor_correlation_before < 0
    assert result.anchor_correlation_after > 0
    assert result.metrics.rmse_m < 1e-3
