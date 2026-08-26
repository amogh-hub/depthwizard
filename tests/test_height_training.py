from __future__ import annotations

import numpy as np

from depthwizard.height_model.training import (
    fit_rgb_ranges,
    fit_robust_range,
    normalize_relative_target,
    normalize_rgb,
    patch_windows,
    spatial_column_holdout,
)


def test_spatial_holdout_has_non_overlapping_gap() -> None:
    train, validation = spatial_column_holdout((64, 100), train_fraction=0.68, gap_px=12)
    assert train.shape == validation.shape == (64, 100)
    assert not np.any(train & validation)
    train_cols = np.flatnonzero(train.any(axis=0))
    validation_cols = np.flatnonzero(validation.any(axis=0))
    assert train_cols.size > 0
    assert validation_cols.size > 0
    assert int(validation_cols.min()) - int(train_cols.max()) > 1


def test_target_normalization_is_fit_only_on_training_mask() -> None:
    values = np.arange(100, dtype=np.float32).reshape(10, 10)
    train = np.zeros((10, 10), dtype=bool)
    train[:, :6] = True
    scale = fit_robust_range(values, train, lower_percentile=0, upper_percentile=100)
    assert scale.lower == 0.0
    assert scale.upper == 95.0

    normalized = normalize_relative_target(values, scale)
    assert normalized[0, 0] == 0.0
    assert normalized[-1, -1] == 1.0


def test_rgb_normalization_and_patch_enumeration() -> None:
    rng = np.random.default_rng(26175)
    rgb = rng.integers(0, 255, size=(64, 96, 3), dtype=np.uint8)
    eligible = np.ones((64, 96), dtype=bool)
    eligible[:, 80:] = False

    ranges = fit_rgb_ranges(rgb, eligible)
    normalized = normalize_rgb(rgb, ranges)
    assert normalized.shape == rgb.shape
    assert normalized.dtype == np.float32
    assert float(normalized.min()) >= 0.0
    assert float(normalized.max()) <= 1.0

    windows = patch_windows(
        eligible,
        patch_size=32,
        stride=16,
        min_valid_fraction=1.0,
    )
    assert windows
    for window in windows:
        patch = eligible[window.row_slice, window.col_slice]
        assert patch.shape == (32, 32)
        assert patch.all()
