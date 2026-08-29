from __future__ import annotations

import numpy as np

RELATIVE_DISPLAY_RELIEF_FRACTION = 0.12
RELATIVE_DISPLAY_SCALE_POLICY = "robust_p02_p98_target_12pct_shorter_xy_span_v1"
_MIN_ROBUST_RELIEF = 1e-8
_MIN_HORIZONTAL_SPAN = 1e-8


def relative_display_vertical_scale(
    values: np.ndarray,
    valid_mask: np.ndarray,
    *,
    span_x: float,
    span_y: float,
    target_relief_fraction: float = RELATIVE_DISPLAY_RELIEF_FRACTION,
) -> float:
    """Return a display-only Z multiplier for dimensionless rDSM surfaces.

    Relative monocular height has no physically meaningful ratio to pixel-space or metre-space XY.
    Rendering raw rDSM values beside scene-scale XY therefore makes otherwise valid geometry look
    flat. This policy preserves the rDSM itself and derives only a visualization multiplier: the
    robust P02-P98 relief is mapped to a fixed fraction of the shorter horizontal scene span.

    The returned multiplier must never be used for scientific measurements, calibration, exported
    elevation values, or metric claims. Those continue to consume the persisted raw rDSM/DSM.
    """
    array = np.asarray(values, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if array.ndim != 2 or valid.shape != array.shape:
        raise ValueError("relative display scaling requires matching 2D value/valid arrays")
    if not (0.0 < target_relief_fraction <= 1.0):
        raise ValueError("target_relief_fraction must be in (0, 1]")

    usable = valid & np.isfinite(array)
    finite = array[usable]
    if finite.size < 4:
        return 1.0

    low, high = np.percentile(finite, [2.0, 98.0])
    robust_relief = float(high - low)
    horizontal_span = min(abs(float(span_x)), abs(float(span_y)))
    if (
        not np.isfinite(robust_relief)
        or robust_relief <= _MIN_ROBUST_RELIEF
        or not np.isfinite(horizontal_span)
        or horizontal_span <= _MIN_HORIZONTAL_SPAN
    ):
        return 1.0

    scale = horizontal_span * target_relief_fraction / robust_relief
    return float(scale) if np.isfinite(scale) and scale > 0.0 else 1.0
