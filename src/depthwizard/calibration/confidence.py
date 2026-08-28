from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConfidenceWeighting:
    """Conservative mapping from model-native confidence to calibration uncertainty.

    The result is deliberately *not* a probability or calibrated error estimate. It is only a
    monotonic reliability transform used to down-weight weaker DEM anchors when a model-native
    confidence raster is actually available.
    """

    uncertainty: np.ndarray | None
    active: bool
    finite_fraction: float
    robust_low: float | None
    robust_high: float | None
    semantics: str
    reason: str | None

    def evidence(self) -> dict[str, object]:
        return {
            "active": self.active,
            "source": "model_native_confidence" if self.active else None,
            "finite_fraction": self.finite_fraction,
            "robust_low": self.robust_low,
            "robust_high": self.robust_high,
            "semantics": self.semantics,
            "reason": self.reason,
            "probability_calibrated": False,
        }


def model_confidence_to_uncertainty(
    confidence: np.ndarray | None,
    *,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> ConfidenceWeighting:
    """Map native confidence monotonically to [0, 1] uncertainty for anchor weighting.

    The mapping is robust to extreme values and never invents confidence. If the supplied field is
    absent, has too little finite support, or is effectively constant, weighting is disabled and the
    DEM-only calibration path remains authoritative. Non-finite confidence cells are assigned the
    weakest reliability only after a usable field has been established, so they cannot increase an
    anchor's influence.
    """
    semantics = "monotonic_model_native_reliability_weight_not_probability_calibrated"
    if confidence is None:
        return ConfidenceWeighting(
            uncertainty=None,
            active=False,
            finite_fraction=0.0,
            robust_low=None,
            robust_high=None,
            semantics=semantics,
            reason="model_native_confidence_unavailable",
        )
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("confidence percentiles must satisfy 0 <= low < high <= 100")

    values = np.asarray(confidence, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("model-native confidence must be a 2D raster")
    finite = np.isfinite(values)
    finite_count = int(finite.sum())
    finite_fraction = float(finite_count / values.size) if values.size else 0.0
    if finite_count < 2:
        return ConfidenceWeighting(
            uncertainty=None,
            active=False,
            finite_fraction=finite_fraction,
            robust_low=None,
            robust_high=None,
            semantics=semantics,
            reason="insufficient_finite_model_native_confidence",
        )

    low, high = np.percentile(values[finite], [low_percentile, high_percentile])
    low = float(low)
    high = float(high)
    span = high - low
    scale_reference = max(abs(low), abs(high), 1.0)
    if not np.isfinite(span) or span <= 1e-9 * scale_reference:
        return ConfidenceWeighting(
            uncertainty=None,
            active=False,
            finite_fraction=finite_fraction,
            robust_low=low,
            robust_high=high,
            semantics=semantics,
            reason="degenerate_model_native_confidence_range",
        )

    reliability = np.zeros(values.shape, dtype=np.float64)
    reliability[finite] = np.clip((values[finite] - low) / span, 0.0, 1.0)
    # Missing confidence must never increase influence. Once the field is usable, non-finite cells
    # receive the weakest reliability (uncertainty=1) rather than being interpreted as confident.
    uncertainty = 1.0 - reliability
    return ConfidenceWeighting(
        uncertainty=uncertainty.astype(np.float32),
        active=True,
        finite_fraction=finite_fraction,
        robust_low=low,
        robust_high=high,
        semantics=semantics,
        reason=None,
    )
