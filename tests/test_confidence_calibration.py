import numpy as np

from depthwizard.calibration.confidence import model_confidence_to_uncertainty


def test_model_confidence_becomes_monotonic_nonprobabilistic_uncertainty() -> None:
    confidence = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape(10, 10)
    result = model_confidence_to_uncertainty(confidence)

    assert result.active is True
    assert result.uncertainty is not None
    assert result.uncertainty.shape == confidence.shape
    assert float(result.uncertainty[0, 0]) > float(result.uncertainty[-1, -1])
    assert np.nanmin(result.uncertainty) >= 0.0
    assert np.nanmax(result.uncertainty) <= 1.0
    evidence = result.evidence()
    assert evidence["probability_calibrated"] is False
    assert evidence["source"] == "model_native_confidence"
    assert "not_probability_calibrated" in str(evidence["semantics"])


def test_missing_or_degenerate_confidence_disables_weighting_instead_of_fabricating_it() -> None:
    missing = model_confidence_to_uncertainty(None)
    assert missing.active is False
    assert missing.uncertainty is None
    assert missing.reason == "model_native_confidence_unavailable"

    constant = model_confidence_to_uncertainty(np.full((8, 8), 0.8, dtype=np.float32))
    assert constant.active is False
    assert constant.uncertainty is None
    assert constant.reason == "degenerate_model_native_confidence_range"


def test_nonfinite_confidence_can_only_reduce_anchor_reliability() -> None:
    confidence = np.arange(25, dtype=np.float32).reshape(5, 5)
    confidence[2, 2] = np.nan
    result = model_confidence_to_uncertainty(confidence)

    assert result.active is True
    assert result.uncertainty is not None
    assert result.uncertainty[2, 2] == 1.0
    assert result.finite_fraction == 24 / 25
