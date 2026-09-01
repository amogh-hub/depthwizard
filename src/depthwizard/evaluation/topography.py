from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from depthwizard.evaluation.metrics import slope_degrees


@dataclass(frozen=True)
class TerrainStratumMetrics:
    name: str
    min_reference_slope_degrees: float
    max_reference_slope_degrees: float | None
    valid_pixels: int
    elevation_mae_m: float
    elevation_rmse_m: float
    slope_mae_degrees: float
    slope_rmse_degrees: float


@dataclass(frozen=True)
class StratifiedTerrainReport:
    valid_pixels: int
    elevation_mae_m: float
    elevation_rmse_m: float
    slope_mae_degrees: float
    slope_rmse_degrees: float
    steep_valid_pixels: int
    steep_elevation_mae_m: float
    steep_elevation_rmse_m: float
    steep_slope_mae_degrees: float
    steep_slope_rmse_degrees: float
    strata: tuple[TerrainStratumMetrics, ...]


@dataclass(frozen=True)
class TerrainPromotionThresholds:
    min_steep_pixels: int = 10_000
    min_steep_elevation_rmse_reduction_fraction: float = 0.05
    min_steep_slope_rmse_reduction_fraction: float = 0.05
    max_overall_rmse_degradation_fraction: float = 0.01
    max_stratum_rmse_degradation_fraction: float = 0.02


@dataclass(frozen=True)
class TerrainPromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    steep_elevation_rmse_reduction_fraction: float
    steep_slope_rmse_reduction_fraction: float


def _error_metrics(error: np.ndarray) -> tuple[float, float]:
    if error.size == 0:
        raise ValueError("terrain stratum has no valid pixels")
    absolute = np.abs(error)
    return float(np.mean(absolute)), float(np.sqrt(np.mean(error**2)))


def _stratum(
    name: str,
    min_slope: float,
    max_slope: float | None,
    *,
    prediction: np.ndarray,
    reference: np.ndarray,
    prediction_slope: np.ndarray,
    reference_slope: np.ndarray,
    valid: np.ndarray,
) -> TerrainStratumMetrics | None:
    mask = valid & (reference_slope >= min_slope)
    if max_slope is not None:
        mask &= reference_slope < max_slope
    count = int(np.count_nonzero(mask))
    if count == 0:
        return None
    elevation_mae, elevation_rmse = _error_metrics(prediction[mask] - reference[mask])
    slope_mae, slope_rmse = _error_metrics(prediction_slope[mask] - reference_slope[mask])
    return TerrainStratumMetrics(
        name=name,
        min_reference_slope_degrees=min_slope,
        max_reference_slope_degrees=max_slope,
        valid_pixels=count,
        elevation_mae_m=elevation_mae,
        elevation_rmse_m=elevation_rmse,
        slope_mae_degrees=slope_mae,
        slope_rmse_degrees=slope_rmse,
    )


def compute_stratified_terrain_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    *,
    gsd_x_m: float,
    gsd_y_m: float,
    valid_mask: np.ndarray | None = None,
    steep_threshold_degrees: float = 30.0,
) -> StratifiedTerrainReport:
    """Evaluate elevation and slope errors across reference-defined terrain difficulty bands.

    Reference slope defines the strata so a candidate model cannot change which pixels are called
    difficult. The standard bands are <15°, 15–30°, 30–45°, and >=45°, with an additional aggregate
    >=30° steep-terrain score used by the promotion gate.
    """
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if pred.shape != ref.shape:
        raise ValueError("prediction and reference must have identical shape")
    if not np.isfinite(gsd_x_m) or gsd_x_m <= 0.0:
        raise ValueError("gsd_x_m must be a positive finite value")
    if not np.isfinite(gsd_y_m) or gsd_y_m <= 0.0:
        raise ValueError("gsd_y_m must be a positive finite value")
    if steep_threshold_degrees <= 0.0 or steep_threshold_degrees >= 90.0:
        raise ValueError("steep_threshold_degrees must be between 0 and 90")

    pred_slope = slope_degrees(pred, gsd_x=gsd_x_m, gsd_y=gsd_y_m).astype(np.float64)
    ref_slope = slope_degrees(ref, gsd_x=gsd_x_m, gsd_y=gsd_y_m).astype(np.float64)
    valid = np.isfinite(pred) & np.isfinite(ref) & np.isfinite(pred_slope) & np.isfinite(ref_slope)
    if valid_mask is not None:
        supplied = np.asarray(valid_mask, dtype=bool)
        if supplied.shape != pred.shape:
            raise ValueError("valid_mask must match prediction/reference shape")
        valid &= supplied
    if not np.any(valid):
        raise ValueError("no valid terrain pixels available for evaluation")

    elevation_mae, elevation_rmse = _error_metrics(pred[valid] - ref[valid])
    slope_mae, slope_rmse = _error_metrics(pred_slope[valid] - ref_slope[valid])

    definitions = (
        ("gentle_lt15", 0.0, 15.0),
        ("moderate_15_30", 15.0, 30.0),
        ("steep_30_45", 30.0, 45.0),
        ("very_steep_ge45", 45.0, None),
    )
    strata: list[TerrainStratumMetrics] = []
    for name, minimum, maximum in definitions:
        metrics = _stratum(
            name,
            minimum,
            maximum,
            prediction=pred,
            reference=ref,
            prediction_slope=pred_slope,
            reference_slope=ref_slope,
            valid=valid,
        )
        if metrics is not None:
            strata.append(metrics)

    steep = valid & (ref_slope >= steep_threshold_degrees)
    steep_count = int(np.count_nonzero(steep))
    if steep_count == 0:
        raise ValueError("reference contains no valid pixels in the requested steep-terrain band")
    steep_elevation_mae, steep_elevation_rmse = _error_metrics(pred[steep] - ref[steep])
    steep_slope_mae, steep_slope_rmse = _error_metrics(pred_slope[steep] - ref_slope[steep])

    return StratifiedTerrainReport(
        valid_pixels=int(np.count_nonzero(valid)),
        elevation_mae_m=elevation_mae,
        elevation_rmse_m=elevation_rmse,
        slope_mae_degrees=slope_mae,
        slope_rmse_degrees=slope_rmse,
        steep_valid_pixels=steep_count,
        steep_elevation_mae_m=steep_elevation_mae,
        steep_elevation_rmse_m=steep_elevation_rmse,
        steep_slope_mae_degrees=steep_slope_mae,
        steep_slope_rmse_degrees=steep_slope_rmse,
        strata=tuple(strata),
    )


def _fractional_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= baseline + 1e-12 else -float("inf")
    return float((baseline - candidate) / baseline)


def terrain_promotion_gate(
    baseline: StratifiedTerrainReport,
    candidate: StratifiedTerrainReport,
    *,
    thresholds: TerrainPromotionThresholds = TerrainPromotionThresholds(),
) -> TerrainPromotionDecision:
    """Require material steep-terrain gains without hiding regressions in easier strata."""
    reasons: list[str] = []
    if candidate.steep_valid_pixels < thresholds.min_steep_pixels:
        reasons.append(
            f"only {candidate.steep_valid_pixels} steep pixels evaluated; "
            f"need at least {thresholds.min_steep_pixels}"
        )
    if baseline.valid_pixels != candidate.valid_pixels:
        reasons.append("baseline and candidate terrain reports do not use identical valid support")

    elevation_reduction = _fractional_reduction(
        baseline.steep_elevation_rmse_m,
        candidate.steep_elevation_rmse_m,
    )
    slope_reduction = _fractional_reduction(
        baseline.steep_slope_rmse_degrees,
        candidate.steep_slope_rmse_degrees,
    )
    if elevation_reduction < thresholds.min_steep_elevation_rmse_reduction_fraction:
        reasons.append(
            f"steep elevation RMSE reduction {elevation_reduction:.3%} is below required "
            f"{thresholds.min_steep_elevation_rmse_reduction_fraction:.3%}"
        )
    if slope_reduction < thresholds.min_steep_slope_rmse_reduction_fraction:
        reasons.append(
            f"steep slope RMSE reduction {slope_reduction:.3%} is below required "
            f"{thresholds.min_steep_slope_rmse_reduction_fraction:.3%}"
        )

    allowed_overall = baseline.elevation_rmse_m * (
        1.0 + thresholds.max_overall_rmse_degradation_fraction
    )
    if candidate.elevation_rmse_m > allowed_overall:
        reasons.append("overall terrain elevation RMSE regressed beyond the allowed tolerance")

    baseline_by_name = {item.name: item for item in baseline.strata}
    candidate_by_name = {item.name: item for item in candidate.strata}
    if baseline_by_name.keys() != candidate_by_name.keys():
        reasons.append("baseline and candidate do not contain identical terrain strata")
    else:
        for name, baseline_stratum in baseline_by_name.items():
            candidate_stratum = candidate_by_name[name]
            if baseline_stratum.valid_pixels != candidate_stratum.valid_pixels:
                reasons.append(f"terrain stratum {name} does not use identical valid support")
                continue
            allowed_elevation = baseline_stratum.elevation_rmse_m * (
                1.0 + thresholds.max_stratum_rmse_degradation_fraction
            )
            allowed_slope = baseline_stratum.slope_rmse_degrees * (
                1.0 + thresholds.max_stratum_rmse_degradation_fraction
            )
            if candidate_stratum.elevation_rmse_m > allowed_elevation:
                reasons.append(f"terrain stratum {name} elevation RMSE regressed")
            if candidate_stratum.slope_rmse_degrees > allowed_slope:
                reasons.append(f"terrain stratum {name} slope RMSE regressed")

    return TerrainPromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        steep_elevation_rmse_reduction_fraction=elevation_reduction,
        steep_slope_rmse_reduction_fraction=slope_reduction,
    )
