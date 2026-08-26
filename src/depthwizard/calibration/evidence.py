from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.contracts import CalibrationResult
from scipy.ndimage import gaussian_filter


_GAUSSIAN_FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


@dataclass(frozen=True)
class EvidenceCalibrationOutput:
    dsm: np.ndarray
    calibration: CalibrationResult
    anchor_mask: np.ndarray
    low_frequency_bias: np.ndarray
    orientation_flipped: bool
    anchor_correlation_before: float
    anchor_correlation_after: float
    frequency_match_sigma_px: float
    anchor_stride_px: int


def build_anchor_weights(
    *,
    dem_valid: np.ndarray,
    ground_probability: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    min_ground_probability: float = 0.55,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a conservative anchor mask and confidence weights.

    Low uncertainty and high ground probability increase reliability. If semantic/uncertainty
    products are unavailable, the caller still receives a valid DEM-only calibration path.
    """
    mask = np.asarray(dem_valid, dtype=bool).copy()
    weights = np.ones(mask.shape, dtype=np.float64)

    if ground_probability is not None:
        gp = np.asarray(ground_probability, dtype=np.float64)
        if gp.shape != mask.shape:
            raise ValueError("ground_probability must match dem_valid")
        mask &= np.isfinite(gp) & (gp >= min_ground_probability)
        weights *= np.clip(gp, 1e-3, 1.0)

    if uncertainty is not None:
        unc = np.asarray(uncertainty, dtype=np.float64)
        if unc.shape != mask.shape:
            raise ValueError("uncertainty must match dem_valid")
        mask &= np.isfinite(unc)
        valid_unc = unc[mask]
        if valid_unc.size:
            reference = max(float(np.median(valid_unc)), 1e-6)
            weights *= 1.0 / (1.0 + np.maximum(unc, 0.0) / reference)

    weights[~mask] = 0.0
    return mask, weights


def _weighted_correlation(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Return weighted Pearson correlation for finite, positive-weight observations."""
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    wv = np.asarray(weights, dtype=np.float64).reshape(-1)
    valid = np.isfinite(xv) & np.isfinite(yv) & np.isfinite(wv) & (wv > 0)
    xv, yv, wv = xv[valid], yv[valid], wv[valid]
    if xv.size < 2:
        return float("nan")
    total = float(np.sum(wv))
    if total <= 0:
        return float("nan")
    mx = float(np.sum(wv * xv) / total)
    my = float(np.sum(wv * yv) / total)
    dx = xv - mx
    dy = yv - my
    covariance = float(np.sum(wv * dx * dy) / total)
    var_x = float(np.sum(wv * dx * dx) / total)
    var_y = float(np.sum(wv * dy * dy) / total)
    denominator = float(np.sqrt(max(var_x * var_y, 0.0)))
    if denominator <= 1e-12:
        return float("nan")
    return covariance / denominator


def _frequency_match_parameters(
    *,
    target_gsd_m: float | None,
    dem_effective_gsd_m: float | None,
) -> tuple[float, int]:
    """Return Gaussian sigma and independent-support stride for coarse DEM evidence.

    ``dem_effective_gsd_m`` is treated as the approximate full width at half maximum (FWHM) of the
    coarse DEM's spatial support. The monocular field is low-pass filtered to that support before
    fitting scale/offset. The aligned DEM may be upsampled to the image grid, but repeated pixels do
    not create independent evidence: anchors are sampled at approximately one point per effective
    DEM support cell.

    Omitting both values preserves the historical same-resolution behavior. Supplying only one is a
    contract error because silently guessing the frequency ratio would make calibration claims
    irreproducible.
    """
    if target_gsd_m is None and dem_effective_gsd_m is None:
        return 0.0, 1
    if target_gsd_m is None or dem_effective_gsd_m is None:
        raise ValueError(
            "target_gsd_m and dem_effective_gsd_m must either both be supplied or both be omitted"
        )
    if not np.isfinite(target_gsd_m) or target_gsd_m <= 0:
        raise ValueError("target_gsd_m must be a finite positive value")
    if not np.isfinite(dem_effective_gsd_m) or dem_effective_gsd_m <= 0:
        raise ValueError("dem_effective_gsd_m must be a finite positive value")

    ratio = float(dem_effective_gsd_m / target_gsd_m)
    if ratio <= 1.0:
        return 0.0, 1
    sigma_px = ratio * _GAUSSIAN_FWHM_TO_SIGMA
    anchor_stride_px = max(1, round(ratio))
    return float(sigma_px), anchor_stride_px


def _normalized_gaussian_filter(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    sigma_px: float,
) -> np.ndarray:
    """Low-pass finite raster values without bleeding NaN/NoData values into valid support."""
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if array.shape != mask.shape:
        raise ValueError("values and valid must have identical shape")
    if sigma_px <= 0:
        result = array.copy()
        result[~mask] = np.nan
        return result

    numerator = gaussian_filter(
        np.where(mask, array, 0.0),
        sigma=sigma_px,
        mode="nearest",
    )
    denominator = gaussian_filter(mask.astype(np.float64), sigma=sigma_px, mode="nearest")
    result = np.full(array.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=result, where=denominator > 1e-8)
    return result


def _independent_support_mask(shape: tuple[int, ...], *, stride_px: int) -> np.ndarray:
    """Return deterministic approximately independent samples on the target raster grid."""
    if len(shape) != 2:
        raise ValueError("DEM calibration currently requires a 2D raster")
    if stride_px < 1:
        raise ValueError("stride_px must be >= 1")
    if stride_px == 1:
        return np.ones(shape, dtype=bool)
    support = np.zeros(shape, dtype=bool)
    offset = stride_px // 2
    support[offset::stride_px, offset::stride_px] = True
    return support


def calibrate_relative_height_with_dem(
    relative_height: np.ndarray,
    dem_aligned: np.ndarray,
    *,
    dem_valid: np.ndarray | None = None,
    ground_probability: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    low_frequency_sigma_px: float = 24.0,
    min_anchors: int = 32,
    min_abs_anchor_correlation: float = 0.05,
    resolve_orientation: bool = True,
    target_gsd_m: float | None = None,
    dem_effective_gsd_m: float | None = None,
) -> EvidenceCalibrationOutput:
    """Convert relative height to metric DSM using DEM evidence without erasing fine structure.

    The monocular prior is scale-agnostic and its vertical polarity can be unreliable after a
    domain shift from natural imagery to overhead remote sensing. DepthWizard therefore uses the
    independent DEM anchors to *diagnose* polarity before fitting metric scale. A negative anchor
    correlation is explicitly reflected, recorded in the output, and then fitted with a positive
    physical scale. Weakly correlated evidence is rejected rather than silently fabricating
    metric elevation.

    When the DEM is coarser than the optical raster, calibration is performed in the DEM's spatial
    frequency band rather than against unresolved high-frequency roofs/edges. Upsampling a coarse
    DEM does not manufacture independent observations: the anchor grid is thinned to approximately
    one sample per effective DEM support cell. The fitted scale/offset is then applied to the *raw*
    relative-height field, so fine image-derived structure is retained, while only a smooth terrain
    residual is added.

    1) conservative DEM/semantic/uncertainty anchors;
    2) optional DEM-frequency matching and independent-support thinning;
    3) evidence-based polarity diagnosis for the frequency-matched relative field;
    4) robust positive scale/offset fit;
    5) smooth residual correction for low-frequency terrain bias only;
    6) preservation of high-frequency image-derived structure.
    """
    rel = np.asarray(relative_height, dtype=np.float64)
    dem = np.asarray(dem_aligned, dtype=np.float64)
    if rel.shape != dem.shape:
        raise ValueError("relative_height and dem_aligned must have identical shape")
    if rel.ndim != 2:
        raise ValueError("relative_height and dem_aligned must be 2D rasters")
    if not (0.0 <= min_abs_anchor_correlation <= 1.0):
        raise ValueError("min_abs_anchor_correlation must be in [0, 1]")
    if min_anchors < 2:
        raise ValueError("min_anchors must be >= 2")

    valid = np.isfinite(rel) & np.isfinite(dem)
    if dem_valid is not None:
        dm = np.asarray(dem_valid, dtype=bool)
        if dm.shape != rel.shape:
            raise ValueError("dem_valid must match relative_height")
        valid &= dm

    frequency_match_sigma_px, anchor_stride_px = _frequency_match_parameters(
        target_gsd_m=target_gsd_m,
        dem_effective_gsd_m=dem_effective_gsd_m,
    )
    calibration_rel = _normalized_gaussian_filter(
        rel,
        valid,
        sigma_px=frequency_match_sigma_px,
    )
    valid &= np.isfinite(calibration_rel)
    independent_support = _independent_support_mask(rel.shape, stride_px=anchor_stride_px)

    anchor_mask, weights = build_anchor_weights(
        dem_valid=valid,
        ground_probability=ground_probability,
        uncertainty=uncertainty,
    )
    anchor_mask &= independent_support
    weights[~anchor_mask] = 0.0

    if int(anchor_mask.sum()) < min_anchors:
        # Semantic filtering can be over-conservative on a difficult scene. Fall back only to all
        # valid *independent* coarse-DEM support, never to every upsampled target pixel.
        anchor_mask = valid & independent_support
        weights = np.ones(rel.shape, dtype=np.float64)
        if uncertainty is not None:
            unc = np.asarray(uncertainty, dtype=np.float64)
            finite_unc = anchor_mask & np.isfinite(unc)
            reference = (
                max(float(np.median(unc[finite_unc])), 1e-6) if np.any(finite_unc) else 1.0
            )
            weights[finite_unc] = 1.0 / (
                1.0 + np.maximum(unc[finite_unc], 0.0) / reference
            )
        weights[~anchor_mask] = 0.0

    anchor_count = int(anchor_mask.sum())
    if anchor_count < min_anchors:
        raise ValueError(
            "insufficient independent reliable DEM anchors: "
            f"{anchor_count}; need at least {min_anchors}"
        )

    anchor_weights = weights[anchor_mask]
    correlation_before = _weighted_correlation(
        calibration_rel[anchor_mask], dem[anchor_mask], anchor_weights
    )
    if not np.isfinite(correlation_before):
        raise ValueError("DEM anchors cannot determine relative-height orientation")
    if abs(correlation_before) < min_abs_anchor_correlation:
        raise ValueError(
            "relative height is too weakly correlated with frequency-matched DEM evidence for "
            f"defensible metric calibration (|r|={abs(correlation_before):.3f} < "
            f"{min_abs_anchor_correlation:.3f})"
        )

    orientation_flipped = bool(resolve_orientation and correlation_before < 0.0)
    oriented_rel = -rel if orientation_flipped else rel
    oriented_calibration_rel = -calibration_rel if orientation_flipped else calibration_rel
    correlation_after = -correlation_before if orientation_flipped else correlation_before

    fit = robust_affine_calibration(
        oriented_calibration_rel[anchor_mask],
        dem[anchor_mask],
        weights=anchor_weights,
        require_positive_scale=True,
    )
    globally_scaled = fit.scale * oriented_rel + fit.offset
    calibration_band_scaled = fit.scale * oriented_calibration_rel + fit.offset

    raw_residual = np.zeros(rel.shape, dtype=np.float64)
    residual_weight = np.zeros(rel.shape, dtype=np.float64)
    raw_residual[anchor_mask] = dem[anchor_mask] - calibration_band_scaled[anchor_mask]
    residual_weight[anchor_mask] = weights[anchor_mask]

    if low_frequency_sigma_px <= 0:
        smooth_bias = np.zeros(rel.shape, dtype=np.float64)
    else:
        numerator = gaussian_filter(raw_residual * residual_weight, sigma=low_frequency_sigma_px)
        denominator = gaussian_filter(residual_weight, sigma=low_frequency_sigma_px)
        smooth_bias = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 1e-8,
        )

    dsm = globally_scaled + smooth_bias
    dsm[~np.isfinite(rel)] = np.nan
    return EvidenceCalibrationOutput(
        dsm=dsm.astype(np.float32),
        calibration=fit,
        anchor_mask=anchor_mask,
        low_frequency_bias=smooth_bias.astype(np.float32),
        orientation_flipped=orientation_flipped,
        anchor_correlation_before=float(correlation_before),
        anchor_correlation_after=float(correlation_after),
        frequency_match_sigma_px=frequency_match_sigma_px,
        anchor_stride_px=anchor_stride_px,
    )
