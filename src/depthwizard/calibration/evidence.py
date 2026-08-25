from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.contracts import CalibrationResult


@dataclass(frozen=True)
class EvidenceCalibrationOutput:
    dsm: np.ndarray
    calibration: CalibrationResult
    anchor_mask: np.ndarray
    low_frequency_bias: np.ndarray


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


def calibrate_relative_height_with_dem(
    relative_height: np.ndarray,
    dem_aligned: np.ndarray,
    *,
    dem_valid: np.ndarray | None = None,
    ground_probability: np.ndarray | None = None,
    uncertainty: np.ndarray | None = None,
    low_frequency_sigma_px: float = 24.0,
    min_anchors: int = 32,
) -> EvidenceCalibrationOutput:
    """Convert relative height to metric DSM using DEM evidence without erasing fine structure.

    1) robust global scale/offset from conservative anchor pixels;
    2) smooth residual field from the DEM to correct only low-frequency terrain bias;
    3) preserve the high-frequency image-derived height structure.
    """
    rel = np.asarray(relative_height, dtype=np.float64)
    dem = np.asarray(dem_aligned, dtype=np.float64)
    if rel.shape != dem.shape:
        raise ValueError("relative_height and dem_aligned must have identical shape")

    valid = np.isfinite(rel) & np.isfinite(dem)
    if dem_valid is not None:
        dm = np.asarray(dem_valid, dtype=bool)
        if dm.shape != rel.shape:
            raise ValueError("dem_valid must match relative_height")
        valid &= dm

    anchor_mask, weights = build_anchor_weights(
        dem_valid=valid,
        ground_probability=ground_probability,
        uncertainty=uncertainty,
    )
    if int(anchor_mask.sum()) < min_anchors:
        # Semantic filtering can be over-conservative on a difficult scene. Falling back to all
        # valid DEM support is preferable to fabricating scale from too few points.
        anchor_mask = valid
        weights = np.ones(rel.shape, dtype=np.float64)
        if uncertainty is not None:
            unc = np.asarray(uncertainty, dtype=np.float64)
            finite_unc = valid & np.isfinite(unc)
            reference = max(float(np.median(unc[finite_unc])), 1e-6) if np.any(finite_unc) else 1.0
            weights[finite_unc] = 1.0 / (1.0 + np.maximum(unc[finite_unc], 0.0) / reference)
        weights[~anchor_mask] = 0.0

    if int(anchor_mask.sum()) < min_anchors:
        raise ValueError(
            f"insufficient reliable DEM anchors: {int(anchor_mask.sum())}; need at least {min_anchors}"
        )

    fit = robust_affine_calibration(
        rel[anchor_mask],
        dem[anchor_mask],
        weights=weights[anchor_mask],
    )
    globally_scaled = fit.scale * rel + fit.offset

    raw_residual = np.zeros(rel.shape, dtype=np.float64)
    residual_weight = np.zeros(rel.shape, dtype=np.float64)
    raw_residual[anchor_mask] = dem[anchor_mask] - globally_scaled[anchor_mask]
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
    )
