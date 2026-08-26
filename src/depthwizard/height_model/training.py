from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from depthwizard.calibration.robust import robust_affine_calibration


class PriorReferenceCalibrationError(ValueError):
    """Raised when a training scene cannot support the positive-height prior convention.

    This is intentionally distinct from generic input/IO errors. A caller may reject a *training*
    sample and try another sample from the same pre-declared geographic group, while alignment,
    CRS, shape, and other engineering failures must still stop the run.
    """


@dataclass(frozen=True)
class RobustRange:
    lower: float
    upper: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


@dataclass(frozen=True)
class PriorReferenceFit:
    """Training-only affine relation between DA3 relative height and metric reference DSM."""

    scale_m_per_prior_unit: float
    offset_m: float
    rmse_m: float
    median_abs_residual_m: float
    samples: int


@dataclass(frozen=True)
class PatchWindow:
    row: int
    col: int
    size: int

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.size)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.size)


def fit_robust_range(
    values: np.ndarray,
    fit_mask: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> RobustRange:
    """Fit normalization statistics using only a declared training mask."""
    array = np.asarray(values, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool)
    if array.shape[:2] != mask.shape:
        raise ValueError("fit_mask must match the first two value dimensions")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("invalid percentile range")
    if array.ndim != 2:
        raise ValueError("fit_robust_range expects a 2D array")
    selected = array[mask & np.isfinite(array)]
    if selected.size < 16:
        raise ValueError("insufficient finite training samples for robust normalization")
    lower, upper = np.percentile(selected, [lower_percentile, upper_percentile])
    if not np.isfinite(lower) or not np.isfinite(upper) or upper - lower <= 1e-8:
        raise ValueError("training samples do not span a usable normalization range")
    return RobustRange(lower=float(lower), upper=float(upper))


def normalize_relative_target(values: np.ndarray, scale: RobustRange) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    normalized = (array - np.float32(scale.lower)) / np.float32(scale.span)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized[~np.isfinite(array)] = np.nan
    return normalized.astype(np.float32, copy=False)


def fit_reference_to_prior(
    geometry_prior: np.ndarray,
    reference_dsm_m: np.ndarray,
    fit_mask: np.ndarray,
) -> PriorReferenceFit:
    """Fit one robust scene-level affine relation using training pixels only.

    The final network predicts a refinement in the native DA3 relative-height coordinate system.
    Dense reference DSMs are metric, so directly normalizing each patch or fitting scale/shift per
    patch lets neighbouring patches learn mutually inconsistent coordinate systems. Instead we fit
    one robust positive affine relation for the declared training region and invert that relation
    to express every reference pixel in the same DA3-relative coordinates.

    This fit is a supervision canonicalization step, not an inference-time calibration. Final
    metric elevation is still recovered from DEM/GCP evidence after height inference.
    """
    geometry = np.asarray(geometry_prior, dtype=np.float64)
    reference = np.asarray(reference_dsm_m, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=bool)
    if geometry.ndim != 2 or reference.ndim != 2 or geometry.shape != reference.shape:
        raise ValueError("geometry_prior and reference_dsm_m must be matching 2D arrays")
    if mask.shape != geometry.shape:
        raise ValueError("fit_mask must match geometry_prior")

    valid = mask & np.isfinite(geometry) & np.isfinite(reference)
    if int(valid.sum()) < 16:
        raise ValueError("insufficient finite training pixels for prior/reference canonicalization")

    geometry_valid = geometry[valid]
    reference_valid = reference[valid]
    geometry_std = float(np.std(geometry_valid))
    reference_std = float(np.std(reference_valid))
    if geometry_std > 1e-12 and reference_std > 1e-12:
        pearson_r = float(np.corrcoef(geometry_valid, reference_valid)[0, 1])
    else:
        pearson_r = float("nan")

    try:
        calibration = robust_affine_calibration(
            geometry_valid,
            reference_valid,
            require_positive_scale=True,
        )
    except ValueError as exc:
        correlation_text = f"{pearson_r:.4f}" if np.isfinite(pearson_r) else "undefined"
        raise PriorReferenceCalibrationError(
            "training prior/reference canonicalization rejected the scene under the required "
            f"positive-height convention (Pearson r={correlation_text}): {exc}"
        ) from exc

    if not np.isfinite(calibration.scale) or calibration.scale <= 1e-8:
        raise PriorReferenceCalibrationError(
            "training prior/reference fit produced an invalid positive scale"
        )
    return PriorReferenceFit(
        scale_m_per_prior_unit=float(calibration.scale),
        offset_m=float(calibration.offset),
        rmse_m=float(calibration.rmse_anchor),
        median_abs_residual_m=float(calibration.median_abs_residual),
        samples=int(calibration.anchors_used),
    )


def canonicalize_reference_to_prior(
    reference_dsm_m: np.ndarray,
    fit: PriorReferenceFit,
) -> np.ndarray:
    """Express a metric DSM in the DA3-relative coordinate system defined by ``fit``."""
    if not np.isfinite(fit.scale_m_per_prior_unit) or fit.scale_m_per_prior_unit <= 1e-8:
        raise ValueError("fit must contain a finite positive scale")
    reference = np.asarray(reference_dsm_m, dtype=np.float32)
    canonical = (
        reference - np.float32(fit.offset_m)
    ) / np.float32(fit.scale_m_per_prior_unit)
    canonical[~np.isfinite(reference)] = np.nan
    return canonical.astype(np.float32, copy=False)


def fit_rgb_ranges(
    rgb: np.ndarray,
    fit_mask: np.ndarray,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> tuple[RobustRange, RobustRange, RobustRange]:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape H x W x 3")

    def fit(channel: int) -> RobustRange:
        return fit_robust_range(
            image[..., channel],
            fit_mask,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

    return fit(0), fit(1), fit(2)


def normalize_rgb(
    rgb: np.ndarray,
    ranges: tuple[RobustRange, RobustRange, RobustRange],
) -> np.ndarray:
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must have shape H x W x 3")
    output = np.empty_like(image, dtype=np.float32)
    for channel, scale in enumerate(ranges):
        output[..., channel] = np.clip(
            (image[..., channel] - np.float32(scale.lower)) / np.float32(scale.span),
            0.0,
            1.0,
        )
    return output


def spatial_column_holdout(
    shape: tuple[int, int],
    *,
    train_fraction: float = 0.68,
    gap_px: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic left/right train-validation regions separated by an exclusion gap."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("shape dimensions must be positive")
    if not 0.5 <= train_fraction <= 0.85:
        raise ValueError("train_fraction must be in [0.5, 0.85]")
    if gap_px < 0:
        raise ValueError("gap_px must be non-negative")
    split = round(width * train_fraction)
    half_gap = gap_px // 2
    train_end = split - half_gap
    val_start = split + (gap_px - half_gap)
    if train_end < 1 or val_start >= width:
        raise ValueError("holdout gap leaves no train or validation region")
    train = np.zeros((height, width), dtype=bool)
    validation = np.zeros((height, width), dtype=bool)
    train[:, :train_end] = True
    validation[:, val_start:] = True
    return train, validation


def patch_windows(
    eligible_mask: np.ndarray,
    *,
    patch_size: int = 192,
    stride: int = 128,
    min_valid_fraction: float = 0.95,
) -> list[PatchWindow]:
    """Enumerate deterministic patch windows wholly supported by an eligibility mask."""
    mask = np.asarray(eligible_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("eligible_mask must be 2D")
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if not 0.0 < min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be in (0, 1]")
    height, width = mask.shape
    if patch_size > height or patch_size > width:
        return []

    rows = list(range(0, height - patch_size + 1, stride))
    cols = list(range(0, width - patch_size + 1, stride))
    if rows[-1] != height - patch_size:
        rows.append(height - patch_size)
    if cols[-1] != width - patch_size:
        cols.append(width - patch_size)

    windows: list[PatchWindow] = []
    required = min_valid_fraction * patch_size * patch_size
    for row in rows:
        for col in cols:
            valid_count = int(mask[row : row + patch_size, col : col + patch_size].sum())
            if valid_count >= required:
                windows.append(PatchWindow(row=row, col=col, size=patch_size))
    return windows
