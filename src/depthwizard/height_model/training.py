from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustRange:
    lower: float
    upper: float

    @property
    def span(self) -> float:
        return self.upper - self.lower


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
