from __future__ import annotations

import numpy as np


def robust_rgb_normalize(
    rgb: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic robust normalization to float32 [0, 1], independently per band."""
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("rgb must have shape HxWx3")
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")

    image = image.astype(np.float32, copy=False)
    valid = np.all(np.isfinite(image), axis=-1)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        if vm.shape != image.shape[:2]:
            raise ValueError("valid_mask must match image height/width")
        valid &= vm
    if not np.any(valid):
        raise ValueError("no valid RGB pixels available for normalization")

    result = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        values = image[..., channel][valid]
        lo, hi = np.percentile(values, [low_percentile, high_percentile])
        scale = max(float(hi - lo), 1e-6)
        result[..., channel] = np.clip((image[..., channel] - lo) / scale, 0.0, 1.0)
    result[~valid] = 0.0
    return result
