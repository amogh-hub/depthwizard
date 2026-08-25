from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.geometry_prior.base import GeometryPrior
from depthwizard.preprocess.stats import (
    RGBNormalizationStats,
    estimate_rgb_normalization_stats,
    normalize_with_stats,
)
from depthwizard.tiling.blend import WeightedTileAccumulator
from depthwizard.tiling.grid import generate_tiles


@dataclass(frozen=True)
class GeometrySceneOutput:
    relative_height: np.ndarray
    confidence: np.ndarray | None
    model_id: str
    normalization: RGBNormalizationStats
    tile_count: int
    harmonized_tiles: int


def _harmonize_tile(
    tile_height: np.ndarray,
    existing: np.ndarray,
    overlap_mask: np.ndarray,
    *,
    min_overlap_pixels: int,
) -> tuple[np.ndarray, bool]:
    mask = overlap_mask & np.isfinite(tile_height) & np.isfinite(existing)
    if int(mask.sum()) < min_overlap_pixels:
        return tile_height, False
    tile_values = tile_height[mask]
    existing_values = existing[mask]
    if np.ptp(tile_values) <= 1e-6 or np.ptp(existing_values) <= 1e-6:
        return tile_height, False
    try:
        fit = robust_affine_calibration(tile_values, existing_values, require_positive_scale=True)
    except ValueError:
        return tile_height, False
    # Reject implausible overlap fits rather than propagating one bad tile across the mosaic.
    if not (0.2 <= fit.scale <= 5.0):
        return tile_height, False
    return (fit.scale * tile_height + fit.offset).astype(np.float32), True


def infer_geometry_scene(
    source_path: str | Path,
    prior: GeometryPrior,
    *,
    band_indices: tuple[int, int, int] = (1, 2, 3),
    tile_size: int = 1024,
    overlap: int = 128,
    harmonize_overlaps: bool = True,
    min_harmonization_pixels: int = 256,
) -> GeometrySceneOutput:
    """Run memory-bounded overlapping geometry inference on a full remote-sensing scene.

    Each new relative-height tile may be robustly scale/offset aligned to the already accumulated
    overlap. This suppresses local monocular scale drift before weighted seam blending.
    """
    stats = estimate_rgb_normalization_stats(source_path, band_indices=band_indices)
    with rasterio.open(source_path) as src:
        if src.count < 3:
            raise ValueError("geometry inference requires at least three RGB bands")
        tiles = generate_tiles(src.height, src.width, tile_size=tile_size, overlap=overlap)
        height_acc = WeightedTileAccumulator(src.height, src.width)
        confidence_acc: WeightedTileAccumulator | None = None
        model_id: str | None = None
        harmonized_tiles = 0

        for tile in tiles:
            window = Window.from_slices(
                (tile.y, tile.y + tile.height),
                (tile.x, tile.x + tile.width),
            )
            rgb = src.read(list(band_indices), window=window).astype(np.float32)
            rgb = np.moveaxis(rgb, 0, -1)
            normalized = normalize_with_stats(rgb, stats)
            prediction = prior.infer(normalized)
            if prediction.relative_height.shape != (tile.height, tile.width):
                raise ValueError(
                    f"geometry prior returned {prediction.relative_height.shape} for tile "
                    f"{(tile.height, tile.width)}"
                )
            model_id = prediction.model_id if model_id is None else model_id
            if prediction.model_id != model_id:
                raise ValueError("geometry prior model_id changed within a single scene job")

            relative_tile = prediction.relative_height.astype(np.float32, copy=False)
            if harmonize_overlaps:
                existing, overlap_mask = height_acc.current_region(
                    tile.y, tile.x, tile.height, tile.width
                )
                relative_tile, changed = _harmonize_tile(
                    relative_tile,
                    existing,
                    overlap_mask,
                    min_overlap_pixels=min_harmonization_pixels,
                )
                harmonized_tiles += int(changed)
            height_acc.add(relative_tile, tile.y, tile.x)

            if prediction.confidence is not None:
                if prediction.confidence.shape != prediction.relative_height.shape:
                    raise ValueError("geometry confidence must match relative-height shape")
                if confidence_acc is None:
                    confidence_acc = WeightedTileAccumulator(src.height, src.width)
                confidence_acc.add(prediction.confidence, tile.y, tile.x)

    return GeometrySceneOutput(
        relative_height=height_acc.finalize(),
        confidence=confidence_acc.finalize() if confidence_acc is not None else None,
        model_id=model_id or "unknown",
        normalization=stats,
        tile_count=len(tiles),
        harmonized_tiles=harmonized_tiles,
    )
