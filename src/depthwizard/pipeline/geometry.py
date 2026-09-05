from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.cancellation import CancellationProbe, raise_if_cancelled
from depthwizard.geometry_prior.base import GeometryPrior
from depthwizard.preprocess.stats import (
    RGBNormalizationStats,
    estimate_rgb_normalization_stats,
    normalize_with_stats,
)
from depthwizard.tiling.blend import WeightedTileAccumulator
from depthwizard.tiling.grid import generate_tiles

_SCENE_NORMALIZE_METADATA_KEY = "scene_normalize_relative_height"
_MIN_SCALE_CORRELATION = 0.35
_MIN_AFFINE_ERROR_IMPROVEMENT = 0.10
_MIN_HARMONIZATION_SCALE = 0.25
_MAX_HARMONIZATION_SCALE = 4.0
# Persisted rDSM artifacts are reusable only under this exact algorithm contract. Bump whenever
# masking, tiling, harmonization, normalization, or prior-output interpretation changes.
GEOMETRY_PIPELINE_CONTRACT = "depthwizard.geometry.v2_nodata_affine_scene_normalization"


@dataclass(frozen=True)
class GeometrySceneOutput:
    relative_height: np.ndarray
    confidence: np.ndarray | None
    model_id: str
    normalization: RGBNormalizationStats
    tile_count: int
    harmonized_tiles: int
    valid_pixel_fraction: float


def normalize_relative_height_scene(
    values: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Apply one robust affine relative-height convention to an assembled scene.

    Scene-global normalization is deliberately performed *after* tile harmonization. This preserves
    inter-tile low-frequency evidence during mosaicking and prevents locally flat tiles from being
    stretched to the same apparent relief as genuinely high-relief tiles.

    P01 and P99 define the affine origin and scale, but values are intentionally **not clipped** to
    that interval. Genuine high/low scene extrema must survive for structural-height analysis and
    downstream metric calibration; clipping would silently flatten the very objects DepthWizard is
    intended to reconstruct.
    """
    if not (0.0 <= low_percentile < high_percentile <= 100.0):
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    field = np.asarray(values, dtype=np.float32)
    if field.ndim != 2:
        raise ValueError("relative-height scene must be a 2D raster")
    valid = np.isfinite(field)
    if not np.any(valid):
        raise ValueError("relative-height scene contains no finite values")

    lo, hi = np.percentile(field[valid], [low_percentile, high_percentile])
    span = float(hi - lo)
    out = np.full(field.shape, np.nan, dtype=np.float32)
    if span <= 1e-6:
        # A genuinely near-constant scene must stay flat; inventing contrast would create relief.
        out[valid] = 0.0
        return out

    out[valid] = ((field[valid] - lo) / span).astype(np.float32)
    return out


def _robust_span(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return 0.0
    low, high = np.percentile(finite, [5.0, 95.0])
    return float(high - low)


def _pearson_correlation(x: np.ndarray, y: np.ndarray) -> float | None:
    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.size < 2 or y64.size != x64.size:
        return None
    x_centered = x64 - np.mean(x64)
    y_centered = y64 - np.mean(y64)
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 1e-12:
        return None
    correlation = float(np.dot(x_centered, y_centered) / denominator)
    return correlation if np.isfinite(correlation) else None


def _offset_aligned_tile(
    tile_height: np.ndarray,
    tile_values: np.ndarray,
    existing_values: np.ndarray,
) -> tuple[np.ndarray, bool, float]:
    offset = float(np.median(existing_values - tile_values))
    if not np.isfinite(offset):
        return tile_height, False, 0.0
    aligned = (tile_height + offset).astype(np.float32)
    changed = abs(offset) > 1e-8
    return aligned, changed, offset


def _harmonize_tile(
    tile_height: np.ndarray,
    existing: np.ndarray,
    overlap_mask: np.ndarray,
    *,
    min_overlap_pixels: int,
) -> tuple[np.ndarray, bool]:
    """Align one tile to accumulated overlap without manufacturing tile-scale relief.

    Offset alignment is the safe baseline because monocular predictions may carry an arbitrary
    additive shift. A full affine scale correction is accepted only when the overlap contains real
    variation, the two predictions are positively correlated, the scale is bounded, and the affine
    fit materially improves median overlap error over offset-only alignment. This prevents a weak or
    nearly flat overlap from rescaling an entire 1024-pixel tile and imprinting the inference grid.
    """
    mask = overlap_mask & np.isfinite(tile_height) & np.isfinite(existing)
    if int(mask.sum()) < min_overlap_pixels:
        return tile_height, False

    tile_values = np.asarray(tile_height[mask], dtype=np.float64)
    existing_values = np.asarray(existing[mask], dtype=np.float64)
    offset_tile, offset_changed, offset = _offset_aligned_tile(
        tile_height,
        tile_values,
        existing_values,
    )

    tile_span = _robust_span(tile_values)
    existing_span = _robust_span(existing_values)
    if tile_span <= 1e-6 or existing_span <= 1e-6:
        return offset_tile, offset_changed

    correlation = _pearson_correlation(tile_values, existing_values)
    if correlation is None or correlation < _MIN_SCALE_CORRELATION:
        return offset_tile, offset_changed

    try:
        fit = robust_affine_calibration(
            tile_values,
            existing_values,
            require_positive_scale=True,
        )
    except ValueError:
        return offset_tile, offset_changed

    if not (_MIN_HARMONIZATION_SCALE <= fit.scale <= _MAX_HARMONIZATION_SCALE):
        return offset_tile, offset_changed

    offset_residual = np.abs(existing_values - (tile_values + offset))
    affine_residual = np.abs(existing_values - (fit.scale * tile_values + fit.offset))
    offset_medae = float(np.median(offset_residual))
    affine_medae = float(np.median(affine_residual))
    if not np.isfinite(offset_medae) or not np.isfinite(affine_medae):
        return offset_tile, offset_changed
    if offset_medae <= 1e-9:
        return offset_tile, offset_changed
    required_max_error = offset_medae * (1.0 - _MIN_AFFINE_ERROR_IMPROVEMENT)
    if affine_medae > required_max_error:
        return offset_tile, offset_changed

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
    cancellation_probe: CancellationProbe | None = None,
) -> GeometrySceneOutput:
    """Run memory-bounded overlapping geometry inference on a full remote-sensing scene.

    Each new relative-height tile may be robustly aligned to the already accumulated overlap. A
    geometry prior may additionally request scene-global normalization through explicit metadata;
    that normalization occurs only after all affine-preserving tile evidence has been harmonized and
    blended.
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
        scene_normalization_required: bool | None = None
        source_valid = np.zeros((src.height, src.width), dtype=bool)

        for tile in tiles:
            raise_if_cancelled(cancellation_probe)
            window = Window.from_slices(
                (tile.y, tile.y + tile.height),
                (tile.x, tile.x + tile.width),
            )
            masked_rgb = src.read(list(band_indices), window=window, masked=True)
            tile_valid = ~np.any(np.ma.getmaskarray(masked_rgb), axis=0)
            rgb_bands = np.asarray(masked_rgb.filled(0.0), dtype=np.float32)
            tile_valid &= np.all(np.isfinite(rgb_bands), axis=0)
            source_valid[
                tile.y : tile.y + tile.height,
                tile.x : tile.x + tile.width,
            ] |= tile_valid
            if not np.any(tile_valid):
                continue
            rgb = rgb_bands
            rgb = np.moveaxis(rgb, 0, -1)
            normalized = normalize_with_stats(rgb, stats)
            prediction = prior.infer(normalized)
            raise_if_cancelled(cancellation_probe)
            if prediction.relative_height.shape != (tile.height, tile.width):
                raise ValueError(
                    f"geometry prior returned {prediction.relative_height.shape} for tile "
                    f"{(tile.height, tile.width)}"
                )
            model_id = prediction.model_id if model_id is None else model_id
            if prediction.model_id != model_id:
                raise ValueError("geometry prior model_id changed within a single scene job")

            requested_scene_normalization = prediction.metadata.get(
                _SCENE_NORMALIZE_METADATA_KEY,
                False,
            )
            if not isinstance(requested_scene_normalization, bool):
                raise TypeError(
                    f"geometry prior metadata {_SCENE_NORMALIZE_METADATA_KEY!r} must be boolean"
                )
            if scene_normalization_required is None:
                scene_normalization_required = requested_scene_normalization
            elif requested_scene_normalization != scene_normalization_required:
                raise ValueError("geometry prior scene-normalization contract changed within a job")

            relative_tile = prediction.relative_height.astype(np.float32, copy=True)
            # The model necessarily receives finite padding for invalid pixels, but its predictions
            # there are never scientific data. Restore the source validity contract before overlap
            # harmonization so NoData borders/holes cannot influence or appear in the rDSM/DSM.
            relative_tile[~tile_valid] = np.nan
            if harmonize_overlaps:
                existing, overlap_mask = height_acc.current_region(
                    tile.y,
                    tile.x,
                    tile.height,
                    tile.width,
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
                confidence_tile = prediction.confidence.astype(np.float32, copy=True)
                confidence_tile[~tile_valid] = np.nan
                confidence_acc.add(confidence_tile, tile.y, tile.x)

    relative_height = height_acc.finalize()
    raise_if_cancelled(cancellation_probe)
    if scene_normalization_required:
        relative_height = normalize_relative_height_scene(relative_height)

    return GeometrySceneOutput(
        relative_height=relative_height,
        confidence=confidence_acc.finalize() if confidence_acc is not None else None,
        model_id=model_id or "unknown",
        normalization=stats,
        tile_count=len(tiles),
        harmonized_tiles=harmonized_tiles,
        valid_pixel_fraction=float(np.mean(source_valid)),
    )
