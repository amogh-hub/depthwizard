from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.windows import Window

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.preprocess.stats import (
    RGBNormalizationStats,
    estimate_rgb_normalization_stats,
    normalize_with_stats,
)
from depthwizard.tiling.blend import WeightedTileAccumulator
from depthwizard.tiling.grid import TileWindow, generate_shifted_tiles, generate_tiles

_SCENE_NORMALIZE_METADATA_KEY = "scene_normalize_relative_height"
_DUAL_LATTICE_METADATA_KEY = "dual_lattice_mosaic"
_MIN_SCALE_CORRELATION = 0.35
_MIN_AFFINE_ERROR_IMPROVEMENT = 0.10
_MIN_HARMONIZATION_SCALE = 0.25
_MAX_HARMONIZATION_SCALE = 4.0


@dataclass(frozen=True)
class GeometrySceneOutput:
    relative_height: np.ndarray
    confidence: np.ndarray | None
    model_id: str
    normalization: RGBNormalizationStats
    tile_count: int
    harmonized_tiles: int


def normalize_relative_height_scene(
    values: np.ndarray,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
) -> np.ndarray:
    """Apply one robust affine relative-height convention to an assembled scene.

    Scene-global normalization is deliberately performed *after* tile harmonization and any
    multi-lattice ensemble. This preserves inter-tile/inter-lattice low-frequency evidence and
    prevents locally flat tiles from being stretched to the same apparent relief as genuinely
    high-relief tiles.

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
    """Align one tile to accumulated overlap without manufacturing tile-scale relief."""
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


def _bool_metadata(prediction: GeometryPriorOutput, key: str) -> bool:
    value = prediction.metadata.get(key, False)
    if not isinstance(value, bool):
        raise TypeError(f"geometry prior metadata {key!r} must be boolean")
    return value


def _infer_tile(
    src: DatasetReader,
    tile: TileWindow,
    prior: GeometryPrior,
    stats: RGBNormalizationStats,
    band_indices: tuple[int, int, int],
) -> GeometryPriorOutput:
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
    return prediction


def _validate_prediction_contract(
    prediction: GeometryPriorOutput,
    *,
    model_id: str,
    scene_normalization_required: bool,
    dual_lattice_requested: bool,
) -> None:
    if prediction.model_id != model_id:
        raise ValueError("geometry prior model_id changed within a single scene job")
    if _bool_metadata(prediction, _SCENE_NORMALIZE_METADATA_KEY) != scene_normalization_required:
        raise ValueError("geometry prior scene-normalization contract changed within a job")
    if _bool_metadata(prediction, _DUAL_LATTICE_METADATA_KEY) != dual_lattice_requested:
        raise ValueError("geometry prior dual-lattice contract changed within a job")


def _assemble_lattice(
    src: DatasetReader,
    tiles: list[TileWindow],
    prior: GeometryPrior,
    stats: RGBNormalizationStats,
    band_indices: tuple[int, int, int],
    *,
    model_id: str,
    scene_normalization_required: bool,
    dual_lattice_requested: bool,
    harmonize_overlaps: bool,
    min_harmonization_pixels: int,
    first_prediction: GeometryPriorOutput | None = None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    height_acc = WeightedTileAccumulator(src.height, src.width)
    confidence_acc: WeightedTileAccumulator | None = None
    harmonized_tiles = 0

    for index, tile in enumerate(tiles):
        prediction = (
            first_prediction
            if index == 0 and first_prediction is not None
            else _infer_tile(src, tile, prior, stats, band_indices)
        )
        _validate_prediction_contract(
            prediction,
            model_id=model_id,
            scene_normalization_required=scene_normalization_required,
            dual_lattice_requested=dual_lattice_requested,
        )

        relative_tile = prediction.relative_height.astype(np.float32, copy=False)
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
            confidence_acc.add(prediction.confidence, tile.y, tile.x)

    return (
        height_acc.finalize(),
        confidence_acc.finalize() if confidence_acc is not None else None,
        harmonized_tiles,
    )


def _align_complete_mosaic(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    min_pixels: int = 4096,
) -> tuple[np.ndarray, bool]:
    """Robustly align one complete affine monocular mosaic to another before ensembling."""
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("candidate and reference mosaics must be same-shaped 2D rasters")
    valid = np.isfinite(candidate) & np.isfinite(reference)
    if int(valid.sum()) < min_pixels:
        raise ValueError("dual-lattice mosaics do not share enough finite support")

    step = max(1, min(candidate.shape) // 512)
    sampled_mask = valid[::step, ::step]
    candidate_values = candidate[::step, ::step][sampled_mask].astype(np.float64)
    reference_values = reference[::step, ::step][sampled_mask].astype(np.float64)
    if candidate_values.size < min_pixels:
        candidate_values = candidate[valid].astype(np.float64)
        reference_values = reference[valid].astype(np.float64)

    offset = float(np.median(reference_values - candidate_values))
    offset_aligned = (candidate + offset).astype(np.float32)
    candidate_span = _robust_span(candidate_values)
    reference_span = _robust_span(reference_values)
    correlation = _pearson_correlation(candidate_values, reference_values)
    if (
        candidate_span <= 1e-6
        or reference_span <= 1e-6
        or correlation is None
        or correlation < _MIN_SCALE_CORRELATION
    ):
        return offset_aligned, abs(offset) > 1e-8

    try:
        fit = robust_affine_calibration(
            candidate_values,
            reference_values,
            require_positive_scale=True,
        )
    except ValueError:
        return offset_aligned, abs(offset) > 1e-8
    if not (_MIN_HARMONIZATION_SCALE <= fit.scale <= _MAX_HARMONIZATION_SCALE):
        return offset_aligned, abs(offset) > 1e-8

    offset_error = float(np.median(np.abs(reference_values - (candidate_values + offset))))
    affine_error = float(
        np.median(np.abs(reference_values - (fit.scale * candidate_values + fit.offset)))
    )
    if not np.isfinite(offset_error) or not np.isfinite(affine_error):
        return offset_aligned, abs(offset) > 1e-8
    if offset_error <= 1e-9:
        return offset_aligned, abs(offset) > 1e-8
    if affine_error > offset_error * (1.0 - _MIN_AFFINE_ERROR_IMPROVEMENT):
        return offset_aligned, abs(offset) > 1e-8
    return (fit.scale * candidate + fit.offset).astype(np.float32), True


def _average_surfaces(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    if primary.shape != secondary.shape:
        raise ValueError("dual-lattice surfaces must have identical shape")
    primary_valid = np.isfinite(primary)
    secondary_valid = np.isfinite(secondary)
    output = np.full(primary.shape, np.nan, dtype=np.float32)
    both = primary_valid & secondary_valid
    output[both] = (0.5 * (primary[both] + secondary[both])).astype(np.float32)
    only_primary = primary_valid & ~secondary_valid
    only_secondary = secondary_valid & ~primary_valid
    output[only_primary] = primary[only_primary]
    output[only_secondary] = secondary[only_secondary]
    return output


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
    """Run memory-bounded geometry inference with optional shifted-lattice ensembling.

    Priors may explicitly request a dual-lattice mosaic. The primary and half-stride-shifted grids
    are assembled independently in affine monocular space, the shifted complete mosaic is robustly
    aligned to the primary mosaic, and both are averaged before the single scene-global relative
    normalization. A fixed tile-context artifact therefore appears at different scene coordinates in
    each lattice instead of surviving phase-locked to one inference grid.
    """
    stats = estimate_rgb_normalization_stats(source_path, band_indices=band_indices)
    with rasterio.open(source_path) as src:
        if src.count < 3:
            raise ValueError("geometry inference requires at least three RGB bands")

        primary_tiles = generate_tiles(src.height, src.width, tile_size=tile_size, overlap=overlap)
        first_prediction = _infer_tile(src, primary_tiles[0], prior, stats, band_indices)
        model_id = first_prediction.model_id
        scene_normalization_required = _bool_metadata(
            first_prediction,
            _SCENE_NORMALIZE_METADATA_KEY,
        )
        dual_lattice_requested = _bool_metadata(first_prediction, _DUAL_LATTICE_METADATA_KEY)
        use_dual_lattice = dual_lattice_requested and len(primary_tiles) > 1

        primary_height, primary_confidence, primary_harmonized = _assemble_lattice(
            src,
            primary_tiles,
            prior,
            stats,
            band_indices,
            model_id=model_id,
            scene_normalization_required=scene_normalization_required,
            dual_lattice_requested=dual_lattice_requested,
            harmonize_overlaps=harmonize_overlaps,
            min_harmonization_pixels=min_harmonization_pixels,
            first_prediction=first_prediction,
        )

        relative_height = primary_height
        confidence = primary_confidence
        tile_count = len(primary_tiles)
        harmonized_tiles = primary_harmonized

        if use_dual_lattice:
            shifted_tiles = generate_shifted_tiles(
                src.height,
                src.width,
                tile_size=tile_size,
                overlap=overlap,
            )
            secondary_height, secondary_confidence, secondary_harmonized = _assemble_lattice(
                src,
                shifted_tiles,
                prior,
                stats,
                band_indices,
                model_id=model_id,
                scene_normalization_required=scene_normalization_required,
                dual_lattice_requested=dual_lattice_requested,
                harmonize_overlaps=harmonize_overlaps,
                min_harmonization_pixels=min_harmonization_pixels,
            )
            secondary_height, lattice_changed = _align_complete_mosaic(
                secondary_height,
                primary_height,
            )
            relative_height = _average_surfaces(primary_height, secondary_height)
            if primary_confidence is not None and secondary_confidence is not None:
                confidence = _average_surfaces(primary_confidence, secondary_confidence)
            elif secondary_confidence is not None:
                confidence = secondary_confidence
            tile_count += len(shifted_tiles)
            harmonized_tiles += secondary_harmonized + int(lattice_changed)

    if scene_normalization_required:
        relative_height = normalize_relative_height_scene(relative_height)

    return GeometrySceneOutput(
        relative_height=relative_height,
        confidence=confidence,
        model_id=model_id,
        normalization=stats,
        tile_count=tile_count,
        harmonized_tiles=harmonized_tiles,
    )
