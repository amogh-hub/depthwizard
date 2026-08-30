from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.preprocess.stats import (
    RGBNormalizationStats,
    estimate_rgb_normalization_stats,
    normalize_with_stats,
)
from depthwizard.tiling.blend import WeightedTileAccumulator
from depthwizard.tiling.grid import TileWindow, generate_tiles

from .scaffold import (
    align_tile_to_scaffold,
    high_frequency_residual,
    resize_field_to_shape,
)

_SCENE_NORMALIZE_METADATA_KEY = "scene_normalize_relative_height"
_SCENE_SCAFFOLD_METADATA_KEY = "scene_global_scaffold"
_GLOBAL_SCAFFOLD_MAX_EDGE = 1536
_ALIGNMENT_SIGMA_FRACTION = 0.125
_RESIDUAL_SIGMA_FRACTION = 0.25
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


def _bool_metadata(prediction: GeometryPriorOutput, key: str) -> bool:
    value = prediction.metadata.get(key, False)
    if not isinstance(value, bool):
        raise TypeError(f"geometry prior metadata {key!r} must be boolean")
    return value


def _read_and_infer_tile(
    src: rasterio.io.DatasetReader,
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


def _overview_shape(height: int, width: int, max_edge: int) -> tuple[int, int]:
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    scale = min(1.0, max_edge / max(height, width))
    return max(2, round(height * scale)), max(2, round(width * scale))


def _infer_global_scaffold(
    src: rasterio.io.DatasetReader,
    prior: GeometryPrior,
    stats: RGBNormalizationStats,
    band_indices: tuple[int, int, int],
    *,
    expected_model_id: str,
) -> GeometryPriorOutput:
    overview_height, overview_width = _overview_shape(
        src.height,
        src.width,
        _GLOBAL_SCAFFOLD_MAX_EDGE,
    )
    rgb = src.read(
        list(band_indices),
        out_shape=(len(band_indices), overview_height, overview_width),
        resampling=Resampling.bilinear,
    ).astype(np.float32)
    rgb = np.moveaxis(rgb, 0, -1)
    normalized = normalize_with_stats(rgb, stats)
    prediction = prior.infer(normalized)
    if prediction.model_id != expected_model_id:
        raise ValueError("geometry prior model_id changed between tile and global-scaffold inference")
    if prediction.relative_height.shape != (overview_height, overview_width):
        raise ValueError(
            "geometry prior returned an unexpected global-scaffold shape: "
            f"{prediction.relative_height.shape} != {(overview_height, overview_width)}"
        )
    return prediction


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

    Priors may request a scene-global scaffold. For those priors, one overview inference establishes
    the scene-wide low-frequency geometry. High-resolution tiles are aligned to that scaffold and
    contribute only high-frequency residual detail, preventing independent tile context from
    manufacturing broad tile-sized plateaus. Other priors retain the overlap-harmonized mosaic path.
    Scene-global normalization, when requested, remains the final scientific affine convention.
    """
    stats = estimate_rgb_normalization_stats(source_path, band_indices=band_indices)
    with rasterio.open(source_path) as src:
        if src.count < 3:
            raise ValueError("geometry inference requires at least three RGB bands")
        tiles = generate_tiles(src.height, src.width, tile_size=tile_size, overlap=overlap)
        first_prediction = _read_and_infer_tile(src, tiles[0], prior, stats, band_indices)
        model_id = first_prediction.model_id
        scene_normalization_required = _bool_metadata(
            first_prediction,
            _SCENE_NORMALIZE_METADATA_KEY,
        )
        scaffold_requested = _bool_metadata(
            first_prediction,
            _SCENE_SCAFFOLD_METADATA_KEY,
        )
        use_scaffold = scaffold_requested and len(tiles) > 1

        scaffold: np.ndarray | None = None
        residual_acc: WeightedTileAccumulator | None = None
        height_acc: WeightedTileAccumulator | None = None
        if use_scaffold:
            overview_prediction = _infer_global_scaffold(
                src,
                prior,
                stats,
                band_indices,
                expected_model_id=model_id,
            )
            if _bool_metadata(overview_prediction, _SCENE_NORMALIZE_METADATA_KEY) != (
                scene_normalization_required
            ):
                raise ValueError("geometry prior scene-normalization contract changed for scaffold")
            if not _bool_metadata(overview_prediction, _SCENE_SCAFFOLD_METADATA_KEY):
                raise ValueError("geometry prior disabled the scaffold contract during overview inference")
            scaffold = resize_field_to_shape(
                overview_prediction.relative_height,
                (src.height, src.width),
            )
            residual_acc = WeightedTileAccumulator(src.height, src.width)
        else:
            height_acc = WeightedTileAccumulator(src.height, src.width)

        confidence_acc: WeightedTileAccumulator | None = None
        harmonized_tiles = 0

        for index, tile in enumerate(tiles):
            prediction = (
                first_prediction
                if index == 0
                else _read_and_infer_tile(src, tile, prior, stats, band_indices)
            )
            if prediction.model_id != model_id:
                raise ValueError("geometry prior model_id changed within a single scene job")
            if _bool_metadata(prediction, _SCENE_NORMALIZE_METADATA_KEY) != (
                scene_normalization_required
            ):
                raise ValueError("geometry prior scene-normalization contract changed within a job")
            if _bool_metadata(prediction, _SCENE_SCAFFOLD_METADATA_KEY) != scaffold_requested:
                raise ValueError("geometry prior scaffold contract changed within a job")

            relative_tile = prediction.relative_height.astype(np.float32, copy=False)
            if use_scaffold:
                assert scaffold is not None and residual_acc is not None
                scaffold_tile = scaffold[
                    tile.y : tile.y + tile.height,
                    tile.x : tile.x + tile.width,
                ]
                alignment_sigma = max(
                    8.0,
                    min(tile.height, tile.width) * _ALIGNMENT_SIGMA_FRACTION,
                )
                residual_sigma = max(
                    16.0,
                    min(tile.height, tile.width) * _RESIDUAL_SIGMA_FRACTION,
                )
                if harmonize_overlaps:
                    aligned_tile, changed = align_tile_to_scaffold(
                        relative_tile,
                        scaffold_tile,
                        lowpass_sigma_px=alignment_sigma,
                        min_pixels=min_harmonization_pixels,
                    )
                    harmonized_tiles += int(changed)
                else:
                    aligned_tile = relative_tile
                residual = high_frequency_residual(
                    aligned_tile,
                    scaffold_tile,
                    sigma_px=residual_sigma,
                )
                residual_acc.add(residual, tile.y, tile.x)
            else:
                assert height_acc is not None
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

    if use_scaffold:
        assert scaffold is not None and residual_acc is not None
        residual_field = residual_acc.finalize(nodata=0.0)
        relative_height = (scaffold + residual_field).astype(np.float32)
    else:
        assert height_acc is not None
        relative_height = height_acc.finalize()

    if scene_normalization_required:
        relative_height = normalize_relative_height_scene(relative_height)

    return GeometrySceneOutput(
        relative_height=relative_height,
        confidence=confidence_acc.finalize() if confidence_acc is not None else None,
        model_id=model_id,
        normalization=stats,
        tile_count=len(tiles),
        harmonized_tiles=harmonized_tiles,
    )
