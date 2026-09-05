from __future__ import annotations

import json
import os
import shutil
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from depthwizard.calibration.evidence import calibrate_relative_height_with_dem
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.evaluation.potsdam import (
    COPERNICUS_GLO30_EFFECTIVE_GSD_M,
    COPERNICUS_GLO30_FILENAME,
    COPERNICUS_GLO30_URL,
    FROZEN_POTSDAM_TILE_IDS,
    POTSDAM_BENCHMARK_GSD_M,
    POTSDAM_CRS,
    POTSDAM_NATIVE_GSD_M,
    POTSDAM_NATIVE_SHAPE,
    PotsdamTilePaths,
    inspect_potsdam_rgb_contract,
    resolve_potsdam_tile_paths,
    write_or_verify_protocol_seal,
)
from depthwizard.geometry_prior.da3 import (
    DA3_CHECKPOINT_SHA256,
    DA3_HF_REVISION,
    DA3_MODEL_ID,
    DA3MonocularPrior,
)
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.training import fit_rgb_ranges, normalize_rgb, patch_windows
from depthwizard.io.raster import read_rgb, write_float_geotiff
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.provenance.manifest import canonical_json_hash, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("DEPTHWIZARD_POTSDAM_ROOT", ROOT / "data" / "external" / "isprs-potsdam")
)
COPDEM_PATH = ROOT / "data" / "external" / "copernicus-dem" / COPERNICUS_GLO30_FILENAME
OUT_DIR = ROOT / "artifacts" / "evaluation" / "potsdam-external-v1"
PROTOCOL_SEAL_PATH = OUT_DIR / "protocol_seal.json"
REPORT_PATH = OUT_DIR / "potsdam_external_report.json"
V4_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-multiscene-v4"
    / "height_model_multiscene_v4.pt"
)
PATCH_SIZE = 192
PATCH_STRIDE = 96
BATCH_SIZE = 2
LOW_FREQUENCY_BIAS_SIGMA_PX = 120.0
DA3_PINNED_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
ISPRS_POTSDAM_SOURCE = (
    "https://isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"
)
COPERNICUS_SOURCE = "https://registry.opendata.aws/copernicus-dem/"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_v4_model(device: torch.device) -> tuple[DepthWizardHeightModel, dict[str, Any]]:
    if not V4_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "frozen V4 checkpoint is missing; run `make height-multiscene-v4` only if the original "
            f"checkpoint was never produced: {V4_CHECKPOINT}"
        )
    checkpoint = torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("V4 checkpoint payload is not a dictionary")
    config_payload = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise TypeError("V4 checkpoint is missing config/state_dict dictionaries")
    config = HeightModelConfig(**config_payload)
    if config.architecture_version != "confidence-gated-v2":
        raise RuntimeError(
            "Potsdam external protocol requires frozen confidence-gated-v2; "
            f"found {config.architecture_version}"
        )
    best_epoch = checkpoint.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch <= 0:
        raise RuntimeError("V4 checkpoint has no promoted learned epoch")
    model = DepthWizardHeightModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def _number(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"external benchmark field {key!r} is not numeric")
    return float(value)


def _download_atomic(url: str, destination: Path) -> Path:
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "DepthWizard-SIH26175/1.0"})
    print(f"Downloading independent Copernicus GLO-30 calibration tile: {destination.name}")
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    if temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Copernicus GLO-30 download produced an empty file")
    temporary.replace(destination)
    return destination


def _benchmark_grid(rgb_path: Path) -> tuple[int, int, object, list[float]]:
    with rasterio.open(rgb_path) as src:
        width = round(src.width * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        height = round(src.height * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        if width <= 0 or height <= 0:
            raise RuntimeError("invalid Potsdam benchmark dimensions")
        transform = from_bounds(*src.bounds, width=width, height=height)
        bounds = [float(value) for value in src.bounds]
    return height, width, transform, bounds


def _ensure_benchmark_rgb(tile: PotsdamTilePaths) -> Path:
    tile_dir = OUT_DIR / "derived" / tile.tile_id
    output = tile_dir / "rgb_025m.tif"
    expected_height, expected_width, expected_transform, _ = _benchmark_grid(tile.rgb)
    source_rgb_sha256 = sha256_file(tile.rgb)
    derivation_sha256 = canonical_json_hash(
        {
            "source_rgb_sha256": source_rgb_sha256,
            "source_gsd_m": POTSDAM_NATIVE_GSD_M,
            "benchmark_gsd_m": POTSDAM_BENCHMARK_GSD_M,
            "resampling": "average",
            "rgb_bands": [1, 2, 3],
        }
    )
    if output.exists():
        with rasterio.open(output) as src:
            tags = src.tags()
            if (
                src.height == expected_height
                and src.width == expected_width
                and src.count == 3
                and src.crs == POTSDAM_CRS
                and src.transform.almost_equals(expected_transform)
                and tags.get("SOURCE_RGB_SHA256") == source_rgb_sha256
                and tags.get("DERIVATION_SHA256") == derivation_sha256
            ):
                return output

    tile_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(tile.rgb) as src:
        source_crs = src.crs or POTSDAM_CRS
        destination = np.zeros((3, expected_height, expected_width), dtype=np.uint8)
        for band_index in range(1, 4):
            reproject(
                source=rasterio.band(src, band_index),
                destination=destination[band_index - 1],
                src_transform=src.transform,
                src_crs=source_crs,
                dst_transform=expected_transform,
                dst_crs=POTSDAM_CRS,
                resampling=Resampling.average,
            )
    profile = {
        "driver": "GTiff",
        "height": expected_height,
        "width": expected_width,
        "count": 3,
        "dtype": "uint8",
        "crs": POTSDAM_CRS,
        "transform": expected_transform,
        "compress": "deflate",
        "tiled": True,
    }
    temporary = output.with_suffix(".tmp.tif")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(destination)
        dst.update_tags(
            DEPTHWIZARD_PRODUCT="POTSDAM_RGB_BENCHMARK_GRID",
            SOURCE_GSD_M=str(POTSDAM_NATIVE_GSD_M),
            BENCHMARK_GSD_M=str(POTSDAM_BENCHMARK_GSD_M),
            SOURCE_TILE=tile.tile_id,
            SOURCE_RGB_SHA256=source_rgb_sha256,
            DERIVATION_SHA256=derivation_sha256,
        )
    temporary.replace(output)
    return output


def _load_reference_after_seal(
    tile: PotsdamTilePaths,
    benchmark_rgb: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load the ISPRS DSM only after the immutable protocol seal exists."""
    if not PROTOCOL_SEAL_PATH.is_file():
        raise RuntimeError("protocol seal must exist before loading any Potsdam reference DSM")
    with rasterio.open(benchmark_rgb) as target:
        destination = np.full((target.height, target.width), np.nan, dtype=np.float32)
        destination_mask = np.zeros((target.height, target.width), dtype=np.uint8)
        dst_transform = target.transform
        dst_crs = target.crs
    if dst_crs is None:
        raise RuntimeError("derived Potsdam benchmark RGB lost its CRS")

    with rasterio.open(tile.reference_dsm) as src:
        if (src.height, src.width) != POTSDAM_NATIVE_SHAPE:
            raise ValueError(
                f"Potsdam reference DSM must be 6000x6000: {tile.reference_dsm}"
            )
        with rasterio.open(tile.rgb) as rgb_src:
            if not src.transform.almost_equals(rgb_src.transform):
                raise ValueError(
                    f"Potsdam RGB/reference affine mismatch for tile {tile.tile_id}"
                )
        source_crs = src.crs or POTSDAM_CRS
        if source_crs != POTSDAM_CRS:
            raise ValueError(
                f"Potsdam reference CRS must resolve to EPSG:32633; got {source_crs}"
            )
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=source_crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        source_mask = src.read_masks(1)
        reproject(
            source=source_mask,
            destination=destination_mask,
            src_transform=src.transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    valid = (destination_mask > 0) & np.isfinite(destination)
    return destination, valid, sha256_file(tile.reference_dsm)


def _aligned_copdem(benchmark_rgb: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(benchmark_rgb) as target:
        destination = np.full((target.height, target.width), np.nan, dtype=np.float32)
        destination_mask = np.zeros((target.height, target.width), dtype=np.uint8)
        dst_transform = target.transform
        dst_crs = target.crs
    if dst_crs is None:
        raise RuntimeError("benchmark target is missing CRS")
    with rasterio.open(COPDEM_PATH) as src:
        if src.crs is None:
            raise ValueError("Copernicus GLO-30 source unexpectedly lacks CRS metadata")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        source_mask = src.read_masks(1)
        reproject(
            source=source_mask,
            destination=destination_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    valid = (destination_mask > 0) & np.isfinite(destination)
    return destination, valid


def _ensure_da3_geometry(
    tile_id: str,
    benchmark_rgb: Path,
    prior: DA3MonocularPrior,
) -> Path:
    output = OUT_DIR / "derived" / tile_id / "da3_rdsm_025m.tif"
    source_rgb_sha256 = sha256_file(benchmark_rgb)
    geometry_config_sha256 = canonical_json_hash(
        {
            "source_rgb_sha256": source_rgb_sha256,
            "model_id": DA3_MODEL_ID,
            "model_revision": DA3_HF_REVISION,
            "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
            "tile_size": 768,
            "overlap": 128,
            "harmonize_overlaps": True,
            "geometry_pipeline_sha256": sha256_file(
                ROOT / "src" / "depthwizard" / "pipeline" / "geometry.py"
            ),
            "da3_adapter_sha256": sha256_file(
                ROOT / "src" / "depthwizard" / "geometry_prior" / "da3.py"
            ),
        }
    )
    if output.exists():
        with rasterio.open(output) as geometry, rasterio.open(benchmark_rgb) as rgb:
            tags = geometry.tags()
            if (
                geometry.height == rgb.height
                and geometry.width == rgb.width
                and geometry.crs == rgb.crs
                and geometry.transform.almost_equals(rgb.transform)
                and tags.get("SOURCE_RGB_SHA256") == source_rgb_sha256
                and tags.get("GEOMETRY_CONFIG_SHA256") == geometry_config_sha256
            ):
                print(f"Reusing external DA3 geometry: Potsdam {tile_id}")
                return output
    print(f"Generating external DA3 geometry: Potsdam {tile_id}")
    scene = infer_geometry_scene(
        benchmark_rgb,
        prior,
        tile_size=768,
        overlap=128,
        harmonize_overlaps=True,
    )
    write_float_geotiff(
        output,
        scene.relative_height,
        template_path=benchmark_rgb,
        description=f"DepthWizard DA3 geometry prior for ISPRS Potsdam {tile_id}",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": scene.model_id,
            "PURPOSE": "sealed_external_cross_dataset_evaluation",
            "SOURCE_TILE": tile_id,
            "SOURCE_RGB_SHA256": source_rgb_sha256,
            "GEOMETRY_CONFIG_SHA256": geometry_config_sha256,
            "MODEL_REVISION": DA3_HF_REVISION,
            "CHECKPOINT_SHA256": DA3_CHECKPOINT_SHA256,
        },
    )
    return output


def _predict_refined(
    model: DepthWizardHeightModel,
    rgb: np.ndarray,
    geometry: np.ndarray,
    input_valid: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = patch_windows(
        input_valid,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        min_valid_fraction=0.90,
    )
    if not windows:
        raise RuntimeError("Potsdam benchmark produced no valid model-inference windows")
    height, width = geometry.shape
    prediction_sum = np.zeros((height, width), dtype=np.float64)
    uncertainty_sum = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    axis = np.maximum(np.hanning(PATCH_SIZE), 0.05)
    blend = np.outer(axis, axis).astype(np.float64)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[start : start + BATCH_SIZE]
            rgb_items: list[np.ndarray] = []
            geometry_items: list[np.ndarray] = []
            for window in batch_windows:
                mask = input_valid[window.row_slice, window.col_slice]
                rgb_patch = rgb[window.row_slice, window.col_slice].transpose(2, 0, 1)
                geometry_patch = geometry[window.row_slice, window.col_slice][None]
                rgb_items.append(np.where(mask[None], rgb_patch, 0.0).astype(np.float32))
                geometry_items.append(
                    np.where(mask[None], geometry_patch, 0.0).astype(np.float32)
                )
            rgb_tensor = torch.from_numpy(np.stack(rgb_items)).to(
                device=device, dtype=torch.float32
            )
            geometry_tensor = torch.from_numpy(np.stack(geometry_items)).to(
                device=device, dtype=torch.float32
            )
            gsd = torch.full(
                (len(batch_windows),),
                POTSDAM_BENCHMARK_GSD_M,
                device=device,
                dtype=torch.float32,
            )
            output = model(rgb_tensor, geometry_tensor, gsd_m=gsd)
            predictions = output.relative_height[:, 0].cpu().numpy()
            uncertainties = output.uncertainty[:, 0].cpu().numpy()
            for index, window in enumerate(batch_windows):
                prediction_sum[window.row_slice, window.col_slice] += predictions[index] * blend
                uncertainty_sum[window.row_slice, window.col_slice] += uncertainties[index] * blend
                weights[window.row_slice, window.col_slice] += blend

    covered = weights > 0
    prediction = np.full((height, width), np.nan, dtype=np.float32)
    uncertainty = np.full((height, width), np.nan, dtype=np.float32)
    prediction[covered] = (prediction_sum[covered] / weights[covered]).astype(np.float32)
    uncertainty[covered] = (uncertainty_sum[covered] / weights[covered]).astype(np.float32)
    return prediction, uncertainty, covered


def _uncertainty_error_correlation(
    uncertainty: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    mask = valid & np.isfinite(uncertainty) & np.isfinite(prediction) & np.isfinite(reference)
    if int(mask.sum()) < 32:
        return None
    uncertainty_values = uncertainty[mask].astype(np.float64)
    errors = np.abs(prediction[mask].astype(np.float64) - reference[mask].astype(np.float64))
    if float(np.std(uncertainty_values)) <= 1e-12 or float(np.std(errors)) <= 1e-12:
        return None
    return float(np.corrcoef(uncertainty_values, errors)[0, 1])


def _evaluate_tile(
    tile: PotsdamTilePaths,
    model: DepthWizardHeightModel,
    prior: DA3MonocularPrior,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray] | tuple[
    dict[str, object], None, None, None
]:
    started = time.perf_counter()
    benchmark_rgb = _ensure_benchmark_rgb(tile)
    geometry_path = _ensure_da3_geometry(tile.tile_id, benchmark_rgb, prior)
    rgb_raw = read_rgb(benchmark_rgb)
    with rasterio.open(geometry_path) as src:
        geometry = src.read(1).astype(np.float32)
        geometry_valid = np.isfinite(geometry)
        if src.nodata is not None:
            geometry_valid &= geometry != src.nodata
    with rasterio.open(benchmark_rgb) as src:
        rgb_valid = src.dataset_mask() > 0
    input_valid = rgb_valid & geometry_valid
    ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, ranges)
    refined, uncertainty, covered = _predict_refined(model, rgb, geometry, input_valid, device)
    model_valid = input_valid & covered & np.isfinite(refined)

    coarse_dem, coarse_valid = _aligned_copdem(benchmark_rgb)
    calibration_valid = input_valid & coarse_valid
    try:
        da3_calibration = calibrate_relative_height_with_dem(
            geometry,
            coarse_dem,
            dem_valid=calibration_valid,
            low_frequency_sigma_px=LOW_FREQUENCY_BIAS_SIGMA_PX,
            target_gsd_m=POTSDAM_BENCHMARK_GSD_M,
            dem_effective_gsd_m=COPERNICUS_GLO30_EFFECTIVE_GSD_M,
        )
        v4_calibration = calibrate_relative_height_with_dem(
            refined,
            coarse_dem,
            dem_valid=model_valid & coarse_valid,
            low_frequency_sigma_px=LOW_FREQUENCY_BIAS_SIGMA_PX,
            target_gsd_m=POTSDAM_BENCHMARK_GSD_M,
            dem_effective_gsd_m=COPERNICUS_GLO30_EFFECTIVE_GSD_M,
        )
    except ValueError as exc:
        return (
            {
                "tile_id": tile.tile_id,
                "status": "REJECTED_CALIBRATION_EVIDENCE",
                "reason": str(exc),
                "wall_time_seconds": time.perf_counter() - started,
            },
            None,
            None,
            None,
        )

    reference, reference_valid, reference_sha256 = _load_reference_after_seal(tile, benchmark_rgb)
    common = (
        reference_valid
        & model_valid
        & np.isfinite(da3_calibration.dsm)
        & np.isfinite(v4_calibration.dsm)
    )
    if int(common.sum()) < 100_000:
        return (
            {
                "tile_id": tile.tile_id,
                "status": "REJECTED_INSUFFICIENT_INDEPENDENT_REFERENCE",
                "reason": f"only {int(common.sum())} common external-reference pixels",
                "wall_time_seconds": time.perf_counter() - started,
            },
            None,
            None,
            None,
        )
    da3_metrics = compute_elevation_metrics(da3_calibration.dsm, reference, valid_mask=common)
    v4_metrics = compute_elevation_metrics(v4_calibration.dsm, reference, valid_mask=common)
    reliability = _uncertainty_error_correlation(
        uncertainty,
        v4_calibration.dsm,
        reference,
        common,
    )
    report: dict[str, object] = {
        "tile_id": tile.tile_id,
        "status": "EVALUATED",
        "rgb_source": str(tile.rgb.resolve()),
        "reference_source": str(tile.reference_dsm.resolve()),
        "reference_sha256": reference_sha256,
        "benchmark_rgb": str(benchmark_rgb.resolve()),
        "geometry_prior": str(geometry_path.resolve()),
        "benchmark_gsd_m": POTSDAM_BENCHMARK_GSD_M,
        "evaluation_pixels": int(common.sum()),
        "da3": da3_metrics.model_dump(),
        "v4": v4_metrics.model_dump(),
        "v4_rmse_delta_m": float(v4_metrics.rmse_m - da3_metrics.rmse_m),
        "v4_mae_delta_m": float(v4_metrics.mae_m - da3_metrics.mae_m),
        "uncertainty_abs_error_pearson": reliability,
        "da3_calibration": {
            **da3_calibration.calibration.model_dump(),
            "anchor_correlation_before": da3_calibration.anchor_correlation_before,
            "orientation_flipped": da3_calibration.orientation_flipped,
            "frequency_match_sigma_px": da3_calibration.frequency_match_sigma_px,
            "anchor_stride_px": da3_calibration.anchor_stride_px,
            "anchors": int(da3_calibration.anchor_mask.sum()),
        },
        "v4_calibration": {
            **v4_calibration.calibration.model_dump(),
            "anchor_correlation_before": v4_calibration.anchor_correlation_before,
            "orientation_flipped": v4_calibration.orientation_flipped,
            "frequency_match_sigma_px": v4_calibration.frequency_match_sigma_px,
            "anchor_stride_px": v4_calibration.anchor_stride_px,
            "anchors": int(v4_calibration.anchor_mask.sum()),
        },
        "rgb_normalization": [asdict(item) for item in ranges],
        "wall_time_seconds": time.perf_counter() - started,
    }
    return (
        report,
        da3_calibration.dsm[common].astype(np.float32),
        v4_calibration.dsm[common].astype(np.float32),
        reference[common].astype(np.float32),
    )


def _protocol_payload(
    tile_paths: list[PotsdamTilePaths],
    rgb_contracts: list[dict[str, object]],
    checkpoint: dict[str, Any],
) -> dict[str, object]:
    source_hash_paths = (
        Path(__file__),
        ROOT / "src" / "depthwizard" / "evaluation" / "potsdam.py",
        ROOT / "src" / "depthwizard" / "calibration" / "evidence.py",
        ROOT / "src" / "depthwizard" / "height_model" / "model.py",
    )
    return {
        "protocol_version": "potsdam-external-v1",
        "purpose": (
            "Frozen external cross-dataset evaluation. ISPRS Potsdam DSM is evaluation-only and "
            "is never used for model selection, blending, scale calibration, or orientation."
        ),
        "selected_tiles": list(FROZEN_POTSDAM_TILE_IDS),
        "tile_files": [
            {
                "tile_id": tile.tile_id,
                "rgb": str(tile.rgb.resolve()),
                "reference_dsm": str(tile.reference_dsm.resolve()),
                "reference_role": "evaluation_only_after_protocol_seal",
            }
            for tile in tile_paths
        ],
        "rgb_contracts": rgb_contracts,
        "dataset": {
            "name": "ISPRS 2D Semantic Labeling Potsdam",
            "source": ISPRS_POTSDAM_SOURCE,
            "native_gsd_m": POTSDAM_NATIVE_GSD_M,
            "native_shape": list(POTSDAM_NATIVE_SHAPE),
            "rgb_channels": "R-G-B",
            "official_grid": "WGS84 UTM zone 33N",
            "effective_crs": POTSDAM_CRS.to_string(),
            "benchmark_gsd_m": POTSDAM_BENCHMARK_GSD_M,
            "benchmark_resampling": "area_average_native_5cm_to_25cm",
        },
        "calibration_evidence": {
            "name": "Copernicus DEM GLO-30 Public 2021",
            "source_registry": COPERNICUS_SOURCE,
            "object_url": COPERNICUS_GLO30_URL,
            "local_sha256": sha256_file(COPDEM_PATH),
            "role": "metric_calibration_only_not_accuracy_reference",
            "effective_gsd_m": COPERNICUS_GLO30_EFFECTIVE_GSD_M,
            "frequency_matching": True,
            "independent_support_thinning": True,
            "low_frequency_bias_sigma_px": LOW_FREQUENCY_BIAS_SIGMA_PX,
        },
        "model": {
            "depthwizard_checkpoint": str(V4_CHECKPOINT.resolve()),
            "depthwizard_checkpoint_sha256": sha256_file(V4_CHECKPOINT),
            "best_epoch": checkpoint.get("best_epoch"),
            "config": checkpoint.get("config"),
            "da3_model": "depth-anything/DA3MONO-LARGE",
            "da3_pinned_source_commit": DA3_PINNED_COMMIT,
            "learned_refinement_policy": "raw_frozen_v4_no_reference_adaptive_blending",
        },
        "prediction": {
            "patch_size": PATCH_SIZE,
            "patch_stride": PATCH_STRIDE,
            "batch_size": BATCH_SIZE,
            "benchmark_gsd_m": POTSDAM_BENCHMARK_GSD_M,
        },
        "metrics": ["RMSE_m", "MAE_m", "Pearson_r"],
        "predeclared_refiner_promotion": {
            "all_four_tiles_must_evaluate": True,
            "aggregate_rmse_must_improve_over_da3": True,
            "aggregate_mae_must_improve_over_da3": True,
            "every_tile_rmse_non_degrading": True,
        },
        "source_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in source_hash_paths},
    }


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(26175)
    np.random.seed(26175)
    torch.set_float32_matmul_precision("high")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tile_paths = [resolve_potsdam_tile_paths(DATASET_ROOT, tile_id) for tile_id in FROZEN_POTSDAM_TILE_IDS]
    rgb_contracts = [inspect_potsdam_rgb_contract(tile.rgb) for tile in tile_paths]
    _download_atomic(COPERNICUS_GLO30_URL, COPDEM_PATH)
    device = _resolve_device()
    model, checkpoint = _load_v4_model(device)

    protocol = _protocol_payload(tile_paths, rgb_contracts, checkpoint)
    protocol_digest = write_or_verify_protocol_seal(PROTOCOL_SEAL_PATH, protocol)
    print(f"Potsdam external protocol sealed before reference load: {protocol_digest}")
    print(f"Frozen external tiles: {list(FROZEN_POTSDAM_TILE_IDS)}")
    print(
        "Scientific separation: Copernicus GLO-30 is calibration-only; ISPRS Potsdam DSM is "
        "evaluation-only and is loaded only after the seal."
    )
    print(
        "Reference-adaptive blending is disabled for this benchmark: frozen raw V4 and DA3 are "
        "calibrated independently with the same external coarse DEM evidence."
    )

    prior = DA3MonocularPrior(device="auto")
    reports: list[dict[str, object]] = []
    da3_values: list[np.ndarray] = []
    v4_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    for tile in tile_paths:
        scene_report, da3, v4, reference = _evaluate_tile(tile, model, prior, device)
        reports.append(scene_report)
        if da3 is None or v4 is None or reference is None:
            print(
                f"potsdam_external {tile.tile_id}: REJECTED | "
                f"{scene_report.get('reason', 'unknown reason')}"
            )
            continue
        da3_values.append(da3)
        v4_values.append(v4)
        reference_values.append(reference)
        da3_section = scene_report["da3"]
        v4_section = scene_report["v4"]
        if not isinstance(da3_section, dict) or not isinstance(v4_section, dict):
            raise TypeError("external scene metric sections are not dictionaries")
        print(
            f"potsdam_external {tile.tile_id}: DA3 {_number(da3_section, 'rmse_m'):.3f} m | "
            f"V4 {_number(v4_section, 'rmse_m'):.3f} m | "
            f"delta {_number(scene_report, 'v4_rmse_delta_m'):+.3f} m"
        )
    del prior

    all_tiles_evaluated = len(da3_values) == len(FROZEN_POTSDAM_TILE_IDS)
    aggregate_da3 = None
    aggregate_v4 = None
    rmse_improvement_fraction: float | None = None
    mae_improvement_fraction: float | None = None
    every_tile_non_degrading = False
    frozen_refiner_promotion = False
    if da3_values:
        da3_all = np.concatenate(da3_values)
        v4_all = np.concatenate(v4_values)
        reference_all = np.concatenate(reference_values)
        aggregate_da3 = compute_elevation_metrics(da3_all, reference_all)
        aggregate_v4 = compute_elevation_metrics(v4_all, reference_all)
        rmse_improvement_fraction = (
            aggregate_da3.rmse_m - aggregate_v4.rmse_m
        ) / aggregate_da3.rmse_m
        mae_improvement_fraction = (
            aggregate_da3.mae_m - aggregate_v4.mae_m
        ) / aggregate_da3.mae_m
        evaluated_deltas = [
            _number(report, "v4_rmse_delta_m")
            for report in reports
            if report.get("status") == "EVALUATED"
        ]
        every_tile_non_degrading = bool(
            all_tiles_evaluated and evaluated_deltas and max(evaluated_deltas) <= 0.0
        )
        frozen_refiner_promotion = bool(
            all_tiles_evaluated
            and aggregate_v4.rmse_m < aggregate_da3.rmse_m
            and aggregate_v4.mae_m < aggregate_da3.mae_m
            and every_tile_non_degrading
        )

    report_payload: dict[str, object] = {
        "execution_status": "PASS" if all_tiles_evaluated else "PARTIAL_EVIDENCE",
        "frozen_refiner_promotion": frozen_refiner_promotion,
        "protocol_sha256": protocol_digest,
        "protocol_seal": str(PROTOCOL_SEAL_PATH.resolve()),
        "scenes": reports,
        "aggregate_da3": aggregate_da3.model_dump() if aggregate_da3 is not None else None,
        "aggregate_v4": aggregate_v4.model_dump() if aggregate_v4 is not None else None,
        "rmse_improvement_fraction": rmse_improvement_fraction,
        "mae_improvement_fraction": mae_improvement_fraction,
        "all_tiles_evaluated": all_tiles_evaluated,
        "every_tile_rmse_non_degrading": every_tile_non_degrading,
        "scientific_limit": (
            "This is a frozen external urban/cross-dataset benchmark and materially advances Gate B, "
            "but it does not alone close Gate B. Terrain-diverse urban/sparse/hilly/forest evidence, "
            "additional published baselines, ablations, and cross-sensor evidence remain required."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    print("DepthWizard sealed Potsdam external evaluation: COMPLETE")
    print(f"All frozen tiles evaluated: {'YES' if all_tiles_evaluated else 'NO'}")
    if (
        aggregate_da3 is not None
        and aggregate_v4 is not None
        and rmse_improvement_fraction is not None
        and mae_improvement_fraction is not None
    ):
        print(
            f"External aggregate RMSE: DA3 {aggregate_da3.rmse_m:.3f} m | "
            f"V4 {aggregate_v4.rmse_m:.3f} m | "
            f"improvement {100.0 * rmse_improvement_fraction:.2f}%"
        )
        print(
            f"External aggregate MAE: DA3 {aggregate_da3.mae_m:.3f} m | "
            f"V4 {aggregate_v4.mae_m:.3f} m | "
            f"improvement {100.0 * mae_improvement_fraction:.2f}%"
        )
    print(f"Every frozen tile RMSE non-degrading: {'YES' if every_tile_non_degrading else 'NO'}")
    print(f"Frozen external refiner promotion: {'YES' if frozen_refiner_promotion else 'NO'}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
