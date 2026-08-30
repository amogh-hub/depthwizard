from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import (
    compute_elevation_metrics,
    compute_slope_metrics,
)
from depthwizard.evaluation.potsdam import (
    POTSDAM_BENCHMARK_GSD_M,
    POTSDAM_CRS,
    POTSDAM_NATIVE_GSD_M,
    benchmark_full_coverage_mask,
    inspect_potsdam_reference_contract,
    resolve_potsdam_tile_paths,
)
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.structure_band import (
    StructureBandConfig,
    physical_highpass_numpy,
    project_structure_correction_numpy,
)
from depthwizard.height_model.training import (
    PatchWindow,
    PriorReferenceFit,
    RobustRange,
    canonicalize_reference_to_prior,
    fit_reference_to_prior,
    fit_rgb_ranges,
    normalize_rgb,
    patch_windows,
)
from depthwizard.height_model.urban_structure import (
    UrbanStructureConfig,
    compute_urban_structure_loss,
)
from depthwizard.io.raster import write_float_geotiff
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training" / "urban-structure-v6"
GEOMETRY_DIR = OUT_DIR / "geometry-native-v2"
DATASET_ROOT = ROOT / "data" / "external" / "isprs-potsdam"
CHECKPOINT_PATH = OUT_DIR / "height_model_urban_structure_v6.pt"
REPORT_PATH = OUT_DIR / "urban_structure_v6_training_report.json"

SEED = 26175
EPOCHS = 18
BATCH_SIZE = 2
LEARNING_RATE = 4.0e-5
PATCH_SIZE = 192
PATCH_STRIDE = 96
SAMPLES_PER_DOMAIN = 600
ANCHOR_COUNT = 64

URBAN_TRAIN_TILE_IDS = ("2_10", "3_13", "5_11", "6_14")
URBAN_VALIDATION_TILE_ID = "3_14"
URBAN_DEVELOPMENT_TILE_ID = "2_14"
BLIND_TILE_IDS = ("4_12", "6_12")

V5_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-structure-band-v5"
    / "height_model_structure_band_v5.pt"
)
EXPECTED_V5_SHA256 = "3fcc1423abaffdeedea5f56ef360866e8d72d32d36d6452d559e40ae070b8abf"

EXPOSED_V2_ROOT = (
    ROOT / "workspace" / "urban-mosaic-corrective" / "5e87670-potsdam-2_14"
)
EXPECTED_V2_RDSM_SHA256 = "620b0430d0b22c7854733cc61bddd319f4769d9d273f87d29baa7a396764a35e"
EXPECTED_V2_DSM_SHA256 = "8bae324c5c6732d92dacd4af0bb321849a85eece0792f80526f369356ef59fe7"
EXPECTED_2_14_REFERENCE_SHA256 = (
    "fdac03cdee3eb36ccf194f7782dc729150385ef0ea820817800ea83c65447046"
)

URBAN_STRUCTURE = UrbanStructureConfig(
    fine_scale_m=2.0,
    mid_scale_m=4.0,
    coarse_scale_m=8.0,
    safety_scale_m=16.0,
    active_threshold_m=0.75,
    max_structure_weight=6.0,
)
SAFETY_STRUCTURE = StructureBandConfig(
    target_outer_scale_m=URBAN_STRUCTURE.coarse_scale_m,
    safety_outer_scale_m=URBAN_STRUCTURE.safety_scale_m,
    structure_threshold_m=URBAN_STRUCTURE.active_threshold_m,
    max_structure_weight=URBAN_STRUCTURE.max_structure_weight,
)


@dataclass
class SceneData:
    scene_id: str
    location_id: str
    domain: str
    role: str
    rgb: np.ndarray
    geometry: np.ndarray
    reference_m: np.ndarray
    input_valid: np.ndarray
    supervision_valid: np.ndarray
    gsd_m: float
    rgb_ranges: tuple[RobustRange, RobustRange, RobustRange]
    target_prior: np.ndarray | None
    prior_reference_fit: PriorReferenceFit | None
    windows: list[PatchWindow]


@dataclass(frozen=True)
class IndexedPatch:
    scene_index: int
    window: PatchWindow


def _assert_protocol_partition() -> None:
    exposed = set(URBAN_TRAIN_TILE_IDS) | {
        URBAN_VALIDATION_TILE_ID,
        URBAN_DEVELOPMENT_TILE_ID,
    }
    blind = set(BLIND_TILE_IDS)
    if exposed & blind:
        raise RuntimeError("blind Potsdam tiles overlap the exposed V6 protocol")
    if len(exposed) != len(URBAN_TRAIN_TILE_IDS) + 2:
        raise RuntimeError("duplicate exposed Potsdam tile in V6 protocol")


def _resolve_device() -> torch.device:
    return legacy.resolve_device()


def _from_ortholoc(scene: legacy.SceneData, role: str) -> SceneData:
    windows = list(scene.windows or [])
    if not windows:
        raise RuntimeError(f"OrthoLoC scene {scene.scene_id} has no windows")
    return SceneData(
        scene_id=scene.scene_id,
        location_id=scene.location_id,
        domain="ortholoc",
        role=role,
        rgb=scene.rgb,
        geometry=scene.geometry,
        reference_m=scene.reference_m,
        input_valid=scene.input_valid,
        supervision_valid=scene.supervision_valid,
        gsd_m=scene.gsd_m,
        rgb_ranges=scene.rgb_ranges,
        target_prior=scene.target_prior,
        prior_reference_fit=scene.prior_reference_fit,
        windows=windows,
    )


def _ensure_native_v2_geometry(
    tile_id: str,
    rgb_path: Path,
    prior: DA3MonocularPrior,
) -> Path:
    output = GEOMETRY_DIR / f"potsdam_{tile_id}_rdsm_native_v2.tif"
    if output.is_file():
        with rasterio.open(output) as geometry, rasterio.open(rgb_path) as rgb:
            if (
                geometry.width == rgb.width
                and geometry.height == rgb.height
                and geometry.transform.almost_equals(rgb.transform)
            ):
                print(f"Reusing V2 urban geometry: Potsdam {tile_id}")
                return output

    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating V2 urban geometry: Potsdam {tile_id}")
    result = infer_geometry_scene(
        rgb_path,
        prior,
        tile_size=1024,
        overlap=128,
        harmonize_overlaps=True,
    )
    write_float_geotiff(
        output,
        result.relative_height,
        template_path=rgb_path,
        description=f"DepthWizard V2 geometry for exposed Potsdam training tile {tile_id}",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": result.model_id,
            "POTSDAM_TILE": tile_id,
            "PURPOSE": "urban_structure_v6_exposed_training",
        },
    )
    return output


def _target_grid(rgb_path: Path) -> tuple[int, int, Affine]:
    with rasterio.open(rgb_path) as src:
        width = round(src.width * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        height = round(src.height * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        transform = from_bounds(*src.bounds, width=width, height=height)
    if width <= 0 or height <= 0:
        raise RuntimeError("invalid Potsdam 0.25 m working grid")
    return height, width, transform


def _read_rgb_025m(path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, height, width),
            resampling=Resampling.average,
        )
        valid = src.dataset_mask(
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ) > 0
    return np.moveaxis(rgb, 0, -1), valid


def _read_float_025m(
    path: Path,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average,
        )
    values = np.asarray(data.filled(np.nan), dtype=np.float32)
    return values, np.isfinite(values)


def _read_potsdam_reference_025m(
    rgb_path: Path,
    reference_path: Path,
    height: int,
    width: int,
    transform: Affine,
) -> tuple[np.ndarray, np.ndarray]:
    inspect_potsdam_reference_contract(
        rgb_path,
        reference_path,
        max_trailing_edge_deficit_px=1,
    )
    destination = np.full((height, width), np.nan, dtype=np.float32)
    destination_mask = np.zeros((height, width), dtype=np.uint8)
    with rasterio.open(reference_path) as src:
        source_crs = src.crs or POTSDAM_CRS
        if source_crs != POTSDAM_CRS:
            raise ValueError("Potsdam reference must resolve to EPSG:32633")
        reference_bounds = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=source_crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=POTSDAM_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        source_mask = src.read_masks(1)
        reproject(
            source=source_mask,
            destination=destination_mask,
            src_transform=src.transform,
            src_crs=source_crs,
            dst_transform=transform,
            dst_crs=POTSDAM_CRS,
            resampling=Resampling.nearest,
        )
    full_coverage = benchmark_full_coverage_mask(
        target_height=height,
        target_width=width,
        target_transform=(
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ),
        reference_bounds=reference_bounds,
    )
    valid = (destination_mask > 0) & np.isfinite(destination) & full_coverage
    return destination, valid


def _load_potsdam_scene(
    tile_id: str,
    role: str,
    prior: DA3MonocularPrior,
) -> SceneData:
    if tile_id in BLIND_TILE_IDS:
        raise RuntimeError(f"refusing to open reserved blind Potsdam tile {tile_id}")
    tile = resolve_potsdam_tile_paths(DATASET_ROOT, tile_id)
    geometry_path = _ensure_native_v2_geometry(tile_id, tile.rgb, prior)
    height, width, transform = _target_grid(tile.rgb)
    rgb_raw, rgb_valid = _read_rgb_025m(tile.rgb, height, width)
    geometry, geometry_valid = _read_float_025m(geometry_path, height, width)
    reference, reference_valid = _read_potsdam_reference_025m(
        tile.rgb,
        tile.reference_dsm,
        height,
        width,
        transform,
    )
    input_valid = rgb_valid & geometry_valid & np.isfinite(geometry)
    supervision_valid = input_valid & reference_valid & np.isfinite(reference)
    if int(supervision_valid.sum()) < 500_000:
        raise RuntimeError(f"Potsdam {tile_id} has insufficient paired valid pixels")

    rgb_ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, rgb_ranges)
    fit = fit_reference_to_prior(geometry, reference, supervision_valid)
    target = canonicalize_reference_to_prior(reference, fit)
    windows = patch_windows(
        supervision_valid,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        min_valid_fraction=0.98,
    )
    if not windows:
        raise RuntimeError(f"Potsdam {tile_id} produced no training/evaluation windows")
    return SceneData(
        scene_id=f"POTSDAM_{tile_id}",
        location_id=tile_id,
        domain="potsdam",
        role=role,
        rgb=rgb,
        geometry=geometry,
        reference_m=reference,
        input_valid=input_valid,
        supervision_valid=supervision_valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        rgb_ranges=rgb_ranges,
        target_prior=target,
        prior_reference_fit=fit,
        windows=windows,
    )


def _build_patch_index(scenes: list[SceneData]) -> list[IndexedPatch]:
    patches: list[IndexedPatch] = []
    for scene_index, scene in enumerate(scenes):
        for window in scene.windows:
            patches.append(IndexedPatch(scene_index=scene_index, window=window))
    if not patches:
        raise RuntimeError("V6 patch index is empty")
    return patches


def _make_batch(
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb_items: list[np.ndarray] = []
    geometry_items: list[np.ndarray] = []
    target_items: list[np.ndarray] = []
    valid_items: list[np.ndarray] = []
    gsd_items: list[float] = []
    for patch in patches:
        scene = scenes[patch.scene_index]
        if scene.target_prior is None:
            raise ValueError(f"scene {scene.scene_id} has no canonical training target")
        window = patch.window
        input_mask = scene.input_valid[window.row_slice, window.col_slice]
        loss_mask = scene.supervision_valid[window.row_slice, window.col_slice]
        rgb_patch = scene.rgb[window.row_slice, window.col_slice].transpose(2, 0, 1)
        geometry_patch = scene.geometry[window.row_slice, window.col_slice][None]
        target_patch = scene.target_prior[window.row_slice, window.col_slice][None]
        rgb_items.append(np.where(input_mask[None], rgb_patch, 0.0).astype(np.float32))
        geometry_items.append(
            np.where(input_mask[None], geometry_patch, 0.0).astype(np.float32)
        )
        target_items.append(np.where(loss_mask[None], target_patch, 0.0).astype(np.float32))
        valid_items.append(loss_mask[None].astype(bool))
        gsd_items.append(scene.gsd_m)
    return (
        torch.from_numpy(np.stack(rgb_items)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(geometry_items)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(target_items)).to(device=device, dtype=torch.float32),
        torch.from_numpy(np.stack(valid_items)).to(device=device, dtype=torch.bool),
        torch.tensor(gsd_items, device=device, dtype=torch.float32),
    )


def _augment_batch(
    rgb: torch.Tensor,
    geometry: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return legacy.augment_batch(rgb, geometry, target, valid, rng)


def _metadata_for_patches(
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales: list[float] = []
    baseline_rmse: list[float] = []
    for patch in patches:
        fit = scenes[patch.scene_index].prior_reference_fit
        if fit is None:
            raise ValueError(f"scene {scenes[patch.scene_index].scene_id} has no fit")
        scales.append(fit.scale_m_per_prior_unit)
        baseline_rmse.append(max(fit.rmse_m, 0.5))
    return (
        torch.tensor(scales, device=device, dtype=torch.float32),
        torch.tensor(baseline_rmse, device=device, dtype=torch.float32),
    )


def _training_objective(
    model: DepthWizardHeightModel,
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    device: torch.device,
    rng: np.random.Generator,
    *,
    augment: bool,
) -> torch.Tensor:
    rgb, geometry, target, valid, gsd = _make_batch(scenes, patches, device)
    scales, baseline_rmse = _metadata_for_patches(scenes, patches, device)
    if augment:
        rgb, geometry, target, valid = _augment_batch(
            rgb,
            geometry,
            target,
            valid,
            rng,
        )
    output = model(rgb, geometry, gsd_m=gsd)
    result = compute_urban_structure_loss(
        output,
        geometry,
        target,
        valid,
        gsd,
        scales,
        baseline_rmse,
        config=URBAN_STRUCTURE,
    )
    return result.total


def _patch_structure_scores(
    scenes: list[SceneData],
    patches: list[IndexedPatch],
) -> np.ndarray:
    scene_maps: list[np.ndarray] = []
    for scene in scenes:
        fit = scene.prior_reference_fit
        if fit is None or scene.target_prior is None:
            raise ValueError(f"scene {scene.scene_id} has no structure target")
        correction_m = (
            scene.target_prior - scene.geometry
        ) * fit.scale_m_per_prior_unit
        scene_maps.append(
            physical_highpass_numpy(
                correction_m,
                scene.supervision_valid,
                gsd_m=scene.gsd_m,
                characteristic_scale_m=URBAN_STRUCTURE.coarse_scale_m,
            )
        )
    scores = np.zeros(len(patches), dtype=np.float64)
    for index, patch in enumerate(patches):
        values = scene_maps[patch.scene_index][patch.window.row_slice, patch.window.col_slice]
        selected = np.abs(values[np.isfinite(values)])
        scores[index] = float(np.percentile(selected, 90.0)) if selected.size else 0.0
    return scores


def _domain_balanced_epoch_indices(
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if scores.shape != (len(patches),):
        raise ValueError("structure scores must match patch index")
    sampled: list[np.ndarray] = []
    for domain in ("ortholoc", "potsdam"):
        domain_indices = np.asarray(
            [
                index
                for index, patch in enumerate(patches)
                if scenes[patch.scene_index].domain == domain
            ],
            dtype=np.int64,
        )
        if domain_indices.size == 0:
            raise RuntimeError(f"V6 has no {domain} training patches")
        scene_counts: dict[int, int] = {}
        for index in domain_indices:
            scene_index = patches[int(index)].scene_index
            scene_counts[scene_index] = scene_counts.get(scene_index, 0) + 1
        weights = np.empty(domain_indices.size, dtype=np.float64)
        for local_index, patch_index in enumerate(domain_indices):
            patch = patches[int(patch_index)]
            scene_count = scene_counts[patch.scene_index]
            structure_boost = np.clip(
                scores[int(patch_index)] / URBAN_STRUCTURE.active_threshold_m,
                0.0,
                URBAN_STRUCTURE.max_structure_weight - 1.0,
            )
            weights[local_index] = (1.0 + structure_boost) / scene_count
        weights /= weights.sum()
        sampled.append(
            rng.choice(
                domain_indices,
                size=SAMPLES_PER_DOMAIN,
                replace=True,
                p=weights,
            )
        )
    order = np.concatenate(sampled)
    rng.shuffle(order)
    return order


def _predict_scene(
    model: DepthWizardHeightModel,
    scene: SceneData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    windows = patch_windows(
        scene.input_valid,
        patch_size=PATCH_SIZE,
        stride=PATCH_STRIDE,
        min_valid_fraction=0.90,
    )
    if not windows:
        raise RuntimeError(f"scene {scene.scene_id} has no inference windows")
    height, width = scene.geometry.shape
    prediction_sum = np.zeros((height, width), dtype=np.float64)
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
                mask = scene.input_valid[window.row_slice, window.col_slice]
                rgb_patch = scene.rgb[window.row_slice, window.col_slice].transpose(2, 0, 1)
                geometry_patch = scene.geometry[window.row_slice, window.col_slice][None]
                rgb_items.append(
                    np.where(mask[None], rgb_patch, 0.0).astype(np.float32)
                )
                geometry_items.append(
                    np.where(mask[None], geometry_patch, 0.0).astype(np.float32)
                )
            rgb = torch.from_numpy(np.stack(rgb_items)).to(
                device=device,
                dtype=torch.float32,
            )
            geometry = torch.from_numpy(np.stack(geometry_items)).to(
                device=device,
                dtype=torch.float32,
            )
            gsd = torch.full(
                (len(batch_windows),),
                scene.gsd_m,
                device=device,
                dtype=torch.float32,
            )
            output = model(rgb, geometry, gsd_m=gsd)
            predictions = output.relative_height[:, 0].cpu().numpy()
            for index, window in enumerate(batch_windows):
                prediction_sum[window.row_slice, window.col_slice] += (
                    predictions[index] * blend
                )
                weights[window.row_slice, window.col_slice] += blend
    covered = weights > 0
    raw = np.full((height, width), np.nan, dtype=np.float32)
    raw[covered] = (prediction_sum[covered] / weights[covered]).astype(np.float32)
    correction_valid = scene.input_valid & covered & np.isfinite(raw)
    correction = raw - scene.geometry
    safe = project_structure_correction_numpy(
        correction,
        correction_valid,
        gsd_m=scene.gsd_m,
        config=SAFETY_STRUCTURE,
    )
    refined = np.full_like(scene.geometry, np.nan, dtype=np.float32)
    valid = correction_valid & np.isfinite(safe)
    refined[valid] = scene.geometry[valid] + safe[valid]
    return refined, valid


def _rmse(values: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[valid & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("no valid values for RMSE")
    return float(np.sqrt(np.mean(selected**2)))


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid & np.isfinite(a) & np.isfinite(b)
    av = np.asarray(a, dtype=np.float64)[mask]
    bv = np.asarray(b, dtype=np.float64)[mask]
    if av.size < 2 or np.std(av) <= 1e-12 or np.std(bv) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def _structure_report(
    base: np.ndarray,
    refined: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    gsd_m: float,
    scale_m: float,
) -> dict[str, float | int | None]:
    base_error_hp = physical_highpass_numpy(
        base - reference,
        valid,
        gsd_m=gsd_m,
        characteristic_scale_m=scale_m,
    )
    refined_error_hp = physical_highpass_numpy(
        refined - reference,
        valid,
        gsd_m=gsd_m,
        characteristic_scale_m=scale_m,
    )
    base_surface_hp = physical_highpass_numpy(
        base,
        valid,
        gsd_m=gsd_m,
        characteristic_scale_m=scale_m,
    )
    refined_surface_hp = physical_highpass_numpy(
        refined,
        valid,
        gsd_m=gsd_m,
        characteristic_scale_m=scale_m,
    )
    reference_hp = physical_highpass_numpy(
        reference,
        valid,
        gsd_m=gsd_m,
        characteristic_scale_m=scale_m,
    )
    hp_valid = (
        valid
        & np.isfinite(base_error_hp)
        & np.isfinite(refined_error_hp)
        & np.isfinite(base_surface_hp)
        & np.isfinite(refined_surface_hp)
        & np.isfinite(reference_hp)
    )
    base_rmse = _rmse(base_error_hp, hp_valid)
    refined_rmse = _rmse(refined_error_hp, hp_valid)
    return {
        "scale_m": scale_m,
        "valid_pixels": int(hp_valid.sum()),
        "base_rmse_m": base_rmse,
        "refined_rmse_m": refined_rmse,
        "improvement_fraction": (base_rmse - refined_rmse) / base_rmse,
        "base_pearson_r": _corr(base_surface_hp, reference_hp, hp_valid),
        "refined_pearson_r": _corr(refined_surface_hp, reference_hp, hp_valid),
    }


def _evaluate_sparse_scene(
    model: DepthWizardHeightModel,
    scene: SceneData,
    device: torch.device,
) -> dict[str, object]:
    refined, refined_valid = _predict_scene(model, scene, device)
    evaluation_valid = scene.supervision_valid & refined_valid & np.isfinite(refined)
    baseline = sparse_anchor_holdout_benchmark(
        scene.geometry,
        scene.reference_m,
        valid_mask=evaluation_valid,
        anchor_count=ANCHOR_COUNT,
        seed=SEED,
        exclusion_radius_px=4,
    )
    try:
        candidate = sparse_anchor_holdout_benchmark(
            refined,
            scene.reference_m,
            valid_mask=evaluation_valid,
            anchor_count=ANCHOR_COUNT,
            seed=SEED,
            exclusion_radius_px=4,
        )
    except ValueError as exc:
        common = baseline.evaluation_mask
        base_metrics = compute_elevation_metrics(
            baseline.prediction,
            scene.reference_m,
            valid_mask=common,
        )
        return {
            "scene_id": scene.scene_id,
            "domain": scene.domain,
            "base": base_metrics.model_dump(),
            "refined": None,
            "rejection": str(exc),
        }

    common = baseline.evaluation_mask & candidate.evaluation_mask
    base_metrics = compute_elevation_metrics(
        baseline.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    refined_metrics = compute_elevation_metrics(
        candidate.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "domain": scene.domain,
        "base": base_metrics.model_dump(),
        "refined": refined_metrics.model_dump(),
        "rejection": None,
    }
    if scene.domain == "potsdam":
        report["structure_4m"] = _structure_report(
            baseline.prediction,
            candidate.prediction,
            scene.reference_m,
            common,
            gsd_m=scene.gsd_m,
            scale_m=4.0,
        )
        report["structure_8m"] = _structure_report(
            baseline.prediction,
            candidate.prediction,
            scene.reference_m,
            common,
            gsd_m=scene.gsd_m,
            scale_m=8.0,
        )
        report["base_slope"] = compute_slope_metrics(
            baseline.prediction,
            scene.reference_m,
            gsd_x=scene.gsd_m,
            gsd_y=scene.gsd_m,
            valid_mask=common,
        ).model_dump()
        report["refined_slope"] = compute_slope_metrics(
            candidate.prediction,
            scene.reference_m,
            gsd_x=scene.gsd_m,
            gsd_y=scene.gsd_m,
            valid_mask=common,
        ).model_dump()
    return report


def _metric(report: dict[str, object], section: str, field: str) -> float:
    mapping = report.get(section)
    if not isinstance(mapping, dict):
        raise TypeError(f"missing metric section {section}")
    value = mapping.get(field)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric {section}.{field} is not numeric")
    return float(value)



def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric {key} is not numeric")
    return float(value)


def _required_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"report section {key} is missing or invalid")
    return value


def _natural_validation_summary(reports: list[dict[str, object]]) -> dict[str, object]:
    base_sq = 0.0
    refined_sq = 0.0
    pixels = 0
    rejections: list[str] = []
    for report in reports:
        base = report.get("base")
        refined = report.get("refined")
        if not isinstance(base, dict):
            raise TypeError("natural validation base metrics missing")
        valid_pixels = int(base["valid_pixels"])
        base_sq += float(base["rmse_m"]) ** 2 * valid_pixels
        pixels += valid_pixels
        if not isinstance(refined, dict):
            rejections.append(
                f"{report['scene_id']}: {report.get('rejection', 'unknown rejection')}"
            )
            continue
        refined_sq += float(refined["rmse_m"]) ** 2 * valid_pixels
    base_rmse = float(np.sqrt(base_sq / pixels))
    refined_rmse = float("inf") if rejections else float(np.sqrt(refined_sq / pixels))
    return {
        "base_rmse_m": base_rmse,
        "refined_rmse_m": None if not np.isfinite(refined_rmse) else refined_rmse,
        "rejections": rejections,
        "non_degrading": not rejections and refined_rmse <= base_rmse + 1e-9,
    }


def _urban_epoch_eligible(
    urban: dict[str, object],
    natural: dict[str, object],
) -> tuple[bool, float]:
    refined = urban.get("refined")
    if not isinstance(refined, dict):
        return False, float("-inf")
    base_rmse = _metric(urban, "base", "rmse_m")
    refined_rmse = _metric(urban, "refined", "rmse_m")
    structure4 = _required_mapping(urban, "structure_4m")
    structure8 = _required_mapping(urban, "structure_8m")
    base_slope = _required_mapping(urban, "base_slope")
    refined_slope = _required_mapping(urban, "refined_slope")

    s4_base = _required_number(structure4, "base_rmse_m")
    s4_refined = _required_number(structure4, "refined_rmse_m")
    s8_base = _required_number(structure8, "base_rmse_m")
    s8_refined = _required_number(structure8, "refined_rmse_m")

    s4_base_r = structure4.get("base_pearson_r")
    s4_refined_r = structure4.get("refined_pearson_r")
    s8_base_r = structure8.get("base_pearson_r")
    s8_refined_r = structure8.get("refined_pearson_r")
    if not isinstance(s4_base_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s4_refined_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s8_base_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s8_refined_r, (int, float)):
        return False, float("-inf")

    natural_non_degrading = natural.get("non_degrading")
    if not isinstance(natural_non_degrading, bool):
        raise TypeError("natural validation non_degrading flag is not boolean")

    slope_ok = _required_number(refined_slope, "rmse_degrees") <= (
        _required_number(base_slope, "rmse_degrees") + 1e-9
    )
    eligible = bool(
        natural_non_degrading
        and refined_rmse < base_rmse - 1e-3
        and s4_refined < s4_base - 1e-3
        and s8_refined < s8_base - 1e-3
        and float(s4_refined_r) >= float(s4_base_r)
        and float(s8_refined_r) >= float(s8_base_r)
        and slope_ok
    )
    if not eligible:
        return False, float("-inf")
    score = (
        (base_rmse - refined_rmse) / base_rmse
        + (s4_base - s4_refined) / s4_base
        + 1.5 * (s8_base - s8_refined) / s8_base
    )
    return True, float(score)


def _warm_start_v5(model: DepthWizardHeightModel) -> None:
    if not V5_CHECKPOINT.is_file():
        raise FileNotFoundError(f"missing frozen V5 checkpoint: {V5_CHECKPOINT}")
    actual_sha = sha256_file(V5_CHECKPOINT)
    if actual_sha != EXPECTED_V5_SHA256:
        raise RuntimeError(
            f"V5 checkpoint SHA mismatch: expected {EXPECTED_V5_SHA256}, got {actual_sha}"
        )
    try:
        checkpoint = torch.load(V5_CHECKPOINT, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(V5_CHECKPOINT, map_location="cpu")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state_dict"), dict):
        raise TypeError("V5 checkpoint is missing state_dict")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    torch.nn.init.zeros_(model.height_residual_head.weight)
    if model.height_residual_head.bias is not None:
        torch.nn.init.zeros_(model.height_residual_head.bias)


def _find_tif_by_sha(root: Path, expected_sha: str, label: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"missing {label} root: {root}")
    for path in sorted(root.rglob("*.tif")):
        if sha256_file(path) == expected_sha:
            return path
    raise FileNotFoundError(f"could not locate frozen {label} with SHA-256 {expected_sha}")


def _sample_scale_fit(
    rdsm: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float]:
    indices = np.flatnonzero(valid & np.isfinite(rdsm) & np.isfinite(dsm))
    if indices.size < 1000:
        raise RuntimeError("insufficient pixels to recover frozen V2 affine scale")
    rng = np.random.default_rng(SEED)
    if indices.size > 200_000:
        indices = rng.choice(indices, size=200_000, replace=False)
    from depthwizard.calibration.robust import robust_affine_calibration

    fit = robust_affine_calibration(
        rdsm.ravel()[indices],
        dsm.ravel()[indices],
        require_positive_scale=True,
    )
    return float(fit.scale), float(fit.offset), float(fit.rmse_anchor)


def _evaluate_exposed_2_14(
    model: DepthWizardHeightModel,
    device: torch.device,
) -> dict[str, object]:
    tile = resolve_potsdam_tile_paths(DATASET_ROOT, URBAN_DEVELOPMENT_TILE_ID)
    if sha256_file(tile.reference_dsm) != EXPECTED_2_14_REFERENCE_SHA256:
        raise RuntimeError("Potsdam 2_14 reference SHA mismatch")
    rdsm_path = _find_tif_by_sha(EXPOSED_V2_ROOT, EXPECTED_V2_RDSM_SHA256, "V2 rDSM")
    dsm_path = _find_tif_by_sha(EXPOSED_V2_ROOT, EXPECTED_V2_DSM_SHA256, "V2 metric DSM")
    height, width, transform = _target_grid(tile.rgb)
    rgb_raw, rgb_valid = _read_rgb_025m(tile.rgb, height, width)
    rdsm, rdsm_valid = _read_float_025m(rdsm_path, height, width)
    base_dsm, dsm_valid = _read_float_025m(dsm_path, height, width)
    reference, ref_valid = _read_potsdam_reference_025m(
        tile.rgb,
        tile.reference_dsm,
        height,
        width,
        transform,
    )
    input_valid = rgb_valid & rdsm_valid & np.isfinite(rdsm)
    ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, ranges)
    scene = SceneData(
        scene_id="POTSDAM_2_14_PRODUCTION_TRANSFER",
        location_id="2_14",
        domain="potsdam",
        role="development",
        rgb=rgb,
        geometry=rdsm,
        reference_m=reference,
        input_valid=input_valid,
        supervision_valid=input_valid & ref_valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        rgb_ranges=ranges,
        target_prior=None,
        prior_reference_fit=None,
        windows=patch_windows(
            input_valid,
            patch_size=PATCH_SIZE,
            stride=PATCH_STRIDE,
            min_valid_fraction=0.90,
        ),
    )
    refined_relative, refined_valid = _predict_scene(model, scene, device)
    metric_scale, metric_offset, affine_rmse = _sample_scale_fit(
        rdsm,
        base_dsm,
        input_valid & dsm_valid,
    )
    del metric_offset
    correction = refined_relative - rdsm
    refined_dsm = np.full_like(base_dsm, np.nan, dtype=np.float32)
    common_refined = refined_valid & dsm_valid & np.isfinite(correction)
    refined_dsm[common_refined] = (
        base_dsm[common_refined]
        + correction[common_refined] * np.float32(metric_scale)
    )
    valid = (
        input_valid
        & dsm_valid
        & ref_valid
        & common_refined
        & np.isfinite(base_dsm)
        & np.isfinite(refined_dsm)
        & np.isfinite(reference)
    )
    base_metrics = compute_elevation_metrics(base_dsm, reference, valid_mask=valid)
    refined_metrics = compute_elevation_metrics(refined_dsm, reference, valid_mask=valid)
    base_slope = compute_slope_metrics(
        base_dsm,
        reference,
        gsd_x=POTSDAM_BENCHMARK_GSD_M,
        gsd_y=POTSDAM_BENCHMARK_GSD_M,
        valid_mask=valid,
    )
    refined_slope = compute_slope_metrics(
        refined_dsm,
        reference,
        gsd_x=POTSDAM_BENCHMARK_GSD_M,
        gsd_y=POTSDAM_BENCHMARK_GSD_M,
        valid_mask=valid,
    )
    structure4 = _structure_report(
        base_dsm,
        refined_dsm,
        reference,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        scale_m=4.0,
    )
    structure8 = _structure_report(
        base_dsm,
        refined_dsm,
        reference,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        scale_m=8.0,
    )
    s4_refined_r = structure4.get("refined_pearson_r")
    s4_base_r = structure4.get("base_pearson_r")
    s8_refined_r = structure8.get("refined_pearson_r")
    s8_base_r = structure8.get("base_pearson_r")
    structure_correlation_ok = False
    if (
        isinstance(s4_refined_r, (int, float))
        and isinstance(s4_base_r, (int, float))
        and isinstance(s8_refined_r, (int, float))
        and isinstance(s8_base_r, (int, float))
    ):
        structure_correlation_ok = (
            float(s4_refined_r) >= float(s4_base_r)
            and float(s8_refined_r) >= float(s8_base_r)
        )
    pass_transfer = bool(
        refined_metrics.rmse_m < base_metrics.rmse_m
        and refined_metrics.mae_m <= base_metrics.mae_m
        and refined_metrics.pearson_r is not None
        and base_metrics.pearson_r is not None
        and refined_metrics.pearson_r >= base_metrics.pearson_r
        and refined_slope.rmse_degrees <= base_slope.rmse_degrees
        and _required_number(structure4, "refined_rmse_m")
        < _required_number(structure4, "base_rmse_m")
        and _required_number(structure8, "refined_rmse_m")
        < _required_number(structure8, "base_rmse_m")
        and structure_correlation_ok
    )
    return {
        "base": base_metrics.model_dump(),
        "refined": refined_metrics.model_dump(),
        "base_slope": base_slope.model_dump(),
        "refined_slope": refined_slope.model_dump(),
        "structure_4m": structure4,
        "structure_8m": structure8,
        "recovered_v2_metric_scale_m_per_relative_unit": metric_scale,
        "v2_affine_reconstruction_rmse_m": affine_rmse,
        "pass": pass_transfer,
        "blind_tiles_touched": False,
    }


def main() -> None:
    _assert_protocol_partition()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = _resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("DepthWizard V6 mixed-domain urban structure training")
    print(f"Urban train: {list(URBAN_TRAIN_TILE_IDS)}")
    print(f"Urban validation: {URBAN_VALIDATION_TILE_ID}")
    print(f"Urban exposed development: {URBAN_DEVELOPMENT_TILE_ID}")
    print(f"Reserved blind and never opened by this script: {list(BLIND_TILE_IDS)}")

    train_discovered = legacy.discover_remote_scenes("train")
    outplace_discovered = legacy.discover_remote_scenes("test_outPlace")
    train_remote, validation_remote, _development_remote = v3.select_diversity_split(
        train_discovered,
        outplace_discovered,
    )
    prior = DA3MonocularPrior(device="auto")
    ortho_train_raw, training_rejections = v3.load_training_scenes_with_replacement(
        train_remote,
        train_discovered,
        prior,
    )
    ortho_validation_raw = [
        legacy.load_scene(scene, "validation", prior, include_target=True)
        for scene in validation_remote
    ]
    ortho_train = [_from_ortholoc(scene, "train") for scene in ortho_train_raw]
    ortho_validation = [
        _from_ortholoc(scene, "validation") for scene in ortho_validation_raw
    ]
    urban_train = [
        _load_potsdam_scene(tile_id, "train", prior)
        for tile_id in URBAN_TRAIN_TILE_IDS
    ]
    urban_validation = _load_potsdam_scene(
        URBAN_VALIDATION_TILE_ID,
        "validation",
        prior,
    )
    del prior

    train_scenes = ortho_train + urban_train
    train_patches = _build_patch_index(train_scenes)
    scores = _patch_structure_scores(train_scenes, train_patches)
    print(
        "Training domains: "
        f"OrthoLoC scenes={len(ortho_train)}, Potsdam scenes={len(urban_train)}, "
        f"raw patches={len(train_patches)}, samples/epoch={2 * SAMPLES_PER_DOMAIN}"
    )
    print(
        "Training-only OrthoLoC quality exclusions: "
        f"{[item['scene_id'] for item in training_rejections] or 'none'}"
    )
    print(
        "Structure-score quantiles m: "
        f"p50={np.percentile(scores, 50):.3f}, "
        f"p75={np.percentile(scores, 75):.3f}, "
        f"p90={np.percentile(scores, 90):.3f}, "
        f"p95={np.percentile(scores, 95):.3f}"
    )

    config = HeightModelConfig(
        architecture_version="bidirectional-cross-scale-v1",
        max_relative_correction=0.60,
    )
    model = DepthWizardHeightModel(config)
    _warm_start_v5(model)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    initial_urban = _evaluate_sparse_scene(model, urban_validation, device)
    initial_natural_reports = [
        _evaluate_sparse_scene(model, scene, device)
        for scene in ortho_validation
    ]
    initial_natural = _natural_validation_summary(initial_natural_reports)
    identity_rmse = _metric(initial_urban, "base", "rmse_m")
    if abs(_metric(initial_urban, "refined", "rmse_m") - identity_rmse) > 1e-6:
        raise RuntimeError("V6 warm-start did not preserve exact identity at epoch 0")

    best_epoch = 0
    best_score = float("-inf")
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = _domain_balanced_epoch_indices(
            train_scenes,
            train_patches,
            scores,
            rng,
        )
        weighted_loss = 0.0
        samples = 0
        for start in range(0, len(order), BATCH_SIZE):
            selected = [
                train_patches[int(index)]
                for index in order[start : start + BATCH_SIZE]
            ]
            optimizer.zero_grad(set_to_none=True)
            loss = _training_objective(
                model,
                train_scenes,
                selected,
                device,
                rng,
                augment=True,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite V6 training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = len(selected)
            weighted_loss += float(loss.detach().cpu()) * count
            samples += count
        train_loss = weighted_loss / samples

        urban_report = _evaluate_sparse_scene(model, urban_validation, device)
        natural_reports = [
            _evaluate_sparse_scene(model, scene, device)
            for scene in ortho_validation
        ]
        natural_summary = _natural_validation_summary(natural_reports)
        eligible, score = _urban_epoch_eligible(urban_report, natural_summary)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "urban_validation": urban_report,
                "natural_validation": natural_summary,
                "eligible": eligible,
                "selection_score": score if np.isfinite(score) else None,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        urban_base = _metric(urban_report, "base", "rmse_m")
        urban_refined = (
            _metric(urban_report, "refined", "rmse_m")
            if isinstance(urban_report.get("refined"), dict)
            else float("inf")
        )
        structure8 = urban_report.get("structure_8m")
        s8_text = "REJECTED"
        if isinstance(structure8, dict):
            s8_text = (
                f"{float(structure8['base_rmse_m']):.3f}->"
                f"{float(structure8['refined_rmse_m']):.3f}m"
            )
        natural_refined = natural_summary["refined_rmse_m"]
        print(
            f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | "
            f"urban RMSE {urban_base:.3f}->{urban_refined:.3f}m | "
            f"urban 8m {s8_text} | "
            f"natural {natural_summary['base_rmse_m']:.3f}->"
            f"{natural_refined if natural_refined is not None else 'REJECTED'} | "
            f"eligible={'YES' if eligible else 'NO'}"
        )
        if eligible and score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    final_urban_validation = _evaluate_sparse_scene(model, urban_validation, device)
    final_natural_validation_reports = [
        _evaluate_sparse_scene(model, scene, device)
        for scene in ortho_validation
    ]
    final_natural_validation = _natural_validation_summary(
        final_natural_validation_reports
    )

    exposed_2_14 = None
    if best_epoch > 0:
        exposed_2_14 = _evaluate_exposed_2_14(model, device)

    operational_candidate = bool(
        best_epoch > 0
        and exposed_2_14 is not None
        and exposed_2_14["pass"]
        and final_natural_validation["non_degrading"]
    )

    checkpoint_payload = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "config": asdict(config),
        "best_epoch": best_epoch,
        "selection_score": best_score if np.isfinite(best_score) else None,
        "v5_warm_start_sha256": EXPECTED_V5_SHA256,
        "urban_train_tiles": list(URBAN_TRAIN_TILE_IDS),
        "urban_validation_tile": URBAN_VALIDATION_TILE_ID,
        "urban_development_tile": URBAN_DEVELOPMENT_TILE_ID,
        "blind_tiles": list(BLIND_TILE_IDS),
        "structure_config": asdict(URBAN_STRUCTURE),
    }
    torch.save(checkpoint_payload, CHECKPOINT_PATH)
    checkpoint_sha = sha256_file(CHECKPOINT_PATH)

    report = {
        "status": (
            "PASS_V6_URBAN_STRUCTURE_CANDIDATE"
            if operational_candidate
            else "REJECT_V6_URBAN_STRUCTURE_CANDIDATE"
        ),
        "research_only": True,
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "selection_score": best_score if np.isfinite(best_score) else None,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "v5_warm_start_sha256": EXPECTED_V5_SHA256,
        "protocol": {
            "urban_train_tiles": list(URBAN_TRAIN_TILE_IDS),
            "urban_validation_tile": URBAN_VALIDATION_TILE_ID,
            "urban_development_tile": URBAN_DEVELOPMENT_TILE_ID,
            "blind_tiles": list(BLIND_TILE_IDS),
            "blind_tiles_touched": False,
            "checkpoint_selection_uses_2_14": False,
            "domain_balanced_samples_per_epoch": 2 * SAMPLES_PER_DOMAIN,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
        },
        "structure_config": asdict(URBAN_STRUCTURE),
        "initial_urban_validation": initial_urban,
        "initial_natural_validation": initial_natural,
        "history": history,
        "final_urban_validation": final_urban_validation,
        "final_natural_validation": final_natural_validation,
        "exposed_2_14_production_transfer": exposed_2_14,
        "operational_candidate": operational_candidate,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== DEPTHWIZARD V6 RESULT ===")
    print(f"Best epoch: {best_epoch}")
    print(
        "Natural validation non-degrading: "
        f"{'YES' if final_natural_validation['non_degrading'] else 'NO'}"
    )
    if exposed_2_14 is not None:
        base = _required_mapping(exposed_2_14, "base")
        refined = _required_mapping(exposed_2_14, "refined")
        s4 = _required_mapping(exposed_2_14, "structure_4m")
        s8 = _required_mapping(exposed_2_14, "structure_8m")
        transfer_pass = exposed_2_14.get("pass")
        if not isinstance(transfer_pass, bool):
            raise TypeError("Potsdam 2_14 transfer gate is not boolean")
        print(
            f"Potsdam 2_14 RMSE: {_required_number(base, 'rmse_m'):.4f} -> "
            f"{_required_number(refined, 'rmse_m'):.4f} m"
        )
        print(
            f"Potsdam 2_14 structure 4m: {_required_number(s4, 'base_rmse_m'):.4f} -> "
            f"{_required_number(s4, 'refined_rmse_m'):.4f} m"
        )
        print(
            f"Potsdam 2_14 structure 8m: {_required_number(s8, 'base_rmse_m'):.4f} -> "
            f"{_required_number(s8, 'refined_rmse_m'):.4f} m"
        )
        print(
            "Potsdam 2_14 production-transfer gate: "
            f"{'PASS' if transfer_pass else 'REJECT'}"
        )
    else:
        print("Potsdam 2_14 was not opened because no validation-eligible learned epoch existed.")
    print(
        "V6 candidate: "
        f"{'PASS_V6_URBAN_STRUCTURE_CANDIDATE' if operational_candidate else 'REJECT_V6_URBAN_STRUCTURE_CANDIDATE'}"
    )
    print(f"Checkpoint SHA-256: {checkpoint_sha}")
    print(f"Report: {REPORT_PATH}")
    print(f"Reserved blind tiles untouched: {list(BLIND_TILE_IDS)}")


if __name__ == "__main__":
    main()
