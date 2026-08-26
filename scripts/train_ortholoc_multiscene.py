from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch

from depthwizard.data.ortholoc import (
    OrthoLoCRemoteScene,
    discover_remote_scenes,
    download_file,
    select_geographic_scenes,
)
from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.height_model.losses import compute_height_losses
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
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
from depthwizard.io.raster import (
    ground_sample_distance_m,
    read_rgb,
    reproject_to_match,
    write_float_geotiff,
)
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.provenance.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "training" / "ortholoc-multiscene"
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-multiscene"
GEOMETRY_DIR = OUT_DIR / "geometry"
SEED = 26175
TRAIN_LOCATIONS = 4
VALIDATION_LOCATIONS = 1
TEST_LOCATIONS = 2
SAMPLES_PER_LOCATION = 1
PATCH_SIZE = 192
STRIDE = 96
EPOCHS = 14
BATCH_SIZE = 2
LEARNING_RATE = 1.5e-4
ANCHOR_COUNT = 64


@dataclass
class SceneData:
    scene_id: str
    location_id: str
    role: str
    source: OrthoLoCRemoteScene
    dop_path: Path
    dsm_path: Path
    geometry_path: Path
    rgb: np.ndarray
    geometry: np.ndarray
    reference_m: np.ndarray
    input_valid: np.ndarray
    supervision_valid: np.ndarray
    gsd_m: float
    rgb_ranges: tuple[RobustRange, RobustRange, RobustRange]
    target_prior: np.ndarray | None = None
    prior_reference_fit: PriorReferenceFit | None = None
    windows: list[PatchWindow] | None = None


@dataclass(frozen=True)
class IndexedPatch:
    scene_index: int
    window: PatchWindow


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _scene_lookup(scenes: list[OrthoLoCRemoteScene]) -> dict[str, OrthoLoCRemoteScene]:
    lookup = {scene.filename: scene for scene in scenes}
    if len(lookup) != len(scenes):
        raise RuntimeError("duplicate OrthoLoC filenames discovered within one split")
    return lookup


def select_experiment_scenes() -> tuple[
    list[OrthoLoCRemoteScene],
    list[OrthoLoCRemoteScene],
    list[OrthoLoCRemoteScene],
]:
    train_discovered = [scene for scene in discover_remote_scenes("train") if scene.same_domain]
    if not train_discovered:
        raise RuntimeError("official OrthoLoC train listing contains no same-domain scenes")
    train_lookup = _scene_lookup(train_discovered)
    train_names = [scene.filename for scene in train_discovered]

    selected_train_names = select_geographic_scenes(
        train_names,
        max_locations=TRAIN_LOCATIONS,
        samples_per_location=SAMPLES_PER_LOCATION,
    )
    train_locations = {
        train_lookup[name].location_id for name in selected_train_names
    }
    selected_validation_names = select_geographic_scenes(
        train_names,
        max_locations=VALIDATION_LOCATIONS,
        samples_per_location=SAMPLES_PER_LOCATION,
        excluded_locations=train_locations,
    )
    validation_locations = {
        train_lookup[name].location_id for name in selected_validation_names
    }
    if train_locations & validation_locations:
        raise RuntimeError("geographic leakage between multiscene train and validation sets")

    test_discovered = [
        scene for scene in discover_remote_scenes("test_outPlace") if scene.same_domain
    ]
    if not test_discovered:
        raise RuntimeError("official OrthoLoC test_outPlace listing contains no same-domain scenes")
    test_lookup = _scene_lookup(test_discovered)
    test_names = [scene.filename for scene in test_discovered]
    selected_test_names = select_geographic_scenes(
        test_names,
        max_locations=TEST_LOCATIONS,
        samples_per_location=SAMPLES_PER_LOCATION,
        excluded_locations=train_locations | validation_locations,
    )
    test_locations = {test_lookup[name].location_id for name in selected_test_names}
    if (train_locations | validation_locations) & test_locations:
        raise RuntimeError("geographic leakage into OrthoLoC out-of-place test set")

    return (
        [train_lookup[name] for name in selected_train_names],
        [train_lookup[name] for name in selected_validation_names],
        [test_lookup[name] for name in selected_test_names],
    )


def materialize_scene(remote: OrthoLoCRemoteScene, role: str) -> tuple[Path, Path]:
    scene_dir = DATA_DIR / role / remote.scene_id
    dop_path = download_file(remote.dop_url, scene_dir / "rgb.tif")
    dsm_path = download_file(remote.dsm_url, scene_dir / "reference_dsm.tif")
    return dop_path, dsm_path


def ensure_geometry(
    remote: OrthoLoCRemoteScene,
    dop_path: Path,
    prior: DA3MonocularPrior,
) -> Path:
    geometry_path = GEOMETRY_DIR / f"{remote.scene_id}_rdsm.tif"
    if geometry_path.exists() and geometry_path.stat().st_size > 0:
        with rasterio.open(geometry_path) as geometry_src, rasterio.open(dop_path) as dop_src:
            if (
                geometry_src.width == dop_src.width
                and geometry_src.height == dop_src.height
                and geometry_src.crs == dop_src.crs
                and geometry_src.transform == dop_src.transform
            ):
                print(f"Reusing DA3 geometry: {remote.scene_id}")
                return geometry_path

    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating DA3 geometry: {remote.scene_id}")
    output = infer_geometry_scene(
        dop_path,
        prior,
        tile_size=768,
        overlap=128,
        harmonize_overlaps=True,
    )
    write_float_geotiff(
        geometry_path,
        output.relative_height,
        template_path=dop_path,
        description=f"DepthWizard DA3 geometry prior for OrthoLoC {remote.scene_id}",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": output.model_id,
            "SCENE_ID": remote.scene_id,
            "PURPOSE": "multiscene_height_training_acceptance",
        },
    )
    return geometry_path


def load_scene(
    remote: OrthoLoCRemoteScene,
    role: str,
    prior: DA3MonocularPrior,
    *,
    include_target: bool,
) -> SceneData:
    dop_path, dsm_path = materialize_scene(remote, role)
    geometry_path = ensure_geometry(remote, dop_path, prior)
    rgb_raw = read_rgb(dop_path)
    with rasterio.open(dop_path) as src:
        source_valid = src.dataset_mask() > 0
    with rasterio.open(geometry_path) as src:
        geometry = src.read(1).astype(np.float32)
        geometry_valid = np.isfinite(geometry)
        if src.nodata is not None:
            geometry_valid &= geometry != src.nodata

    reference, reference_valid = reproject_to_match(dsm_path, geometry_path)
    reference = reference.astype(np.float32, copy=False)
    input_valid = source_valid & geometry_valid
    supervision_valid = input_valid & reference_valid & np.isfinite(reference)
    if int(supervision_valid.sum()) < 50_000:
        raise RuntimeError(f"{remote.scene_id} has insufficient paired valid pixels")

    metric_gsd = ground_sample_distance_m(dop_path)
    if metric_gsd is None:
        raise RuntimeError(f"{remote.scene_id} is missing metric GSD")
    gsd_m = float(np.sqrt(metric_gsd[0] * metric_gsd[1]))
    rgb_ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, rgb_ranges)

    fit: PriorReferenceFit | None = None
    target: np.ndarray | None = None
    if include_target:
        fit = fit_reference_to_prior(geometry, reference, supervision_valid)
        target = canonicalize_reference_to_prior(reference, fit)

    windows = patch_windows(
        supervision_valid,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_fraction=0.98,
    )
    if not windows:
        raise RuntimeError(f"{remote.scene_id} produced no training/evaluation windows")

    return SceneData(
        scene_id=remote.scene_id,
        location_id=remote.location_id,
        role=role,
        source=remote,
        dop_path=dop_path,
        dsm_path=dsm_path,
        geometry_path=geometry_path,
        rgb=rgb,
        geometry=geometry,
        reference_m=reference,
        input_valid=input_valid,
        supervision_valid=supervision_valid,
        gsd_m=gsd_m,
        rgb_ranges=rgb_ranges,
        target_prior=target,
        prior_reference_fit=fit,
        windows=windows,
    )


def build_patch_index(scenes: list[SceneData]) -> list[IndexedPatch]:
    patches: list[IndexedPatch] = []
    for scene_index, scene in enumerate(scenes):
        for window in scene.windows or []:
            patches.append(IndexedPatch(scene_index=scene_index, window=window))
    if not patches:
        raise RuntimeError("multiscene patch index is empty")
    return patches


def make_batch(
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not patches:
        raise ValueError("make_batch requires at least one patch")
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


def augment_batch(
    rgb: torch.Tensor,
    geometry: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if rng.random() < 0.5:
        rgb, geometry, target, valid = (
            torch.flip(value, dims=(-1,)) for value in (rgb, geometry, target, valid)
        )
    if rng.random() < 0.5:
        rgb, geometry, target, valid = (
            torch.flip(value, dims=(-2,)) for value in (rgb, geometry, target, valid)
        )
    rotations = int(rng.integers(0, 4))
    if rotations:
        rgb, geometry, target, valid = (
            torch.rot90(value, rotations, dims=(-2, -1))
            for value in (rgb, geometry, target, valid)
        )
    return rgb, geometry, target, valid


def validation_loss(
    model: DepthWizardHeightModel,
    scenes: list[SceneData],
    patches: list[IndexedPatch],
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(patches), BATCH_SIZE):
            batch = make_batch(scenes, patches[start : start + BATCH_SIZE], device)
            rgb, geometry, target, valid, gsd = batch
            output = model(rgb, geometry, gsd_m=gsd)
            losses.append(float(compute_height_losses(output, target, valid).total.cpu()))
    if not losses:
        raise RuntimeError("multiscene validation produced no batches")
    return float(np.mean(losses))


def predict_scene(
    model: DepthWizardHeightModel,
    scene: SceneData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = patch_windows(
        scene.input_valid,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_fraction=0.90,
    )
    if not windows:
        raise RuntimeError(f"{scene.scene_id} has no inference windows")
    height, width = scene.geometry.shape
    height_sum = np.zeros((height, width), dtype=np.float64)
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
                input_mask = scene.input_valid[window.row_slice, window.col_slice]
                rgb_patch = scene.rgb[window.row_slice, window.col_slice].transpose(2, 0, 1)
                geometry_patch = scene.geometry[window.row_slice, window.col_slice][None]
                rgb_items.append(np.where(input_mask[None], rgb_patch, 0.0).astype(np.float32))
                geometry_items.append(
                    np.where(input_mask[None], geometry_patch, 0.0).astype(np.float32)
                )
            rgb = torch.from_numpy(np.stack(rgb_items)).to(device=device, dtype=torch.float32)
            geometry = torch.from_numpy(np.stack(geometry_items)).to(
                device=device, dtype=torch.float32
            )
            gsd = torch.full((len(batch_windows),), scene.gsd_m, device=device)
            output = model(rgb, geometry, gsd_m=gsd)
            heights = output.relative_height[:, 0].cpu().numpy()
            uncertainties = output.uncertainty[:, 0].cpu().numpy()
            for index, window in enumerate(batch_windows):
                height_sum[window.row_slice, window.col_slice] += heights[index] * blend
                uncertainty_sum[window.row_slice, window.col_slice] += uncertainties[index] * blend
                weights[window.row_slice, window.col_slice] += blend

    covered = weights > 0
    prediction = np.full((height, width), np.nan, dtype=np.float32)
    uncertainty = np.full((height, width), np.nan, dtype=np.float32)
    prediction[covered] = (height_sum[covered] / weights[covered]).astype(np.float32)
    uncertainty[covered] = (uncertainty_sum[covered] / weights[covered]).astype(np.float32)
    return prediction, uncertainty, covered


def uncertainty_error_correlation(
    uncertainty: np.ndarray,
    absolute_error_m: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    mask = valid & np.isfinite(uncertainty) & np.isfinite(absolute_error_m)
    u = uncertainty[mask].astype(np.float64)
    e = absolute_error_m[mask].astype(np.float64)
    if u.size < 2 or float(np.std(u)) <= 1e-12 or float(np.std(e)) <= 1e-12:
        return None
    return float(np.corrcoef(u, e)[0, 1])


def evaluate_scene(
    model: DepthWizardHeightModel,
    scene: SceneData,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    refined, uncertainty, covered = predict_scene(model, scene, device)
    evaluation_valid = scene.supervision_valid & covered & np.isfinite(refined)
    baseline = sparse_anchor_holdout_benchmark(
        scene.geometry,
        scene.reference_m,
        valid_mask=evaluation_valid,
        anchor_count=ANCHOR_COUNT,
        seed=SEED,
        exclusion_radius_px=4,
    )
    refined_result = sparse_anchor_holdout_benchmark(
        refined,
        scene.reference_m,
        valid_mask=evaluation_valid,
        anchor_count=ANCHOR_COUNT,
        seed=SEED,
        exclusion_radius_px=4,
    )
    common_mask = baseline.evaluation_mask & refined_result.evaluation_mask
    baseline_metrics = compute_elevation_metrics(
        baseline.prediction, scene.reference_m, valid_mask=common_mask
    )
    refined_metrics = compute_elevation_metrics(
        refined_result.prediction, scene.reference_m, valid_mask=common_mask
    )
    abs_error = np.abs(refined_result.prediction - scene.reference_m)
    reliability = uncertainty_error_correlation(uncertainty, abs_error, common_mask)
    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "location_id": scene.location_id,
        "gsd_m": scene.gsd_m,
        "heldout_pixels": int(common_mask.sum()),
        "da3": baseline_metrics.model_dump(),
        "depthwizard": refined_metrics.model_dump(),
        "rmse_delta_m": float(refined_metrics.rmse_m - baseline_metrics.rmse_m),
        "mae_delta_m": float(refined_metrics.mae_m - baseline_metrics.mae_m),
        "uncertainty_abs_error_pearson": reliability,
    }
    return (
        report,
        baseline.prediction[common_mask],
        refined_result.prediction[common_mask],
        scene.reference_m[common_mask],
        uncertainty[common_mask],
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_remote, validation_remote, test_remote = select_experiment_scenes()
    print(
        "Geographic split: "
        f"train={[scene.location_id for scene in train_remote]} | "
        f"validation={[scene.location_id for scene in validation_remote]} | "
        f"test_outPlace={[scene.location_id for scene in test_remote]}"
    )

    prior = DA3MonocularPrior(device="auto")
    train_scenes = [
        load_scene(scene, "train", prior, include_target=True) for scene in train_remote
    ]
    validation_scenes = [
        load_scene(scene, "validation", prior, include_target=True)
        for scene in validation_remote
    ]
    test_scenes = [
        load_scene(scene, "test_outPlace", prior, include_target=False) for scene in test_remote
    ]
    del prior

    for scene in train_scenes:
        fit = scene.prior_reference_fit
        if fit is not None:
            print(
                f"train {scene.scene_id}: canonicalization "
                f"{fit.scale_m_per_prior_unit:.3f} m/prior + {fit.offset_m:.3f} m | "
                f"RMSE {fit.rmse_m:.3f} m | patches={len(scene.windows or [])}"
            )

    train_patches = build_patch_index(train_scenes)
    validation_patches = build_patch_index(validation_scenes)
    config = HeightModelConfig()
    model = DepthWizardHeightModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    initial_val = validation_loss(model, validation_scenes, validation_patches, device)
    best_val = initial_val
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    print(
        f"Training multiscene refiner: patches={len(train_patches)} | "
        f"validation={len(validation_patches)} | device={device}"
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_patches))
        epoch_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_patches[int(index)] for index in order[start : start + BATCH_SIZE]]
            rgb, geometry, target, valid, gsd = make_batch(train_scenes, selected, device)
            rgb, geometry, target, valid = augment_batch(rgb, geometry, target, valid, rng)
            optimizer.zero_grad(set_to_none=True)
            output = model(rgb, geometry, gsd_m=gsd)
            loss = compute_height_losses(output, target, valid).total
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite multiscene training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val = validation_loss(model, validation_scenes, validation_patches, device)
        train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | val {val:.5f}")
        if val < best_val:
            best_val = val
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    scene_reports: list[dict[str, object]] = []
    baseline_values: list[np.ndarray] = []
    refined_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    uncertainty_values: list[np.ndarray] = []
    for scene in test_scenes:
        scene_report, baseline, refined, reference, uncertainty = evaluate_scene(
            model, scene, device
        )
        scene_reports.append(scene_report)
        baseline_values.append(baseline)
        refined_values.append(refined)
        reference_values.append(reference)
        uncertainty_values.append(uncertainty)
        da3_rmse = float(scene_report["da3"]["rmse_m"])  # type: ignore[index]
        refined_rmse = float(scene_report["depthwizard"]["rmse_m"])  # type: ignore[index]
        print(
            f"test {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"DepthWizard {refined_rmse:.3f} m"
        )

    baseline_all = np.concatenate(baseline_values)
    refined_all = np.concatenate(refined_values)
    reference_all = np.concatenate(reference_values)
    uncertainty_all = np.concatenate(uncertainty_values)
    aggregate_da3 = compute_elevation_metrics(baseline_all, reference_all)
    aggregate_refined = compute_elevation_metrics(refined_all, reference_all)
    aggregate_reliability = uncertainty_error_correlation(
        uncertainty_all,
        np.abs(refined_all - reference_all),
        np.ones_like(reference_all, dtype=bool),
    )
    rmse_improvement_fraction = (
        aggregate_da3.rmse_m - aggregate_refined.rmse_m
    ) / aggregate_da3.rmse_m
    promoted = bool(rmse_improvement_fraction > 0.0)

    checkpoint_path = OUT_DIR / "height_model_multiscene.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val,
        "training_locations": [scene.location_id for scene in train_scenes],
        "validation_locations": [scene.location_id for scene in validation_scenes],
        "test_locations": [scene.location_id for scene in test_scenes],
        "purpose": "geographically disjoint OrthoLoC multiscene training acceptance",
    }
    torch.save(checkpoint, checkpoint_path)

    source_manifest = []
    for scene in train_scenes + validation_scenes + test_scenes:
        source_manifest.append(
            {
                "scene_id": scene.scene_id,
                "role": scene.role,
                "location_id": scene.location_id,
                "dop_url": scene.source.dop_url,
                "dsm_url": scene.source.dsm_url,
                "dop_sha256": sha256_file(scene.dop_path),
                "dsm_sha256": sha256_file(scene.dsm_path),
                "geometry_sha256": sha256_file(scene.geometry_path),
            }
        )

    report = {
        "status": "PASS_MULTISCENE_PIPELINE",
        "model_promoted_over_da3": promoted,
        "purpose": (
            "Geographically disjoint multi-scene engineering acceptance using official OrthoLoC "
            "train and test_outPlace data. Sparse anchors recover metric scale; this is not yet "
            "the final independent-source, four-terrain SIH benchmark."
        ),
        "dataset": "OrthoLoC",
        "dataset_license": "CC BY-NC-SA 4.0",
        "device": str(device),
        "model_config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "train_locations": [scene.location_id for scene in train_scenes],
            "validation_locations": [scene.location_id for scene in validation_scenes],
            "test_outPlace_locations": [scene.location_id for scene in test_scenes],
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "train_patches": len(train_patches),
            "validation_patches": len(validation_patches),
            "initial_validation_loss": initial_val,
            "best_validation_loss": best_val,
            "best_epoch": best_epoch,
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "evaluation": {
            "protocol": "test_outPlace_geographic_holdout_plus_64_sparse_metric_anchors",
            "scenes": scene_reports,
            "aggregate_da3": aggregate_da3.model_dump(),
            "aggregate_depthwizard": aggregate_refined.model_dump(),
            "rmse_delta_m": float(aggregate_refined.rmse_m - aggregate_da3.rmse_m),
            "rmse_improvement_fraction": float(rmse_improvement_fraction),
            "uncertainty_abs_error_pearson": aggregate_reliability,
        },
        "sources": source_manifest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = OUT_DIR / "multiscene_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard OrthoLoC multiscene training acceptance: PASS")
    print(f"Best epoch: {best_epoch} | validation loss: {best_val:.5f}")
    print(
        f"Aggregate outPlace DA3 RMSE: {aggregate_da3.rmse_m:.3f} m | "
        f"DepthWizard: {aggregate_refined.rmse_m:.3f} m"
    )
    print(f"RMSE improvement: {100.0 * rmse_improvement_fraction:.2f}%")
    print(f"Model promoted over DA3: {'YES' if promoted else 'NO'}")
    print(f"Checkpoint SHA-256: {report['checkpoint_sha256']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
