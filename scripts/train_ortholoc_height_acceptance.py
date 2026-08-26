from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio
import torch

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.height_model.losses import compute_height_losses
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.training import (
    PatchWindow,
    fit_rgb_ranges,
    fit_robust_range,
    normalize_relative_target,
    normalize_rgb,
    patch_windows,
    spatial_column_holdout,
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
DATA_DIR = ROOT / "data" / "benchmark" / "ortholoc-demo"
GEOMETRY_DIR = ROOT / "artifacts" / "benchmark" / "ortholoc-demo"
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-acceptance"
BASE_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/demo"
DOP_URL = f"{BASE_URL}/urban_residential_DOP.tif"
DSM_URL = f"{BASE_URL}/urban_residential_DSM.tif"
USER_AGENT = "DepthWizard-SIH26175/0.2 training-acceptance"
SEED = 26175
PATCH_SIZE = 192
STRIDE = 128
TRAIN_FRACTION = 0.68
HOLDOUT_GAP_PX = 64
EPOCHS = 12
BATCH_SIZE = 2
LEARNING_RATE = 2e-4
ANCHOR_COUNT = 64


def download(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty download: {url}")
    path.write_bytes(payload)
    return path


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_geometry_prior(dop_path: Path) -> Path:
    rdsm_path = GEOMETRY_DIR / "rdsm.tif"
    if rdsm_path.exists() and rdsm_path.stat().st_size > 0:
        with rasterio.open(rdsm_path) as src, rasterio.open(dop_path) as dop:
            if src.width == dop.width and src.height == dop.height and src.crs == dop.crs:
                print(f"Reusing cached DA3 geometry: {rdsm_path}")
                return rdsm_path

    print("Generating cached DA3MONO-LARGE geometry prior...")
    GEOMETRY_DIR.mkdir(parents=True, exist_ok=True)
    prior = DA3MonocularPrior(device="auto")
    scene = infer_geometry_scene(
        dop_path,
        prior,
        tile_size=768,
        overlap=128,
        harmonize_overlaps=True,
    )
    write_float_geotiff(
        rdsm_path,
        scene.relative_height,
        template_path=dop_path,
        description="DepthWizard DA3 relative geometry prior for OrthoLoC training acceptance",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": scene.model_id,
            "PURPOSE": "height_model_training_acceptance",
        },
    )
    return rdsm_path


def make_batch(
    windows: list[PatchWindow],
    *,
    rgb: np.ndarray,
    geometry: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if not windows:
        raise ValueError("make_batch requires at least one patch window")

    rgb_batch = np.stack(
        [rgb[w.row_slice, w.col_slice].transpose(2, 0, 1) for w in windows],
        axis=0,
    ).astype(np.float32, copy=False)
    geometry_batch = np.stack(
        [geometry[w.row_slice, w.col_slice] for w in windows],
        axis=0,
    )[:, None].astype(np.float32, copy=False)
    target_batch = np.stack(
        [target[w.row_slice, w.col_slice] for w in windows],
        axis=0,
    )[:, None].astype(np.float32, copy=False)
    valid_batch = np.stack(
        [valid[w.row_slice, w.col_slice] for w in windows],
        axis=0,
    )[:, None].astype(bool, copy=False)

    # Invalid source/geometry/reference values must not leak extreme NoData sentinels into
    # neighbouring valid convolutions even though the loss itself is masked.
    rgb_batch = np.where(valid_batch, rgb_batch, np.float32(0.0)).astype(np.float32, copy=False)
    geometry_batch = np.where(valid_batch, geometry_batch, np.float32(0.0)).astype(
        np.float32,
        copy=False,
    )
    target_batch = np.where(valid_batch, target_batch, np.float32(0.0)).astype(
        np.float32,
        copy=False,
    )

    return (
        torch.from_numpy(rgb_batch).to(device=device, dtype=torch.float32),
        torch.from_numpy(geometry_batch).to(device=device, dtype=torch.float32),
        torch.from_numpy(target_batch).to(device=device, dtype=torch.float32),
        torch.from_numpy(valid_batch).to(device=device, dtype=torch.bool),
    )


def augment_batch(
    tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rgb, geometry, target, valid = tensors
    if rng.random() < 0.5:
        rgb = torch.flip(rgb, dims=(-1,))
        geometry = torch.flip(geometry, dims=(-1,))
        target = torch.flip(target, dims=(-1,))
        valid = torch.flip(valid, dims=(-1,))
    if rng.random() < 0.5:
        rgb = torch.flip(rgb, dims=(-2,))
        geometry = torch.flip(geometry, dims=(-2,))
        target = torch.flip(target, dims=(-2,))
        valid = torch.flip(valid, dims=(-2,))
    rotations = int(rng.integers(0, 4))
    if rotations:
        rgb = torch.rot90(rgb, rotations, dims=(-2, -1))
        geometry = torch.rot90(geometry, rotations, dims=(-2, -1))
        target = torch.rot90(target, rotations, dims=(-2, -1))
        valid = torch.rot90(valid, rotations, dims=(-2, -1))
    return rgb, geometry, target, valid


def validation_loss(
    model: DepthWizardHeightModel,
    windows: list[PatchWindow],
    *,
    rgb: np.ndarray,
    geometry: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    gsd_m: float,
    device: torch.device,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[start : start + BATCH_SIZE]
            rgb_t, geometry_t, target_t, valid_t = make_batch(
                batch_windows,
                rgb=rgb,
                geometry=geometry,
                target=target,
                valid=valid,
                device=device,
            )
            gsd = torch.full((rgb_t.shape[0],), gsd_m, device=device)
            output = model(rgb_t, geometry_t, gsd_m=gsd)
            loss = compute_height_losses(output, target_t, valid_t).total
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("validation produced no batches")
    return float(np.mean(losses))


def predict_windows(
    model: DepthWizardHeightModel,
    windows: list[PatchWindow],
    *,
    rgb: np.ndarray,
    geometry: np.ndarray,
    valid: np.ndarray,
    gsd_m: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = int(geometry.shape[0]), int(geometry.shape[1])
    accumulator = np.zeros((height, width), dtype=np.float64)
    weights = np.zeros((height, width), dtype=np.float64)
    axis = np.maximum(np.hanning(PATCH_SIZE), 0.05)
    blend = np.outer(axis, axis).astype(np.float64)

    model.eval()
    with torch.inference_mode():
        for start in range(0, len(windows), BATCH_SIZE):
            batch_windows = windows[start : start + BATCH_SIZE]
            rgb_t, geometry_t, _, _ = make_batch(
                batch_windows,
                rgb=rgb,
                geometry=geometry,
                target=geometry,
                valid=valid,
                device=device,
            )
            gsd = torch.full((rgb_t.shape[0],), gsd_m, device=device)
            predictions = model(rgb_t, geometry_t, gsd_m=gsd).relative_height
            predicted = predictions[:, 0].detach().cpu().numpy()
            for index, window in enumerate(batch_windows):
                accumulator[window.row_slice, window.col_slice] += predicted[index] * blend
                weights[window.row_slice, window.col_slice] += blend

    output = np.full((height, width), np.nan, dtype=np.float32)
    covered = weights > 0
    output[covered] = (accumulator[covered] / weights[covered]).astype(np.float32)
    return output, covered


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dop_path = download(DOP_URL, DATA_DIR / "urban_residential_DOP.tif")
    dsm_path = download(DSM_URL, DATA_DIR / "urban_residential_DSM.tif")
    rdsm_path = ensure_geometry_prior(dop_path)

    rgb_raw = read_rgb(dop_path)
    with rasterio.open(dop_path) as src:
        source_valid = src.dataset_mask() > 0
    with rasterio.open(rdsm_path) as src:
        geometry = src.read(1).astype(np.float32)
        geometry_valid = np.isfinite(geometry)
        if src.nodata is not None:
            geometry_valid &= geometry != src.nodata
    reference, reference_valid = reproject_to_match(dsm_path, rdsm_path)
    paired_valid = source_valid & geometry_valid & reference_valid & np.isfinite(reference)

    raster_shape = (int(reference.shape[0]), int(reference.shape[1]))
    train_region, validation_region = spatial_column_holdout(
        raster_shape,
        train_fraction=TRAIN_FRACTION,
        gap_px=HOLDOUT_GAP_PX,
    )
    train_mask = paired_valid & train_region
    validation_mask = paired_valid & validation_region
    if int(train_mask.sum()) < 10_000 or int(validation_mask.sum()) < 10_000:
        raise RuntimeError("insufficient spatially disjoint paired pixels for training acceptance")

    target_scale = fit_robust_range(reference, train_mask)
    target = normalize_relative_target(reference, target_scale)
    rgb_ranges = fit_rgb_ranges(rgb_raw, train_mask)
    rgb = normalize_rgb(rgb_raw, rgb_ranges)

    train_windows = patch_windows(
        train_mask,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_fraction=0.98,
    )
    validation_windows = patch_windows(
        validation_mask,
        patch_size=PATCH_SIZE,
        stride=STRIDE,
        min_valid_fraction=0.98,
    )
    if not train_windows or not validation_windows:
        raise RuntimeError(
            f"training windows unavailable: train={len(train_windows)} val={len(validation_windows)}"
        )

    metric_gsd = ground_sample_distance_m(dop_path)
    if metric_gsd is None:
        raise RuntimeError("training acceptance requires georeferenced imagery")
    gsd_m = float(np.sqrt(metric_gsd[0] * metric_gsd[1]))

    config = HeightModelConfig()
    model = DepthWizardHeightModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    initial_validation_loss = validation_loss(
        model,
        validation_windows,
        rgb=rgb,
        geometry=geometry,
        target=target,
        valid=validation_mask,
        gsd_m=gsd_m,
        device=device,
    )
    best_validation_loss = initial_validation_loss
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()

    print(
        f"Training DepthWizard refiner on {len(train_windows)} patches; "
        f"validation={len(validation_windows)} patches; device={device}"
    )
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(train_windows))
        epoch_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            indices = order[start : start + BATCH_SIZE]
            windows = [train_windows[int(index)] for index in indices]
            batch = make_batch(
                windows,
                rgb=rgb,
                geometry=geometry,
                target=target,
                valid=train_mask,
                device=device,
            )
            rgb_t, geometry_t, target_t, valid_t = augment_batch(batch, rng)
            gsd = torch.full((rgb_t.shape[0],), gsd_m, device=device)

            optimizer.zero_grad(set_to_none=True)
            output = model(rgb_t, geometry_t, gsd_m=gsd)
            loss = compute_height_losses(output, target_t, valid_t).total
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_loss = validation_loss(
            model,
            validation_windows,
            rgb=rgb,
            geometry=geometry,
            target=target,
            valid=validation_mask,
            gsd_m=gsd_m,
            device=device,
        )
        mean_train_loss = float(np.mean(epoch_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": mean_train_loss,
                "validation_loss": val_loss,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch:02d}/{EPOCHS} | train {mean_train_loss:.5f} | "
            f"val {val_loss:.5f}"
        )
        if val_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    refined, covered = predict_windows(
        model,
        validation_windows,
        rgb=rgb,
        geometry=geometry,
        valid=validation_mask,
        gsd_m=gsd_m,
        device=device,
    )
    evaluation_valid = validation_mask & covered & np.isfinite(refined)
    baseline_benchmark = sparse_anchor_holdout_benchmark(
        geometry,
        reference,
        valid_mask=evaluation_valid,
        anchor_count=ANCHOR_COUNT,
        seed=SEED,
        exclusion_radius_px=4,
    )
    refined_benchmark = sparse_anchor_holdout_benchmark(
        refined,
        reference,
        valid_mask=evaluation_valid,
        anchor_count=ANCHOR_COUNT,
        seed=SEED,
        exclusion_radius_px=4,
    )

    refined_path = OUT_DIR / "validation_refined_rdsm.tif"
    write_float_geotiff(
        refined_path,
        refined,
        template_path=rdsm_path,
        description="DepthWizard spatial-holdout refined relative DSM",
        tags={
            "PURPOSE": "training_pipeline_acceptance",
            "EVALUATION_SCOPE": "spatial_holdout_same_scene_not_final_benchmark",
        },
    )

    checkpoint_path = OUT_DIR / "height_model_acceptance.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "target_range_m": asdict(target_scale),
        "rgb_ranges": [asdict(scale) for scale in rgb_ranges],
        "source": "OrthoLoC demo / urban_residential",
        "purpose": "training pipeline acceptance only",
    }
    torch.save(checkpoint, checkpoint_path)

    baseline_metrics = baseline_benchmark.metrics.model_dump()
    refined_metrics = refined_benchmark.metrics.model_dump()
    report = {
        "status": "PASS_TRAINING_PIPELINE",
        "purpose": (
            "Real-data training acceptance on a spatially disjoint region of one OrthoLoC demo "
            "scene. This is not the final cross-scene or cross-sensor scientific benchmark."
        ),
        "dataset": "OrthoLoC demo / urban_residential",
        "dataset_license": "CC BY-NC-SA 4.0",
        "device": str(device),
        "model_config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patch_size": PATCH_SIZE,
            "stride": STRIDE,
            "train_fraction": TRAIN_FRACTION,
            "holdout_gap_px": HOLDOUT_GAP_PX,
            "train_windows": len(train_windows),
            "validation_windows": len(validation_windows),
            "initial_validation_loss": initial_validation_loss,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "gsd_m": {"x": metric_gsd[0], "y": metric_gsd[1], "conditioning": gsd_m},
        "target_normalization_m": asdict(target_scale),
        "evaluation": {
            "protocol": "same_scene_spatial_holdout_plus_64_sparse_metric_anchors",
            "heldout_pixels": int(refined_benchmark.evaluation_mask.sum()),
            "da3_baseline": baseline_metrics,
            "depthwizard_refined": refined_metrics,
            "rmse_delta_m": float(refined_metrics["rmse_m"] - baseline_metrics["rmse_m"]),
            "mae_delta_m": float(refined_metrics["mae_m"] - baseline_metrics["mae_m"]),
        },
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "refined_rdsm": str(refined_path.resolve()),
    }
    report_path = OUT_DIR / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard real-data height training acceptance: PASS")
    print(f"Best epoch: {best_epoch} | validation loss: {best_validation_loss:.5f}")
    print(
        f"DA3 holdout RMSE: {baseline_metrics['rmse_m']:.3f} m | "
        f"DepthWizard: {refined_metrics['rmse_m']:.3f} m"
    )
    print(f"Checkpoint SHA-256: {report['checkpoint_sha256']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
