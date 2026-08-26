from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import torch

from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.multiscene import (
    balanced_group_order,
    compute_multiscene_refinement_loss,
)
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-multiscene-v2"
SEED = legacy.SEED
EPOCHS = legacy.EPOCHS
BATCH_SIZE = legacy.BATCH_SIZE
LEARNING_RATE = 1.0e-4


def _metric_scales_for_patches(
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
) -> torch.Tensor:
    scales: list[float] = []
    for patch in patches:
        fit = scenes[patch.scene_index].prior_reference_fit
        if fit is None:
            raise ValueError(
                f"scene {scenes[patch.scene_index].scene_id} has no prior/reference training fit"
            )
        scales.append(fit.scale_m_per_prior_unit)
    return torch.tensor(scales, device=device, dtype=torch.float32)


def _training_objective(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
    *,
    augment: bool,
    rng: np.random.Generator,
) -> torch.Tensor:
    rgb, geometry, target, valid, gsd = legacy.make_batch(scenes, patches, device)
    scales = _metric_scales_for_patches(scenes, patches, device)
    if augment:
        rgb, geometry, target, valid = legacy.augment_batch(rgb, geometry, target, valid, rng)
    output = model(rgb, geometry, gsd_m=gsd)
    return compute_multiscene_refinement_loss(
        output,
        geometry,
        target,
        valid,
        scales,
    ).total


def validation_objective(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(patches), BATCH_SIZE):
            selected = patches[start : start + BATCH_SIZE]
            loss = _training_objective(
                model,
                scenes,
                selected,
                device,
                augment=False,
                rng=rng,
            )
            losses.append(float(loss.cpu()))
    if not losses:
        raise RuntimeError("metric-aware validation produced no batches")
    return float(np.mean(losses))


def calibrated_scene_metrics(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    device: torch.device,
) -> tuple[float, float]:
    """Return aggregate DA3/refined RMSE under the actual sparse-anchor calibration protocol."""
    baseline_values: list[np.ndarray] = []
    refined_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    for scene in scenes:
        _, baseline, refined, reference, _ = legacy.evaluate_scene(model, scene, device)
        baseline_values.append(baseline)
        refined_values.append(refined)
        reference_values.append(reference)
    baseline_all = np.concatenate(baseline_values)
    refined_all = np.concatenate(refined_values)
    reference_all = np.concatenate(reference_values)
    baseline_metrics = compute_elevation_metrics(baseline_all, reference_all)
    refined_metrics = compute_elevation_metrics(refined_all, reference_all)
    return baseline_metrics.rmse_m, refined_metrics.rmse_m


def _balanced_epoch_indices(
    patches: list[legacy.IndexedPatch],
    rng: np.random.Generator,
) -> np.ndarray:
    return balanced_group_order([patch.scene_index for patch in patches], rng)


def _dict_metric(report: dict[str, object], key: str, metric: str) -> float:
    section = cast(dict[str, object], report[key])
    return float(section[metric])


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = legacy.resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_remote, validation_remote, development_remote = legacy.select_experiment_scenes()
    print(
        "Geographic split: "
        f"train={[scene.location_id for scene in train_remote]} | "
        f"validation={[scene.location_id for scene in validation_remote]} | "
        f"development_outPlace={[scene.location_id for scene in development_remote]}"
    )
    print(
        "Protocol note: L08/L50 are now a development out-of-place benchmark because their "
        "previous results have been inspected; they will not be used as final untouched evidence."
    )

    prior = legacy.DA3MonocularPrior(device="auto")
    train_scenes = [
        legacy.load_scene(scene, "train", prior, include_target=True) for scene in train_remote
    ]
    validation_scenes = [
        legacy.load_scene(scene, "validation", prior, include_target=True)
        for scene in validation_remote
    ]
    development_scenes = [
        legacy.load_scene(scene, "test_outPlace", prior, include_target=False)
        for scene in development_remote
    ]
    del prior

    for scene in train_scenes:
        fit = scene.prior_reference_fit
        if fit is None:
            raise RuntimeError(f"training scene {scene.scene_id} has no canonicalization fit")
        correction_scale = fit.rmse_m / fit.scale_m_per_prior_unit
        print(
            f"train {scene.scene_id}: {fit.scale_m_per_prior_unit:.3f} m/prior + "
            f"{fit.offset_m:.3f} m | baseline-fit RMSE {fit.rmse_m:.3f} m | "
            f"RMSE/prior-scale {correction_scale:.3f} | patches={len(scene.windows or [])}"
        )

    train_patches = legacy.build_patch_index(train_scenes)
    validation_patches = legacy.build_patch_index(validation_scenes)
    group_sizes = {
        scene_index: sum(patch.scene_index == scene_index for patch in train_patches)
        for scene_index in range(len(train_scenes))
    }
    balanced_epoch_size = max(group_sizes.values()) * len(group_sizes)
    print(
        f"Scene-balanced sampling: raw patches={len(train_patches)} | "
        f"per-scene={group_sizes} | samples/epoch={balanced_epoch_size}"
    )

    config = HeightModelConfig()
    model = DepthWizardHeightModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    initial_objective = validation_objective(
        model,
        validation_scenes,
        validation_patches,
        device,
        rng,
    )
    validation_da3_rmse, initial_validation_rmse = calibrated_scene_metrics(
        model,
        validation_scenes,
        device,
    )
    best_validation_rmse = initial_validation_rmse
    best_validation_objective = initial_objective
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, float | int]] = []

    print(
        f"Initial validation calibration: DA3 {validation_da3_rmse:.3f} m | "
        f"identity refiner {initial_validation_rmse:.3f} m"
    )
    print(
        f"Training metric-aware multiscene refiner: device={device} | epochs={EPOCHS} | "
        "checkpoint criterion=sparse-anchor validation RMSE"
    )
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = _balanced_epoch_indices(train_patches, rng)
        epoch_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_patches[int(index)] for index in order[start : start + BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            loss = _training_objective(
                model,
                train_scenes,
                selected,
                device,
                augment=True,
                rng=rng,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite metric-aware training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_objective = validation_objective(
            model,
            validation_scenes,
            validation_patches,
            device,
            rng,
        )
        val_da3_rmse, val_refined_rmse = calibrated_scene_metrics(
            model,
            validation_scenes,
            device,
        )
        train_loss = float(np.mean(epoch_losses))
        delta = val_refined_rmse - val_da3_rmse
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_objective": val_objective,
                "validation_da3_rmse_m": val_da3_rmse,
                "validation_depthwizard_rmse_m": val_refined_rmse,
                "validation_rmse_delta_m": delta,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | "
            f"val-objective {val_objective:.5f} | val-RMSE {val_refined_rmse:.3f} m "
            f"(DA3 {val_da3_rmse:.3f}, delta {delta:+.3f})"
        )
        if val_refined_rmse < best_validation_rmse - 1e-6:
            best_validation_rmse = val_refined_rmse
            best_validation_objective = val_objective
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
    for scene in development_scenes:
        scene_report, baseline, refined, reference, uncertainty = legacy.evaluate_scene(
            model,
            scene,
            device,
        )
        scene_reports.append(scene_report)
        baseline_values.append(baseline)
        refined_values.append(refined)
        reference_values.append(reference)
        uncertainty_values.append(uncertainty)
        da3_rmse = _dict_metric(scene_report, "da3", "rmse_m")
        refined_rmse = _dict_metric(scene_report, "depthwizard", "rmse_m")
        print(
            f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"DepthWizard {refined_rmse:.3f} m | delta {refined_rmse - da3_rmse:+.3f} m"
        )

    baseline_all = np.concatenate(baseline_values)
    refined_all = np.concatenate(refined_values)
    reference_all = np.concatenate(reference_values)
    uncertainty_all = np.concatenate(uncertainty_values)
    aggregate_da3 = compute_elevation_metrics(baseline_all, reference_all)
    aggregate_refined = compute_elevation_metrics(refined_all, reference_all)
    aggregate_reliability = legacy.uncertainty_error_correlation(
        uncertainty_all,
        np.abs(refined_all - reference_all),
        np.ones_like(reference_all, dtype=bool),
    )
    rmse_improvement_fraction = (
        aggregate_da3.rmse_m - aggregate_refined.rmse_m
    ) / aggregate_da3.rmse_m
    promoted = bool(rmse_improvement_fraction > 0.0 and best_epoch > 0)

    checkpoint_path = OUT_DIR / "height_model_multiscene_v2.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
        "best_validation_objective": best_validation_objective,
        "training_locations": [scene.location_id for scene in train_scenes],
        "validation_locations": [scene.location_id for scene in validation_scenes],
        "development_outplace_locations": [scene.location_id for scene in development_scenes],
        "selection_metric": "geographically_disjoint_validation_sparse_anchor_rmse",
        "purpose": "metric-aware geographically disjoint OrthoLoC development acceptance",
    }
    torch.save(checkpoint, checkpoint_path)

    source_manifest: list[dict[str, object]] = []
    for scene in train_scenes + validation_scenes + development_scenes:
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
        "status": "PASS_MULTISCENE_V2_PIPELINE",
        "model_promoted_over_da3_on_development_outplace": promoted,
        "purpose": (
            "Metric-aware, scene-balanced multiscene development acceptance. L08/L50 are no "
            "longer treated as untouched test evidence because their previous results informed "
            "training-system development. Final Gate B claims require a new independent holdout."
        ),
        "dataset": "OrthoLoC",
        "dataset_license": "CC BY-NC-SA 4.0",
        "device": str(device),
        "model_config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "train_locations": [scene.location_id for scene in train_scenes],
            "validation_locations": [scene.location_id for scene in validation_scenes],
            "development_outPlace_locations": [scene.location_id for scene in development_scenes],
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "patch_size": legacy.PATCH_SIZE,
            "stride": legacy.STRIDE,
            "raw_train_patches": len(train_patches),
            "scene_balanced_samples_per_epoch": balanced_epoch_size,
            "validation_patches": len(validation_patches),
            "initial_validation_objective": initial_objective,
            "initial_validation_sparse_anchor_rmse_m": initial_validation_rmse,
            "best_validation_objective": best_validation_objective,
            "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
            "best_epoch": best_epoch,
            "checkpoint_selection": "minimum sparse-anchor validation RMSE; epoch 0 is DA3 identity",
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "development_evaluation": {
            "protocol": "previously_inspected_test_outPlace_plus_64_sparse_metric_anchors",
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
    report_path = OUT_DIR / "multiscene_v2_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard metric-aware multiscene development acceptance: PASS")
    print(
        f"Best epoch: {best_epoch} | validation sparse-anchor RMSE: "
        f"{best_validation_rmse:.3f} m | DA3: {validation_da3_rmse:.3f} m"
    )
    print(
        f"Aggregate development-outPlace DA3 RMSE: {aggregate_da3.rmse_m:.3f} m | "
        f"DepthWizard: {aggregate_refined.rmse_m:.3f} m"
    )
    print(f"Development RMSE improvement: {100.0 * rmse_improvement_fraction:.2f}%")
    print(f"Development model promoted over DA3: {'YES' if promoted else 'NO'}")
    print(f"Checkpoint SHA-256: {report['checkpoint_sha256']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
