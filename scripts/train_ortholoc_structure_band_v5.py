from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.structure_band import (
    StructureBandConfig,
    StructureBandLossResult,
    compute_structure_band_loss,
    physical_highpass_numpy,
    project_structure_correction_numpy,
)
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v2 as v2
from scripts import train_ortholoc_multiscene_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-structure-band-v5"
SEED = legacy.SEED
EPOCHS = legacy.EPOCHS
BATCH_SIZE = legacy.BATCH_SIZE
LEARNING_RATE = 8.0e-5
STRUCTURE = StructureBandConfig(
    target_outer_scale_m=8.0,
    safety_outer_scale_m=16.0,
    structure_threshold_m=1.0,
    max_structure_weight=5.0,
)
MAX_RELATIVE_CORRECTION = 0.50


def _training_metadata_for_patches(
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales: list[float] = []
    baseline_rmse: list[float] = []
    for patch in patches:
        fit = scenes[patch.scene_index].prior_reference_fit
        if fit is None:
            raise ValueError(
                f"scene {scenes[patch.scene_index].scene_id} has no prior/reference training fit"
            )
        if not np.isfinite(fit.scale_m_per_prior_unit) or fit.scale_m_per_prior_unit <= 0:
            raise ValueError("training scene has invalid positive metric scale")
        if not np.isfinite(fit.rmse_m) or fit.rmse_m <= 0:
            raise ValueError("training scene has invalid DA3 baseline RMSE")
        scales.append(fit.scale_m_per_prior_unit)
        baseline_rmse.append(fit.rmse_m)
    return (
        torch.tensor(scales, device=device, dtype=torch.float32),
        torch.tensor(baseline_rmse, device=device, dtype=torch.float32),
    )


def _training_objective(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
    *,
    augment: bool,
    rng: np.random.Generator,
) -> StructureBandLossResult:
    rgb, geometry, target, valid, gsd = legacy.make_batch(scenes, patches, device)
    scales, baseline_rmse = _training_metadata_for_patches(scenes, patches, device)
    if augment:
        rgb, geometry, target, valid = legacy.augment_batch(rgb, geometry, target, valid, rng)
    output = model(rgb, geometry, gsd_m=gsd)
    return compute_structure_band_loss(
        output,
        geometry,
        target,
        valid,
        gsd,
        scales,
        baseline_rmse,
        config=STRUCTURE,
    )


def validation_objective(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
    rng: np.random.Generator,
) -> float:
    model.eval()
    weighted_total = 0.0
    samples = 0
    with torch.inference_mode():
        for start in range(0, len(patches), BATCH_SIZE):
            selected = patches[start : start + BATCH_SIZE]
            result = _training_objective(
                model,
                scenes,
                selected,
                device,
                augment=False,
                rng=rng,
            )
            count = len(selected)
            weighted_total += float(result.total.cpu()) * count
            samples += count
    if samples == 0:
        raise RuntimeError("structure-band validation produced no batches")
    return weighted_total / samples


def _scene_structure_map(scene: legacy.SceneData) -> np.ndarray:
    if scene.target_prior is None or scene.prior_reference_fit is None:
        raise ValueError(f"scene {scene.scene_id} has no canonical structure target")
    correction_m = (
        scene.target_prior - scene.geometry
    ) * scene.prior_reference_fit.scale_m_per_prior_unit
    return physical_highpass_numpy(
        correction_m,
        scene.supervision_valid,
        gsd_m=scene.gsd_m,
        characteristic_scale_m=STRUCTURE.target_outer_scale_m,
    )


def _patch_structure_scores(
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
) -> np.ndarray:
    scene_maps = [_scene_structure_map(scene) for scene in scenes]
    scores = np.zeros(len(patches), dtype=np.float64)
    for index, patch in enumerate(patches):
        values = scene_maps[patch.scene_index][patch.window.row_slice, patch.window.col_slice]
        selected = np.abs(values[np.isfinite(values)])
        scores[index] = float(np.percentile(selected, 90.0)) if selected.size else 0.0
    return scores


def _structure_balanced_epoch_indices(
    patches: list[legacy.IndexedPatch],
    scores: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if scores.shape != (len(patches),):
        raise ValueError("structure score vector must match patch index")
    groups: dict[int, list[int]] = {}
    for index, patch in enumerate(patches):
        groups.setdefault(patch.scene_index, []).append(index)
    if not groups:
        raise ValueError("cannot sample an empty patch set")

    samples_per_scene = max(len(indices) for indices in groups.values())
    selected_groups: list[np.ndarray] = []
    for scene_index in sorted(groups):
        indices = np.asarray(groups[scene_index], dtype=np.int64)
        scene_scores = scores[indices]
        # Every patch keeps base probability 1.0. Object-rich patches gain up to 4x additional
        # weight, so quiet terrain remains represented while roofs/trees are seen more frequently.
        structure_boost = np.clip(
            scene_scores / STRUCTURE.structure_threshold_m,
            0.0,
            STRUCTURE.max_structure_weight - 1.0,
        )
        probabilities = 1.0 + structure_boost
        probabilities /= probabilities.sum()
        selected_groups.append(
            rng.choice(indices, size=samples_per_scene, replace=True, p=probabilities)
        )
    order = np.concatenate(selected_groups)
    rng.shuffle(order)
    return order


def _predict_structure_scene(
    model: DepthWizardHeightModel,
    scene: legacy.SceneData,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_refined, uncertainty, covered = legacy.predict_scene(model, scene, device)
    valid = scene.input_valid & covered & np.isfinite(raw_refined)
    raw_correction = raw_refined - scene.geometry
    safe_correction = project_structure_correction_numpy(
        raw_correction,
        valid,
        gsd_m=scene.gsd_m,
        config=STRUCTURE,
    )
    refined = np.full(scene.geometry.shape, np.nan, dtype=np.float32)
    safe_valid = valid & np.isfinite(safe_correction)
    refined[safe_valid] = scene.geometry[safe_valid] + safe_correction[safe_valid]
    return refined, uncertainty, safe_valid


def evaluate_scene_safely(
    model: DepthWizardHeightModel,
    scene: legacy.SceneData,
    device: torch.device,
) -> v2.SafeSceneEvaluation:
    refined, uncertainty, covered = _predict_structure_scene(model, scene, device)
    evaluation_valid = scene.supervision_valid & covered & np.isfinite(refined)

    baseline = sparse_anchor_holdout_benchmark(
        scene.geometry,
        scene.reference_m,
        valid_mask=evaluation_valid,
        anchor_count=legacy.ANCHOR_COUNT,
        seed=legacy.SEED,
        exclusion_radius_px=4,
    )
    try:
        refined_result = sparse_anchor_holdout_benchmark(
            refined,
            scene.reference_m,
            valid_mask=evaluation_valid,
            anchor_count=legacy.ANCHOR_COUNT,
            seed=legacy.SEED,
            exclusion_radius_px=4,
        )
    except ValueError as exc:
        common = baseline.evaluation_mask
        baseline_metrics = compute_elevation_metrics(
            baseline.prediction,
            scene.reference_m,
            valid_mask=common,
        )
        reason = str(exc)
        return v2.SafeSceneEvaluation(
            report={
                "scene_id": scene.scene_id,
                "location_id": scene.location_id,
                "gsd_m": scene.gsd_m,
                "heldout_pixels": int(common.sum()),
                "da3": baseline_metrics.model_dump(),
                "depthwizard": None,
                "depthwizard_calibration_valid": False,
                "depthwizard_calibration_rejection": reason,
                "uncertainty_abs_error_pearson": None,
            },
            baseline_values=baseline.prediction[common],
            refined_values=None,
            reference_values=scene.reference_m[common],
            uncertainty_values=None,
            refined_rejection_reason=reason,
        )

    common = baseline.evaluation_mask & refined_result.evaluation_mask
    baseline_metrics = compute_elevation_metrics(
        baseline.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    refined_metrics = compute_elevation_metrics(
        refined_result.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    reliability = legacy.uncertainty_error_correlation(
        uncertainty,
        np.abs(refined_result.prediction - scene.reference_m),
        common,
    )
    return v2.SafeSceneEvaluation(
        report={
            "scene_id": scene.scene_id,
            "location_id": scene.location_id,
            "gsd_m": scene.gsd_m,
            "heldout_pixels": int(common.sum()),
            "da3": baseline_metrics.model_dump(),
            "depthwizard": refined_metrics.model_dump(),
            "depthwizard_calibration_valid": True,
            "depthwizard_calibration_rejection": None,
            "rmse_delta_m": float(refined_metrics.rmse_m - baseline_metrics.rmse_m),
            "mae_delta_m": float(refined_metrics.mae_m - baseline_metrics.mae_m),
            "uncertainty_abs_error_pearson": reliability,
        },
        baseline_values=baseline.prediction[common],
        refined_values=refined_result.prediction[common],
        reference_values=scene.reference_m[common],
        uncertainty_values=uncertainty[common],
        refined_rejection_reason=None,
    )


def calibrated_scene_metrics(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    device: torch.device,
) -> tuple[float, float, list[str]]:
    baseline_values: list[np.ndarray] = []
    refined_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    rejections: list[str] = []
    for scene in scenes:
        evaluation = evaluate_scene_safely(model, scene, device)
        baseline_values.append(evaluation.baseline_values)
        reference_values.append(evaluation.reference_values)
        if evaluation.refined_values is None:
            reason = evaluation.refined_rejection_reason or "unspecified calibration rejection"
            rejections.append(f"{scene.scene_id}: {reason}")
        else:
            refined_values.append(evaluation.refined_values)

    baseline_all = np.concatenate(baseline_values)
    reference_all = np.concatenate(reference_values)
    baseline_metrics = compute_elevation_metrics(baseline_all, reference_all)
    if rejections:
        return baseline_metrics.rmse_m, float("inf"), rejections
    refined_all = np.concatenate(refined_values)
    refined_metrics = compute_elevation_metrics(refined_all, reference_all)
    return baseline_metrics.rmse_m, refined_metrics.rmse_m, []


def _metric(report: dict[str, object], section: str, metric: str) -> float:
    mapping = report.get(section)
    if not isinstance(mapping, dict):
        raise TypeError(f"report section {section!r} is not a metric mapping")
    value = mapping.get(metric)
    if not isinstance(value, (int, float)):
        raise TypeError(f"report metric {section}.{metric} is not numeric")
    return float(value)


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = legacy.resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_discovered = legacy.discover_remote_scenes("train")
    outplace_discovered = legacy.discover_remote_scenes("test_outPlace")
    train_remote, validation_remote, development_remote = v3.select_diversity_split(
        train_discovered,
        outplace_discovered,
    )
    print(
        "Structure-band geographic split: "
        f"train_locations={sorted({scene.location_id for scene in train_remote})} | "
        f"validation_locations={sorted({scene.location_id for scene in validation_remote})} | "
        f"development_outPlace={sorted({scene.location_id for scene in development_remote})}"
    )
    print(
        "Selection boundary: checkpoint epoch is chosen only by fixed geographically disjoint "
        "OrthoLoC validation sparse-anchor RMSE. Potsdam 2_14 is not used by this script."
    )

    prior = legacy.DA3MonocularPrior(device="auto")
    train_scenes, training_rejections = v3.load_training_scenes_with_replacement(
        train_remote,
        train_discovered,
        prior,
    )
    validation_scenes = [
        legacy.load_scene(scene, "validation", prior, include_target=True)
        for scene in validation_remote
    ]
    development_scenes = [
        legacy.load_scene(scene, "test_outPlace", prior, include_target=False)
        for scene in development_remote
    ]
    del prior

    train_patches = legacy.build_patch_index(train_scenes)
    validation_patches = legacy.build_patch_index(validation_scenes)
    structure_scores = _patch_structure_scores(train_scenes, train_patches)
    score_quantiles = {
        "p25_m": float(np.percentile(structure_scores, 25)),
        "p50_m": float(np.percentile(structure_scores, 50)),
        "p75_m": float(np.percentile(structure_scores, 75)),
        "p90_m": float(np.percentile(structure_scores, 90)),
        "p95_m": float(np.percentile(structure_scores, 95)),
    }
    scene_patch_counts = {
        scene_index: sum(patch.scene_index == scene_index for patch in train_patches)
        for scene_index in range(len(train_scenes))
    }
    balanced_epoch_size = max(scene_patch_counts.values()) * len(scene_patch_counts)
    print(
        f"Structure-aware sampling: patches={len(train_patches)} | "
        f"per-scene={scene_patch_counts} | samples/epoch={balanced_epoch_size} | "
        f"score_quantiles_m={score_quantiles}"
    )

    config = HeightModelConfig(
        architecture_version="bidirectional-cross-scale-v1",
        max_relative_correction=MAX_RELATIVE_CORRECTION,
    )
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
    validation_da3_rmse, initial_rmse, initial_rejections = calibrated_scene_metrics(
        model,
        validation_scenes,
        device,
    )
    if initial_rejections or not np.isfinite(initial_rmse):
        raise RuntimeError(
            "epoch-0 DA3 identity calibration unexpectedly failed: " + "; ".join(initial_rejections)
        )
    if abs(initial_rmse - validation_da3_rmse) > 1e-6:
        raise RuntimeError("epoch 0 violated exact DA3 identity")

    best_epoch = 0
    best_validation_rmse = initial_rmse
    best_validation_objective = initial_objective
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, object]] = []
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = _structure_balanced_epoch_indices(train_patches, structure_scores, rng)
        total_loss = 0.0
        train_samples = 0
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_patches[int(index)] for index in order[start : start + BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            result = _training_objective(
                model,
                train_scenes,
                selected,
                device,
                augment=True,
                rng=rng,
            )
            if not torch.isfinite(result.total):
                raise RuntimeError(f"non-finite structure-band loss at epoch {epoch}")
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            count = len(selected)
            total_loss += float(result.total.detach().cpu()) * count
            train_samples += count

        val_objective = validation_objective(
            model,
            validation_scenes,
            validation_patches,
            device,
            rng,
        )
        val_da3_rmse, val_refined_rmse, val_rejections = calibrated_scene_metrics(
            model,
            validation_scenes,
            device,
        )
        train_loss = total_loss / train_samples
        delta = val_refined_rmse - val_da3_rmse if np.isfinite(val_refined_rmse) else float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_objective": val_objective,
                "validation_da3_rmse_m": val_da3_rmse,
                "validation_structure_rmse_m": (
                    val_refined_rmse if np.isfinite(val_refined_rmse) else None
                ),
                "validation_rmse_delta_m": delta if np.isfinite(delta) else None,
                "validation_calibration_valid": not val_rejections,
                "validation_calibration_rejections": val_rejections,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        if val_rejections:
            print(
                f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | "
                f"val-objective {val_objective:.5f} | validation REJECTED | "
                f"{' ; '.join(val_rejections)}"
            )
        else:
            print(
                f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | "
                f"val-objective {val_objective:.5f} | RMSE {val_refined_rmse:.3f} m "
                f"(DA3 {val_da3_rmse:.3f}, delta {delta:+.3f})"
            )
            if val_refined_rmse < best_validation_rmse - 1e-6:
                best_epoch = epoch
                best_validation_rmse = val_refined_rmse
                best_validation_objective = val_objective
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
        scheduler.step()

    elapsed = time.perf_counter() - started
    model.load_state_dict(best_state)
    model.to(device)

    development_reports: list[dict[str, object]] = []
    baseline_values: list[np.ndarray] = []
    refined_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    development_rejections: list[str] = []
    scene_deltas: list[float] = []
    for scene in development_scenes:
        evaluation = evaluate_scene_safely(model, scene, device)
        development_reports.append(evaluation.report)
        baseline_values.append(evaluation.baseline_values)
        reference_values.append(evaluation.reference_values)
        da3_rmse = _metric(evaluation.report, "da3", "rmse_m")
        if evaluation.refined_values is None:
            reason = evaluation.refined_rejection_reason or "unspecified calibration rejection"
            development_rejections.append(f"{scene.scene_id}: {reason}")
            print(f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | structure REJECTED")
            continue
        refined_values.append(evaluation.refined_values)
        refined_rmse = _metric(evaluation.report, "depthwizard", "rmse_m")
        scene_deltas.append(refined_rmse - da3_rmse)
        print(
            f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"structure {refined_rmse:.3f} m | delta {refined_rmse - da3_rmse:+.3f} m"
        )

    baseline_all = np.concatenate(baseline_values)
    reference_all = np.concatenate(reference_values)
    aggregate_da3 = compute_elevation_metrics(baseline_all, reference_all)
    aggregate_refined = None
    if not development_rejections:
        aggregate_refined = compute_elevation_metrics(np.concatenate(refined_values), reference_all)

    validation_improvement = (
        validation_da3_rmse - best_validation_rmse
    ) / validation_da3_rmse
    development_non_degrading = bool(scene_deltas and max(scene_deltas) <= 0.0)
    development_improvement = (
        (aggregate_da3.rmse_m - aggregate_refined.rmse_m) / aggregate_da3.rmse_m
        if aggregate_refined is not None
        else None
    )
    promoted = bool(
        best_epoch > 0
        and validation_improvement > 0.0
        and aggregate_refined is not None
        and development_improvement is not None
        and development_improvement > 0.0
        and development_non_degrading
    )

    checkpoint_path = OUT_DIR / "height_model_structure_band_v5.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "structure_band": asdict(STRUCTURE),
        "seed": SEED,
        "best_epoch": best_epoch,
        "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
        "best_validation_objective": best_validation_objective,
        "selection_metric": "geographically_disjoint_validation_sparse_anchor_rmse",
        "training_locations": [scene.location_id for scene in train_scenes],
        "validation_locations": [scene.location_id for scene in validation_scenes],
        "development_outplace_locations": [scene.location_id for scene in development_scenes],
        "potsdam_2_14_used_for_training_or_selection": False,
    }
    torch.save(checkpoint, checkpoint_path)

    report = {
        "status": "PASS_STRUCTURE_BAND_TRAINING_PIPELINE",
        "operationally_promoted_on_ortholoc_development": promoted,
        "dataset": "OrthoLoC",
        "device": str(device),
        "model_config": asdict(config),
        "structure_band": asdict(STRUCTURE),
        "training_only_quality_rejections": training_rejections,
        "structure_patch_score_quantiles_m": score_quantiles,
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "raw_patches": len(train_patches),
            "structure_balanced_samples_per_epoch": balanced_epoch_size,
            "best_epoch": best_epoch,
            "initial_validation_rmse_m": initial_rmse,
            "best_validation_rmse_m": best_validation_rmse,
            "validation_da3_rmse_m": validation_da3_rmse,
            "validation_improvement_fraction": validation_improvement,
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "development": {
            "scenes": development_reports,
            "calibration_rejections": development_rejections,
            "aggregate_da3": aggregate_da3.model_dump(),
            "aggregate_structure": (
                aggregate_refined.model_dump() if aggregate_refined is not None else None
            ),
            "rmse_improvement_fraction": development_improvement,
            "all_scenes_non_degrading": development_non_degrading,
        },
        "scientific_boundary": (
            "Potsdam 2_14 is exposed development data but is not used by this training/selection "
            "script. Potsdam 4_12 and 6_12 remain reserved blind final urban confirmation scenes."
        ),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = OUT_DIR / "structure_band_v5_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard structure-band OrthoLoC training pipeline: PASS")
    print(
        f"Best epoch: {best_epoch} | validation DA3 {validation_da3_rmse:.3f} m | "
        f"structure {best_validation_rmse:.3f} m | "
        f"improvement {100.0 * validation_improvement:.2f}%"
    )
    if aggregate_refined is None or development_improvement is None:
        print("Development aggregate: structure candidate REJECTED")
    else:
        print(
            f"Development aggregate: DA3 {aggregate_da3.rmse_m:.3f} m | "
            f"structure {aggregate_refined.rmse_m:.3f} m | "
            f"improvement {100.0 * development_improvement:.2f}%"
        )
    print(f"All development scenes non-degrading: {'YES' if development_non_degrading else 'NO'}")
    print(f"Operational development promotion: {'YES' if promoted else 'NO'}")
    print(f"Checkpoint SHA-256: {report['checkpoint_sha256']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
