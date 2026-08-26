from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from depthwizard.data.ortholoc import discover_remote_scenes
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.confidence import (
    ConfidenceGatedLossResult,
    compute_confidence_gated_refinement_loss,
)
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v2 as v2
from scripts import train_ortholoc_multiscene_v3 as v3

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-multiscene-v4"
SEED = legacy.SEED
EPOCHS = legacy.EPOCHS
BATCH_SIZE = legacy.BATCH_SIZE
LEARNING_RATE = 6.0e-5


@dataclass(frozen=True)
class ObjectiveSummary:
    total: float
    predicted_gate_mean: float
    oracle_gate_mean: float


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
) -> ConfidenceGatedLossResult:
    rgb, geometry, target, valid, gsd = legacy.make_batch(scenes, patches, device)
    scales, baseline_rmse = _training_metadata_for_patches(scenes, patches, device)
    if augment:
        rgb, geometry, target, valid = legacy.augment_batch(rgb, geometry, target, valid, rng)
    output = model(rgb, geometry, gsd_m=gsd)
    return compute_confidence_gated_refinement_loss(
        output,
        geometry,
        target,
        valid,
        scales,
        baseline_rmse,
    )


def validation_objective(
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    patches: list[legacy.IndexedPatch],
    device: torch.device,
    rng: np.random.Generator,
) -> ObjectiveSummary:
    model.eval()
    weighted_total = 0.0
    weighted_predicted_gate = 0.0
    weighted_oracle_gate = 0.0
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
            weighted_predicted_gate += float(result.predicted_gate_mean.cpu()) * count
            weighted_oracle_gate += float(result.oracle_gate_mean.cpu()) * count
            samples += count
    if samples == 0:
        raise RuntimeError("confidence-gated validation produced no batches")
    return ObjectiveSummary(
        total=weighted_total / samples,
        predicted_gate_mean=weighted_predicted_gate / samples,
        oracle_gate_mean=weighted_oracle_gate / samples,
    )


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_float32_matmul_precision("high")
    rng = np.random.default_rng(SEED)
    device = legacy.resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_discovered = discover_remote_scenes("train")
    outplace_discovered = discover_remote_scenes("test_outPlace")
    train_remote, validation_remote, development_remote = v3.select_diversity_split(
        train_discovered,
        outplace_discovered,
    )
    print(
        "Confidence-gated geographic split: "
        f"train_locations={sorted({scene.location_id for scene in train_remote})} | "
        f"validation_locations={sorted({scene.location_id for scene in validation_remote})} | "
        f"development_outPlace={sorted({scene.location_id for scene in development_remote})}"
    )
    print(
        "Protocol lock: V4 reuses the V3 geographic composition and training-only replacement "
        "policy so the learning formulation is the principal changed variable. Validation scenes "
        "remain fixed and development outPlace remains previously inspected, non-final evidence."
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

    print(
        f"Accepted training scenes ({len(train_scenes)}): "
        f"{[scene.scene_id for scene in train_scenes]}"
    )
    print(
        "Training-only quality exclusions: "
        f"{[item['scene_id'] for item in training_rejections] or 'none'}"
    )
    print(f"Fixed validation scenes: {[scene.scene_id for scene in validation_scenes]}")

    for scene in train_scenes:
        fit = scene.prior_reference_fit
        if fit is None:
            raise RuntimeError(f"training scene {scene.scene_id} has no canonicalization fit")
        print(
            f"train {scene.scene_id}: scale={fit.scale_m_per_prior_unit:.3f} m/prior | "
            f"DA3 fit RMSE={fit.rmse_m:.3f} m | patches={len(scene.windows or [])}"
        )

    train_patches = legacy.build_patch_index(train_scenes)
    validation_patches = legacy.build_patch_index(validation_scenes)
    scene_patch_counts = {
        scene_index: sum(patch.scene_index == scene_index for patch in train_patches)
        for scene_index in range(len(train_scenes))
    }
    balanced_epoch_size = max(scene_patch_counts.values()) * len(scene_patch_counts)
    print(
        f"Confidence-gated sampling: raw patches={len(train_patches)} | "
        f"scene_patch_counts={scene_patch_counts} | samples/epoch={balanced_epoch_size}"
    )

    config = HeightModelConfig(architecture_version="confidence-gated-v2")
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
    validation_da3_rmse, initial_validation_rmse, initial_rejections = v2.calibrated_scene_metrics(
        model,
        validation_scenes,
        device,
    )
    if initial_rejections or not np.isfinite(initial_validation_rmse):
        raise RuntimeError(
            "identity DA3 confidence-validation calibration was unexpectedly rejected: "
            + "; ".join(initial_rejections)
        )
    if abs(initial_validation_rmse - validation_da3_rmse) > 1e-6:
        raise RuntimeError("confidence-gated epoch 0 violated exact DA3 identity")

    best_validation_rmse = initial_validation_rmse
    best_validation_objective = initial_objective.total
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, object]] = []

    print(
        f"Initial confidence validation: DA3 {validation_da3_rmse:.3f} m | "
        f"identity refiner {initial_validation_rmse:.3f} m | "
        f"predicted gate {initial_objective.predicted_gate_mean:.3f}"
    )
    print(
        f"Training confidence-gated refiner: architecture={config.architecture_version} | "
        f"device={device} | epochs={EPOCHS} | lr={LEARNING_RATE:.1e} | "
        "checkpoint criterion=fixed geographically disjoint sparse-anchor validation RMSE"
    )
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = v2._balanced_epoch_indices(train_patches, rng)
        weighted_train_loss = 0.0
        weighted_train_gate = 0.0
        weighted_train_oracle_gate = 0.0
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
                raise RuntimeError(f"non-finite confidence-gated training loss at epoch {epoch}")
            result.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            count = len(selected)
            weighted_train_loss += float(result.total.detach().cpu()) * count
            weighted_train_gate += float(result.predicted_gate_mean.detach().cpu()) * count
            weighted_train_oracle_gate += float(result.oracle_gate_mean.detach().cpu()) * count
            train_samples += count

        if train_samples != len(order):
            raise RuntimeError("confidence-gated epoch accounting mismatch")
        train_loss = weighted_train_loss / train_samples
        train_gate = weighted_train_gate / train_samples
        train_oracle_gate = weighted_train_oracle_gate / train_samples

        val_objective = validation_objective(
            model,
            validation_scenes,
            validation_patches,
            device,
            rng,
        )
        val_da3_rmse, val_refined_rmse, val_rejections = v2.calibrated_scene_metrics(
            model,
            validation_scenes,
            device,
        )
        delta = val_refined_rmse - val_da3_rmse if np.isfinite(val_refined_rmse) else float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_predicted_gate_mean": train_gate,
                "train_oracle_gate_mean": train_oracle_gate,
                "validation_objective": val_objective.total,
                "validation_predicted_gate_mean": val_objective.predicted_gate_mean,
                "validation_oracle_gate_mean": val_objective.oracle_gate_mean,
                "validation_da3_rmse_m": val_da3_rmse,
                "validation_depthwizard_rmse_m": (
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
                f"gate train={train_gate:.3f}/{train_oracle_gate:.3f} "
                f"val={val_objective.predicted_gate_mean:.3f}/"
                f"{val_objective.oracle_gate_mean:.3f} | val-RMSE REJECTED "
                f"(DA3 {val_da3_rmse:.3f}) | {'; '.join(val_rejections)}"
            )
        else:
            print(
                f"epoch {epoch:02d}/{EPOCHS} | train {train_loss:.5f} | "
                f"gate train={train_gate:.3f}/{train_oracle_gate:.3f} "
                f"val={val_objective.predicted_gate_mean:.3f}/"
                f"{val_objective.oracle_gate_mean:.3f} | val-RMSE {val_refined_rmse:.3f} m "
                f"(DA3 {val_da3_rmse:.3f}, delta {delta:+.3f})"
            )
            if val_refined_rmse < best_validation_rmse - 1e-6:
                best_validation_rmse = val_refined_rmse
                best_validation_objective = val_objective.total
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
    development_rejections: list[str] = []
    scene_deltas_m: list[float] = []
    for scene in development_scenes:
        evaluation = v2.evaluate_scene_safely(model, scene, device)
        scene_reports.append(evaluation.report)
        baseline_values.append(evaluation.baseline_values)
        reference_values.append(evaluation.reference_values)
        da3_rmse = v2._dict_metric(evaluation.report, "da3", "rmse_m")
        if evaluation.refined_values is None:
            reason = evaluation.refined_rejection_reason or "unspecified calibration rejection"
            development_rejections.append(f"{scene.scene_id}: {reason}")
            print(
                f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
                f"DepthWizard REJECTED | {reason}"
            )
            continue

        refined_values.append(evaluation.refined_values)
        if evaluation.uncertainty_values is None:
            raise RuntimeError("valid refined development evaluation is missing uncertainty values")
        uncertainty_values.append(evaluation.uncertainty_values)
        refined_rmse = v2._dict_metric(evaluation.report, "depthwizard", "rmse_m")
        delta_m = refined_rmse - da3_rmse
        scene_deltas_m.append(delta_m)
        print(
            f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"DepthWizard {refined_rmse:.3f} m | delta {delta_m:+.3f} m"
        )

    baseline_all = np.concatenate(baseline_values)
    reference_all = np.concatenate(reference_values)
    aggregate_da3 = compute_elevation_metrics(baseline_all, reference_all)

    aggregate_refined = None
    aggregate_reliability = None
    rmse_improvement_fraction = None
    if not development_rejections:
        refined_all = np.concatenate(refined_values)
        uncertainty_all = np.concatenate(uncertainty_values)
        aggregate_refined = compute_elevation_metrics(refined_all, reference_all)
        aggregate_reliability = legacy.uncertainty_error_correlation(
            uncertainty_all,
            np.abs(refined_all - reference_all),
            np.ones_like(reference_all, dtype=bool),
        )
        rmse_improvement_fraction = (
            aggregate_da3.rmse_m - aggregate_refined.rmse_m
        ) / aggregate_da3.rmse_m

    validation_improvement_fraction = (
        validation_da3_rmse - best_validation_rmse
    ) / validation_da3_rmse
    promoted = bool(
        aggregate_refined is not None
        and rmse_improvement_fraction is not None
        and rmse_improvement_fraction > 0.0
        and validation_improvement_fraction > 0.0
        and best_epoch > 0
    )

    checkpoint_path = OUT_DIR / "height_model_multiscene_v4.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "seed": SEED,
        "experiment": "ortholoc-multiscene-v4-confidence-gated",
        "best_epoch": best_epoch,
        "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
        "best_validation_objective": best_validation_objective,
        "validation_improvement_fraction": validation_improvement_fraction,
        "training_scene_ids": [scene.scene_id for scene in train_scenes],
        "training_locations": sorted({scene.location_id for scene in train_scenes}),
        "training_quality_exclusions": training_rejections,
        "validation_scene_ids": [scene.scene_id for scene in validation_scenes],
        "validation_locations": sorted({scene.location_id for scene in validation_scenes}),
        "development_outplace_locations": sorted(
            {scene.location_id for scene in development_scenes}
        ),
        "selection_metric": "fixed_two_location_validation_sparse_anchor_rmse",
        "purpose": "confidence-gated geographically disjoint OrthoLoC development acceptance",
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
        "status": "PASS_MULTISCENE_V4_CONFIDENCE_PIPELINE",
        "model_promoted_over_da3_on_development_outplace": promoted,
        "purpose": (
            "Confidence-gated residual development after V3 demonstrated a real validation "
            "improvement but a slight negative aggregate on previously inspected outPlace scenes. "
            "V4 preserves the V3 data split and isolates a conservative learning-formulation "
            "change. Final Gate B evidence still requires a new untouched holdout."
        ),
        "dataset": "OrthoLoC",
        "dataset_license": "CC BY-NC-SA 4.0",
        "device": str(device),
        "model_config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "train_scene_ids": [scene.scene_id for scene in train_scenes],
            "train_locations": sorted({scene.location_id for scene in train_scenes}),
            "training_quality_exclusions": training_rejections,
            "validation_scene_ids": [scene.scene_id for scene in validation_scenes],
            "validation_locations": sorted({scene.location_id for scene in validation_scenes}),
            "development_outPlace_locations": sorted(
                {scene.location_id for scene in development_scenes}
            ),
        },
        "training": {
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "patch_size": legacy.PATCH_SIZE,
            "stride": legacy.STRIDE,
            "raw_train_patches": len(train_patches),
            "scene_patch_counts": scene_patch_counts,
            "scene_balanced_samples_per_epoch": balanced_epoch_size,
            "validation_patches": len(validation_patches),
            "initial_validation_objective": asdict(initial_objective),
            "initial_validation_sparse_anchor_rmse_m": initial_validation_rmse,
            "best_validation_objective": best_validation_objective,
            "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
            "validation_improvement_fraction": validation_improvement_fraction,
            "best_epoch": best_epoch,
            "checkpoint_selection": (
                "minimum fixed disjoint validation RMSE; epoch 0 is exact DA3 identity"
            ),
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "development_evaluation": {
            "protocol": "previously_inspected_test_outPlace_plus_64_sparse_metric_anchors",
            "scenes": scene_reports,
            "scene_rmse_deltas_m": scene_deltas_m,
            "calibration_rejections": development_rejections,
            "aggregate_da3": aggregate_da3.model_dump(),
            "aggregate_depthwizard": (
                aggregate_refined.model_dump() if aggregate_refined is not None else None
            ),
            "rmse_delta_m": (
                float(aggregate_refined.rmse_m - aggregate_da3.rmse_m)
                if aggregate_refined is not None
                else None
            ),
            "rmse_improvement_fraction": (
                float(rmse_improvement_fraction)
                if rmse_improvement_fraction is not None
                else None
            ),
            "uncertainty_abs_error_pearson": aggregate_reliability,
        },
        "sources": source_manifest,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
    }
    report_path = OUT_DIR / "multiscene_v4_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard confidence-gated multiscene development acceptance: PASS")
    print(
        f"Best epoch: {best_epoch} | confidence validation sparse-anchor RMSE: "
        f"{best_validation_rmse:.3f} m | DA3: {validation_da3_rmse:.3f} m | "
        f"validation improvement: {100.0 * validation_improvement_fraction:.2f}%"
    )
    if aggregate_refined is None or rmse_improvement_fraction is None:
        print(
            f"Aggregate development-outPlace DA3 RMSE: {aggregate_da3.rmse_m:.3f} m | "
            "DepthWizard: REJECTED"
        )
        print("Development RMSE improvement: unavailable because calibration was rejected")
    else:
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
