from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from depthwizard.data.ortholoc import (
    OrthoLoCRemoteScene,
    discover_remote_scenes,
    select_geographic_scenes,
)
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.training import PriorReferenceCalibrationError
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v2 as v2

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-multiscene-v3"
SEED = legacy.SEED
EPOCHS = legacy.EPOCHS
BATCH_SIZE = legacy.BATCH_SIZE
LEARNING_RATE = 6.0e-5
TRAIN_LOCATIONS = 5
TRAIN_SAMPLES_PER_LOCATION = 2
VALIDATION_LOCATIONS = 2
VALIDATION_SAMPLES_PER_LOCATION = 1
DEVELOPMENT_LOCATIONS = 2
DEVELOPMENT_SAMPLES_PER_LOCATION = 1


def _scene_lookup(scenes: list[OrthoLoCRemoteScene]) -> dict[str, OrthoLoCRemoteScene]:
    lookup = {scene.filename: scene for scene in scenes}
    if len(lookup) != len(scenes):
        raise RuntimeError("duplicate OrthoLoC filenames discovered within one split")
    return lookup


def select_diversity_split(
    train_discovered: list[OrthoLoCRemoteScene],
    outplace_discovered: list[OrthoLoCRemoteScene],
) -> tuple[
    list[OrthoLoCRemoteScene],
    list[OrthoLoCRemoteScene],
    list[OrthoLoCRemoteScene],
]:
    """Build a larger geographically disjoint development protocol.

    V1/V2 used one sample from each of four training locations and one validation location. Those
    experiments showed that both the one-way and bidirectional refiners could reduce training loss
    while degrading the unseen location. V3 therefore changes the data evidence before changing
    the architecture again: five train locations contribute two independent DOP/DSM scenes each,
    while two entirely different train-split locations are reserved for model selection. The
    previously inspected out-of-place scenes remain development-only evidence.
    """
    train_pool = [scene for scene in train_discovered if scene.same_domain]
    if not train_pool:
        raise RuntimeError("official OrthoLoC train listing contains no same-domain scenes")
    train_lookup = _scene_lookup(train_pool)
    train_names = [scene.filename for scene in train_pool]

    train_names_selected = select_geographic_scenes(
        train_names,
        max_locations=TRAIN_LOCATIONS,
        samples_per_location=TRAIN_SAMPLES_PER_LOCATION,
    )
    train_selected = [train_lookup[name] for name in train_names_selected]
    train_location_counts = Counter(scene.location_id for scene in train_selected)
    incomplete = {
        location: count
        for location, count in train_location_counts.items()
        if count != TRAIN_SAMPLES_PER_LOCATION
    }
    if incomplete:
        raise RuntimeError(
            "diversity acceptance requires exactly two scenes per training location; "
            f"insufficient same-domain samples for {incomplete}"
        )
    train_locations = set(train_location_counts)

    validation_names_selected = select_geographic_scenes(
        train_names,
        max_locations=VALIDATION_LOCATIONS,
        samples_per_location=VALIDATION_SAMPLES_PER_LOCATION,
        excluded_locations=train_locations,
    )
    validation_selected = [train_lookup[name] for name in validation_names_selected]
    validation_locations = {scene.location_id for scene in validation_selected}
    if train_locations & validation_locations:
        raise RuntimeError("geographic leakage between V3 training and validation locations")

    outplace_pool = [scene for scene in outplace_discovered if scene.same_domain]
    if not outplace_pool:
        raise RuntimeError("official OrthoLoC test_outPlace listing contains no same-domain scenes")
    outplace_lookup = _scene_lookup(outplace_pool)
    outplace_names = [scene.filename for scene in outplace_pool]
    development_names_selected = select_geographic_scenes(
        outplace_names,
        max_locations=DEVELOPMENT_LOCATIONS,
        samples_per_location=DEVELOPMENT_SAMPLES_PER_LOCATION,
        excluded_locations=train_locations | validation_locations,
    )
    development_selected = [outplace_lookup[name] for name in development_names_selected]
    development_locations = {scene.location_id for scene in development_selected}
    if (train_locations | validation_locations) & development_locations:
        raise RuntimeError("geographic leakage into V3 out-of-place development locations")

    return train_selected, validation_selected, development_selected


def training_candidate_groups(
    selected_train: list[OrthoLoCRemoteScene],
    train_discovered: list[OrthoLoCRemoteScene],
) -> dict[str, list[OrthoLoCRemoteScene]]:
    """Return deterministic same-location fallback queues for training-only quality filtering.

    Geographic groups are fixed *before* any DSM-dependent quality decision. Within each declared
    training location, the originally selected scenes are attempted first and remaining same-domain
    scenes are ordered by filename. This lets the training set replace a DA3/reference pair that
    violates the required positive-height convention without changing geographic composition.

    This helper is deliberately training-only. Validation scenes must never be replaced after
    looking at their reference relationship because doing so would condition model selection on
    validation labels.
    """
    selected_locations = sorted({scene.location_id for scene in selected_train})
    selected_by_location: dict[str, list[OrthoLoCRemoteScene]] = {
        location: sorted(
            [scene for scene in selected_train if scene.location_id == location],
            key=lambda scene: scene.filename,
        )
        for location in selected_locations
    }
    discovered_by_location: dict[str, list[OrthoLoCRemoteScene]] = {
        location: sorted(
            [
                scene
                for scene in train_discovered
                if scene.same_domain and scene.location_id == location
            ],
            key=lambda scene: scene.filename,
        )
        for location in selected_locations
    }

    groups: dict[str, list[OrthoLoCRemoteScene]] = {}
    for location in selected_locations:
        preferred = selected_by_location[location]
        preferred_names = {scene.filename for scene in preferred}
        remaining = [
            scene
            for scene in discovered_by_location[location]
            if scene.filename not in preferred_names
        ]
        candidates = preferred + remaining
        if len(candidates) < TRAIN_SAMPLES_PER_LOCATION:
            raise RuntimeError(
                f"training location {location} exposes only {len(candidates)} same-domain scenes; "
                f"{TRAIN_SAMPLES_PER_LOCATION} are required"
            )
        groups[location] = candidates
    return groups


def load_training_scenes_with_replacement(
    selected_train: list[OrthoLoCRemoteScene],
    train_discovered: list[OrthoLoCRemoteScene],
    prior: legacy.DA3MonocularPrior,
) -> tuple[list[legacy.SceneData], list[dict[str, str]]]:
    """Load exactly two usable training scenes per fixed geographic group.

    Only ``PriorReferenceCalibrationError`` is recoverable here. All IO, alignment, CRS, shape,
    GSD, and implementation errors remain fatal. Rejections are recorded so the final provenance
    states exactly which training examples were screened and why.
    """
    groups = training_candidate_groups(selected_train, train_discovered)
    accepted: list[legacy.SceneData] = []
    rejections: list[dict[str, str]] = []

    for location, candidates in groups.items():
        accepted_for_location = 0
        for remote in candidates:
            try:
                scene = legacy.load_scene(remote, "train", prior, include_target=True)
            except PriorReferenceCalibrationError as exc:
                reason = str(exc)
                rejections.append(
                    {
                        "scene_id": remote.scene_id,
                        "location_id": location,
                        "reason": reason,
                    }
                )
                print(
                    f"training quality rejection: {remote.scene_id} | {reason} | "
                    "trying next pre-declared same-location candidate"
                )
                continue

            accepted.append(scene)
            accepted_for_location += 1
            if accepted_for_location == TRAIN_SAMPLES_PER_LOCATION:
                break

        if accepted_for_location != TRAIN_SAMPLES_PER_LOCATION:
            rejected_ids = [
                item["scene_id"] for item in rejections if item["location_id"] == location
            ]
            raise RuntimeError(
                f"training location {location} could supply only {accepted_for_location}/"
                f"{TRAIN_SAMPLES_PER_LOCATION} positive-height scenes after deterministic "
                f"training-only quality screening; rejected={rejected_ids}"
            )

    counts = Counter(scene.location_id for scene in accepted)
    expected = {
        location: TRAIN_SAMPLES_PER_LOCATION
        for location in sorted({scene.location_id for scene in selected_train})
    }
    if counts != expected:
        raise RuntimeError(
            f"training replacement violated fixed geographic composition: got={dict(counts)} "
            f"expected={expected}"
        )
    return accepted, rejections


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
    train_remote, validation_remote, development_remote = select_diversity_split(
        train_discovered,
        outplace_discovered,
    )
    print(
        "Diversity geographic split: "
        f"train_locations={sorted({scene.location_id for scene in train_remote})} | "
        f"validation_locations={sorted({scene.location_id for scene in validation_remote})} | "
        f"development_outPlace={sorted({scene.location_id for scene in development_remote})}"
    )
    print(
        f"Initial training candidates ({len(train_remote)}): "
        f"{[scene.scene_id for scene in train_remote]}"
    )
    print(f"Fixed validation scenes ({len(validation_remote)}): {[scene.scene_id for scene in validation_remote]}")
    print(
        "Protocol note: training-only prior/reference quality screening may replace a rejected "
        "scene only within its already-declared training location. Validation scenes are fixed "
        "before inspection and are never label-conditioned replacements. Out-of-place development "
        "scenes have already been inspected and are not eligible as final Gate B evidence."
    )

    prior = legacy.DA3MonocularPrior(device="auto")
    train_scenes, training_quality_rejections = load_training_scenes_with_replacement(
        train_remote,
        train_discovered,
        prior,
    )
    try:
        validation_scenes = [
            legacy.load_scene(scene, "validation", prior, include_target=True)
            for scene in validation_remote
        ]
    except PriorReferenceCalibrationError as exc:
        raise RuntimeError(
            "a fixed V3 validation scene cannot support the positive-height DA3 calibration "
            "protocol. The run is intentionally aborted rather than selecting a replacement after "
            "examining validation reference data; define a new protocol version before changing "
            "validation membership."
        ) from exc
    development_scenes = [
        legacy.load_scene(scene, "test_outPlace", prior, include_target=False)
        for scene in development_remote
    ]
    del prior

    print(
        f"Accepted training scenes ({len(train_scenes)}): {[scene.scene_id for scene in train_scenes]}"
    )
    if training_quality_rejections:
        print(
            "Training-only quality exclusions: "
            f"{[item['scene_id'] for item in training_quality_rejections]}"
        )

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
    scene_patch_counts = {
        scene_index: sum(patch.scene_index == scene_index for patch in train_patches)
        for scene_index in range(len(train_scenes))
    }
    balanced_epoch_size = max(scene_patch_counts.values()) * len(scene_patch_counts)
    location_patch_counts = {
        location: sum(
            scene_patch_counts[index]
            for index, scene in enumerate(train_scenes)
            if scene.location_id == location
        )
        for location in sorted({scene.location_id for scene in train_scenes})
    }
    print(
        f"Diversity sampling: raw patches={len(train_patches)} | "
        f"scene_patch_counts={scene_patch_counts} | location_patch_counts={location_patch_counts} | "
        f"scene-balanced samples/epoch={balanced_epoch_size}"
    )

    config = HeightModelConfig()
    model = DepthWizardHeightModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    initial_objective = v2.validation_objective(
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
            "identity DA3 diversity-validation calibration was unexpectedly rejected: "
            + "; ".join(initial_rejections)
        )

    best_validation_rmse = initial_validation_rmse
    best_validation_objective = initial_objective
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, object]] = []

    print(
        f"Initial diversity validation: DA3 {validation_da3_rmse:.3f} m | "
        f"identity refiner {initial_validation_rmse:.3f} m"
    )
    print(
        f"Training diversity-first refiner: architecture={config.architecture_version} | "
        f"device={device} | epochs={EPOCHS} | lr={LEARNING_RATE:.1e} | "
        "checkpoint criterion=geographically disjoint sparse-anchor validation RMSE"
    )
    started = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = v2._balanced_epoch_indices(train_patches, rng)
        epoch_losses: list[float] = []
        for start in range(0, len(order), BATCH_SIZE):
            selected = [train_patches[int(index)] for index in order[start : start + BATCH_SIZE]]
            optimizer.zero_grad(set_to_none=True)
            loss = v2._training_objective(
                model,
                train_scenes,
                selected,
                device,
                augment=True,
                rng=rng,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite diversity training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        val_objective = v2.validation_objective(
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
        train_loss = float(np.mean(epoch_losses))
        delta = val_refined_rmse - val_da3_rmse if np.isfinite(val_refined_rmse) else float("inf")
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_objective": val_objective,
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
                f"val-objective {val_objective:.5f} | val-RMSE REJECTED "
                f"(DA3 {val_da3_rmse:.3f}) | {'; '.join(val_rejections)}"
            )
        else:
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
    development_rejections: list[str] = []
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
        print(
            f"development {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"DepthWizard {refined_rmse:.3f} m | delta {refined_rmse - da3_rmse:+.3f} m"
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

    promoted = bool(
        aggregate_refined is not None
        and rmse_improvement_fraction is not None
        and rmse_improvement_fraction > 0.0
        and best_epoch > 0
    )

    checkpoint_path = OUT_DIR / "height_model_multiscene_v3.pt"
    checkpoint = {
        "state_dict": best_state,
        "config": asdict(config),
        "seed": SEED,
        "experiment": "ortholoc-multiscene-v3-diversity-first",
        "best_epoch": best_epoch,
        "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
        "best_validation_objective": best_validation_objective,
        "training_scene_ids": [scene.scene_id for scene in train_scenes],
        "training_locations": sorted({scene.location_id for scene in train_scenes}),
        "training_quality_rejections": training_quality_rejections,
        "validation_scene_ids": [scene.scene_id for scene in validation_scenes],
        "validation_locations": sorted({scene.location_id for scene in validation_scenes}),
        "development_outplace_locations": sorted(
            {scene.location_id for scene in development_scenes}
        ),
        "selection_metric": "geographically_disjoint_two-location_validation_sparse_anchor_rmse",
        "purpose": "diversity-first geographically disjoint OrthoLoC development acceptance",
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
        "status": "PASS_MULTISCENE_V3_DIVERSITY_PIPELINE",
        "model_promoted_over_da3_on_development_outplace": promoted,
        "purpose": (
            "Diversity-first development acceptance after V1/V2 showed that small four-location "
            "training did not transfer. Five fixed training locations contribute two usable scenes "
            "each after training-only positive-height quality screening, and two different fixed "
            "locations are held out for checkpoint selection. Validation membership is never "
            "changed using reference-derived quality information. Development outPlace results are "
            "not final evidence because those locations have already been inspected."
        ),
        "dataset": "OrthoLoC",
        "dataset_license": "CC BY-NC-SA 4.0",
        "device": str(device),
        "model_config": asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "train_scene_ids": [scene.scene_id for scene in train_scenes],
            "train_locations": sorted({scene.location_id for scene in train_scenes}),
            "training_quality_rejections": training_quality_rejections,
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
            "location_patch_counts": location_patch_counts,
            "scene_balanced_samples_per_epoch": balanced_epoch_size,
            "validation_patches": len(validation_patches),
            "initial_validation_objective": initial_objective,
            "initial_validation_sparse_anchor_rmse_m": initial_validation_rmse,
            "best_validation_objective": best_validation_objective,
            "best_validation_sparse_anchor_rmse_m": best_validation_rmse,
            "best_epoch": best_epoch,
            "checkpoint_selection": "minimum disjoint validation RMSE; epoch 0 is exact DA3 identity",
            "wall_time_seconds": elapsed,
            "history": history,
        },
        "development_evaluation": {
            "protocol": "previously_inspected_test_outPlace_plus_64_sparse_metric_anchors",
            "scenes": scene_reports,
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
    report_path = OUT_DIR / "multiscene_v3_training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard diversity-first multiscene development acceptance: PASS")
    print(
        f"Best epoch: {best_epoch} | diversity validation sparse-anchor RMSE: "
        f"{best_validation_rmse:.3f} m | DA3: {validation_da3_rmse:.3f} m"
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
