from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from depthwizard.data.ortholoc import OrthoLoCRemoteScene, discover_remote_scenes
from depthwizard.evaluation.adaptive import adaptive_sparse_anchor_holdout_benchmark
from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel
from depthwizard.provenance.manifest import canonical_json_hash, sha256_file
from scripts import evaluate_ortholoc_adaptive_refinement as adaptive_eval
from scripts import train_ortholoc_multiscene as legacy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "ortholoc-frozen-same-domain-v1"
OUT_DIR = ROOT / "artifacts" / "evaluation" / PROTOCOL_VERSION
SEAL_PATH = OUT_DIR / "protocol_seal.json"
REPORT_PATH = OUT_DIR / "frozen_holdout_report.json"
HOLDOUT_LOCATIONS = 4
SAMPLES_PER_LOCATION = 2
EXCLUDED_LOCATIONS = frozenset(
    {
        "L01",
        "L02",
        "L03",
        "L04",
        "L05",
        "L06",
        "L07",
        "L08",
        "L50",
    }
)


def _location_sort_key(location_id: str) -> tuple[int, str]:
    suffix = location_id[1:] if location_id.startswith("L") else ""
    return (int(suffix), location_id) if suffix.isdigit() else (10**9, location_id)


def select_frozen_holdout_scenes(
    discovered: list[OrthoLoCRemoteScene],
) -> list[OrthoLoCRemoteScene]:
    """Select holdout scenes using listing metadata only, before any DSM target is loaded.

    The V1 protocol is intentionally deterministic. It excludes every location already used for
    training, validation, or development, keeps same-domain OrthoLoC samples only, chooses the first
    four remaining location IDs in numeric order, and then chooses the first two filenames within
    each location. A location that cannot provide two samples is not eligible. No target-dependent
    replacement is permitted after the protocol is sealed.
    """
    grouped: dict[str, list[OrthoLoCRemoteScene]] = defaultdict(list)
    for scene in discovered:
        if not scene.same_domain or scene.location_id in EXCLUDED_LOCATIONS:
            continue
        grouped[scene.location_id].append(scene)

    for scenes in grouped.values():
        scenes.sort(key=lambda scene: scene.filename)
    eligible_locations = sorted(
        (
            location
            for location, scenes in grouped.items()
            if len(scenes) >= SAMPLES_PER_LOCATION
        ),
        key=_location_sort_key,
    )
    if len(eligible_locations) < HOLDOUT_LOCATIONS:
        raise RuntimeError(
            "frozen holdout selection stopped before loading any DSM targets: "
            f"need {HOLDOUT_LOCATIONS} untouched same-domain locations with at least "
            f"{SAMPLES_PER_LOCATION} scenes each, found {len(eligible_locations)} "
            f"({eligible_locations})"
        )

    selected_locations = eligible_locations[:HOLDOUT_LOCATIONS]
    selected = [
        scene
        for location in selected_locations
        for scene in grouped[location][:SAMPLES_PER_LOCATION]
    ]
    if len(selected) != HOLDOUT_LOCATIONS * SAMPLES_PER_LOCATION:
        raise RuntimeError("frozen holdout selector violated the declared scene-count invariant")
    if {scene.location_id for scene in selected} & EXCLUDED_LOCATIONS:
        raise RuntimeError("frozen holdout selector leaked a previously inspected location")
    return selected


def _checkpoint_contract(checkpoint: dict[str, Any]) -> tuple[int, dict[str, object]]:
    best_epoch = checkpoint.get("best_epoch")
    config = checkpoint.get("config")
    if not isinstance(best_epoch, int) or best_epoch <= 0:
        raise TypeError("frozen holdout requires a learned V4 checkpoint with best_epoch > 0")
    if not isinstance(config, dict):
        raise TypeError("frozen holdout checkpoint config is not a dictionary")
    return best_epoch, config


def build_protocol_payload(
    selected: list[OrthoLoCRemoteScene],
    checkpoint: dict[str, Any],
) -> dict[str, object]:
    """Build the complete pre-target protocol contract that is hashed into the local seal."""
    best_epoch, config = _checkpoint_contract(checkpoint)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "One-time untouched same-domain geographic holdout for the already-frozen V4 model "
            "and V5 anchor-evidence adaptive policy."
        ),
        "selection": {
            "source_split": "test_outPlace",
            "same_domain_only": True,
            "excluded_locations": sorted(EXCLUDED_LOCATIONS, key=_location_sort_key),
            "holdout_locations": HOLDOUT_LOCATIONS,
            "samples_per_location": SAMPLES_PER_LOCATION,
            "selection_rule": (
                "numeric location order after exclusions; first two filenames per eligible location"
            ),
            "selected_scene_ids": [scene.scene_id for scene in selected],
            "selected_location_ids": [scene.location_id for scene in selected],
        },
        "model": {
            "checkpoint_path": str(adaptive_eval.V4_CHECKPOINT.resolve()),
            "checkpoint_sha256": sha256_file(adaptive_eval.V4_CHECKPOINT),
            "best_epoch": best_epoch,
            "config": config,
        },
        "adaptive_policy": {
            "candidate_weights": list(adaptive_eval.CANDIDATE_WEIGHTS),
            "cv_folds": adaptive_eval.CV_FOLDS,
            "cv_seed": adaptive_eval.CV_SEED,
            "safety_margin_fraction": adaptive_eval.SAFETY_MARGIN_FRACTION,
            "near_best_fraction": adaptive_eval.NEAR_BEST_FRACTION,
            "anchor_count": legacy.ANCHOR_COUNT,
            "anchor_seed": legacy.SEED,
            "heldout_exclusion_radius_px": 4,
            "selection_evidence": "calibration_anchors_only",
        },
        "source_hashes": {
            "frozen_holdout_evaluator": sha256_file(Path(__file__)),
            "adaptive_policy_core": sha256_file(
                ROOT / "src" / "depthwizard" / "evaluation" / "adaptive.py"
            ),
            "adaptive_development_evaluator": sha256_file(
                ROOT / "scripts" / "evaluate_ortholoc_adaptive_refinement.py"
            ),
        },
        "immutability_rule": (
            "Once this seal exists, any model, policy, evaluator, or scene-membership change must "
            "use a new protocol version. Do not delete the seal to retest a modified system."
        ),
    }


def seal_protocol(payload: dict[str, object]) -> str:
    """Atomically seal the protocol before holdout raster/reference content is loaded."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    digest = canonical_json_hash(payload)
    seal = {
        "state": "SEALED_BEFORE_HOLDOUT_TARGET_LOAD",
        "protocol_sha256": digest,
        "protocol": payload,
    }
    if SEAL_PATH.exists():
        existing = json.loads(SEAL_PATH.read_text(encoding="utf-8"))
        if not isinstance(existing, dict):
            raise TypeError("existing frozen holdout seal is not a JSON object")
        existing_digest = existing.get("protocol_sha256")
        if existing_digest != digest:
            raise RuntimeError(
                "frozen holdout seal mismatch. This holdout has already been committed to a "
                "different model/policy/evaluator/scene contract. Preserve the old seal and define "
                "a new protocol version instead of rerunning a modified system."
            )
        print(f"Reusing frozen protocol seal: {digest}")
        return digest

    temporary = SEAL_PATH.with_suffix(".json.part")
    temporary.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    temporary.replace(SEAL_PATH)
    print(f"Frozen protocol sealed before target load: {digest}")
    return digest


def _candidate_scores_json(scores: object) -> list[dict[str, object]]:
    if not isinstance(scores, tuple):
        raise TypeError("candidate scores must be a tuple")
    return [asdict(score) for score in scores]


def evaluate_operational_scene(
    model: DepthWizardHeightModel,
    scene: legacy.SceneData,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate frozen DA3/V4/adaptive paths without making model-only failure fatal."""
    refined, uncertainty, covered = legacy.predict_scene(model, scene, device)
    valid = scene.supervision_valid & covered & np.isfinite(refined)
    adaptive = adaptive_sparse_anchor_holdout_benchmark(
        scene.geometry,
        refined,
        scene.reference_m,
        valid_mask=valid,
        anchor_count=legacy.ANCHOR_COUNT,
        seed=legacy.SEED,
        exclusion_radius_px=4,
        candidate_weights=adaptive_eval.CANDIDATE_WEIGHTS,
        cv_folds=adaptive_eval.CV_FOLDS,
        cv_seed=adaptive_eval.CV_SEED,
        safety_margin_fraction=adaptive_eval.SAFETY_MARGIN_FRACTION,
        near_best_fraction=adaptive_eval.NEAR_BEST_FRACTION,
    )

    model_metrics: dict[str, object] | None = None
    model_rejection: str | None = None
    try:
        model_only = sparse_anchor_holdout_benchmark(
            refined,
            scene.reference_m,
            valid_mask=valid,
            anchor_count=legacy.ANCHOR_COUNT,
            seed=legacy.SEED,
            exclusion_radius_px=4,
        )
    except ValueError as exc:
        model_rejection = str(exc)
    else:
        if not np.array_equal(model_only.evaluation_mask, adaptive.baseline.evaluation_mask):
            raise RuntimeError("model-only and frozen adaptive evaluation masks diverged")
        model_metrics = model_only.metrics.model_dump()

    common = adaptive.baseline.evaluation_mask
    adaptive_abs_error = np.abs(adaptive.adaptive.prediction - scene.reference_m)
    reliability = legacy.uncertainty_error_correlation(uncertainty, adaptive_abs_error, common)
    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "location_id": scene.location_id,
        "gsd_m": scene.gsd_m,
        "heldout_pixels": int(common.sum()),
        "da3": adaptive.baseline.metrics.model_dump(),
        "v4_model_only": model_metrics,
        "v4_model_only_rejection_reason": model_rejection,
        "adaptive": adaptive.adaptive.metrics.model_dump(),
        "adaptive_rmse_delta_m": float(
            adaptive.adaptive.metrics.rmse_m - adaptive.baseline.metrics.rmse_m
        ),
        "selected_refinement_weight": adaptive.selection.selected_weight,
        "selection_evidence_valid": adaptive.selection.selection_evidence_valid,
        "fallback_reason": adaptive.selection.fallback_reason,
        "anchor_cv_baseline_rmse_m": adaptive.selection.baseline_cv_rmse_m,
        "anchor_cv_selected_rmse_m": adaptive.selection.selected_cv_rmse_m,
        "anchor_cv_relative_improvement": adaptive.selection.relative_cv_improvement,
        "candidate_scores": _candidate_scores_json(adaptive.selection.candidate_scores),
        "uncertainty_abs_error_pearson": reliability,
    }
    return (
        report,
        adaptive.baseline.prediction[common],
        adaptive.adaptive.prediction[common],
        scene.reference_m[common],
    )


def aggregate_holdout(
    scene_reports: list[dict[str, object]],
    baseline_values: list[np.ndarray],
    adaptive_values: list[np.ndarray],
    reference_values: list[np.ndarray],
    rejections: list[dict[str, str]],
) -> dict[str, object]:
    if not baseline_values or not adaptive_values or not reference_values:
        return {
            "aggregate_available": False,
            "scene_rejections": rejections,
            "primary_same_domain_pass": False,
        }

    baseline = np.concatenate(baseline_values)
    adaptive = np.concatenate(adaptive_values)
    reference = np.concatenate(reference_values)
    baseline_metrics = compute_elevation_metrics(baseline, reference)
    adaptive_metrics = compute_elevation_metrics(adaptive, reference)
    rmse_improvement = (baseline_metrics.rmse_m - adaptive_metrics.rmse_m) / baseline_metrics.rmse_m
    mae_improvement = (baseline_metrics.mae_m - adaptive_metrics.mae_m) / baseline_metrics.mae_m

    deltas: list[float] = []
    selected_weights: list[float] = []
    for scene_report in scene_reports:
        delta = scene_report.get("adaptive_rmse_delta_m")
        weight = scene_report.get("selected_refinement_weight")
        if isinstance(delta, (int, float)):
            deltas.append(float(delta))
        if isinstance(weight, (int, float)):
            selected_weights.append(float(weight))

    all_non_degrading = bool(deltas and max(deltas) <= 0.0)
    learned_refinement_used = any(weight > 0.0 for weight in selected_weights)
    expected_scenes = HOLDOUT_LOCATIONS * SAMPLES_PER_LOCATION
    complete = len(scene_reports) == expected_scenes and not rejections
    primary_pass = bool(
        complete
        and rmse_improvement > 0.0
        and mae_improvement > 0.0
        and all_non_degrading
        and learned_refinement_used
    )
    return {
        "aggregate_available": True,
        "aggregate_da3": baseline_metrics.model_dump(),
        "aggregate_adaptive": adaptive_metrics.model_dump(),
        "rmse_improvement_fraction": float(rmse_improvement),
        "mae_improvement_fraction": float(mae_improvement),
        "scene_rmse_deltas_m": deltas,
        "all_scenes_non_degrading": all_non_degrading,
        "learned_refinement_used_on_any_scene": learned_refinement_used,
        "evaluated_scene_count": len(scene_reports),
        "expected_scene_count": expected_scenes,
        "scene_rejections": rejections,
        "primary_same_domain_pass": primary_pass,
    }


def _nested_metric(mapping: dict[str, object], section: str, metric: str) -> float:
    section_value = mapping.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(f"report section {section!r} is not a dictionary")
    value = section_value.get(metric)
    if not isinstance(value, (int, float)):
        raise TypeError(f"report metric {section}.{metric} is not numeric")
    return float(value)


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(legacy.SEED)
    np.random.seed(legacy.SEED)
    torch.set_float32_matmul_precision("high")
    device = legacy.resolve_device()

    model, checkpoint = adaptive_eval._load_v4_model(device)
    discovered = discover_remote_scenes("test_outPlace")
    selected_remote = select_frozen_holdout_scenes(discovered)
    payload = build_protocol_payload(selected_remote, checkpoint)
    digest = seal_protocol(payload)

    selected_locations = sorted({scene.location_id for scene in selected_remote}, key=_location_sort_key)
    print(
        "Frozen untouched OrthoLoC holdout: "
        f"locations={selected_locations} | scenes={[scene.scene_id for scene in selected_remote]}"
    )
    print(
        "Integrity note: model, adaptive policy, evaluator hashes, and scene membership were sealed "
        "before any selected holdout DSM/reference was loaded. No target-dependent replacements "
        "are permitted."
    )

    prior = legacy.DA3MonocularPrior(device="auto")
    scenes = [
        legacy.load_scene(remote, "test_outPlace", prior, include_target=False)
        for remote in selected_remote
    ]
    del prior

    reports: list[dict[str, object]] = []
    baseline_values: list[np.ndarray] = []
    adaptive_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    rejections: list[dict[str, str]] = []

    for scene in scenes:
        try:
            scene_report, baseline, adaptive, reference = evaluate_operational_scene(
                model,
                scene,
                device,
            )
        except ValueError as exc:
            reason = str(exc)
            rejections.append({"scene_id": scene.scene_id, "reason": reason})
            print(f"frozen_holdout {scene.scene_id}: REJECTED | {reason}")
            continue

        reports.append(scene_report)
        baseline_values.append(baseline)
        adaptive_values.append(adaptive)
        reference_values.append(reference)
        da3_rmse = _nested_metric(scene_report, "da3", "rmse_m")
        adaptive_rmse = _nested_metric(scene_report, "adaptive", "rmse_m")
        weight_raw = scene_report.get("selected_refinement_weight")
        evidence_raw = scene_report.get("selection_evidence_valid")
        if not isinstance(weight_raw, (int, float)) or not isinstance(evidence_raw, bool):
            raise TypeError("frozen holdout scene report contains invalid policy diagnostics")
        print(
            f"frozen_holdout {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
            f"adaptive {adaptive_rmse:.3f} m | weight={float(weight_raw):.2f} | "
            f"selection-evidence={'valid' if evidence_raw else 'fallback'}"
        )

    aggregate = aggregate_holdout(
        reports,
        baseline_values,
        adaptive_values,
        reference_values,
        rejections,
    )
    primary_pass_raw = aggregate.get("primary_same_domain_pass")
    if not isinstance(primary_pass_raw, bool):
        raise TypeError("frozen holdout aggregate pass state is not boolean")

    report = {
        "status": (
            "PASS_FROZEN_SAME_DOMAIN_GEOGRAPHIC_HOLDOUT"
            if primary_pass_raw
            else "FAIL_FROZEN_SAME_DOMAIN_GEOGRAPHIC_HOLDOUT"
        ),
        "protocol_sha256": digest,
        "protocol_seal": str(SEAL_PATH.resolve()),
        "selected_scenes": [scene.scene_id for scene in selected_remote],
        "aggregate": aggregate,
        "scene_reports": reports,
        "scientific_scope": (
            "This is a one-time untouched same-domain geographic holdout within OrthoLoC, "
            "conditional on the declared 64 sparse height anchors. It is not zero-shot metric "
            "depth, not an independent-sensor reference, and does not by itself close Gate B."
        ),
        "post_result_rule": (
            "Do not tune V4/V5 or this protocol against these holdout outcomes. Any future model or "
            "policy change makes these scenes development evidence only and requires a new external "
            "untouched benchmark for final claims."
        ),
    }
    temporary_report = REPORT_PATH.with_suffix(".json.part")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(REPORT_PATH)

    print(
        "DepthWizard frozen same-domain geographic holdout: "
        f"{'PASS' if primary_pass_raw else 'FAIL'}"
    )
    if aggregate.get("aggregate_available") is True:
        da3_rmse = _nested_metric(aggregate, "aggregate_da3", "rmse_m")
        adaptive_rmse = _nested_metric(aggregate, "aggregate_adaptive", "rmse_m")
        rmse_gain = aggregate.get("rmse_improvement_fraction")
        mae_gain = aggregate.get("mae_improvement_fraction")
        non_degrading = aggregate.get("all_scenes_non_degrading")
        if not isinstance(rmse_gain, (int, float)) or not isinstance(mae_gain, (int, float)):
            raise TypeError("frozen holdout aggregate gains are not numeric")
        if not isinstance(non_degrading, bool):
            raise TypeError("frozen holdout non-degradation state is not boolean")
        print(
            f"Frozen aggregate: DA3 {da3_rmse:.3f} m | adaptive {adaptive_rmse:.3f} m | "
            f"RMSE improvement {100.0 * float(rmse_gain):.2f}% | "
            f"MAE improvement {100.0 * float(mae_gain):.2f}%"
        )
        print(f"All frozen scenes non-degrading: {'YES' if non_degrading else 'NO'}")
    print(f"Frozen scene rejections: {len(rejections)}")
    print(f"Protocol seal: {SEAL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
