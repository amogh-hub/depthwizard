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
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.provenance.manifest import canonical_json_hash, sha256_file
from scripts import evaluate_ortholoc_adaptive_refinement as adaptive_eval
from scripts import evaluate_ortholoc_frozen_holdout as frozen_v1
from scripts import train_ortholoc_multiscene as legacy

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "ortholoc-frozen-outplace-location-v2"
OUT_DIR = ROOT / "artifacts" / "evaluation" / PROTOCOL_VERSION
SEAL_PATH = OUT_DIR / "protocol_seal.json"
REPORT_PATH = OUT_DIR / "frozen_location_holdout_report.json"
SAMPLES_PER_LOCATION = 2
EXCLUDED_LOCATIONS = frozen_v1.EXCLUDED_LOCATIONS


def _location_sort_key(location_id: str) -> tuple[int, str]:
    return frozen_v1._location_sort_key(location_id)


def select_frozen_location_scenes(
    discovered: list[OrthoLoCRemoteScene],
) -> list[OrthoLoCRemoteScene]:
    """Select the remaining untouched same-domain out-of-place location using metadata only.

    V1 required four untouched test-outPlace locations with two same-domain scenes each. Dataset
    listing metadata showed that this cardinality is unavailable after excluding every geography
    already used by V3/V4/V5 development. V2 therefore makes the strongest remaining claim that
    the source split can support without recycling an inspected geography: one untouched location,
    two deterministic same-domain scenes.

    Selection uses filenames, split membership, and location IDs only. DSM/reference values are not
    loaded or inspected here. No target-dependent replacement is allowed after the protocol seal.
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
    if not eligible_locations:
        raise RuntimeError(
            "frozen location holdout selection stopped before loading any DSM targets: no "
            f"untouched same-domain test_outPlace location provides {SAMPLES_PER_LOCATION} scenes"
        )

    location = eligible_locations[0]
    selected = grouped[location][:SAMPLES_PER_LOCATION]
    if len(selected) != SAMPLES_PER_LOCATION:
        raise RuntimeError("frozen location selector violated the declared scene-count invariant")
    if any(scene.location_id != location for scene in selected):
        raise RuntimeError("frozen location selector mixed geographic groups")
    if location in EXCLUDED_LOCATIONS:
        raise RuntimeError("frozen location selector leaked a previously inspected location")
    return selected


def _checkpoint_contract(checkpoint: dict[str, Any]) -> tuple[int, dict[str, object]]:
    best_epoch = checkpoint.get("best_epoch")
    config = checkpoint.get("config")
    if not isinstance(best_epoch, int) or best_epoch <= 0:
        raise TypeError("frozen location holdout requires a learned V4 checkpoint")
    if not isinstance(config, dict):
        raise TypeError("frozen location holdout checkpoint config is not a dictionary")
    return best_epoch, config


def build_protocol_payload(
    selected: list[OrthoLoCRemoteScene],
    checkpoint: dict[str, Any],
) -> dict[str, object]:
    """Build the complete pre-target contract hashed into the V2 protocol seal."""
    if len(selected) != SAMPLES_PER_LOCATION:
        raise ValueError("protocol payload requires exactly the declared number of scenes")
    location_ids = {scene.location_id for scene in selected}
    if len(location_ids) != 1:
        raise ValueError("protocol payload requires exactly one geographic location")
    best_epoch, config = _checkpoint_contract(checkpoint)
    return {
        "protocol_version": PROTOCOL_VERSION,
        "purpose": (
            "One-time untouched same-domain out-of-place location holdout for the frozen V4 model "
            "and V5 calibration-evidence adaptive policy."
        ),
        "selection": {
            "source_split": "test_outPlace",
            "same_domain_only": True,
            "excluded_locations": sorted(EXCLUDED_LOCATIONS, key=_location_sort_key),
            "holdout_locations": 1,
            "samples_per_location": SAMPLES_PER_LOCATION,
            "selection_rule": (
                "first numeric untouched location exposing at least two same-domain scenes; first "
                "two filenames in that location"
            ),
            "selected_scene_ids": [scene.scene_id for scene in selected],
            "selected_location_ids": sorted(location_ids, key=_location_sort_key),
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
            "frozen_location_evaluator": sha256_file(Path(__file__)),
            "shared_frozen_scene_evaluator": sha256_file(
                ROOT / "scripts" / "evaluate_ortholoc_frozen_holdout.py"
            ),
            "adaptive_policy_core": sha256_file(
                ROOT / "src" / "depthwizard" / "evaluation" / "adaptive.py"
            ),
            "adaptive_development_evaluator": sha256_file(
                ROOT / "scripts" / "evaluate_ortholoc_adaptive_refinement.py"
            ),
        },
        "scope_limit": (
            "This protocol is a single untouched geographic location because the remaining "
            "same-domain test_outPlace metadata cannot support V1's four-location requirement. "
            "It is confirmatory evidence only and cannot close Gate B geographic or cross-sensor "
            "generalization."
        ),
        "immutability_rule": (
            "Once this seal exists, any model, policy, evaluator, or scene-membership change must "
            "use a new protocol version. Do not delete the seal to retest a modified system."
        ),
    }


def seal_protocol(payload: dict[str, object]) -> str:
    """Atomically seal V2 before any selected holdout target is loaded."""
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
            raise TypeError("existing frozen location seal is not a JSON object")
        if existing.get("protocol_sha256") != digest:
            raise RuntimeError(
                "frozen location seal mismatch. Preserve the existing seal and define a new "
                "protocol version instead of rerunning a modified system."
            )
        print(f"Reusing frozen location protocol seal: {digest}")
        return digest

    temporary = SEAL_PATH.with_suffix(".json.part")
    temporary.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    temporary.replace(SEAL_PATH)
    print(f"Frozen location protocol sealed before target load: {digest}")
    return digest


def _relative_improvement(baseline: float, candidate: float) -> float:
    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("error metrics must be non-negative")
    if baseline == 0.0:
        return 0.0
    return float((baseline - candidate) / baseline)


def aggregate_location_holdout(
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
            "frozen_location_pass": False,
        }

    baseline = np.concatenate(baseline_values)
    adaptive = np.concatenate(adaptive_values)
    reference = np.concatenate(reference_values)
    baseline_metrics = compute_elevation_metrics(baseline, reference)
    adaptive_metrics = compute_elevation_metrics(adaptive, reference)
    rmse_improvement = _relative_improvement(
        baseline_metrics.rmse_m,
        adaptive_metrics.rmse_m,
    )
    mae_improvement = _relative_improvement(
        baseline_metrics.mae_m,
        adaptive_metrics.mae_m,
    )

    deltas: list[float] = []
    weights: list[float] = []
    for scene_report in scene_reports:
        delta = scene_report.get("adaptive_rmse_delta_m")
        weight = scene_report.get("selected_refinement_weight")
        if isinstance(delta, (int, float)):
            deltas.append(float(delta))
        if isinstance(weight, (int, float)):
            weights.append(float(weight))

    complete = len(scene_reports) == SAMPLES_PER_LOCATION and not rejections
    all_non_degrading = len(deltas) == SAMPLES_PER_LOCATION and max(deltas) <= 0.0
    learned_refinement_used = any(weight > 0.0 for weight in weights)
    frozen_pass = bool(
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
        "rmse_improvement_fraction": rmse_improvement,
        "mae_improvement_fraction": mae_improvement,
        "scene_rmse_deltas_m": deltas,
        "all_scenes_non_degrading": all_non_degrading,
        "learned_refinement_used_on_any_scene": learned_refinement_used,
        "evaluated_scene_count": len(scene_reports),
        "expected_scene_count": SAMPLES_PER_LOCATION,
        "scene_rejections": rejections,
        "frozen_location_pass": frozen_pass,
    }


def _nested_metric(mapping: dict[str, object], section: str, metric: str) -> float:
    section_value = mapping.get(section)
    if not isinstance(section_value, dict):
        raise TypeError(f"report section {section!r} is not a dictionary")
    value = section_value.get(metric)
    if not isinstance(value, (int, float)):
        raise TypeError(f"report metric {section}.{metric} is not numeric")
    return float(value)


def _candidate_scores_json(scores: object) -> list[dict[str, object]]:
    if not isinstance(scores, tuple):
        raise TypeError("candidate scores must be a tuple")
    return [asdict(score) for score in scores]


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(legacy.SEED)
    np.random.seed(legacy.SEED)
    torch.set_float32_matmul_precision("high")
    device = legacy.resolve_device()

    model, checkpoint = adaptive_eval._load_v4_model(device)
    discovered = discover_remote_scenes("test_outPlace")
    selected_remote = select_frozen_location_scenes(discovered)
    payload = build_protocol_payload(selected_remote, checkpoint)
    digest = seal_protocol(payload)

    location = selected_remote[0].location_id
    print(
        "Frozen untouched OrthoLoC out-place location: "
        f"location={location} | scenes={[scene.scene_id for scene in selected_remote]}"
    )
    print(
        "Integrity note: scene membership, model/checkpoint, adaptive policy, and evaluator hashes "
        "were sealed before any selected DSM/reference was loaded. V1's infeasible four-location "
        "requirement was not weakened after observing target values; no V1 target was loaded."
    )

    reports: list[dict[str, object]] = []
    baseline_values: list[np.ndarray] = []
    adaptive_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    rejections: list[dict[str, str]] = []

    prior = legacy.DA3MonocularPrior(device="auto")
    try:
        for remote in selected_remote:
            try:
                scene = legacy.load_scene(remote, "test_outPlace", prior, include_target=False)
                scene_report, baseline, adaptive, reference = frozen_v1.evaluate_operational_scene(
                    model,
                    scene,
                    device,
                )
            except ValueError as exc:
                reason = str(exc)
                rejections.append({"scene_id": remote.scene_id, "reason": reason})
                print(f"frozen_location {remote.scene_id}: REJECTED | {reason}")
                continue

            # Preserve complete adaptive policy diagnostics in this independent report.
            if "candidate_scores" not in scene_report:
                raise RuntimeError("shared frozen evaluator omitted adaptive candidate diagnostics")
            candidate_scores = scene_report.get("candidate_scores")
            if not isinstance(candidate_scores, list):
                # Historical helper currently serializes candidate scores as a list. Fail rather
                # than silently accepting a schema drift that would weaken report reproducibility.
                raise TypeError("shared frozen evaluator candidate_scores schema changed")

            reports.append(scene_report)
            baseline_values.append(baseline)
            adaptive_values.append(adaptive)
            reference_values.append(reference)
            da3_rmse = _nested_metric(scene_report, "da3", "rmse_m")
            adaptive_rmse = _nested_metric(scene_report, "adaptive", "rmse_m")
            weight_raw = scene_report.get("selected_refinement_weight")
            evidence_raw = scene_report.get("selection_evidence_valid")
            if not isinstance(weight_raw, (int, float)) or not isinstance(evidence_raw, bool):
                raise TypeError("frozen location scene report contains invalid policy diagnostics")
            print(
                f"frozen_location {remote.scene_id}: DA3 {da3_rmse:.3f} m | "
                f"adaptive {adaptive_rmse:.3f} m | weight={float(weight_raw):.2f} | "
                f"selection-evidence={'valid' if evidence_raw else 'fallback'}"
            )
    finally:
        del prior

    aggregate = aggregate_location_holdout(
        reports,
        baseline_values,
        adaptive_values,
        reference_values,
        rejections,
    )
    pass_raw = aggregate.get("frozen_location_pass")
    if not isinstance(pass_raw, bool):
        raise TypeError("frozen location aggregate pass state is not boolean")

    report = {
        "status": (
            "PASS_FROZEN_OUTPLACE_LOCATION_HOLDOUT"
            if pass_raw
            else "FAIL_FROZEN_OUTPLACE_LOCATION_HOLDOUT"
        ),
        "protocol_sha256": digest,
        "protocol_seal": str(SEAL_PATH.resolve()),
        "selected_location": location,
        "selected_scenes": [scene.scene_id for scene in selected_remote],
        "aggregate": aggregate,
        "scene_reports": reports,
        "scientific_scope": (
            "This is a one-time untouched same-domain test_outPlace location holdout conditional "
            "on 64 sparse calibration anchors. It confirms behavior on one previously unseen "
            "OrthoLoC geography only. It is not a multi-geography Gate B result, not zero-shot "
            "metric depth, and not an independent cross-sensor/reference benchmark."
        ),
        "post_result_rule": (
            "Do not tune V4/V5 against this location after the result is observed. Any future "
            "model/policy change makes these scenes development evidence and requires a new "
            "external untouched benchmark for scientific claims."
        ),
    }
    temporary = REPORT_PATH.with_suffix(".json.part")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(REPORT_PATH)

    print(f"DepthWizard frozen out-place location holdout: {'PASS' if pass_raw else 'FAIL'}")
    if aggregate.get("aggregate_available") is True:
        da3_rmse = _nested_metric(aggregate, "aggregate_da3", "rmse_m")
        adaptive_rmse = _nested_metric(aggregate, "aggregate_adaptive", "rmse_m")
        rmse_gain = aggregate.get("rmse_improvement_fraction")
        mae_gain = aggregate.get("mae_improvement_fraction")
        non_degrading = aggregate.get("all_scenes_non_degrading")
        learned_used = aggregate.get("learned_refinement_used_on_any_scene")
        if not isinstance(rmse_gain, (int, float)) or not isinstance(mae_gain, (int, float)):
            raise TypeError("frozen location aggregate gains are not numeric")
        if not isinstance(non_degrading, bool) or not isinstance(learned_used, bool):
            raise TypeError("frozen location aggregate boolean diagnostics are invalid")
        print(
            f"Frozen location aggregate: DA3 {da3_rmse:.3f} m | adaptive {adaptive_rmse:.3f} m | "
            f"RMSE improvement {100.0 * float(rmse_gain):.2f}% | "
            f"MAE improvement {100.0 * float(mae_gain):.2f}%"
        )
        print(f"All frozen location scenes non-degrading: {'YES' if non_degrading else 'NO'}")
        print(f"Learned refinement used on frozen location: {'YES' if learned_used else 'NO'}")
    print(f"Frozen location scene rejections: {len(rejections)}")
    print(f"Protocol seal: {SEAL_PATH}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
