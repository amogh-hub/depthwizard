from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.structure_band import project_structure_correction_numpy
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v3 as v3
from scripts import train_ortholoc_structure_band_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-structure-band-v5"
    / "height_model_structure_band_v5.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "3fcc1423abaffdeedea5f56ef360866e8d72d32d36d6452d559e40ae070b8abf"
OUT_PATH = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-structure-band-v5"
    / "structure_band_v5_evidence_gate_diagnostic.json"
)
ANCHOR_RMSE_TOLERANCE_M = 1e-9


def _load_checkpoint(device: torch.device) -> DepthWizardHeightModel:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"missing frozen V5 checkpoint: {CHECKPOINT_PATH}")
    actual_sha = sha256_file(CHECKPOINT_PATH)
    if actual_sha != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "V5 checkpoint SHA mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha}"
        )
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    config = HeightModelConfig(**checkpoint["config"])
    model = DepthWizardHeightModel(config)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def _evaluate_scene(
    model: DepthWizardHeightModel,
    scene: legacy.SceneData,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    raw_refined, _uncertainty, covered = legacy.predict_scene(model, scene, device)
    candidate_valid = scene.input_valid & covered & np.isfinite(raw_refined)
    raw_correction = raw_refined - scene.geometry
    safe_correction = project_structure_correction_numpy(
        raw_correction,
        candidate_valid,
        gsd_m=scene.gsd_m,
        config=v5.STRUCTURE,
    )
    refined = np.full(scene.geometry.shape, np.nan, dtype=np.float32)
    refined_valid = candidate_valid & np.isfinite(safe_correction)
    refined[refined_valid] = scene.geometry[refined_valid] + safe_correction[refined_valid]
    evaluation_valid = scene.supervision_valid & refined_valid & np.isfinite(refined)

    baseline = sparse_anchor_holdout_benchmark(
        scene.geometry,
        scene.reference_m,
        valid_mask=evaluation_valid,
        anchor_count=legacy.ANCHOR_COUNT,
        seed=legacy.SEED,
        exclusion_radius_px=4,
    )

    refined_result = None
    refined_rejection = None
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
        refined_rejection = str(exc)

    if refined_result is None:
        common = baseline.evaluation_mask
        baseline_metrics = compute_elevation_metrics(
            baseline.prediction,
            scene.reference_m,
            valid_mask=common,
        )
        decision = "FALLBACK_BASE_GEOMETRY"
        reason = f"refined calibration rejected: {refined_rejection}"
        chosen_prediction = baseline.prediction
        chosen_metrics = baseline_metrics
        refined_metrics = None
        refined_anchor_rmse = None
        refined_scale = None
        refined_orientation_flipped = None
    else:
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
        refined_anchor_rmse = float(refined_result.calibration.rmse_anchor)
        refined_scale = float(refined_result.calibration.scale)
        refined_orientation_flipped = bool(refined_result.orientation_flipped)
        if refined_anchor_rmse <= baseline.calibration.rmse_anchor + ANCHOR_RMSE_TOLERANCE_M:
            decision = "ACCEPT_STRUCTURE_REFINEMENT"
            reason = "refined sparse-anchor calibration residual is non-worse than base geometry"
            chosen_prediction = refined_result.prediction
            chosen_metrics = refined_metrics
        else:
            decision = "FALLBACK_BASE_GEOMETRY"
            reason = (
                "refined sparse-anchor calibration residual worsened: "
                f"{refined_anchor_rmse:.6f} m > {baseline.calibration.rmse_anchor:.6f} m"
            )
            chosen_prediction = baseline.prediction
            chosen_metrics = baseline_metrics

    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "location_id": scene.location_id,
        "gsd_m": scene.gsd_m,
        "gate_uses_heldout_evaluation_pixels": False,
        "decision": decision,
        "decision_reason": reason,
        "base_anchor_rmse_m": float(baseline.calibration.rmse_anchor),
        "base_scale": float(baseline.calibration.scale),
        "base_orientation_flipped": bool(baseline.orientation_flipped),
        "refined_anchor_rmse_m": refined_anchor_rmse,
        "refined_scale": refined_scale,
        "refined_orientation_flipped": refined_orientation_flipped,
        "refined_calibration_rejection": refined_rejection,
        "heldout_pixels": int(common.sum()),
        "base_heldout": baseline_metrics.model_dump(),
        "refined_heldout": refined_metrics.model_dump() if refined_metrics is not None else None,
        "gated_heldout": chosen_metrics.model_dump(),
        "gated_rmse_delta_vs_base_m": float(chosen_metrics.rmse_m - baseline_metrics.rmse_m),
    }
    return (
        report,
        baseline.prediction[common],
        chosen_prediction[common],
        scene.reference_m[common],
    )


def _evaluate_split(
    name: str,
    model: DepthWizardHeightModel,
    scenes: list[legacy.SceneData],
    device: torch.device,
) -> dict[str, object]:
    reports: list[dict[str, object]] = []
    base_values: list[np.ndarray] = []
    gated_values: list[np.ndarray] = []
    reference_values: list[np.ndarray] = []
    scene_deltas: list[float] = []

    print(f"\n=== {name.upper()} ===")
    for scene in scenes:
        report, base, gated, reference = _evaluate_scene(model, scene, device)
        reports.append(report)
        base_values.append(base)
        gated_values.append(gated)
        reference_values.append(reference)
        delta = float(report["gated_rmse_delta_vs_base_m"])
        scene_deltas.append(delta)
        refined_status = (
            "REJECTED"
            if report["refined_heldout"] is None
            else f"RMSE {report['refined_heldout']['rmse_m']:.3f} m"
        )
        print(
            f"{scene.scene_id}: base {report['base_heldout']['rmse_m']:.3f} m | "
            f"refined {refined_status} | gate {report['decision']} | "
            f"gated {report['gated_heldout']['rmse_m']:.3f} m | delta {delta:+.3f} m"
        )
        print(
            f"  anchor RMSE base={report['base_anchor_rmse_m']:.6f} m | "
            f"refined={report['refined_anchor_rmse_m']} | {report['decision_reason']}"
        )

    base_metrics = compute_elevation_metrics(
        np.concatenate(base_values),
        np.concatenate(reference_values),
    )
    gated_metrics = compute_elevation_metrics(
        np.concatenate(gated_values),
        np.concatenate(reference_values),
    )
    improvement = (base_metrics.rmse_m - gated_metrics.rmse_m) / base_metrics.rmse_m
    all_non_degrading = bool(scene_deltas and max(scene_deltas) <= 1e-9)
    print(
        f"{name} aggregate: base {base_metrics.rmse_m:.3f} m | "
        f"gated {gated_metrics.rmse_m:.3f} m | improvement {100.0 * improvement:.2f}%"
    )
    print(f"{name} all scenes non-degrading: {'YES' if all_non_degrading else 'NO'}")
    return {
        "scenes": reports,
        "aggregate_base": base_metrics.model_dump(),
        "aggregate_gated": gated_metrics.model_dump(),
        "rmse_improvement_fraction": improvement,
        "all_scenes_non_degrading": all_non_degrading,
    }


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(legacy.SEED)
    np.random.seed(legacy.SEED)
    torch.set_float32_matmul_precision("high")
    device = legacy.resolve_device()

    model = _load_checkpoint(device)
    train_discovered = legacy.discover_remote_scenes("train")
    outplace_discovered = legacy.discover_remote_scenes("test_outPlace")
    _train_remote, validation_remote, development_remote = v3.select_diversity_split(
        train_discovered,
        outplace_discovered,
    )

    prior = legacy.DA3MonocularPrior(device="auto")
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
        "Evidence gate: structure refinement is accepted only when sparse calibration remains "
        "physically valid and its anchor RMSE is non-worse than the unchanged base geometry."
    )
    print(
        "Gate selection uses calibration anchors only; held-out evaluation pixels are never used "
        "to accept or reject the learned correction."
    )

    validation = _evaluate_split("validation", model, validation_scenes, device)
    development = _evaluate_split("development", model, development_scenes, device)
    diagnostic_pass = bool(
        validation["rmse_improvement_fraction"] > 0.0
        and development["rmse_improvement_fraction"] >= 0.0
        and development["all_scenes_non_degrading"]
    )

    payload = {
        "status": "PASS_V5_EVIDENCE_GATE_DIAGNOSTIC" if diagnostic_pass else "REJECT_V5_EVIDENCE_GATE_DIAGNOSTIC",
        "research_only": True,
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "policy": {
            "refined_must_calibrate": True,
            "refined_anchor_rmse_must_be_non_worse_than_base": True,
            "anchor_rmse_tolerance_m": ANCHOR_RMSE_TOLERANCE_M,
            "heldout_pixels_used_for_gate_decision": False,
        },
        "validation": validation,
        "development": development,
        "potsdam_2_14_used": False,
        "blind_potsdam_4_12_6_12_touched": False,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nEvidence-gate diagnostic: {payload['status']}")
    print(f"Report: {OUT_PATH}")


if __name__ == "__main__":
    main()
