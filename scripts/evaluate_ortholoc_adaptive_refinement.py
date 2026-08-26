from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from depthwizard.data.ortholoc import discover_remote_scenes
from depthwizard.evaluation.adaptive import adaptive_sparse_anchor_holdout_benchmark
from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.provenance.manifest import sha256_file
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_multiscene_v3 as v3
from scripts import train_ortholoc_multiscene_v4 as v4

ROOT = Path(__file__).resolve().parents[1]
V4_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-multiscene-v4"
    / "height_model_multiscene_v4.pt"
)
OUT_DIR = ROOT / "artifacts" / "evaluation" / "ortholoc-adaptive-refinement-v5"
REPORT_PATH = OUT_DIR / "adaptive_refinement_report.json"
CANDIDATE_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
CV_FOLDS = 4
CV_SEED = 104729
SAFETY_MARGIN_FRACTION = 0.01
NEAR_BEST_FRACTION = 0.005


def _load_v4_model(device: torch.device) -> tuple[DepthWizardHeightModel, dict[str, Any]]:
    if not V4_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "V4 checkpoint is missing. Run `make height-multiscene-v4` first so the adaptive "
            f"acceptance can evaluate the frozen learned model: {V4_CHECKPOINT}"
        )
    checkpoint = torch.load(V4_CHECKPOINT, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("V4 checkpoint payload is not a dictionary")
    config_payload = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise RuntimeError("V4 checkpoint is missing config/state_dict")
    config = HeightModelConfig(**config_payload)
    if config.architecture_version != "confidence-gated-v2":
        raise RuntimeError(
            "adaptive acceptance requires the confidence-gated-v2 V4 checkpoint; "
            f"found {config.architecture_version}"
        )
    best_epoch = checkpoint.get("best_epoch")
    if not isinstance(best_epoch, int) or best_epoch <= 0:
        raise RuntimeError("V4 checkpoint did not contain a promoted learned validation epoch")

    model = DepthWizardHeightModel(config)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def _candidate_scores_json(scores: object) -> list[dict[str, object]]:
    if not isinstance(scores, tuple):
        raise TypeError("candidate scores must be a tuple")
    return [asdict(score) for score in scores]


def _evaluate_scene(
    model: DepthWizardHeightModel,
    scene: legacy.SceneData,
    device: torch.device,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
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
        candidate_weights=CANDIDATE_WEIGHTS,
        cv_folds=CV_FOLDS,
        cv_seed=CV_SEED,
        safety_margin_fraction=SAFETY_MARGIN_FRACTION,
        near_best_fraction=NEAR_BEST_FRACTION,
    )

    model_only = sparse_anchor_holdout_benchmark(
        refined,
        scene.reference_m,
        valid_mask=valid,
        anchor_count=legacy.ANCHOR_COUNT,
        seed=legacy.SEED,
        exclusion_radius_px=4,
    )
    if not np.array_equal(model_only.evaluation_mask, adaptive.baseline.evaluation_mask):
        raise RuntimeError("model-only and adaptive evaluation masks diverged")

    common = adaptive.baseline.evaluation_mask
    baseline_metrics = compute_elevation_metrics(
        adaptive.baseline.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    model_metrics = compute_elevation_metrics(
        model_only.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    adaptive_metrics = compute_elevation_metrics(
        adaptive.adaptive.prediction,
        scene.reference_m,
        valid_mask=common,
    )
    adaptive_abs_error = np.abs(adaptive.adaptive.prediction - scene.reference_m)
    reliability = legacy.uncertainty_error_correlation(uncertainty, adaptive_abs_error, common)

    report: dict[str, object] = {
        "scene_id": scene.scene_id,
        "location_id": scene.location_id,
        "gsd_m": scene.gsd_m,
        "heldout_pixels": int(common.sum()),
        "da3": baseline_metrics.model_dump(),
        "v4_model_only": model_metrics.model_dump(),
        "adaptive": adaptive_metrics.model_dump(),
        "v4_model_only_rmse_delta_m": float(model_metrics.rmse_m - baseline_metrics.rmse_m),
        "adaptive_rmse_delta_m": float(adaptive_metrics.rmse_m - baseline_metrics.rmse_m),
        "selected_refinement_weight": adaptive.selection.selected_weight,
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


def _aggregate(
    reports: list[dict[str, object]],
    baseline_values: list[np.ndarray],
    adaptive_values: list[np.ndarray],
    reference_values: list[np.ndarray],
) -> dict[str, object]:
    baseline = np.concatenate(baseline_values)
    adaptive = np.concatenate(adaptive_values)
    reference = np.concatenate(reference_values)
    baseline_metrics = compute_elevation_metrics(baseline, reference)
    adaptive_metrics = compute_elevation_metrics(adaptive, reference)
    improvement = (baseline_metrics.rmse_m - adaptive_metrics.rmse_m) / baseline_metrics.rmse_m
    scene_deltas = [
        float(report["adaptive_rmse_delta_m"])
        for report in reports
        if isinstance(report.get("adaptive_rmse_delta_m"), (int, float))
    ]
    return {
        "scenes": reports,
        "aggregate_da3": baseline_metrics.model_dump(),
        "aggregate_adaptive": adaptive_metrics.model_dump(),
        "rmse_delta_m": float(adaptive_metrics.rmse_m - baseline_metrics.rmse_m),
        "rmse_improvement_fraction": float(improvement),
        "scene_rmse_deltas_m": scene_deltas,
        "all_scenes_non_degrading": bool(scene_deltas and max(scene_deltas) <= 0.0),
    }


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE", "1")
    torch.manual_seed(legacy.SEED)
    np.random.seed(legacy.SEED)
    torch.set_float32_matmul_precision("high")
    device = legacy.resolve_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model, checkpoint = _load_v4_model(device)
    train_discovered = discover_remote_scenes("train")
    outplace_discovered = discover_remote_scenes("test_outPlace")
    _, validation_remote, development_remote = v3.select_diversity_split(
        train_discovered,
        outplace_discovered,
    )

    print(
        "Evidence-adaptive protocol: frozen V4 checkpoint + fixed V3/V4 validation/development "
        "geography + anchor-only nested CV blend selection"
    )
    print(
        f"Policy: weights={CANDIDATE_WEIGHTS} | folds={CV_FOLDS} | "
        f"minimum anchor-CV gain={100.0 * SAFETY_MARGIN_FRACTION:.1f}% | "
        f"near-best conservative band={100.0 * NEAR_BEST_FRACTION:.1f}%"
    )
    print(
        "Scientific note: policy selection uses only the declared 64 sparse calibration anchors; "
        "raster held-out pixels are never used to select the blend. L08/L50 remain development "
        "evidence only, not final untouched Gate B evidence."
    )

    prior = legacy.DA3MonocularPrior(device="auto")
    validation_scenes = [
        legacy.load_scene(scene, "validation", prior, include_target=False)
        for scene in validation_remote
    ]
    development_scenes = [
        legacy.load_scene(scene, "test_outPlace", prior, include_target=False)
        for scene in development_remote
    ]
    del prior

    role_outputs: dict[str, dict[str, object]] = {}
    for role, scenes in (
        ("fixed_validation", validation_scenes),
        ("development_outPlace", development_scenes),
    ):
        reports: list[dict[str, object]] = []
        baseline_values: list[np.ndarray] = []
        adaptive_values: list[np.ndarray] = []
        reference_values: list[np.ndarray] = []
        for scene in scenes:
            scene_report, baseline, adaptive, reference = _evaluate_scene(model, scene, device)
            reports.append(scene_report)
            baseline_values.append(baseline)
            adaptive_values.append(adaptive)
            reference_values.append(reference)
            da3_rmse = float(scene_report["da3"]["rmse_m"])  # type: ignore[index]
            model_rmse = float(scene_report["v4_model_only"]["rmse_m"])  # type: ignore[index]
            adaptive_rmse = float(scene_report["adaptive"]["rmse_m"])  # type: ignore[index]
            selected = float(scene_report["selected_refinement_weight"])
            cv_gain = 100.0 * float(scene_report["anchor_cv_relative_improvement"])
            print(
                f"{role} {scene.scene_id}: DA3 {da3_rmse:.3f} m | "
                f"V4 {model_rmse:.3f} m | adaptive {adaptive_rmse:.3f} m | "
                f"weight={selected:.2f} | anchor-CV gain={cv_gain:.2f}%"
            )
        role_outputs[role] = _aggregate(
            reports,
            baseline_values,
            adaptive_values,
            reference_values,
        )

    validation = role_outputs["fixed_validation"]
    development = role_outputs["development_outPlace"]
    validation_improvement = float(validation["rmse_improvement_fraction"])
    development_improvement = float(development["rmse_improvement_fraction"])
    development_non_degrading = bool(development["all_scenes_non_degrading"])
    operationally_promoted = bool(
        validation_improvement > 0.0
        and development_improvement > 0.0
        and development_non_degrading
    )

    report = {
        "status": "PASS_ADAPTIVE_REFINEMENT_PIPELINE",
        "operationally_promoted_on_development": operationally_promoted,
        "purpose": (
            "Evaluate calibration-evidence adaptive trust of the frozen V4 learned residual. "
            "The policy uses nested cross-validation inside the declared sparse anchor set and "
            "never observes raster held-out evaluation pixels during blend selection."
        ),
        "model_checkpoint": str(V4_CHECKPOINT.resolve()),
        "model_checkpoint_sha256": sha256_file(V4_CHECKPOINT),
        "model_best_epoch": checkpoint.get("best_epoch"),
        "model_config": checkpoint.get("config"),
        "policy": {
            "candidate_weights": list(CANDIDATE_WEIGHTS),
            "cv_folds": CV_FOLDS,
            "cv_seed": CV_SEED,
            "safety_margin_fraction": SAFETY_MARGIN_FRACTION,
            "near_best_fraction": NEAR_BEST_FRACTION,
            "anchor_count": legacy.ANCHOR_COUNT,
            "anchor_seed": legacy.SEED,
            "heldout_exclusion_radius_px": 4,
            "selection_evidence": "calibration_anchors_only",
        },
        "fixed_validation": validation,
        "development_outPlace": development,
        "scientific_limit": (
            "L08/L50 and the fixed L06/L07 validation locations have influenced development. "
            "Final Gate B claims require a new untouched geographic/cross-sensor reference set."
        ),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    validation_da3 = float(validation["aggregate_da3"]["rmse_m"])  # type: ignore[index]
    validation_adaptive = float(validation["aggregate_adaptive"]["rmse_m"])  # type: ignore[index]
    development_da3 = float(development["aggregate_da3"]["rmse_m"])  # type: ignore[index]
    development_adaptive = float(development["aggregate_adaptive"]["rmse_m"])  # type: ignore[index]
    print("DepthWizard evidence-adaptive refinement acceptance: PASS")
    print(
        f"Fixed validation aggregate: DA3 {validation_da3:.3f} m | "
        f"adaptive {validation_adaptive:.3f} m | improvement {100.0 * validation_improvement:.2f}%"
    )
    print(
        f"Development aggregate: DA3 {development_da3:.3f} m | "
        f"adaptive {development_adaptive:.3f} m | improvement {100.0 * development_improvement:.2f}%"
    )
    print(f"All development scenes non-degrading: {'YES' if development_non_degrading else 'NO'}")
    print(f"Operational development promotion: {'YES' if operationally_promoted else 'NO'}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
