"""Exact-head analytical qualification without modifying the frozen release checkout."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root))

    from depthwizard.calibration.gcp_io import inspect_ground_control_point_file
    from depthwizard.contracts import GroundControlPointEvidence, ProcessingRequest
    from depthwizard.pipeline.policy import EstimatorPath, current_production_estimator_decision
    from scripts import release_train_4_scientific_analytical_smoke as rt4

    rt4.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rt4._write_inputs()
    inspected = inspect_ground_control_point_file(rt4.GCP_PATH)
    require(inspected.point_count >= 6, "fewer than six distributed GCPs were generated")
    gcp_evidence = GroundControlPointEvidence(
        source_path=inspected.source_path,
        sha256=inspected.sha256,
    )

    dem_manifest, dem_calibration = rt4._run_project(
        "dem",
        ProcessingRequest(
            source=rt4.SOURCE_PATH,
            output_dir=rt4.OUTPUT_DIR / "dem-project",
            dem_path=rt4.DEM_PATH,
            requested_output="dsm",
        ),
    )
    require(dem_calibration.get("mode") == "dem", "DEM calibration mode changed")
    dem_error = rt4._metric_error(dem_manifest)
    dem_high_frequency = rt4._high_frequency_preservation(dem_manifest)
    require(bool(dem_high_frequency["preserved"]), "DEM path lost high-frequency structure")

    gcp_manifest, gcp_calibration = rt4._run_project(
        "gcp",
        ProcessingRequest(
            source=rt4.SOURCE_PATH,
            output_dir=rt4.OUTPUT_DIR / "gcp-project",
            gcps=inspected.points,
            gcp_evidence=gcp_evidence,
            requested_output="dsm",
        ),
    )
    require(gcp_calibration.get("mode") == "gcp", "GCP calibration mode changed")
    gcp_source = rt4._verified_gcp_source(gcp_calibration)
    require(gcp_source.get("identity_verified") is True, "GCP CSV identity was not verified")
    gcp_error = rt4._metric_error(gcp_manifest)

    fusion_manifest, fusion_calibration = rt4._run_project(
        "dem-gcp",
        ProcessingRequest(
            source=rt4.SOURCE_PATH,
            output_dir=rt4.OUTPUT_DIR / "dem-gcp-project",
            dem_path=rt4.DEM_PATH,
            gcps=inspected.points,
            gcp_evidence=gcp_evidence,
            requested_output="dsm",
        ),
    )
    require(fusion_calibration.get("mode") == "dem_gcp", "DEM plus GCP mode changed")
    fusion_evidence = fusion_calibration.get("evidence")
    require(isinstance(fusion_evidence, dict), "DEM plus GCP evidence is missing")
    require(
        fusion_evidence.get("fusion_method")
        == "dem_scale_then_gcp_robust_global_datum_offset",
        "DEM plus GCP method is not the frozen robust global datum-offset contract",
    )
    fusion_error = rt4._metric_error(fusion_manifest)

    weak_abort = rt4._weak_evidence_abort(inspected.points)
    mutation = rt4._mutation_rejection(inspected.points, inspected.sha256)
    structure = rt4._structure_report(rt4.OUTPUT_DIR / "gcp-project")
    previews = rt4._preview_report(rt4.OUTPUT_DIR / "gcp-project")
    policy = current_production_estimator_decision()
    require(
        policy.selected_path is EstimatorPath.CALIBRATED_DA3,
        "qualification detected an unapproved estimator promotion",
    )
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    gcp_report = {
        "schema_version": 2,
        "status": "PASS_ENGINEERING_CALIBRATION_GCP",
        "git_head": git_head,
        "mode": "gcp",
        "project_id": gcp_manifest.project_id,
        "gcp_file_sha256": inspected.sha256,
        "gcp_count": inspected.point_count,
        "metric_error_against_synthetic_truth": gcp_error,
        "weak_evidence_abort": weak_abort,
        "post_inspection_mutation_rejection": mutation,
        "calibration_evidence": gcp_calibration,
        "scientific_boundary": (
            "Deterministic calibration engineering qualification on the fresh runner; "
            "not independent accuracy evidence."
        ),
    }
    rt4._write_report("calibration_gcp_report.json", gcp_report)
    integrated = {
        "schema_version": 2,
        "status": "PASS_RT4_SCIENTIFIC_ANALYTICAL_ENGINEERING_PATH",
        "git_head": git_head,
        "production_estimator_policy": policy.as_dict(),
        "production_estimator_remains_calibrated_da3": True,
        "consumed_benchmark_rerun": False,
        "model_promotion_claim": False,
        "reference_data_used_for_calibration": False,
        "confidence_fabricated": False,
        "gcp_csv_identity_verified": True,
        "gcp_post_inspection_mutation_rejected": True,
        "weak_evidence_aborted": True,
        "dem_only": {
            "mode": "dem",
            "metric_error_against_synthetic_truth": dem_error,
            "high_frequency_preservation": dem_high_frequency,
        },
        "gcp_only": gcp_report,
        "dem_gcp": {
            "mode": "dem_gcp",
            "fusion_method": fusion_evidence.get("fusion_method"),
            "metric_error_against_synthetic_truth": fusion_error,
            "calibration_evidence": fusion_calibration,
        },
        "structural_height": structure,
        "derived_previews": previews,
        "scientific_boundary": (
            "Fresh-runner engineering closure for calibration, provenance and analyst tools. "
            "Independent DSM accuracy remains owned by the frozen science campaign."
        ),
    }
    output = rt4._write_report("release-train-4-analytical-acceptance.json", integrated)
    print(json.dumps({"status": integrated["status"], "report": str(output)}, indent=2))


if __name__ == "__main__":
    main()
