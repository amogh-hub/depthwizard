#!/usr/bin/env python3
"""Stage the frozen Potsdam 2_14 science output as an operator-workstation project.

This qualification-only helper DOES NOT rerun DA3, calibration, or validation. It verifies the
already-qualified final-science prediction and calibration evidence identities, copies only the
metric prediction into a normal DepthWizard project-owned artifact location, and records the
calibration DEM SHA in project provenance so production reference validation can enforce exact-file
independence. The official Potsdam reference DSM remains outside the project and is not consumed by
this helper; the analyst must select it later through the packaged UI's Validate reference action.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
from pathlib import Path

from depthwizard.contracts import ProjectRunStatus
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file

QUALIFIED_HEAD = "188c5111e73a8f0cb3b5022233b22fdebdb19d2a"
EXPECTED_PREDICTION_SHA = "116bb397c813bff2afdfcae1747503353d01a27364c7e6db947d7574299730ef"
EXPECTED_CALIBRATION_DEM_SHA = "96e3c9de4049cff5d60f5e927c6d574f8cafcfde6164fa427a3ceab9fb11126d"
MODEL_ID = "DA3MONO-LARGE"


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git_head(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare frozen Potsdam 2_14 operator project")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/operator-urban-potsdam-2-14"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve(strict=True)
    head = _git_head(repo)
    if head != QUALIFIED_HEAD:
        raise SystemExit(f"wrong git head: expected {QUALIFIED_HEAD}, got {head}")
    if subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise SystemExit("worktree is not clean; operator project staging requires the clean final corrective checkout")

    source = repo / "data/external/isprs-potsdam/2_Ortho_RGB/top_potsdam_2_14_RGB.tif"
    prediction = repo / "workspace/final-science-data/predictions/potsdam-2-14-urban-test/metric-dsm.tif"
    calibration_dem = repo / "workspace/final-science-data/predictions/potsdam-2-14-urban-test/calibration/copernicus-glo30-mosaic.tif"
    reference = repo / "data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif"

    for label, path in (
        ("Potsdam RGB", source),
        ("frozen metric prediction", prediction),
        ("frozen Copernicus calibration DEM", calibration_dem),
        ("official Potsdam reference DSM", reference),
    ):
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")

    prediction_sha = sha256_file(prediction)
    if prediction_sha != EXPECTED_PREDICTION_SHA:
        raise SystemExit(
            f"frozen prediction identity mismatch: expected {EXPECTED_PREDICTION_SHA}, got {prediction_sha}"
        )
    calibration_sha = sha256_file(calibration_dem)
    if calibration_sha != EXPECTED_CALIBRATION_DEM_SHA:
        raise SystemExit(
            "frozen calibration DEM identity mismatch: "
            f"expected {EXPECTED_CALIBRATION_DEM_SHA}, got {calibration_sha}"
        )

    project_dir = args.output
    if not project_dir.is_absolute():
        project_dir = repo / project_dir
    project_dir = project_dir.resolve(strict=False)
    if project_dir.exists():
        shutil.rmtree(project_dir)
    products = project_dir / "products"
    products.mkdir(parents=True)
    dsm_path = products / "dsm.tif"
    shutil.copy2(prediction, dsm_path)

    source_sha = sha256_file(source)
    manifest = ProjectManifest.create_or_load(project_dir, source)
    manifest.set_identity(
        source_sha256=source_sha,
        input_kind="georeferenced",
        geometry_config_sha256=_label_sha(
            f"operator-stage:{QUALIFIED_HEAD}:{MODEL_ID}:potsdam-2-14:geometry"
        ),
        run_config_sha256=_label_sha(
            f"operator-stage:{QUALIFIED_HEAD}:{MODEL_ID}:potsdam-2-14:metric-final-science"
        ),
    )
    manifest.set_estimator(
        {
            "selected_model_id": MODEL_ID,
            "qualification_staged_from_final_science": True,
            "qualified_source_git_head": QUALIFIED_HEAD,
            "frozen_prediction_sha256": EXPECTED_PREDICTION_SHA,
        }
    )
    manifest.register_artifact(
        "dsm",
        dsm_path,
        semantics="metric_dsm_from_frozen_final_science_prediction",
        units="m",
        sha256=sha256_file(dsm_path),
    )
    manifest.record_stage(
        ProcessingStage.INGEST,
        status="completed",
        details={
            "source_path": str(source),
            "source_sha256": source_sha,
            "operator_qualification_stage": True,
        },
    )
    manifest.record_stage(
        ProcessingStage.GEOMETRY,
        status="completed",
        details={
            "model_id": MODEL_ID,
            "source_prediction_sha256": EXPECTED_PREDICTION_SHA,
            "staged_from_final_science": True,
        },
    )
    manifest.record_stage(
        ProcessingStage.CALIBRATION,
        status="completed",
        artifacts={"dsm": str(dsm_path)},
        details={
            "method": "robust_affine_huber_irls",
            "staged_from_frozen_final_science": True,
            "evidence": {
                "dem": {
                    "path": str(calibration_dem),
                    "sha256": calibration_sha,
                    "role": "independent_coarse_calibration_evidence",
                }
            },
        },
    )
    manifest.record_stage(
        ProcessingStage.COMPLETE,
        status="completed",
        details={
            "operator_qualification_stage": True,
            "reference_not_consumed_by_staging_helper": True,
            "reference_path_for_manual_ui_selection": str(reference),
        },
    )
    manifest.mark_status(ProjectRunStatus.COMPLETE)

    ProjectManifest.load(project_dir)

    print("PASS_OPERATOR_URBAN_PROJECT_STAGED")
    print(f"git_head={head}")
    print(f"project_dir={project_dir}")
    print(f"source={source}")
    print(f"prediction_sha256={prediction_sha}")
    print(f"calibration_dem_sha256={calibration_sha}")
    print(f"reference_for_manual_validation={reference}")
    print("reference_consumed_by_staging_helper=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
