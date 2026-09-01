from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.contracts import ReferenceValidationRequest
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file

EXPOSED_TILE_ID = "2_14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete downstream diagnostic for the already-exposed Potsdam 2_14 operator "
            "scene: project reference validation, deterministic official semantic-mask preparation, "
            "and per-building height evaluation. Sealed blind tiles are not accepted."
        )
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--semantic-label", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ground-policy", choices=("strict", "expanded"), default="strict")
    return parser.parse_args()


def _run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def _contains_exposed_tile(path: Path) -> bool:
    normalized = path.name.casefold().replace("-", "_")
    return EXPOSED_TILE_ID in normalized or "02_14" in normalized


def _require_exposed_identity(
    *,
    manifest: ProjectManifest,
    reference: Path,
    semantic_label: Path,
) -> None:
    source = Path(manifest.source_path)
    if not _contains_exposed_tile(source):
        raise ValueError(
            f"project source does not identify exposed Potsdam {EXPOSED_TILE_ID}: {source}"
        )
    if not _contains_exposed_tile(reference):
        raise ValueError(
            f"reference filename does not identify exposed Potsdam {EXPOSED_TILE_ID}: {reference}"
        )
    if not _contains_exposed_tile(semantic_label):
        raise ValueError(
            f"semantic-label filename does not identify exposed Potsdam {EXPOSED_TILE_ID}: "
            f"{semantic_label}"
        )


def main() -> int:
    args = parse_args()
    if not (args.project_dir / "project-manifest.json").is_file():
        raise FileNotFoundError("project manifest does not exist")
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)
    if not args.semantic_label.is_file():
        raise FileNotFoundError(args.semantic_label)

    manifest = ProjectManifest.load(args.project_dir)
    _require_exposed_identity(
        manifest=manifest,
        reference=args.reference,
        semantic_label=args.semantic_label,
    )
    prediction = manifest.artifact_path("dsm")
    if prediction is None or not prediction.is_file():
        raise ValueError("exposed diagnosis requires a completed metric DSM artifact")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = args.output_dir / "semantic-masks"
    building_dir = args.output_dir / "building-height"

    validation = validate_project_reference(
        ReferenceValidationRequest(
            project_dir=args.project_dir,
            reference_path=args.reference,
            reference_label="ISPRS Potsdam 2_14 exposed development reference",
        )
    )

    _run(
        sys.executable,
        str(CODE_ROOT / "qualification" / "prepare_potsdam_2_14_semantic_masks.py"),
        "--semantic-label",
        str(args.semantic_label),
        "--reference",
        str(args.reference),
        "--output-dir",
        str(masks_dir),
        "--ground-policy",
        args.ground_policy,
        "--tile-id",
        EXPOSED_TILE_ID,
    )
    building_mask = masks_dir / "potsdam-2_14-building-mask.tif"
    ground_mask = masks_dir / f"potsdam-2_14-ground-mask-{args.ground_policy}.tif"

    _run(
        sys.executable,
        str(CODE_ROOT / "qualification" / "evaluate_urban_building_height.py"),
        "--prediction",
        str(prediction),
        "--reference",
        str(args.reference),
        "--building-mask",
        str(building_mask),
        "--ground-mask",
        str(ground_mask),
        "--output-dir",
        str(building_dir),
        "--label",
        "potsdam-2_14-exposed-vnext-diagnostic",
        "--tile-id",
        EXPOSED_TILE_ID,
    )

    building_report_path = building_dir / "building-height-report.json"
    building_payload = json.loads(building_report_path.read_text(encoding="utf-8"))
    building_report = building_payload["report"]
    diagnosis_path = args.output_dir / "exposed-potsdam-2_14-diagnosis.json"
    diagnosis = {
        "schema_version": 1,
        "tile_id": EXPOSED_TILE_ID,
        "status": "EXPOSED_DEVELOPMENT_DIAGNOSTIC",
        "claim_boundary": (
            "This artifact is restricted to the already-exposed Potsdam 2_14 development scene. "
            "It is downstream evaluation evidence only and does not authorize opening sealed blind "
            "tiles 4_12 or 6_12."
        ),
        "project": {
            "project_dir": str(args.project_dir.resolve()),
            "project_id": manifest.project_id,
            "project_manifest_sha256": sha256_file(args.project_dir / "project-manifest.json"),
            "source_path": str(manifest.source_path),
            "prediction": str(prediction.resolve()),
            "prediction_sha256": sha256_file(prediction),
        },
        "reference": {
            "path": str(args.reference.resolve()),
            "sha256": sha256_file(args.reference),
            "validation": validation.model_dump(mode="json"),
        },
        "semantic_label": {
            "path": str(args.semantic_label.resolve()),
            "sha256": sha256_file(args.semantic_label),
            "ground_policy": args.ground_policy,
        },
        "building_height": {
            "report_path": str(building_report_path.resolve()),
            "report_sha256": sha256_file(building_report_path),
            "evaluated_buildings": len(building_report["evaluated_instance_ids"]),
            "prediction_failures": len(building_report["prediction_failure_ids"]),
            "height_mae_m": building_report["height_mae_m"],
            "height_rmse_m": building_report["height_rmse_m"],
            "height_p90_abs_error_m": building_report["height_p90_abs_error_m"],
            "within_2m_fraction": building_report["within_2m_fraction"],
            "catastrophic_over_3m_fraction": building_report["catastrophic_over_3m_fraction"],
            "top_mae_m": building_report["top_mae_m"],
            "ground_mae_m": building_report["ground_mae_m"],
        },
    }
    temporary = diagnosis_path.with_suffix(diagnosis_path.suffix + ".tmp")
    temporary.write_text(json.dumps(diagnosis, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(diagnosis_path)

    print(f"diagnosis={diagnosis_path}")
    print(f"global_rmse_m={validation.elevation.rmse_m:.6f}")
    print(f"global_mae_m={validation.elevation.mae_m:.6f}")
    print(f"slope_rmse_degrees={validation.slope.rmse_degrees:.6f}")
    print(f"building_height_rmse_m={building_report['height_rmse_m']:.6f}")
    print(f"building_height_mae_m={building_report['height_mae_m']:.6f}")
    print(f"building_top_mae_m={building_report['top_mae_m']:.6f}")
    print(f"building_ground_mae_m={building_report['ground_mae_m']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
