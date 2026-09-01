from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import rasterio

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.contracts import ReferenceValidationRequest
from depthwizard.evaluation.potsdam_semantics import decode_potsdam_semantic_labels
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file

EXPOSED_TILE_ID = "2_14"
CANONICAL_GROUND_POLICY = "strict"
PROTOCOL_VERSION = "potsdam-2_14-building-diagnostic-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete downstream diagnostic for the already-exposed Potsdam 2_14 operator "
            "scene: project reference validation, deterministic official semantic-mask preparation, "
            "and per-building height evaluation. The canonical protocol is frozen to strict "
            "impervious-surface ground support. Sealed blind tiles are not accepted."
        )
    )
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--semantic-label",
        type=Path,
        help=(
            "Explicit official ISPRS semantic label for exposed tile 2_14. If omitted, "
            "--dataset-root is scanned and exactly one palette-valid, exact-grid label must resolve."
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Potsdam dataset root used only to auto-resolve the exposed 2_14 semantic label.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=CODE_ROOT, check=True)


def _contains_exposed_tile(path: Path) -> bool:
    normalized = path.name.casefold().replace("-", "_")
    return EXPOSED_TILE_ID in normalized or "02_14" in normalized


def _semantic_name_candidate(path: Path) -> bool:
    if path.suffix.casefold() not in {".tif", ".tiff"}:
        return False
    normalized = path.name.casefold().replace("-", "_")
    return _contains_exposed_tile(path) and ("label" in normalized or "ground_truth" in normalized)


def _palette_valid_exact_grid_label(candidate: Path, reference: Path) -> bool:
    try:
        with rasterio.open(reference) as ref, rasterio.open(candidate) as labels:
            if labels.count != 3:
                return False
            if (labels.height, labels.width) != (ref.height, ref.width):
                return False
            if labels.crs is not None and ref.crs is not None and labels.crs != ref.crs:
                return False
            if not labels.transform.is_identity and not labels.transform.almost_equals(ref.transform):
                return False
            rgb = np.moveaxis(labels.read((1, 2, 3)), 0, -1)
        decoded = decode_potsdam_semantic_labels(rgb)
    except (OSError, ValueError):
        return False
    return (
        decoded.class_pixel_counts["building"] > 0
        and decoded.class_pixel_counts["impervious"] > 0
    )


def _resolve_semantic_label(
    *,
    explicit: Path | None,
    dataset_root: Path | None,
    reference: Path,
) -> tuple[Path, str]:
    if explicit is not None:
        if not explicit.is_file():
            raise FileNotFoundError(explicit)
        if not _contains_exposed_tile(explicit):
            raise ValueError(
                f"semantic-label filename does not identify exposed Potsdam {EXPOSED_TILE_ID}: "
                f"{explicit}"
            )
        if not _palette_valid_exact_grid_label(explicit, reference):
            raise ValueError(
                "explicit semantic label does not satisfy the official ISPRS palette and exact-grid "
                f"contract against the exposed reference: {explicit}"
            )
        return explicit, "explicit"

    if dataset_root is None:
        raise ValueError("provide --semantic-label or --dataset-root for exposed label resolution")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    name_candidates = sorted(
        path for path in dataset_root.rglob("*") if path.is_file() and _semantic_name_candidate(path)
    )
    valid = [
        path for path in name_candidates if _palette_valid_exact_grid_label(path, reference)
    ]
    if not valid:
        rendered = ", ".join(str(path) for path in name_candidates[:12]) or "none"
        raise FileNotFoundError(
            "could not auto-resolve an official palette-valid exact-grid semantic label for exposed "
            f"Potsdam {EXPOSED_TILE_ID}; filename candidates inspected: {rendered}"
        )
    if len(valid) > 1:
        rendered = ", ".join(str(path) for path in valid)
        raise RuntimeError(
            "ambiguous exposed semantic labels: more than one official-palette exact-grid candidate "
            f"was found. Pass --semantic-label explicitly before viewing results: {rendered}"
        )
    return valid[0], "dataset_root_unique_palette_exact_grid"


def _git_identity() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=CODE_ROOT,
            text=True,
        ).strip()
        tracked_status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=CODE_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("qualification requires an attributable Git checkout") from exc
    if tracked_status:
        raise RuntimeError(
            "tracked repository files are modified; commit or restore them before generating "
            "qualification evidence"
        )
    if len(sha) != 40:
        raise RuntimeError(f"unexpected Git SHA from qualification checkout: {sha!r}")
    return sha


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

    git_sha = _git_identity()
    semantic_label, semantic_resolution = _resolve_semantic_label(
        explicit=args.semantic_label,
        dataset_root=args.dataset_root,
        reference=args.reference,
    )

    manifest = ProjectManifest.load(args.project_dir)
    _require_exposed_identity(
        manifest=manifest,
        reference=args.reference,
        semantic_label=semantic_label,
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
        str(semantic_label),
        "--reference",
        str(args.reference),
        "--output-dir",
        str(masks_dir),
        "--ground-policy",
        CANONICAL_GROUND_POLICY,
        "--tile-id",
        EXPOSED_TILE_ID,
    )
    building_mask = masks_dir / "potsdam-2_14-building-mask.tif"
    ground_mask = masks_dir / f"potsdam-2_14-ground-mask-{CANONICAL_GROUND_POLICY}.tif"

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
        "protocol_version": PROTOCOL_VERSION,
        "tile_id": EXPOSED_TILE_ID,
        "status": "EXPOSED_DEVELOPMENT_DIAGNOSTIC",
        "claim_boundary": (
            "This artifact is restricted to the already-exposed Potsdam 2_14 development scene. "
            "It is downstream evaluation evidence only and does not authorize opening sealed blind "
            "tiles 4_12 or 6_12. The canonical ground policy is frozen before result inspection."
        ),
        "qualification_runtime": {
            "git_sha": git_sha,
            "tracked_worktree_clean": True,
            "runner_sha256": sha256_file(Path(__file__)),
            "semantic_label_resolution": semantic_resolution,
        },
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
            "path": str(semantic_label.resolve()),
            "sha256": sha256_file(semantic_label),
            "ground_policy": CANONICAL_GROUND_POLICY,
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
    print(f"protocol_version={PROTOCOL_VERSION}")
    print(f"qualification_git_sha={git_sha}")
    print(f"semantic_label={semantic_label}")
    print(f"semantic_label_resolution={semantic_resolution}")
    print(f"ground_policy={CANONICAL_GROUND_POLICY}")
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
