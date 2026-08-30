#!/usr/bin/env python3
"""Compare already-frozen Potsdam 2_14 candidates against the official reference DSM.

All candidate predictions in this script were frozen before evaluator truth was exposed. This stage
therefore performs a fair retrospective comparison. Running it permanently changes Potsdam 2_14
from a blind qualification scene into a development/diagnostic scene; future final acceptance must
use a different untouched Potsdam tile.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from depthwizard.evaluation.report import validate_geospatial_dsm

HEAD = "d2c94772a60eea6b113107c3a3f2985503802f87"
REFERENCE_REL = Path("data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif")
OUTPUT_REL = Path("workspace/potsdam-2-14-frozen-candidate-reference-comparison")
OLD_METRIC_REL = Path(
    "workspace/final-science-data/predictions/potsdam-2-14-urban-test/metric-dsm.tif"
)
OLD_METRIC_SHA256 = "116bb397c813bff2afdfcae1747503353d01a27364c7e6db947d7574299730ef"

CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "id": "old_frozen_339bdf4",
        "source_head": "339bdf485149f552db846543b9e09377b567c19c",
        "metric_path": OLD_METRIC_REL,
        "metric_sha256": OLD_METRIC_SHA256,
        "freeze_path": None,
        "pipeline_revision": "historical_frozen_single_lattice",
    },
    {
        "id": "v2_scene_global_affine",
        "source_head": "5e876709ad5a98cafb03a5c60ab7d84b135c6ffe",
        "metric_path": Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/metric-dsm.tif"),
        "freeze_path": Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/pre-reference-freeze.json"),
        "pipeline_revision": "scene-global-affine-mosaic-v2",
    },
    {
        "id": "v3_global_scaffold_residual",
        "source_head": "2a452f691ab235f69c1884bbb7b32885cc535213",
        "metric_path": Path("workspace/urban-mosaic-v3/2a452f6-potsdam-2_14/metric-dsm.tif"),
        "freeze_path": Path("workspace/urban-mosaic-v3/2a452f6-potsdam-2_14/pre-reference-freeze.json"),
        "pipeline_revision": "global-scaffold-residual-mosaic-v3",
    },
    {
        "id": "v4_dual_lattice",
        "source_head": HEAD,
        "metric_path": Path("workspace/urban-mosaic-v4/d2c9477-potsdam-2_14/metric-dsm.tif"),
        "freeze_path": Path("workspace/urban-mosaic-v4/d2c9477-potsdam-2_14/pre-reference-freeze.json"),
        "pipeline_revision": "dual-lattice-affine-mosaic-v4",
    },
)


class ComparisonError(RuntimeError):
    pass


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkout(root: Path) -> None:
    current = git(root, "rev-parse", "HEAD")
    if current != HEAD:
        raise ComparisonError(f"requires exact v4 source head {HEAD}; current={current}")
    if git(root, "status", "--porcelain"):
        raise ComparisonError("repository must be clean before reference comparison")


def frozen_identity(root: Path, candidate: dict[str, Any]) -> tuple[Path, str]:
    metric_path = (root / candidate["metric_path"]).resolve(strict=True)
    explicit_sha = candidate.get("metric_sha256")
    freeze_rel = candidate.get("freeze_path")
    if explicit_sha is not None:
        expected_sha = str(explicit_sha)
    else:
        if not isinstance(freeze_rel, Path):
            raise ComparisonError(f"candidate {candidate['id']} has no frozen identity source")
        freeze_path = (root / freeze_rel).resolve(strict=True)
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
        if freeze.get("reference_values_consumed") is not False:
            raise ComparisonError(f"candidate {candidate['id']} was not frozen reference-blind")
        if freeze.get("reference_raster_opened_or_hashed") is not False:
            raise ComparisonError(f"candidate {candidate['id']} freeze says reference was already opened")
        if freeze.get("git_head") != candidate["source_head"]:
            raise ComparisonError(f"candidate {candidate['id']} freeze source head mismatch")
        record = freeze.get("metric_dsm")
        if not isinstance(record, dict):
            raise ComparisonError(f"candidate {candidate['id']} freeze lacks metric_dsm identity")
        frozen_path = record.get("path")
        frozen_sha = record.get("sha256")
        if not isinstance(frozen_path, str) or not isinstance(frozen_sha, str):
            raise ComparisonError(f"candidate {candidate['id']} metric_dsm identity malformed")
        if Path(frozen_path).resolve() != metric_path:
            raise ComparisonError(f"candidate {candidate['id']} metric path differs from freeze")
        expected_sha = frozen_sha
    actual_sha = sha256(metric_path)
    if actual_sha != expected_sha:
        raise ComparisonError(
            f"candidate {candidate['id']} SHA mismatch: expected={expected_sha} actual={actual_sha}"
        )
    return metric_path, actual_sha


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    output = (root / OUTPUT_REL).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)

    # Verify every candidate while the official reference is still unopened in this process.
    identities: dict[str, tuple[Path, str]] = {}
    for candidate in CANDIDATES:
        identities[str(candidate["id"])] = frozen_identity(root, candidate)

    reference = (root / REFERENCE_REL).resolve(strict=True)
    reference_sha = sha256(reference)
    exposed_at = datetime.now(timezone.utc).isoformat()
    exposure = {
        "schema_version": 1,
        "status": "REFERENCE_EXPOSED_FOR_DEVELOPMENT_DIAGNOSTICS",
        "scene": "ISPRS Potsdam 2_14",
        "reference_path": str(reference),
        "reference_sha256": reference_sha,
        "exposed_at_utc": exposed_at,
        "policy": (
            "Potsdam 2_14 is no longer eligible as a blind final acceptance scene after this run; "
            "reserve a different untouched Potsdam tile for final confirmation."
        ),
    }
    (output / "reference-exposure.json").write_text(
        json.dumps(exposure, indent=2), encoding="utf-8"
    )

    results: dict[str, Any] = {}
    pre_hashes = {candidate_id: digest for candidate_id, (_, digest) in identities.items()}

    for candidate in CANDIDATES:
        candidate_id = str(candidate["id"])
        metric_path, metric_sha = identities[candidate_id]
        candidate_output = output / candidate_id
        payload = validate_geospatial_dsm(metric_path, reference, candidate_output)
        results[candidate_id] = {
            "source_head": candidate["source_head"],
            "pipeline_revision": candidate["pipeline_revision"],
            "metric_dsm": {"path": str(metric_path), "sha256": metric_sha},
            "official": payload["official"],
            "slope": payload["diagnostics"]["slope"],
            "ground_sample_distance_m": payload["diagnostics"]["ground_sample_distance_m"],
            "residual_path": str((candidate_output / "residual.tif").resolve()),
        }

    # Evaluation must be read-only with respect to frozen candidates.
    for candidate_id, (metric_path, _) in identities.items():
        after = sha256(metric_path)
        if after != pre_hashes[candidate_id]:
            raise ComparisonError(f"evaluation mutated frozen candidate {candidate_id}")

    by_rmse = sorted(
        results,
        key=lambda item: float(results[item]["official"]["rmse_m"]),
    )
    by_slope_rmse = sorted(
        results,
        key=lambda item: float(results[item]["slope"]["rmse_degrees"]),
    )
    by_correlation = sorted(
        results,
        key=lambda item: (
            results[item]["official"]["pearson_r"] is None,
            -float(results[item]["official"]["pearson_r"] or -1.0),
        ),
    )

    report = {
        "schema_version": 1,
        "status": "PASS_FROZEN_CANDIDATE_REFERENCE_COMPARISON",
        "comparison_head": HEAD,
        "scene": "ISPRS Potsdam 2_14",
        "reference": {"path": str(reference), "sha256": reference_sha},
        "reference_exposed_for_future_development": True,
        "blind_final_acceptance_allowed_on_2_14": False,
        "results": results,
        "rankings": {
            "lowest_elevation_rmse": by_rmse,
            "lowest_slope_rmse": by_slope_rmse,
            "highest_pearson_correlation": by_correlation,
        },
    }
    report_path = output / "frozen-candidate-reference-comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print("PASS_FROZEN_CANDIDATE_REFERENCE_COMPARISON")
    print(f"report={report_path}")
    print(f"reference_sha256={reference_sha}")
    print("potsdam_2_14_blind_status=EXPOSED_DEVELOPMENT_ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
