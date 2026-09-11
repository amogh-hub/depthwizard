from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from depthwizard.data.registry import load_registry
from depthwizard.provenance.manifest import sha256_file

EVALUATION_SPLITS = {"test", "cross_sensor_test"}
ROOT = Path(__file__).resolve().parents[1]


class PredictionFreezeError(ValueError):
    """The draft prediction set cannot be frozen without violating campaign identity rules."""


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise PredictionFreezeError("draft prediction manifest root must be an object")
    return payload


def _resolve(path_value: object, *, base_dir: Path, context: str) -> Path:
    if not isinstance(path_value, str) or not path_value.strip():
        raise PredictionFreezeError(f"{context} must be a non-empty file path")
    path = Path(path_value)
    candidate = path if path.is_absolute() else base_dir / path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    return resolved


def _portable_path(path: Path, *, output_dir: Path) -> str:
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def freeze_prediction_manifest(
    registry_path: Path,
    draft_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Freeze checkpoint/prediction identities without opening evaluation reference rasters.

    The draft schema is intentionally smaller than the final CampaignManifest schema:

    schema_version: 2
    model_id: ...
    checkpoint_path: ...
    predictions:
      - scene_id: ...
        prediction_path: ...
        calibration_evidence_paths: [...]
        notes: optional

    This function reads the registry only to determine the exact evaluation scene IDs. It does not
    inspect, hash, align or open any reference raster. Reference evaluation remains a later step.
    """
    registry_file = registry_path.resolve(strict=True)
    draft_file = draft_path.resolve(strict=True)
    registry = load_registry(registry_file)
    draft = _load_mapping(draft_file)
    if draft.get("schema_version") != 2:
        raise PredictionFreezeError("draft prediction schema_version must be 2")
    model_id = draft.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        raise PredictionFreezeError("draft model_id must be a non-empty string")

    draft_base = draft_file.parent
    checkpoint = _resolve(
        draft.get("checkpoint_path"),
        base_dir=draft_base,
        context="production checkpoint",
    )
    raw_predictions = draft.get("predictions")
    if not isinstance(raw_predictions, list) or not raw_predictions:
        raise PredictionFreezeError("draft predictions must be a non-empty list")

    expected_scene_ids = {
        scene.scene_id for scene in registry.scenes if scene.split in EVALUATION_SPLITS
    }
    if not expected_scene_ids:
        raise PredictionFreezeError("registry contains no test/cross_sensor_test evaluation scenes")

    by_scene: dict[str, dict[str, Any]] = {}
    for index, raw_entry in enumerate(raw_predictions):
        if not isinstance(raw_entry, dict):
            raise PredictionFreezeError(f"draft prediction #{index + 1} must be an object")
        scene_id = raw_entry.get("scene_id")
        if not isinstance(scene_id, str) or not scene_id.strip():
            raise PredictionFreezeError(f"draft prediction #{index + 1} has invalid scene_id")
        if scene_id in by_scene:
            raise PredictionFreezeError(f"duplicate draft prediction for scene {scene_id!r}")
        by_scene[scene_id] = raw_entry

    supplied_scene_ids = set(by_scene)
    missing = sorted(expected_scene_ids - supplied_scene_ids)
    extra = sorted(supplied_scene_ids - expected_scene_ids)
    if missing or extra:
        raise PredictionFreezeError(
            "draft predictions must exactly match registry evaluation scenes; "
            f"missing={missing}, extra={extra}"
        )

    output_file = output_path.resolve(strict=False)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frozen_entries: list[dict[str, Any]] = []
    freeze_evidence: list[dict[str, Any]] = []
    for scene_id in sorted(expected_scene_ids):
        entry = by_scene[scene_id]
        prediction = _resolve(
            entry.get("prediction_path"),
            base_dir=draft_base,
            context=f"prediction for {scene_id}",
        )
        raw_calibration = entry.get("calibration_evidence_paths")
        if not isinstance(raw_calibration, list) or not raw_calibration:
            raise PredictionFreezeError(
                f"prediction {scene_id!r} requires at least one calibration evidence path"
            )
        calibration_paths = [
            _resolve(
                value,
                base_dir=draft_base,
                context=f"calibration evidence for {scene_id}",
            )
            for value in raw_calibration
        ]
        prediction_sha = sha256_file(prediction)
        vertical_fields: dict[str, str] = {}
        for key in (
            "prediction_vertical_datum",
            "reference_vertical_datum",
            "prediction_elevation_reference",
            "reference_elevation_reference",
        ):
            value = entry.get(key)
            if not isinstance(value, str) or not value.strip():
                raise PredictionFreezeError(
                    f"prediction {scene_id!r} requires non-empty vertical field {key!r}"
                )
            vertical_fields[key] = value.strip()
        if (
            vertical_fields["prediction_vertical_datum"].casefold()
            != vertical_fields["reference_vertical_datum"].casefold()
        ):
            raise PredictionFreezeError(
                f"prediction {scene_id!r} vertical datum does not match its reference"
            )
        placeholders = {"unknown", "unspecified", "none", "null", "n/a", "na", "tbd"}
        if any(
            vertical_fields[key].casefold() in placeholders
            for key in ("prediction_vertical_datum", "reference_vertical_datum")
        ):
            raise PredictionFreezeError(
                f"prediction {scene_id!r} vertical datum must be explicit, not a placeholder"
            )
        if (
            vertical_fields["prediction_elevation_reference"]
            != vertical_fields["reference_elevation_reference"]
        ):
            raise PredictionFreezeError(
                f"prediction {scene_id!r} elevation-reference type does not match its reference"
            )
        if vertical_fields["prediction_elevation_reference"] not in {
            "orthometric",
            "ellipsoidal",
            "local",
        }:
            raise PredictionFreezeError(
                f"prediction {scene_id!r} uses an unsupported elevation-reference type"
            )
        frozen_entries.append(
            {
                "scene_id": scene_id,
                "prediction_path": _portable_path(
                    prediction,
                    output_dir=output_file.parent,
                ),
                "prediction_sha256": prediction_sha,
                "calibration_evidence_paths": [
                    _portable_path(path, output_dir=output_file.parent)
                    for path in calibration_paths
                ],
                "prediction_vertical_units": "m",
                "reference_vertical_units": "m",
                **vertical_fields,
                "notes": entry.get("notes"),
            }
        )
        freeze_evidence.append(
            {
                "scene_id": scene_id,
                "prediction_path": str(prediction),
                "prediction_sha256": prediction_sha,
                "calibration_evidence": [
                    {"path": str(path), "sha256": sha256_file(path)} for path in calibration_paths
                ],
            }
        )

    git_head = _git_head()
    frozen = {
        "schema_version": 2,
        "git_head": git_head,
        "model_id": model_id,
        "checkpoint_sha256": sha256_file(checkpoint),
        "predictions": frozen_entries,
    }
    temporary = output_file.with_suffix(output_file.suffix + ".tmp")
    temporary.write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    temporary.replace(output_file)

    report = {
        "schema_version": 1,
        "status": "PASS_FINAL_SCIENCE_PREDICTION_FREEZE",
        "git_head": git_head,
        "registry_path": str(registry_file),
        "registry_sha256": sha256_file(registry_file),
        "draft_path": str(draft_file),
        "draft_sha256": sha256_file(draft_file),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": frozen["checkpoint_sha256"],
        "frozen_manifest_path": str(output_file),
        "frozen_manifest_sha256": sha256_file(output_file),
        "evaluation_scene_count": len(frozen_entries),
        "predictions": freeze_evidence,
        "reference_rasters_opened_or_hashed": False,
        "claim_boundary": (
            "Identity-freeze step only. This report proves checkpoint/prediction/calibration-evidence "
            "byte identities were frozen before the separate reference evaluator was run."
        ),
    }
    report_path = output_file.with_name(f"{output_file.stem}-freeze-report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the DepthWizard final-science checkpoint and prediction SHA-256 identities."
    )
    parser.add_argument("registry", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("output", type=Path, help="Output frozen prediction YAML")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = freeze_prediction_manifest(args.registry, args.draft, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
