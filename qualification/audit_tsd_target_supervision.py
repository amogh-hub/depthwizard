from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = "potsdam-tsd-target-supervision-audit-v1"
TARGET_PROTOCOL_VERSION = "potsdam-tsd-metric-targets-v1"
EXPECTED_TRAIN_TILE_IDS = frozenset(
    {"6_7", "6_8", "6_9", "6_10", "7_7", "7_8", "7_9", "7_10"}
)
EXPECTED_DEV_TILE_IDS = frozenset({"2_10", "2_11", "2_12", "3_10", "4_10"})
FORBIDDEN_TILE_IDS = frozenset({"2_14", "3_14", "4_12", "6_12", "3_13", "6_14"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _height_bin(value: float) -> str:
    if value < 2.0:
        return "<2m"
    if value < 5.0:
        return "2-5m"
    if value < 8.0:
        return "5-8m"
    if value < 12.0:
        return "8-12m"
    if value < 20.0:
        return "12-20m"
    return ">=20m"


def _fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / max(float(denominator), 1.0)


def _tile_report(tile: dict[str, Any]) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]]]:
    tile_id = str(tile["tile_id"])
    role = str(tile["role"])
    if role not in {"train", "dev"}:
        raise ValueError(f"invalid TSD role for {tile_id}: {role}")
    if tile_id in FORBIDDEN_TILE_IDS:
        raise ValueError(f"forbidden tile present in target manifest: {tile_id}")

    instances_path = Path(str(tile["instances"])).resolve()
    if not instances_path.is_file():
        raise FileNotFoundError(instances_path)
    if sha256_file(instances_path) != str(tile["instances_sha256"]):
        raise ValueError(f"instance-evidence hash mismatch for {tile_id}")

    evidence = _load_json(instances_path)
    if evidence.get("tile_id") != tile_id or evidence.get("role") != role:
        raise ValueError(f"instance-evidence identity mismatch for {tile_id}")
    instances_raw = evidence.get("instances")
    if not isinstance(instances_raw, list):
        raise TypeError(f"instances must be a list for {tile_id}")
    instances = [item for item in instances_raw if isinstance(item, dict)]
    if len(instances) != len(instances_raw):
        raise TypeError(f"all instance records must be objects for {tile_id}")

    accepted = [item for item in instances if item.get("accepted") is True]
    rejected = [item for item in instances if item.get("accepted") is False]
    if len(accepted) != int(tile["accepted_instance_count"]):
        raise ValueError(f"accepted-instance count mismatch for {tile_id}")
    if len(rejected) != int(tile["rejected_instance_count"]):
        raise ValueError(f"rejected-instance count mismatch for {tile_id}")

    reasons: Counter[str] = Counter(str(item.get("reason")) for item in rejected)
    accepted_heights = [
        float(item["reference_height_m"])
        for item in accepted
        if item.get("reference_height_m") is not None
    ]
    rejected_measured_heights = [
        float(item["reference_height_m"])
        for item in rejected
        if item.get("reference_height_m") is not None
    ]
    accepted_height_bins = Counter(_height_bin(value) for value in accepted_heights)
    rejected_height_bins = Counter(_height_bin(value) for value in rejected_measured_heights)

    tall_rejections: list[dict[str, Any]] = []
    for item in rejected:
        height = item.get("reference_height_m")
        if height is None or float(height) < 8.0:
            continue
        tall_rejections.append(
            {
                "tile_id": tile_id,
                "role": role,
                "instance_id": int(item["instance_id"]),
                "reference_height_m": float(height),
                "area_m2": float(item["area_m2"]),
                "reason": str(item["reason"]),
                "ground_candidate_pixels": int(item["ground_candidate_pixels"]),
                "ground_inlier_pixels": int(item["ground_inlier_pixels"]),
                "ground_inlier_fraction": item.get("ground_inlier_fraction"),
                "ground_sector_coverage": item.get("ground_sector_coverage"),
                "negative_agl_fraction": item.get("negative_agl_fraction"),
            }
        )

    source_building_pixels = int(tile["source_semantic_building_pixels"])
    accepted_building_pixels = int(tile["accepted_building_pixels"])
    accepted_ge_8m = sum(value >= 8.0 for value in accepted_heights)
    rejected_ge_8m = sum(value >= 8.0 for value in rejected_measured_heights)
    accepted_ge_12m = sum(value >= 12.0 for value in accepted_heights)
    rejected_ge_12m = sum(value >= 12.0 for value in rejected_measured_heights)

    report: dict[str, Any] = {
        "tile_id": tile_id,
        "role": role,
        "accepted_instances": len(accepted),
        "rejected_instances": len(rejected),
        "instance_acceptance_fraction": _fraction(len(accepted), len(instances)),
        "source_semantic_building_pixels": source_building_pixels,
        "accepted_building_pixels": accepted_building_pixels,
        "accepted_building_pixel_fraction": _fraction(
            accepted_building_pixels, source_building_pixels
        ),
        "accepted_instance_area_m2": sum(float(item["area_m2"]) for item in accepted),
        "rejected_instance_area_m2": sum(float(item["area_m2"]) for item in rejected),
        "accepted_reference_height_bins": dict(sorted(accepted_height_bins.items())),
        "rejected_measured_height_bins": dict(sorted(rejected_height_bins.items())),
        "accepted_height_ge_8m": accepted_ge_8m,
        "rejected_height_ge_8m": rejected_ge_8m,
        "accepted_height_ge_12m": accepted_ge_12m,
        "rejected_height_ge_12m": rejected_ge_12m,
        "max_accepted_reference_height_m": max(accepted_heights) if accepted_heights else None,
        "max_rejected_reference_height_m": (
            max(rejected_measured_heights) if rejected_measured_heights else None
        ),
        "rejection_reasons": dict(sorted(reasons.items())),
    }
    return report, reasons, tall_rejections


def _aggregate(reports: list[dict[str, Any]], role: str | None) -> dict[str, Any]:
    selected = [item for item in reports if role is None or item["role"] == role]
    accepted = sum(int(item["accepted_instances"]) for item in selected)
    rejected = sum(int(item["rejected_instances"]) for item in selected)
    source_pixels = sum(int(item["source_semantic_building_pixels"]) for item in selected)
    accepted_pixels = sum(int(item["accepted_building_pixels"]) for item in selected)
    accepted_8 = sum(int(item["accepted_height_ge_8m"]) for item in selected)
    rejected_8 = sum(int(item["rejected_height_ge_8m"]) for item in selected)
    accepted_12 = sum(int(item["accepted_height_ge_12m"]) for item in selected)
    rejected_12 = sum(int(item["rejected_height_ge_12m"]) for item in selected)
    return {
        "accepted_instances": accepted,
        "rejected_instances": rejected,
        "instance_acceptance_fraction": _fraction(accepted, accepted + rejected),
        "source_semantic_building_pixels": source_pixels,
        "accepted_building_pixels": accepted_pixels,
        "accepted_building_pixel_fraction": _fraction(accepted_pixels, source_pixels),
        "accepted_height_ge_8m": accepted_8,
        "rejected_height_ge_8m": rejected_8,
        "measured_ge_8m_rejection_fraction": _fraction(rejected_8, accepted_8 + rejected_8),
        "accepted_height_ge_12m": accepted_12,
        "rejected_height_ge_12m": rejected_12,
        "measured_ge_12m_rejection_fraction": _fraction(
            rejected_12, accepted_12 + rejected_12
        ),
    }


def audit_target_supervision(
    target_manifest: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    target_manifest = target_manifest.resolve()
    manifest_sha = sha256_file(target_manifest)
    if expected_manifest_sha256 is not None and manifest_sha != expected_manifest_sha256:
        raise ValueError(
            "target manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {manifest_sha}"
        )
    manifest = _load_json(target_manifest)
    if manifest.get("status") != "GENERATED_TSD_METRIC_TARGETS":
        raise ValueError("target manifest is not a generated TSD metric-target manifest")
    if manifest.get("protocol_version") != TARGET_PROTOCOL_VERSION:
        raise ValueError("unexpected TSD metric-target protocol")
    tiles_raw = manifest.get("tiles")
    if not isinstance(tiles_raw, list) or len(tiles_raw) != 13:
        raise ValueError("TSD target audit requires exactly 13 frozen campaign tiles")
    tiles = [item for item in tiles_raw if isinstance(item, dict)]
    if len(tiles) != len(tiles_raw):
        raise TypeError("all TSD target tile records must be objects")

    reports: list[dict[str, Any]] = []
    all_reasons: Counter[str] = Counter()
    role_reasons: dict[str, Counter[str]] = {"train": Counter(), "dev": Counter()}
    tall_rejections: list[dict[str, Any]] = []
    for tile in tiles:
        report, reasons, tile_tall = _tile_report(tile)
        reports.append(report)
        all_reasons.update(reasons)
        role_reasons[str(report["role"])].update(reasons)
        tall_rejections.extend(tile_tall)

    tile_ids = {str(item["tile_id"]) for item in reports}
    expected = EXPECTED_TRAIN_TILE_IDS | EXPECTED_DEV_TILE_IDS
    if tile_ids != expected:
        raise ValueError("target manifest tile identities do not match the frozen campaign")
    if not tile_ids.isdisjoint(FORBIDDEN_TILE_IDS):
        raise ValueError("forbidden tile leaked into target supervision audit")
    train_ids = {str(item["tile_id"]) for item in reports if item["role"] == "train"}
    dev_ids = {str(item["tile_id"]) for item in reports if item["role"] == "dev"}
    if train_ids != EXPECTED_TRAIN_TILE_IDS or dev_ids != EXPECTED_DEV_TILE_IDS:
        raise ValueError("target supervision train/dev roles changed")

    tall_rejections.sort(key=lambda item: float(item["reference_height_m"]), reverse=True)
    return {
        "schema_version": 1,
        "status": "AUDITED_TSD_TARGET_SUPERVISION",
        "protocol_version": PROTOCOL_VERSION,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": manifest_sha,
        "train": _aggregate(reports, "train"),
        "dev": _aggregate(reports, "dev"),
        "overall": _aggregate(reports, None),
        "rejection_reasons_overall": dict(sorted(all_reasons.items())),
        "rejection_reasons_train": dict(sorted(role_reasons["train"].items())),
        "rejection_reasons_dev": dict(sorted(role_reasons["dev"].items())),
        "tiles": reports,
        "rejected_measured_structures_ge_8m": tall_rejections,
        "training_authorized": False,
        "training_authorization_reason": (
            "Target-supervision audit is diagnostic. Training remains fail-closed until the "
            "measured tall-structure and rejection-reason distribution is reviewed and a "
            "predeclared training gate is committed."
        ),
        "claim_boundary": (
            "Diagnostic audit of already generated frozen train/dev target evidence only. "
            "No original Potsdam raster, reserved tile, historical challenge-test tile, "
            "external-evaluation tile, exposed-corrective tile or sealed-blind tile is opened."
        ),
    }


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def print_audit_summary(audit: dict[str, Any]) -> None:
    print("===== TSD TARGET SUPERVISION AUDIT =====")
    for role in ("train", "dev", "overall"):
        stats = audit[role]
        total = int(stats["accepted_instances"]) + int(stats["rejected_instances"])
        print(
            f"{role}: accepted={stats['accepted_instances']}/{total} "
            f"({_percent(float(stats['instance_acceptance_fraction']))}) "
            "building_pixel_support="
            f"{_percent(float(stats['accepted_building_pixel_fraction']))} "
            f"accepted>=8m={stats['accepted_height_ge_8m']} "
            f"rejected>=8m={stats['rejected_height_ge_8m']} "
            f"accepted>=12m={stats['accepted_height_ge_12m']} "
            f"rejected>=12m={stats['rejected_height_ge_12m']}"
        )
    print("top_rejection_reasons:")
    reasons = sorted(
        audit["rejection_reasons_overall"].items(), key=lambda item: (-int(item[1]), item[0])
    )
    for reason, count in reasons[:8]:
        print(f"  {count}: {reason}")
    tall = audit["rejected_measured_structures_ge_8m"]
    print(f"measured_rejected_structures_ge_8m={len(tall)}")
    for item in tall[:10]:
        print(
            "  "
            f"tile={item['tile_id']} instance={item['instance_id']} "
            f"height_m={float(item['reference_height_m']):.3f} "
            f"area_m2={float(item['area_m2']):.2f} reason={item['reason']}"
        )
    print("training_authorized=false")
    print("sealed_blind_tile_payloads_consumed=false")
    print("original_dataset_rasters_opened=false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit frozen TSD target supervision.")
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_target_supervision(
        args.target_manifest,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    _write_json_atomic(args.output, audit)
    print_audit_summary(audit)
    print(f"audit={args.output.resolve()}")
    print(f"audit_sha256={sha256_file(args.output.resolve())}")
    print("TSD_TARGET_SUPERVISION_AUDIT=COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
