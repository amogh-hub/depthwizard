from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.errors import NotGeoreferencedWarning

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.potsdam import (
    inspect_potsdam_reference_contract,
    inspect_potsdam_rgb_contract,
)
from depthwizard.evaluation.potsdam_semantics import decode_potsdam_semantic_labels
from depthwizard.height_model.terrain_structure_split import (
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    RESERVED_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    initial_tsd_campaign_split,
)
from depthwizard.height_model.terrain_structure_targets import (
    TerrainStructureTargetConfig,
    prepare_metric_terrain_structure_targets,
)
from depthwizard.provenance.manifest import sha256_file

TARGET_PROTOCOL_VERSION = "potsdam-tsd-metric-targets-v1"
GROUND_POLICY = "strict_impervious_only"
TARGET_PACK_FORMAT = "npz-compressed-metric-terrain-structure-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate auditable metric Terrain-Structure Decomposition targets only for the frozen "
            "Potsdam train/dev campaign. Every TIFF, TFW, and semantic-label identity is re-hashed "
            "against the frozen split before raster content is consumed."
        )
    )
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def _tracked_source_identity() -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "refusing TSD target generation from a tracked-dirty worktree; commit or restore first"
        )
    if len(head) != 40:
        raise RuntimeError(f"unexpected Git SHA: {head!r}")
    return head, branch


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _validate_frozen_split(payload: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    split = initial_tsd_campaign_split()
    expected_train = tuple(split.train_tile_ids)
    expected_dev = tuple(split.dev_tile_ids)
    expected_all = set(expected_train) | set(expected_dev)

    if payload.get("schema_version") != 4:
        raise ValueError("TSD target generation requires frozen split-manifest schema_version=4")
    if payload.get("status") != "FROZEN_TSD_SUPERVISION_SPLIT":
        raise ValueError("split manifest is not in FROZEN_TSD_SUPERVISION_SPLIT state")
    if payload.get("protocol_version") != TSD_SPLIT_PROTOCOL_VERSION:
        raise ValueError("split protocol version mismatch")
    if payload.get("campaign_protocol_version") != INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION:
        raise ValueError("campaign protocol version mismatch")
    if tuple(payload.get("train_tile_ids", ())) != expected_train:
        raise ValueError("frozen train tile ids differ from the predeclared campaign")
    if tuple(payload.get("dev_tile_ids", ())) != expected_dev:
        raise ValueError("frozen dev tile ids differ from the predeclared campaign")

    buffers = set(payload.get("campaign_buffer_tile_ids_withheld", ()))
    if buffers != set(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS):
        raise ValueError("campaign buffer population differs from the predeclared campaign")

    records_raw = payload.get("tiles")
    if not isinstance(records_raw, list):
        raise TypeError("split manifest tiles must be a list")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in records_raw:
        if not isinstance(raw, dict):
            raise TypeError("split manifest tile records must be objects")
        tile_id = raw.get("tile_id")
        role = raw.get("role")
        if not isinstance(tile_id, str) or tile_id not in expected_all:
            raise ValueError(f"unexpected TSD target tile record: {tile_id!r}")
        if tile_id in seen:
            raise ValueError(f"duplicate TSD target tile record: {tile_id}")
        seen.add(tile_id)
        expected_role = "train" if tile_id in expected_train else "dev"
        if role != expected_role:
            raise ValueError(
                f"frozen role mismatch for {tile_id}: expected {expected_role}, got {role!r}"
            )
        if tile_id in RESERVED_TILE_IDS or tile_id in HISTORICAL_CHALLENGE_TEST_TILE_IDS:
            raise ValueError(f"prohibited tile reached TSD target generation: {tile_id}")
        required_identity_keys = (
            "rgb",
            "rgb_sha256",
            "rgb_world_file",
            "rgb_world_file_sha256",
            "reference_dsm",
            "reference_dsm_sha256",
            "reference_dsm_world_file",
            "reference_dsm_world_file_sha256",
            "semantic_label",
            "semantic_label_sha256",
            "rgb_contract",
            "reference_contract",
            "semantic_label_metadata",
        )
        missing = [key for key in required_identity_keys if key not in raw]
        if missing:
            raise ValueError(f"frozen tile {tile_id} is missing identity fields: {missing}")
        records.append(raw)

    if seen != expected_all:
        missing = sorted(expected_all - seen)
        raise ValueError(f"frozen split manifest is missing target tiles: {missing}")
    return tuple(records)


def _require_inside_dataset(path: Path, dataset_root: Path, *, role: str) -> Path:
    resolved = path.resolve()
    root = dataset_root.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{role} escapes the frozen dataset root: {resolved}")
    return resolved


def _verify_hash(path: Path, expected: object, *, role: str) -> str:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid frozen SHA-256 for {role}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"frozen identity mismatch for {role}: expected {expected}, got {actual}: {path}"
        )
    return actual


def _affine_values(transform: Affine) -> tuple[float, float, float, float, float, float]:
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def _label_alignment_basis(
    *,
    label_shape: tuple[int, int],
    rgb_shape: tuple[int, int],
    label_transform: Affine,
    rgb_transform: Affine,
    label_crs: CRS | None,
    rgb_crs: CRS | None,
) -> str:
    if label_shape != rgb_shape:
        raise ValueError("semantic label and RGB must share the exact native pixel dimensions")
    if label_transform.is_identity:
        return "official_same_tile_exact_pixel_grid"
    if not label_transform.almost_equals(rgb_transform):
        raise ValueError("semantic-label affine transform disagrees with frozen RGB grid")
    if label_crs is not None and rgb_crs is not None and label_crs != rgb_crs:
        raise ValueError("semantic-label CRS disagrees with frozen RGB grid")
    return "geospatial_transform_match"


def _verify_record_identities(
    record: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Path]:
    tile_id = str(record["tile_id"])
    fields = {
        "rgb": ("rgb", "rgb_sha256"),
        "rgb_world_file": ("rgb_world_file", "rgb_world_file_sha256"),
        "reference_dsm": ("reference_dsm", "reference_dsm_sha256"),
        "reference_dsm_world_file": (
            "reference_dsm_world_file",
            "reference_dsm_world_file_sha256",
        ),
        "semantic_label": ("semantic_label", "semantic_label_sha256"),
    }
    verified: dict[str, Path] = {}
    for role, (path_key, hash_key) in fields.items():
        raw_path = record[path_key]
        if not isinstance(raw_path, str):
            raise TypeError(f"frozen {path_key} for {tile_id} must be a path string")
        path = _require_inside_dataset(Path(raw_path), dataset_root, role=f"{tile_id} {role}")
        _verify_hash(path, record[hash_key], role=f"{tile_id} {role}")
        verified[role] = path

    if verified["rgb_world_file"].suffix.casefold() != ".tfw":
        raise ValueError(f"frozen RGB georeference sidecar is not .tfw for {tile_id}")
    if verified["reference_dsm_world_file"].suffix.casefold() != ".tfw":
        raise ValueError(f"frozen DSM georeference sidecar is not .tfw for {tile_id}")

    current_rgb = inspect_potsdam_rgb_contract(verified["rgb"])
    current_reference = inspect_potsdam_reference_contract(
        verified["rgb"], verified["reference_dsm"]
    )
    if current_rgb != record["rgb_contract"]:
        raise ValueError(f"RGB geospatial contract changed since split freeze for {tile_id}")
    if current_reference != record["reference_contract"]:
        raise ValueError(f"DSM geospatial contract changed since split freeze for {tile_id}")
    return verified


def _read_reference(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        band = src.read(1, masked=True)
    if np.ma.isMaskedArray(band):
        return np.asarray(band.filled(np.nan), dtype=np.float64)
    return np.asarray(band, dtype=np.float64)


def _read_semantics(
    label_path: Path,
    rgb_path: Path,
    *,
    reference_shape: tuple[int, int],
    expected_metadata: object,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(label_path) as label, rasterio.open(rgb_path) as rgb:
            if label.count != 3:
                raise ValueError(f"Potsdam semantic label must have exactly 3 bands: {label_path}")
            alignment_basis = _label_alignment_basis(
                label_shape=(label.height, label.width),
                rgb_shape=(rgb.height, rgb.width),
                label_transform=label.transform,
                rgb_transform=rgb.transform,
                label_crs=label.crs,
                rgb_crs=rgb.crs,
            )
            current_metadata = {
                "width": label.width,
                "height": label.height,
                "bands": label.count,
                "dtype": label.dtypes[0],
                "crs": label.crs.to_string() if label.crs is not None else None,
                "transform_is_identity": bool(label.transform.is_identity),
                "alignment_basis": alignment_basis,
            }
            if current_metadata != expected_metadata:
                raise ValueError(
                    f"semantic-label metadata/alignment changed since split freeze: {label_path}"
                )
            rgb_labels = np.moveaxis(label.read((1, 2, 3)), 0, -1)

    decoded = decode_potsdam_semantic_labels(rgb_labels)
    height, width = reference_shape
    if height > decoded.building.shape[0] or width > decoded.building.shape[1]:
        raise ValueError("reference DSM extends beyond semantic-label native coverage")
    return (
        decoded.building[:height, :width],
        decoded.strict_ground[:height, :width],
        {
            "alignment_basis": alignment_basis,
            "class_pixel_counts": decoded.class_pixel_counts,
            "unlabeled_pixels": decoded.unlabeled_pixels,
            "native_label_shape": [decoded.building.shape[0], decoded.building.shape[1]],
            "target_crop_shape": [height, width],
        },
    )


def _write_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _reason_counts(instances: tuple[object, ...]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for instance in instances:
        reason = getattr(instance, "reason", None)
        if isinstance(reason, str):
            counter[reason] += 1
    return dict(sorted(counter.items()))


def _generate_tile(
    record: dict[str, Any],
    *,
    dataset_root: Path,
    output_dir: Path,
    config: TerrainStructureTargetConfig,
) -> dict[str, Any]:
    tile_id = str(record["tile_id"])
    verified = _verify_record_identities(record, dataset_root)
    reference = _read_reference(verified["reference_dsm"])
    reference_shape = (int(reference.shape[0]), int(reference.shape[1]))
    building, strict_ground, semantic_evidence = _read_semantics(
        verified["semantic_label"],
        verified["rgb"],
        reference_shape=reference_shape,
        expected_metadata=record["semantic_label_metadata"],
    )

    targets = prepare_metric_terrain_structure_targets(
        reference,
        building,
        strict_ground,
        gsd_x_m=0.05,
        gsd_y_m=0.05,
        config=config,
    )
    valid_pixels = int(np.count_nonzero(targets.valid_mask))
    building_pixels = int(np.count_nonzero(targets.building_mask))
    ground_pixels = int(np.count_nonzero(targets.ground_mask))
    if valid_pixels <= 0 or building_pixels <= 0 or ground_pixels <= 0:
        raise RuntimeError(f"TSD target generation produced empty scientific support for {tile_id}")

    tile_dir = output_dir / tile_id
    target_path = tile_dir / "metric-targets.npz"
    instances_path = tile_dir / "instances.json"
    if target_path.exists() or instances_path.exists():
        raise FileExistsError(
            f"refusing to overwrite existing TSD target evidence for {tile_id}: {tile_dir}"
        )

    _write_npz_atomic(
        target_path,
        terrain_m=np.asarray(targets.terrain_m, dtype=np.float32),
        above_ground_m=np.asarray(targets.above_ground_m, dtype=np.float32),
        valid_mask=np.asarray(targets.valid_mask, dtype=np.uint8),
        building_mask=np.asarray(targets.building_mask, dtype=np.uint8),
        ground_mask=np.asarray(targets.ground_mask, dtype=np.uint8),
        boundary_mask=np.asarray(targets.boundary_mask, dtype=np.uint8),
    )
    instances_payload = {
        "schema_version": 1,
        "protocol_version": TARGET_PROTOCOL_VERSION,
        "tile_id": tile_id,
        "role": record["role"],
        "instances": [asdict(instance) for instance in targets.instances],
    }
    _write_json_atomic(instances_path, instances_payload)

    accepted = len(targets.accepted_instance_ids)
    rejected = len(targets.rejected_instance_ids)
    return {
        "tile_id": tile_id,
        "role": record["role"],
        "target_pack": str(target_path.resolve()),
        "target_pack_sha256": sha256_file(target_path),
        "instances": str(instances_path.resolve()),
        "instances_sha256": sha256_file(instances_path),
        "target_shape": [int(reference.shape[0]), int(reference.shape[1])],
        "native_gsd_m": 0.05,
        "valid_pixels": valid_pixels,
        "accepted_building_pixels": building_pixels,
        "strict_ground_pixels": ground_pixels,
        "source_semantic_building_pixels": int(np.count_nonzero(building)),
        "accepted_instance_count": accepted,
        "rejected_instance_count": rejected,
        "rejection_reason_counts": _reason_counts(targets.instances),
        "semantic_evidence": semantic_evidence,
        "input_identity": {
            "rgb_sha256": record["rgb_sha256"],
            "rgb_world_file_sha256": record["rgb_world_file_sha256"],
            "reference_dsm_sha256": record["reference_dsm_sha256"],
            "reference_dsm_world_file_sha256": record[
                "reference_dsm_world_file_sha256"
            ],
            "semantic_label_sha256": record["semantic_label_sha256"],
        },
    }


def main() -> int:
    args = parse_args()
    source_sha, source_branch = _tracked_source_identity()
    split_payload = _load_json(args.split_manifest)
    records = _validate_frozen_split(split_payload)

    dataset_root_raw = split_payload.get("dataset_root")
    if not isinstance(dataset_root_raw, str):
        raise TypeError("frozen split dataset_root must be a path string")
    dataset_root = Path(dataset_root_raw).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"target output directory is not empty; refusing overwrite: {args.output_dir}"
        )
    if args.manifest.exists():
        raise FileExistsError(f"target manifest already exists; refusing overwrite: {args.manifest}")

    config = TerrainStructureTargetConfig()
    tile_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        tile_id = record["tile_id"]
        print(
            f"[{index}/{len(records)}] generating metric TSD targets for Potsdam {tile_id} ...",
            flush=True,
        )
        result = _generate_tile(
            record,
            dataset_root=dataset_root,
            output_dir=args.output_dir,
            config=config,
        )
        tile_records.append(result)
        print(
            "TARGET PASS: "
            f"{tile_id} accepted_instances={result['accepted_instance_count']} "
            f"rejected_instances={result['rejected_instance_count']} "
            f"valid_pixels={result['valid_pixels']}",
            flush=True,
        )

    manifest = {
        "schema_version": 1,
        "protocol_version": TARGET_PROTOCOL_VERSION,
        "status": "GENERATED_TSD_METRIC_TARGETS",
        "qualification_git_sha": source_sha,
        "qualification_git_branch": source_branch,
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "split_qualification_git_sha": split_payload["qualification_git_sha"],
        "split_protocol_version": split_payload["protocol_version"],
        "campaign_protocol_version": split_payload["campaign_protocol_version"],
        "ground_policy": GROUND_POLICY,
        "target_pack_format": TARGET_PACK_FORMAT,
        "target_config": asdict(config),
        "metric_to_relative_canonicalization_performed": False,
        "claim_boundary": (
            "These artifacts contain metric terrain/above-ground supervision derived only from the "
            "frozen Potsdam training/development reference DSMs and official participant semantic "
            "labels. Every RGB/DSM TIFF, mandatory TFW sidecar, and label is hash-verified against "
            "the frozen split before use. Identity-transform labels are aligned only by the official "
            "same-tile exact-pixel-grid contract. Strict ground means impervious surfaces only. "
            "No DA3 prior/reference scale or offset is fitted here; relative canonicalization remains "
            "an explicit training-time step. Reserved, spatial-buffer, historical challenge-test, "
            "external-evaluation, exposed-corrective, and sealed-blind tiles are not consumed."
        ),
        "reserved_tiles_not_consumed": sorted(RESERVED_TILE_IDS),
        "historical_challenge_test_tiles_not_consumed": sorted(
            HISTORICAL_CHALLENGE_TEST_TILE_IDS
        ),
        "campaign_buffer_tiles_not_consumed": sorted(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS),
        "tiles": tile_records,
    }
    _write_json_atomic(args.manifest, manifest)

    print(f"target_manifest={args.manifest}")
    print(f"target_manifest_sha256={sha256_file(args.manifest)}")
    print(f"qualification_git_sha={source_sha}")
    print(f"protocol_version={TARGET_PROTOCOL_VERSION}")
    print(f"tile_count={len(tile_records)}")
    print(f"train_tile_count={sum(item['role'] == 'train' for item in tile_records)}")
    print(f"dev_tile_count={sum(item['role'] == 'dev' for item in tile_records)}")
    print("ground_policy=strict_impervious_only")
    print("metric_to_relative_canonicalization_performed=false")
    print("sealed_blind_tile_payloads_consumed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
