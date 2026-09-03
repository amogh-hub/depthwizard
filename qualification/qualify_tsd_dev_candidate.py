from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import rasterio
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.evaluation.building_height import (
    BuildingHeightBenchmarkReport,
    BuildingHeightInstance,
    BuildingHeightPromotionDecision,
    building_height_promotion_gate,
    evaluate_building_height_instances,
)
from depthwizard.height_model.terrain_structure import TerrainStructureConfig, TerrainStructureModel
from depthwizard.provenance.manifest import sha256_file

PROTOCOL_VERSION = "potsdam-tsd-dev-qualification-v1"
EXPECTED_TARGET_MANIFEST_SHA256 = (
    "cffe83e74d6a58adf5ea4919f70de35fd0a2b7a3532d1d272a9d2e927f20716c"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "9c786ce22488b530cf07602d52cbcb0481f919eb2ec2cb9b9cd0a44b71c2bb43"
)
EXPECTED_TRAINING_REPORT_SHA256 = (
    "c2bc750d8cd170efbb3c088751dcb82f1ff27e6dc4265571ba14a2b066614664"
)
EXPECTED_BEST_CHECKPOINT_SHA256 = (
    "292d2d1feee176987964391df6a0f3a4a37970dcf05ec143927b0fcd90e84206"
)
EXPECTED_TRAINING_SOURCE_SHA = "eda6cf68ea4233ea99311de45f4e9ebefc03456b"
EXPECTED_DEV_TILE_IDS = ("2_10", "2_11", "2_12", "3_10", "4_10")
FORBIDDEN_TILE_IDS = frozenset({"2_14", "3_14", "4_12", "6_12", "3_13", "6_14"})
NATIVE_GSD_M = 0.05
INFERENCE_PATCH_SIZE = 512
INFERENCE_OVERLAP = 64
MIN_TALL_BUILDINGS = 50
MIN_TALL_MAE_REDUCTION_FRACTION = 0.10
MAX_VALID_RMSE_DEGRADATION_FRACTION = 0.01
MAX_BUILDING_SURFACE_MAE_DEGRADATION_FRACTION = 0.01
MAX_TILE_HEIGHT_MAE_DEGRADATION_FRACTION = 0.10


class QualificationError(RuntimeError):
    pass


class QualificationCheck(TypedDict):
    name: str
    passed: bool
    actual: float
    operator: str
    threshold: float


class TallMetrics(TypedDict):
    count: int
    baseline_mae_m: float
    candidate_mae_m: float
    mae_reduction_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Qualify the frozen epoch-2 TSD candidate against the calibrated DA3 dev baseline."
    )
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _require_source_identity(root: Path) -> tuple[str, str]:
    branch = _git_output(root, "branch", "--show-current")
    if branch != "engineering/terrain-structure-vnext":
        raise QualificationError(f"wrong branch for TSD qualification: {branch}")
    dirty = _git_output(root, "status", "--short", "--untracked-files=no")
    if dirty:
        raise QualificationError("refusing TSD qualification from a tracked-dirty worktree")
    head = _git_output(root, "rev-parse", "HEAD")
    if len(head) != 40:
        raise QualificationError("could not resolve exact qualification source SHA")
    return head, branch


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _fractional_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= baseline + 1e-12 else -float("inf")
    return float((baseline - candidate) / baseline)


def _window_starts(length: int, size: int, overlap: int) -> tuple[int, ...]:
    if length < size:
        raise ValueError("inference patch exceeds scene dimension")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch size")
    stride = size - overlap
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if not starts or starts[-1] != final:
        starts.append(final)
    return tuple(starts)


def _blend_weight(size: int) -> np.ndarray:
    if size < 2:
        raise ValueError("blend size must be at least 2")
    axis = np.hanning(size).astype(np.float32)
    axis = np.maximum(axis, np.float32(0.05))
    return np.outer(axis, axis).astype(np.float32)


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled(np.nan), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _read_target_pack(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as pack:
        required = {
            "terrain_m",
            "above_ground_m",
            "valid_mask",
            "building_mask",
            "ground_mask",
            "boundary_mask",
        }
        if set(pack.files) != required:
            raise QualificationError(f"unexpected target-pack arrays for {path}")
        return {
            "terrain_m": np.asarray(pack["terrain_m"], dtype=np.float32),
            "above_ground_m": np.asarray(pack["above_ground_m"], dtype=np.float32),
            "valid_mask": np.asarray(pack["valid_mask"], dtype=bool),
            "building_mask": np.asarray(pack["building_mask"], dtype=bool),
            "ground_mask": np.asarray(pack["ground_mask"], dtype=bool),
        }


def _write_prediction(path: Path, values: np.ndarray, *, template_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(template_path) as template:
        profile = template.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            dtype="float32",
            nodata=np.nan,
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(np.asarray(values, dtype=np.float32), 1)
            dst.update_tags(
                DEPTHWIZARD_PRODUCT="TSD_DEV_RELATIVE_DSM_RESEARCH_CANDIDATE",
                PROTOCOL_VERSION=PROTOCOL_VERSION,
                PRODUCTION_PROMOTED="false",
            )
    temporary.replace(path)


def _load_model(checkpoint_path: Path, device: torch.device) -> TerrainStructureModel:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("TSD checkpoint must contain a mapping")
    if payload.get("protocol_version") != "potsdam-tsd-urban-train-v1":
        raise QualificationError("unexpected TSD checkpoint protocol")
    if payload.get("training_source_git_sha") != EXPECTED_TRAINING_SOURCE_SHA:
        raise QualificationError("TSD checkpoint source identity mismatch")
    epoch = payload.get("epoch")
    if not isinstance(epoch, int) or epoch != 2:
        raise QualificationError("frozen TSD candidate is not the epoch-2 checkpoint")
    if payload.get("target_manifest_sha256") != EXPECTED_TARGET_MANIFEST_SHA256:
        raise QualificationError("TSD checkpoint target identity mismatch")
    if payload.get("split_manifest_sha256") != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise QualificationError("TSD checkpoint split identity mismatch")
    raw_config = payload.get("model_config")
    if not isinstance(raw_config, dict):
        raise TypeError("TSD checkpoint model_config must be an object")
    config = TerrainStructureConfig(**raw_config)
    model = TerrainStructureModel(config).to(device)
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise TypeError("TSD checkpoint model_state_dict must be an object")
    model.load_state_dict(state)
    model.eval()
    return model


def _prediction_metadata_matches(
    metadata_path: Path,
    prediction_path: Path,
    *,
    tile_id: str,
    rgb_sha256: str,
    geometry_sha256: str,
) -> bool:
    if prediction_path.is_file() != metadata_path.is_file():
        raise QualificationError(f"incomplete cached qualification prediction for {tile_id}")
    if not prediction_path.is_file():
        return False
    metadata = _load_json(metadata_path)
    required: dict[str, object] = {
        "protocol_version": PROTOCOL_VERSION,
        "tile_id": tile_id,
        "rgb_sha256": rgb_sha256,
        "geometry_sha256": geometry_sha256,
        "checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
        "patch_size": INFERENCE_PATCH_SIZE,
        "overlap": INFERENCE_OVERLAP,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise QualificationError(f"cached TSD dev prediction metadata mismatch: {tile_id}: {key}")
    if metadata.get("prediction_sha256") != sha256_file(prediction_path):
        raise QualificationError(f"cached TSD dev prediction hash mismatch for {tile_id}")
    return True


def _predict_relative_scene(
    *,
    tile_id: str,
    rgb_path: Path,
    rgb_sha256: str,
    geometry_path: Path,
    prediction_path: Path,
    metadata_path: Path,
    model: TerrainStructureModel,
    device: torch.device,
    qualification_source_sha: str,
) -> np.ndarray:
    geometry_sha256 = sha256_file(geometry_path)
    if _prediction_metadata_matches(
        metadata_path,
        prediction_path,
        tile_id=tile_id,
        rgb_sha256=rgb_sha256,
        geometry_sha256=geometry_sha256,
    ):
        print(f"dev_prediction reuse Potsdam {tile_id}", flush=True)
        return _read_band(prediction_path)

    with rasterio.open(rgb_path) as rgb_src, rasterio.open(geometry_path) as geometry_src:
        if rgb_src.count < 3:
            raise QualificationError(f"RGB has fewer than three bands: {tile_id}")
        if (rgb_src.height, rgb_src.width) != (geometry_src.height, geometry_src.width):
            raise QualificationError(f"RGB/geometry shape mismatch: {tile_id}")
        if not rgb_src.transform.almost_equals(geometry_src.transform):
            raise QualificationError(f"RGB/geometry transform mismatch: {tile_id}")
        height = int(rgb_src.height)
        width = int(rgb_src.width)
        row_starts = _window_starts(height, INFERENCE_PATCH_SIZE, INFERENCE_OVERLAP)
        col_starts = _window_starts(width, INFERENCE_PATCH_SIZE, INFERENCE_OVERLAP)
        blend = _blend_weight(INFERENCE_PATCH_SIZE)
        accumulated = np.zeros((height, width), dtype=np.float32)
        weights = np.zeros((height, width), dtype=np.float32)
        gsd = torch.tensor([NATIVE_GSD_M], dtype=torch.float32, device=device)

        total = len(row_starts) * len(col_starts)
        completed = 0
        print(f"dev_prediction Potsdam {tile_id} patches={total} ...", flush=True)
        with torch.inference_mode():
            for row in row_starts:
                for col in col_starts:
                    window = (
                        (row, row + INFERENCE_PATCH_SIZE),
                        (col, col + INFERENCE_PATCH_SIZE),
                    )
                    rgb_np = rgb_src.read((1, 2, 3), window=window).astype(np.float32)
                    if float(np.nanmax(rgb_np)) > 1.0:
                        rgb_np /= 255.0
                    rgb_np = np.clip(rgb_np, 0.0, 1.0)
                    geometry_np = geometry_src.read(1, window=window).astype(np.float32)
                    if not np.all(np.isfinite(geometry_np)):
                        raise QualificationError(f"non-finite frozen DA3 geometry in {tile_id}")
                    rgb_tensor = torch.from_numpy(rgb_np[None]).to(device=device)
                    geometry_tensor = torch.from_numpy(geometry_np[None, None]).to(device=device)
                    output = model(rgb_tensor, geometry_tensor, gsd_m=gsd)
                    relative = np.asarray(
                        output.relative_height[0, 0].detach().cpu(), dtype=np.float32
                    )
                    rows = slice(row, row + INFERENCE_PATCH_SIZE)
                    cols = slice(col, col + INFERENCE_PATCH_SIZE)
                    accumulated[rows, cols] += relative * blend
                    weights[rows, cols] += blend
                    completed += 1
                    if completed % 50 == 0 or completed == total:
                        print(f"dev_prediction {tile_id} patch={completed}/{total}", flush=True)

    if np.any(weights <= 0.0):
        raise QualificationError(f"uncovered TSD inference pixels for {tile_id}")
    prediction = accumulated / weights
    if not np.all(np.isfinite(prediction)):
        raise QualificationError(f"non-finite TSD dev prediction for {tile_id}")
    _write_prediction(prediction_path, prediction, template_path=geometry_path)
    metadata = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "tile_id": tile_id,
        "rgb": str(rgb_path),
        "rgb_sha256": rgb_sha256,
        "geometry": str(geometry_path),
        "geometry_sha256": geometry_sha256,
        "prediction": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
        "patch_size": INFERENCE_PATCH_SIZE,
        "overlap": INFERENCE_OVERLAP,
        "qualification_source_git_sha": qualification_source_sha,
        "metric_calibration_performed_by_model": False,
        "production_promoted": False,
        "sealed_blind_tile_payloads_consumed": False,
    }
    _write_json_atomic(metadata_path, metadata)
    print(f"dev_prediction PASS {tile_id} sha256={metadata['prediction_sha256']}", flush=True)
    return prediction


def _pixel_metrics(
    prediction: np.ndarray, reference: np.ndarray, mask: np.ndarray
) -> dict[str, float | int]:
    support = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(reference)
    count = int(np.count_nonzero(support))
    if count <= 0:
        raise QualificationError("pixel metric has no valid support")
    error = np.asarray(prediction[support] - reference[support], dtype=np.float64)
    return {
        "pixels": count,
        "mae_m": float(np.mean(np.abs(error))),
        "rmse_m": float(np.sqrt(np.mean(error**2))),
        "bias_m": float(np.mean(error)),
    }


def _aggregate_building_reports(
    reports: list[tuple[str, BuildingHeightBenchmarkReport]],
) -> BuildingHeightBenchmarkReport:
    eligible: list[int] = []
    evaluated: list[int] = []
    failures: list[int] = []
    instances: list[BuildingHeightInstance] = []
    skipped = 0
    for tile_index, (_, report) in enumerate(reports, start=1):
        offset = tile_index * 100_000
        eligible.extend(offset + value for value in report.eligible_instance_ids)
        evaluated.extend(offset + value for value in report.evaluated_instance_ids)
        failures.extend(offset + value for value in report.prediction_failure_ids)
        instances.extend(replace(item, instance_id=offset + item.instance_id) for item in report.instances)
        skipped += report.skipped_reference_instances
    if not instances:
        raise QualificationError("no aggregate building instances available")
    height_error = np.asarray([item.height_error_m for item in instances], dtype=np.float64)
    height_abs = np.abs(height_error)
    top_abs = np.abs(np.asarray([item.top_error_m for item in instances], dtype=np.float64))
    ground_abs = np.abs(np.asarray([item.ground_error_m for item in instances], dtype=np.float64))
    reference_heights = np.asarray([item.reference_height_m for item in instances], dtype=np.float64)
    predicted_heights = np.asarray([item.predicted_height_m for item in instances], dtype=np.float64)
    return BuildingHeightBenchmarkReport(
        eligible_instance_ids=tuple(eligible),
        evaluated_instance_ids=tuple(evaluated),
        prediction_failure_ids=tuple(failures),
        skipped_reference_instances=skipped,
        instances=tuple(instances),
        height_mae_m=float(np.mean(height_abs)),
        height_rmse_m=float(np.sqrt(np.mean(height_error**2))),
        height_bias_m=float(np.mean(height_error)),
        height_median_abs_error_m=float(np.median(height_abs)),
        height_p90_abs_error_m=float(np.percentile(height_abs, 90)),
        height_p95_abs_error_m=float(np.percentile(height_abs, 95)),
        top_mae_m=float(np.mean(top_abs)),
        ground_mae_m=float(np.mean(ground_abs)),
        within_1m_fraction=float(np.mean(height_abs <= 1.0)),
        within_2m_fraction=float(np.mean(height_abs <= 2.0)),
        catastrophic_over_3m_fraction=float(np.mean(height_abs > 3.0)),
        mean_reference_height_m=float(np.mean(reference_heights)),
        mean_predicted_height_m=float(np.mean(predicted_heights)),
    )


def _tall_mae(
    report: BuildingHeightBenchmarkReport, minimum_height_m: float = 8.0
) -> tuple[int, float]:
    selected = [
        abs(item.height_error_m)
        for item in report.instances
        if item.reference_height_m >= minimum_height_m
    ]
    if not selected:
        raise QualificationError("no tall buildings available for TSD dev qualification")
    return len(selected), float(np.mean(np.asarray(selected, dtype=np.float64)))


def _check(
    name: str, passed: bool, actual: float, operator: str, threshold: float
) -> QualificationCheck:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": float(actual),
        "operator": operator,
        "threshold": float(threshold),
    }


def _metric_value(metrics: dict[str, float | int], key: str) -> float:
    return float(metrics[key])


def _qualification_checks(
    *,
    baseline_buildings: BuildingHeightBenchmarkReport,
    candidate_buildings: BuildingHeightBenchmarkReport,
    baseline_valid: dict[str, float | int],
    candidate_valid: dict[str, float | int],
    baseline_building_surface: dict[str, float | int],
    candidate_building_surface: dict[str, float | int],
    per_tile_mae_degradation: list[float],
) -> tuple[list[QualificationCheck], TallMetrics, BuildingHeightPromotionDecision]:
    promotion = building_height_promotion_gate(baseline_buildings, candidate_buildings)
    tall_count_baseline, tall_mae_baseline = _tall_mae(baseline_buildings)
    tall_count_candidate, tall_mae_candidate = _tall_mae(candidate_buildings)
    if tall_count_baseline != tall_count_candidate:
        raise QualificationError("candidate and baseline tall-building populations differ")
    tall_reduction = _fractional_reduction(tall_mae_baseline, tall_mae_candidate)
    valid_rmse_ratio = _metric_value(candidate_valid, "rmse_m") / _metric_value(
        baseline_valid, "rmse_m"
    )
    building_surface_mae_ratio = _metric_value(
        candidate_building_surface, "mae_m"
    ) / _metric_value(baseline_building_surface, "mae_m")
    max_tile_degradation = max(per_tile_mae_degradation) if per_tile_mae_degradation else float("inf")
    checks: list[QualificationCheck] = [
        _check(
            "building_height_promotion_gate",
            promotion.passed,
            1.0 if promotion.passed else 0.0,
            "==",
            1.0,
        ),
        _check(
            "tall_building_count",
            tall_count_candidate >= MIN_TALL_BUILDINGS,
            float(tall_count_candidate),
            ">=",
            float(MIN_TALL_BUILDINGS),
        ),
        _check(
            "tall_building_mae_reduction",
            tall_reduction >= MIN_TALL_MAE_REDUCTION_FRACTION,
            tall_reduction,
            ">=",
            MIN_TALL_MAE_REDUCTION_FRACTION,
        ),
        _check(
            "valid_dsm_rmse_ratio",
            valid_rmse_ratio <= 1.0 + MAX_VALID_RMSE_DEGRADATION_FRACTION,
            valid_rmse_ratio,
            "<=",
            1.0 + MAX_VALID_RMSE_DEGRADATION_FRACTION,
        ),
        _check(
            "building_surface_mae_ratio",
            building_surface_mae_ratio <= 1.0 + MAX_BUILDING_SURFACE_MAE_DEGRADATION_FRACTION,
            building_surface_mae_ratio,
            "<=",
            1.0 + MAX_BUILDING_SURFACE_MAE_DEGRADATION_FRACTION,
        ),
        _check(
            "max_tile_height_mae_degradation",
            max_tile_degradation <= MAX_TILE_HEIGHT_MAE_DEGRADATION_FRACTION,
            max_tile_degradation,
            "<=",
            MAX_TILE_HEIGHT_MAE_DEGRADATION_FRACTION,
        ),
    ]
    tall: TallMetrics = {
        "count": tall_count_candidate,
        "baseline_mae_m": tall_mae_baseline,
        "candidate_mae_m": tall_mae_candidate,
        "mae_reduction_fraction": tall_reduction,
    }
    return checks, tall, promotion


def main() -> int:
    args = parse_args()
    root = _repo_root()
    source_sha, source_branch = _require_source_identity(root)
    target_manifest_path = args.target_manifest.resolve()
    training_dir = args.training_dir.resolve()
    output_path = args.output.resolve()

    if sha256_file(target_manifest_path) != EXPECTED_TARGET_MANIFEST_SHA256:
        raise QualificationError("frozen target manifest identity mismatch")
    training_report_path = training_dir / "training_report.json"
    checkpoint_path = training_dir / "tsd_potsdam_v1_best.pt"
    if sha256_file(training_report_path) != EXPECTED_TRAINING_REPORT_SHA256:
        raise QualificationError("frozen TSD training report identity mismatch")
    if sha256_file(checkpoint_path) != EXPECTED_BEST_CHECKPOINT_SHA256:
        raise QualificationError("frozen TSD best checkpoint identity mismatch")

    training_report = _load_json(training_report_path)
    if training_report.get("status") != "TRAINED_TSD_CANDIDATE_NOT_PROMOTED":
        raise QualificationError("TSD training report is not in frozen candidate state")
    if training_report.get("best_epoch") != 2:
        raise QualificationError("TSD frozen candidate best epoch changed")
    if training_report.get("training_source_git_sha") != EXPECTED_TRAINING_SOURCE_SHA:
        raise QualificationError("TSD training report source SHA changed")
    if training_report.get("production_promoted") is not False:
        raise QualificationError("TSD training report unexpectedly claims production promotion")

    if output_path.is_file():
        prior = _load_json(output_path)
        if prior.get("checkpoint_sha256") != EXPECTED_BEST_CHECKPOINT_SHA256:
            raise QualificationError("existing dev qualification belongs to another checkpoint")
        if prior.get("target_manifest_sha256") != EXPECTED_TARGET_MANIFEST_SHA256:
            raise QualificationError("existing dev qualification belongs to another target manifest")
        print("TSD_DEV_QUALIFICATION=ALREADY_COMPLETE")
        print(f"qualification_passed={str(bool(prior.get('qualification_passed'))).lower()}")
        print(f"qualification_report={output_path}")
        print(f"qualification_report_sha256={sha256_file(output_path)}")
        print("production_promoted=false")
        print("sealed_blind_tile_payloads_consumed=false")
        return 0

    target_manifest = _load_json(target_manifest_path)
    split_raw = target_manifest.get("split_manifest")
    if not isinstance(split_raw, str):
        raise TypeError("target manifest split_manifest must be a string path")
    split_path = Path(split_raw).resolve()
    if sha256_file(split_path) != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise QualificationError("frozen split identity mismatch")
    split = _load_json(split_path)
    if tuple(split.get("dev_tile_ids", ())) != EXPECTED_DEV_TILE_IDS:
        raise QualificationError("frozen dev tile population changed")
    if not set(EXPECTED_DEV_TILE_IDS).isdisjoint(FORBIDDEN_TILE_IDS):
        raise QualificationError("forbidden tile leaked into dev constants")

    target_tiles_raw = target_manifest.get("tiles")
    split_tiles_raw = split.get("tiles")
    if not isinstance(target_tiles_raw, list) or not isinstance(split_tiles_raw, list):
        raise TypeError("target/split tile records must be lists")
    target_by_id = {
        str(item["tile_id"]): item for item in target_tiles_raw if isinstance(item, dict)
    }
    split_by_id = {
        str(item["tile_id"]): item for item in split_tiles_raw if isinstance(item, dict)
    }

    device = _resolve_device()
    model = _load_model(checkpoint_path, device)
    print("===== FROZEN TSD DEV QUALIFICATION =====")
    print(f"device={device}")
    print(f"qualification_source_git_sha={source_sha}")
    print(f"training_source_git_sha={EXPECTED_TRAINING_SOURCE_SHA}")
    print(f"checkpoint_sha256={EXPECTED_BEST_CHECKPOINT_SHA256}")
    print(f"training_report_sha256={EXPECTED_TRAINING_REPORT_SHA256}")
    print(f"target_manifest_sha256={EXPECTED_TARGET_MANIFEST_SHA256}")
    print(f"dev_tiles={','.join(EXPECTED_DEV_TILE_IDS)}")
    print("sealed_blind_tile_payloads_consumed=false")

    baseline_reports: list[tuple[str, BuildingHeightBenchmarkReport]] = []
    candidate_reports: list[tuple[str, BuildingHeightBenchmarkReport]] = []
    tile_results: list[dict[str, object]] = []
    baseline_valid_sums: list[tuple[int, float, float]] = []
    candidate_valid_sums: list[tuple[int, float, float]] = []
    baseline_building_sums: list[tuple[int, float, float]] = []
    candidate_building_sums: list[tuple[int, float, float]] = []
    per_tile_degradation: list[float] = []

    prediction_dir = training_dir / "dev-qualification" / "predictions"
    for index, tile_id in enumerate(EXPECTED_DEV_TILE_IDS, start=1):
        target_item = target_by_id.get(tile_id)
        split_item = split_by_id.get(tile_id)
        if not isinstance(target_item, dict) or not isinstance(split_item, dict):
            raise QualificationError(f"missing frozen dev record for {tile_id}")
        if target_item.get("role") != "dev" or split_item.get("role") != "dev":
            raise QualificationError(f"role mismatch for frozen dev tile {tile_id}")
        rgb_path = Path(str(split_item["rgb"])).resolve()
        target_pack_path = Path(str(target_item["target_pack"])).resolve()
        if sha256_file(rgb_path) != str(split_item["rgb_sha256"]):
            raise QualificationError(f"RGB identity changed for {tile_id}")
        if sha256_file(target_pack_path) != str(target_item["target_pack_sha256"]):
            raise QualificationError(f"target-pack identity changed for {tile_id}")
        geometry_path = training_dir / "geometry" / f"potsdam_{tile_id}_da3_relative.tif"
        fit_path = training_dir / "scene-fits" / f"potsdam_{tile_id}_prior_reference_fit.json"
        geometry_meta_path = training_dir / "geometry" / f"potsdam_{tile_id}_da3_relative.json"
        geometry_meta = _load_json(geometry_meta_path)
        geometry_sha = sha256_file(geometry_path)
        if geometry_meta.get("geometry_sha256") != geometry_sha:
            raise QualificationError(f"DA3 geometry identity changed for {tile_id}")
        fit = _load_json(fit_path)
        if fit.get("geometry_sha256") != geometry_sha:
            raise QualificationError(f"scene fit geometry identity mismatch for {tile_id}")
        if fit.get("target_pack_sha256") != str(target_item["target_pack_sha256"]):
            raise QualificationError(f"scene fit target identity mismatch for {tile_id}")
        scale = float(fit["scale_m_per_prior_unit"])
        offset = float(fit["offset_m"])
        if not math.isfinite(scale) or scale <= 0.0 or not math.isfinite(offset):
            raise QualificationError(f"invalid frozen scene fit for {tile_id}")

        prediction_path = prediction_dir / f"potsdam_{tile_id}_tsd_relative.tif"
        metadata_path = prediction_dir / f"potsdam_{tile_id}_tsd_relative.json"
        candidate_relative = _predict_relative_scene(
            tile_id=tile_id,
            rgb_path=rgb_path,
            rgb_sha256=str(split_item["rgb_sha256"]),
            geometry_path=geometry_path,
            prediction_path=prediction_path,
            metadata_path=metadata_path,
            model=model,
            device=device,
            qualification_source_sha=source_sha,
        )
        geometry_relative = _read_band(geometry_path)
        pack = _read_target_pack(target_pack_path)
        reference_dsm = pack["terrain_m"] + pack["above_ground_m"]
        valid = pack["valid_mask"] & np.isfinite(reference_dsm) & np.isfinite(geometry_relative)
        building = pack["building_mask"] & valid
        ground = pack["ground_mask"] & valid
        baseline_dsm = geometry_relative * scale + offset
        candidate_dsm = candidate_relative * scale + offset

        baseline_buildings = evaluate_building_height_instances(
            baseline_dsm,
            reference_dsm,
            building,
            gsd_x_m=NATIVE_GSD_M,
            gsd_y_m=NATIVE_GSD_M,
            ground_candidate_mask=ground,
        )
        candidate_buildings = evaluate_building_height_instances(
            candidate_dsm,
            reference_dsm,
            building,
            gsd_x_m=NATIVE_GSD_M,
            gsd_y_m=NATIVE_GSD_M,
            ground_candidate_mask=ground,
        )
        baseline_reports.append((tile_id, baseline_buildings))
        candidate_reports.append((tile_id, candidate_buildings))
        baseline_valid = _pixel_metrics(baseline_dsm, reference_dsm, valid)
        candidate_valid = _pixel_metrics(candidate_dsm, reference_dsm, valid)
        baseline_building = _pixel_metrics(baseline_dsm, reference_dsm, building)
        candidate_building = _pixel_metrics(candidate_dsm, reference_dsm, building)
        baseline_ground = _pixel_metrics(baseline_dsm, reference_dsm, ground)
        candidate_ground = _pixel_metrics(candidate_dsm, reference_dsm, ground)

        baseline_valid_sums.append(
            (
                int(baseline_valid["pixels"]),
                float(baseline_valid["mae_m"]),
                float(baseline_valid["rmse_m"]),
            )
        )
        candidate_valid_sums.append(
            (
                int(candidate_valid["pixels"]),
                float(candidate_valid["mae_m"]),
                float(candidate_valid["rmse_m"]),
            )
        )
        baseline_building_sums.append(
            (
                int(baseline_building["pixels"]),
                float(baseline_building["mae_m"]),
                float(baseline_building["rmse_m"]),
            )
        )
        candidate_building_sums.append(
            (
                int(candidate_building["pixels"]),
                float(candidate_building["mae_m"]),
                float(candidate_building["rmse_m"]),
            )
        )
        tile_degradation = (
            candidate_buildings.height_mae_m / baseline_buildings.height_mae_m - 1.0
            if baseline_buildings.height_mae_m > 1e-12
            else float("inf")
        )
        per_tile_degradation.append(float(tile_degradation))
        tile_results.append(
            {
                "tile_id": tile_id,
                "baseline_building_height": asdict(baseline_buildings),
                "candidate_building_height": asdict(candidate_buildings),
                "baseline_valid_dsm": baseline_valid,
                "candidate_valid_dsm": candidate_valid,
                "baseline_building_surface": baseline_building,
                "candidate_building_surface": candidate_building,
                "baseline_strict_ground_surface": baseline_ground,
                "candidate_strict_ground_surface": candidate_ground,
                "height_mae_degradation_fraction": tile_degradation,
                "candidate_prediction": str(prediction_path),
                "candidate_prediction_sha256": sha256_file(prediction_path),
            }
        )
        print(
            f"dev[{index}/5] tile={tile_id} "
            f"building_mae baseline={baseline_buildings.height_mae_m:.4f}m "
            f"candidate={candidate_buildings.height_mae_m:.4f}m "
            f"valid_rmse baseline={_metric_value(baseline_valid, 'rmse_m'):.4f}m "
            f"candidate={_metric_value(candidate_valid, 'rmse_m'):.4f}m",
            flush=True,
        )

    def combine_pixel_metrics(items: list[tuple[int, float, float]]) -> dict[str, float | int]:
        total = sum(count for count, _, _ in items)
        if total <= 0:
            raise QualificationError("cannot combine empty pixel metrics")
        mae = sum(count * value for count, value, _ in items) / total
        mse = sum(count * (rmse**2) for count, _, rmse in items) / total
        return {"pixels": total, "mae_m": float(mae), "rmse_m": float(math.sqrt(mse))}

    baseline_aggregate = _aggregate_building_reports(baseline_reports)
    candidate_aggregate = _aggregate_building_reports(candidate_reports)
    baseline_valid_aggregate = combine_pixel_metrics(baseline_valid_sums)
    candidate_valid_aggregate = combine_pixel_metrics(candidate_valid_sums)
    baseline_building_aggregate = combine_pixel_metrics(baseline_building_sums)
    candidate_building_aggregate = combine_pixel_metrics(candidate_building_sums)
    checks, tall, promotion = _qualification_checks(
        baseline_buildings=baseline_aggregate,
        candidate_buildings=candidate_aggregate,
        baseline_valid=baseline_valid_aggregate,
        candidate_valid=candidate_valid_aggregate,
        baseline_building_surface=baseline_building_aggregate,
        candidate_building_surface=candidate_building_aggregate,
        per_tile_mae_degradation=per_tile_degradation,
    )
    qualification_passed = all(item["passed"] for item in checks)
    report = {
        "schema_version": 1,
        "status": "TSD_DEV_QUALIFICATION_PASS" if qualification_passed else "TSD_DEV_QUALIFICATION_FAIL",
        "protocol_version": PROTOCOL_VERSION,
        "qualification_source_git_sha": source_sha,
        "qualification_source_git_branch": source_branch,
        "training_source_git_sha": EXPECTED_TRAINING_SOURCE_SHA,
        "training_report": str(training_report_path),
        "training_report_sha256": EXPECTED_TRAINING_REPORT_SHA256,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": EXPECTED_BEST_CHECKPOINT_SHA256,
        "best_epoch": 2,
        "target_manifest": str(target_manifest_path),
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "split_manifest": str(split_path),
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "dev_tile_ids": list(EXPECTED_DEV_TILE_IDS),
        "baseline": {
            "building_height": asdict(baseline_aggregate),
            "valid_dsm": baseline_valid_aggregate,
            "building_surface": baseline_building_aggregate,
        },
        "candidate": {
            "building_height": asdict(candidate_aggregate),
            "valid_dsm": candidate_valid_aggregate,
            "building_surface": candidate_building_aggregate,
        },
        "promotion_gate": asdict(promotion),
        "tall_buildings_ge_8m": tall,
        "checks": checks,
        "tiles": tile_results,
        "qualification_passed": qualification_passed,
        "production_promoted": False,
        "exposed_corrective_2_14_consumed": False,
        "external_evaluation_3_14_consumed": False,
        "sealed_blind_tile_payloads_consumed": False,
        "claim_boundary": (
            "Development qualification of the frozen epoch-2 TSD research candidate against the "
            "calibrated DA3 baseline on only the five predeclared frozen dev scenes. The same "
            "reference-derived scene fit is applied to baseline and candidate solely for apples-to-apples "
            "development evaluation. This is not independent metric calibration, exposed-corrective "
            "evaluation, external evaluation, sealed-blind evidence, or production promotion."
        ),
    }
    _write_json_atomic(output_path, report)

    print()
    print("===== TSD DEV QUALIFICATION DECISION =====")
    print(
        f"building_height_mae baseline={baseline_aggregate.height_mae_m:.4f}m "
        f"candidate={candidate_aggregate.height_mae_m:.4f}m "
        f"reduction={_fractional_reduction(baseline_aggregate.height_mae_m, candidate_aggregate.height_mae_m):.2%}"
    )
    print(
        f"building_height_rmse baseline={baseline_aggregate.height_rmse_m:.4f}m "
        f"candidate={candidate_aggregate.height_rmse_m:.4f}m "
        f"reduction={_fractional_reduction(baseline_aggregate.height_rmse_m, candidate_aggregate.height_rmse_m):.2%}"
    )
    print(
        f"tall_ge_8m_mae baseline={tall['baseline_mae_m']:.4f}m "
        f"candidate={tall['candidate_mae_m']:.4f}m "
        f"reduction={tall['mae_reduction_fraction']:.2%} n={tall['count']}"
    )
    print(
        f"valid_dsm_rmse baseline={_metric_value(baseline_valid_aggregate, 'rmse_m'):.4f}m "
        f"candidate={_metric_value(candidate_valid_aggregate, 'rmse_m'):.4f}m"
    )
    for item in checks:
        print(
            f"{'PASS' if item['passed'] else 'FAIL'}: {item['name']} "
            f"actual={item['actual']:.6f} required {item['operator']} {item['threshold']:.6f}"
        )
    print(f"qualification_passed={str(qualification_passed).lower()}")
    print(f"qualification_report={output_path}")
    print(f"qualification_report_sha256={hashlib.sha256(output_path.read_bytes()).hexdigest()}")
    print("production_promoted=false")
    print("exposed_corrective_2_14_consumed=false")
    print("external_evaluation_3_14_consumed=false")
    print("sealed_blind_tile_payloads_consumed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
