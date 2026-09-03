from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from depthwizard.geometry_prior.da3 import (
    DA3_CHECKPOINT_SHA256,
    DA3_HF_REVISION,
    DA3_MODEL_ID,
    DA3_MODEL_SOURCE,
    DA3MonocularPrior,
)
from depthwizard.height_model.terrain_structure import (
    TerrainStructureConfig,
    TerrainStructureModel,
)
from depthwizard.height_model.terrain_structure_campaign import (
    assess_tsd_training_authorization,
)
from depthwizard.height_model.terrain_structure_loss import (
    TerrainStructureTargets,
    compute_terrain_structure_loss,
)
from depthwizard.height_model.terrain_structure_targets import (
    MetricTerrainStructureTargets,
    RelativeTerrainStructureTargets,
    canonicalize_metric_terrain_structure_targets,
)
from depthwizard.height_model.training import PriorReferenceFit, fit_reference_to_prior
from depthwizard.io.raster import write_float_geotiff
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.provenance.manifest import sha256_file

PROTOCOL_VERSION = "potsdam-tsd-urban-train-v1"
EXPECTED_TARGET_MANIFEST_SHA256 = (
    "cffe83e74d6a58adf5ea4919f70de35fd0a2b7a3532d1d272a9d2e927f20716c"
)
EXPECTED_SPLIT_MANIFEST_SHA256 = (
    "9c786ce22488b530cf07602d52cbcb0481f919eb2ec2cb9b9cd0a44b71c2bb43"
)
EXPECTED_TRAIN_TILE_IDS = ("6_7", "6_8", "6_9", "6_10", "7_7", "7_8", "7_9", "7_10")
EXPECTED_DEV_TILE_IDS = ("2_10", "2_11", "2_12", "3_10", "4_10")
FORBIDDEN_TILE_IDS = frozenset({"2_14", "3_14", "4_12", "6_12", "3_13", "6_14"})

SEED = 26175
EPOCHS = 12
PATCH_SIZE = 384
TRAIN_PATCHES_PER_TILE_PER_EPOCH = 24
DEV_PATCHES_PER_TILE = 12
LEARNING_RATE = 2.0e-4
WEIGHT_DECAY = 1.0e-4
GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 4
MIN_EPOCHS_BEFORE_EARLY_STOP = 6
NATIVE_GSD_M = 0.05
MIN_FREE_BYTES = 4 * 1024**3


@dataclass(frozen=True)
class SceneRecord:
    tile_id: str
    role: str
    rgb_path: Path
    rgb_sha256: str
    target_pack: Path
    target_pack_sha256: str
    geometry_path: Path
    geometry_metadata_path: Path


@dataclass(frozen=True)
class LoadedScene:
    record: SceneRecord
    rgb: np.ndarray
    geometry: np.ndarray
    targets: RelativeTerrainStructureTargets
    metric_agl_m: np.ndarray
    fit: PriorReferenceFit


@dataclass(frozen=True)
class PatchWindow:
    row: int
    col: int
    size: int
    sampling_class: str

    @property
    def row_slice(self) -> slice:
        return slice(self.row, self.row + self.size)

    @property
    def col_slice(self) -> slice:
        return slice(self.col, self.col + self.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the frozen first Potsdam TSD campaign.")
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


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


def _torch_save_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _require_source_identity() -> tuple[str, str]:
    branch = _git_output("branch", "--show-current")
    if branch != "engineering/terrain-structure-vnext":
        raise RuntimeError(f"wrong branch for TSD training: {branch}")
    dirty = _git_output("status", "--short", "--untracked-files=no")
    if dirty:
        raise RuntimeError("refusing TSD training from a tracked-dirty worktree")
    head = _git_output("rev-parse", "HEAD")
    if len(head) != 40:
        raise RuntimeError("could not resolve exact training source commit")
    return head, branch


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _require_free_space(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < MIN_FREE_BYTES:
        raise RuntimeError(
            f"TSD training requires at least {MIN_FREE_BYTES / 1024**3:.1f} GiB free; "
            f"found {free / 1024**3:.1f} GiB"
        )


def _validate_authorization(
    target_manifest: Path,
    audit_path: Path,
    authorization_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if sha256_file(target_manifest) != EXPECTED_TARGET_MANIFEST_SHA256:
        raise RuntimeError("TSD trainer target-manifest identity mismatch")
    target_payload = _load_json(target_manifest)
    audit = _load_json(audit_path)
    authorization = _load_json(authorization_path)
    if authorization.get("target_manifest_sha256") != EXPECTED_TARGET_MANIFEST_SHA256:
        raise RuntimeError("TSD authorization target identity mismatch")
    if authorization.get("audit_sha256") != sha256_file(audit_path):
        raise RuntimeError("TSD authorization audit identity mismatch")
    recomputed = assess_tsd_training_authorization(audit)
    if recomputed.get("training_authorized") is not True:
        raise RuntimeError("current TSD audit does not satisfy the frozen training gate")
    if authorization.get("training_authorized") is not True:
        raise RuntimeError("TSD training has not been authorized")
    if authorization.get("gate") != recomputed.get("gate"):
        raise RuntimeError("TSD authorization gate differs from the committed gate")
    return target_payload, audit, authorization


def _split_manifest_from_target(target_payload: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    split_raw = target_payload.get("split_manifest")
    if not isinstance(split_raw, str):
        raise TypeError("target manifest split_manifest must be a path string")
    split_path = Path(split_raw).resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    actual = sha256_file(split_path)
    if actual != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError(
            f"frozen split manifest changed: expected {EXPECTED_SPLIT_MANIFEST_SHA256}, got {actual}"
        )
    if target_payload.get("split_manifest_sha256") != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise RuntimeError("target manifest is bound to a different split identity")
    split = _load_json(split_path)
    if tuple(split.get("train_tile_ids", ())) != EXPECTED_TRAIN_TILE_IDS:
        raise RuntimeError("TSD training tile population changed")
    if tuple(split.get("dev_tile_ids", ())) != EXPECTED_DEV_TILE_IDS:
        raise RuntimeError("TSD dev tile population changed")
    return split_path, split


def _build_scene_records(
    target_payload: dict[str, Any],
    split: dict[str, Any],
    output_dir: Path,
) -> tuple[list[SceneRecord], list[SceneRecord]]:
    split_tiles_raw = split.get("tiles")
    target_tiles_raw = target_payload.get("tiles")
    if not isinstance(split_tiles_raw, list) or not isinstance(target_tiles_raw, list):
        raise TypeError("TSD split/target tiles must be lists")
    split_by_id = {
        str(item["tile_id"]): item for item in split_tiles_raw if isinstance(item, dict)
    }
    target_by_id = {
        str(item["tile_id"]): item for item in target_tiles_raw if isinstance(item, dict)
    }
    expected = set(EXPECTED_TRAIN_TILE_IDS) | set(EXPECTED_DEV_TILE_IDS)
    if set(split_by_id) != expected or set(target_by_id) != expected:
        raise RuntimeError("TSD scene identities do not match the frozen campaign")
    if not expected.isdisjoint(FORBIDDEN_TILE_IDS):
        raise RuntimeError("forbidden TSD tile leaked into campaign constants")

    def make(tile_id: str, role: str) -> SceneRecord:
        split_item = split_by_id[tile_id]
        target_item = target_by_id[tile_id]
        if split_item.get("role") != role or target_item.get("role") != role:
            raise RuntimeError(f"TSD role mismatch for {tile_id}")
        rgb_path = Path(str(split_item["rgb"])).resolve()
        target_pack = Path(str(target_item["target_pack"])).resolve()
        if not rgb_path.is_file() or not target_pack.is_file():
            raise FileNotFoundError(f"missing frozen TSD input for {tile_id}")
        rgb_hash = str(split_item["rgb_sha256"])
        target_hash = str(target_item["target_pack_sha256"])
        if sha256_file(rgb_path) != rgb_hash:
            raise RuntimeError(f"frozen RGB identity changed for {tile_id}")
        if sha256_file(target_pack) != target_hash:
            raise RuntimeError(f"frozen target-pack identity changed for {tile_id}")
        geometry_dir = output_dir / "geometry"
        return SceneRecord(
            tile_id=tile_id,
            role=role,
            rgb_path=rgb_path,
            rgb_sha256=rgb_hash,
            target_pack=target_pack,
            target_pack_sha256=target_hash,
            geometry_path=geometry_dir / f"potsdam_{tile_id}_da3_relative.tif",
            geometry_metadata_path=geometry_dir / f"potsdam_{tile_id}_da3_relative.json",
        )

    train = [make(tile_id, "train") for tile_id in EXPECTED_TRAIN_TILE_IDS]
    dev = [make(tile_id, "dev") for tile_id in EXPECTED_DEV_TILE_IDS]
    return train, dev


def _validate_reusable_geometry(record: SceneRecord) -> bool:
    geometry_exists = record.geometry_path.is_file()
    metadata_exists = record.geometry_metadata_path.is_file()
    if geometry_exists != metadata_exists:
        raise RuntimeError(
            f"incomplete cached geometry evidence for {record.tile_id}; remove the lone artifact"
        )
    if not geometry_exists:
        return False
    metadata = _load_json(record.geometry_metadata_path)
    required = {
        "tile_id": record.tile_id,
        "rgb_sha256": record.rgb_sha256,
        "model_id": DA3_MODEL_ID,
        "model_source": DA3_MODEL_SOURCE,
        "model_revision": DA3_HF_REVISION,
        "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"cached geometry metadata mismatch for {record.tile_id}: {key}")
    if metadata.get("geometry_sha256") != sha256_file(record.geometry_path):
        raise RuntimeError(f"cached geometry hash mismatch for {record.tile_id}")
    with rasterio.open(record.geometry_path) as geometry, rasterio.open(record.rgb_path) as rgb:
        if geometry.width != rgb.width or geometry.height != rgb.height:
            raise RuntimeError(f"cached geometry shape mismatch for {record.tile_id}")
        if not geometry.transform.almost_equals(rgb.transform):
            raise RuntimeError(f"cached geometry transform mismatch for {record.tile_id}")
    return True


def _ensure_geometry(
    records: list[SceneRecord],
    *,
    source_sha: str,
    prior: DA3MonocularPrior,
) -> None:
    for index, record in enumerate(records, start=1):
        if _validate_reusable_geometry(record):
            print(f"geometry[{index}/{len(records)}] reuse Potsdam {record.tile_id}", flush=True)
            continue
        record.geometry_path.parent.mkdir(parents=True, exist_ok=True)
        partial = record.geometry_path.with_suffix(record.geometry_path.suffix + ".partial")
        partial.unlink(missing_ok=True)
        print(f"geometry[{index}/{len(records)}] DA3 Potsdam {record.tile_id} ...", flush=True)
        result = infer_geometry_scene(
            record.rgb_path,
            prior,
            tile_size=1024,
            overlap=128,
            harmonize_overlaps=True,
        )
        write_float_geotiff(
            partial,
            result.relative_height,
            template_path=record.rgb_path,
            description=(
                f"DepthWizard TSD training geometry prior for frozen Potsdam {record.tile_id}"
            ),
            tags={
                "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
                "MODEL_ID": result.model_id,
                "POTSDAM_TILE": record.tile_id,
                "PURPOSE": PROTOCOL_VERSION,
            },
        )
        partial.replace(record.geometry_path)
        metadata = {
            "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION,
            "tile_id": record.tile_id,
            "role": record.role,
            "rgb": str(record.rgb_path),
            "rgb_sha256": record.rgb_sha256,
            "geometry": str(record.geometry_path),
            "geometry_sha256": sha256_file(record.geometry_path),
            "model_id": DA3_MODEL_ID,
            "model_source": DA3_MODEL_SOURCE,
            "model_revision": DA3_HF_REVISION,
            "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
            "tile_count": result.tile_count,
            "harmonized_tiles": result.harmonized_tiles,
            "training_source_git_sha": source_sha,
            "metric_calibration_performed": False,
            "sealed_blind_tile_payloads_consumed": False,
        }
        _write_json_atomic(record.geometry_metadata_path, metadata)
        print(
            f"geometry PASS {record.tile_id} sha256={metadata['geometry_sha256']}",
            flush=True,
        )


def _metric_targets_from_pack(path: Path) -> MetricTerrainStructureTargets:
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
            raise RuntimeError(f"unexpected TSD target-pack arrays: {path}: {sorted(pack.files)}")
        terrain = np.asarray(pack["terrain_m"], dtype=np.float32)
        agl = np.asarray(pack["above_ground_m"], dtype=np.float32)
        valid = np.asarray(pack["valid_mask"], dtype=bool)
        building = np.asarray(pack["building_mask"], dtype=bool)
        ground = np.asarray(pack["ground_mask"], dtype=bool)
        boundary = np.asarray(pack["boundary_mask"], dtype=bool)
    if terrain.shape != agl.shape or terrain.shape != valid.shape:
        raise RuntimeError("TSD target-pack array shape mismatch")
    dsm = terrain + agl
    dsm[~valid] = np.nan
    return MetricTerrainStructureTargets(
        dsm_m=dsm,
        terrain_m=terrain,
        above_ground_m=agl,
        valid_mask=valid,
        building_mask=building,
        ground_mask=ground,
        boundary_mask=boundary,
        instances=(),
    )


def _read_geometry(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
    if np.ma.isMaskedArray(data):
        return np.asarray(data.filled(np.nan), dtype=np.float32)
    return np.asarray(data, dtype=np.float32)


def _read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count < 3:
            raise RuntimeError(f"Potsdam RGB has fewer than three bands: {path}")
        rgb = np.moveaxis(src.read((1, 2, 3)), 0, -1).astype(np.float32)
    if float(np.nanmax(rgb)) > 1.0:
        rgb /= 255.0
    return np.clip(rgb, 0.0, 1.0)


def _fit_path(output_dir: Path, tile_id: str) -> Path:
    return output_dir / "scene-fits" / f"potsdam_{tile_id}_prior_reference_fit.json"


def _fit_for_scene(
    record: SceneRecord,
    metric: MetricTerrainStructureTargets,
    geometry: np.ndarray,
    output_dir: Path,
) -> PriorReferenceFit:
    path = _fit_path(output_dir, record.tile_id)
    geometry_hash = sha256_file(record.geometry_path)
    if path.is_file():
        payload = _load_json(path)
        if payload.get("target_pack_sha256") != record.target_pack_sha256:
            raise RuntimeError(f"cached TSD scene fit target mismatch for {record.tile_id}")
        if payload.get("geometry_sha256") != geometry_hash:
            raise RuntimeError(f"cached TSD scene fit geometry mismatch for {record.tile_id}")
        return PriorReferenceFit(
            scale_m_per_prior_unit=float(payload["scale_m_per_prior_unit"]),
            offset_m=float(payload["offset_m"]),
            rmse_m=float(payload["rmse_m"]),
            median_abs_residual_m=float(payload["median_abs_residual_m"]),
            samples=int(payload["samples"]),
        )
    valid = np.asarray(metric.valid_mask, dtype=bool) & np.isfinite(geometry)
    fit = fit_reference_to_prior(geometry, metric.dsm_m, valid)
    payload = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "tile_id": record.tile_id,
        "role": record.role,
        "target_pack_sha256": record.target_pack_sha256,
        "geometry_sha256": geometry_hash,
        **asdict(fit),
        "training_only_supervision_canonicalization": True,
        "inference_metric_calibration": False,
    }
    _write_json_atomic(path, payload)
    return fit


def _load_scene(record: SceneRecord, output_dir: Path) -> LoadedScene:
    metric = _metric_targets_from_pack(record.target_pack)
    geometry = _read_geometry(record.geometry_path)
    if geometry.shape != metric.valid_mask.shape:
        raise RuntimeError(f"geometry/target shape mismatch for {record.tile_id}")
    fit = _fit_for_scene(record, metric, geometry, output_dir)
    relative = canonicalize_metric_terrain_structure_targets(metric, fit)
    rgb = _read_rgb(record.rgb_path)
    if rgb.shape[:2] != geometry.shape:
        raise RuntimeError(f"RGB/geometry shape mismatch for {record.tile_id}")
    return LoadedScene(
        record=record,
        rgb=rgb,
        geometry=geometry,
        targets=relative,
        metric_agl_m=np.asarray(metric.above_ground_m, dtype=np.float32),
        fit=fit,
    )


def _window_from_center(row: int, col: int, shape: tuple[int, int], size: int) -> tuple[int, int]:
    height, width = shape
    if size > height or size > width:
        raise ValueError("TSD patch size exceeds scene dimensions")
    top = min(max(row - size // 2, 0), height - size)
    left = min(max(col - size // 2, 0), width - size)
    return top, left


def _sample_windows(
    scene: LoadedScene,
    count: int,
    rng: np.random.Generator,
) -> list[PatchWindow]:
    valid = np.asarray(scene.targets.valid_mask, dtype=bool) & np.isfinite(scene.geometry)
    building = np.asarray(scene.targets.building_mask, dtype=bool) & valid
    tall = building & (scene.metric_agl_m >= 8.0)
    indices = {
        "tall": np.flatnonzero(tall),
        "building": np.flatnonzero(building),
        "valid": np.flatnonzero(valid),
    }
    if indices["valid"].size == 0 or indices["building"].size == 0 or indices["tall"].size == 0:
        raise RuntimeError(f"scene {scene.record.tile_id} lacks required TSD patch support")
    classes = ("tall", "building", "valid")
    windows: list[PatchWindow] = []
    width = valid.shape[1]
    for index in range(count):
        sampling_class = classes[index % len(classes)]
        candidates = indices[sampling_class]
        chosen: PatchWindow | None = None
        for _ in range(64):
            flat = int(candidates[int(rng.integers(0, candidates.size))])
            row, col = divmod(flat, width)
            top, left = _window_from_center(row, col, valid.shape, PATCH_SIZE)
            patch_valid = valid[top : top + PATCH_SIZE, left : left + PATCH_SIZE]
            if float(np.mean(patch_valid)) < 0.20:
                continue
            chosen = PatchWindow(top, left, PATCH_SIZE, sampling_class)
            break
        if chosen is None:
            raise RuntimeError(
                f"could not sample a sufficiently supported {sampling_class} patch for "
                f"{scene.record.tile_id}"
            )
        windows.append(chosen)
    rng.shuffle(windows)
    return windows


def _patch_tensors(
    scene: LoadedScene,
    window: PatchWindow,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, TerrainStructureTargets, torch.Tensor, torch.Tensor]:
    rows = window.row_slice
    cols = window.col_slice
    geometry_np = scene.geometry[rows, cols]
    input_valid = np.isfinite(geometry_np)
    valid = np.asarray(scene.targets.valid_mask[rows, cols], dtype=bool) & input_valid
    building = np.asarray(scene.targets.building_mask[rows, cols], dtype=bool) & valid
    ground = np.asarray(scene.targets.ground_mask[rows, cols], dtype=bool) & valid
    boundary = np.asarray(scene.targets.boundary_mask[rows, cols], dtype=bool) & valid

    def field(values: np.ndarray) -> torch.Tensor:
        patch = np.asarray(values[rows, cols], dtype=np.float32)
        patch = np.where(valid, patch, 0.0).astype(np.float32)
        return torch.from_numpy(patch[None, None]).to(device=device)

    rgb_np = np.asarray(scene.rgb[rows, cols], dtype=np.float32).transpose(2, 0, 1)
    rgb = torch.from_numpy(rgb_np[None]).to(device=device)
    geometry = torch.from_numpy(
        np.where(input_valid, geometry_np, 0.0).astype(np.float32)[None, None]
    ).to(device=device)
    targets = TerrainStructureTargets(
        relative_dsm=field(scene.targets.relative_dsm),
        terrain_relative=field(scene.targets.terrain_relative),
        above_ground_relative=field(scene.targets.above_ground_relative),
        valid_mask=torch.from_numpy(valid[None, None]).to(device=device),
        building_mask=torch.from_numpy(building[None, None]).to(device=device),
        ground_mask=torch.from_numpy(ground[None, None]).to(device=device),
        boundary_mask=torch.from_numpy(boundary[None, None]).to(device=device),
    )
    scale = torch.tensor([scene.fit.scale_m_per_prior_unit], dtype=torch.float32, device=device)
    gsd = torch.tensor([NATIVE_GSD_M], dtype=torch.float32, device=device)
    return rgb, geometry, targets, scale, gsd


def _augment(
    rgb: torch.Tensor,
    geometry: torch.Tensor,
    targets: TerrainStructureTargets,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, TerrainStructureTargets]:
    k = int(rng.integers(0, 4))
    flip = bool(rng.integers(0, 2))

    def transform(tensor: torch.Tensor) -> torch.Tensor:
        output = torch.rot90(tensor, k=k, dims=(-2, -1)) if k else tensor
        return torch.flip(output, dims=(-1,)) if flip else output

    return (
        transform(rgb),
        transform(geometry),
        TerrainStructureTargets(
            relative_dsm=transform(targets.relative_dsm),
            terrain_relative=transform(targets.terrain_relative),
            above_ground_relative=transform(targets.above_ground_relative),
            valid_mask=transform(targets.valid_mask),
            building_mask=transform(targets.building_mask),
            ground_mask=transform(targets.ground_mask),
            boundary_mask=transform(targets.boundary_mask),
        ),
    )


def _finite_scalar(value: torch.Tensor, name: str) -> float:
    scalar = float(value.detach().cpu())
    if not math.isfinite(scalar):
        raise RuntimeError(f"non-finite TSD {name}: {scalar}")
    return scalar


def _train_epoch(
    model: TerrainStructureModel,
    optimizer: torch.optim.Optimizer,
    records: list[SceneRecord],
    output_dir: Path,
    device: torch.device,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    model.train()
    order = list(records)
    rng.shuffle(order)
    total_loss = 0.0
    steps = 0
    class_counts = {"tall": 0, "building": 0, "valid": 0}
    for record in order:
        scene = _load_scene(record, output_dir)
        windows = _sample_windows(scene, TRAIN_PATCHES_PER_TILE_PER_EPOCH, rng)
        for window in windows:
            rgb, geometry, targets, scale, gsd = _patch_tensors(scene, window, device)
            rgb, geometry, targets = _augment(rgb, geometry, targets, rng)
            optimizer.zero_grad(set_to_none=True)
            output = model(rgb, geometry, gsd_m=gsd)
            loss = compute_terrain_structure_loss(
                output,
                targets,
                scale,
                gsd,
                config=model.config,
            )
            scalar = _finite_scalar(loss.total, "training loss")
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            total_loss += scalar
            steps += 1
            class_counts[window.sampling_class] += 1
        del scene
    if steps == 0:
        raise RuntimeError("TSD training epoch produced zero optimization steps")
    return {
        "loss": total_loss / steps,
        "steps": steps,
        "tall_patches": class_counts["tall"],
        "building_patches": class_counts["building"],
        "valid_patches": class_counts["valid"],
    }


def _evaluate_dev(
    model: TerrainStructureModel,
    records: list[SceneRecord],
    output_dir: Path,
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    totals = {
        "agl_abs": 0.0,
        "agl_count": 0,
        "tall_agl_abs": 0.0,
        "tall_agl_count": 0,
        "roof_abs": 0.0,
        "roof_count": 0,
        "ground_abs": 0.0,
        "ground_count": 0,
        "loss": 0.0,
        "steps": 0,
    }
    rng = np.random.default_rng(SEED + 9001)
    with torch.inference_mode():
        for record in records:
            scene = _load_scene(record, output_dir)
            windows = _sample_windows(scene, DEV_PATCHES_PER_TILE, rng)
            for window in windows:
                rgb, geometry, targets, scale, gsd = _patch_tensors(scene, window, device)
                output = model(rgb, geometry, gsd_m=gsd)
                loss = compute_terrain_structure_loss(
                    output,
                    targets,
                    scale,
                    gsd,
                    config=model.config,
                )
                totals["loss"] += _finite_scalar(loss.total, "dev loss")
                totals["steps"] += 1

                scale_map = scale.view(1, 1, 1, 1)
                offset = torch.tensor(
                    [scene.fit.offset_m], dtype=torch.float32, device=device
                ).view(1, 1, 1, 1)
                pred_dsm_m = output.relative_height * scale_map + offset
                pred_terrain_m = output.terrain_relative * scale_map + offset
                pred_agl_m = output.above_ground_amplitude_relative * scale_map
                target_dsm_m = targets.relative_dsm * scale_map + offset
                target_terrain_m = targets.terrain_relative * scale_map + offset
                target_agl_m = targets.above_ground_relative * scale_map
                building = targets.building_mask.to(dtype=torch.bool)
                ground = targets.ground_mask.to(dtype=torch.bool)
                tall = building & (target_agl_m >= 8.0)

                def accumulate(name: str, error: torch.Tensor, mask: torch.Tensor) -> None:
                    count = int(mask.sum().detach().cpu())
                    if count <= 0:
                        return
                    totals[f"{name}_abs"] += float(error[mask].abs().sum().detach().cpu())
                    totals[f"{name}_count"] += count

                accumulate("agl", pred_agl_m - target_agl_m, building)
                accumulate("tall_agl", pred_agl_m - target_agl_m, tall)
                accumulate("roof", pred_dsm_m - target_dsm_m, building)
                accumulate("ground", pred_terrain_m - target_terrain_m, ground)
            del scene

    def mae(name: str) -> float:
        count = int(totals[f"{name}_count"])
        if count <= 0:
            raise RuntimeError(f"TSD dev evaluation has no {name} support")
        return float(totals[f"{name}_abs"]) / count

    steps = int(totals["steps"])
    if steps <= 0:
        raise RuntimeError("TSD dev evaluation produced zero steps")
    agl_mae = mae("agl")
    tall_agl_mae = mae("tall_agl")
    roof_mae = mae("roof")
    ground_mae = mae("ground")
    score = 2.0 * tall_agl_mae + agl_mae + roof_mae + ground_mae
    return {
        "loss": float(totals["loss"]) / steps,
        "steps": steps,
        "agl_mae_m": agl_mae,
        "tall_agl_mae_m": tall_agl_mae,
        "roof_mae_m": roof_mae,
        "ground_mae_m": ground_mae,
        "selection_score": score,
        "agl_pixels": int(totals["agl_count"]),
        "tall_agl_pixels": int(totals["tall_agl_count"]),
        "roof_pixels": int(totals["roof_count"]),
        "ground_pixels": int(totals["ground_count"]),
    }


def _checkpoint_payload(
    *,
    model: TerrainStructureModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int,
    best_score: float,
    source_sha: str,
    target_manifest_sha256: str,
    split_manifest_sha256: str,
    authorization_sha256: str,
    history: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "model_architecture_version": model.config.architecture_version,
        "model_config": asdict(model.config),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "training_source_git_sha": source_sha,
        "target_manifest_sha256": target_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "authorization_sha256": authorization_sha256,
        "seed": SEED,
        "history": history,
        "production_promoted": False,
        "sealed_blind_tile_payloads_consumed": False,
    }


def _load_resume(
    path: Path,
    model: TerrainStructureModel,
    optimizer: torch.optim.Optimizer,
    *,
    source_sha: str,
    authorization_sha256: str,
    device: torch.device,
) -> tuple[int, int, float, list[dict[str, object]]]:
    if not path.is_file():
        return 0, 0, float("inf"), []
    payload = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("TSD resume checkpoint must be a mapping")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError("TSD resume checkpoint protocol mismatch")
    if payload.get("training_source_git_sha") != source_sha:
        raise RuntimeError("TSD resume checkpoint was produced by a different source commit")
    if payload.get("authorization_sha256") != authorization_sha256:
        raise RuntimeError("TSD resume checkpoint was produced under a different authorization")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    history_raw = payload.get("history", [])
    if not isinstance(history_raw, list):
        raise TypeError("TSD checkpoint history must be a list")
    history = [item for item in history_raw if isinstance(item, dict)]
    if len(history) != len(history_raw):
        raise TypeError("TSD checkpoint history items must be objects")
    return (
        int(payload["epoch"]),
        int(payload["best_epoch"]),
        float(payload["best_score"]),
        history,
    )


def main() -> int:
    args = parse_args()
    source_sha, source_branch = _require_source_identity()
    target_manifest = args.target_manifest.resolve()
    audit_path = args.audit.resolve()
    authorization_path = args.authorization.resolve()
    output_dir = args.output_dir.resolve()
    _require_free_space(output_dir)

    target_payload, _, _ = _validate_authorization(
        target_manifest,
        audit_path,
        authorization_path,
    )
    split_path, split = _split_manifest_from_target(target_payload)
    train_records, dev_records = _build_scene_records(target_payload, split, output_dir)
    all_records = train_records + dev_records

    report_path = output_dir / "training_report.json"
    best_path = output_dir / "tsd_potsdam_v1_best.pt"
    last_path = output_dir / "tsd_potsdam_v1_last.pt"
    if report_path.is_file():
        report = _load_json(report_path)
        if report.get("status") == "TRAINED_TSD_CANDIDATE_NOT_PROMOTED":
            print("TSD_TRAINING=ALREADY_COMPLETE")
            print(f"best_checkpoint={best_path}")
            print(f"best_checkpoint_sha256={sha256_file(best_path)}")
            print("production_promoted=false")
            print("sealed_blind_tile_payloads_consumed=false")
            return 0
        raise RuntimeError("existing TSD training report is not a completed campaign report")

    device = _resolve_device()
    print(f"device={device}")
    print(f"protocol_version={PROTOCOL_VERSION}")
    print(f"training_source_git_sha={source_sha}")
    print(f"target_manifest_sha256={EXPECTED_TARGET_MANIFEST_SHA256}")
    print(f"split_manifest_sha256={sha256_file(split_path)}")
    print(f"authorization_sha256={sha256_file(authorization_path)}")
    print(f"train_tiles={','.join(record.tile_id for record in train_records)}")
    print(f"dev_tiles={','.join(record.tile_id for record in dev_records)}")
    print("sealed_blind_tile_payloads_consumed=false")

    prior = DA3MonocularPrior(device="auto")
    _ensure_geometry(all_records, source_sha=source_sha, prior=prior)
    del prior
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    config = TerrainStructureConfig()
    model = TerrainStructureModel(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    authorization_hash = sha256_file(authorization_path)
    completed_epoch, best_epoch, best_score, history = _load_resume(
        last_path,
        model,
        optimizer,
        source_sha=source_sha,
        authorization_sha256=authorization_hash,
        device=device,
    )
    if completed_epoch:
        print(
            f"resume=true completed_epoch={completed_epoch} best_epoch={best_epoch} "
            f"best_score={best_score:.6f}",
            flush=True,
        )

    stale_epochs = max(0, completed_epoch - best_epoch)
    for epoch in range(completed_epoch + 1, EPOCHS + 1):
        train_metrics = _train_epoch(
            model,
            optimizer,
            train_records,
            output_dir,
            device,
            rng,
        )
        dev_metrics = _evaluate_dev(model, dev_records, output_dir, device)
        score = float(dev_metrics["selection_score"])
        improved = score < best_score - 1e-6
        if improved:
            best_score = score
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train": train_metrics,
            "dev": dev_metrics,
            "improved": improved,
        }
        history.append(epoch_record)
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            source_sha=source_sha,
            target_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256,
            split_manifest_sha256=EXPECTED_SPLIT_MANIFEST_SHA256,
            authorization_sha256=authorization_hash,
            history=history,
        )
        _torch_save_atomic(last_path, payload)
        if improved:
            _torch_save_atomic(best_path, payload)
        _write_json_atomic(
            output_dir / "training_progress.json",
            {
                "status": "TRAINING_TSD_CANDIDATE",
                "protocol_version": PROTOCOL_VERSION,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "latest": epoch_record,
                "training_source_git_sha": source_sha,
                "production_promoted": False,
                "sealed_blind_tile_payloads_consumed": False,
            },
        )
        print(
            f"epoch={epoch}/{EPOCHS} train_loss={float(train_metrics['loss']):.6f} "
            f"dev_loss={float(dev_metrics['loss']):.6f} "
            f"dev_tall_agl_mae_m={float(dev_metrics['tall_agl_mae_m']):.4f} "
            f"dev_agl_mae_m={float(dev_metrics['agl_mae_m']):.4f} "
            f"dev_roof_mae_m={float(dev_metrics['roof_mae_m']):.4f} "
            f"dev_ground_mae_m={float(dev_metrics['ground_mae_m']):.4f} "
            f"score={score:.4f} improved={str(improved).lower()}",
            flush=True,
        )

        if (
            epoch >= MIN_EPOCHS_BEFORE_EARLY_STOP
            and stale_epochs >= EARLY_STOPPING_PATIENCE
        ):
            print(f"early_stopping=true epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if not best_path.is_file():
        raise RuntimeError("TSD training completed without a best checkpoint")
    final_report = {
        "schema_version": 1,
        "status": "TRAINED_TSD_CANDIDATE_NOT_PROMOTED",
        "protocol_version": PROTOCOL_VERSION,
        "training_source_git_sha": source_sha,
        "training_source_git_branch": source_branch,
        "target_manifest": str(target_manifest),
        "target_manifest_sha256": EXPECTED_TARGET_MANIFEST_SHA256,
        "split_manifest": str(split_path),
        "split_manifest_sha256": EXPECTED_SPLIT_MANIFEST_SHA256,
        "audit": str(audit_path),
        "audit_sha256": sha256_file(audit_path),
        "authorization": str(authorization_path),
        "authorization_sha256": authorization_hash,
        "train_tile_ids": list(EXPECTED_TRAIN_TILE_IDS),
        "dev_tile_ids": list(EXPECTED_DEV_TILE_IDS),
        "model_config": asdict(config),
        "training_hyperparameters": {
            "seed": SEED,
            "max_epochs": EPOCHS,
            "patch_size_px": PATCH_SIZE,
            "native_gsd_m": NATIVE_GSD_M,
            "patch_extent_m": PATCH_SIZE * NATIVE_GSD_M,
            "train_patches_per_tile_per_epoch": TRAIN_PATCHES_PER_TILE_PER_EPOCH,
            "dev_patches_per_tile": DEV_PATCHES_PER_TILE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_clip_norm": GRAD_CLIP_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "min_epochs_before_early_stop": MIN_EPOCHS_BEFORE_EARLY_STOP,
            "sampling_cycle": ["tall>=8m", "building", "valid"],
            "dev_selection_score": "2*tall_agl_mae + agl_mae + roof_mae + ground_mae",
        },
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "best_epoch": best_epoch,
        "best_selection_score": best_score,
        "history": history,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": sha256_file(best_path),
        "last_checkpoint": str(last_path),
        "last_checkpoint_sha256": sha256_file(last_path),
        "production_promoted": False,
        "sealed_blind_tile_payloads_consumed": False,
        "claim_boundary": (
            "Research candidate trained only on the frozen first Potsdam TSD train split and selected "
            "only with its frozen dev split. This report is not a production promotion or an exposed, "
            "external, or sealed-blind evaluation result. Metric target-to-relative conversion is a "
            "training-only supervision canonicalization; inference still requires independent metric "
            "calibration evidence."
        ),
    }
    _write_json_atomic(report_path, final_report)
    print("TSD_TRAINING=COMPLETE_NOT_PROMOTED")
    print(f"best_epoch={best_epoch}")
    print(f"best_selection_score={best_score:.6f}")
    print(f"best_checkpoint={best_path}")
    print(f"best_checkpoint_sha256={sha256_file(best_path)}")
    print(f"training_report={report_path}")
    print(f"training_report_sha256={sha256_file(report_path)}")
    print("production_promoted=false")
    print("sealed_blind_tile_payloads_consumed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
