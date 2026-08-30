#!/usr/bin/env python3
"""Fresh reference-free Potsdam 2_14 qualification for dual-lattice DA3 mosaic v4.

Consumes only RGB, pinned DA3 weights, frozen Copernicus GLO-30 calibration evidence, and previously
frozen prediction rasters used as negative/control baselines. The official Potsdam DSM is never
opened, stat'ed for raster content, or hashed. The v4 prediction is SHA-frozen before a like-for-like
896 px phase-fold comparison against original, v2, and rejected v3 mosaics.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

HEAD = "d2c94772a60eea6b113107c3a3f2985503802f87"
PIPELINE_REVISION = "dual-lattice-affine-mosaic-v4"
CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
CHECKPOINT_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
RGB_REL = Path("data/external/isprs-potsdam/2_Ortho_RGB/top_potsdam_2_14_RGB.tif")
CALIBRATION_REL = Path(
    "workspace/final-science-data/predictions/potsdam-2-14-urban-test/"
    "calibration/copernicus-glo30-mosaic.tif"
)
CALIBRATION_SHA256 = "96e3c9de4049cff5d60f5e927c6d574f8cafcfde6164fa427a3ceab9fb11126d"
OLD_RDSM_REL = Path(
    "workspace/final-science-data/predictions/potsdam-2-14-urban-test/reconstruction/rdsm.tif"
)
OLD_RDSM_SHA256 = "be5d01c1aa149efbfdad4b9f469aa5529c3d7c2bb727cff80e55b8750c5ad4bb"
V2_RDSM_REL = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/reconstruction/rdsm.tif")
V2_RDSM_SHA256 = "620b0430d0b22c7854733cc61bddd319f4769d9d273f87d29baa7a396764a35e"
V3_RDSM_REL = Path("workspace/urban-mosaic-v3/2a452f6-potsdam-2_14/reconstruction/rdsm.tif")
V3_RDSM_SHA256 = "f13bdb0f989f0b129555c9bb97b32a4bb7ccdf2e951c0cd452c2d29d2376d4c0"
OUTPUT_REL = Path("workspace/urban-mosaic-v4/d2c9477-potsdam-2_14")
REFERENCE_STRING_ONLY = "data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif"
TILE_SIZE = 1024
OVERLAP = 128
TARGET_STRIDE = TILE_SIZE - OVERLAP
EXPECTED_TILE_COUNT = 98
DOWNSAMPLE = 8
CONTROL_PERIODS = (768, 832, 960, 1024)
PERIODS = (768, 832, TARGET_STRIDE, 960, 1024)


class QualificationError(RuntimeError):
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
        raise QualificationError(f"requires exact source head {HEAD}; current={current}")
    if git(root, "status", "--porcelain"):
        raise QualificationError("repository must be clean")


def find_checkpoint() -> Path:
    explicit = os.environ.get("DEPTHWIZARD_DA3_CHECKPOINT", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(
        Path.home()
        / ".cache/huggingface/hub/models--depth-anything--DA3MONO-LARGE/snapshots"
        / CHECKPOINT_REVISION
        / "model.safetensors"
    )
    for candidate in candidates:
        if candidate.is_file() and sha256(candidate) == CHECKPOINT_SHA256:
            return candidate.resolve()
    raise QualificationError("pinned DA3 checkpoint not found with required SHA-256")


def require_hash(path: Path, expected: str, label: str) -> Path:
    resolved = path.resolve(strict=True)
    actual = sha256(resolved)
    if actual != expected:
        raise QualificationError(f"{label} SHA mismatch: {actual} != {expected}")
    return resolved


def require_surface(path: Path) -> None:
    with rasterio.open(path) as src:
        if src.shape != (6000, 6000):
            raise QualificationError(f"expected 6000x6000 Potsdam surface, got {src.shape}: {path}")
        sample = src.read(1, out_shape=(64, 64), resampling=Resampling.average)
        if not np.any(np.isfinite(sample)):
            raise QualificationError(f"surface has no finite values: {path}")


def run_cli(root: Path, *args: str) -> None:
    command = [sys.executable, "-m", "depthwizard.cli", *args]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def read_downsampled(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.shape != (6000, 6000):
            raise QualificationError(f"expected 6000x6000 Potsdam surface: {path}")
        height = math.ceil(src.height / DOWNSAMPLE)
        width = math.ceil(src.width / DOWNSAMPLE)
        values = src.read(1, out_shape=(height, width), resampling=Resampling.average).astype(np.float32)
        valid = src.read_masks(1, out_shape=(height, width), resampling=Resampling.nearest) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    if float(valid.mean()) < 0.99:
        raise QualificationError(f"phase diagnostic expects near-complete coverage: {path}")
    fill = float(np.median(values[valid]))
    return np.where(valid, values, fill).astype(np.float32)


def robust_normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)].astype(np.float64)
    p01, p99 = np.percentile(finite, [1.0, 99.0])
    span = float(p99 - p01)
    if span <= 1e-12:
        raise QualificationError("surface has no robust relief span")
    return ((values.astype(np.float64) - p01) / span).astype(np.float32)


def box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32, copy=True)
    padded = np.pad(values.astype(np.float64), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    summed = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (summed / float(size * size)).astype(np.float32)


def remove_plane(cell: np.ndarray) -> np.ndarray:
    height, width = cell.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    design = np.column_stack([np.ones(cell.size), xx.reshape(-1), yy.reshape(-1)])
    target = cell.reshape(-1).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return (target - design @ coefficients).reshape(cell.shape).astype(np.float32)


def phase_fold(values: np.ndarray, period_source_px: int) -> dict[str, float | int | None]:
    if period_source_px % DOWNSAMPLE != 0:
        raise QualificationError("phase-fold period must be divisible by diagnostic downsample")
    period = period_source_px // DOWNSAMPLE
    normalized = robust_normalize(values)
    rows = normalized.shape[0] // period
    cols = normalized.shape[1] // period
    if rows < 4 or cols < 4:
        raise QualificationError(f"insufficient phase repeats for {period_source_px}px")
    crop = normalized[: rows * period, : cols * period]
    cells: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            cell = crop[row * period : (row + 1) * period, col * period : (col + 1) * period]
            cells.append(remove_plane(box_blur(cell, radius=2)))
    stack = np.stack(cells, axis=0).astype(np.float64)
    fingerprint = np.mean(stack, axis=0)
    typical_energy = float(np.mean(stack * stack))
    fingerprint_energy = float(np.mean(fingerprint * fingerprint))
    coherence = 0.0 if typical_energy <= 1e-15 else fingerprint_energy / typical_energy
    flat = stack.reshape(stack.shape[0], -1)
    flat -= np.mean(flat, axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    usable = norms > 1e-15
    normalized_flat = flat[usable] / norms[usable, None]
    mean_pairwise = 0.0
    if normalized_flat.shape[0] >= 2:
        correlations = normalized_flat @ normalized_flat.T
        tri = np.triu_indices(correlations.shape[0], k=1)
        mean_pairwise = float(np.mean(correlations[tri]))
    return {
        "period_source_px": period_source_px,
        "phase_cell_count": int(stack.shape[0]),
        "phase_coherence_energy_fraction": coherence,
        "phase_coherence_rms_ratio": math.sqrt(max(coherence, 0.0)),
        "mean_pairwise_phase_correlation": mean_pairwise,
        "fingerprint_rms": math.sqrt(max(fingerprint_energy, 0.0)),
        "typical_cell_rms": math.sqrt(max(typical_energy, 0.0)),
    }


def survey(path: Path) -> dict[str, Any]:
    values = read_downsampled(path)
    periods: dict[str, Any] = {str(period): phase_fold(values, period) for period in PERIODS}
    target = periods[str(TARGET_STRIDE)]
    controls = np.asarray(
        [float(periods[str(period)]["phase_coherence_energy_fraction"]) for period in CONTROL_PERIODS],
        dtype=np.float64,
    )
    control_median = float(np.median(controls))
    target["control_period_median_coherence"] = control_median
    target["stride_specificity_vs_control_periods"] = (
        float(target["phase_coherence_energy_fraction"]) / control_median
        if control_median > 1e-15
        else None
    )
    return periods


def safe_ratio(new: float | None, baseline: float | None) -> float | None:
    if new is None or baseline is None or abs(baseline) <= 1e-15:
        return None
    return float(new / baseline)


def compare(new_periods: dict[str, Any], baseline_periods: dict[str, Any]) -> dict[str, float | None]:
    new = new_periods[str(TARGET_STRIDE)]
    baseline = baseline_periods[str(TARGET_STRIDE)]
    return {
        "coherence_new_over_baseline": safe_ratio(
            float(new["phase_coherence_energy_fraction"]),
            float(baseline["phase_coherence_energy_fraction"]),
        ),
        "rms_ratio_new_over_baseline": safe_ratio(
            float(new["phase_coherence_rms_ratio"]),
            float(baseline["phase_coherence_rms_ratio"]),
        ),
        "abs_mean_pairwise_new_over_baseline": safe_ratio(
            abs(float(new["mean_pairwise_phase_correlation"])),
            abs(float(baseline["mean_pairwise_phase_correlation"])),
        ),
        "stride_specificity_new_over_baseline": safe_ratio(
            new.get("stride_specificity_vs_control_periods"),
            baseline.get("stride_specificity_vs_control_periods"),
        ),
    }


def validate_existing(output: Path) -> bool:
    freeze_path = output / "pre-reference-freeze.json"
    diagnostic_path = output / "v4-stride-phase-diagnostic.json"
    if not freeze_path.is_file() or not diagnostic_path.is_file():
        return False
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if freeze.get("git_head") != HEAD:
        raise QualificationError("existing freeze belongs to another source head")
    if freeze.get("reference_values_consumed") is not False:
        raise QualificationError("existing freeze violates reference separation")
    if freeze.get("reference_raster_opened_or_hashed") is not False:
        raise QualificationError("existing freeze does not preserve unopened-reference boundary")
    for key in ("relative_dsm", "metric_dsm", "calibration_evidence"):
        record = freeze.get(key)
        if not isinstance(record, dict):
            raise QualificationError(f"existing freeze missing {key}")
        path_text = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path_text, str) or not isinstance(digest, str):
            raise QualificationError(f"malformed frozen identity for {key}")
        path = Path(path_text)
        if not path.is_file() or sha256(path) != digest:
            raise QualificationError(f"frozen identity mismatch for {key}")
    print("PASS_V4_POTSDAM_PRE_REFERENCE_FREEZE_REUSED")
    print(f"freeze={freeze_path}")
    print(f"diagnostic={diagnostic_path}")
    print("reference_values_consumed=false")
    print("reference_raster_opened_or_hashed=false")
    return True


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    checkpoint = find_checkpoint()
    rgb = (root / RGB_REL).resolve(strict=True)
    calibration = require_hash(root / CALIBRATION_REL, CALIBRATION_SHA256, "calibration evidence")
    old_rdsm = require_hash(root / OLD_RDSM_REL, OLD_RDSM_SHA256, "old defective rDSM")
    v2_rdsm = require_hash(root / V2_RDSM_REL, V2_RDSM_SHA256, "v2 rDSM")
    v3_rdsm = require_hash(root / V3_RDSM_REL, V3_RDSM_SHA256, "v3 rDSM")
    for baseline in (old_rdsm, v2_rdsm, v3_rdsm):
        require_surface(baseline)

    output = (root / OUTPUT_REL).resolve(strict=False)
    if output.exists():
        if validate_existing(output):
            return 0
        raise QualificationError(f"incomplete prior v4 qualification directory exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    reconstruction = output / "reconstruction"
    rdsm = reconstruction / "rdsm.tif"
    metric = output / "metric-dsm.tif"
    freeze_path = output / "pre-reference-freeze.json"
    diagnostic_path = output / "v4-stride-phase-diagnostic.json"

    print("=== POTSDAM 2_14 V4 DUAL-LATTICE PRE-REFERENCE QUALIFICATION ===")
    print(f"git_head={HEAD}")
    print(f"pipeline_revision={PIPELINE_REVISION}")
    print(f"source_rgb={rgb}")
    print("primary_tiles=49")
    print("shifted_tiles=49")
    print("reference_values_consumed=false")
    print("reference_raster_opened_or_hashed=false")

    run_cli(
        root,
        "reconstruct-da3",
        str(rgb),
        str(reconstruction),
        "--tile-size",
        str(TILE_SIZE),
        "--overlap",
        str(OVERLAP),
    )
    require_surface(rdsm)
    reconstruction_report_path = reconstruction / "reconstruction_report.json"
    reconstruction_report = json.loads(reconstruction_report_path.read_text(encoding="utf-8"))
    if reconstruction_report.get("model_id") != "DA3MONO-LARGE":
        raise QualificationError(f"unexpected model_id: {reconstruction_report.get('model_id')}")
    if reconstruction_report.get("tile_count") != EXPECTED_TILE_COUNT:
        raise QualificationError(
            f"expected {EXPECTED_TILE_COUNT} dual-lattice tile inferences; "
            f"got {reconstruction_report.get('tile_count')}"
        )

    run_cli(root, "calibrate-dem", str(rdsm), str(calibration), str(metric))
    require_surface(metric)

    freeze = {
        "schema_version": 1,
        "status": "FROZEN_PRE_REFERENCE_V4",
        "git_head": HEAD,
        "geometry_pipeline_revision": PIPELINE_REVISION,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "target_stride_px": TARGET_STRIDE,
        "tile_count": EXPECTED_TILE_COUNT,
        "source_rgb": {"path": str(rgb), "sha256": sha256(rgb)},
        "model_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "calibration_evidence": {"path": str(calibration), "sha256": sha256(calibration)},
        "relative_dsm": {"path": str(rdsm.resolve()), "sha256": sha256(rdsm)},
        "metric_dsm": {"path": str(metric.resolve()), "sha256": sha256(metric)},
        "reference_path_string_only": REFERENCE_STRING_ONLY,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
    }
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print("PASS_V4_POTSDAM_PRE_REFERENCE_FREEZE")
    print(f"freeze={freeze_path}")
    print(f"rdsm_sha256={freeze['relative_dsm']['sha256']}")
    print(f"metric_dsm_sha256={freeze['metric_dsm']['sha256']}")

    old_periods = survey(old_rdsm)
    v2_periods = survey(v2_rdsm)
    v3_periods = survey(v3_rdsm)
    v4_periods = survey(rdsm)
    report = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_NO_PASS_CLAIM",
        "git_head": HEAD,
        "geometry_pipeline_revision": PIPELINE_REVISION,
        "target_stride_px": TARGET_STRIDE,
        "control_periods_px": list(CONTROL_PERIODS),
        "old_896": old_periods[str(TARGET_STRIDE)],
        "v2_896": v2_periods[str(TARGET_STRIDE)],
        "v3_896": v3_periods[str(TARGET_STRIDE)],
        "v4_896": v4_periods[str(TARGET_STRIDE)],
        "v4_over_old": compare(v4_periods, old_periods),
        "v4_over_v2": compare(v4_periods, v2_periods),
        "v4_over_v3": compare(v4_periods, v3_periods),
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "purpose": "reference-free negative-control comparison of exact 896 px inference-grid periodicity",
    }
    diagnostic_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS_V4_STRIDE_PHASE_DIAGNOSTIC_GENERATED")
    print(f"diagnostic={diagnostic_path}")
    print("reference_values_consumed=false")
    print("reference_raster_opened_or_hashed=false")
    print("DIAGNOSTIC_ONLY_NO_PASS_CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
