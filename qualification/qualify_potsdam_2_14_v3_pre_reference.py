#!/usr/bin/env python3
"""Fresh Potsdam 2_14 qualification for the v3 global-scaffold residual mosaic.

This stage consumes only RGB, the pinned DA3 checkpoint, and the already-frozen independent
Copernicus GLO-30 calibration mosaic. It never opens or hashes the official Potsdam DSM.
After the new prediction is SHA-frozen, it compares 896 px phase-locking against both the original
known-defective mosaic and the v2 corrective mosaic. The diagnostic intentionally makes no PASS
claim; its purpose is to prove whether v3 materially removes the exact inference-grid signature.
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

HEAD = "2a452f691ab235f69c1884bbb7b32885cc535213"
PIPELINE_REVISION = "global-scaffold-residual-mosaic-v3"
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
V2_RDSM_REL = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/reconstruction/rdsm.tif")
OUTPUT_REL = Path("workspace/urban-mosaic-v3/2a452f6-potsdam-2_14")
TILE_SIZE = 1024
OVERLAP = 128
TARGET_STRIDE = TILE_SIZE - OVERLAP
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


def run_cli(root: Path, *args: str) -> None:
    command = [sys.executable, "-m", "depthwizard.cli", *args]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def require_surface(path: Path, *, expected_shape: tuple[int, int] = (6000, 6000)) -> None:
    if not path.is_file():
        raise QualificationError(f"missing surface: {path}")
    with rasterio.open(path) as src:
        if src.shape != expected_shape:
            raise QualificationError(f"unexpected raster shape {src.shape}: {path}")
        values = src.read(1, out_shape=(64, 64), resampling=Resampling.average)
        if not np.any(np.isfinite(values)):
            raise QualificationError(f"surface contains no finite values: {path}")


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
        raise QualificationError(f"diagnostic expects near-complete coverage: {path}")
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


def phase_fold(values: np.ndarray, period_source_px: int) -> dict[str, float | int]:
    if period_source_px % DOWNSAMPLE != 0:
        raise QualificationError("phase-fold period must divide by diagnostic downsample")
    period = period_source_px // DOWNSAMPLE
    normalized = robust_normalize(values)
    rows = normalized.shape[0] // period
    cols = normalized.shape[1] // period
    if rows < 4 or cols < 4:
        raise QualificationError(f"insufficient repeats for period {period_source_px}")
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
    periods = {str(period): phase_fold(values, period) for period in PERIODS}
    target = periods[str(TARGET_STRIDE)]
    controls = [float(periods[str(period)]["phase_coherence_energy_fraction"]) for period in CONTROL_PERIODS]
    control_median = float(np.median(np.asarray(controls, dtype=np.float64)))
    target["control_period_median_coherence"] = control_median
    target["stride_specificity_vs_control_periods"] = (
        float(target["phase_coherence_energy_fraction"]) / control_median
        if control_median > 1e-15
        else None
    )
    return periods


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or abs(denominator) <= 1e-15:
        return None
    return float(numerator / denominator)


def target_metrics(periods: dict[str, Any]) -> dict[str, Any]:
    return periods[str(TARGET_STRIDE)]


def comparison(new: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float | None]:
    new_target = target_metrics(new)
    base_target = target_metrics(baseline)
    return {
        "coherence_new_over_baseline": safe_ratio(
            float(new_target["phase_coherence_energy_fraction"]),
            float(base_target["phase_coherence_energy_fraction"]),
        ),
        "rms_ratio_new_over_baseline": safe_ratio(
            float(new_target["phase_coherence_rms_ratio"]),
            float(base_target["phase_coherence_rms_ratio"]),
        ),
        "abs_mean_pairwise_new_over_baseline": safe_ratio(
            abs(float(new_target["mean_pairwise_phase_correlation"])),
            abs(float(base_target["mean_pairwise_phase_correlation"])),
        ),
        "stride_specificity_new_over_baseline": safe_ratio(
            new_target.get("stride_specificity_vs_control_periods"),
            base_target.get("stride_specificity_vs_control_periods"),
        ),
    }


def validate_existing(output: Path) -> bool:
    freeze = output / "pre-reference-freeze.json"
    diagnostic = output / "v3-stride-phase-diagnostic.json"
    if not freeze.is_file() or not diagnostic.is_file():
        return False
    payload = json.loads(freeze.read_text(encoding="utf-8"))
    if payload.get("git_head") != HEAD:
        raise QualificationError("existing freeze belongs to another head")
    if payload.get("reference_values_consumed") is not False:
        raise QualificationError("existing freeze violates reference separation")
    for key in ("relative_dsm", "metric_dsm", "calibration_evidence"):
        record = payload.get(key)
        if not isinstance(record, dict):
            raise QualificationError(f"existing freeze missing {key}")
        path_text = record.get("path")
        digest = record.get("sha256")
        if not isinstance(path_text, str) or not isinstance(digest, str):
            raise QualificationError(f"malformed existing identity: {key}")
        path = Path(path_text)
        if not path.is_file() or sha256(path) != digest:
            raise QualificationError(f"existing frozen identity mismatch: {key}")
    print("PASS_V3_POTSDAM_PRE_REFERENCE_FREEZE_REUSED")
    print(f"freeze={freeze}")
    print(f"diagnostic={diagnostic}")
    print("reference_values_consumed=false")
    print("DIAGNOSTIC_ONLY_NO_PASS_CLAIM")
    return True


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    checkpoint = find_checkpoint()
    rgb = (root / RGB_REL).resolve(strict=True)
    calibration = (root / CALIBRATION_REL).resolve(strict=True)
    if sha256(calibration) != CALIBRATION_SHA256:
        raise QualificationError("Copernicus calibration evidence SHA does not match frozen identity")
    old_rdsm = (root / OLD_RDSM_REL).resolve(strict=True)
    v2_rdsm = (root / V2_RDSM_REL).resolve(strict=True)
    require_surface(old_rdsm)
    require_surface(v2_rdsm)

    output = (root / OUTPUT_REL).resolve(strict=False)
    if output.exists():
        if validate_existing(output):
            return 0
        raise QualificationError(f"incomplete prior v3 qualification directory exists: {output}")
    output.mkdir(parents=True, exist_ok=False)

    reconstruction = output / "reconstruction"
    rdsm = reconstruction / "rdsm.tif"
    metric = output / "metric-dsm.tif"
    freeze_path = output / "pre-reference-freeze.json"
    diagnostic_path = output / "v3-stride-phase-diagnostic.json"

    print("=== POTSDAM 2_14 V3 PRE-REFERENCE QUALIFICATION ===")
    print(f"git_head={HEAD}")
    print(f"pipeline_revision={PIPELINE_REVISION}")
    print(f"source_rgb={rgb}")
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
    report_path = reconstruction / "reconstruction_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("model_id") != "DA3MONO-LARGE":
        raise QualificationError(f"unexpected model_id: {report.get('model_id')}")
    if report.get("tile_count") != 49:
        raise QualificationError(f"expected 49 production tiles; got {report.get('tile_count')}")

    run_cli(root, "calibrate-dem", str(rdsm), str(calibration), str(metric))
    require_surface(metric)

    freeze = {
        "schema_version": 1,
        "status": "FROZEN_PRE_REFERENCE_V3",
        "git_head": HEAD,
        "geometry_pipeline_revision": PIPELINE_REVISION,
        "tile_size": TILE_SIZE,
        "overlap": OVERLAP,
        "stride": TARGET_STRIDE,
        "source_rgb": {"path": str(rgb), "sha256": sha256(rgb)},
        "model_checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "relative_dsm": {"path": str(rdsm.resolve()), "sha256": sha256(rdsm)},
        "metric_dsm": {"path": str(metric.resolve()), "sha256": sha256(metric)},
        "calibration_evidence": {"path": str(calibration), "sha256": sha256(calibration)},
        "reconstruction_report": report,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
    }
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print("PASS_V3_POTSDAM_PRE_REFERENCE_FREEZE")
    print(f"freeze={freeze_path}")
    print(f"rdsm_sha256={freeze['relative_dsm']['sha256']}")
    print(f"metric_dsm_sha256={freeze['metric_dsm']['sha256']}")

    old_survey = survey(old_rdsm)
    v2_survey = survey(v2_rdsm)
    v3_survey = survey(rdsm)
    diagnostic = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_NO_PASS_CLAIM",
        "git_head": HEAD,
        "geometry_pipeline_revision": PIPELINE_REVISION,
        "target_stride_px": TARGET_STRIDE,
        "control_periods_px": list(CONTROL_PERIODS),
        "old_known_defective": {
            "path": str(old_rdsm),
            "sha256": sha256(old_rdsm),
            "period_survey": old_survey,
        },
        "v2_scene_global_affine": {
            "path": str(v2_rdsm),
            "sha256": sha256(v2_rdsm),
            "period_survey": v2_survey,
        },
        "v3_global_scaffold_residual": {
            "path": str(rdsm.resolve()),
            "sha256": sha256(rdsm),
            "period_survey": v3_survey,
        },
        "v3_over_old": comparison(v3_survey, old_survey),
        "v3_over_v2": comparison(v3_survey, v2_survey),
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "purpose": "reference-free negative-control test for exact 896 px inference-grid phase locking",
    }
    diagnostic_path.write_text(json.dumps(diagnostic, indent=2), encoding="utf-8")
    print(json.dumps({
        "old_896": target_metrics(old_survey),
        "v2_896": target_metrics(v2_survey),
        "v3_896": target_metrics(v3_survey),
        "v3_over_old": diagnostic["v3_over_old"],
        "v3_over_v2": diagnostic["v3_over_v2"],
    }, indent=2))
    print("PASS_V3_STRIDE_PHASE_DIAGNOSTIC_GENERATED")
    print(f"diagnostic={diagnostic_path}")
    print("reference_values_consumed=false")
    print("reference_raster_opened_or_hashed=false")
    print("DIAGNOSTIC_ONLY_NO_PASS_CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
