#!/usr/bin/env python3
"""Analyze the exposed Potsdam 2_14 residual to design a structure-specific refiner.

This is development-only analysis. The official reference has already been exposed by the frozen
candidate comparison, so this script MUST NOT be used as blind acceptance evidence. It does not
modify any prediction or production source.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import rasterio
from scipy.ndimage import gaussian_filter

HEAD = "d2c94772a60eea6b113107c3a3f2985503802f87"
V2_RDSM_REL = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/reconstruction/rdsm.tif")
V2_RDSM_SHA256 = "620b0430d0b22c7854733cc61bddd319f4769d9d273f87d29baa7a396764a35e"
V2_METRIC_REL = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14/metric-dsm.tif")
V2_METRIC_SHA256 = "8bae324c5c6732d92dacd4af0bb321849a85eece0792f80526f369356ef59fe7"
REFERENCE_REL = Path("data/external/isprs-potsdam/1_DSM/dsm_potsdam_02_14.tif")
REFERENCE_SHA256 = "fdac03cdee3eb36ccf194f7782dc729150385ef0ea820817800ea83c65447046"
OUT_REL = Path("workspace/potsdam-2-14-structure-residual-analysis/structure-residual-analysis.json")
OLD_MAX_RELATIVE_CORRECTION = 0.35
PHYSICAL_SCALES_M = (0.5, 1.0, 2.0, 4.0, 8.0)


class DiagnosticError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> Path:
    path = path.resolve(strict=True)
    actual = sha256(path)
    if actual != expected:
        raise DiagnosticError(f"{label} SHA mismatch: {actual} != {expected}")
    return path


def finite_quantiles(values: np.ndarray, qs: tuple[float, ...]) -> dict[str, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise DiagnosticError("no finite values available for quantiles")
    return {f"p{int(q):02d}": float(np.percentile(finite, q)) for q in qs}


def fit_exact_metric_affine(relative: np.ndarray, metric: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    # Metric DSM is generated from the relative field by one scene-level affine calibration. A
    # sparse deterministic sample is enough to recover that persisted mapping without reference use.
    yy, xx = np.nonzero(valid)
    if yy.size < 1000:
        raise DiagnosticError("not enough common rDSM/metric pixels")
    step = max(1, yy.size // 200_000)
    r = relative[yy[::step], xx[::step]].astype(np.float64)
    m = metric[yy[::step], xx[::step]].astype(np.float64)
    design = np.column_stack((r, np.ones_like(r)))
    scale, offset = np.linalg.lstsq(design, m, rcond=None)[0]
    residual = m - (scale * r + offset)
    return float(scale), float(offset), float(np.sqrt(np.mean(residual * residual)))


def normalized_gaussian(values: np.ndarray, valid: np.ndarray, sigma_px: float) -> np.ndarray:
    # Mask-normalized filtering prevents nodata from contaminating local low-frequency estimates.
    data = np.where(valid, values, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    numerator = gaussian_filter(data, sigma=sigma_px, mode="reflect")
    denominator = gaussian_filter(weight, sigma=sigma_px, mode="reflect")
    out = np.full(values.shape, np.nan, dtype=np.float32)
    usable = valid & (denominator > 1e-6)
    out[usable] = numerator[usable] / denominator[usable]
    return out


def rmse(values: np.ndarray, valid: np.ndarray) -> float:
    selected = values[valid & np.isfinite(values)].astype(np.float64)
    return float(np.sqrt(np.mean(selected * selected)))


def main() -> None:
    root = Path.cwd().resolve(strict=True)
    current = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if current != HEAD:
        raise DiagnosticError(f"requires exact checkout {HEAD}; current={current}")
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip():
        raise DiagnosticError("repository must be clean")

    rdsm_path = require_hash(root / V2_RDSM_REL, V2_RDSM_SHA256, "v2 rDSM")
    metric_path = require_hash(root / V2_METRIC_REL, V2_METRIC_SHA256, "v2 metric DSM")
    reference_path = require_hash(root / REFERENCE_REL, REFERENCE_SHA256, "Potsdam reference")

    with rasterio.open(rdsm_path) as src:
        rdsm = src.read(1).astype(np.float32)
        rdsm_valid = (src.read_masks(1) > 0) & np.isfinite(rdsm)
    with rasterio.open(metric_path) as src:
        metric = src.read(1).astype(np.float32)
        metric_valid = (src.read_masks(1) > 0) & np.isfinite(metric)
        gsd_x = abs(float(src.transform.a))
        gsd_y = abs(float(src.transform.e))
    with rasterio.open(reference_path) as src:
        reference = src.read(1).astype(np.float32)
        reference_valid = (src.read_masks(1) > 0) & np.isfinite(reference)

    if rdsm.shape != metric.shape or metric.shape != reference.shape:
        raise DiagnosticError(f"shape mismatch: rDSM={rdsm.shape}, metric={metric.shape}, ref={reference.shape}")
    valid = rdsm_valid & metric_valid & reference_valid
    if int(valid.sum()) < 1_000_000:
        raise DiagnosticError("unexpectedly small valid support")

    scale, offset, affine_reconstruction_rmse = fit_exact_metric_affine(rdsm, metric, valid)
    old_cap_m = abs(scale) * OLD_MAX_RELATIVE_CORRECTION
    correction = reference - metric
    abs_correction = np.abs(correction)

    structure_scales: dict[str, object] = {}
    for physical_m in PHYSICAL_SCALES_M:
        # sigma is half the named physical support scale; ~95% Gaussian support spans ~4 sigma.
        sigma_m = physical_m / 2.0
        sigma_px = max(0.75, sigma_m / max((gsd_x + gsd_y) * 0.5, 1e-9))
        low = normalized_gaussian(correction, valid, sigma_px)
        high = correction - low
        high_valid = valid & np.isfinite(high)
        high_abs = np.abs(high)
        structure_mask = high_valid & (high_abs >= 2.0)
        structure_scales[f"{physical_m:.1f}m"] = {
            "gaussian_sigma_px": float(sigma_px),
            "total_residual_rmse_m": rmse(correction, valid),
            "high_frequency_residual_rmse_m": rmse(high, high_valid),
            "high_frequency_energy_fraction": float(
                (rmse(high, high_valid) ** 2) / max(rmse(correction, valid) ** 2, 1e-12)
            ),
            "pixels_abs_high_frequency_ge_2m_fraction": float(structure_mask.sum() / high_valid.sum()),
            "abs_high_frequency_quantiles_m": finite_quantiles(high_abs[high_valid], (50, 75, 90, 95, 99)),
            "old_v4_cap_exceeded_within_structure_fraction": (
                float((high_abs[structure_mask] > old_cap_m).mean()) if np.any(structure_mask) else 0.0
            ),
        }

    global_abs = abs_correction[valid]
    report = {
        "schema_version": 1,
        "status": "PASS_EXPOSED_STRUCTURE_RESIDUAL_DIAGNOSTIC",
        "scene": "ISPRS Potsdam 2_14",
        "blind_status": "EXPOSED_DEVELOPMENT_ONLY",
        "source_head": HEAD,
        "baseline": "v2_scene_global_affine",
        "v2_rdsm_sha256": V2_RDSM_SHA256,
        "v2_metric_dsm_sha256": V2_METRIC_SHA256,
        "reference_sha256": REFERENCE_SHA256,
        "valid_pixels": int(valid.sum()),
        "pixel_size_map_units": {"x": gsd_x, "y": gsd_y},
        "recovered_metric_affine": {
            "scale_m_per_relative_unit": scale,
            "offset_m": offset,
            "reconstruction_rmse_m": affine_reconstruction_rmse,
        },
        "old_v4_architecture_cap": {
            "max_relative_correction": OLD_MAX_RELATIVE_CORRECTION,
            "maximum_metric_correction_before_gate_m": old_cap_m,
            "global_reference_residual_exceeding_cap_fraction": float((global_abs > old_cap_m).mean()),
        },
        "reference_minus_v2_metric_residual": {
            "mean_m": float(np.mean(correction[valid], dtype=np.float64)),
            "rmse_m": rmse(correction, valid),
            "abs_quantiles_m": finite_quantiles(global_abs, (50, 75, 90, 95, 99, 99.5)),
        },
        "physical_scale_decomposition": structure_scales,
        "interpretation_boundary": (
            "Development diagnostic only. It may guide architecture/loss design on Potsdam 2_14, "
            "but no final acceptance claim may be made on this exposed scene."
        ),
    }
    output = root / OUT_REL
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("PASS_EXPOSED_STRUCTURE_RESIDUAL_DIAGNOSTIC")
    print(f"report={output}")


if __name__ == "__main__":
    main()
