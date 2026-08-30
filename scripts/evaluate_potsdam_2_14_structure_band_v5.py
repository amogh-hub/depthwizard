from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from depthwizard.calibration.robust import robust_affine_calibration
from depthwizard.evaluation.metrics import compute_elevation_metrics, compute_slope_metrics
from depthwizard.evaluation.potsdam import (
    POTSDAM_BENCHMARK_GSD_M,
    POTSDAM_CRS,
    POTSDAM_NATIVE_GSD_M,
    resolve_potsdam_tile_paths,
)
from depthwizard.height_model.model import DepthWizardHeightModel, HeightModelConfig
from depthwizard.height_model.structure_band import (
    physical_highpass_numpy,
    project_structure_correction_numpy,
)
from depthwizard.height_model.training import fit_rgb_ranges, normalize_rgb
from depthwizard.provenance.manifest import sha256_file
from scripts import evaluate_potsdam_external as potsdam_eval
from scripts import train_ortholoc_multiscene as legacy
from scripts import train_ortholoc_structure_band_v5 as v5

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "external" / "isprs-potsdam"
V2_ROOT = ROOT / "workspace" / "urban-mosaic-corrective" / "5e87670-potsdam-2_14"
OUT_DIR = ROOT / "artifacts" / "training" / "ortholoc-structure-band-v5" / "potsdam-2_14-exposed"
CHECKPOINT_PATH = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-structure-band-v5"
    / "height_model_structure_band_v5.pt"
)
EXPECTED_CHECKPOINT_SHA256 = "3fcc1423abaffdeedea5f56ef360866e8d72d32d36d6452d559e40ae070b8abf"
EXPECTED_V2_RDSM_SHA256 = "620b0430d0b22c7854733cc61bddd319f4769d9d273f87d29baa7a396764a35e"
EXPECTED_V2_DSM_SHA256 = "8bae324c5c6732d92dacd4af0bb321849a85eece0792f80526f369356ef59fe7"
EXPECTED_REFERENCE_SHA256 = "fdac03cdee3eb36ccf194f7782dc729150385ef0ea820817800ea83c65447046"
TILE_ID = "2_14"


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_tif_by_sha(root: Path, expected_sha256: str, label: str) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"missing {label} search root: {root}")
    candidates = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )
    if not candidates:
        raise FileNotFoundError(f"no TIFF candidates found while locating {label}: {root}")
    for path in candidates:
        if sha256_file(path) == expected_sha256:
            return path
    rendered = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"could not locate frozen {label} with SHA-256 {expected_sha256} under {root}.\n"
        f"Candidates inspected:\n{rendered}"
    )


def _load_model(device: torch.device) -> DepthWizardHeightModel:
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"missing frozen V5 checkpoint: {CHECKPOINT_PATH}")
    actual = sha256_file(CHECKPOINT_PATH)
    if actual != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"V5 checkpoint SHA mismatch: expected {EXPECTED_CHECKPOINT_SHA256}, got {actual}"
        )
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("V5 checkpoint payload is not a dictionary")
    config_payload = checkpoint.get("config")
    state_dict = checkpoint.get("state_dict")
    if not isinstance(config_payload, dict) or not isinstance(state_dict, dict):
        raise TypeError("V5 checkpoint is missing config/state_dict")
    model = DepthWizardHeightModel(HeightModelConfig(**config_payload))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _target_grid(rgb_path: Path) -> tuple[int, int, object]:
    with rasterio.open(rgb_path) as src:
        width = round(src.width * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        height = round(src.height * POTSDAM_NATIVE_GSD_M / POTSDAM_BENCHMARK_GSD_M)
        transform = from_bounds(*src.bounds, width=width, height=height)
    return height, width, transform


def _validate_native_grid(candidate: Path, rgb_path: Path, label: str) -> None:
    with rasterio.open(candidate) as src, rasterio.open(rgb_path) as rgb:
        if src.width != rgb.width or src.height != rgb.height:
            raise ValueError(
                f"{label} native shape mismatch: {src.width}x{src.height} vs RGB {rgb.width}x{rgb.height}"
            )
        if not src.transform.almost_equals(rgb.transform):
            raise ValueError(f"{label} affine transform does not match Potsdam RGB")
        src_crs = src.crs or POTSDAM_CRS
        rgb_crs = rgb.crs or POTSDAM_CRS
        if src_crs != POTSDAM_CRS or rgb_crs != POTSDAM_CRS:
            raise ValueError(f"{label} and RGB must resolve to EPSG:32633")


def _read_rgb_025m(path: Path, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, height, width),
            resampling=Resampling.average,
        )
        mask = src.dataset_mask(out_shape=(height, width), resampling=Resampling.nearest) > 0
    return np.moveaxis(rgb, 0, -1), mask


def _read_float_025m(
    path: Path,
    height: int,
    width: int,
    *,
    resampling: Resampling = Resampling.average,
) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=resampling,
        )
    values = np.asarray(data.filled(np.nan), dtype=np.float32)
    valid = np.isfinite(values)
    return values, valid


def _write_float(path: Path, values: np.ndarray, transform: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(values, dtype=np.float32)
    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": POTSDAM_CRS,
        "transform": transform,
        "nodata": -9999.0,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": min(512, max(16, (arr.shape[1] // 16) * 16)),
        "blockysize": min(512, max(16, (arr.shape[0] // 16) * 16)),
    }
    encoded = np.where(np.isfinite(arr), arr, np.float32(-9999.0))
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(encoded.astype(np.float32), 1)
        dst.update_tags(
            DEPTHWIZARD_RESEARCH_ONLY="true",
            POTSDAM_TILE=TILE_ID,
            POTSDAM_BLIND_STATUS="EXPOSED_DEVELOPMENT_ONLY",
            V5_CHECKPOINT_SHA256=EXPECTED_CHECKPOINT_SHA256,
        )


def _sample_scale_fit(
    rdsm: np.ndarray,
    dsm: np.ndarray,
    valid: np.ndarray,
) -> tuple[float, float, float]:
    indices = np.flatnonzero(valid & np.isfinite(rdsm) & np.isfinite(dsm))
    if indices.size < 1000:
        raise RuntimeError("insufficient valid pixels to recover frozen v2 relative-to-metric scale")
    rng = np.random.default_rng(26175)
    if indices.size > 200_000:
        indices = rng.choice(indices, size=200_000, replace=False)
    fit = robust_affine_calibration(
        rdsm.ravel()[indices],
        dsm.ravel()[indices],
        require_positive_scale=True,
    )
    return float(fit.scale), float(fit.offset), float(fit.rmse_anchor)


def _rmse(values: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[valid & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("no valid values available for RMSE")
    return float(np.sqrt(np.mean(selected**2)))


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid & np.isfinite(a) & np.isfinite(b)
    av = np.asarray(a, dtype=np.float64)[mask]
    bv = np.asarray(b, dtype=np.float64)[mask]
    if av.size < 2 or float(np.std(av)) <= 1e-12 or float(np.std(bv)) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def _structure_scale_report(
    base_dsm: np.ndarray,
    refined_dsm: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    scale_m: float,
) -> dict[str, object]:
    base_error = base_dsm - reference
    refined_error = refined_dsm - reference
    base_error_hp = physical_highpass_numpy(
        base_error,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        characteristic_scale_m=scale_m,
    )
    refined_error_hp = physical_highpass_numpy(
        refined_error,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        characteristic_scale_m=scale_m,
    )
    base_surface_hp = physical_highpass_numpy(
        base_dsm,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        characteristic_scale_m=scale_m,
    )
    refined_surface_hp = physical_highpass_numpy(
        refined_dsm,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        characteristic_scale_m=scale_m,
    )
    reference_hp = physical_highpass_numpy(
        reference,
        valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        characteristic_scale_m=scale_m,
    )
    hp_valid = (
        valid
        & np.isfinite(base_error_hp)
        & np.isfinite(refined_error_hp)
        & np.isfinite(base_surface_hp)
        & np.isfinite(refined_surface_hp)
        & np.isfinite(reference_hp)
    )
    base_rmse = _rmse(base_error_hp, hp_valid)
    refined_rmse = _rmse(refined_error_hp, hp_valid)
    return {
        "characteristic_scale_m": scale_m,
        "valid_pixels": int(hp_valid.sum()),
        "base_highpass_error_rmse_m": base_rmse,
        "refined_highpass_error_rmse_m": refined_rmse,
        "rmse_delta_m": refined_rmse - base_rmse,
        "rmse_improvement_fraction": (base_rmse - refined_rmse) / base_rmse,
        "base_structure_pearson_r": _corr(base_surface_hp, reference_hp, hp_valid),
        "refined_structure_pearson_r": _corr(refined_surface_hp, reference_hp, hp_valid),
    }


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    torch.manual_seed(26175)
    np.random.seed(26175)
    torch.set_float32_matmul_precision("high")
    device = _resolve_device()

    tile = resolve_potsdam_tile_paths(DATASET_ROOT, TILE_ID)
    reference_sha = sha256_file(tile.reference_dsm)
    if reference_sha != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError(
            f"Potsdam 2_14 reference SHA mismatch: expected {EXPECTED_REFERENCE_SHA256}, got {reference_sha}"
        )

    v2_rdsm_path = _find_tif_by_sha(V2_ROOT, EXPECTED_V2_RDSM_SHA256, "v2 rDSM")
    v2_dsm_path = _find_tif_by_sha(V2_ROOT, EXPECTED_V2_DSM_SHA256, "v2 metric DSM")
    _validate_native_grid(v2_rdsm_path, tile.rgb, "v2 rDSM")
    _validate_native_grid(v2_dsm_path, tile.rgb, "v2 metric DSM")
    _validate_native_grid(tile.reference_dsm, tile.rgb, "Potsdam reference DSM")

    height, width, transform = _target_grid(tile.rgb)
    rgb_raw, rgb_valid = _read_rgb_025m(tile.rgb, height, width)
    rdsm, rdsm_valid = _read_float_025m(v2_rdsm_path, height, width)
    base_dsm, dsm_valid = _read_float_025m(v2_dsm_path, height, width)
    reference, reference_valid = _read_float_025m(tile.reference_dsm, height, width)

    input_valid = rgb_valid & rdsm_valid & np.isfinite(rdsm)
    evaluation_valid = input_valid & dsm_valid & reference_valid & np.isfinite(base_dsm) & np.isfinite(reference)
    if int(evaluation_valid.sum()) < 100_000:
        raise RuntimeError("Potsdam 2_14 benchmark grid has insufficient common valid pixels")

    ranges = fit_rgb_ranges(rgb_raw, input_valid)
    rgb = normalize_rgb(rgb_raw, ranges)
    model = _load_model(device)

    print(
        "Potsdam 2_14 exposed-development transfer: frozen v2 broad terrain/calibration is preserved; "
        "V5 contributes only a 16 m safety-projected local structure correction."
    )
    print(
        "Official Potsdam DSM is evaluation-only. No reference value is used to train V5 or fit the "
        "v2 relative-to-metric scale in this diagnostic."
    )
    print(f"Working grid: {width}x{height} at {POTSDAM_BENCHMARK_GSD_M:.2f} m GSD")
    print(f"v2 rDSM: {v2_rdsm_path}")
    print(f"v2 DSM:  {v2_dsm_path}")

    raw_refined, uncertainty, covered = potsdam_eval._predict_refined(
        model,
        rgb,
        rdsm,
        input_valid,
        device,
    )
    correction_valid = input_valid & covered & np.isfinite(raw_refined)
    raw_correction = raw_refined - rdsm
    safe_correction = project_structure_correction_numpy(
        raw_correction,
        correction_valid,
        gsd_m=POTSDAM_BENCHMARK_GSD_M,
        config=v5.STRUCTURE,
    )
    safe_valid = correction_valid & np.isfinite(safe_correction)

    scale_fit_valid = input_valid & dsm_valid
    metric_scale, metric_offset, affine_reconstruction_rmse = _sample_scale_fit(
        rdsm,
        base_dsm,
        scale_fit_valid,
    )
    refined_dsm = np.full_like(base_dsm, np.nan, dtype=np.float32)
    refined_valid = safe_valid & dsm_valid & np.isfinite(base_dsm)
    refined_dsm[refined_valid] = (
        base_dsm[refined_valid]
        + safe_correction[refined_valid] * np.float32(metric_scale)
    )

    common = evaluation_valid & refined_valid & np.isfinite(refined_dsm)
    base_metrics = compute_elevation_metrics(base_dsm, reference, valid_mask=common)
    refined_metrics = compute_elevation_metrics(refined_dsm, reference, valid_mask=common)
    base_slope = compute_slope_metrics(
        base_dsm,
        reference,
        gsd_x=POTSDAM_BENCHMARK_GSD_M,
        gsd_y=POTSDAM_BENCHMARK_GSD_M,
        valid_mask=common,
    )
    refined_slope = compute_slope_metrics(
        refined_dsm,
        reference,
        gsd_x=POTSDAM_BENCHMARK_GSD_M,
        gsd_y=POTSDAM_BENCHMARK_GSD_M,
        valid_mask=common,
    )

    structure_scales = {
        f"{scale:g}m": _structure_scale_report(
            base_dsm,
            refined_dsm,
            reference,
            common,
            scale,
        )
        for scale in (2.0, 4.0, 8.0)
    }

    correction_m = np.full_like(base_dsm, np.nan, dtype=np.float32)
    correction_m[refined_valid] = safe_correction[refined_valid] * np.float32(metric_scale)
    selected_correction = np.abs(correction_m[common])
    correction_stats = {
        "mean_abs_m": float(np.mean(selected_correction)),
        "p50_abs_m": float(np.percentile(selected_correction, 50)),
        "p90_abs_m": float(np.percentile(selected_correction, 90)),
        "p95_abs_m": float(np.percentile(selected_correction, 95)),
        "p99_abs_m": float(np.percentile(selected_correction, 99)),
        "max_abs_m": float(np.max(selected_correction)),
    }

    overall_non_degrading = refined_metrics.rmse_m <= base_metrics.rmse_m + 1e-9
    slope_non_degrading = refined_slope.rmse_degrees <= base_slope.rmse_degrees + 1e-9
    structure_8m_improves = (
        structure_scales["8m"]["refined_highpass_error_rmse_m"]
        < structure_scales["8m"]["base_highpass_error_rmse_m"]
    )
    promising = bool(overall_non_degrading and slope_non_degrading and structure_8m_improves)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_float(OUT_DIR / "v2_base_dsm_025m.tif", base_dsm, transform)
    _write_float(OUT_DIR / "v5_refined_dsm_025m.tif", refined_dsm, transform)
    _write_float(OUT_DIR / "v5_structure_correction_m_025m.tif", correction_m, transform)
    _write_float(OUT_DIR / "reference_dsm_025m.tif", reference, transform)
    _write_float(OUT_DIR / "v2_abs_error_025m.tif", np.abs(base_dsm - reference), transform)
    _write_float(OUT_DIR / "v5_abs_error_025m.tif", np.abs(refined_dsm - reference), transform)

    payload = {
        "status": (
            "PROMISING_V5_POTSDAM_2_14_EXPOSED_TRANSFER"
            if promising
            else "REJECT_V5_POTSDAM_2_14_EXPOSED_TRANSFER"
        ),
        "research_only": True,
        "potsdam_tile": TILE_ID,
        "potsdam_blind_status": "EXPOSED_DEVELOPMENT_ONLY",
        "blind_potsdam_4_12_6_12_touched": False,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "v2_rdsm_sha256": EXPECTED_V2_RDSM_SHA256,
        "v2_dsm_sha256": EXPECTED_V2_DSM_SHA256,
        "reference_sha256": EXPECTED_REFERENCE_SHA256,
        "working_gsd_m": POTSDAM_BENCHMARK_GSD_M,
        "working_shape": [height, width],
        "metric_correction_scale": {
            "scale_m_per_relative_unit": metric_scale,
            "offset_m_diagnostic_only": metric_offset,
            "v2_affine_reconstruction_rmse_m": affine_reconstruction_rmse,
            "semantics": (
                "Only the recovered positive v2 scale converts V5 local relative correction to metres. "
                "The frozen v2 metric DSM remains the broad-terrain base; the recovered offset is not "
                "reapplied."
            ),
        },
        "base_elevation": base_metrics.model_dump(),
        "refined_elevation": refined_metrics.model_dump(),
        "rmse_delta_m": float(refined_metrics.rmse_m - base_metrics.rmse_m),
        "rmse_improvement_fraction": float(
            (base_metrics.rmse_m - refined_metrics.rmse_m) / base_metrics.rmse_m
        ),
        "base_slope": base_slope.model_dump(),
        "refined_slope": refined_slope.model_dump(),
        "slope_rmse_delta_degrees": float(
            refined_slope.rmse_degrees - base_slope.rmse_degrees
        ),
        "structure_scales": structure_scales,
        "correction_stats_m": correction_stats,
        "acceptance": {
            "overall_rmse_non_degrading": overall_non_degrading,
            "slope_rmse_non_degrading": slope_non_degrading,
            "eight_m_structure_error_improves": structure_8m_improves,
            "promising_for_next_operator_stage": promising,
        },
        "artifacts": {
            "base_dsm": str((OUT_DIR / "v2_base_dsm_025m.tif").resolve()),
            "refined_dsm": str((OUT_DIR / "v5_refined_dsm_025m.tif").resolve()),
            "structure_correction_m": str(
                (OUT_DIR / "v5_structure_correction_m_025m.tif").resolve()
            ),
            "reference_dsm": str((OUT_DIR / "reference_dsm_025m.tif").resolve()),
            "base_abs_error": str((OUT_DIR / "v2_abs_error_025m.tif").resolve()),
            "refined_abs_error": str((OUT_DIR / "v5_abs_error_025m.tif").resolve()),
        },
    }
    report_path = OUT_DIR / "potsdam_2_14_v5_exposed_transfer.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== OVERALL ELEVATION ===")
    print(
        f"v2 RMSE {base_metrics.rmse_m:.4f} m | V5 {refined_metrics.rmse_m:.4f} m | "
        f"delta {refined_metrics.rmse_m - base_metrics.rmse_m:+.4f} m | "
        f"improvement {100.0 * payload['rmse_improvement_fraction']:.2f}%"
    )
    print(
        f"v2 MAE {base_metrics.mae_m:.4f} m | V5 {refined_metrics.mae_m:.4f} m | "
        f"Pearson {base_metrics.pearson_r} -> {refined_metrics.pearson_r}"
    )

    print("\n=== SLOPE ===")
    print(
        f"v2 slope RMSE {base_slope.rmse_degrees:.4f} deg | "
        f"V5 {refined_slope.rmse_degrees:.4f} deg | "
        f"delta {payload['slope_rmse_delta_degrees']:+.4f} deg"
    )

    print("\n=== STRUCTURE-BAND ERROR ===")
    for key, report in structure_scales.items():
        print(
            f"{key}: v2 {report['base_highpass_error_rmse_m']:.4f} m | "
            f"V5 {report['refined_highpass_error_rmse_m']:.4f} m | "
            f"improvement {100.0 * report['rmse_improvement_fraction']:.2f}% | "
            f"corr {report['base_structure_pearson_r']} -> {report['refined_structure_pearson_r']}"
        )

    print("\n=== CORRECTION MAGNITUDE ===")
    print(json.dumps(correction_stats, indent=2))
    print(f"\nPotsdam 2_14 exposed transfer: {payload['status']}")
    print(f"Report: {report_path}")
    print(f"Refined DSM for visual inspection: {payload['artifacts']['refined_dsm']}")


if __name__ == "__main__":
    main()
