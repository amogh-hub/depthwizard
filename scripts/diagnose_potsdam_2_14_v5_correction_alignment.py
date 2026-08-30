from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from depthwizard.evaluation.potsdam import (
    POTSDAM_BENCHMARK_GSD_M,
    resolve_potsdam_tile_paths,
)
from depthwizard.height_model.structure_band import physical_highpass_numpy
from depthwizard.provenance.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "data" / "external" / "isprs-potsdam"
V2_ROOT = ROOT / "workspace" / "urban-mosaic-corrective" / "5e87670-potsdam-2_14"
V5_ROOT = (
    ROOT
    / "artifacts"
    / "training"
    / "ortholoc-structure-band-v5"
    / "potsdam-2_14-exposed"
)
REFINED_PATH = V5_ROOT / "v5_refined_dsm_025m.tif"
OUT_PATH = V5_ROOT / "potsdam_2_14_v5_correction_alignment.json"
TILE_ID = "2_14"
EXPECTED_V2_DSM_SHA256 = "8bae324c5c6732d92dacd4af0bb321849a85eece0792f80526f369356ef59fe7"
EXPECTED_REFERENCE_SHA256 = "fdac03cdee3eb36ccf194f7782dc729150385ef0ea820817800ea83c65447046"


def _find_tif_by_sha(root: Path, expected_sha256: str, label: str) -> Path:
    for path in sorted(root.rglob("*.tif")):
        if sha256_file(path) == expected_sha256:
            return path
    for path in sorted(root.rglob("*.tiff")):
        if sha256_file(path) == expected_sha256:
            return path
    raise FileNotFoundError(f"could not locate {label} with SHA-256 {expected_sha256} under {root}")


def _read_025m(path: Path, height: int = 1200, width: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        data = src.read(
            1,
            out_shape=(height, width),
            masked=True,
            resampling=Resampling.average,
        )
    values = np.asarray(data.filled(np.nan), dtype=np.float32)
    return values, np.isfinite(values)


def _corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid & np.isfinite(a) & np.isfinite(b)
    av = np.asarray(a, dtype=np.float64)[mask]
    bv = np.asarray(b, dtype=np.float64)[mask]
    if av.size < 2 or float(np.std(av)) <= 1e-12 or float(np.std(bv)) <= 1e-12:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def _rmse(values: np.ndarray, valid: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)[valid & np.isfinite(values)]
    if selected.size == 0:
        raise ValueError("no valid values")
    return float(np.sqrt(np.mean(selected**2)))


def _optimal_gain(correction: np.ndarray, needed: np.ndarray, valid: np.ndarray) -> float | None:
    mask = valid & np.isfinite(correction) & np.isfinite(needed)
    c = np.asarray(correction, dtype=np.float64)[mask]
    n = np.asarray(needed, dtype=np.float64)[mask]
    denominator = float(np.dot(c, c))
    if denominator <= 1e-12:
        return None
    return float(np.dot(c, n) / denominator)


def _sign_agreement(correction: np.ndarray, needed: np.ndarray, valid: np.ndarray, percentile: float) -> dict[str, float | int | None]:
    mask = valid & np.isfinite(correction) & np.isfinite(needed)
    abs_needed = np.abs(needed[mask])
    if abs_needed.size == 0:
        return {"pixels": 0, "threshold_m": None, "agreement_fraction": None}
    threshold = float(np.percentile(abs_needed, percentile))
    selected = mask & (np.abs(needed) >= threshold)
    count = int(selected.sum())
    if count == 0:
        return {"pixels": 0, "threshold_m": threshold, "agreement_fraction": None}
    agreement = np.sign(correction[selected]) == np.sign(needed[selected])
    return {
        "pixels": count,
        "threshold_m": threshold,
        "agreement_fraction": float(np.mean(agreement)),
    }


def _alignment_report(correction: np.ndarray, needed: np.ndarray, valid: np.ndarray) -> dict[str, object]:
    gain = _optimal_gain(correction, needed, valid)
    baseline_rmse = _rmse(needed, valid)
    current_rmse = _rmse(needed - correction, valid)
    optimal_rmse = None
    if gain is not None:
        optimal_rmse = _rmse(needed - gain * correction, valid)
    return {
        "valid_pixels": int((valid & np.isfinite(correction) & np.isfinite(needed)).sum()),
        "correction_needed_pearson_r": _corr(correction, needed, valid),
        "optimal_scalar_gain_diagnostic_only": gain,
        "zero_correction_rmse_m": baseline_rmse,
        "current_v5_correction_rmse_m": current_rmse,
        "optimal_scalar_rmse_m_diagnostic_only": optimal_rmse,
        "sign_agreement_top50_needed": _sign_agreement(correction, needed, valid, 50.0),
        "sign_agreement_top75_needed": _sign_agreement(correction, needed, valid, 75.0),
        "sign_agreement_top90_needed": _sign_agreement(correction, needed, valid, 90.0),
    }


def main() -> None:
    tile = resolve_potsdam_tile_paths(DATASET_ROOT, TILE_ID)
    if sha256_file(tile.reference_dsm) != EXPECTED_REFERENCE_SHA256:
        raise RuntimeError("Potsdam 2_14 reference SHA mismatch")
    base_path = _find_tif_by_sha(V2_ROOT, EXPECTED_V2_DSM_SHA256, "frozen v2 metric DSM")
    if not REFINED_PATH.is_file():
        raise FileNotFoundError(f"missing V5 exposed-transfer DSM: {REFINED_PATH}")

    base, base_valid = _read_025m(base_path)
    reference, reference_valid = _read_025m(tile.reference_dsm)
    refined, refined_valid = _read_025m(REFINED_PATH)
    valid = base_valid & reference_valid & refined_valid
    if int(valid.sum()) < 100_000:
        raise RuntimeError("insufficient common valid pixels for correction-alignment diagnostic")

    correction = refined - base
    needed = reference - base

    report: dict[str, object] = {
        "status": "POTSDAM_2_14_V5_CORRECTION_ALIGNMENT_DIAGNOSTIC",
        "research_only": True,
        "potsdam_tile": TILE_ID,
        "potsdam_blind_status": "EXPOSED_DEVELOPMENT_ONLY",
        "blind_potsdam_4_12_6_12_touched": False,
        "base_dsm": str(base_path.resolve()),
        "refined_dsm": str(REFINED_PATH.resolve()),
        "reference_dsm": str(tile.reference_dsm.resolve()),
        "overall": _alignment_report(correction, needed, valid),
        "physical_scales": {},
    }

    print("Potsdam 2_14 V5 correction-alignment diagnostic")
    print("This is diagnosis only: the official exposed reference is not used to train or promote V5.")
    print("Blind Potsdam 4_12 and 6_12 are not accessed.")

    overall = report["overall"]
    assert isinstance(overall, dict)
    print("\n=== RAW LOCAL CORRECTION VS NEEDED RESIDUAL ===")
    print(
        f"corr={overall['correction_needed_pearson_r']} | "
        f"optimal gain={overall['optimal_scalar_gain_diagnostic_only']} | "
        f"zero RMSE={overall['zero_correction_rmse_m']:.4f} m | "
        f"V5 RMSE={overall['current_v5_correction_rmse_m']:.4f} m | "
        f"optimal-scalar RMSE={overall['optimal_scalar_rmse_m_diagnostic_only']}"
    )

    physical = report["physical_scales"]
    assert isinstance(physical, dict)
    for scale_m in (2.0, 4.0, 8.0):
        correction_hp = physical_highpass_numpy(
            correction,
            valid,
            gsd_m=POTSDAM_BENCHMARK_GSD_M,
            characteristic_scale_m=scale_m,
        )
        needed_hp = physical_highpass_numpy(
            needed,
            valid,
            gsd_m=POTSDAM_BENCHMARK_GSD_M,
            characteristic_scale_m=scale_m,
        )
        hp_valid = valid & np.isfinite(correction_hp) & np.isfinite(needed_hp)
        scale_report = _alignment_report(correction_hp, needed_hp, hp_valid)
        physical[f"{int(scale_m)}m"] = scale_report
        print(f"\n=== {int(scale_m)} m STRUCTURE SCALE ===")
        print(
            f"corr={scale_report['correction_needed_pearson_r']} | "
            f"optimal gain={scale_report['optimal_scalar_gain_diagnostic_only']} | "
            f"zero RMSE={scale_report['zero_correction_rmse_m']:.4f} m | "
            f"V5 RMSE={scale_report['current_v5_correction_rmse_m']:.4f} m | "
            f"optimal-scalar RMSE={scale_report['optimal_scalar_rmse_m_diagnostic_only']}"
        )
        top90 = scale_report["sign_agreement_top90_needed"]
        assert isinstance(top90, dict)
        print(
            f"top-10%-needed sign agreement={top90['agreement_fraction']} "
            f"over {top90['pixels']} pixels"
        )

    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport: {OUT_PATH}")


if __name__ == "__main__":
    main()
