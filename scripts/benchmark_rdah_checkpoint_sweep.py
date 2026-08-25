from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from benchmark_rdah_ortholoc import DA2Prior

from depthwizard.baselines.rdah import infer_rdah, load_rdah_model
from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.io.raster import read_rgb, reproject_to_match, write_float_geotiff

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark" / "ortholoc-demo"
DA3_REPORT = ROOT / "artifacts" / "benchmark" / "ortholoc-demo" / "benchmark_report.json"
SWEEP_MANIFEST = (
    ROOT / "checkpoints" / "baselines" / "rdah" / "sweep" / "checkpoint_sweep.json"
)
OUT_DIR = ROOT / "artifacts" / "benchmark" / "rdah-checkpoint-sweep"
ANCHOR_BUDGETS = (8, 16, 32, 64)


def load_manifest() -> list[dict[str, Any]]:
    if not SWEEP_MANIFEST.exists():
        raise RuntimeError("RDAH sweep manifest missing; run 'make rdah-sweep-setup' first")
    payload = json.loads(SWEEP_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("checkpoints"), list):
        raise TypeError("RDAH sweep manifest is invalid")
    result = [item for item in payload["checkpoints"] if isinstance(item, dict)]
    if not result:
        raise RuntimeError("RDAH sweep manifest contains no checkpoints")
    return result


def da3_results() -> dict[int, dict[str, Any]]:
    if not DA3_REPORT.exists():
        raise RuntimeError("DA3 benchmark report missing; run 'make benchmark-ortholoc-demo' first")
    payload = json.loads(DA3_REPORT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("DA3 benchmark report is invalid")
    results = payload.get("results")
    if not isinstance(results, list):
        raise TypeError("DA3 benchmark report has no results")
    return {
        int(item["anchor_count"]): item
        for item in results
        if isinstance(item, dict) and "anchor_count" in item
    }


def _da3_rmse(
    da3_by_anchor: dict[int, dict[str, Any]],
    anchors: int,
) -> float | None:
    item = da3_by_anchor.get(anchors)
    metrics = item.get("metrics") if isinstance(item, dict) else None
    if not isinstance(metrics, dict) or "rmse_m" not in metrics:
        return None
    return float(metrics["rmse_m"])


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    xv = np.asarray(x, dtype=np.float64).reshape(-1)
    yv = np.asarray(y, dtype=np.float64).reshape(-1)
    finite = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[finite], yv[finite]
    if xv.size < 2:
        return None
    if float(np.std(xv)) <= 1e-12 or float(np.std(yv)) <= 1e-12:
        return None
    value = float(np.corrcoef(xv, yv)[0, 1])
    return value if np.isfinite(value) else None


def _output_stats(values: np.ndarray) -> dict[str, float | int]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise ValueError("RDAH output has no finite diagnostic pixels")
    return {
        "count": int(data.size),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "p01": float(np.percentile(data, 1)),
        "p99": float(np.percentile(data, 99)),
    }


def _anchor_result(result: dict[str, Any], anchors: int) -> dict[str, Any] | None:
    entries = result.get("results")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if isinstance(entry, dict) and int(entry.get("anchor_count", -1)) == anchors:
            return entry
    return None


def benchmark_checkpoint(
    checkpoint_info: dict[str, Any],
    *,
    rgb: np.ndarray,
    da2_depth: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    da3_by_anchor: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checkpoint = Path(str(checkpoint_info["checkpoint"]))
    file_id = int(checkpoint_info["figshare_file_id"])
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    model, device = load_rdah_model(checkpoint)
    started = time.perf_counter()
    rdah_height = infer_rdah(model, rgb, da2_depth, device=device)
    elapsed = time.perf_counter() - started

    diagnostic_mask = valid & np.isfinite(rdah_height)
    raw_correlation = _pearson(rdah_height[diagnostic_mask], reference[diagnostic_mask])
    raw_stats = _output_stats(rdah_height[diagnostic_mask])

    checkpoint_dir = OUT_DIR / f"figshare-{file_id}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    height_path = checkpoint_dir / "rdah_height_like.tif"
    write_float_geotiff(
        height_path,
        rdah_height,
        template_path=DATA_DIR / "urban_residential_DOP.tif",
        description="Published RDAH-Net checkpoint output for cross-domain geometry screening",
        tags={
            "MODEL": "RDAH-Net",
            "FIGSHARE_FILE_ID": str(file_id),
            "TRAINING_DOMAIN": "unresolved_from_public_filename",
            "SEMANTICS": "cross_domain_geometry_screen_not_final_accuracy_claim",
        },
    )

    results: list[dict[str, Any]] = []
    for anchors in ANCHOR_BUDGETS:
        da3_rmse = _da3_rmse(da3_by_anchor, anchors)
        try:
            holdout = sparse_anchor_holdout_benchmark(
                rdah_height,
                reference,
                valid_mask=valid,
                anchor_count=anchors,
                seed=26175,
                exclusion_radius_px=4,
            )
        except ValueError as exc:
            results.append(
                {
                    "anchor_count": anchors,
                    "status": "FAILED_SCREEN",
                    "error": str(exc),
                    "da3_rmse_m": da3_rmse,
                }
            )
            continue

        metrics = holdout.metrics.model_dump()
        rdah_rmse = float(metrics["rmse_m"])
        results.append(
            {
                "anchor_count": anchors,
                "status": "PASS",
                "heldout_pixels": int(holdout.evaluation_mask.sum()),
                "metrics": metrics,
                "orientation_flipped": holdout.orientation_flipped,
                "anchor_correlation_before": holdout.anchor_correlation_before,
                "anchor_correlation_after": holdout.anchor_correlation_after,
                "da3_rmse_m": da3_rmse,
                "rmse_delta_m_rdah_minus_da3": (
                    rdah_rmse - da3_rmse if da3_rmse is not None else None
                ),
            }
        )

    del model
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()

    payload = {
        "figshare_file_id": file_id,
        "figshare_file_name": checkpoint_info.get("figshare_file_name"),
        "checkpoint_sha256": checkpoint_info.get("sha256"),
        "training_domain": "unresolved_from_public_filename",
        "device": device,
        "inference_seconds": elapsed,
        "height_like": str(height_path.resolve()),
        "raw_output_vs_reference_pearson_r": raw_correlation,
        "raw_output_stats": raw_stats,
        "results": results,
    }
    (checkpoint_dir / "report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    checkpoints = load_manifest()
    da3_by_anchor = da3_results()
    dop_path = DATA_DIR / "urban_residential_DOP.tif"
    dsm_path = DATA_DIR / "urban_residential_DSM.tif"
    if not dop_path.exists() or not dsm_path.exists():
        raise RuntimeError("OrthoLoC cache missing; run 'make benchmark-ortholoc-demo' first")

    rgb = read_rgb(dop_path)
    if rgb.shape[0] % 128 or rgb.shape[1] % 128:
        raise RuntimeError(f"RDAH requires dimensions divisible by 128; got {rgb.shape[:2]}")

    print("Generating Depth Anything v2 prior once for the complete checkpoint sweep...")
    da2 = DA2Prior()
    prior_started = time.perf_counter()
    da2_depth = da2.infer(rgb)
    prior_seconds = time.perf_counter() - prior_started

    reference, reference_valid = reproject_to_match(dsm_path, dop_path)
    valid = np.isfinite(reference) & reference_valid
    if int(valid.sum()) < 1000:
        raise RuntimeError("insufficient valid reference pixels")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    checkpoint_failures: list[dict[str, object]] = []

    for item in checkpoints:
        file_id = int(item["figshare_file_id"])
        print(f"Screening published RDAH checkpoint Figshare file {file_id}...")
        try:
            result = benchmark_checkpoint(
                item,
                rgb=rgb,
                da2_depth=da2_depth,
                reference=reference,
                valid=valid,
                da3_by_anchor=da3_by_anchor,
            )
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
            checkpoint_failures.append({"figshare_file_id": file_id, "error": str(exc)})
            print(f"  checkpoint {file_id}: INFERENCE/LOAD FAILED — {exc}")
            continue

        completed.append(result)
        result_64 = _anchor_result(result, 64)
        raw_r = result.get("raw_output_vs_reference_pearson_r")
        raw_std = result["raw_output_stats"]["std"]
        raw_r_text = "n/a" if raw_r is None else f"{float(raw_r):.4f}"
        if isinstance(result_64, dict) and result_64.get("status") == "PASS":
            metrics = result_64["metrics"]
            print(
                f"  file {file_id}: 64-anchor PASS | RMSE {float(metrics['rmse_m']):.3f} m | "
                f"MAE {float(metrics['mae_m']):.3f} m | r {metrics['pearson_r']} | "
                f"raw r {raw_r_text} | raw std {float(raw_std):.6f}"
            )
        else:
            error = result_64.get("error") if isinstance(result_64, dict) else "missing 64-anchor result"
            print(
                f"  file {file_id}: 64-anchor FAILED_SCREEN — {error} | "
                f"raw r {raw_r_text} | raw std {float(raw_std):.6f}"
            )

    screenable: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for item in completed:
        result_64 = _anchor_result(item, 64)
        if isinstance(result_64, dict) and result_64.get("status") == "PASS":
            metrics = result_64.get("metrics")
            if isinstance(metrics, dict) and "rmse_m" in metrics:
                screenable.append((item, result_64))

    screenable.sort(key=lambda pair: float(pair[1]["metrics"]["rmse_m"]))
    status = "PASS_SCREENING" if screenable else "SCREEN_COMPLETE_NO_COMPATIBLE_CHECKPOINT"

    report: dict[str, Any] = {
        "status": status,
        "dataset": "OrthoLoC demo / urban_residential",
        "protocol": "same_disjoint_sparse_anchor_holdout_as_DA3_baseline",
        "scope": (
            "Cross-domain geometry screening only. RDAH-Net is trained for nDSM/AGL-style "
            "height, while this OrthoLoC scene supplies DSM; checkpoint training-domain labels "
            "are also unresolved from the public filenames. Scientific calibration guards are "
            "not relaxed to force a metric."
        ),
        "da2_prior_device": da2.device,
        "da2_prior_seconds": prior_seconds,
        "checkpoint_results": completed,
        "checkpoint_failures": checkpoint_failures,
    }
    if screenable:
        best, best_64 = screenable[0]
        report["screening_best_64_anchor_figshare_file_id"] = best["figshare_file_id"]
        report["screening_best_64_anchor_rmse_m"] = best_64["metrics"]["rmse_m"]

    report_path = OUT_DIR / "checkpoint_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"DepthWizard published RDAH checkpoint sweep: {status}")
    print(f"DA2 prior device/time: {da2.device} / {prior_seconds:.2f} s")
    if screenable:
        for item, result_64 in screenable:
            metrics = result_64["metrics"]
            delta = result_64.get("rmse_delta_m_rdah_minus_da3")
            delta_text = "n/a" if delta is None else f"{float(delta):+.3f} m"
            print(
                f"file {item['figshare_file_id']} | 64 anchors | "
                f"RMSE {float(metrics['rmse_m']):.3f} m | "
                f"MAE {float(metrics['mae_m']):.3f} m | "
                f"r {metrics['pearson_r']} | vs DA3 {delta_text}"
            )
    else:
        print("No published RDAH checkpoint passed the guarded 64-anchor cross-domain screen.")
        for item in completed:
            result_64 = _anchor_result(item, 64)
            raw_r = item.get("raw_output_vs_reference_pearson_r")
            raw_std = item["raw_output_stats"]["std"]
            raw_r_text = "n/a" if raw_r is None else f"{float(raw_r):.4f}"
            reason = result_64.get("error") if isinstance(result_64, dict) else "missing result"
            print(
                f"file {item['figshare_file_id']} | 64 anchors FAILED | raw r {raw_r_text} | "
                f"raw std {float(raw_std):.6f} | {reason}"
            )
    print("Checkpoint training domains remain unresolved; results are screening evidence only.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
