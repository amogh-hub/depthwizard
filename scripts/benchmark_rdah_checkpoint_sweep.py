from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
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
        raise RuntimeError("RDAH sweep manifest is invalid")
    result = [item for item in payload["checkpoints"] if isinstance(item, dict)]
    if not result:
        raise RuntimeError("RDAH sweep manifest contains no checkpoints")
    return result


def da3_results() -> dict[int, dict[str, Any]]:
    if not DA3_REPORT.exists():
        raise RuntimeError("DA3 benchmark report missing; run 'make benchmark-ortholoc-demo' first")
    payload = json.loads(DA3_REPORT.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("DA3 benchmark report is invalid")
    results = payload.get("results")
    if not isinstance(results, list):
        raise RuntimeError("DA3 benchmark report has no results")
    return {
        int(item["anchor_count"]): item
        for item in results
        if isinstance(item, dict) and "anchor_count" in item
    }


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
        holdout = sparse_anchor_holdout_benchmark(
            rdah_height,
            reference,
            valid_mask=valid,
            anchor_count=anchors,
            seed=26175,
            exclusion_radius_px=4,
        )
        metrics = holdout.metrics.model_dump()
        da3_item = da3_by_anchor.get(anchors)
        da3_metrics = da3_item.get("metrics") if isinstance(da3_item, dict) else None
        da3_rmse = (
            float(da3_metrics["rmse_m"])
            if isinstance(da3_metrics, dict) and "rmse_m" in da3_metrics
            else None
        )
        rdah_rmse = float(metrics["rmse_m"])
        results.append(
            {
                "anchor_count": anchors,
                "heldout_pixels": int(holdout.evaluation_mask.sum()),
                "metrics": metrics,
                "orientation_flipped": holdout.orientation_flipped,
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
    failures: list[dict[str, object]] = []

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
            failures.append({"figshare_file_id": file_id, "error": str(exc)})
            print(f"  checkpoint {file_id}: FAILED — {exc}")
            continue
        completed.append(result)
        result_64 = next(
            entry for entry in result["results"] if int(entry["anchor_count"]) == 64
        )
        metrics_64 = result_64["metrics"]
        print(
            f"  checkpoint {file_id}: RMSE {float(metrics_64['rmse_m']):.3f} m | "
            f"MAE {float(metrics_64['mae_m']):.3f} m | "
            f"r {metrics_64['pearson_r']}"
        )

    if not completed:
        raise RuntimeError("every published RDAH checkpoint failed compatibility screening")

    ranking = sorted(
        completed,
        key=lambda item: float(
            next(entry for entry in item["results"] if int(entry["anchor_count"]) == 64)[
                "metrics"
            ]["rmse_m"]
        ),
    )
    best = ranking[0]
    best_64 = next(entry for entry in best["results"] if int(entry["anchor_count"]) == 64)

    report = {
        "status": "PASS_SCREENING",
        "dataset": "OrthoLoC demo / urban_residential",
        "protocol": "same_disjoint_sparse_anchor_holdout_as_DA3_baseline",
        "scope": (
            "cross-domain geometry screening only; checkpoint training-domain labels are unresolved "
            "from the public filenames, and the best checkpoint is not a final independent score"
        ),
        "da2_prior_device": da2.device,
        "da2_prior_seconds": prior_seconds,
        "completed": completed,
        "failures": failures,
        "screening_best_64_anchor_figshare_file_id": best["figshare_file_id"],
        "screening_best_64_anchor_rmse_m": best_64["metrics"]["rmse_m"],
    }
    report_path = OUT_DIR / "checkpoint_sweep_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("DepthWizard published RDAH checkpoint sweep: PASS")
    print(f"DA2 prior device/time: {da2.device} / {prior_seconds:.2f} s")
    for item in ranking:
        result_64 = next(
            entry for entry in item["results"] if int(entry["anchor_count"]) == 64
        )
        metrics = result_64["metrics"]
        delta = result_64["rmse_delta_m_rdah_minus_da3"]
        print(
            f"file {item['figshare_file_id']} | 64 anchors | "
            f"RMSE {float(metrics['rmse_m']):.3f} m | MAE {float(metrics['mae_m']):.3f} m | "
            f"r {metrics['pearson_r']} | vs DA3 {float(delta):+.3f} m"
        )
    print("Checkpoint training domains remain unresolved; results are screening evidence only.")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
