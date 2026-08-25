from __future__ import annotations

import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch

from depthwizard.baselines.rdah import infer_rdah, load_rdah_model
from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_slope_metrics
from depthwizard.io.raster import read_rgb, reproject_to_match, write_float_geotiff

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark" / "ortholoc-demo"
DA3_DIR = ROOT / "artifacts" / "benchmark" / "ortholoc-demo"
OUT_DIR = ROOT / "artifacts" / "benchmark" / "rdah-ortholoc-demo"
DA2_VENDOR = ROOT / ".vendor" / "depth-anything-v2"
DA2_WEIGHT = ROOT / "checkpoints" / "baselines" / "da2" / "depth_anything_v2_vits.pth"
RDAH_SELECTION = ROOT / "checkpoints" / "baselines" / "rdah" / "checkpoint_selection.json"
ANCHOR_BUDGETS = (8, 16, 32, 64)


class DA2Prior:
    def __init__(self) -> None:
        if not DA2_VENDOR.exists() or not DA2_WEIGHT.exists():
            raise RuntimeError("RDAH baseline is not prepared; run 'make rdah-setup' first")
        sys.path.insert(0, str(DA2_VENDOR))
        try:
            module: Any = importlib.import_module("depth_anything_v2.dpt")
        finally:
            sys.path.pop(0)
        model_type = module.DepthAnythingV2
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else "cpu"
        )
        self.model = model_type(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        state = torch.load(DA2_WEIGHT, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state)
        self.model = self.model.to(torch.device(self.device)).eval()

    def infer(self, rgb: np.ndarray) -> np.ndarray:
        # Official DA2 infer_image expects OpenCV BGR and internally converts to RGB.
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        with torch.inference_mode():
            depth = self.model.infer_image(bgr, input_size=518)
        result = np.asarray(depth, dtype=np.float32)
        if result.shape != rgb.shape[:2] or not np.all(np.isfinite(result)):
            raise RuntimeError("Depth Anything v2 returned an invalid relative-depth prior")
        return result


def load_checkpoint_path() -> Path:
    if not RDAH_SELECTION.exists():
        raise RuntimeError("RDAH checkpoint selection is missing; run 'make rdah-setup' first")
    payload = json.loads(RDAH_SELECTION.read_text(encoding="utf-8"))
    checkpoint = Path(str(payload["checkpoint"]))
    if not checkpoint.exists():
        raise RuntimeError(f"selected RDAH checkpoint is missing: {checkpoint}")
    return checkpoint


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    dop_path = DATA_DIR / "urban_residential_DOP.tif"
    dsm_path = DATA_DIR / "urban_residential_DSM.tif"
    da3_report_path = DA3_DIR / "benchmark_report.json"
    if not dop_path.exists() or not dsm_path.exists() or not da3_report_path.exists():
        raise RuntimeError(
            "OrthoLoC benchmark cache is missing; run 'make benchmark-ortholoc-demo' first"
        )
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rgb = read_rgb(dop_path)
    if rgb.shape[0] % 128 or rgb.shape[1] % 128:
        raise RuntimeError(
            f"released RDAH graph requires dimensions divisible by 128; got {rgb.shape[:2]}"
        )

    print("Generating the faithful RDAH-Net Depth Anything v2 prior...")
    da2 = DA2Prior()
    prior_started = time.perf_counter()
    da2_depth = da2.infer(rgb)
    prior_seconds = time.perf_counter() - prior_started
    np.save(OUT_DIR / "da2_relative_depth.npy", da2_depth)

    checkpoint = load_checkpoint_path()
    print("Running the released RDAH-Net architecture on MPS/available device...")
    model, rdah_device = load_rdah_model(checkpoint)
    rdah_started = time.perf_counter()
    rdah_height = infer_rdah(model, rgb, da2_depth, device=rdah_device)
    rdah_seconds = time.perf_counter() - rdah_started

    rdah_path = OUT_DIR / "rdah_height_like.tif"
    write_float_geotiff(
        rdah_path,
        rdah_height,
        template_path=dop_path,
        description="RDAH-Net cross-domain nDSM-like output for geometry benchmarking",
        tags={
            "MODEL": "RDAH-Net",
            "RELATIVE_DEPTH_PRIOR": "Depth Anything v2 Small",
            "SEMANTICS": "cross_domain_ndsm_like_not_claimed_as_absolute_dsm",
        },
    )

    reference, reference_valid = reproject_to_match(dsm_path, rdah_path)
    valid = np.isfinite(rdah_height) & np.isfinite(reference) & reference_valid
    if int(valid.sum()) < 1000:
        raise RuntimeError("insufficient valid paired pixels for RDAH holdout benchmark")

    with rasterio.open(dop_path) as src:
        gsd_x = abs(float(src.transform.a)) if not src.transform.is_identity else 1.0
        gsd_y = abs(float(src.transform.e)) if not src.transform.is_identity else 1.0

    rdah_results: list[dict[str, Any]] = []
    for anchors in ANCHOR_BUDGETS:
        benchmark = sparse_anchor_holdout_benchmark(
            rdah_height,
            reference,
            valid_mask=valid,
            anchor_count=anchors,
            seed=26175,
            exclusion_radius_px=4,
        )
        slope = compute_slope_metrics(
            benchmark.prediction,
            reference,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            valid_mask=benchmark.evaluation_mask,
        )
        rdah_results.append(
            {
                "anchor_count": anchors,
                "heldout_pixels": int(benchmark.evaluation_mask.sum()),
                "orientation_flipped": benchmark.orientation_flipped,
                "calibration": benchmark.calibration.model_dump(),
                "metrics": benchmark.metrics.model_dump(),
                "slope_metrics": slope.model_dump(),
            }
        )

    da3_report: dict[str, Any] = json.loads(da3_report_path.read_text(encoding="utf-8"))
    da3_by_anchor: dict[int, dict[str, Any]] = {
        int(item["anchor_count"]): item
        for item in da3_report.get("results", [])
        if isinstance(item, dict) and "anchor_count" in item
    }

    comparison: list[dict[str, Any]] = []
    for rdah_item in rdah_results:
        anchors = int(rdah_item["anchor_count"])
        da3_item = da3_by_anchor.get(anchors)
        if da3_item is None:
            continue
        rdah_metrics = rdah_item["metrics"]
        da3_metrics = da3_item.get("metrics")
        if not isinstance(rdah_metrics, dict) or not isinstance(da3_metrics, dict):
            continue
        comparison.append(
            {
                "anchor_count": anchors,
                "da3_rmse_m": float(da3_metrics["rmse_m"]),
                "rdah_rmse_m": float(rdah_metrics["rmse_m"]),
                "rmse_delta_m_rdah_minus_da3": (
                    float(rdah_metrics["rmse_m"]) - float(da3_metrics["rmse_m"])
                ),
                "da3_mae_m": float(da3_metrics["mae_m"]),
                "rdah_mae_m": float(rdah_metrics["mae_m"]),
                "da3_pearson_r": da3_metrics.get("pearson_r"),
                "rdah_pearson_r": rdah_metrics.get("pearson_r"),
            }
        )

    payload = {
        "status": "PASS",
        "dataset": "OrthoLoC demo / urban_residential",
        "protocol": "same_disjoint_sparse_anchor_holdout_as_DA3_baseline",
        "interpretation": (
            "This comparison tests cross-domain spatial geometry after the same sparse affine "
            "calibration. RDAH-Net is trained for nDSM, while OrthoLoC supplies a DSM, so its raw "
            "output is not claimed as an absolute-DSM metric prediction here."
        ),
        "rdah_checkpoint": str(checkpoint.resolve()),
        "rdah_device": rdah_device,
        "da2_device": da2.device,
        "da2_prior_seconds": prior_seconds,
        "rdah_inference_seconds": rdah_seconds,
        "rdah_results": rdah_results,
        "comparison": comparison,
    }
    report_path = OUT_DIR / "comparison_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DepthWizard RDAH vs DA3 OrthoLoC geometry benchmark: PASS")
    print(f"DA2 prior device/time: {da2.device} / {prior_seconds:.2f} s")
    print(f"RDAH device/time: {rdah_device} / {rdah_seconds:.2f} s")
    for item in comparison:
        delta = float(item["rmse_delta_m_rdah_minus_da3"])
        winner = "RDAH" if delta < 0 else "DA3"
        print(
            f"{int(item['anchor_count']):>2} anchors | "
            f"DA3 RMSE {float(item['da3_rmse_m']):.3f} m | "
            f"RDAH RMSE {float(item['rdah_rmse_m']):.3f} m | "
            f"winner {winner} | delta {delta:+.3f} m"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
