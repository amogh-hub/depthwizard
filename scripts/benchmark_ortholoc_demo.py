from __future__ import annotations

import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import rasterio

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.evaluation.metrics import compute_slope_metrics
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.raster import reproject_to_match, write_float_geotiff
from depthwizard.pipeline.geometry import infer_geometry_scene

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark" / "ortholoc-demo"
OUT_DIR = ROOT / "artifacts" / "benchmark" / "ortholoc-demo"
BASE_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/demo"
DOP_URL = f"{BASE_URL}/urban_residential_DOP.tif"
DSM_URL = f"{BASE_URL}/urban_residential_DSM.tif"
USER_AGENT = "DepthWizard-SIH26175/0.2 benchmark"
ANCHOR_BUDGETS = (8, 16, 32, 64)


def download(url: str, path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty download: {url}")
    path.write_bytes(payload)
    return path


def main() -> None:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading/caching OrthoLoC paired DOP + DSM demo scene...")
    dop_path = download(DOP_URL, DATA_DIR / "urban_residential_DOP.tif")
    dsm_path = download(DSM_URL, DATA_DIR / "urban_residential_DSM.tif")

    with rasterio.open(dop_path) as src:
        if src.count < 3:
            raise RuntimeError("OrthoLoC DOP must contain at least three optical bands")
        source_shape = (src.height, src.width)
        source_crs = src.crs.to_string() if src.crs is not None else None
        gsd_x = abs(float(src.transform.a)) if not src.transform.is_identity else 1.0
        gsd_y = abs(float(src.transform.e)) if not src.transform.is_identity else 1.0

    print(f"Scene: {source_shape[1]} x {source_shape[0]} | CRS: {source_crs or 'none'}")
    print("Running DA3MONO-LARGE reconstruction once for all anchor budgets...")
    prior = DA3MonocularPrior(device="auto")
    started = time.perf_counter()
    scene = infer_geometry_scene(
        dop_path,
        prior,
        tile_size=768,
        overlap=128,
        harmonize_overlaps=True,
    )
    inference_seconds = time.perf_counter() - started

    rdsm_path = OUT_DIR / "rdsm.tif"
    write_float_geotiff(
        rdsm_path,
        scene.relative_height,
        template_path=dop_path,
        description="DepthWizard DA3 relative DSM for OrthoLoC benchmark",
        tags={
            "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
            "MODEL_ID": scene.model_id,
            "BENCHMARK_DATASET": "OrthoLoC demo",
        },
    )

    reference, reference_valid = reproject_to_match(dsm_path, rdsm_path)
    valid = np.isfinite(scene.relative_height) & np.isfinite(reference) & reference_valid
    if int(valid.sum()) < 1000:
        raise RuntimeError("insufficient valid paired pixels for benchmark")

    results: list[dict[str, object]] = []
    for anchors in ANCHOR_BUDGETS:
        benchmark = sparse_anchor_holdout_benchmark(
            scene.relative_height,
            reference,
            valid_mask=valid,
            anchor_count=anchors,
            seed=26175,
            exclusion_radius_px=4,
        )
        metrics = benchmark.metrics
        slope = compute_slope_metrics(
            benchmark.prediction,
            reference,
            gsd_x=gsd_x,
            gsd_y=gsd_y,
            valid_mask=benchmark.evaluation_mask,
        )
        results.append(
            {
                "anchor_count": anchors,
                "heldout_pixels": int(benchmark.evaluation_mask.sum()),
                "orientation_flipped": benchmark.orientation_flipped,
                "anchor_correlation_before": benchmark.anchor_correlation_before,
                "anchor_correlation_after": benchmark.anchor_correlation_after,
                "calibration": benchmark.calibration.model_dump(),
                "metrics": metrics.model_dump(),
                "slope_metrics": slope.model_dump(),
            }
        )

        prediction_path = OUT_DIR / f"prediction_{anchors:02d}_anchors.tif"
        write_float_geotiff(
            prediction_path,
            benchmark.prediction,
            template_path=rdsm_path,
            description=f"DepthWizard held-out DSM prediction with {anchors} sparse anchors",
            tags={
                "EVALUATION_PROTOCOL": "disjoint_sparse_anchor_holdout",
                "ANCHOR_COUNT": str(anchors),
                "BENCHMARK_DATASET": "OrthoLoC demo",
            },
        )

    payload = {
        "status": "PASS",
        "dataset": "OrthoLoC demo / urban_residential",
        "dataset_source": "TUM OrthoLoC public dataset",
        "dataset_license": "CC BY-NC-SA 4.0",
        "protocol": "disjoint_sparse_anchor_holdout",
        "protocol_note": (
            "The paired reference DSM supplies only the declared sparse calibration anchors. "
            "RMSE/MAE/correlation are computed on disjoint held-out pixels outside a 4-pixel "
            "exclusion radius. Metrics are conditional on each anchor budget, not zero-shot scores."
        ),
        "model": scene.model_id,
        "device": prior._resolved_device or "unknown",
        "shape": list(scene.relative_height.shape),
        "crs": source_crs,
        "gsd_x": gsd_x,
        "gsd_y": gsd_y,
        "tile_count": scene.tile_count,
        "harmonized_tiles": scene.harmonized_tiles,
        "inference_seconds": inference_seconds,
        "anchor_budgets": list(ANCHOR_BUDGETS),
        "results": results,
    }
    report_path = OUT_DIR / "benchmark_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DepthWizard OrthoLoC paired RGB-DSM benchmark: PASS")
    print(f"Model/device: {scene.model_id} / {prior._resolved_device or 'unknown'}")
    print(f"Tiles: {scene.tile_count} ({scene.harmonized_tiles} overlap-harmonized)")
    print(f"Inference wall time: {inference_seconds:.2f} s")
    for item in results:
        metrics = item["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{item['anchor_count']:>2} anchors | "
            f"RMSE {metrics['rmse_m']:.3f} m | MAE {metrics['mae_m']:.3f} m | "
            f"r {metrics['pearson_r']} | held-out {item['heldout_pixels']:,} px"
        )
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
