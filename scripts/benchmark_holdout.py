from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from depthwizard.evaluation.holdout import sparse_anchor_holdout_benchmark
from depthwizard.io.raster import reproject_to_match, write_float_geotiff


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a DepthWizard relative DSM against an independent reference DSM using disjoint sparse calibration anchors."
    )
    parser.add_argument("relative_height", type=Path)
    parser.add_argument("reference_dsm", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--anchors", type=int, default=64)
    parser.add_argument("--seed", type=int, default=26175)
    parser.add_argument("--exclusion-radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.relative_height) as src:
        relative = src.read(1).astype(np.float32)
        valid = np.isfinite(relative)
        if src.nodata is not None:
            valid &= relative != src.nodata

    reference, reference_valid = reproject_to_match(args.reference_dsm, args.relative_height)
    valid &= reference_valid & np.isfinite(reference)

    result = sparse_anchor_holdout_benchmark(
        relative,
        reference,
        valid_mask=valid,
        anchor_count=args.anchors,
        seed=args.seed,
        exclusion_radius_px=args.exclusion_radius,
    )

    prediction_path = args.output_dir / "holdout_prediction_dsm.tif"
    write_float_geotiff(
        prediction_path,
        result.prediction,
        template_path=args.relative_height,
        description="DepthWizard sparse-anchor holdout DSM prediction (metres)",
        tags={
            "DEPTHWIZARD_PRODUCT": "SPARSE_ANCHOR_HOLDOUT_DSM_METRES",
            "ANCHOR_COUNT": str(args.anchors),
            "SPLIT_SEED": str(args.seed),
            "EVALUATION_PROTOCOL": "disjoint_sparse_anchor_holdout",
        },
    )

    payload = {
        "protocol": "disjoint_sparse_anchor_holdout",
        "semantics": (
            "Global scale/offset is fitted only on declared sparse anchors; metrics are computed "
            "on disjoint held-out reference pixels outside the anchor exclusion radius."
        ),
        "relative_height": str(args.relative_height.resolve()),
        "reference_dsm": str(args.reference_dsm.resolve()),
        "prediction_dsm": str(prediction_path.resolve()),
        "anchor_count": args.anchors,
        "seed": args.seed,
        "exclusion_radius_px": args.exclusion_radius,
        "heldout_pixels": int(result.evaluation_mask.sum()),
        "orientation_flipped": result.orientation_flipped,
        "anchor_correlation_before": result.anchor_correlation_before,
        "anchor_correlation_after": result.anchor_correlation_after,
        "calibration": result.calibration.model_dump(),
        "metrics": result.metrics.model_dump(),
    }
    report_path = args.output_dir / "holdout_metrics.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("DepthWizard sparse-anchor holdout benchmark: PASS")
    print(f"Anchors: {args.anchors} | held-out pixels: {payload['heldout_pixels']:,}")
    print(f"RMSE: {result.metrics.rmse_m:.3f} m")
    print(f"MAE: {result.metrics.mae_m:.3f} m")
    print(f"Pearson r: {result.metrics.pearson_r if result.metrics.pearson_r is not None else 'n/a'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
