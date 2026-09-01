#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.topography import compute_stratified_terrain_metrics  # noqa: E402
from depthwizard.io.raster import ground_sample_distance_m  # noqa: E402
from depthwizard.provenance.manifest import sha256_file  # noqa: E402


def _grid_signature(path: Path) -> tuple[int, int, object, object]:
    with rasterio.open(path) as src:
        return src.height, src.width, src.crs, src.transform


def _require_exact_grid(reference: Path, prediction: Path) -> None:
    expected = _grid_signature(reference)
    actual = _grid_signature(prediction)
    if not (
        actual[0] == expected[0]
        and actual[1] == expected[1]
        and actual[2] == expected[2]
        and actual[3].almost_equals(expected[3])
    ):
        raise ValueError("stratified terrain evaluation requires exact-grid prediction/reference")


def _read_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float64)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != float(src.nodata)
    values[~valid] = np.nan
    return values, valid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate elevation and slope errors by reference-defined terrain difficulty. "
            "Use aligned downstream reference products; never feed reference values into inference."
        )
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="stratified-terrain-evaluation")
    parser.add_argument("--steep-threshold-degrees", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for path in (args.prediction, args.reference):
        if not path.is_file():
            raise FileNotFoundError(path)
    _require_exact_grid(args.reference, args.prediction)

    prediction, prediction_valid = _read_surface(args.prediction)
    reference, reference_valid = _read_surface(args.reference)
    gsd = ground_sample_distance_m(args.reference)
    if gsd is None:
        raise ValueError("stratified terrain evaluation requires trustworthy physical GSD")

    report = compute_stratified_terrain_metrics(
        prediction,
        reference,
        gsd_x_m=gsd[0],
        gsd_y_m=gsd[1],
        valid_mask=prediction_valid & reference_valid,
        steep_threshold_degrees=args.steep_threshold_degrees,
    )
    payload = {
        "schema_version": 1,
        "label": args.label,
        "claim_boundary": "Downstream evaluation only; reference values are not reconstruction evidence.",
        "inputs": {
            "prediction": str(args.prediction.resolve()),
            "prediction_sha256": sha256_file(args.prediction),
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
            "gsd_x_m": gsd[0],
            "gsd_y_m": gsd[1],
        },
        "config": {"steep_threshold_degrees": args.steep_threshold_degrees},
        "report": asdict(report),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)

    print(f"terrain_report={args.output}")
    print(f"overall_elevation_rmse_m={report.elevation_rmse_m:.6f}")
    print(f"steep_elevation_rmse_m={report.steep_elevation_rmse_m:.6f}")
    print(f"steep_slope_rmse_degrees={report.steep_slope_rmse_degrees:.6f}")
    print(f"steep_valid_pixels={report.steep_valid_pixels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
