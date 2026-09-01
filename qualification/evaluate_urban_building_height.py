#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.building_height import evaluate_building_height_instances  # noqa: E402
from depthwizard.io.raster import ground_sample_distance_m  # noqa: E402
from depthwizard.provenance.manifest import sha256_file  # noqa: E402


def _read_float_surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float64)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != float(src.nodata)
    values[~valid] = np.nan
    return values, valid


def _read_binary_mask(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        values = src.read(1)
        valid = src.read_masks(1) > 0
    return valid & (values != 0)


def _grid_signature(path: Path) -> tuple[int, int, object, object]:
    with rasterio.open(path) as src:
        return src.height, src.width, src.crs, src.transform


def _require_exact_grid(reference: Path, *others: Path) -> None:
    expected = _grid_signature(reference)
    for path in others:
        actual = _grid_signature(path)
        same = (
            actual[0] == expected[0]
            and actual[1] == expected[1]
            and actual[2] == expected[2]
            and actual[3].almost_equals(expected[3])
        )
        if not same:
            raise ValueError(
                f"urban building benchmark requires exact-grid inputs; {path} does not match {reference}"
            )


def _write_csv(path: Path, instances: list[dict[str, object]]) -> None:
    if not instances:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(instances[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(instances)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a candidate metric DSM on explicit building instances. This is an exposed "
            "development diagnostic and must not be used to consume sealed blind tiles."
        )
    )
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--building-mask", type=Path, required=True)
    parser.add_argument("--ground-mask", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", default="exposed-urban-building-height")
    parser.add_argument("--min-building-area-m2", type=float, default=20.0)
    parser.add_argument("--min-reference-height-m", type=float, default=2.0)
    parser.add_argument("--roof-inset-m", type=float, default=0.50)
    parser.add_argument("--ground-inner-buffer-m", type=float, default=1.50)
    parser.add_argument("--ground-outer-buffer-m", type=float, default=8.00)
    parser.add_argument("--min-structure-pixels", type=int, default=16)
    parser.add_argument("--min-ground-pixels", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = [args.prediction, args.reference, args.building_mask]
    if args.ground_mask is not None:
        inputs.append(args.ground_mask)
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    _require_exact_grid(args.reference, *[path for path in inputs if path != args.reference])
    prediction, prediction_valid = _read_float_surface(args.prediction)
    reference, reference_valid = _read_float_surface(args.reference)
    buildings = _read_binary_mask(args.building_mask)
    ground = _read_binary_mask(args.ground_mask) if args.ground_mask is not None else None

    common_valid = prediction_valid & reference_valid
    prediction = np.where(common_valid, prediction, np.nan)
    reference = np.where(common_valid, reference, np.nan)
    buildings &= reference_valid
    if ground is not None:
        ground &= reference_valid

    gsd = ground_sample_distance_m(args.reference)
    if gsd is None:
        raise ValueError("urban building benchmark requires trustworthy physical GSD")

    report = evaluate_building_height_instances(
        prediction,
        reference,
        buildings,
        gsd_x_m=gsd[0],
        gsd_y_m=gsd[1],
        ground_candidate_mask=ground,
        min_building_area_m2=args.min_building_area_m2,
        min_reference_height_m=args.min_reference_height_m,
        roof_inset_m=args.roof_inset_m,
        ground_inner_buffer_m=args.ground_inner_buffer_m,
        ground_outer_buffer_m=args.ground_outer_buffer_m,
        min_structure_pixels=args.min_structure_pixels,
        min_ground_pixels=args.min_ground_pixels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "building-height-report.json"
    instances_path = args.output_dir / "building-height-instances.csv"
    payload = {
        "schema_version": 1,
        "label": args.label,
        "claim_boundary": (
            "Exposed development diagnostic only. Reference values are used solely downstream for "
            "evaluation and must never enter inference, calibration, model selection on sealed blind "
            "tiles, or production reconstruction."
        ),
        "inputs": {
            "prediction": str(args.prediction.resolve()),
            "prediction_sha256": sha256_file(args.prediction),
            "reference": str(args.reference.resolve()),
            "reference_sha256": sha256_file(args.reference),
            "building_mask": str(args.building_mask.resolve()),
            "building_mask_sha256": sha256_file(args.building_mask),
            "ground_mask": str(args.ground_mask.resolve()) if args.ground_mask else None,
            "ground_mask_sha256": sha256_file(args.ground_mask) if args.ground_mask else None,
            "gsd_x_m": gsd[0],
            "gsd_y_m": gsd[1],
        },
        "config": {
            "min_building_area_m2": args.min_building_area_m2,
            "min_reference_height_m": args.min_reference_height_m,
            "roof_inset_m": args.roof_inset_m,
            "ground_inner_buffer_m": args.ground_inner_buffer_m,
            "ground_outer_buffer_m": args.ground_outer_buffer_m,
            "min_structure_pixels": args.min_structure_pixels,
            "min_ground_pixels": args.min_ground_pixels,
        },
        "report": asdict(report),
    }
    temporary = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)
    _write_csv(instances_path, [asdict(instance) for instance in report.instances])

    print(f"building_height_report={report_path}")
    print(f"building_height_instances={instances_path}")
    print(f"evaluated_buildings={len(report.evaluated_instance_ids)}")
    print(f"prediction_failures={len(report.prediction_failure_ids)}")
    print(f"height_mae_m={report.height_mae_m:.6f}")
    print(f"height_rmse_m={report.height_rmse_m:.6f}")
    print(f"height_p90_abs_error_m={report.height_p90_abs_error_m:.6f}")
    print(f"within_2m_fraction={report.within_2m_fraction:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
