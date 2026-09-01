#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.potsdam import POTSDAM_NATIVE_GSD_M  # noqa: E402
from depthwizard.evaluation.potsdam_semantics import (  # noqa: E402
    ISPRS_CLASS_COLORS,
    decode_potsdam_semantic_labels,
)
from depthwizard.io.raster import ground_sample_distance_m  # noqa: E402
from depthwizard.provenance.manifest import sha256_file  # noqa: E402

EXPOSED_TILE_ID = "2_14"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare exact-grid building and ground masks for the exposed Potsdam 2_14 development "
            "scene from the official ISPRS semantic label raster. This tool intentionally refuses "
            "all other tile ids so sealed blind evidence cannot be consumed by mistake."
        )
    )
    parser.add_argument("--semantic-label", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-policy",
        choices=("strict", "expanded"),
        default="strict",
        help="strict=impervious only; expanded=impervious + low vegetation",
    )
    parser.add_argument("--tile-id", default=EXPOSED_TILE_ID)
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.count != 3:
            raise ValueError(f"semantic label raster must contain exactly 3 bands: {path}")
        return np.moveaxis(src.read((1, 2, 3)), 0, -1)


def _write_mask(
    path: Path,
    mask: np.ndarray,
    known: np.ndarray,
    *,
    reference: Path,
    semantics: str,
) -> None:
    with rasterio.open(reference) as src:
        profile = src.profile.copy()
    values = np.zeros(mask.shape, dtype=np.uint8)
    values[mask] = 1
    values[~known] = 255
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        predictor=1,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(values, 1)
        dst.update_tags(
            DEPTHWIZARD_PRODUCT="qualification_mask",
            DEPTHWIZARD_SEMANTICS=semantics,
            DEPTHWIZARD_TILE_ID=EXPOSED_TILE_ID,
        )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.tile_id != EXPOSED_TILE_ID:
        raise ValueError(
            f"this exposed-development preparation tool is hard-locked to tile {EXPOSED_TILE_ID}; "
            "blind or alternate tiles must not be opened through this workflow"
        )
    if not args.semantic_label.is_file():
        raise FileNotFoundError(args.semantic_label)
    if not args.reference.is_file():
        raise FileNotFoundError(args.reference)

    with rasterio.open(args.semantic_label) as labels, rasterio.open(args.reference) as reference:
        if (labels.height, labels.width) != (reference.height, reference.width):
            raise ValueError(
                "semantic labels and reference DSM must have identical pixel dimensions for the "
                "exposed Potsdam benchmark"
            )
        if labels.crs is not None and reference.crs is not None and labels.crs != reference.crs:
            raise ValueError("semantic labels and reference DSM declare different CRS values")
        if (
            labels.transform is not None
            and not labels.transform.is_identity
            and not labels.transform.almost_equals(reference.transform)
        ):
            raise ValueError("semantic labels and reference DSM declare different affine grids")

    gsd = ground_sample_distance_m(args.reference)
    if gsd is None:
        raise ValueError("reference DSM must provide trustworthy physical GSD")
    if not np.isclose(gsd[0], POTSDAM_NATIVE_GSD_M, atol=1e-5) or not np.isclose(
        gsd[1], POTSDAM_NATIVE_GSD_M, atol=1e-5
    ):
        raise ValueError(
            "exposed Potsdam semantic preparation requires the native 5 cm reference grid; "
            f"got {gsd[0]:.6f} x {gsd[1]:.6f} m"
        )

    decoded = decode_potsdam_semantic_labels(_read_rgb(args.semantic_label))
    if decoded.class_pixel_counts["building"] == 0:
        raise ValueError("official semantic label contains no building pixels")
    ground_mask = decoded.strict_ground if args.ground_policy == "strict" else decoded.expanded_ground
    if int(np.count_nonzero(ground_mask)) == 0:
        raise ValueError(f"selected {args.ground_policy} ground policy contains no valid pixels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    building_path = args.output_dir / "potsdam-2_14-building-mask.tif"
    ground_path = args.output_dir / f"potsdam-2_14-ground-mask-{args.ground_policy}.tif"
    manifest_path = args.output_dir / "potsdam-2_14-semantic-mask-manifest.json"

    _write_mask(
        building_path,
        decoded.building,
        decoded.known,
        reference=args.reference,
        semantics="official_isprs_building_class",
    )
    _write_mask(
        ground_path,
        ground_mask,
        decoded.known,
        reference=args.reference,
        semantics=(
            "official_isprs_impervious_ground_candidates"
            if args.ground_policy == "strict"
            else "official_isprs_impervious_plus_low_vegetation_ground_candidates"
        ),
    )

    payload = {
        "schema_version": 1,
        "tile_id": EXPOSED_TILE_ID,
        "status": "EXPOSED_DEVELOPMENT_ONLY",
        "claim_boundary": (
            "Masks are derived deterministically from the official ISPRS semantic labels for the "
            "already-exposed Potsdam 2_14 development scene. This workflow is hard-locked against "
            "sealed blind tiles and must never be generalized to consume them before protocol freeze."
        ),
        "semantic_label": str(args.semantic_label.resolve()),
        "semantic_label_sha256": sha256_file(args.semantic_label),
        "reference": str(args.reference.resolve()),
        "reference_sha256": sha256_file(args.reference),
        "reference_gsd_m": [gsd[0], gsd[1]],
        "official_rgb_palette": {name: list(color) for name, color in ISPRS_CLASS_COLORS.items()},
        "unlabeled_pixels": decoded.unlabeled_pixels,
        "class_pixel_counts": decoded.class_pixel_counts,
        "ground_policy": args.ground_policy,
        "building_mask": str(building_path.resolve()),
        "building_mask_sha256": sha256_file(building_path),
        "ground_mask": str(ground_path.resolve()),
        "ground_mask_sha256": sha256_file(ground_path),
    }
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)

    print(f"building_mask={building_path}")
    print(f"ground_mask={ground_path}")
    print(f"manifest={manifest_path}")
    print(f"building_pixels={decoded.class_pixel_counts['building']}")
    print(f"ground_pixels={int(np.count_nonzero(ground_mask))}")
    print(f"unlabeled_pixels={decoded.unlabeled_pixels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
