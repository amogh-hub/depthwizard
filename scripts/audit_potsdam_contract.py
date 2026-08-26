from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TypedDict

import rasterio

from depthwizard.evaluation.potsdam import (
    FROZEN_POTSDAM_TILE_IDS,
    POTSDAM_NATIVE_SHAPE,
    resolve_potsdam_tile_paths,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(
    os.environ.get("DEPTHWIZARD_POTSDAM_ROOT", ROOT / "data" / "external" / "isprs-potsdam")
)
OUT_DIR = ROOT / "artifacts" / "evaluation" / "potsdam-external-v1"
AUDIT_PATH = OUT_DIR / "contract_audit_after_abort.json"
PROTOCOL_SEAL_PATH = OUT_DIR / "protocol_seal.json"


class RasterMetadata(TypedDict):
    path: str
    file_bytes: int
    width: int
    height: int
    count: int
    dtypes: list[str]
    driver: str
    crs: str | None
    transform: list[float]
    bounds: list[float]
    nodata: float | int | None
    block_shapes: list[list[int]]


def _world_file_values(path: Path) -> list[float]:
    values = [
        float(line.strip())
        for line in path.read_text(encoding="ascii").splitlines()
        if line.strip()
    ]
    if len(values) != 6:
        raise ValueError(f"world file must contain exactly six numeric lines: {path}")
    return values


def _metadata(path: Path) -> RasterMetadata:
    """Read raster metadata only; never decode any raster pixel values."""
    with rasterio.open(path) as src:
        return {
            "path": str(path.resolve()),
            "file_bytes": path.stat().st_size,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtypes": list(src.dtypes),
            "driver": src.driver,
            "crs": src.crs.to_string() if src.crs is not None else None,
            "transform": [
                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),
            ],
            "bounds": [float(value) for value in src.bounds],
            "nodata": src.nodata,
            "block_shapes": [list(shape) for shape in src.block_shapes],
        }


def _almost_equal(
    left: list[float],
    right: list[float],
    tolerance: float = 1e-9,
) -> bool:
    return len(left) == len(right) and all(
        abs(a - b) <= tolerance for a, b in zip(left, right, strict=True)
    )


def main() -> None:
    if not DATASET_ROOT.is_dir():
        raise FileNotFoundError(f"Potsdam dataset root does not exist: {DATASET_ROOT}")

    print("=== DEPTHWIZARD POTSDAM READ-ONLY CONTRACT AUDIT ===")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Existing protocol seal: {'YES' if PROTOCOL_SEAL_PATH.is_file() else 'NO'}")
    print("DSM pixel values read: NO")
    print()

    tiles: list[dict[str, object]] = []
    mismatch_count = 0

    for tile_id in FROZEN_POTSDAM_TILE_IDS:
        tile = resolve_potsdam_tile_paths(DATASET_ROOT, tile_id)
        rgb_meta = _metadata(tile.rgb)
        dsm_meta = _metadata(tile.reference_dsm)

        rgb_tfw = tile.rgb.with_suffix(".tfw")
        dsm_tfw = tile.reference_dsm.with_suffix(".tfw")
        if not rgb_tfw.is_file():
            raise FileNotFoundError(f"missing RGB world file: {rgb_tfw}")
        if not dsm_tfw.is_file():
            raise FileNotFoundError(f"missing DSM world file: {dsm_tfw}")

        rgb_world = _world_file_values(rgb_tfw)
        dsm_world = _world_file_values(dsm_tfw)

        rgb_shape = [rgb_meta["height"], rgb_meta["width"]]
        dsm_shape = [dsm_meta["height"], dsm_meta["width"]]
        expected_shape = [POTSDAM_NATIVE_SHAPE[0], POTSDAM_NATIVE_SHAPE[1]]

        checks = {
            "rgb_matches_declared_native_shape": rgb_shape == expected_shape,
            "dsm_matches_declared_native_shape": dsm_shape == expected_shape,
            "rgb_dsm_shape_equal": rgb_shape == dsm_shape,
            "rgb_dsm_transform_equal": _almost_equal(
                rgb_meta["transform"], dsm_meta["transform"]
            ),
            "rgb_dsm_bounds_equal": _almost_equal(rgb_meta["bounds"], dsm_meta["bounds"]),
            "rgb_dsm_world_file_equal": _almost_equal(rgb_world, dsm_world),
        }

        failed = [name for name, passed in checks.items() if not passed]
        mismatch_count += len(failed)

        tiles.append(
            {
                "tile_id": tile_id,
                "rgb": rgb_meta,
                "dsm": dsm_meta,
                "rgb_world_file": str(rgb_tfw.resolve()),
                "dsm_world_file": str(dsm_tfw.resolve()),
                "rgb_world_values": rgb_world,
                "dsm_world_values": dsm_world,
                "checks": checks,
                "failed_checks": failed,
            }
        )

        print(
            f"{tile_id}: RGB {rgb_meta['width']}x{rgb_meta['height']} | "
            f"DSM {dsm_meta['width']}x{dsm_meta['height']}"
        )
        print(f"  transform equal:  {checks['rgb_dsm_transform_equal']}")
        print(f"  bounds equal:     {checks['rgb_dsm_bounds_equal']}")
        print(f"  world file equal: {checks['rgb_dsm_world_file_equal']}")
        print(f"  failed checks:    {failed if failed else 'NONE'}")
        print()

    payload: dict[str, object] = {
        "schema": "depthwizard.potsdam-contract-audit.v1",
        "purpose": (
            "Read-only post-abort metadata audit. No DSM raster pixel values are decoded. "
            "Used only to diagnose the external-v1 data-contract failure without changing the "
            "sealed model, tile membership, calibration source, metrics, or promotion rule."
        ),
        "dataset_root": str(DATASET_ROOT.resolve()),
        "protocol_seal_present": PROTOCOL_SEAL_PATH.is_file(),
        "declared_native_shape": list(POTSDAM_NATIVE_SHAPE),
        "frozen_tile_ids": list(FROZEN_POTSDAM_TILE_IDS),
        "mismatch_count": mismatch_count,
        "tiles": tiles,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = AUDIT_PATH.with_suffix(AUDIT_PATH.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(AUDIT_PATH)

    print("=== AUDIT COMPLETE ===")
    print(f"Metadata mismatches recorded: {mismatch_count}")
    print(f"Audit artifact: {AUDIT_PATH}")
    print("DSM pixel values read: NO")
    print("Dataset files modified: NO")


if __name__ == "__main__":
    main()
