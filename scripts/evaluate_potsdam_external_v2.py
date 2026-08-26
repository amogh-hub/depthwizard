from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.warp import reproject

from depthwizard.evaluation.potsdam import (
    POTSDAM_CRS,
    PotsdamTilePaths,
    benchmark_full_coverage_mask,
    inspect_potsdam_reference_contract,
    protocol_sha256,
)
from depthwizard.provenance.manifest import sha256_file
from scripts import evaluate_potsdam_external as v1

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "artifacts" / "evaluation" / "potsdam-external-v2"
PROTOCOL_SEAL_PATH = OUT_DIR / "protocol_seal.json"
REPORT_PATH = OUT_DIR / "potsdam_external_report.json"
V1_PROTOCOL_SEAL_PATH = ROOT / "artifacts" / "evaluation" / "potsdam-external-v1" / "protocol_seal.json"
MAX_TRAILING_EDGE_DEFICIT_PX = 1
V1_PROTOCOL_SHA256 = "6452480ae7cc55d63eff5cab9bf6b449bfb00222591e79070854c57dc6e35923"

# External-v2 is intentionally a thin adapter over the frozen v1 inference/calibration core.
# Direct module references and assignments make the compatibility seam explicit and statically
# checkable while preserving the historical v1 implementation unchanged.
_CORE_PROTOCOL_PAYLOAD = v1._protocol_payload
_CORE_EVALUATE_TILE = v1._evaluate_tile
_CORE_LOAD_V4_MODEL = v1._load_v4_model
_SEALED_REFERENCE_CONTRACTS: dict[str, dict[str, object]] = {}
_REFERENCE_RUNTIME: dict[str, dict[str, object]] = {}


def _verify_historical_v1_seal() -> None:
    if not V1_PROTOCOL_SEAL_PATH.is_file():
        raise FileNotFoundError(
            "historical external-v1 protocol seal is missing; do not create external-v2 without "
            f"the preserved v1 evidence record: {V1_PROTOCOL_SEAL_PATH}"
        )
    document = json.loads(V1_PROTOCOL_SEAL_PATH.read_text(encoding="utf-8"))
    actual = document.get("protocol_sha256")
    if actual != V1_PROTOCOL_SHA256:
        raise RuntimeError(
            "historical external-v1 protocol seal digest does not match the preserved run: "
            f"expected {V1_PROTOCOL_SHA256}, got {actual!r}"
        )


def _assert_v2_unconsumed() -> None:
    if PROTOCOL_SEAL_PATH.exists():
        raise RuntimeError(
            "external-v2 protocol seal already exists. Treat v2 as consumed; do not delete or "
            "rewrite its seal to rerun a changed evaluator."
        )


def _reference_contracts(tile_paths: list[PotsdamTilePaths]) -> dict[str, dict[str, object]]:
    contracts: dict[str, dict[str, object]] = {}
    for tile in tile_paths:
        contracts[tile.tile_id] = inspect_potsdam_reference_contract(
            tile.rgb,
            tile.reference_dsm,
            max_trailing_edge_deficit_px=MAX_TRAILING_EDGE_DEFICIT_PX,
        )
    return contracts


def _protocol_payload_v2(
    tile_paths: list[PotsdamTilePaths],
    rgb_contracts: list[dict[str, object]],
    checkpoint: dict[str, Any],
) -> dict[str, object]:
    payload = _CORE_PROTOCOL_PAYLOAD(tile_paths, rgb_contracts, checkpoint)
    reference_contracts = _reference_contracts(tile_paths)
    _SEALED_REFERENCE_CONTRACTS.clear()
    _SEALED_REFERENCE_CONTRACTS.update(reference_contracts)

    payload["protocol_version"] = "potsdam-external-v2"
    payload["purpose"] = (
        "Frozen external cross-dataset evaluation after a metadata-only correction to the v1 "
        "reference-shape contract. ISPRS Potsdam DSM remains evaluation-only and is never used for "
        "model selection, blending, scale calibration, orientation, or promotion criteria."
    )
    payload["reference_metadata_contract_v2"] = {
        "origin": (
            "external-v1 aborted after sealing when official tile 3_13 was found to contain one "
            "fewer native column than its 6000x6000 RGB tile while retaining the identical affine "
            "transform/world-file origin. The post-abort audit inspected metadata only and read no "
            "DSM pixel values."
        ),
        "v1_protocol_sha256": V1_PROTOCOL_SHA256,
        "allowed_trailing_edge_deficit_native_px": MAX_TRAILING_EDGE_DEFICIT_PX,
        "required_affine_policy": "identical_rgb_reference_affine_and_upper_left_origin",
        "required_crs": POTSDAM_CRS.to_string(),
        "partial_benchmark_cell_policy": "exclude_any_25cm_cell_not_fully_covered_by_reference",
        "model_checkpoint_changed": False,
        "calibration_source_changed": False,
        "selected_tiles_changed": False,
        "metrics_changed": False,
        "promotion_rule_changed": False,
        "tile_contracts": reference_contracts,
    }
    source_sha256 = payload.get("source_sha256")
    if not isinstance(source_sha256, dict):
        raise TypeError("v1 protocol payload source_sha256 is not a dictionary")
    source_sha256[str(Path(__file__).relative_to(ROOT))] = sha256_file(Path(__file__))
    return payload


def _load_reference_after_seal_v2(
    tile: PotsdamTilePaths,
    benchmark_rgb: Path,
) -> tuple[np.ndarray, np.ndarray, str]:
    if not PROTOCOL_SEAL_PATH.is_file():
        raise RuntimeError("external-v2 protocol seal must exist before loading Potsdam DSM values")

    contract = inspect_potsdam_reference_contract(
        tile.rgb,
        tile.reference_dsm,
        max_trailing_edge_deficit_px=MAX_TRAILING_EDGE_DEFICIT_PX,
    )
    sealed_contract = _SEALED_REFERENCE_CONTRACTS.get(tile.tile_id)
    if sealed_contract is None:
        raise RuntimeError(f"tile {tile.tile_id} has no sealed reference metadata contract")
    if contract != sealed_contract:
        raise RuntimeError(
            f"Potsdam reference metadata changed after protocol seal for tile {tile.tile_id}"
        )

    with rasterio.open(benchmark_rgb) as target:
        destination = np.full((target.height, target.width), np.nan, dtype=np.float32)
        destination_mask = np.zeros((target.height, target.width), dtype=np.uint8)
        dst_transform = target.transform
        dst_crs = target.crs
        target_height = target.height
        target_width = target.width
    if dst_crs is None:
        raise RuntimeError("derived Potsdam benchmark RGB lost its CRS")

    with rasterio.open(tile.reference_dsm) as src:
        source_crs = src.crs or POTSDAM_CRS
        if source_crs != POTSDAM_CRS:
            raise ValueError(
                f"Potsdam reference CRS must resolve to EPSG:32633; got {source_crs}"
            )
        reference_bounds = (
            float(src.bounds.left),
            float(src.bounds.bottom),
            float(src.bounds.right),
            float(src.bounds.top),
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=source_crs,
            src_nodata=src.nodata,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        source_mask = src.read_masks(1)
        reproject(
            source=source_mask,
            destination=destination_mask,
            src_transform=src.transform,
            src_crs=source_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )

    full_coverage = benchmark_full_coverage_mask(
        target_height=target_height,
        target_width=target_width,
        target_transform=(
            float(dst_transform.a),
            float(dst_transform.b),
            float(dst_transform.c),
            float(dst_transform.d),
            float(dst_transform.e),
            float(dst_transform.f),
        ),
        reference_bounds=reference_bounds,
    )
    valid = (destination_mask > 0) & np.isfinite(destination) & full_coverage
    _REFERENCE_RUNTIME[tile.tile_id] = {
        "reference_contract": contract,
        "benchmark_cells_total": int(full_coverage.size),
        "benchmark_cells_fully_covered": int(full_coverage.sum()),
        "benchmark_full_coverage_fraction": float(full_coverage.mean()),
    }
    return destination, valid, sha256_file(tile.reference_dsm)


def _evaluate_tile_v2(
    tile: PotsdamTilePaths,
    model: Any,
    prior: Any,
    device: Any,
) -> Any:
    result = _CORE_EVALUATE_TILE(tile, model, prior, device)
    report = result[0]
    runtime = _REFERENCE_RUNTIME.get(tile.tile_id)
    if runtime is not None:
        report["reference_contract_v2"] = runtime
    return result


def _configure_v2() -> None:
    v1.OUT_DIR = OUT_DIR
    v1.PROTOCOL_SEAL_PATH = PROTOCOL_SEAL_PATH
    v1.REPORT_PATH = REPORT_PATH
    v1._protocol_payload = _protocol_payload_v2
    v1._load_reference_after_seal = _load_reference_after_seal_v2
    v1._evaluate_tile = _evaluate_tile_v2


def preflight() -> None:
    """Validate the complete v2 metadata/provenance contract without consuming its seal."""
    _verify_historical_v1_seal()
    _assert_v2_unconsumed()

    tile_paths = [
        v1.resolve_potsdam_tile_paths(v1.DATASET_ROOT, tile_id)
        for tile_id in v1.FROZEN_POTSDAM_TILE_IDS
    ]
    rgb_contracts = [v1.inspect_potsdam_rgb_contract(tile.rgb) for tile in tile_paths]
    if not v1.COPDEM_PATH.is_file():
        raise FileNotFoundError(
            "Copernicus GLO-30 calibration tile is missing; external-v1 should have downloaded it: "
            f"{v1.COPDEM_PATH}"
        )
    model, checkpoint = _CORE_LOAD_V4_MODEL(torch.device("cpu"))
    del model
    protocol = _protocol_payload_v2(tile_paths, rgb_contracts, checkpoint)
    candidate_digest = protocol_sha256(protocol)

    print("=== DEPTHWIZARD POTSDAM EXTERNAL-V2 PREFLIGHT ===")
    print("Historical v1 seal: VERIFIED")
    print("V2 seal already exists: NO")
    print(f"Frozen tiles: {list(v1.FROZEN_POTSDAM_TILE_IDS)}")
    for tile_id, contract in _SEALED_REFERENCE_CONTRACTS.items():
        print(
            f"{tile_id}: reference shape {contract['reference_shape']} | "
            f"missing trailing rows {contract['missing_trailing_rows']} | "
            f"columns {contract['missing_trailing_columns']}"
        )
    print(f"Candidate v2 protocol SHA256: {candidate_digest}")
    print("Potsdam DSM pixel values read: NO")
    print("V2 protocol seal created: NO")
    print("Ready for one sealed external-v2 execution. ✅")


def main() -> None:
    _verify_historical_v1_seal()
    _assert_v2_unconsumed()
    _configure_v2()
    print("DepthWizard Potsdam external-v2: metadata-contract correction only")
    print(f"Historical external-v1 seal preserved: {V1_PROTOCOL_SHA256}")
    print(
        "V2 keeps the same four tiles, frozen V4 checkpoint, DA3 baseline, Copernicus calibration, "
        "metrics, and promotion rule."
    )
    print(
        "Reference correction: permit <=1 missing trailing native edge pixel only with identical "
        "affine/CRS; exclude any 25 cm evaluation cell lacking full reference footprint."
    )
    v1.main()


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if not arguments:
        main()
    elif arguments == ["--preflight-only"]:
        preflight()
    else:
        raise SystemExit("usage: python -m scripts.evaluate_potsdam_external_v2 [--preflight-only]")
