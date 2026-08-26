from __future__ import annotations

import os
from pathlib import Path
from typing import NoReturn

import numpy as np
import torch

from depthwizard.evaluation.potsdam import PotsdamTilePaths
from scripts import evaluate_potsdam_external as v1
from scripts import evaluate_potsdam_external_v2 as v2


class _ReferenceBoundaryReached(RuntimeError):
    """Sentinel proving target-free execution reached the reference-load boundary."""


def _stop_before_reference_values(
    tile: PotsdamTilePaths,
    benchmark_rgb: Path,
) -> NoReturn:
    del benchmark_rgb
    raise _ReferenceBoundaryReached(tile.tile_id)


def main() -> None:
    """Exercise all expensive pre-reference runtime paths without consuming external-v2."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    v2._verify_historical_v1_seal()
    v2._assert_v2_unconsumed()

    tile_paths = [
        v1.resolve_potsdam_tile_paths(v1.DATASET_ROOT, tile_id)
        for tile_id in v1.FROZEN_POTSDAM_TILE_IDS
    ]
    reference_contracts = v2._reference_contracts(tile_paths)
    if set(reference_contracts) != set(v1.FROZEN_POTSDAM_TILE_IDS):
        raise RuntimeError("execution preflight did not resolve every frozen reference contract")
    if not v1.COPDEM_PATH.is_file():
        raise FileNotFoundError(
            "Copernicus GLO-30 calibration tile is missing; run the non-consuming metadata "
            f"preflight first: {v1.COPDEM_PATH}"
        )

    torch.manual_seed(26175)
    np.random.seed(26175)
    torch.set_float32_matmul_precision("high")

    original_out_dir = v1.OUT_DIR
    original_reference_loader = v1._load_reference_after_seal
    prior = None
    reached: list[str] = []

    try:
        # Use the v2 derived-output namespace, but never create the v2 seal or report.
        v1.OUT_DIR = v2.OUT_DIR
        v1._load_reference_after_seal = _stop_before_reference_values

        device = v1._resolve_device()
        model, _checkpoint = v1._load_v4_model(device)
        prior = v1.DA3MonocularPrior(device="auto")

        print("=== DEPTHWIZARD POTSDAM EXTERNAL-V2 EXECUTION PREFLIGHT ===")
        print(f"Device: {device}")
        print("V2 protocol seal created: NO")
        print("Potsdam DSM pixel values read: NO")
        print(
            "Testing each frozen tile through RGB preparation, DA3 inference, V4 refinement, "
            "and Copernicus GLO-30 calibration."
        )

        for tile in tile_paths:
            try:
                result = v1._evaluate_tile(tile, model, prior, device)
            except _ReferenceBoundaryReached as exc:
                if str(exc) != tile.tile_id:
                    raise RuntimeError(
                        "execution preflight reached a mismatched reference boundary: "
                        f"expected {tile.tile_id}, got {exc}"
                    ) from exc
                reached.append(tile.tile_id)
                print(f"{tile.tile_id}: PRE-REFERENCE RUNTIME PASS")
                continue

            scene_report = result[0]
            raise RuntimeError(
                "execution preflight failed before the protected reference boundary for "
                f"tile {tile.tile_id}: status={scene_report.get('status')!r}, "
                f"reason={scene_report.get('reason')!r}"
            )
    finally:
        v1._load_reference_after_seal = original_reference_loader
        v1.OUT_DIR = original_out_dir
        if prior is not None:
            del prior

    expected = list(v1.FROZEN_POTSDAM_TILE_IDS)
    if reached != expected:
        raise RuntimeError(
            f"execution preflight did not reach all frozen tiles in order: {reached} != {expected}"
        )
    if v2.PROTOCOL_SEAL_PATH.exists():
        raise RuntimeError("execution preflight unexpectedly created the external-v2 protocol seal")

    print("=== EXECUTION PREFLIGHT COMPLETE ===")
    print(f"Frozen tiles reaching protected reference boundary: {len(reached)}/{len(expected)}")
    print("Potsdam DSM pixel values read: NO")
    print("V2 protocol seal created: NO")
    print("Target-free model/calibration runtime: PASS")
    print("Ready for one sealed external-v2 execution. ✅")


if __name__ == "__main__":
    main()
