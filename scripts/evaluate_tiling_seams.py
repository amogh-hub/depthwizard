from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio

from depthwizard.evaluation.seams import evaluate_tiling_seams
from depthwizard.provenance.manifest import sha256_file


def _surface(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"surface raster does not exist: {path}")
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError("surface raster contains no bands")
        values = src.read(1).astype(np.float32)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    values[~valid] = np.nan
    return values, valid


def build_report(surface_path: Path, *, tile_size: int, overlap: int) -> dict[str, object]:
    surface, valid = _surface(surface_path)
    metrics = evaluate_tiling_seams(
        surface,
        tile_size=tile_size,
        overlap=overlap,
        valid_mask=valid,
    )
    return {
        "schema": "depthwizard.tiling-seam-evidence.v1",
        "surface_path": str(surface_path.resolve()),
        "surface_sha256": sha256_file(surface_path),
        "surface_shape": [int(surface.shape[0]), int(surface.shape[1])],
        "valid_pixels": int(valid.sum()),
        "metrics": metrics.as_dict(),
        "claim_boundary": (
            "This report measures tile-boundary discontinuity relative to ordinary local raster "
            "gradients. It does not by itself establish DSM accuracy or geographic generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic DepthWizard tile-seam evidence from a persisted surface."
    )
    parser.add_argument("surface", type=Path)
    parser.add_argument("--tile-size", type=int, required=True)
    parser.add_argument("--overlap", type=int, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = build_report(args.surface, tile_size=args.tile_size, overlap=args.overlap)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)
    print(f"DepthWizard tiling seam evidence: {args.output}")


if __name__ == "__main__":
    main()
