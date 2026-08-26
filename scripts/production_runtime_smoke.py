from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from depthwizard.contracts import ProcessingRequest
from depthwizard.pipeline.runtime import ProductionElevationRuntime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real DepthWizard production elevation runtime on a local source raster."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dem", type=Path, default=None)
    parser.add_argument("--relative-only", action="store_true")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    requested_output = "rdsm" if args.relative_only else None
    result = ProductionElevationRuntime().run(
        ProcessingRequest(
            source=args.source,
            output_dir=args.output_dir,
            dem_path=args.dem,
            requested_output=requested_output,
            tile_size=args.tile_size,
            overlap=args.overlap,
        ),
        job_id="local-production-smoke",
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
