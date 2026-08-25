from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

import rasterio

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "demo"
SOURCE = DATA_DIR / "RGB.byte.tif"
OUTPUT_DIR = ROOT / "artifacts" / "demo" / "geotiff-rdsm"
SOURCE_URL = "https://raw.githubusercontent.com/rasterio/rasterio/main/tests/data/RGB.byte.tif"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE.exists():
        print("Downloading public georeferenced RGB GeoTIFF smoke scene...")
        urllib.request.urlretrieve(SOURCE_URL, SOURCE)

    with rasterio.open(SOURCE) as src:
        if src.count < 3:
            raise SystemExit(f"Expected at least 3 bands, got {src.count}")
        if src.crs is None:
            raise SystemExit("Demo GeoTIFF unexpectedly has no CRS")
        print("Demo GeoTIFF:", SOURCE)
        print("Shape:", (src.height, src.width))
        print("Bands:", src.count)
        print("CRS:", src.crs)
        print("Transform:", src.transform)

    command = [
        sys.executable,
        "-m",
        "depthwizard.cli",
        "reconstruct-da3",
        str(SOURCE),
        str(OUTPUT_DIR),
        "--tile-size",
        "1024",
        "--overlap",
        "128",
    ]
    print("Running DepthWizard DA3 reconstruction...")
    subprocess.run(command, cwd=ROOT, check=True)
    print("DepthWizard georeferenced rDSM demo: PASS")
    print("Artifacts:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
