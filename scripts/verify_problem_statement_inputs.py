from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.io.raster import inspect_raster, read_rgb

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "acceptance" / "sih26175-input-formats"
REPORT_NAME = "input-format-contract.json"


class InputFormatQualificationFailure(RuntimeError):
    """The literal SIH26175 PNG/JPG/TIFF input contract was not satisfied."""


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _rgb_fixture() -> np.ndarray:
    rows, cols = np.mgrid[:48, :64]
    return np.stack(
        [
            (cols * 3 + rows) % 255,
            (rows * 5 + 31) % 255,
            ((rows + cols) * 2 + 67) % 255,
        ],
        axis=0,
    ).astype(np.uint8)


def _write_rgb(
    path: Path,
    *,
    driver: str,
    georeferenced: bool,
) -> None:
    data = _rgb_fixture()
    kwargs: dict[str, Any] = {
        "driver": driver,
        "height": data.shape[1],
        "width": data.shape[2],
        "count": 3,
        "dtype": "uint8",
    }
    if georeferenced:
        kwargs.update(
            crs="EPSG:32643",
            transform=from_origin(500000.0, 1400000.0, 2.0, 2.0),
        )
    with rasterio.open(path, "w", **kwargs) as dst:
        dst.write(data)


def _assert_non_georeferenced(path: Path, *, label: str) -> dict[str, object]:
    metadata = inspect_raster(path)
    rgb = read_rgb(path)
    if metadata.width != 64 or metadata.height != 48 or metadata.count != 3:
        raise InputFormatQualificationFailure(f"{label} dimensions/bands changed: {metadata}")
    if metadata.crs is not None:
        raise InputFormatQualificationFailure(f"{label} unexpectedly acquired CRS metadata")
    if metadata.ground_sample_distance_x is not None or metadata.ground_sample_distance_y is not None:
        raise InputFormatQualificationFailure(f"{label} invented metric ground spacing")
    if rgb.shape != (48, 64, 3):
        raise InputFormatQualificationFailure(f"{label} RGB read shape is {rgb.shape}")
    return {
        "path": str(path.resolve()),
        "driver": label,
        "width": metadata.width,
        "height": metadata.height,
        "bands": metadata.count,
        "crs": metadata.crs,
        "ground_sample_distance_x": metadata.ground_sample_distance_x,
        "ground_sample_distance_y": metadata.ground_sample_distance_y,
        "elevation_contract": "relative_only_no_metric_claim",
        "status": "PASS",
    }


def _assert_geotiff(path: Path) -> dict[str, object]:
    metadata = inspect_raster(path)
    rgb = read_rgb(path)
    if metadata.width != 64 or metadata.height != 48 or metadata.count != 3:
        raise InputFormatQualificationFailure(f"GeoTIFF dimensions/bands changed: {metadata}")
    if metadata.crs != "EPSG:32643":
        raise InputFormatQualificationFailure(f"GeoTIFF CRS is not preserved: {metadata.crs!r}")
    if metadata.ground_sample_distance_x is None or metadata.ground_sample_distance_y is None:
        raise InputFormatQualificationFailure("GeoTIFF did not expose trustworthy metric GSD")
    if not 1.99 <= metadata.ground_sample_distance_x <= 2.01:
        raise InputFormatQualificationFailure(
            f"GeoTIFF ground GSD X is not approximately 2 m: {metadata.ground_sample_distance_x}"
        )
    if not 1.99 <= metadata.ground_sample_distance_y <= 2.01:
        raise InputFormatQualificationFailure(
            f"GeoTIFF ground GSD Y is not approximately 2 m: {metadata.ground_sample_distance_y}"
        )
    if rgb.shape != (48, 64, 3):
        raise InputFormatQualificationFailure(f"GeoTIFF RGB read shape is {rgb.shape}")
    return {
        "path": str(path.resolve()),
        "driver": "GTiff",
        "width": metadata.width,
        "height": metadata.height,
        "bands": metadata.count,
        "crs": metadata.crs,
        "ground_sample_distance_x": metadata.ground_sample_distance_x,
        "ground_sample_distance_y": metadata.ground_sample_distance_y,
        "elevation_contract": "georeferenced_metric_calibration_eligible",
        "status": "PASS",
    }


def run_input_format_qualification(output_dir: Path = DEFAULT_OUTPUT_DIR) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_dir = output_dir / "fixtures"
    if fixture_dir.exists():
        shutil.rmtree(fixture_dir)
    fixture_dir.mkdir(parents=True, exist_ok=True)

    png_path = fixture_dir / "single-view-rgb.png"
    jpg_path = fixture_dir / "single-view-rgb.jpg"
    tiff_path = fixture_dir / "single-view-rgb.tiff"
    _write_rgb(png_path, driver="PNG", georeferenced=False)
    _write_rgb(jpg_path, driver="JPEG", georeferenced=False)
    _write_rgb(tiff_path, driver="GTiff", georeferenced=True)

    formats = {
        "png": _assert_non_georeferenced(png_path, label="PNG"),
        "jpg": _assert_non_georeferenced(jpg_path, label="JPEG"),
        "tiff": _assert_geotiff(tiff_path),
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_SIH26175_INPUT_FORMAT_CONTRACT",
        "git_head": _git_head(),
        "problem_statement_contract": {
            "non_georeferenced_png": "relative DSM pathway; no CRS/metres invented",
            "non_georeferenced_jpg": "relative DSM pathway; no CRS/metres invented",
            "georeferenced_tiff": "CRS/GSD preserved and eligible for evidence calibration to metric DSM",
        },
        "formats": formats,
        "claim_boundary": (
            "This harness proves literal PNG/JPG/TIFF ingestion, RGB readability and truthful "
            "geospatial metadata semantics. Production DA3 reconstruction and metric calibration "
            "remain covered by exact-head packaged acceptance and operator qualification."
        ),
    }
    report_path = output_dir / REPORT_NAME
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def main() -> int:
    report = run_input_format_qualification()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
