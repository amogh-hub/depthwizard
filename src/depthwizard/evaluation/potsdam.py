from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.crs import CRS

POTSDAM_NATIVE_GSD_M = 0.05
POTSDAM_BENCHMARK_GSD_M = 0.25
POTSDAM_NATIVE_SHAPE = (6000, 6000)
POTSDAM_CRS = CRS.from_epsg(32633)
FROZEN_POTSDAM_TILE_IDS = ("2_10", "3_13", "5_11", "6_14")
COPERNICUS_GLO30_EFFECTIVE_GSD_M = 30.0
COPERNICUS_GLO30_TILE = "Copernicus_DSM_COG_10_N52_00_E013_00_DEM"
COPERNICUS_GLO30_FILENAME = f"{COPERNICUS_GLO30_TILE}.tif"
COPERNICUS_GLO30_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    f"{COPERNICUS_GLO30_TILE}/{COPERNICUS_GLO30_FILENAME}"
)


@dataclass(frozen=True)
class PotsdamTilePaths:
    tile_id: str
    rgb: Path
    reference_dsm: Path


def _normalized_names(root: Path) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            files.setdefault(path.name.casefold(), []).append(path)
    return files


def _unique_named_file(index: dict[str, list[Path]], names: tuple[str, ...], role: str) -> Path:
    matches: list[Path] = []
    for name in names:
        matches.extend(index.get(name.casefold(), []))
    unique = sorted(set(matches))
    if not unique:
        raise FileNotFoundError(
            f"missing Potsdam {role}; expected one of: {', '.join(names)}"
        )
    if len(unique) > 1:
        rendered = ", ".join(str(path) for path in unique)
        raise RuntimeError(f"ambiguous Potsdam {role}; found multiple candidates: {rendered}")
    return unique[0]


def resolve_potsdam_tile_paths(root: str | Path, tile_id: str) -> PotsdamTilePaths:
    """Resolve one official Potsdam RGB/DSM pair without opening the DSM target.

    The resolver only inspects filenames. This lets the external protocol seal scene membership and
    all model/calibration settings before any reference raster values are observed.
    """
    dataset_root = Path(root)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Potsdam dataset root does not exist: {dataset_root}")
    parts = tile_id.split("_")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ValueError(f"invalid Potsdam tile id: {tile_id!r}")
    row, col = (int(part) for part in parts)
    index = _normalized_names(dataset_root)
    rgb = _unique_named_file(
        index,
        (
            f"top_potsdam_{row}_{col}_RGB.tif",
            f"top_potsdam_{row}_{col}_RGB.tiff",
        ),
        f"RGB tile {tile_id}",
    )
    reference = _unique_named_file(
        index,
        (
            f"dsm_potsdam_{row:02d}_{col:02d}.tif",
            f"dsm_potsdam_{row:02d}_{col:02d}.tiff",
            f"dsm_potsdam_{row}_{col}.tif",
            f"dsm_potsdam_{row}_{col}.tiff",
        ),
        f"reference DSM tile {tile_id}",
    )
    return PotsdamTilePaths(tile_id=tile_id, rgb=rgb, reference_dsm=reference)


def inspect_potsdam_rgb_contract(path: str | Path) -> dict[str, object]:
    """Validate the public Potsdam RGB contract without touching its reference DSM."""
    rgb_path = Path(path)
    with rasterio.open(rgb_path) as src:
        if (src.height, src.width) != POTSDAM_NATIVE_SHAPE:
            raise ValueError(
                f"Potsdam RGB must be 6000x6000; got {src.width}x{src.height}: {rgb_path}"
            )
        if src.count < 3:
            raise ValueError(f"Potsdam RGB must contain at least three bands: {rgb_path}")
        if src.transform.is_identity:
            raise ValueError(
                "Potsdam RGB has no usable affine transform. Keep the official .tfw beside the "
                f"TIFF before running the benchmark: {rgb_path}"
            )
        if abs(float(src.transform.b)) > 1e-9 or abs(float(src.transform.d)) > 1e-9:
            raise ValueError("Potsdam external benchmark currently requires a north-up affine grid")
        pixel_x = abs(float(src.transform.a))
        pixel_y = abs(float(src.transform.e))
        if not np.isclose(pixel_x, POTSDAM_NATIVE_GSD_M, atol=1e-5) or not np.isclose(
            pixel_y, POTSDAM_NATIVE_GSD_M, atol=1e-5
        ):
            raise ValueError(
                "Potsdam RGB affine pixel spacing does not match the official 5 cm contract: "
                f"{pixel_x:.6f} x {pixel_y:.6f} m"
            )
        if src.crs is not None and src.crs != POTSDAM_CRS:
            raise ValueError(
                f"Potsdam RGB CRS must be WGS84 / UTM zone 33N (EPSG:32633); got {src.crs}"
            )
        return {
            "path": str(rgb_path.resolve()),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs_from_tiff": src.crs.to_string() if src.crs is not None else None,
            "effective_crs": POTSDAM_CRS.to_string(),
            "native_gsd_m": POTSDAM_NATIVE_GSD_M,
            "bounds": [float(value) for value in src.bounds],
            "transform": [
                float(src.transform.a),
                float(src.transform.b),
                float(src.transform.c),
                float(src.transform.d),
                float(src.transform.e),
                float(src.transform.f),
            ],
        }


def validate_trailing_edge_reference_shape(
    rgb_shape: tuple[int, int],
    reference_shape: tuple[int, int],
    *,
    max_trailing_edge_deficit_px: int = 1,
) -> tuple[int, int]:
    """Validate a metadata-only trailing-edge shape defect.

    Shape tuples use ``(height, width)``. The reference may be smaller only on its trailing
    bottom/right edges because an identical affine transform fixes the same upper-left origin.
    """
    if max_trailing_edge_deficit_px < 0:
        raise ValueError("max_trailing_edge_deficit_px must be non-negative")
    rgb_height, rgb_width = rgb_shape
    reference_height, reference_width = reference_shape
    missing_rows = rgb_height - reference_height
    missing_cols = rgb_width - reference_width
    if missing_rows < 0 or missing_cols < 0:
        raise ValueError("Potsdam reference DSM may not extend beyond the RGB native shape")
    if missing_rows > max_trailing_edge_deficit_px or missing_cols > max_trailing_edge_deficit_px:
        raise ValueError(
            "Potsdam reference DSM trailing-edge deficit exceeds the sealed metadata tolerance: "
            f"missing_rows={missing_rows}, missing_cols={missing_cols}, "
            f"allowed={max_trailing_edge_deficit_px}"
        )
    return missing_rows, missing_cols


def inspect_potsdam_reference_contract(
    rgb_path: str | Path,
    reference_path: str | Path,
    *,
    max_trailing_edge_deficit_px: int = 1,
) -> dict[str, object]:
    """Inspect RGB/reference metadata only; never decode reference DSM pixel values.

    External-v2 permits at most one missing native 5 cm row/column at a trailing edge when the
    affine transform, CRS, and upper-left origin are otherwise identical. This formalizes the
    metadata discrepancy observed after external-v1 aborted on tile 3_13 without consulting DSM
    values or changing model/promotion decisions.
    """
    rgb_path = Path(rgb_path)
    reference_path = Path(reference_path)
    inspect_potsdam_rgb_contract(rgb_path)
    with rasterio.open(rgb_path) as rgb, rasterio.open(reference_path) as reference:
        if reference.count < 1:
            raise ValueError(f"Potsdam reference DSM has no raster band: {reference_path}")
        if not reference.transform.almost_equals(rgb.transform):
            raise ValueError(
                "Potsdam RGB/reference affine mismatch; external-v2 only permits a trailing-edge "
                f"shape deficit with identical affine metadata: {reference_path}"
            )
        reference_crs = reference.crs or POTSDAM_CRS
        rgb_crs = rgb.crs or POTSDAM_CRS
        if reference_crs != POTSDAM_CRS or rgb_crs != POTSDAM_CRS:
            raise ValueError(
                "Potsdam RGB/reference CRS must resolve to WGS84 / UTM zone 33N (EPSG:32633)"
            )
        missing_rows, missing_cols = validate_trailing_edge_reference_shape(
            (rgb.height, rgb.width),
            (reference.height, reference.width),
            max_trailing_edge_deficit_px=max_trailing_edge_deficit_px,
        )
        coverage_fraction = (reference.height * reference.width) / (rgb.height * rgb.width)
        return {
            "rgb_shape": [rgb.height, rgb.width],
            "reference_shape": [reference.height, reference.width],
            "missing_trailing_rows": missing_rows,
            "missing_trailing_columns": missing_cols,
            "native_coverage_fraction": float(coverage_fraction),
            "max_trailing_edge_deficit_px": max_trailing_edge_deficit_px,
            "reference_dtype": reference.dtypes[0],
            "effective_crs": POTSDAM_CRS.to_string(),
            "transform": [
                float(reference.transform.a),
                float(reference.transform.b),
                float(reference.transform.c),
                float(reference.transform.d),
                float(reference.transform.e),
                float(reference.transform.f),
            ],
            "rgb_bounds": [float(value) for value in rgb.bounds],
            "reference_bounds": [float(value) for value in reference.bounds],
        }


def benchmark_full_coverage_mask(
    *,
    target_height: int,
    target_width: int,
    target_transform: tuple[float, float, float, float, float, float],
    reference_bounds: tuple[float, float, float, float],
    atol: float = 1e-9,
) -> np.ndarray:
    """Mask benchmark cells whose entire footprint lies inside reference DSM coverage."""
    a, b, c, d, e, f = target_transform
    if target_height <= 0 or target_width <= 0:
        raise ValueError("target benchmark dimensions must be positive")
    if abs(b) > atol or abs(d) > atol or a <= 0.0 or e >= 0.0:
        raise ValueError("full-coverage masking requires a north-up affine transform")
    left, bottom, right, top = reference_bounds
    col_left = c + np.arange(target_width, dtype=np.float64) * a
    col_right = col_left + a
    row_top = f + np.arange(target_height, dtype=np.float64) * e
    row_bottom = row_top + e
    covered_cols = (col_left >= left - atol) & (col_right <= right + atol)
    covered_rows = (row_top <= top + atol) & (row_bottom >= bottom - atol)
    return covered_rows[:, None] & covered_cols[None, :]


def canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def protocol_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def write_or_verify_protocol_seal(path: str | Path, payload: dict[str, object]) -> str:
    """Atomically create an immutable protocol seal or verify an identical existing seal."""
    seal_path = Path(path)
    digest = protocol_sha256(payload)
    document = {"protocol_sha256": digest, "protocol": payload}
    if seal_path.exists():
        existing = json.loads(seal_path.read_text(encoding="utf-8"))
        if existing != document:
            raise RuntimeError(
                "external benchmark protocol seal already exists with different contents; "
                "do not mutate or reuse a consumed benchmark protocol"
            )
        return digest
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = seal_path.with_suffix(seal_path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(seal_path)
    return digest
