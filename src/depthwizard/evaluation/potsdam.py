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
