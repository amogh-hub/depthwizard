#!/usr/bin/env python3
"""Reference-free negative-control diagnostic for Potsdam 2_14 tile imprint.

This diagnostic deliberately does not open the official Potsdam DSM. It compares the frozen
corrective rDSM against the previously generated known-defective rDSM for the exact same RGB scene.
The goal is not to optimize an evaluation score. It is to verify that a claimed engineering
artifact detector can distinguish the corrected pipeline from its own negative control.

Signals:
1. tile-affine basis explained fraction: how much high-pass scene structure can be explained by the
   overlapping Hann-window tile lattice, including per-tile offset and scale terms;
2. tile-transition correlation: correlation between surface gradient energy and transitions in the
   theoretical tile-dominance field;
3. tile-core span dispersion: descriptive evidence of whether local relief has been artificially
   equalized across inference tiles.

No PASS threshold is declared here. The script emits a diagnostic-only report and preview so that a
subsequent acceptance rule can be justified from a detector that actually separates the known
negative control. Reference bytes remain unopened and unhashed.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.enums import Resampling

from depthwizard.tiling.grid import generate_tiles

HEAD = "5e876709ad5a98cafb03a5c60ab7d84b135c6ffe"
OUTPUT = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14")
FREEZE_NAME = "pre-reference-freeze.json"
REPORT_NAME = "tile-imprint-negative-control-diagnostic.json"
PREVIEW_NAME = "tile-imprint-negative-control-preview.png"
OLD_RDSM = Path(
    "workspace/final-science-data/predictions/"
    "potsdam-2-14-urban-test/reconstruction/rdsm.tif"
)
FULL_HEIGHT = 6000
FULL_WIDTH = 6000
TILE_SIZE = 1024
OVERLAP = 128
DOWNSAMPLE = 8
ANALYSIS_STEP = 2


class DiagnosticError(RuntimeError):
    """Reference-free diagnostic failed closed."""


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_checkout(root: Path) -> None:
    current = git(root, "rev-parse", "HEAD")
    if current != HEAD:
        raise DiagnosticError(f"diagnostic requires exact corrective head {HEAD}; current={current}")
    if git(root, "status", "--porcelain"):
        raise DiagnosticError("repository must be clean before diagnostic")


def load_frozen_new_rdsm(root: Path) -> tuple[Path, dict[str, Any]]:
    freeze_path = root / OUTPUT / FREEZE_NAME
    if not freeze_path.is_file():
        raise DiagnosticError(f"missing pre-reference freeze: {freeze_path}")
    payload = json.loads(freeze_path.read_text(encoding="utf-8"))
    if payload.get("git_head") != HEAD:
        raise DiagnosticError("pre-reference freeze belongs to another source head")
    if payload.get("reference_values_consumed") is not False:
        raise DiagnosticError("pre-reference freeze does not preserve reference-value separation")
    if payload.get("reference_raster_opened_or_hashed") is not False:
        raise DiagnosticError("pre-reference freeze does not preserve unopened-reference boundary")
    record = payload.get("relative_dsm")
    if not isinstance(record, dict):
        raise DiagnosticError("freeze is missing relative_dsm identity")
    path_text = record.get("path")
    expected_hash = record.get("sha256")
    if not isinstance(path_text, str) or not isinstance(expected_hash, str):
        raise DiagnosticError("malformed frozen relative_dsm identity")
    path = Path(path_text)
    if not path.is_file() or sha256(path) != expected_hash:
        raise DiagnosticError("frozen corrective rDSM no longer matches its SHA-256")
    return path.resolve(), payload


def read_downsampled(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        if src.height != FULL_HEIGHT or src.width != FULL_WIDTH:
            raise DiagnosticError(
                f"expected {FULL_WIDTH}x{FULL_HEIGHT} Potsdam surface, got {src.width}x{src.height}: {path}"
            )
        height = math.ceil(src.height / DOWNSAMPLE)
        width = math.ceil(src.width / DOWNSAMPLE)
        values = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        ).astype(np.float32)
        mask = src.read_masks(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ) > 0
        mask &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            mask &= values != np.float32(src.nodata)
    if float(mask.mean()) < 0.99:
        raise DiagnosticError(f"diagnostic expects near-complete Potsdam coverage: {path}")
    fill = float(np.median(values[mask]))
    return np.where(mask, values, fill).astype(np.float32)


def robust_affine_normalize(values: np.ndarray) -> np.ndarray:
    finite = values[np.isfinite(values)].astype(np.float64)
    p01, p99 = np.percentile(finite, [1.0, 99.0])
    span = float(p99 - p01)
    if span <= 1e-12:
        raise DiagnosticError("surface has no robust relief span")
    return ((values.astype(np.float64) - p01) / span).astype(np.float32)


def box_blur(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return values.astype(np.float32, copy=True)
    padded = np.pad(values.astype(np.float64), radius, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    summed = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (summed / float(size * size)).astype(np.float32)


def lowpass(values: np.ndarray) -> np.ndarray:
    # Three modest box filters approximate a smooth Gaussian without adding a qualification-only
    # dependency. At 8x downsampling, radius 10 corresponds to ~80 source pixels per pass.
    result = values.astype(np.float32, copy=True)
    for _ in range(3):
        result = box_blur(result, radius=10)
    return result


def hann_1d(coords: np.ndarray, start: int, length: int) -> np.ndarray:
    local = coords - float(start)
    inside = (local >= 0.0) & (local <= float(length - 1))
    out = np.zeros(coords.shape, dtype=np.float32)
    if length <= 1:
        out[inside] = 1.0
        return out
    phase = np.clip(local[inside] / float(length - 1), 0.0, 1.0)
    weight = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
    out[inside] = np.maximum(weight, 1e-3).astype(np.float32)
    return out


def tile_basis(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    y_full = (np.arange(height, dtype=np.float64) + 0.5) * (FULL_HEIGHT / height) - 0.5
    x_full = (np.arange(width, dtype=np.float64) + 0.5) * (FULL_WIDTH / width) - 0.5
    y_idx = np.arange(0, height, ANALYSIS_STEP)
    x_idx = np.arange(0, width, ANALYSIS_STEP)
    y_sample = y_full[y_idx]
    x_sample = x_full[x_idx]
    tiles = generate_tiles(FULL_HEIGHT, FULL_WIDTH, tile_size=TILE_SIZE, overlap=OVERLAP)

    raw_columns: list[np.ndarray] = []
    total = np.zeros((y_sample.size, x_sample.size), dtype=np.float32)
    for tile in tiles:
        wy = hann_1d(y_sample, tile.y, tile.height)
        wx = hann_1d(x_sample, tile.x, tile.width)
        weight = np.outer(wy, wx).astype(np.float32, copy=False)
        raw_columns.append(weight)
        total += weight
    if np.any(total <= 0.0):
        raise DiagnosticError("theoretical tile-weight lattice has uncovered samples")

    columns = np.column_stack([(column / total).reshape(-1) for column in raw_columns]).astype(
        np.float32,
        copy=False,
    )
    dominance = np.sum(columns * columns, axis=1).reshape(y_sample.size, x_sample.size)
    return columns, dominance.astype(np.float32, copy=False)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x64 = np.asarray(x, dtype=np.float64).reshape(-1)
    y64 = np.asarray(y, dtype=np.float64).reshape(-1)
    x64 -= float(np.mean(x64))
    y64 -= float(np.mean(y64))
    denominator = float(np.linalg.norm(x64) * np.linalg.norm(y64))
    if denominator <= 1e-15:
        return 0.0
    return float(np.dot(x64, y64) / denominator)


def tile_affine_basis_score(values: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    normalized = robust_affine_normalize(values)
    smooth = lowpass(normalized)
    residual = normalized - smooth
    residual_s = residual[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1).astype(np.float64)
    smooth_s = smooth[::ANALYSIS_STEP, ::ANALYSIS_STEP].reshape(-1).astype(np.float64)

    # Drop the final tile basis because all normalized tile weights sum to one. Include offset and
    # scale terms for each remaining tile plus a low-order global polynomial nuisance model.
    b = basis[:, :-1].astype(np.float64, copy=False)
    rows = residual_s.size
    h = values[::ANALYSIS_STEP, ::ANALYSIS_STEP].shape[0]
    w = values[::ANALYSIS_STEP, ::ANALYSIS_STEP].shape[1]
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, h),
        np.linspace(-1.0, 1.0, w),
        indexing="ij",
    )
    nuisance = np.column_stack(
        [
            np.ones(rows),
            xx.reshape(-1),
            yy.reshape(-1),
            (xx * yy).reshape(-1),
            (xx * xx).reshape(-1),
            (yy * yy).reshape(-1),
        ]
    )
    design = np.column_stack([nuisance, b, b * smooth_s[:, None]])
    coefficients, *_ = np.linalg.lstsq(design, residual_s, rcond=1e-7)
    fitted = design @ coefficients
    centered = residual_s - float(np.mean(residual_s))
    sst = float(np.dot(centered, centered))
    errors = residual_s - fitted
    sse = float(np.dot(errors, errors))
    r2 = 0.0 if sst <= 1e-15 else max(0.0, min(1.0, 1.0 - sse / sst))
    return {
        "tile_affine_basis_explained_fraction": r2,
        "highpass_rms": float(np.sqrt(np.mean(residual_s * residual_s))),
    }


def tile_transition_score(values: np.ndarray, dominance: np.ndarray) -> dict[str, float]:
    normalized = robust_affine_normalize(values)
    sampled = normalized[::ANALYSIS_STEP, ::ANALYSIS_STEP]
    gy, gx = np.gradient(sampled.astype(np.float64))
    surface_gradient = np.hypot(gx, gy)
    dy, dx = np.gradient(dominance.astype(np.float64))
    tile_transition = np.hypot(dx, dy)
    return {
        "gradient_to_tile_transition_correlation": pearson(surface_gradient, tile_transition),
        "absolute_highpass_to_tile_dominance_correlation": pearson(
            np.abs(sampled - lowpass(normalized)[::ANALYSIS_STEP, ::ANALYSIS_STEP]),
            dominance,
        ),
    }


def tile_core_span_stats(values: np.ndarray) -> dict[str, Any]:
    normalized = robust_affine_normalize(values)
    scale_y = normalized.shape[0] / FULL_HEIGHT
    scale_x = normalized.shape[1] / FULL_WIDTH
    spans: list[float] = []
    for tile in generate_tiles(FULL_HEIGHT, FULL_WIDTH, tile_size=TILE_SIZE, overlap=OVERLAP):
        margin = OVERLAP
        y0 = int(round((tile.y + margin) * scale_y))
        y1 = int(round((tile.y + tile.height - margin) * scale_y))
        x0 = int(round((tile.x + margin) * scale_x))
        x1 = int(round((tile.x + tile.width - margin) * scale_x))
        if y1 - y0 < 8 or x1 - x0 < 8:
            continue
        core = normalized[y0:y1, x0:x1].astype(np.float64)
        p05, p95 = np.percentile(core, [5.0, 95.0])
        spans.append(float(p95 - p05))
    array = np.asarray(spans, dtype=np.float64)
    mean = float(np.mean(array))
    return {
        "tile_core_count": int(array.size),
        "tile_core_span_mean": mean,
        "tile_core_span_median": float(np.median(array)),
        "tile_core_span_std": float(np.std(array)),
        "tile_core_span_cv": 0.0 if mean <= 1e-12 else float(np.std(array) / mean),
        "tile_core_span_min": float(np.min(array)),
        "tile_core_span_max": float(np.max(array)),
    }


def metrics(values: np.ndarray, basis: np.ndarray, dominance: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    result.update(tile_affine_basis_score(values, basis))
    result.update(tile_transition_score(values, dominance))
    result.update(tile_core_span_stats(values))
    return result


def grayscale(values: np.ndarray) -> Image.Image:
    normalized = robust_affine_normalize(values)
    clipped = np.clip(normalized, 0.0, 1.0)
    return Image.fromarray((clipped * 255.0).astype(np.uint8), mode="L").convert("RGB")


def gradient_preview(values: np.ndarray) -> Image.Image:
    normalized = robust_affine_normalize(values)
    gy, gx = np.gradient(normalized.astype(np.float64))
    gradient = np.hypot(gx, gy)
    p99 = float(np.percentile(gradient, 99.0))
    display = np.zeros(gradient.shape, dtype=np.float32)
    if p99 > 1e-12:
        display = np.clip(gradient / p99, 0.0, 1.0).astype(np.float32)
    return Image.fromarray((display * 255.0).astype(np.uint8), mode="L").convert("RGB")


def add_label(image: Image.Image, text: str) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    draw.rectangle((0, 0, min(result.width, 430), 34), fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0), font=ImageFont.load_default())
    return result


def write_preview(old: np.ndarray, new: np.ndarray, output: Path) -> None:
    panels = [
        add_label(grayscale(old), "OLD known-defective rDSM (robust normalized)"),
        add_label(grayscale(new), "NEW scene-global rDSM (robust normalized)"),
        add_label(gradient_preview(old), "OLD gradient magnitude"),
        add_label(gradient_preview(new), "NEW gradient magnitude"),
    ]
    width = panels[0].width
    height = panels[0].height
    canvas = Image.new("RGB", (width * 2, height * 2), "white")
    for index, panel in enumerate(panels):
        canvas.paste(panel, ((index % 2) * width, (index // 2) * height))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def ratio(new: float, old: float) -> float | None:
    if abs(old) <= 1e-15:
        return None
    return float(new / old)


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    new_path, freeze = load_frozen_new_rdsm(root)
    old_path = (root / OLD_RDSM).resolve(strict=True)

    new_values = read_downsampled(new_path)
    old_values = read_downsampled(old_path)
    if new_values.shape != old_values.shape:
        raise DiagnosticError("old/new downsampled surfaces do not share a diagnostic grid")

    basis, dominance = tile_basis(*new_values.shape)
    old_metrics = metrics(old_values, basis, dominance)
    new_metrics = metrics(new_values, basis, dominance)

    comparison = {
        "tile_affine_basis_explained_fraction_new_over_old": ratio(
            float(new_metrics["tile_affine_basis_explained_fraction"]),
            float(old_metrics["tile_affine_basis_explained_fraction"]),
        ),
        "gradient_to_tile_transition_correlation_new_over_old": ratio(
            abs(float(new_metrics["gradient_to_tile_transition_correlation"])),
            abs(float(old_metrics["gradient_to_tile_transition_correlation"])),
        ),
        "absolute_highpass_to_tile_dominance_correlation_new_over_old": ratio(
            abs(float(new_metrics["absolute_highpass_to_tile_dominance_correlation"])),
            abs(float(old_metrics["absolute_highpass_to_tile_dominance_correlation"])),
        ),
        "tile_core_span_cv_new_over_old": ratio(
            float(new_metrics["tile_core_span_cv"]),
            float(old_metrics["tile_core_span_cv"]),
        ),
    }

    report = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_NO_PASS_CLAIM",
        "git_head": HEAD,
        "corrective_geometry_pipeline_revision": freeze.get("geometry_pipeline_revision"),
        "new_rdsm": {"path": str(new_path), "sha256": sha256(new_path), "metrics": new_metrics},
        "old_known_defective_rdsm": {
            "path": str(old_path),
            "sha256": sha256(old_path),
            "metrics": old_metrics,
        },
        "comparison": comparison,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "purpose": (
            "negative-control engineering diagnostic only; no evaluator truth used and no acceptance "
            "threshold declared"
        ),
    }
    output_root = root / OUTPUT
    report_path = output_root / REPORT_NAME
    preview_path = output_root / PREVIEW_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_preview(old_values, new_values, preview_path)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS_TILE_IMPRINT_NEGATIVE_CONTROL_DIAGNOSTIC_GENERATED")
    print(f"report={report_path}")
    print(f"preview={preview_path}")
    print("reference_values_consumed=false")
    print("DIAGNOSTIC_ONLY_NO_PASS_CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
