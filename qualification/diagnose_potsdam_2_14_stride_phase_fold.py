#!/usr/bin/env python3
"""Reference-free stride phase-fold diagnostic for Potsdam 2_14 tile imprint.

The visible defect was a repeated rounded-square pattern aligned with the tiled DA3 inference
lattice. This diagnostic therefore tests the most direct signature of that failure: whether a
repeatable surface fingerprint is phase-locked to the 896 px production stride.

It compares the already frozen corrective rDSM against the known-defective rDSM for the exact same
RGB scene. The official Potsdam DSM is never opened, stat'ed for raster content, or hashed here.
No PASS threshold is declared in this script; it is a negative-control diagnostic used to determine
whether 896 px periodicity actually separates the old and new pipelines before reference evaluation.
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

HEAD = "5e876709ad5a98cafb03a5c60ab7d84b135c6ffe"
OUTPUT = Path("workspace/urban-mosaic-corrective/5e87670-potsdam-2_14")
FREEZE_NAME = "pre-reference-freeze.json"
REPORT_NAME = "stride-phase-fold-negative-control-diagnostic.json"
PREVIEW_NAME = "stride-phase-fold-negative-control-preview.png"
OLD_RDSM = Path(
    "workspace/final-science-data/predictions/"
    "potsdam-2-14-urban-test/reconstruction/rdsm.tif"
)
FULL_HEIGHT = 6000
FULL_WIDTH = 6000
DOWNSAMPLE = 8
TARGET_STRIDE = 896
CONTROL_PERIODS = (768, 832, 960, 1024)
PERIODS = (*CONTROL_PERIODS[:2], TARGET_STRIDE, *CONTROL_PERIODS[2:])


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
                f"expected {FULL_WIDTH}x{FULL_HEIGHT} Potsdam surface, got "
                f"{src.width}x{src.height}: {path}"
            )
        height = math.ceil(src.height / DOWNSAMPLE)
        width = math.ceil(src.width / DOWNSAMPLE)
        values = src.read(
            1,
            out_shape=(height, width),
            resampling=Resampling.average,
        ).astype(np.float32)
        valid = src.read_masks(
            1,
            out_shape=(height, width),
            resampling=Resampling.nearest,
        ) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    if float(valid.mean()) < 0.99:
        raise DiagnosticError(f"diagnostic expects near-complete Potsdam coverage: {path}")
    fill = float(np.median(values[valid]))
    return np.where(valid, values, fill).astype(np.float32)


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
    padded = np.pad(values.astype(np.float64), radius, mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(0).cumsum(1)
    size = 2 * radius + 1
    summed = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    return (summed / float(size * size)).astype(np.float32)


def remove_plane(cell: np.ndarray) -> np.ndarray:
    height, width = cell.shape
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, height),
        np.linspace(-1.0, 1.0, width),
        indexing="ij",
    )
    design = np.column_stack(
        [np.ones(cell.size), xx.reshape(-1), yy.reshape(-1)]
    )
    target = cell.reshape(-1).astype(np.float64)
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return residual.reshape(cell.shape).astype(np.float32)


def phase_fold(values: np.ndarray, period_source_px: int) -> dict[str, Any]:
    if period_source_px % DOWNSAMPLE != 0:
        raise DiagnosticError("phase-fold periods must be divisible by downsample factor")
    period = period_source_px // DOWNSAMPLE
    normalized = robust_affine_normalize(values)
    rows = normalized.shape[0] // period
    cols = normalized.shape[1] // period
    if rows < 4 or cols < 4:
        raise DiagnosticError(f"insufficient phase repetitions for period {period_source_px}")

    crop = normalized[: rows * period, : cols * period]
    cells: list[np.ndarray] = []
    for row in range(rows):
        for col in range(cols):
            cell = crop[
                row * period : (row + 1) * period,
                col * period : (col + 1) * period,
            ]
            # Suppress tiny roofs/trees while retaining broad tile-scale shape, then remove each
            # cell's independent plane so genuine regional slope cannot masquerade as periodicity.
            prepared = remove_plane(box_blur(cell, radius=2))
            cells.append(prepared)

    stack = np.stack(cells, axis=0).astype(np.float64)
    fingerprint = np.mean(stack, axis=0)
    typical_energy = float(np.mean(stack * stack))
    fingerprint_energy = float(np.mean(fingerprint * fingerprint))
    coherence = 0.0 if typical_energy <= 1e-15 else fingerprint_energy / typical_energy
    coherence_rms_ratio = math.sqrt(max(coherence, 0.0))
    random_baseline = 1.0 / float(stack.shape[0])

    flat = stack.reshape(stack.shape[0], -1)
    flat -= np.mean(flat, axis=1, keepdims=True)
    norms = np.linalg.norm(flat, axis=1)
    usable = norms > 1e-15
    normalized_flat = flat[usable] / norms[usable, None]
    if normalized_flat.shape[0] >= 2:
        correlations = normalized_flat @ normalized_flat.T
        tri = np.triu_indices(correlations.shape[0], k=1)
        pairwise = correlations[tri]
        mean_pairwise = float(np.mean(pairwise))
        mean_abs_pairwise = float(np.mean(np.abs(pairwise)))
        p90_abs_pairwise = float(np.percentile(np.abs(pairwise), 90.0))
    else:
        mean_pairwise = 0.0
        mean_abs_pairwise = 0.0
        p90_abs_pairwise = 0.0

    return {
        "period_source_px": period_source_px,
        "period_analysis_px": period,
        "phase_rows": rows,
        "phase_cols": cols,
        "phase_cell_count": int(stack.shape[0]),
        "phase_coherence_energy_fraction": coherence,
        "phase_coherence_rms_ratio": coherence_rms_ratio,
        "independent_random_energy_baseline": random_baseline,
        "coherence_over_random_baseline": (
            coherence / random_baseline if random_baseline > 0.0 else None
        ),
        "mean_pairwise_phase_correlation": mean_pairwise,
        "mean_absolute_pairwise_phase_correlation": mean_abs_pairwise,
        "p90_absolute_pairwise_phase_correlation": p90_abs_pairwise,
        "fingerprint_rms": float(np.sqrt(fingerprint_energy)),
        "typical_cell_rms": float(np.sqrt(typical_energy)),
        "fingerprint": fingerprint.astype(np.float32),
    }


def numeric(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "fingerprint"}


def survey(values: np.ndarray) -> tuple[dict[str, dict[str, Any]], dict[int, np.ndarray]]:
    metrics: dict[str, dict[str, Any]] = {}
    fingerprints: dict[int, np.ndarray] = {}
    for period in PERIODS:
        result = phase_fold(values, period)
        fingerprints[period] = result["fingerprint"]
        metrics[str(period)] = numeric(result)
    target = metrics[str(TARGET_STRIDE)]
    controls = [metrics[str(period)] for period in CONTROL_PERIODS]
    control_coherence = np.asarray(
        [float(item["phase_coherence_energy_fraction"]) for item in controls],
        dtype=np.float64,
    )
    target["control_period_median_coherence"] = float(np.median(control_coherence))
    target["stride_specificity_vs_control_periods"] = (
        float(target["phase_coherence_energy_fraction"]) / float(np.median(control_coherence))
        if float(np.median(control_coherence)) > 1e-15
        else None
    )
    return metrics, fingerprints


def safe_ratio(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or abs(old) <= 1e-15:
        return None
    return float(new / old)


def fingerprint_panel(fingerprint: np.ndarray, label: str, scale: float) -> Image.Image:
    if scale <= 1e-15:
        display = np.full(fingerprint.shape, 127, dtype=np.uint8)
    else:
        normalized = np.clip(fingerprint / scale, -1.0, 1.0)
        display = np.round((normalized * 0.5 + 0.5) * 255.0).astype(np.uint8)
    image = Image.fromarray(display, mode="L").convert("RGB")
    image = image.resize((image.width * 4, image.height * 4), resample=Image.Resampling.BILINEAR)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, min(image.width, 520), 36), fill=(255, 255, 255))
    draw.text((8, 9), label, fill=(0, 0, 0), font=ImageFont.load_default())
    return image


def write_preview(old_fp: np.ndarray, new_fp: np.ndarray, output: Path) -> None:
    combined = np.concatenate([np.abs(old_fp).reshape(-1), np.abs(new_fp).reshape(-1)])
    scale = float(np.percentile(combined, 99.0))
    old_panel = fingerprint_panel(old_fp, "OLD 896 px phase-fold fingerprint", scale)
    new_panel = fingerprint_panel(new_fp, "NEW 896 px phase-fold fingerprint", scale)
    canvas = Image.new("RGB", (old_panel.width * 2, old_panel.height), "white")
    canvas.paste(old_panel, (0, 0))
    canvas.paste(new_panel, (old_panel.width, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    root = Path.cwd().resolve(strict=True)
    require_checkout(root)
    new_path, freeze = load_frozen_new_rdsm(root)
    old_path = (root / OLD_RDSM).resolve(strict=True)

    new_values = read_downsampled(new_path)
    old_values = read_downsampled(old_path)
    if new_values.shape != old_values.shape:
        raise DiagnosticError("old/new surfaces do not share the diagnostic grid")

    old_metrics, old_fingerprints = survey(old_values)
    new_metrics, new_fingerprints = survey(new_values)
    old_target = old_metrics[str(TARGET_STRIDE)]
    new_target = new_metrics[str(TARGET_STRIDE)]

    comparison = {
        "stride_phase_coherence_new_over_old": safe_ratio(
            float(new_target["phase_coherence_energy_fraction"]),
            float(old_target["phase_coherence_energy_fraction"]),
        ),
        "stride_phase_rms_ratio_new_over_old": safe_ratio(
            float(new_target["phase_coherence_rms_ratio"]),
            float(old_target["phase_coherence_rms_ratio"]),
        ),
        "stride_mean_pairwise_correlation_new_over_old": safe_ratio(
            abs(float(new_target["mean_pairwise_phase_correlation"])),
            abs(float(old_target["mean_pairwise_phase_correlation"])),
        ),
        "stride_specificity_new_over_old": safe_ratio(
            new_target.get("stride_specificity_vs_control_periods"),
            old_target.get("stride_specificity_vs_control_periods"),
        ),
    }

    report = {
        "schema_version": 1,
        "status": "DIAGNOSTIC_ONLY_NO_PASS_CLAIM",
        "git_head": HEAD,
        "geometry_pipeline_revision": freeze.get("geometry_pipeline_revision"),
        "method": "phase_fold_periodicity_after_local_plane_removal",
        "target_stride_px": TARGET_STRIDE,
        "control_periods_px": list(CONTROL_PERIODS),
        "old_known_defective_rdsm": {
            "path": str(old_path),
            "sha256": sha256(old_path),
            "period_survey": old_metrics,
        },
        "new_corrective_rdsm": {
            "path": str(new_path),
            "sha256": sha256(new_path),
            "period_survey": new_metrics,
        },
        "comparison": comparison,
        "reference_values_consumed": False,
        "reference_raster_opened_or_hashed": False,
        "purpose": (
            "negative-control engineering diagnostic only; tests exact 896 px phase-locking "
            "against nearby control periods before evaluator truth is exposed"
        ),
    }

    output_root = root / OUTPUT
    report_path = output_root / REPORT_NAME
    preview_path = output_root / PREVIEW_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_preview(
        old_fingerprints[TARGET_STRIDE],
        new_fingerprints[TARGET_STRIDE],
        preview_path,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print("PASS_STRIDE_PHASE_FOLD_NEGATIVE_CONTROL_DIAGNOSTIC_GENERATED")
    print(f"report={report_path}")
    print(f"preview={preview_path}")
    print("reference_values_consumed=false")
    print("DIAGNOSTIC_ONLY_NO_PASS_CLAIM")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
