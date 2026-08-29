#!/usr/bin/env python3
"""Generate final-science metric DSM predictions without opening evaluation references.

For every test/cross_sensor_test scene in the frozen registry this helper:
1. hashes the RGB input;
2. reconstructs the production DA3 rDSM through the frozen DepthWizard CLI;
3. downloads independent Copernicus GLO-30 tiles covering the RGB footprint;
4. mosaics those coarse tiles as calibration-only evidence;
5. runs the frozen `calibrate-dem` CLI to emit a metric DSM;
6. writes the draft prediction manifest used by `freeze_final_science_manifest.py`.

Reference paths in the registry are never opened, stat'ed for content, or hashed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.merge import merge
from rasterio.warp import transform_bounds

from depthwizard.data.registry import load_registry

FROZEN_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
DA3_MODEL_ID = "DA3MONO-LARGE"
DA3_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
DA3_CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
COPDEM_BASE = "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com"


class PredictionPreparationError(RuntimeError):
    """Final-science prediction preparation failed closed."""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_frozen_checkout(root: Path) -> None:
    head = _git(root, "rev-parse", "HEAD")
    if head != FROZEN_HEAD:
        raise PredictionPreparationError(
            f"prediction generation must run from {FROZEN_HEAD}; current={head}"
        )
    if _git(root, "status", "--porcelain"):
        raise PredictionPreparationError("repository must be clean before prediction generation")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_da3_checkpoint(root: Path) -> Path:
    explicit = os.environ.get("DEPTHWIZARD_DA3_CHECKPOINT", "").strip()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.append(
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--depth-anything--DA3MONO-LARGE"
        / "snapshots"
        / DA3_REVISION
        / "model.safetensors"
    )
    for candidate in candidates:
        if candidate.is_file() and _sha256(candidate) == DA3_CHECKPOINT_SHA256:
            return candidate.resolve()

    search_roots = [
        root / "apps" / "desktop" / "src-tauri" / "resources" / "depthwizard-core-runtime",
        root
        / "apps"
        / "desktop"
        / "src-tauri"
        / "target"
        / "release"
        / "bundle"
        / "macos"
        / "DepthWizard.app"
        / "Contents"
        / "Resources",
    ]
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for candidate in search_root.rglob("model.safetensors"):
            if candidate.is_file() and _sha256(candidate) == DA3_CHECKPOINT_SHA256:
                return candidate.resolve()
    raise PredictionPreparationError(
        "could not locate the pinned DA3 model.safetensors. Set DEPTHWIZARD_DA3_CHECKPOINT to "
        "the exact file whose SHA-256 is " + DA3_CHECKPOINT_SHA256
    )


def _copdem_tile_name(latitude: int, longitude: int) -> str:
    lat = f"{'N' if latitude >= 0 else 'S'}{abs(latitude):02d}_00"
    lon = f"{'E' if longitude >= 0 else 'W'}{abs(longitude):03d}_00"
    return f"Copernicus_DSM_COG_10_{lat}_{lon}_DEM"


def _download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DepthWizard-SIH26175-final-science/1"},
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
    except urllib.error.HTTPError as exc:
        temporary.unlink(missing_ok=True)
        raise PredictionPreparationError(f"Copernicus DEM HTTP {exc.code}: {url}") from exc
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise PredictionPreparationError(f"Copernicus DEM download was empty: {url}")
    temporary.replace(destination)


def _required_copdem_tiles(rgb_path: Path) -> list[tuple[int, int]]:
    with rasterio.open(rgb_path) as src:
        if src.crs is None or src.transform.is_identity:
            raise PredictionPreparationError(
                f"final metric science RGB must be georeferenced: {rgb_path}"
            )
        left, bottom, right, top = transform_bounds(
            src.crs,
            CRS.from_epsg(4326),
            *src.bounds,
            densify_pts=21,
        )
    lon_start = math.floor(left)
    lon_stop = math.ceil(right)
    lat_start = math.floor(bottom)
    lat_stop = math.ceil(top)
    tiles = [
        (lat, lon)
        for lat in range(lat_start, lat_stop)
        for lon in range(lon_start, lon_stop)
    ]
    if not tiles:
        raise PredictionPreparationError(f"could not derive Copernicus tile coverage for {rgb_path}")
    return tiles


def _copdem_mosaic(rgb_path: Path, cache_root: Path, output: Path) -> list[Path]:
    source_paths: list[Path] = []
    for lat, lon in _required_copdem_tiles(rgb_path):
        tile = _copdem_tile_name(lat, lon)
        filename = f"{tile}.tif"
        url = f"{COPDEM_BASE}/{tile}/{filename}"
        destination = cache_root / tile / filename
        _download(url, destination)
        source_paths.append(destination.resolve())

    datasets = [rasterio.open(path) for path in source_paths]
    try:
        mosaic, transform = merge(datasets)
        profile = datasets[0].profile.copy()
        profile.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=transform,
            count=mosaic.shape[0],
            compress="deflate",
            tiled=True,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp.tif")
        with rasterio.open(temporary, "w", **profile) as dst:
            dst.write(mosaic)
        temporary.replace(output)
    finally:
        for dataset in datasets:
            dataset.close()
    return source_paths


def _run(root: Path, *args: str) -> None:
    command = [sys.executable, "-m", "depthwizard.cli", *args]
    print("+ " + " ".join(command), file=sys.stderr)
    subprocess.run(command, cwd=root, check=True)


def _portable(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate frozen-head metric DSM final-science predictions.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("workspace/final-science-data/frozen-registry.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workspace/final-science-data/predictions"),
    )
    parser.add_argument(
        "--copdem-cache",
        type=Path,
        default=Path("workspace/final-science-data/calibration/copdem-cache"),
    )
    parser.add_argument(
        "--draft",
        type=Path,
        default=Path("workspace/final-science-data/predictions-draft.yaml"),
    )
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.repo.resolve(strict=True)
    _require_frozen_checkout(root)
    registry_path = (root / args.registry).resolve(strict=True)
    frozen_manifest = root / "workspace" / "final-science-data" / "frozen-predictions.yaml"
    freeze_report = frozen_manifest.with_name("frozen-predictions-freeze-report.json")
    if frozen_manifest.exists() or freeze_report.exists():
        raise PredictionPreparationError(
            "prediction identities are already frozen; refusing to regenerate or overwrite them"
        )
    registry = load_registry(registry_path)
    registry.assert_integrity(require_files=True)
    checkpoint = _find_da3_checkpoint(root)

    output_root = (root / args.output_root).resolve(strict=False)
    copdem_cache = (root / args.copdem_cache).resolve(strict=False)
    draft_path = (root / args.draft).resolve(strict=False)
    entries: list[dict[str, Any]] = []
    generation: list[dict[str, Any]] = []

    for scene in registry.scenes:
        if scene.split not in {"test", "cross_sensor_test"}:
            continue
        rgb = scene.rgb_path.resolve(strict=True)
        scene_dir = output_root / scene.scene_id
        reconstruction_dir = scene_dir / "reconstruction"
        rdsm = reconstruction_dir / "rdsm.tif"
        calibration = scene_dir / "calibration" / "copernicus-glo30-mosaic.tif"
        metric_dsm = scene_dir / "metric-dsm.tif"
        generation_report = scene_dir / "prediction-generation.json"

        rgb_sha = _sha256(rgb)
        reusable = False
        if generation_report.is_file() and metric_dsm.is_file() and calibration.is_file():
            prior = json.loads(generation_report.read_text(encoding="utf-8"))
            reusable = (
                isinstance(prior, dict)
                and prior.get("git_head") == FROZEN_HEAD
                and prior.get("source_sha256") == rgb_sha
                and prior.get("model_checkpoint_sha256") == DA3_CHECKPOINT_SHA256
                and prior.get("metric_dsm_sha256") == _sha256(metric_dsm)
                and prior.get("calibration_evidence_sha256") == _sha256(calibration)
            )
        if not reusable:
            if scene_dir.exists():
                shutil.rmtree(scene_dir)
            scene_dir.mkdir(parents=True, exist_ok=True)
            _run(
                root,
                "reconstruct-da3",
                str(rgb),
                str(reconstruction_dir),
                "--tile-size",
                str(args.tile_size),
                "--overlap",
                str(args.overlap),
            )
            if not rdsm.is_file():
                raise PredictionPreparationError(f"DA3 reconstruction did not create {rdsm}")
            copdem_sources = _copdem_mosaic(rgb, copdem_cache, calibration)
            _run(root, "calibrate-dem", str(rdsm), str(calibration), str(metric_dsm))
            if not metric_dsm.is_file():
                raise PredictionPreparationError(f"metric calibration did not create {metric_dsm}")
            report = {
                "schema_version": 1,
                "status": "PASS_FINAL_SCIENCE_PREDICTION_GENERATION",
                "git_head": FROZEN_HEAD,
                "scene_id": scene.scene_id,
                "source_rgb": str(rgb),
                "source_sha256": rgb_sha,
                "model_id": DA3_MODEL_ID,
                "model_checkpoint": str(checkpoint),
                "model_checkpoint_sha256": DA3_CHECKPOINT_SHA256,
                "relative_dsm": str(rdsm),
                "calibration_evidence": str(calibration),
                "calibration_evidence_sha256": _sha256(calibration),
                "calibration_sources": [str(path) for path in copdem_sources],
                "metric_dsm": str(metric_dsm),
                "metric_dsm_sha256": _sha256(metric_dsm),
                "reference_raster_opened_or_hashed": False,
            }
            generation_report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        else:
            report = json.loads(generation_report.read_text(encoding="utf-8"))

        entries.append(
            {
                "scene_id": scene.scene_id,
                "prediction_path": _portable(metric_dsm.resolve(), draft_path.parent),
                "calibration_evidence_paths": [
                    _portable(calibration.resolve(), draft_path.parent)
                ],
                "notes": (
                    "Production DA3MONO-LARGE relative geometry calibrated only with independent "
                    "Copernicus GLO-30 evidence. Evaluation reference was not opened or hashed."
                ),
            }
        )
        generation.append(report)

    expected = {scene.scene_id for scene in registry.scenes if scene.split in {"test", "cross_sensor_test"}}
    if {entry["scene_id"] for entry in entries} != expected:
        raise PredictionPreparationError("generated prediction set does not exactly match evaluation scenes")

    draft = {
        "schema_version": 1,
        "model_id": DA3_MODEL_ID,
        "checkpoint_path": _portable(checkpoint, draft_path.parent),
        "predictions": sorted(entries, key=lambda item: str(item["scene_id"])),
    }
    draft_path.parent.mkdir(parents=True, exist_ok=True)
    draft_path.write_text(yaml.safe_dump(draft, sort_keys=False), encoding="utf-8")
    summary_path = draft_path.with_name("prediction-generation-report.json")
    summary = {
        "schema_version": 1,
        "status": "PASS_FINAL_SCIENCE_PREDICTIONS_READY_TO_FREEZE",
        "git_head": FROZEN_HEAD,
        "registry": str(registry_path),
        "model_id": DA3_MODEL_ID,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
        "evaluation_scene_count": len(entries),
        "reference_rasters_opened_or_hashed": False,
        "draft_manifest": str(draft_path),
        "scenes": generation,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": summary["status"],
        "evaluation_scene_count": len(entries),
        "draft_manifest": str(draft_path),
        "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
