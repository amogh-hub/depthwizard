from __future__ import annotations

"""DepthWizard SIH26175 final-science Stage 1 orchestration.

NON-PRODUCTION HELPER.

This file intentionally lives only on the qualification/final-evidence-orchestration branch.
It must be executed while the working tree itself remains on the frozen production head
339bdf485149f552db846543b9e09377b567c19c (for example by materializing it with `git show`).

Stage 1 performs only the reference-blind half of the final campaign:
- verifies the frozen production head/worktree;
- downloads immutable RGB inputs and independent Copernicus GLO-30 calibration evidence;
- prepares a new, previously-unused Potsdam urban RGB tile without reading its DSM values;
- runs the frozen production calibrated-DA3 runtime for all evaluation scenes;
- freezes checkpoint/prediction/calibration-evidence SHA-256 identities;
- NEVER downloads, opens, hashes, rasterizes, or evaluates the final LiDAR/DSM references.

Do not add reference handling here. Reference materialization belongs to Stage 2 only after the
prediction freeze report says PASS_FINAL_SCIENCE_PREDICTION_FREEZE.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

import rasterio
import yaml
from affine import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling

from depthwizard.contracts import ProcessingRequest, ProjectRunStatus
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file

PRODUCTION_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
DA3_CHECKPOINT_SHA256 = "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
DA3_REVISION = "f465978e618db8cc79c83b8bbf24964857db1875"
USER_AGENT = "DepthWizard-SIH26175-final-science/1.0"

ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = ROOT / "artifacts" / "final-science-inputs"
SCIENCE_ROOT = ROOT / "artifacts" / "final-science"
PREDICTION_ROOT = SCIENCE_ROOT / "predictions"
PROJECT_ROOT = SCIENCE_ROOT / "projects"

PUBLIC_SCENES = {
    "sparse-sjer-2018-259000-4110000": {
        "terrain": "sparse",
        "dataset": "NeonTreeEvaluation-NEON-AOP",
        "geographic_group": "sjer-2018-259000-4110000",
        "rgb_rel": "sparse-sjer/rgb/2018_SJER_3_259000_4110000_image.tif",
        "rgb_url": "https://zenodo.org/records/5593238/files/2018_SJER_3_259000_4110000_image.tif?download=1",
        "rgb_md5": "79b3804e212761275f8612b1fc3f0a8f",
        "dem_rel": "sparse-sjer/calibration/copernicus-N37-W120.tif",
        "dem_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N37_00_W120_00_DEM/Copernicus_DSM_COG_10_N37_00_W120_00_DEM.tif",
        "reference_rel": "references/sparse-sjer-dsm.tif",
        "nominal_gsd_m": 0.1,
    },
    "hilly-niwo-2018-450000-4426000": {
        "terrain": "hilly",
        "dataset": "NeonTreeEvaluation-NEON-AOP",
        "geographic_group": "niwo-2018-450000-4426000",
        "rgb_rel": "hilly-niwo/rgb/2018_NIWO_2_450000_4426000_image_crop.tif",
        "rgb_url": "https://zenodo.org/records/5593238/files/2018_NIWO_2_450000_4426000_image_crop.tif?download=1",
        "rgb_md5": "c8f700eca920c6f0b93d16e6e26cc5a7",
        "dem_rel": "hilly-niwo/calibration/copernicus-N39-W106.tif",
        "dem_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N39_00_W106_00_DEM/Copernicus_DSM_COG_10_N39_00_W106_00_DEM.tif",
        "reference_rel": "references/hilly-niwo-dsm.tif",
        "nominal_gsd_m": 0.1,
    },
    "forest-harv-2018-733000-4698000": {
        "terrain": "forested",
        "dataset": "NeonTreeEvaluation-NEON-AOP",
        "geographic_group": "harv-2018-733000-4698000",
        "rgb_rel": "forest-harv/rgb/2018_HARV_5_733000_4698000_image_crop.tif",
        "rgb_url": "https://zenodo.org/records/5593238/files/2018_HARV_5_733000_4698000_image_crop.tif?download=1",
        "rgb_md5": "a8dbe47fbdb5d94b281e3dbc2ec76e27",
        "dem_rel": "forest-harv/calibration/copernicus-N42-W073.tif",
        "dem_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N42_00_W073_00_DEM/Copernicus_DSM_COG_10_N42_00_W073_00_DEM.tif",
        "reference_rel": "references/forest-harv-dsm.tif",
        "nominal_gsd_m": 0.1,
    },
    "cross-bart-2018-322000-4882000": {
        "terrain": "forested",
        "dataset": "NeonTreeEvaluation-NEON-AOP",
        "geographic_group": "bart-2018-322000-4882000",
        "rgb_rel": "cross-bart/rgb/2018_BART_4_322000_4882000_image_crop.tif",
        "rgb_url": "https://zenodo.org/records/5593238/files/2018_BART_4_322000_4882000_image_crop.tif?download=1",
        "rgb_md5": "06bccf59dc274b97494466cc62c1a5a5",
        "dem_rel": "cross-bart/calibration/copernicus-N44-W078.tif",
        "dem_url": "https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_N44_00_W078_00_DEM/Copernicus_DSM_COG_10_N44_00_W078_00_DEM.tif",
        "reference_rel": "references/cross-bart-dsm.tif",
        "nominal_gsd_m": 0.1,
    },
}

ORTHOLOC_DOP_URL = (
    "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/demo/urban_residential_DOP.tif"
)
POTSDAM_COPDEM_URL = (
    "https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/"
    "Copernicus_DSM_COG_10_N52_00_E013_00_DEM/"
    "Copernicus_DSM_COG_10_N52_00_E013_00_DEM.tif"
)


def run(*args: str) -> str:
    completed = subprocess.run(
        list(args), cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def assert_frozen_tree() -> None:
    head = run("git", "rev-parse", "HEAD")
    if head != PRODUCTION_HEAD:
        raise SystemExit(
            f"REFUSED: execute Stage 1 only while the worktree itself is at frozen production head {PRODUCTION_HEAD}; got {head}"
        )
    dirty = run("git", "status", "--porcelain")
    if dirty:
        raise SystemExit("REFUSED: frozen production worktree is not clean")
    if not (ROOT / ".venv" / "bin" / "python").is_file():
        raise SystemExit("REFUSED: exact-head .venv is missing")


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path, *, expected_md5: str | None = None) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        if expected_md5 is None or digest(destination, "md5") == expected_md5:
            return destination
        raise SystemExit(f"REFUSED: cached file checksum mismatch: {destination}")
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise SystemExit(f"download produced an empty file: {url}")
    if expected_md5 is not None and digest(temporary, "md5") != expected_md5:
        actual = digest(temporary, "md5")
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSED: immutable archive MD5 mismatch for {destination.name}: expected={expected_md5}, actual={actual}"
        )
    temporary.replace(destination)
    return destination


def locate_da3_checkpoint() -> Path:
    candidates = [
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--depth-anything--DA3MONO-LARGE"
        / "snapshots"
        / DA3_REVISION
        / "model.safetensors",
        Path.home()
        / "Library"
        / "Caches"
        / "huggingface"
        / "hub"
        / "models--depth-anything--DA3MONO-LARGE"
        / "snapshots"
        / DA3_REVISION
        / "model.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file() and sha256_file(candidate) == DA3_CHECKPOINT_SHA256:
            return candidate.resolve()
    search_roots = [
        Path.home() / ".cache" / "huggingface" / "hub" / "models--depth-anything--DA3MONO-LARGE" / "snapshots",
        Path.home() / "Library" / "Caches" / "huggingface" / "hub" / "models--depth-anything--DA3MONO-LARGE" / "snapshots",
    ]
    for base in search_roots:
        if not base.is_dir():
            continue
        for candidate in base.glob("*/model.safetensors"):
            if candidate.is_file() and sha256_file(candidate) == DA3_CHECKPOINT_SHA256:
                return candidate.resolve()
    raise SystemExit(
        "REFUSED: pinned DA3 model.safetensors was not found in the qualified Hugging Face cache; do not substitute another checkpoint"
    )


def locate_unused_potsdam_rgb() -> Path:
    roots = []
    env_root = os.environ.get("DEPTHWIZARD_POTSDAM_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(ROOT / "data" / "external" / "isprs-potsdam")
    names = {"top_potsdam_4_12_rgb.tif", "top_potsdam_4_12_rgb.tiff"}
    matches: list[Path] = []
    for base in roots:
        if not base.is_dir():
            continue
        matches.extend(
            path for path in base.rglob("*") if path.is_file() and path.name.casefold() in names
        )
    unique = sorted({path.resolve() for path in matches})
    if len(unique) != 1:
        raise SystemExit(
            "Stage 1 needs exactly one previously-unused official Potsdam RGB tile 4_12 under data/external/isprs-potsdam (or DEPTHWIZARD_POTSDAM_ROOT). "
            f"Found {len(unique)}. The DSM/reference is deliberately NOT needed or opened in Stage 1."
        )
    return unique[0]


def prepare_potsdam_rgb(source: Path) -> Path:
    output = INPUT_ROOT / "urban" / "potsdam-4_12-rgb-025m.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.is_file() and output.stat().st_size > 0:
        return output
    with rasterio.open(source) as src:
        if src.count < 3:
            raise SystemExit("Potsdam 4_12 RGB has fewer than 3 bands")
        if src.transform.is_identity:
            raise SystemExit(
                "Potsdam 4_12 RGB has no affine transform; keep its official .tfw beside the TIFF"
            )
        if src.width != 6000 or src.height != 6000:
            raise SystemExit(
                f"Potsdam 4_12 RGB must be the official 6000x6000 tile; got {src.width}x{src.height}"
            )
        out_width = 1200
        out_height = 1200
        data = src.read(
            [1, 2, 3],
            out_shape=(3, out_height, out_width),
            resampling=Resampling.average,
        )
        transform = src.transform * Affine.scale(src.width / out_width, src.height / out_height)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            width=out_width,
            height=out_height,
            count=3,
            dtype=str(data.dtype),
            crs=CRS.from_epsg(32633),
            transform=transform,
            compress="deflate",
            tiled=True,
        )
    temporary = output.with_suffix(".tmp.tif")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(data)
        dst.update_tags(
            DEPTHWIZARD_ROLE="FINAL_SCIENCE_REFERENCE_BLIND_RGB_INPUT",
            SOURCE_DATASET="ISPRS Potsdam",
            SOURCE_TILE="4_12",
            SOURCE_NATIVE_GSD_M="0.05",
            FINAL_SCIENCE_GSD_M="0.25",
            REFERENCE_OPENED="false",
        )
    temporary.replace(output)
    return output


def build_inputs() -> tuple[dict[str, tuple[Path, Path]], Path]:
    eval_inputs: dict[str, tuple[Path, Path]] = {}
    for scene_id, spec in PUBLIC_SCENES.items():
        rgb = download(
            spec["rgb_url"],
            INPUT_ROOT / spec["rgb_rel"],
            expected_md5=spec["rgb_md5"],
        )
        dem = download(spec["dem_url"], INPUT_ROOT / spec["dem_rel"])
        eval_inputs[scene_id] = (rgb, dem)

    ortholoc = download(
        ORTHOLOC_DOP_URL,
        INPUT_ROOT / "train" / "ortholoc" / "urban_residential_DOP.tif",
    )

    potsdam_source = locate_unused_potsdam_rgb()
    potsdam_rgb = prepare_potsdam_rgb(potsdam_source)
    potsdam_dem = download(
        POTSDAM_COPDEM_URL,
        INPUT_ROOT / "urban" / "calibration" / "copernicus-N52-E013.tif",
    )
    eval_inputs["urban-potsdam-4_12"] = (potsdam_rgb, potsdam_dem)
    return eval_inputs, ortholoc


def write_registry(eval_inputs: dict[str, tuple[Path, Path]], ortholoc: Path) -> Path:
    scenes: list[dict[str, object]] = [
        {
            "scene_id": "development-ortholoc-urban-residential",
            "dataset": "TUM-OrthoLoC-development-lineage",
            "split": "train",
            "terrain": "urban",
            "geographic_group": "ortholoc-urban-residential",
            "sensor": "ortholoc-public-orthophoto-development-lineage",
            "rgb_path": os.path.relpath(ortholoc, SCIENCE_ROOT),
            "reference_path": None,
            "license_id": "OrthoLoC-upstream-terms",
            "notes": "DepthWizard project-development lineage sentinel only; NOT a claim about DA3 foundation-model pretraining.",
        },
        {
            "scene_id": "urban-potsdam-4_12",
            "dataset": "ISPRS-Potsdam",
            "split": "test",
            "terrain": "urban",
            "geographic_group": "potsdam-4_12",
            "sensor": "BSF-Swissphoto-Potsdam-airborne-TOP",
            "rgb_path": os.path.relpath(eval_inputs["urban-potsdam-4_12"][0], SCIENCE_ROOT),
            "reference_path": "references/urban-potsdam-4_12-dsm.tif",
            "nominal_gsd_m": 0.25,
            "license_id": "ISPRS-Potsdam-benchmark-terms",
            "notes": "Previously unused final-campaign urban tile. Reference values must remain unopened until prediction freeze.",
        },
    ]
    for scene_id, spec in PUBLIC_SCENES.items():
        split = "cross_sensor_test" if scene_id.startswith("cross-") else "test"
        scenes.append(
            {
                "scene_id": scene_id,
                "dataset": spec["dataset"],
                "split": split,
                "terrain": spec["terrain"],
                "geographic_group": spec["geographic_group"],
                "sensor": "NEON-AOP-airborne-RGB",
                "rgb_path": os.path.relpath(eval_inputs[scene_id][0], SCIENCE_ROOT),
                "reference_path": spec["reference_rel"],
                "nominal_gsd_m": spec["nominal_gsd_m"],
                "license_id": "Zenodo-5593238",
                "notes": (
                    "Formal cross-sensor holdout relative to explicit OrthoLoC project-development lineage; not a claim about DA3 pretraining."
                    if split == "cross_sensor_test"
                    else None
                ),
            }
        )
    registry = SCIENCE_ROOT / "registry.yaml"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump({"schema_version": 1, "scenes": scenes}, sort_keys=False),
        encoding="utf-8",
    )
    return registry


def run_predictions(eval_inputs: dict[str, tuple[Path, Path]]) -> dict[str, tuple[Path, Path]]:
    PREDICTION_ROOT.mkdir(parents=True, exist_ok=True)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    results: dict[str, tuple[Path, Path]] = {}
    runtime = ProductionElevationRuntime()
    ordered_ids = [
        "urban-potsdam-4_12",
        "sparse-sjer-2018-259000-4110000",
        "hilly-niwo-2018-450000-4426000",
        "forest-harv-2018-733000-4698000",
        "cross-bart-2018-322000-4882000",
    ]
    for scene_id in ordered_ids:
        rgb, dem = eval_inputs[scene_id]
        project_dir = PROJECT_ROOT / scene_id
        result = runtime.run(
            ProcessingRequest(
                source=rgb,
                output_dir=project_dir,
                dem_path=dem,
                requested_output="dsm",
                tile_size=1024,
                overlap=128,
                harmonize_overlaps=True,
                low_frequency_sigma_px=24.0,
            ),
            job_id=f"final-science-{scene_id}",
        )
        if result.status != ProjectRunStatus.COMPLETE or result.primary_product is None:
            raise SystemExit(f"production prediction failed for {scene_id}: {result.as_dict()}")
        prediction = PREDICTION_ROOT / f"{scene_id}-dsm.tif"
        shutil.copy2(result.primary_product, prediction)
        results[scene_id] = (prediction, dem)
        print(f"PREDICTION_FROZEN_CANDIDATE {scene_id} {sha256_file(prediction)}")
    return results


def write_draft_and_freeze(
    registry: Path,
    predictions: dict[str, tuple[Path, Path]],
    checkpoint: Path,
) -> Path:
    draft = SCIENCE_ROOT / "predictions-draft.yaml"
    frozen = SCIENCE_ROOT / "predictions-frozen.yaml"
    entries = []
    for scene_id in sorted(predictions):
        prediction, dem = predictions[scene_id]
        entries.append(
            {
                "scene_id": scene_id,
                "prediction_path": os.path.relpath(prediction, SCIENCE_ROOT),
                "calibration_evidence_paths": [os.path.relpath(dem, SCIENCE_ROOT)],
                "notes": "Frozen production calibrated-DA3 metric DSM; evaluator reference not opened during generation.",
            }
        )
    payload = {
        "schema_version": 1,
        "model_id": "calibrated-da3-production",
        "checkpoint_path": str(checkpoint),
        "predictions": entries,
    }
    draft.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    python = ROOT / ".venv" / "bin" / "python"
    subprocess.run(
        [
            str(python),
            str(ROOT / "scripts" / "freeze_final_science_manifest.py"),
            str(registry),
            str(draft),
            str(frozen),
        ],
        cwd=ROOT,
        check=True,
    )
    report = frozen.with_name(f"{frozen.stem}-freeze-report.json")
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("status") != "PASS_FINAL_SCIENCE_PREDICTION_FREEZE":
        raise SystemExit(f"prediction freeze did not pass: {report}")
    if data.get("git_head") != PRODUCTION_HEAD:
        raise SystemExit(f"prediction freeze was not produced by frozen production head: {report}")
    if data.get("reference_rasters_opened_or_hashed") is not False:
        raise SystemExit(f"reference-blind freeze invariant failed: {report}")
    return report


def main() -> int:
    assert_frozen_tree()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    checkpoint = locate_da3_checkpoint()
    print(f"FROZEN_HEAD {PRODUCTION_HEAD}")
    print(f"DA3_CHECKPOINT {checkpoint}")
    print("REFERENCE_RASTERS_OPENED_OR_HASHED false")
    eval_inputs, ortholoc = build_inputs()
    registry = write_registry(eval_inputs, ortholoc)
    predictions = run_predictions(eval_inputs)
    freeze_report = write_draft_and_freeze(registry, predictions, checkpoint)
    final = {
        "status": "PASS_FINAL_SCIENCE_STAGE1_REFERENCE_BLIND",
        "git_head": PRODUCTION_HEAD,
        "registry": str(registry),
        "freeze_report": str(freeze_report),
        "reference_rasters_opened_or_hashed": False,
        "next_step": "Only now may Stage 2 materialize independent Potsdam/Zenodo LiDAR references and run the final evaluator.",
    }
    out = SCIENCE_ROOT / "stage1-reference-blind-report.json"
    out.write_text(json.dumps(final, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(final, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
