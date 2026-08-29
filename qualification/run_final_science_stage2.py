from __future__ import annotations

"""DepthWizard SIH26175 final-science Stage 2 orchestration.

NON-PRODUCTION HELPER. Execute only from an ignored artifacts/ materialization while the actual
worktree remains on frozen production head 339bdf485149f552db846543b9e09377b567c19c.

Stage 2 is fail-closed: it refuses to download/open/hash any evaluation reference until Stage 1 has
already emitted PASS_FINAL_SCIENCE_PREDICTION_FREEZE on the exact production head with
reference_rasters_opened_or_hashed=false.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS

PRODUCTION_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
USER_AGENT = "DepthWizard-SIH26175-final-science-reference/1.0"

ROOT = Path(__file__).resolve().parents[1]
SCIENCE_ROOT = ROOT / "artifacts" / "final-science"
INPUT_ROOT = ROOT / "artifacts" / "final-science-inputs"
REFERENCE_ROOT = SCIENCE_ROOT / "references"
REGISTRY = SCIENCE_ROOT / "registry.yaml"
FROZEN_MANIFEST = SCIENCE_ROOT / "predictions-frozen.yaml"
FREEZE_REPORT = SCIENCE_ROOT / "predictions-frozen-freeze-report.json"
STAGE1_REPORT = SCIENCE_ROOT / "stage1-reference-blind-report.json"

NATURAL_REFERENCES = {
    "sparse-sjer": {
        "url": "https://zenodo.org/records/5593238/files/2018_SJER_3_259000_4110000_image.laz?download=1",
        "md5": "ad5549a021ced41c59e96a587fb4d37d",
        "laz": INPUT_ROOT / "sparse-sjer" / "reference" / "2018_SJER_3_259000_4110000_image.laz",
        "out": REFERENCE_ROOT / "sparse-sjer-dsm.tif",
        "epsg": 32611,
        "resolution_m": 1.0,
    },
    "hilly-niwo": {
        "url": "https://zenodo.org/records/5593238/files/2018_NIWO_2_450000_4426000_image_crop.laz?download=1",
        "md5": "adb2a40470af9de4acea1bed8ce1cac2",
        "laz": INPUT_ROOT / "hilly-niwo" / "reference" / "2018_NIWO_2_450000_4426000_image_crop.laz",
        "out": REFERENCE_ROOT / "hilly-niwo-dsm.tif",
        "epsg": 32613,
        "resolution_m": 1.0,
    },
    "forest-harv": {
        "url": "https://zenodo.org/records/5593238/files/2018_HARV_5_733000_4698000_image_crop.laz?download=1",
        "md5": "1281d93764ddbd52cbd9348f7754459e",
        "laz": INPUT_ROOT / "forest-harv" / "reference" / "2018_HARV_5_733000_4698000_image_crop.laz",
        "out": REFERENCE_ROOT / "forest-harv-dsm.tif",
        "epsg": 32618,
        "resolution_m": 1.0,
    },
    "cross-bart": {
        "url": "https://zenodo.org/records/5593238/files/2018_BART_4_322000_4882000_image_crop.laz?download=1",
        "md5": "dda4b19cdc50a8bce71834a0ea8324d0",
        "laz": INPUT_ROOT / "cross-bart" / "reference" / "2018_BART_4_322000_4882000_image_crop.laz",
        "out": REFERENCE_ROOT / "cross-bart-dsm.tif",
        "epsg": 32618,
        "resolution_m": 1.0,
    },
}


def run(*args: str, capture: bool = True) -> str:
    completed = subprocess.run(
        list(args), cwd=ROOT, check=True, capture_output=capture, text=True
    )
    return completed.stdout.strip() if capture else ""


def assert_frozen_tree() -> None:
    head = run("git", "rev-parse", "HEAD")
    if head != PRODUCTION_HEAD:
        raise SystemExit(
            f"REFUSED: Stage 2 requires frozen production head {PRODUCTION_HEAD}; got {head}"
        )
    if run("git", "status", "--porcelain"):
        raise SystemExit("REFUSED: frozen production worktree is not clean")


def assert_reference_gate_open() -> None:
    required = [STAGE1_REPORT, FREEZE_REPORT, REGISTRY, FROZEN_MANIFEST]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "REFUSED: Stage 1 freeze evidence is incomplete; no reference may be exposed. missing="
            + repr(missing)
        )
    stage1 = json.loads(STAGE1_REPORT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_REPORT.read_text(encoding="utf-8"))
    if stage1.get("status") != "PASS_FINAL_SCIENCE_STAGE1_REFERENCE_BLIND":
        raise SystemExit("REFUSED: Stage 1 did not pass")
    if stage1.get("git_head") != PRODUCTION_HEAD:
        raise SystemExit("REFUSED: Stage 1 was not produced by the frozen production head")
    if stage1.get("reference_rasters_opened_or_hashed") is not False:
        raise SystemExit("REFUSED: Stage 1 does not prove reference-blind prediction generation")
    if freeze.get("status") != "PASS_FINAL_SCIENCE_PREDICTION_FREEZE":
        raise SystemExit("REFUSED: prediction identity freeze did not pass")
    if freeze.get("git_head") != PRODUCTION_HEAD:
        raise SystemExit("REFUSED: prediction freeze was not produced by the frozen production head")
    if freeze.get("reference_rasters_opened_or_hashed") is not False:
        raise SystemExit("REFUSED: prediction-freeze report does not preserve the reference boundary")
    if int(freeze.get("evaluation_scene_count", 0)) != 5:
        raise SystemExit("REFUSED: prediction freeze does not contain exactly five evaluation scenes")


def digest(path: Path, algorithm: str) -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, destination: Path, expected_md5: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        actual = digest(destination, "md5")
        if actual != expected_md5:
            raise SystemExit(
                f"REFUSED: cached immutable reference MD5 mismatch for {destination}: expected={expected_md5}, actual={actual}"
            )
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        raise SystemExit(f"reference download produced an empty file: {url}")
    actual = digest(temporary, "md5")
    if actual != expected_md5:
        temporary.unlink(missing_ok=True)
        raise SystemExit(
            f"REFUSED: immutable reference MD5 mismatch: expected={expected_md5}, actual={actual}"
        )
    temporary.replace(destination)
    return destination


def extract_laz_points(laz: Path, npz: Path) -> Path:
    npz.parent.mkdir(parents=True, exist_ok=True)
    helper = SCIENCE_ROOT / ".read_laz_reference.py"
    helper.write_text(
        textwrap.dedent(
            """
            import sys
            from pathlib import Path
            import laspy
            import numpy as np

            src = Path(sys.argv[1])
            dst = Path(sys.argv[2])
            cloud = laspy.read(src)
            x = np.asarray(cloud.x, dtype=np.float64)
            y = np.asarray(cloud.y, dtype=np.float64)
            z = np.asarray(cloud.z, dtype=np.float64)
            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
            if int(valid.sum()) < 128:
                raise SystemExit(f"insufficient finite LiDAR points: {int(valid.sum())}")
            np.savez_compressed(dst, x=x[valid], y=y[valid], z=z[valid])
            """
        ).lstrip(),
        encoding="utf-8",
    )
    # Evaluation-only dependency isolation. This does not alter the frozen production environment.
    subprocess.run(
        [
            "uv",
            "run",
            "--no-project",
            "--with",
            "laspy[lazrs]==2.6.1",
            "--with",
            "numpy>=1.26,<3",
            "python",
            str(helper),
            str(laz),
            str(npz),
        ],
        cwd=ROOT,
        check=True,
    )
    return npz


def rasterize_surface(npz: Path, output: Path, *, epsg: int, resolution_m: float) -> Path:
    payload = np.load(npz)
    x = np.asarray(payload["x"], dtype=np.float64)
    y = np.asarray(payload["y"], dtype=np.float64)
    z = np.asarray(payload["z"], dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x = x[finite]
    y = y[finite]
    z = z[finite]
    if x.size < 128:
        raise SystemExit(f"insufficient LiDAR points for DSM rasterization: {x.size}")

    left = float(np.floor(np.min(x) / resolution_m) * resolution_m)
    right = float(np.ceil(np.max(x) / resolution_m) * resolution_m)
    bottom = float(np.floor(np.min(y) / resolution_m) * resolution_m)
    top = float(np.ceil(np.max(y) / resolution_m) * resolution_m)
    width = max(1, int(round((right - left) / resolution_m)))
    height = max(1, int(round((top - bottom) / resolution_m)))
    transform = Affine(resolution_m, 0.0, left, 0.0, -resolution_m, top)

    cols = np.floor((x - left) / resolution_m).astype(np.int64)
    rows = np.floor((top - y) / resolution_m).astype(np.int64)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[inside]
    cols = cols[inside]
    z = z[inside]
    flat = rows * width + cols
    order = np.argsort(flat, kind="stable")
    flat_sorted = flat[order]
    z_sorted = z[order]
    unique, starts = np.unique(flat_sorted, return_index=True)
    maxima = np.maximum.reduceat(z_sorted, starts)

    nodata = np.float32(-9999.0)
    surface = np.full(height * width, nodata, dtype=np.float32)
    surface[unique] = maxima.astype(np.float32)
    surface = surface.reshape(height, width)
    if int(np.count_nonzero(surface != nodata)) < 128:
        raise SystemExit("rasterized LiDAR DSM has fewer than 128 occupied cells")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.tif")
    with rasterio.open(
        temporary,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        crs=CRS.from_epsg(epsg),
        transform=transform,
        nodata=float(nodata),
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(surface, 1)
        dst.set_band_description(1, "Independent LiDAR maximum-surface DSM reference")
        dst.update_tags(
            DEPTHWIZARD_ROLE="FINAL_SCIENCE_EVALUATION_ONLY_REFERENCE",
            SOURCE="Zenodo 5593238 NeonTreeEvaluation LiDAR",
            RASTERIZATION="1m grid; maximum observed LiDAR Z per occupied cell",
            CALIBRATION_USE="prohibited",
        )
    temporary.replace(output)
    return output


def locate_potsdam_reference() -> Path:
    roots = []
    env_root = os.environ.get("DEPTHWIZARD_POTSDAM_ROOT")
    if env_root:
        roots.append(Path(env_root))
    roots.append(ROOT / "data" / "external" / "isprs-potsdam")
    names = {
        "dsm_potsdam_04_12.tif",
        "dsm_potsdam_04_12.tiff",
        "dsm_potsdam_4_12.tif",
        "dsm_potsdam_4_12.tiff",
    }
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
            "Stage 2 needs exactly one official Potsdam DSM tile 4_12 under data/external/isprs-potsdam (or DEPTHWIZARD_POTSDAM_ROOT). "
            f"Found {len(unique)}. Do not substitute a calibration DEM or nDSM."
        )
    return unique[0]


def prepare_potsdam_reference(source: Path) -> Path:
    output = REFERENCE_ROOT / "urban-potsdam-4_12-dsm.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source) as src:
        if src.count < 1:
            raise SystemExit("Potsdam 4_12 DSM has no elevation band")
        if src.transform.is_identity:
            raise SystemExit(
                "Potsdam 4_12 DSM has no affine transform; keep its official .tfw beside the TIFF"
            )
        data = src.read(1)
        profile = src.profile.copy()
        profile.update(
            driver="GTiff",
            count=1,
            crs=CRS.from_epsg(32633),
            transform=src.transform,
            compress="deflate",
            tiled=True,
        )
    temporary = output.with_suffix(".tmp.tif")
    with rasterio.open(temporary, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "Independent ISPRS Potsdam DSM evaluation reference")
        dst.update_tags(
            DEPTHWIZARD_ROLE="FINAL_SCIENCE_EVALUATION_ONLY_REFERENCE",
            SOURCE_DATASET="ISPRS Potsdam",
            SOURCE_TILE="4_12",
            CALIBRATION_USE="prohibited",
        )
    temporary.replace(output)
    return output


def materialize_references() -> dict[str, str]:
    references: dict[str, str] = {}
    potsdam = prepare_potsdam_reference(locate_potsdam_reference())
    references["urban-potsdam-4_12"] = str(potsdam)
    for key, spec in NATURAL_REFERENCES.items():
        laz = download(spec["url"], spec["laz"], spec["md5"])
        npz = SCIENCE_ROOT / "reference-point-cache" / f"{key}.npz"
        extract_laz_points(laz, npz)
        dsm = rasterize_surface(
            npz,
            spec["out"],
            epsg=int(spec["epsg"]),
            resolution_m=float(spec["resolution_m"]),
        )
        references[key] = str(dsm)
    return references


def run_final_evaluator() -> dict[str, object]:
    python = ROOT / ".venv" / "bin" / "python"
    subprocess.run(
        [
            str(python),
            str(ROOT / "scripts" / "evaluate_final_science_campaign.py"),
            str(REGISTRY),
            str(FROZEN_MANIFEST),
            str(SCIENCE_ROOT),
        ],
        cwd=ROOT,
        check=True,
    )
    report_path = SCIENCE_ROOT / "domain_generalization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    requirements = report.get("requirements", {})
    expected = {
        "required_test_terrains": ["urban", "sparse", "hilly", "forested"],
        "test_terrain_coverage": ["forested", "hilly", "sparse", "urban"],
    }
    if report.get("protocol") != "depthwizard_final_science_campaign_v1":
        raise SystemExit("final science evaluator produced the wrong protocol")
    if requirements.get("required_test_terrains") != expected["required_test_terrains"]:
        raise SystemExit("final science report lost required terrain contract")
    if sorted(requirements.get("test_terrain_coverage", [])) != expected["test_terrain_coverage"]:
        raise SystemExit("final science report did not cover all four required terrains")
    for field in (
        "geographic_split_integrity",
        "cross_sensor_train_sensor_separation",
        "checkpoint_identity_frozen",
        "prediction_identity_freeze",
        "reference_independence",
    ):
        if requirements.get(field) != "passed":
            raise SystemExit(f"final science requirement did not pass: {field}")
    return report


def main() -> int:
    assert_frozen_tree()
    assert_reference_gate_open()
    print("REFERENCE_GATE OPEN: frozen prediction identities verified; evaluation references may now be materialized")
    references = materialize_references()
    report = run_final_evaluator()
    completion = subprocess.run(
        [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "scripts" / "check_sih26175_completion.py")],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    out = SCIENCE_ROOT / "stage2-reference-evaluation-report.json"
    payload = {
        "status": "PASS_FINAL_SCIENCE_STAGE2_REFERENCE_EVALUATION",
        "git_head": PRODUCTION_HEAD,
        "references": references,
        "domain_generalization_report": str(SCIENCE_ROOT / "domain_generalization_report.json"),
        "test_overall": report.get("test_overall"),
        "cross_sensor_overall": report.get("cross_sensor_overall"),
        "terrain": report.get("terrain"),
        "completion_checker_stdout": completion.stdout,
        "completion_checker_returncode": completion.returncode,
        "claim_boundary": (
            "Reference materialization/evaluation occurred only after exact-head checkpoint and prediction identities were frozen. "
            "Natural LiDAR references are maximum-surface rasters from the immutable Zenodo 5593238 point clouds; they were never calibration inputs."
        ),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
