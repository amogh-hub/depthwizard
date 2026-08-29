#!/usr/bin/env python3
"""Build the minimum frozen SIH26175 final-science registry without opening references.

Frozen scene policy:
- train lineage: first deterministic OrthoLoC training DOP (reference not required)
- urban test: first fresh valid Potsdam tile frozen by input preflight
- sparse test: NEON CPER deterministic co-acquired RGB/DSM tile
- hilly test: NEON NIWO deterministic co-acquired RGB/DSM tile
- forested test: NEON HARV deterministic co-acquired RGB/DSM tile
- cross-sensor holdout: second fresh valid Potsdam tile frozen by input preflight

The input preflight excludes all DepthWizard external-v2 consumed Potsdam tiles
(2_10, 3_13, 5_11, 6_14) and validates metadata without decoding reference heights.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

from depthwizard.data.ortholoc import discover_remote_scenes, download_file
from depthwizard.data.registry import DatasetRegistry, load_registry
from depthwizard.evaluation.potsdam import FROZEN_POTSDAM_TILE_IDS, resolve_potsdam_tile_paths

FROZEN_HEAD = "339bdf485149f552db846543b9e09377b567c19c"


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _require_frozen_head(root: Path) -> None:
    head = _git_head(root)
    if head != FROZEN_HEAD:
        raise RuntimeError(f"final-science registry must be frozen from {FROZEN_HEAD}; current={head}")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    if dirty:
        raise RuntimeError("repository must be clean before freezing the final-science registry")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _potsdam_selection(preflight: dict[str, Any]) -> tuple[str, str]:
    if preflight.get("status") != "PASS_FINAL_SCIENCE_INPUT_PREFLIGHT":
        raise RuntimeError(
            "final-science input preflight is not PASS; do not freeze a registry from incomplete inputs"
        )
    potsdam = preflight.get("potsdam")
    if not isinstance(potsdam, dict):
        raise RuntimeError("input preflight is missing Potsdam selection")
    urban = potsdam.get("urban_test_tile")
    cross = potsdam.get("cross_sensor_tile")
    if not isinstance(urban, str) or not isinstance(cross, str) or not urban or not cross:
        raise RuntimeError("input preflight did not freeze two Potsdam tile IDs")
    if urban == cross:
        raise RuntimeError("urban and cross-sensor Potsdam scenes must be distinct")
    consumed = set(FROZEN_POTSDAM_TILE_IDS)
    if urban in consumed or cross in consumed:
        raise RuntimeError("input preflight selected a consumed Potsdam tile")
    return urban, cross


def _neon_scene(acquisition: dict[str, Any], site: str, terrain: str) -> dict[str, Any]:
    scenes = acquisition.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("NEON acquisition report is missing scenes")
    match = next(
        (entry for entry in scenes if isinstance(entry, dict) and entry.get("site") == site),
        None,
    )
    if not isinstance(match, dict):
        raise RuntimeError(f"NEON acquisition report is missing {site}")
    if match.get("terrain") != terrain:
        raise RuntimeError(f"NEON terrain mismatch for {site}: {match.get('terrain')!r}")
    rgb = match.get("rgb")
    reference = match.get("reference")
    selected = match.get("selected_tile")
    if not isinstance(rgb, dict) or not isinstance(reference, dict) or not isinstance(selected, dict):
        raise RuntimeError(f"NEON acquisition report is incomplete for {site}")
    rgb_path = Path(str(rgb.get("path", ""))).resolve(strict=True)
    reference_path = Path(str(reference.get("path", ""))).resolve(strict=True)
    easting = selected.get("easting")
    northing = selected.get("northing")
    month = match.get("month")
    return {
        "scene_id": f"neon-{site.lower()}-{easting}-{northing}",
        "dataset": "NEON AOP",
        "split": "test",
        "terrain": terrain,
        "geographic_group": f"neon-{site.lower()}-{easting}-{northing}",
        "sensor": "NEON AOP Optech Gemini high-resolution digital camera",
        "rgb_path": str(rgb_path),
        "reference_path": str(reference_path),
        "nominal_gsd_m": 0.1,
        "license_id": "NEON Data Use and Policy",
        "notes": (
            f"{site} acquisition {month}; RGB DP3.30010.001 and LiDAR DSM DP3.30024.001. "
            "Reference DSM is evaluation-only and must remain unopened/unhashed until prediction freeze."
        ),
    }


def _ortholoc_train_scene(destination_root: Path) -> dict[str, Any]:
    remote = discover_remote_scenes("train")
    if not remote:
        raise RuntimeError("OrthoLoC train discovery returned no scenes")
    selected = sorted(remote, key=lambda scene: (scene.location_id, scene.filename))[0]
    path = download_file(selected.dop_url, destination_root / selected.filename)
    return {
        "scene_id": f"ortholoc-train-{selected.scene_id.lower()}",
        "dataset": "OrthoLoC",
        "split": "train",
        "terrain": "urban",
        "geographic_group": f"ortholoc-{selected.location_id.lower()}-train-lineage",
        "sensor": "OrthoLoC aerial orthophoto development lineage",
        "rgb_path": str(path.resolve()),
        "reference_path": None,
        "nominal_gsd_m": None,
        "license_id": "OrthoLoC official dataset terms",
        "notes": (
            "Training-lineage identity only. This scene is not evaluated by the final campaign and its "
            "DSM is intentionally not downloaded here."
        ),
    }


def _potsdam_scene(root: Path, tile_id: str, *, split: str, role: str) -> dict[str, Any]:
    if tile_id in set(FROZEN_POTSDAM_TILE_IDS):
        raise RuntimeError(f"refusing consumed DepthWizard Potsdam tile {tile_id}")
    paths = resolve_potsdam_tile_paths(root, tile_id)
    return {
        "scene_id": f"potsdam-{tile_id.replace('_', '-')}-{role}",
        "dataset": "ISPRS Potsdam",
        "split": split,
        "terrain": "urban",
        "geographic_group": f"potsdam-{tile_id}",
        "sensor": "BSF Swissphoto Potsdam airborne true orthophoto",
        "rgb_path": str(paths.rgb.resolve()),
        "reference_path": str(paths.reference_dsm.resolve()),
        "nominal_gsd_m": 0.05,
        "license_id": "ISPRS Potsdam benchmark terms",
        "notes": (
            f"Preflight-frozen unused tile {tile_id}; official RGB input and absolute DSM reference "
            "on the same UTM grid. Reference DSM is evaluation-only and must remain unopened/unhashed "
            "until prediction freeze."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the frozen DepthWizard final-science registry.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--neon-report",
        type=Path,
        default=Path("workspace/final-science-data/neon-acquisition.json"),
    )
    parser.add_argument(
        "--input-preflight",
        type=Path,
        default=Path("workspace/final-science-data/input-preflight.json"),
    )
    parser.add_argument(
        "--potsdam-root",
        type=Path,
        default=Path("data/external/isprs-potsdam"),
    )
    parser.add_argument(
        "--ortholoc-train-root",
        type=Path,
        default=Path("workspace/final-science-data/train-lineage/ortholoc"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workspace/final-science-data/frozen-registry.yaml"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("workspace/final-science-data/registry-freeze-report.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    repo = args.repo.resolve(strict=True)
    _require_frozen_head(repo)
    acquisition = _read_json((repo / args.neon_report).resolve(strict=True))
    if acquisition.get("reference_values_opened_or_hashed") is not False:
        raise RuntimeError("NEON acquisition report does not preserve the sealed-reference boundary")
    preflight = _read_json((repo / args.input_preflight).resolve(strict=True))
    urban_test_tile, cross_sensor_tile = _potsdam_selection(preflight)
    potsdam_root = (repo / args.potsdam_root).resolve(strict=True)
    ortholoc_root = (repo / args.ortholoc_train_root).resolve(strict=False)

    scenes = [
        _ortholoc_train_scene(ortholoc_root),
        _potsdam_scene(potsdam_root, urban_test_tile, split="test", role="urban-test"),
        _neon_scene(acquisition, "CPER", "sparse"),
        _neon_scene(acquisition, "NIWO", "hilly"),
        _neon_scene(acquisition, "HARV", "forested"),
        _potsdam_scene(
            potsdam_root,
            cross_sensor_tile,
            split="cross_sensor_test",
            role="cross-sensor",
        ),
    ]
    registry_payload = {"schema_version": 1, "scenes": scenes}
    output = (repo / args.output).resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(registry_payload, sort_keys=False), encoding="utf-8")

    registry: DatasetRegistry = load_registry(output)
    registry.assert_integrity(require_files=True)
    if set(registry.terrain_coverage().get("test", set())) != {"urban", "sparse", "hilly", "forested"}:
        raise RuntimeError("final test split does not contain exactly the required four terrain classes")

    report = {
        "schema_version": 2,
        "status": "PASS_FINAL_SCIENCE_REGISTRY_FREEZE",
        "git_head": FROZEN_HEAD,
        "registry": str(output),
        "scene_count": len(registry.scenes),
        "test_terrain_coverage": sorted(registry.terrain_coverage().get("test", set())),
        "cross_sensor_scene_count": sum(1 for scene in registry.scenes if scene.split == "cross_sensor_test"),
        "consumed_potsdam_tiles_excluded": list(FROZEN_POTSDAM_TILE_IDS),
        "frozen_potsdam_test_tile": urban_test_tile,
        "frozen_potsdam_cross_sensor_tile": cross_sensor_tile,
        "input_preflight": str((repo / args.input_preflight).resolve(strict=True)),
        "reference_rasters_opened_or_hashed": False,
    }
    report_path = (repo / args.report).resolve(strict=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
