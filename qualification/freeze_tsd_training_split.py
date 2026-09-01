from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import rasterio

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.potsdam import (
    inspect_potsdam_reference_contract,
    inspect_potsdam_rgb_contract,
    resolve_potsdam_tile_paths,
)
from depthwizard.height_model.terrain_structure_split import (
    EXPOSED_CORRECTIVE_TILE_IDS,
    EXTERNAL_EVALUATION_TILE_IDS,
    SEALED_BLIND_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    TerrainStructureDataSplit,
    freeze_tsd_data_split,
)
from depthwizard.provenance.manifest import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an auditable Potsdam supervision split for Terrain-Structure Decomposition. "
            "Reserved exposed/evaluation/blind tile ids are rejected before any dataset traversal."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--train-tile", action="append", required=True, dest="train_tiles")
    parser.add_argument("--dev-tile", action="append", required=True, dest="dev_tiles")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _tracked_source_identity() -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "refusing to freeze TSD split from a tracked-dirty worktree; commit or restore changes first"
        )
    if not head:
        raise RuntimeError("could not resolve qualification source commit")
    return head, branch


def _unique_semantic_label(dataset_root: Path, tile_id: str) -> Path:
    row_text, col_text = tile_id.split("_")
    expected = f"top_potsdam_{int(row_text)}_{int(col_text)}_label.tif".casefold()
    matches = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.name.casefold() == expected
    )
    if not matches:
        raise FileNotFoundError(
            f"missing official semantic label for Potsdam {tile_id}; expected {expected}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"ambiguous official semantic label for Potsdam {tile_id}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _semantic_metadata(path: Path) -> dict[str, object]:
    with rasterio.open(path) as src:
        if src.count != 3:
            raise ValueError(f"Potsdam semantic label must have exactly 3 bands: {path}")
        if (src.height, src.width) != (6000, 6000):
            raise ValueError(
                f"Potsdam semantic label must be 6000x6000; got {src.width}x{src.height}: {path}"
            )
        return {
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "dtype": src.dtypes[0],
            "crs": src.crs.to_string() if src.crs is not None else None,
            "transform_is_identity": bool(src.transform.is_identity),
        }


def _tile_record(dataset_root: Path, tile_id: str, role: str) -> dict[str, object]:
    paths = resolve_potsdam_tile_paths(dataset_root, tile_id)
    label = _unique_semantic_label(dataset_root, tile_id)
    rgb_contract = inspect_potsdam_rgb_contract(paths.rgb)
    reference_contract = inspect_potsdam_reference_contract(paths.rgb, paths.reference_dsm)
    label_metadata = _semantic_metadata(label)
    if (label_metadata["height"], label_metadata["width"]) != (
        rgb_contract["height"],
        rgb_contract["width"],
    ):
        raise ValueError(f"semantic-label dimensions disagree with RGB for Potsdam {tile_id}")
    return {
        "tile_id": tile_id,
        "role": role,
        "rgb": str(paths.rgb.resolve()),
        "rgb_sha256": sha256_file(paths.rgb),
        "reference_dsm": str(paths.reference_dsm.resolve()),
        "reference_dsm_sha256": sha256_file(paths.reference_dsm),
        "semantic_label": str(label.resolve()),
        "semantic_label_sha256": sha256_file(label),
        "rgb_contract": rgb_contract,
        "reference_contract": reference_contract,
        "semantic_label_metadata": label_metadata,
    }


def _write_manifest(
    output: Path,
    *,
    split: TerrainStructureDataSplit,
    dataset_root: Path,
    source_sha: str,
    source_branch: str,
    records: list[dict[str, object]],
) -> None:
    payload = {
        "schema_version": 1,
        "protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "status": "FROZEN_TSD_SUPERVISION_SPLIT",
        "qualification_git_sha": source_sha,
        "qualification_git_branch": source_branch,
        "dataset_root": str(dataset_root.resolve()),
        "claim_boundary": (
            "Only the listed training/development tiles may contribute reference DSM or semantic "
            "supervision to this TSD research campaign. Exposed 2_14, external evaluation 3_14, and "
            "sealed blind 4_12/6_12 are prohibited from training, development loss, early stopping, "
            "hyperparameter selection, or target generation."
        ),
        "train_tile_ids": list(split.train_tile_ids),
        "dev_tile_ids": list(split.dev_tile_ids),
        "reserved": {
            "exposed_corrective": list(EXPOSED_CORRECTIVE_TILE_IDS),
            "external_evaluation": list(EXTERNAL_EVALUATION_TILE_IDS),
            "sealed_blind": list(SEALED_BLIND_TILE_IDS),
        },
        "tiles": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)


def main() -> int:
    args = parse_args()

    # Deliberately validate the requested ids before checking dataset-root existence or traversing it.
    # A blind/reserved id therefore fails without touching that tile's filesystem entries.
    split = freeze_tsd_data_split(args.train_tiles, args.dev_tiles)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    source_sha, source_branch = _tracked_source_identity()

    records: list[dict[str, object]] = []
    for tile_id in split.train_tile_ids:
        records.append(_tile_record(args.dataset_root, tile_id, "train"))
    for tile_id in split.dev_tile_ids:
        records.append(_tile_record(args.dataset_root, tile_id, "dev"))

    _write_manifest(
        args.output,
        split=split,
        dataset_root=args.dataset_root,
        source_sha=source_sha,
        source_branch=source_branch,
        records=records,
    )
    print(f"split_manifest={args.output}")
    print(f"split_manifest_sha256={sha256_file(args.output)}")
    print(f"qualification_git_sha={source_sha}")
    print(f"train_tiles={','.join(split.train_tile_ids)}")
    print(f"dev_tiles={','.join(split.dev_tile_ids)}")
    print("reserved_tiles_not_consumed=2_14,3_14,4_12,6_12")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
