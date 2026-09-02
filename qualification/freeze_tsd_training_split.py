from __future__ import annotations

import argparse
import json
import subprocess
import sys
import warnings
from pathlib import Path

import rasterio
from rasterio.errors import NotGeoreferencedWarning

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
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    PARTICIPANT_GROUND_TRUTH_TILE_IDS,
    SEALED_BLIND_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    TerrainStructureDataSplit,
    freeze_tsd_data_split,
    initial_tsd_campaign_split,
)
from depthwizard.provenance.manifest import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze an auditable Potsdam supervision split for Terrain-Structure Decomposition. "
            "Only historical participant ground-truth tiles are legal, with DepthWizard reserved "
            "tiles removed before any dataset traversal."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        choices=(INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,),
        help="Freeze the predeclared spatially separated first urban TSD campaign.",
    )
    parser.add_argument("--train-tile", action="append", dest="train_tiles")
    parser.add_argument("--dev-tile", action="append", dest="dev_tiles")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.campaign is not None:
        if args.train_tiles or args.dev_tiles:
            parser.error("--campaign cannot be combined with explicit --train-tile/--dev-tile")
    elif not args.train_tiles or not args.dev_tiles:
        parser.error("provide --campaign or both --train-tile and --dev-tile")
    return args


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


def _unique_tfw(raster_path: Path) -> Path:
    matches = sorted(
        path
        for path in raster_path.parent.iterdir()
        if path.is_file()
        and path.stem.casefold() == raster_path.stem.casefold()
        and path.suffix.casefold() == ".tfw"
    )
    if not matches:
        raise FileNotFoundError(
            f"missing mandatory Potsdam .tfw georeference sidecar for {raster_path}"
        )
    if len(matches) != 1:
        raise RuntimeError(
            f"ambiguous Potsdam .tfw sidecars for {raster_path}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _semantic_metadata(path: Path, rgb_path: Path) -> dict[str, object]:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(path) as src, rasterio.open(rgb_path) as rgb:
            if src.count != 3:
                raise ValueError(f"Potsdam semantic label must have exactly 3 bands: {path}")
            if (src.height, src.width) != (6000, 6000):
                raise ValueError(
                    f"Potsdam semantic label must be 6000x6000; got {src.width}x{src.height}: {path}"
                )
            if (src.height, src.width) != (rgb.height, rgb.width):
                raise ValueError(f"semantic-label dimensions disagree with RGB: {path}")

            if src.transform.is_identity:
                alignment_basis = "official_same_tile_exact_pixel_grid"
            else:
                if not src.transform.almost_equals(rgb.transform):
                    raise ValueError(f"semantic-label affine transform disagrees with RGB: {path}")
                if src.crs is not None and rgb.crs is not None and src.crs != rgb.crs:
                    raise ValueError(f"semantic-label CRS disagrees with RGB: {path}")
                alignment_basis = "geospatial_transform_match"

            return {
                "width": src.width,
                "height": src.height,
                "bands": src.count,
                "dtype": src.dtypes[0],
                "crs": src.crs.to_string() if src.crs is not None else None,
                "transform_is_identity": bool(src.transform.is_identity),
                "alignment_basis": alignment_basis,
            }


def _tile_record(dataset_root: Path, tile_id: str, role: str) -> dict[str, object]:
    paths = resolve_potsdam_tile_paths(dataset_root, tile_id)
    label = _unique_semantic_label(dataset_root, tile_id)
    rgb_world_file = _unique_tfw(paths.rgb)
    dsm_world_file = _unique_tfw(paths.reference_dsm)
    rgb_contract = inspect_potsdam_rgb_contract(paths.rgb)
    reference_contract = inspect_potsdam_reference_contract(paths.rgb, paths.reference_dsm)
    label_metadata = _semantic_metadata(label, paths.rgb)
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
        "rgb_world_file": str(rgb_world_file.resolve()),
        "rgb_world_file_sha256": sha256_file(rgb_world_file),
        "reference_dsm": str(paths.reference_dsm.resolve()),
        "reference_dsm_sha256": sha256_file(paths.reference_dsm),
        "reference_dsm_world_file": str(dsm_world_file.resolve()),
        "reference_dsm_world_file_sha256": sha256_file(dsm_world_file),
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
    campaign_protocol_version: str | None,
) -> None:
    payload = {
        "schema_version": 4,
        "protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "campaign_protocol_version": campaign_protocol_version,
        "status": "FROZEN_TSD_SUPERVISION_SPLIT",
        "qualification_git_sha": source_sha,
        "qualification_git_branch": source_branch,
        "dataset_root": str(dataset_root.resolve()),
        "claim_boundary": (
            "Only listed training/development tiles may contribute reference DSM or semantic "
            "supervision. The legal population is the historical Potsdam participant ground-truth "
            "set minus DepthWizard reserved evidence. Historical challenge-test tiles remain "
            "prohibited even if later label packages are locally available. Exposed 2_14, external "
            "evaluation 3_14, and sealed blind 4_12/6_12 are prohibited from training, development "
            "loss, early stopping, hyperparameter selection, or target generation. For the named "
            "initial campaign, the recorded spatial-buffer tiles are also withheld from supervision. "
            "RGB/DSM TIFF identities and their mandatory .tfw georeference sidecars are hashed "
            "independently so a later world-file edit cannot silently change the frozen grid."
        ),
        "participant_ground_truth_tile_ids": sorted(PARTICIPANT_GROUND_TRUTH_TILE_IDS),
        "supervision_eligible_tile_ids": sorted(SUPERVISION_ELIGIBLE_TILE_IDS),
        "historical_challenge_test_tile_ids_prohibited": sorted(
            HISTORICAL_CHALLENGE_TEST_TILE_IDS
        ),
        "train_tile_ids": list(split.train_tile_ids),
        "dev_tile_ids": list(split.dev_tile_ids),
        "campaign_buffer_tile_ids_withheld": (
            sorted(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS)
            if campaign_protocol_version == INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION
            else []
        ),
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

    # Validate every requested id before checking dataset-root existence or traversing it. Reserved,
    # historical challenge-test, or arbitrary nonparticipant ids therefore fail without resolving
    # any of their filesystem entries.
    if args.campaign == INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION:
        split = initial_tsd_campaign_split()
        campaign_protocol_version: str | None = INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION
    else:
        split = freeze_tsd_data_split(args.train_tiles, args.dev_tiles)
        campaign_protocol_version = None

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
        campaign_protocol_version=campaign_protocol_version,
    )
    print(f"split_manifest={args.output}")
    print(f"split_manifest_sha256={sha256_file(args.output)}")
    print(f"qualification_git_sha={source_sha}")
    print(f"protocol_version={TSD_SPLIT_PROTOCOL_VERSION}")
    print(f"campaign_protocol_version={campaign_protocol_version or 'custom'}")
    print(f"train_tiles={','.join(split.train_tile_ids)}")
    print(f"dev_tiles={','.join(split.dev_tile_ids)}")
    if campaign_protocol_version == INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION:
        print(
            "campaign_buffer_tiles_withheld="
            + ",".join(sorted(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS))
        )
    print("world_file_hashes_frozen=true")
    print("reserved_tiles_not_consumed=2_14,3_14,4_12,6_12")
    print("historical_challenge_test_tiles_not_consumed=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
