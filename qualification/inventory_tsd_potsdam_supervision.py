from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.height_model.terrain_structure_split import RESERVED_TILE_IDS

_LABEL_PATTERN = re.compile(r"^top_potsdam_(\d+)_(\d+)_label\.tiff?$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory non-reserved Potsdam tiles that have RGB, reference DSM, and official semantic "
            "label filenames. This command reads filename metadata only and never opens raster content."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _filename_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def _has_unique(index: dict[str, list[Path]], names: tuple[str, ...]) -> bool:
    matches: set[Path] = set()
    for name in names:
        matches.update(index.get(name.casefold(), []))
    return len(matches) == 1


def _candidate_tile_ids(index: dict[str, list[Path]]) -> tuple[str, ...]:
    label_tiles: set[str] = set()
    for paths in index.values():
        for path in paths:
            match = _LABEL_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            tile_id = f"{int(match.group(1))}_{int(match.group(2))}"
            # Reserved ids are discarded from filename metadata before any candidate resolution.
            if tile_id not in RESERVED_TILE_IDS:
                label_tiles.add(tile_id)

    complete: list[str] = []
    for tile_id in sorted(label_tiles, key=lambda value: tuple(int(part) for part in value.split("_"))):
        row, col = (int(part) for part in tile_id.split("_"))
        has_rgb = _has_unique(
            index,
            (
                f"top_potsdam_{row}_{col}_RGB.tif",
                f"top_potsdam_{row}_{col}_RGB.tiff",
            ),
        )
        has_dsm = _has_unique(
            index,
            (
                f"dsm_potsdam_{row:02d}_{col:02d}.tif",
                f"dsm_potsdam_{row:02d}_{col:02d}.tiff",
                f"dsm_potsdam_{row}_{col}.tif",
                f"dsm_potsdam_{row}_{col}.tiff",
            ),
        )
        if has_rgb and has_dsm:
            complete.append(tile_id)
    return tuple(complete)


def main() -> int:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    index = _filename_index(args.dataset_root)
    candidates = _candidate_tile_ids(index)
    payload = {
        "schema_version": 1,
        "mode": "filename_metadata_only",
        "dataset_root": str(args.dataset_root.resolve()),
        "candidate_supervision_tile_ids": list(candidates),
        "candidate_count": len(candidates),
        "reserved_tile_ids_omitted": sorted(RESERVED_TILE_IDS),
        "claim_boundary": (
            "Inventory is based on filenames only. Raster pixels were not opened, decoded, hashed, "
            "or used for model selection. Reserved exposed/evaluation/blind ids are omitted."
        ),
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(args.output)
        print(f"inventory={args.output}")
    print(f"candidate_count={len(candidates)}")
    print("candidate_supervision_tile_ids=" + ",".join(candidates))
    print("reserved_tile_ids_omitted=" + ",".join(sorted(RESERVED_TILE_IDS)))
    print("raster_content_opened=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
