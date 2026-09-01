from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.height_model.terrain_structure_split import RESERVED_TILE_IDS

_RGB_PATTERN = re.compile(r"^top_potsdam_(\d+)_(\d+)_rgb\.tiff?$", re.IGNORECASE)
_DSM_PATTERN = re.compile(r"^dsm_potsdam_(\d+)_(\d+)\.tiff?$", re.IGNORECASE)
_LABEL_PATTERN = re.compile(r"^top_potsdam_(\d+)_(\d+)_label\.tiff?$", re.IGNORECASE)


@dataclass(frozen=True)
class TileInventory:
    tile_id: str
    rgb_count: int
    dsm_count: int
    label_count: int
    status: str

    @property
    def complete(self) -> bool:
        return self.status == "COMPLETE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory filename metadata for non-reserved Potsdam TSD supervision candidates. "
            "The command never opens raster content and discards reserved tile ids before reporting."
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


def _tile_id_from_name(name: str) -> str | None:
    for pattern in (_RGB_PATTERN, _DSM_PATTERN, _LABEL_PATTERN):
        match = pattern.fullmatch(name)
        if match is not None:
            return f"{int(match.group(1))}_{int(match.group(2))}"
    return None


def _match_count(index: dict[str, list[Path]], names: tuple[str, ...]) -> int:
    matches: set[Path] = set()
    for name in names:
        matches.update(index.get(name.casefold(), []))
    return len(matches)


def _component_counts(index: dict[str, list[Path]], tile_id: str) -> tuple[int, int, int]:
    row, col = (int(part) for part in tile_id.split("_"))
    rgb_count = _match_count(
        index,
        (
            f"top_potsdam_{row}_{col}_RGB.tif",
            f"top_potsdam_{row}_{col}_RGB.tiff",
        ),
    )
    dsm_count = _match_count(
        index,
        (
            f"dsm_potsdam_{row:02d}_{col:02d}.tif",
            f"dsm_potsdam_{row:02d}_{col:02d}.tiff",
            f"dsm_potsdam_{row}_{col}.tif",
            f"dsm_potsdam_{row}_{col}.tiff",
        ),
    )
    label_count = _match_count(
        index,
        (
            f"top_potsdam_{row}_{col}_label.tif",
            f"top_potsdam_{row}_{col}_label.tiff",
        ),
    )
    return rgb_count, dsm_count, label_count


def _status(rgb_count: int, dsm_count: int, label_count: int) -> str:
    counts = {"RGB": rgb_count, "DSM": dsm_count, "LABEL": label_count}
    ambiguous = [name for name, count in counts.items() if count > 1]
    if ambiguous:
        return "AMBIGUOUS_" + "_".join(ambiguous)
    missing = [name for name, count in counts.items() if count == 0]
    if missing:
        return "MISSING_" + "_".join(missing)
    return "COMPLETE"


def _safe_tile_ids(index: dict[str, list[Path]]) -> tuple[str, ...]:
    discovered: set[str] = set()
    for paths in index.values():
        for path in paths:
            tile_id = _tile_id_from_name(path.name)
            if tile_id is not None and tile_id not in RESERVED_TILE_IDS:
                discovered.add(tile_id)
    return tuple(
        sorted(discovered, key=lambda value: tuple(int(part) for part in value.split("_")))
    )


def _inventory(index: dict[str, list[Path]]) -> tuple[TileInventory, ...]:
    records: list[TileInventory] = []
    for tile_id in _safe_tile_ids(index):
        rgb_count, dsm_count, label_count = _component_counts(index, tile_id)
        records.append(
            TileInventory(
                tile_id=tile_id,
                rgb_count=rgb_count,
                dsm_count=dsm_count,
                label_count=label_count,
                status=_status(rgb_count, dsm_count, label_count),
            )
        )
    return tuple(records)


def main() -> int:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)

    index = _filename_index(args.dataset_root)
    records = _inventory(index)
    candidates = tuple(record.tile_id for record in records if record.complete)
    incomplete = tuple(record.tile_id for record in records if not record.complete)

    payload = {
        "schema_version": 2,
        "mode": "filename_metadata_only",
        "dataset_root": str(args.dataset_root.resolve()),
        "safe_discovered_tile_count": len(records),
        "candidate_supervision_tile_ids": list(candidates),
        "candidate_count": len(candidates),
        "incomplete_safe_tile_ids": list(incomplete),
        "tiles": [asdict(record) for record in records],
        "reserved_tile_ids_omitted": sorted(RESERVED_TILE_IDS),
        "claim_boundary": (
            "Inventory is based on filenames only. Raster pixels were not opened, decoded, hashed, "
            "or used for model selection. Reserved exposed/evaluation/blind ids are discarded before "
            "tile diagnostics are constructed. Counts greater than one fail as ambiguous rather than "
            "silently selecting a file."
        ),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(args.output)
        print(f"inventory={args.output}")

    print(f"safe_discovered_tile_count={len(records)}")
    print(f"candidate_count={len(candidates)}")
    print("candidate_supervision_tile_ids=" + ",".join(candidates))
    print("incomplete_safe_tile_ids=" + ",".join(incomplete))
    for record in records:
        print(
            f"tile={record.tile_id} rgb={record.rgb_count} dsm={record.dsm_count} "
            f"label={record.label_count} status={record.status}"
        )
    print("reserved_tile_ids_omitted=" + ",".join(sorted(RESERVED_TILE_IDS)))
    print("raster_content_opened=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
