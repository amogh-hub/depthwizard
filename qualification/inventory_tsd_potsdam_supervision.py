from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.height_model.terrain_structure_split import (
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    RESERVED_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
)


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

    @property
    def locally_present(self) -> bool:
        return self.rgb_count > 0 or self.dsm_count > 0 or self.label_count > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory filename metadata for the legal Potsdam TSD supervision population. The "
            "command never opens raster content, never enumerates challenge-test scenes as candidates, "
            "and keeps DepthWizard's exposed/evaluation/blind tiles outside supervision."
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


def _ordered_eligible_tile_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            SUPERVISION_ELIGIBLE_TILE_IDS,
            key=lambda value: tuple(int(part) for part in value.split("_")),
        )
    )


def _inventory(index: dict[str, list[Path]]) -> tuple[TileInventory, ...]:
    records: list[TileInventory] = []
    for tile_id in _ordered_eligible_tile_ids():
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
    partial = tuple(record.tile_id for record in records if record.locally_present and not record.complete)
    absent = tuple(record.tile_id for record in records if not record.locally_present)
    locally_present = tuple(record for record in records if record.locally_present)

    payload = {
        "schema_version": 3,
        "mode": "filename_metadata_only",
        "protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "dataset_root": str(args.dataset_root.resolve()),
        "supervision_eligible_tile_ids": list(_ordered_eligible_tile_ids()),
        "supervision_eligible_tile_count": len(records),
        "locally_present_eligible_tile_count": len(locally_present),
        "candidate_supervision_tile_ids": list(candidates),
        "candidate_count": len(candidates),
        "partial_eligible_tile_ids": list(partial),
        "absent_eligible_tile_ids": list(absent),
        "tiles": [asdict(record) for record in records],
        "reserved_tile_ids_omitted": sorted(RESERVED_TILE_IDS),
        "historical_challenge_test_tile_ids_prohibited": sorted(HISTORICAL_CHALLENGE_TEST_TILE_IDS),
        "claim_boundary": (
            "Inventory is based on filenames only. Raster pixels were not opened, decoded, hashed, "
            "or used for model selection. Candidate construction is restricted a priori to the "
            "historical participant ground-truth population minus DepthWizard reserved tiles. "
            "Historical challenge-test scenes are never candidate supervision, even if later label "
            "packages are present locally. Counts greater than one fail as ambiguous rather than "
            "silently selecting a file."
        ),
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(args.output)
        print(f"inventory={args.output}")

    print(f"protocol_version={TSD_SPLIT_PROTOCOL_VERSION}")
    print(f"supervision_eligible_tile_count={len(records)}")
    print(f"locally_present_eligible_tile_count={len(locally_present)}")
    print(f"candidate_count={len(candidates)}")
    print("candidate_supervision_tile_ids=" + ",".join(candidates))
    print("partial_eligible_tile_ids=" + ",".join(partial))
    print("absent_eligible_tile_ids=" + ",".join(absent))
    for record in locally_present:
        print(
            f"tile={record.tile_id} rgb={record.rgb_count} dsm={record.dsm_count} "
            f"label={record.label_count} status={record.status}"
        )
    print("reserved_tile_ids_omitted=" + ",".join(sorted(RESERVED_TILE_IDS)))
    print(
        "historical_challenge_test_tile_ids_prohibited="
        + ",".join(sorted(HISTORICAL_CHALLENGE_TEST_TILE_IDS))
    )
    print("raster_content_opened=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
