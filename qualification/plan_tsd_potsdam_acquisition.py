from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.height_model.terrain_structure_split import (
    INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    initial_tsd_campaign_split,
)


@dataclass(frozen=True)
class RequiredFile:
    tile_id: str
    role: str
    component: str
    expected_filename: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan the exact missing local inputs for the predeclared first Potsdam TSD campaign. "
            "The planner consumes only the filename-metadata inventory JSON and never opens rasters."
        )
    )
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _expected_filename(tile_id: str, component: str) -> str:
    row_text, col_text = tile_id.split("_")
    row = int(row_text)
    col = int(col_text)
    if component == "rgb":
        return f"top_potsdam_{row}_{col}_RGB.tif"
    if component == "dsm":
        return f"dsm_potsdam_{row:02d}_{col:02d}.tif"
    if component == "label":
        return f"top_potsdam_{row}_{col}_label.tif"
    raise ValueError(f"unsupported component: {component}")


def _load_inventory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("TSD inventory must be a JSON object")
    if payload.get("protocol_version") != TSD_SPLIT_PROTOCOL_VERSION:
        raise ValueError(
            "inventory protocol mismatch: "
            f"expected {TSD_SPLIT_PROTOCOL_VERSION}, got {payload.get('protocol_version')!r}"
        )
    if payload.get("mode") != "filename_metadata_only":
        raise ValueError("acquisition planning requires filename_metadata_only inventory evidence")
    eligible = payload.get("supervision_eligible_tile_ids")
    if not isinstance(eligible, list) or set(eligible) != set(SUPERVISION_ELIGIBLE_TILE_IDS):
        raise ValueError("inventory legal supervision population does not match current split protocol")
    return payload


def _tile_records(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_tiles = payload.get("tiles")
    if not isinstance(raw_tiles, list):
        raise ValueError("inventory is missing tile records")
    records: dict[str, dict[str, Any]] = {}
    for raw in raw_tiles:
        if not isinstance(raw, dict) or not isinstance(raw.get("tile_id"), str):
            raise ValueError("inventory contains malformed tile record")
        tile_id = raw["tile_id"]
        if tile_id in records:
            raise ValueError(f"inventory contains duplicate tile record: {tile_id}")
        records[tile_id] = raw
    if set(records) != set(SUPERVISION_ELIGIBLE_TILE_IDS):
        raise ValueError("inventory tile records do not exactly cover the legal supervision population")
    return records


def _count(record: dict[str, Any], component: str) -> int:
    value = record.get(f"{component}_count")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid {component}_count for tile {record.get('tile_id')}")
    return value


def _role(tile_id: str) -> str:
    if tile_id in INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS:
        return "train"
    if tile_id in INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS:
        return "dev"
    raise ValueError(f"tile {tile_id} is not active in the initial TSD campaign")


def build_acquisition_plan(payload: dict[str, Any]) -> dict[str, Any]:
    split = initial_tsd_campaign_split()
    records = _tile_records(payload)
    active_tiles = split.supervised_tile_ids

    required_files: list[RequiredFile] = []
    ready: list[str] = []
    partial: list[str] = []
    absent: list[str] = []
    ambiguous: list[str] = []

    for tile_id in active_tiles:
        record = records[tile_id]
        counts = {component: _count(record, component) for component in ("rgb", "dsm", "label")}
        ambiguous_components = [name for name, count in counts.items() if count > 1]
        if ambiguous_components:
            ambiguous.append(tile_id)
            continue

        present_count = sum(count == 1 for count in counts.values())
        if present_count == 3:
            ready.append(tile_id)
        elif present_count == 0:
            absent.append(tile_id)
        else:
            partial.append(tile_id)

        for component, count in counts.items():
            if count == 0:
                required_files.append(
                    RequiredFile(
                        tile_id=tile_id,
                        role=_role(tile_id),
                        component=component,
                        expected_filename=_expected_filename(tile_id, component),
                    )
                )

    if ambiguous:
        raise RuntimeError(
            "refusing acquisition plan while active campaign tiles have ambiguous duplicate inputs: "
            + ",".join(ambiguous)
        )

    missing_by_component = {
        component: [item.tile_id for item in required_files if item.component == component]
        for component in ("rgb", "dsm", "label")
    }
    return {
        "schema_version": 1,
        "split_protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "campaign_protocol_version": INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
        "source_inventory_dataset_root": payload.get("dataset_root"),
        "train_tile_ids": list(INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS),
        "dev_tile_ids": list(INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS),
        "buffer_tile_ids_not_required": sorted(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS),
        "active_tile_count": len(active_tiles),
        "ready_active_tile_ids": ready,
        "partial_active_tile_ids": partial,
        "absent_active_tile_ids": absent,
        "missing_rgb_tile_ids": missing_by_component["rgb"],
        "missing_dsm_tile_ids": missing_by_component["dsm"],
        "missing_label_tile_ids": missing_by_component["label"],
        "missing_rgb_count": len(missing_by_component["rgb"]),
        "missing_dsm_count": len(missing_by_component["dsm"]),
        "missing_label_count": len(missing_by_component["label"]),
        "required_files": [asdict(item) for item in required_files],
        "claim_boundary": (
            "This plan is derived only from the v2 filename-metadata inventory and a split fixed from "
            "tile coordinates before new raster content is acquired. Buffer, reserved, exposed, "
            "external-evaluation, and historical challenge-test tiles are not required by this "
            "campaign. No raster content is opened by this planner."
        ),
    }


def main() -> int:
    args = parse_args()
    payload = _load_inventory(args.inventory)
    plan = build_acquisition_plan(payload)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(args.output)
        print(f"acquisition_plan={args.output}")

    print(f"campaign_protocol_version={plan['campaign_protocol_version']}")
    print("train_tile_ids=" + ",".join(plan["train_tile_ids"]))
    print("dev_tile_ids=" + ",".join(plan["dev_tile_ids"]))
    print("buffer_tile_ids_not_required=" + ",".join(plan["buffer_tile_ids_not_required"]))
    print(f"active_tile_count={plan['active_tile_count']}")
    print("ready_active_tile_ids=" + ",".join(plan["ready_active_tile_ids"]))
    print("partial_active_tile_ids=" + ",".join(plan["partial_active_tile_ids"]))
    print("absent_active_tile_ids=" + ",".join(plan["absent_active_tile_ids"]))
    for component in ("rgb", "dsm", "label"):
        print(f"missing_{component}_count={plan[f'missing_{component}_count']}")
        print(f"missing_{component}_tile_ids=" + ",".join(plan[f"missing_{component}_tile_ids"]))
    print("raster_content_opened=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
