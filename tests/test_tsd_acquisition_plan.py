from __future__ import annotations

import pytest

from depthwizard.height_model.terrain_structure_split import (
    INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
)
from qualification.plan_tsd_potsdam_acquisition import build_acquisition_plan


def _inventory_payload() -> dict[str, object]:
    records = [
        {
            "tile_id": tile_id,
            "rgb_count": 0,
            "dsm_count": 0,
            "label_count": 0,
            "status": "MISSING_RGB_DSM_LABEL",
        }
        for tile_id in sorted(SUPERVISION_ELIGIBLE_TILE_IDS)
    ]
    return {
        "schema_version": 3,
        "mode": "filename_metadata_only",
        "protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "dataset_root": "/dataset",
        "supervision_eligible_tile_ids": sorted(SUPERVISION_ELIGIBLE_TILE_IDS),
        "tiles": records,
    }


def _record(payload: dict[str, object], tile_id: str) -> dict[str, object]:
    tiles = payload["tiles"]
    assert isinstance(tiles, list)
    for record in tiles:
        assert isinstance(record, dict)
        if record["tile_id"] == tile_id:
            return record
    raise AssertionError(tile_id)


def test_acquisition_plan_matches_observed_two_partial_local_tiles() -> None:
    payload = _inventory_payload()
    for tile_id in ("2_10", "5_11"):
        record = _record(payload, tile_id)
        record["rgb_count"] = 1
        record["dsm_count"] = 1
        record["status"] = "MISSING_LABEL"

    plan = build_acquisition_plan(payload)

    assert plan["active_tile_count"] == 13
    assert plan["train_tile_ids"] == list(INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS)
    assert plan["dev_tile_ids"] == list(INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS)
    assert plan["buffer_tile_ids_not_required"] == sorted(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS)
    assert plan["ready_active_tile_ids"] == []
    assert plan["partial_active_tile_ids"] == ["2_10"]
    assert "5_11" not in plan["partial_active_tile_ids"]
    assert plan["missing_rgb_count"] == 12
    assert plan["missing_dsm_count"] == 12
    assert plan["missing_label_count"] == 13
    assert "2_10" not in plan["missing_rgb_tile_ids"]
    assert "2_10" not in plan["missing_dsm_tile_ids"]
    assert "2_10" in plan["missing_label_tile_ids"]
    assert "5_11" not in plan["missing_label_tile_ids"]


def test_acquisition_plan_fails_closed_on_active_duplicate_component() -> None:
    payload = _inventory_payload()
    record = _record(payload, "6_7")
    record["rgb_count"] = 2
    record["dsm_count"] = 1
    record["label_count"] = 1

    with pytest.raises(RuntimeError, match="ambiguous duplicate inputs: 6_7"):
        build_acquisition_plan(payload)


def test_acquisition_plan_ignores_buffer_components_by_design() -> None:
    payload = _inventory_payload()
    record = _record(payload, "5_11")
    record["rgb_count"] = 7
    record["dsm_count"] = 7
    record["label_count"] = 7

    plan = build_acquisition_plan(payload)

    assert "5_11" in plan["buffer_tile_ids_not_required"]
    assert all(item["tile_id"] != "5_11" for item in plan["required_files"])
