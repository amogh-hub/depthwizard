from pathlib import Path

from depthwizard.height_model.terrain_structure_split import (
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    RESERVED_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
)
from qualification.inventory_tsd_potsdam_supervision import TileInventory, _inventory


def _index(*names: str) -> dict[str, list[Path]]:
    return {name.casefold(): [Path("/dataset") / name] for name in names}


def _record(records: tuple[TileInventory, ...], tile_id: str) -> TileInventory:
    return next(record for record in records if record.tile_id == tile_id)


def test_inventory_reports_missing_label_for_eligible_rgb_dsm_tile() -> None:
    records = _inventory(
        _index(
            "top_potsdam_2_10_RGB.tif",
            "dsm_potsdam_02_10.tif",
        )
    )

    assert len(records) == len(SUPERVISION_ELIGIBLE_TILE_IDS)
    record = _record(records, "2_10")
    assert record.rgb_count == 1
    assert record.dsm_count == 1
    assert record.label_count == 0
    assert record.status == "MISSING_LABEL"
    assert record.locally_present
    assert not record.complete


def test_inventory_marks_complete_eligible_supervision_tile() -> None:
    records = _inventory(
        _index(
            "top_potsdam_5_11_RGB.tif",
            "dsm_potsdam_05_11.tif",
            "top_potsdam_5_11_label.tif",
        )
    )

    record = _record(records, "5_11")
    assert record.status == "COMPLETE"
    assert record.complete


def test_inventory_never_constructs_reserved_or_challenge_test_records() -> None:
    names: list[str] = []
    prohibited = RESERVED_TILE_IDS | HISTORICAL_CHALLENGE_TEST_TILE_IDS
    for tile_id in prohibited:
        row, col = tile_id.split("_")
        names.extend(
            (
                f"top_potsdam_{row}_{col}_RGB.tif",
                f"dsm_potsdam_{int(row):02d}_{int(col):02d}.tif",
                f"top_potsdam_{row}_{col}_label.tif",
            )
        )

    records = _inventory(_index(*names))
    assert {record.tile_id for record in records} == SUPERVISION_ELIGIBLE_TILE_IDS
    assert all(not record.locally_present for record in records)


def test_inventory_fails_ambiguous_component_instead_of_selecting_one() -> None:
    index = _index(
        "top_potsdam_6_11_RGB.tif",
        "dsm_potsdam_06_11.tif",
        "top_potsdam_6_11_label.tif",
    )
    index["top_potsdam_6_11_rgb.tif"].append(Path("/duplicate/top_potsdam_6_11_RGB.tif"))

    record = _record(_inventory(index), "6_11")

    assert record.status == "AMBIGUOUS_RGB"
    assert not record.complete
