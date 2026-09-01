from pathlib import Path

from qualification.inventory_tsd_potsdam_supervision import _inventory
from depthwizard.height_model.terrain_structure_split import RESERVED_TILE_IDS


def _index(*names: str) -> dict[str, list[Path]]:
    return {name.casefold(): [Path("/dataset") / name] for name in names}


def test_inventory_reports_missing_label_for_safe_rgb_dsm_tile() -> None:
    records = _inventory(
        _index(
            "top_potsdam_2_10_RGB.tif",
            "dsm_potsdam_02_10.tif",
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record.tile_id == "2_10"
    assert record.rgb_count == 1
    assert record.dsm_count == 1
    assert record.label_count == 0
    assert record.status == "MISSING_LABEL"
    assert not record.complete


def test_inventory_marks_complete_safe_supervision_tile() -> None:
    records = _inventory(
        _index(
            "top_potsdam_5_11_RGB.tif",
            "dsm_potsdam_05_11.tif",
            "top_potsdam_5_11_label.tif",
        )
    )

    assert len(records) == 1
    record = records[0]
    assert record.tile_id == "5_11"
    assert record.status == "COMPLETE"
    assert record.complete


def test_inventory_omits_reserved_ids_before_diagnostics() -> None:
    names: list[str] = []
    for tile_id in RESERVED_TILE_IDS:
        row, col = tile_id.split("_")
        names.extend(
            (
                f"top_potsdam_{row}_{col}_RGB.tif",
                f"dsm_potsdam_{int(row):02d}_{int(col):02d}.tif",
                f"top_potsdam_{row}_{col}_label.tif",
            )
        )

    assert _inventory(_index(*names)) == ()


def test_inventory_fails_ambiguous_component_instead_of_selecting_one() -> None:
    index = _index(
        "top_potsdam_6_14_RGB.tif",
        "dsm_potsdam_06_14.tif",
        "top_potsdam_6_14_label.tif",
    )
    index["top_potsdam_6_14_rgb.tif"].append(Path("/duplicate/top_potsdam_6_14_RGB.tif"))

    records = _inventory(index)

    assert len(records) == 1
    assert records[0].status == "AMBIGUOUS_RGB"
    assert not records[0].complete
