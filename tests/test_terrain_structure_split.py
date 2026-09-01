from __future__ import annotations

import pytest

from depthwizard.height_model.terrain_structure_split import (
    RESERVED_TILE_IDS,
    SEALED_BLIND_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    assert_tsd_supervision_tile_allowed,
    freeze_tsd_data_split,
    validate_potsdam_tile_id,
)


def test_tsd_split_freezes_disjoint_non_reserved_tiles() -> None:
    split = freeze_tsd_data_split(
        ["2_10", "3_13", "5_11", "6_14"],
        ["5_12"],
    )

    assert split.protocol_version == TSD_SPLIT_PROTOCOL_VERSION
    assert split.train_tile_ids == ("2_10", "3_13", "5_11", "6_14")
    assert split.dev_tile_ids == ("5_12",)
    assert not (set(split.supervised_tile_ids) & RESERVED_TILE_IDS)


@pytest.mark.parametrize("tile_id", ["2_14", "3_14", "4_12", "6_12"])
def test_tsd_split_refuses_every_reserved_tile_before_supervision(tile_id: str) -> None:
    with pytest.raises(ValueError, match="refuses reserved tile"):
        freeze_tsd_data_split(["2_10", tile_id], ["5_12"])
    with pytest.raises(ValueError, match="refuses reserved tile"):
        freeze_tsd_data_split(["2_10", "3_13"], [tile_id])
    with pytest.raises(ValueError, match="refuses reserved tile"):
        assert_tsd_supervision_tile_allowed(tile_id)


def test_blind_ids_are_explicitly_part_of_reserved_contract() -> None:
    assert set(SEALED_BLIND_TILE_IDS) <= RESERVED_TILE_IDS


def test_tsd_split_refuses_overlap_duplicates_and_undersized_partition() -> None:
    with pytest.raises(ValueError, match="overlap"):
        freeze_tsd_data_split(["2_10", "3_13"], ["3_13"])
    with pytest.raises(ValueError, match="duplicate"):
        freeze_tsd_data_split(["2_10", "2_10"], ["5_12"])
    with pytest.raises(ValueError, match="at least two training"):
        freeze_tsd_data_split(["2_10"], ["5_12"])
    with pytest.raises(ValueError, match="at least one development"):
        freeze_tsd_data_split(["2_10", "3_13"], [])


@pytest.mark.parametrize("tile_id", ["2-10", "x_y", "02_10", "2_", "", " 2_10x "])
def test_tile_id_validation_is_strict(tile_id: str) -> None:
    with pytest.raises(ValueError, match="invalid Potsdam tile id"):
        validate_potsdam_tile_id(tile_id)


def test_tile_id_validation_normalizes_outer_whitespace_only() -> None:
    assert validate_potsdam_tile_id(" 2_10 ") == "2_10"
