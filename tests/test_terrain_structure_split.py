from __future__ import annotations

import pytest

from depthwizard.height_model.terrain_structure_split import (
    BLIND_SPATIAL_BUFFER_TILE_IDS,
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS,
    PARTICIPANT_GROUND_TRUTH_TILE_IDS,
    RESERVED_TILE_IDS,
    SEALED_BLIND_TILE_IDS,
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TRAIN_DEV_SPATIAL_BUFFER_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    assert_tsd_supervision_tile_allowed,
    freeze_tsd_data_split,
    initial_tsd_campaign_split,
    validate_potsdam_tile_id,
)


def _coordinate(tile_id: str) -> tuple[int, int]:
    row, col = tile_id.split("_")
    return int(row), int(col)


def test_tsd_split_freezes_disjoint_participant_ground_truth_tiles() -> None:
    split = freeze_tsd_data_split(
        ["2_10", "3_10", "5_11", "6_11"],
        ["5_12"],
    )

    assert split.protocol_version == TSD_SPLIT_PROTOCOL_VERSION
    assert split.train_tile_ids == ("2_10", "3_10", "5_11", "6_11")
    assert split.dev_tile_ids == ("5_12",)
    assert set(split.supervised_tile_ids) <= SUPERVISION_ELIGIBLE_TILE_IDS
    assert not (set(split.supervised_tile_ids) & RESERVED_TILE_IDS)


@pytest.mark.parametrize("tile_id", ["2_14", "3_14", "4_12", "6_12"])
def test_tsd_split_refuses_every_reserved_tile_before_supervision(tile_id: str) -> None:
    with pytest.raises(ValueError, match="refuses reserved tile"):
        freeze_tsd_data_split(["2_10", tile_id], ["5_12"])
    with pytest.raises(ValueError, match="refuses reserved tile"):
        freeze_tsd_data_split(["2_10", "3_10"], [tile_id])
    with pytest.raises(ValueError, match="refuses reserved tile"):
        assert_tsd_supervision_tile_allowed(tile_id)


@pytest.mark.parametrize("tile_id", ["3_13", "6_14", "5_15"])
def test_tsd_split_refuses_historical_challenge_test_tiles(tile_id: str) -> None:
    assert tile_id in HISTORICAL_CHALLENGE_TEST_TILE_IDS
    with pytest.raises(ValueError, match="historical Potsdam challenge-test evidence"):
        assert_tsd_supervision_tile_allowed(tile_id)
    with pytest.raises(ValueError, match="historical Potsdam challenge-test evidence"):
        freeze_tsd_data_split(["2_10", tile_id], ["5_12"])


def test_tsd_split_refuses_arbitrary_nonparticipant_tile() -> None:
    with pytest.raises(ValueError, match="not in the historical participant ground-truth"):
        assert_tsd_supervision_tile_allowed("8_8")


def test_supervision_pool_is_participant_ground_truth_minus_reserved() -> None:
    assert SUPERVISION_ELIGIBLE_TILE_IDS == PARTICIPANT_GROUND_TRUTH_TILE_IDS - RESERVED_TILE_IDS
    assert "3_13" not in SUPERVISION_ELIGIBLE_TILE_IDS
    assert "6_14" not in SUPERVISION_ELIGIBLE_TILE_IDS


def test_blind_ids_are_explicitly_part_of_reserved_contract() -> None:
    assert set(SEALED_BLIND_TILE_IDS) <= RESERVED_TILE_IDS


def test_initial_tsd_campaign_partition_is_predeclared_and_complete() -> None:
    split = initial_tsd_campaign_split()

    assert INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION == "potsdam-tsd-urban-spatial-v1"
    assert split.train_tile_ids == INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS
    assert split.dev_tile_ids == INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS
    assert len(split.train_tile_ids) == 8
    assert len(split.dev_tile_ids) == 5
    assert len(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS) == 9
    assert (
        set(split.supervised_tile_ids) | set(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS)
        == set(SUPERVISION_ELIGIBLE_TILE_IDS)
    )
    assert not (set(split.supervised_tile_ids) & set(INITIAL_TSD_CAMPAIGN_BUFFER_TILE_IDS))
    assert not (BLIND_SPATIAL_BUFFER_TILE_IDS & TRAIN_DEV_SPATIAL_BUFFER_TILE_IDS)


def test_initial_campaign_blind_buffer_covers_all_eligible_eight_neighbours() -> None:
    for blind_id in SEALED_BLIND_TILE_IDS:
        blind_row, blind_col = _coordinate(blind_id)
        eligible_neighbours = {
            tile_id
            for tile_id in SUPERVISION_ELIGIBLE_TILE_IDS
            if max(
                abs(_coordinate(tile_id)[0] - blind_row),
                abs(_coordinate(tile_id)[1] - blind_col),
            )
            == 1
        }
        assert eligible_neighbours <= BLIND_SPATIAL_BUFFER_TILE_IDS


def test_initial_campaign_train_and_dev_are_not_immediate_neighbours() -> None:
    split = initial_tsd_campaign_split()
    for train_id in split.train_tile_ids:
        train_row, train_col = _coordinate(train_id)
        for dev_id in split.dev_tile_ids:
            dev_row, dev_col = _coordinate(dev_id)
            assert max(abs(train_row - dev_row), abs(train_col - dev_col)) > 1


def test_tsd_split_refuses_overlap_duplicates_and_undersized_partition() -> None:
    with pytest.raises(ValueError, match="overlap"):
        freeze_tsd_data_split(["2_10", "3_10"], ["3_10"])
    with pytest.raises(ValueError, match="duplicate"):
        freeze_tsd_data_split(["2_10", "2_10"], ["5_12"])
    with pytest.raises(ValueError, match="at least two training"):
        freeze_tsd_data_split(["2_10"], ["5_12"])
    with pytest.raises(ValueError, match="at least one development"):
        freeze_tsd_data_split(["2_10", "3_10"], [])


@pytest.mark.parametrize("tile_id", ["2-10", "x_y", "02_10", "2_", "", " 2_10x "])
def test_tile_id_validation_is_strict(tile_id: str) -> None:
    with pytest.raises(ValueError, match="invalid Potsdam tile id"):
        validate_potsdam_tile_id(tile_id)


def test_tile_id_validation_normalizes_outer_whitespace_only() -> None:
    assert validate_potsdam_tile_id(" 2_10 ") == "2_10"
