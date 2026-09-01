from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

TSD_SPLIT_PROTOCOL_VERSION: Final = "terrain-structure-training-split-v2"

# Conservative reconstruction of the historical Potsdam 2D semantic-labeling participant split.
# ISPRS states that only the ground-truth portion is participant supervision and that the remaining
# scenes are benchmark evaluation. The concrete historical tile enumeration below is cross-checked
# against the long-standing TorchGeo Potsdam2D challenge split. Keeping this list explicit prevents
# later 'all labels' packages from silently widening the TSD supervision population.
PARTICIPANT_GROUND_TRUTH_TILE_IDS: Final[frozenset[str]] = frozenset(
    {
        "2_10",
        "2_11",
        "2_12",
        "3_10",
        "3_11",
        "3_12",
        "4_10",
        "4_11",
        "4_12",
        "5_10",
        "5_11",
        "5_12",
        "6_10",
        "6_11",
        "6_12",
        "6_7",
        "6_8",
        "6_9",
        "7_10",
        "7_11",
        "7_12",
        "7_7",
        "7_8",
        "7_9",
    }
)

HISTORICAL_CHALLENGE_TEST_TILE_IDS: Final[frozenset[str]] = frozenset(
    {
        "2_13",
        "2_14",
        "3_13",
        "3_14",
        "4_13",
        "4_14",
        "4_15",
        "5_13",
        "5_14",
        "5_15",
        "6_13",
        "6_14",
        "6_15",
        "7_13",
    }
)

EXPOSED_CORRECTIVE_TILE_IDS: Final[tuple[str, ...]] = ("2_14",)
EXTERNAL_EVALUATION_TILE_IDS: Final[tuple[str, ...]] = ("3_14",)
SEALED_BLIND_TILE_IDS: Final[tuple[str, ...]] = ("4_12", "6_12")
RESERVED_TILE_IDS: Final[frozenset[str]] = frozenset(
    EXPOSED_CORRECTIVE_TILE_IDS + EXTERNAL_EVALUATION_TILE_IDS + SEALED_BLIND_TILE_IDS
)
SUPERVISION_ELIGIBLE_TILE_IDS: Final[frozenset[str]] = frozenset(
    PARTICIPANT_GROUND_TRUTH_TILE_IDS - RESERVED_TILE_IDS
)
_TILE_PATTERN = re.compile(r"^[1-9][0-9]*_[1-9][0-9]*$")


@dataclass(frozen=True)
class TerrainStructureDataSplit:
    """Frozen tile-level supervision partition for TSD research.

    TSD train/dev supervision is restricted to the historical participant ground-truth population,
    minus DepthWizard's own sealed blind tiles. Exposed corrective evidence, external evaluation
    evidence, historical challenge-test scenes, and arbitrary later-labelled scenes are not legal in
    training, development, early stopping, or hyperparameter selection.
    """

    train_tile_ids: tuple[str, ...]
    dev_tile_ids: tuple[str, ...]
    protocol_version: str = TSD_SPLIT_PROTOCOL_VERSION

    def __post_init__(self) -> None:
        if self.protocol_version != TSD_SPLIT_PROTOCOL_VERSION:
            raise ValueError(f"unsupported TSD split protocol: {self.protocol_version}")
        _validate_partition(self.train_tile_ids, self.dev_tile_ids)

    @property
    def supervised_tile_ids(self) -> tuple[str, ...]:
        return self.train_tile_ids + self.dev_tile_ids


def validate_potsdam_tile_id(tile_id: str) -> str:
    normalized = tile_id.strip()
    if not _TILE_PATTERN.fullmatch(normalized):
        raise ValueError(f"invalid Potsdam tile id: {tile_id!r}")
    return normalized


def _normalize_unique(role: str, tile_ids: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    normalized = tuple(validate_potsdam_tile_id(tile_id) for tile_id in tile_ids)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"duplicate tile id in {role} split")
    return normalized


def _reserved_reason(tile_id: str) -> str:
    if tile_id in EXPOSED_CORRECTIVE_TILE_IDS:
        return "already-exposed corrective/evaluation evidence"
    if tile_id in EXTERNAL_EVALUATION_TILE_IDS:
        return "external/cross-sensor evaluation evidence"
    if tile_id in SEALED_BLIND_TILE_IDS:
        return "sealed blind evidence"
    raise ValueError(f"tile {tile_id!r} is not reserved")


def _supervision_rejection_reason(tile_id: str) -> str:
    if tile_id in RESERVED_TILE_IDS:
        return _reserved_reason(tile_id)
    if tile_id in HISTORICAL_CHALLENGE_TEST_TILE_IDS:
        return "historical Potsdam challenge-test evidence"
    return "not in the historical participant ground-truth supervision population"


def assert_tsd_supervision_tile_allowed(tile_id: str) -> None:
    """Fail before filesystem resolution when a tile is not legal TSD supervision."""

    normalized = validate_potsdam_tile_id(tile_id)
    if normalized not in SUPERVISION_ELIGIBLE_TILE_IDS:
        if normalized in RESERVED_TILE_IDS:
            raise ValueError(
                f"TSD supervision refuses reserved tile {normalized}: {_reserved_reason(normalized)}"
            )
        raise ValueError(
            f"TSD supervision refuses tile {normalized}: {_supervision_rejection_reason(normalized)}"
        )


def _validate_partition(
    train_tile_ids: tuple[str, ...] | list[str],
    dev_tile_ids: tuple[str, ...] | list[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    train = _normalize_unique("training", train_tile_ids)
    dev = _normalize_unique("development", dev_tile_ids)
    if len(train) < 2:
        raise ValueError("TSD split requires at least two training tiles")
    if not dev:
        raise ValueError("TSD split requires at least one development tile")
    overlap = set(train) & set(dev)
    if overlap:
        raise ValueError(f"training/development tile overlap: {', '.join(sorted(overlap))}")
    for tile_id in train + dev:
        assert_tsd_supervision_tile_allowed(tile_id)
    return train, dev


def freeze_tsd_data_split(
    train_tile_ids: tuple[str, ...] | list[str],
    dev_tile_ids: tuple[str, ...] | list[str],
) -> TerrainStructureDataSplit:
    """Normalize and freeze a legal TSD supervision split without consulting raster contents."""

    train, dev = _validate_partition(train_tile_ids, dev_tile_ids)
    return TerrainStructureDataSplit(train_tile_ids=train, dev_tile_ids=dev)
