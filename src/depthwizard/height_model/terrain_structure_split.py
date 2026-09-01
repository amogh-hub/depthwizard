from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final

TSD_SPLIT_PROTOCOL_VERSION: Final = "terrain-structure-training-split-v1"
EXPOSED_CORRECTIVE_TILE_IDS: Final[tuple[str, ...]] = ("2_14",)
EXTERNAL_EVALUATION_TILE_IDS: Final[tuple[str, ...]] = ("3_14",)
SEALED_BLIND_TILE_IDS: Final[tuple[str, ...]] = ("4_12", "6_12")
RESERVED_TILE_IDS: Final[frozenset[str]] = frozenset(
    EXPOSED_CORRECTIVE_TILE_IDS + EXTERNAL_EVALUATION_TILE_IDS + SEALED_BLIND_TILE_IDS
)
_TILE_PATTERN = re.compile(r"^[1-9][0-9]*_[1-9][0-9]*$")


@dataclass(frozen=True)
class TerrainStructureDataSplit:
    """Frozen tile-level supervision partition for TSD research.

    The exposed corrective tile, external evaluation tile, and sealed blind tiles are not legal in
    either training or development. The development split is reserved for ordinary model selection,
    early stopping, and hyperparameter iteration; exposed 2_14 remains an external corrective gate.
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
    for role, tile_ids in (("training", train), ("development", dev)):
        for tile_id in tile_ids:
            if tile_id in RESERVED_TILE_IDS:
                raise ValueError(
                    f"{role} split refuses reserved tile {tile_id}: {_reserved_reason(tile_id)}"
                )
    return train, dev


def freeze_tsd_data_split(
    train_tile_ids: tuple[str, ...] | list[str],
    dev_tile_ids: tuple[str, ...] | list[str],
) -> TerrainStructureDataSplit:
    """Normalize and freeze a legal TSD supervision split without consulting raster contents."""

    train, dev = _validate_partition(train_tile_ids, dev_tile_ids)
    return TerrainStructureDataSplit(train_tile_ids=train, dev_tile_ids=dev)


def assert_tsd_supervision_tile_allowed(tile_id: str) -> None:
    """Fail before any filesystem resolution when a reserved tile is requested for supervision."""

    normalized = validate_potsdam_tile_id(tile_id)
    if normalized in RESERVED_TILE_IDS:
        raise ValueError(
            f"TSD supervision refuses reserved tile {normalized}: {_reserved_reason(normalized)}"
        )
