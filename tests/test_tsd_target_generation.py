from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
from rasterio.crs import CRS

from depthwizard.height_model.terrain_structure_split import initial_tsd_campaign_split
from qualification.generate_tsd_potsdam_targets import (
    _label_alignment_basis,
    _validate_frozen_split,
    _write_npz_atomic,
)


def _frozen_payload() -> dict[str, object]:
    split = initial_tsd_campaign_split()
    records: list[dict[str, object]] = []
    for role, tile_ids in (("train", split.train_tile_ids), ("dev", split.dev_tile_ids)):
        for tile_id in tile_ids:
            records.append(
                {
                    "tile_id": tile_id,
                    "role": role,
                    "rgb": f"/dataset/{tile_id}/rgb.tif",
                    "rgb_sha256": "a" * 64,
                    "rgb_world_file": f"/dataset/{tile_id}/rgb.tfw",
                    "rgb_world_file_sha256": "b" * 64,
                    "reference_dsm": f"/dataset/{tile_id}/dsm.tif",
                    "reference_dsm_sha256": "c" * 64,
                    "reference_dsm_world_file": f"/dataset/{tile_id}/dsm.tfw",
                    "reference_dsm_world_file_sha256": "d" * 64,
                    "semantic_label": f"/dataset/{tile_id}/label.tif",
                    "semantic_label_sha256": "e" * 64,
                    "rgb_contract": {},
                    "reference_contract": {},
                    "semantic_label_metadata": {},
                }
            )
    return {
        "schema_version": 4,
        "status": "FROZEN_TSD_SUPERVISION_SPLIT",
        "protocol_version": "terrain-structure-training-split-v2",
        "campaign_protocol_version": "potsdam-tsd-urban-spatial-v1",
        "train_tile_ids": list(split.train_tile_ids),
        "dev_tile_ids": list(split.dev_tile_ids),
        "campaign_buffer_tile_ids_withheld": [
            "3_11",
            "3_12",
            "4_11",
            "5_10",
            "5_11",
            "5_12",
            "6_11",
            "7_11",
            "7_12",
        ],
        "tiles": records,
    }


def test_target_generation_requires_world_file_bound_schema() -> None:
    payload = _frozen_payload()
    records = _validate_frozen_split(payload)
    assert len(records) == 13

    payload["schema_version"] = 3
    with pytest.raises(ValueError, match="schema_version=4"):
        _validate_frozen_split(payload)


def test_target_generation_rejects_missing_world_file_identity() -> None:
    payload = _frozen_payload()
    tiles = payload["tiles"]
    assert isinstance(tiles, list)
    del tiles[0]["rgb_world_file_sha256"]

    with pytest.raises(ValueError, match="missing identity fields"):
        _validate_frozen_split(payload)


def test_identity_transform_label_uses_official_same_tile_pixel_grid_contract() -> None:
    basis = _label_alignment_basis(
        label_shape=(6000, 6000),
        rgb_shape=(6000, 6000),
        label_transform=Affine.identity(),
        rgb_transform=Affine(0.05, 0.0, 100.0, 0.0, -0.05, 200.0),
        label_crs=None,
        rgb_crs=CRS.from_epsg(32633),
    )
    assert basis == "official_same_tile_exact_pixel_grid"


def test_georeferenced_label_must_match_rgb_transform_and_crs() -> None:
    transform = Affine(0.05, 0.0, 100.0, 0.0, -0.05, 200.0)
    crs = CRS.from_epsg(32633)
    assert (
        _label_alignment_basis(
            label_shape=(6000, 6000),
            rgb_shape=(6000, 6000),
            label_transform=transform,
            rgb_transform=transform,
            label_crs=crs,
            rgb_crs=crs,
        )
        == "geospatial_transform_match"
    )

    with pytest.raises(ValueError, match="affine transform disagrees"):
        _label_alignment_basis(
            label_shape=(6000, 6000),
            rgb_shape=(6000, 6000),
            label_transform=Affine(0.05, 0.0, 101.0, 0.0, -0.05, 200.0),
            rgb_transform=transform,
            label_crs=crs,
            rgb_crs=crs,
        )


def test_target_pack_writer_is_atomic_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "metric-targets.npz"
    terrain = np.array([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32)
    mask = np.array([[1, 0], [1, 1]], dtype=np.uint8)

    _write_npz_atomic(destination, terrain_m=terrain, valid_mask=mask)

    assert destination.is_file()
    assert not destination.with_suffix(".npz.tmp").exists()
    with np.load(destination) as pack:
        assert np.allclose(pack["terrain_m"], terrain, equal_nan=True)
        assert np.array_equal(pack["valid_mask"], mask)


def test_target_generator_direct_launcher_is_importable() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, "qualification/generate_tsd_potsdam_targets.py", "--help"],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--split-manifest" in completed.stdout
