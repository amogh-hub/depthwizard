from __future__ import annotations

import json
from pathlib import Path

import pytest

from depthwizard.height_model.terrain_structure_split import initial_tsd_campaign_split
from qualification.freeze_tsd_training_split import _unique_tfw, _write_manifest


def test_unique_tfw_is_mandatory_and_case_insensitive(tmp_path: Path) -> None:
    raster = tmp_path / "top_potsdam_2_10_RGB.tif"
    raster.write_bytes(b"rgb")
    world = tmp_path / "top_potsdam_2_10_RGB.TFW"
    world.write_text("0.05\n0\n0\n-0.05\n0\n0\n", encoding="utf-8")

    assert _unique_tfw(raster) == world


def test_unique_tfw_fails_closed_when_missing(tmp_path: Path) -> None:
    raster = tmp_path / "dsm_potsdam_02_10.tif"
    raster.write_bytes(b"dsm")

    with pytest.raises(FileNotFoundError, match="mandatory Potsdam .tfw"):
        _unique_tfw(raster)


def test_unique_tfw_fails_closed_when_ambiguous(tmp_path: Path) -> None:
    raster = tmp_path / "dsm_potsdam_02_10.tif"
    raster.write_bytes(b"dsm")
    (tmp_path / "dsm_potsdam_02_10.tfw").write_text("one", encoding="utf-8")
    (tmp_path / "dsm_potsdam_02_10.TFW").write_text("two", encoding="utf-8")

    matches = [
        path
        for path in tmp_path.iterdir()
        if path.stem.casefold() == raster.stem.casefold()
        and path.suffix.casefold() == ".tfw"
    ]
    if len(matches) < 2:
        pytest.skip("the test filesystem is case-insensitive and cannot represent this ambiguity")

    with pytest.raises(RuntimeError, match="ambiguous Potsdam .tfw"):
        _unique_tfw(raster)


def test_split_manifest_schema_freezes_georeference_identity_contract(tmp_path: Path) -> None:
    output = tmp_path / "split.json"
    split = initial_tsd_campaign_split()
    record: dict[str, object] = {
        "tile_id": split.train_tile_ids[0],
        "role": "train",
        "rgb_world_file_sha256": "a" * 64,
        "reference_dsm_world_file_sha256": "b" * 64,
    }

    _write_manifest(
        output,
        split=split,
        dataset_root=tmp_path,
        source_sha="c" * 40,
        source_branch="engineering/terrain-structure-vnext",
        records=[record],
        campaign_protocol_version="potsdam-tsd-urban-spatial-v1",
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 4
    assert "mandatory .tfw georeference sidecars are hashed" in payload["claim_boundary"]
    assert payload["tiles"][0]["rgb_world_file_sha256"] == "a" * 64
    assert payload["tiles"][0]["reference_dsm_world_file_sha256"] == "b" * 64
