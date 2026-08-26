from __future__ import annotations

import json
from pathlib import Path

import pytest

from depthwizard.data.ortholoc import OrthoLoCRemoteScene
from scripts import evaluate_ortholoc_frozen_holdout as frozen


def _scene(location: str, index: int, *, sample_type: str = "R") -> OrthoLoCRemoteScene:
    filename = f"{location}_{sample_type}{index:04d}.tif"
    root = "https://example.test/test_outPlace"
    return OrthoLoCRemoteScene(
        scene_id=filename.removesuffix(".tif"),
        split="test_outPlace",
        location_id=location,
        dop_url=f"{root}/DOPs/{filename}",
        dsm_url=f"{root}/DSMs/{filename}",
    )


def test_frozen_holdout_selection_is_deterministic_and_excludes_inspected_locations() -> None:
    discovered = [
        _scene(location, index)
        for location in ("L08", "L09", "L10", "L11", "L12", "L13", "L50")
        for index in range(3)
    ]
    discovered.extend([_scene("L09", 99, sample_type="xDOP"), _scene("L10", 99, sample_type="xDSM")])

    forward = frozen.select_frozen_holdout_scenes(discovered)
    reverse = frozen.select_frozen_holdout_scenes(list(reversed(discovered)))

    expected_ids = [
        "L09_R0000",
        "L09_R0001",
        "L10_R0000",
        "L10_R0001",
        "L11_R0000",
        "L11_R0001",
        "L12_R0000",
        "L12_R0001",
    ]
    assert [scene.scene_id for scene in forward] == expected_ids
    assert [scene.scene_id for scene in reverse] == expected_ids
    assert not ({scene.location_id for scene in forward} & frozen.EXCLUDED_LOCATIONS)
    assert all(scene.same_domain for scene in forward)


def test_frozen_holdout_requires_four_locations_with_two_scenes_before_target_load() -> None:
    discovered = [
        _scene(location, index)
        for location in ("L09", "L10", "L11")
        for index in range(2)
    ]
    discovered.append(_scene("L12", 0))

    with pytest.raises(RuntimeError, match="before loading any DSM targets"):
        frozen.select_frozen_holdout_scenes(discovered)


def test_frozen_holdout_location_sort_is_numeric_not_lexicographic() -> None:
    discovered = [
        _scene(location, index)
        for location in ("L9", "L10", "L11", "L12", "L100")
        for index in range(2)
    ]

    selected = frozen.select_frozen_holdout_scenes(discovered)

    assert [scene.location_id for scene in selected[::2]] == ["L9", "L10", "L11", "L12"]


def test_protocol_seal_reuses_identical_payload_and_rejects_modified_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seal_path = tmp_path / "protocol_seal.json"
    monkeypatch.setattr(frozen, "OUT_DIR", tmp_path)
    monkeypatch.setattr(frozen, "SEAL_PATH", seal_path)
    payload: dict[str, object] = {"protocol_version": "test", "value": 1}

    first = frozen.seal_protocol(payload)
    second = frozen.seal_protocol(payload)

    assert first == second
    stored = json.loads(seal_path.read_text(encoding="utf-8"))
    assert stored["state"] == "SEALED_BEFORE_HOLDOUT_TARGET_LOAD"
    assert stored["protocol_sha256"] == first

    with pytest.raises(RuntimeError, match="frozen holdout seal mismatch"):
        frozen.seal_protocol({"protocol_version": "test", "value": 2})
