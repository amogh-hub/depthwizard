from __future__ import annotations

from collections import Counter

import pytest

from depthwizard.data.ortholoc import OrthoLoCRemoteScene
from scripts.train_ortholoc_multiscene_v3 import select_diversity_split


def _scene(location: str, index: int, split: str) -> OrthoLoCRemoteScene:
    filename = f"{location}_R{index:04d}.tif"
    root = f"https://example.test/{split}"
    return OrthoLoCRemoteScene(
        scene_id=filename.removesuffix(".tif"),
        split=split,
        location_id=location,
        dop_url=f"{root}/DOPs/{filename}",
        dsm_url=f"{root}/DSMs/{filename}",
    )


def test_diversity_split_uses_five_train_and_two_disjoint_validation_locations() -> None:
    train = [
        _scene(location, index, "train")
        for location in ("L01", "L02", "L03", "L04", "L05", "L06", "L07")
        for index in range(2)
    ]
    outplace = [_scene("L08", 0, "test_outPlace"), _scene("L50", 0, "test_outPlace")]

    selected_train, selected_validation, selected_development = select_diversity_split(
        train,
        outplace,
    )

    train_counts = Counter(scene.location_id for scene in selected_train)
    assert train_counts == {"L01": 2, "L02": 2, "L03": 2, "L04": 2, "L05": 2}
    assert {scene.location_id for scene in selected_validation} == {"L06", "L07"}
    assert {scene.location_id for scene in selected_development} == {"L08", "L50"}
    assert set(train_counts).isdisjoint(scene.location_id for scene in selected_validation)


def test_diversity_split_rejects_training_location_without_two_same_domain_scenes() -> None:
    train = [
        _scene(location, index, "train")
        for location in ("L01", "L02", "L03", "L04", "L05", "L06", "L07")
        for index in range(2)
        if not (location == "L05" and index == 1)
    ]
    outplace = [_scene("L08", 0, "test_outPlace"), _scene("L50", 0, "test_outPlace")]

    with pytest.raises(RuntimeError, match="exactly two scenes per training location"):
        select_diversity_split(train, outplace)
