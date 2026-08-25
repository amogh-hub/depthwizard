from pathlib import Path

import pytest

from depthwizard.data.registry import DatasetRegistry, DatasetScene


def scene(scene_id: str, split: str, group: str, sensor: str) -> DatasetScene:
    return DatasetScene(
        scene_id=scene_id,
        dataset="d",
        split=split,
        terrain="urban",
        geographic_group=group,
        sensor=sensor,
        rgb_path=Path(f"/{scene_id}.tif"),
    )


def test_registry_rejects_geographic_leakage() -> None:
    registry = DatasetRegistry(scenes=[
        scene("a", "train", "same-city", "sensor-a"),
        scene("b", "test", "same-city", "sensor-a"),
    ])
    with pytest.raises(ValueError, match="geographic split leakage"):
        registry.assert_integrity()


def test_registry_rejects_cross_sensor_holdout_using_training_sensor() -> None:
    registry = DatasetRegistry(scenes=[
        scene("a", "train", "city-a", "sensor-a"),
        scene("b", "cross_sensor_test", "city-b", "sensor-a"),
    ])
    with pytest.raises(ValueError, match="training sensor"):
        registry.assert_integrity()
