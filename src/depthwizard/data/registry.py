from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

TerrainClass = Literal["urban", "sparse", "hilly", "forested"]
Split = Literal["train", "validation", "test", "cross_sensor_test"]


class DatasetScene(BaseModel):
    scene_id: str = Field(min_length=1)
    dataset: str = Field(min_length=1)
    split: Split
    terrain: TerrainClass
    geographic_group: str = Field(min_length=1)
    sensor: str = Field(min_length=1)
    rgb_path: Path
    reference_path: Path | None = None
    semantic_path: Path | None = None
    nominal_gsd_m: float | None = Field(default=None, gt=0)
    license_id: str | None = None
    notes: str | None = None


class DatasetRegistry(BaseModel):
    schema_version: int = 1
    scenes: list[DatasetScene]

    def assert_integrity(self, *, require_files: bool = False) -> None:
        seen_ids: set[str] = set()
        for scene in self.scenes:
            if scene.scene_id in seen_ids:
                raise ValueError(f"duplicate scene_id: {scene.scene_id}")
            seen_ids.add(scene.scene_id)
            if require_files:
                if not scene.rgb_path.exists():
                    raise ValueError(f"missing RGB file for {scene.scene_id}: {scene.rgb_path}")
                if scene.reference_path is not None and not scene.reference_path.exists():
                    raise ValueError(
                        f"missing reference file for {scene.scene_id}: {scene.reference_path}"
                    )

        split_by_group: dict[str, set[str]] = defaultdict(set)
        for scene in self.scenes:
            split_by_group[scene.geographic_group].add(scene.split)
        leakage = {
            group: sorted(splits)
            for group, splits in split_by_group.items()
            if len(splits) > 1
        }
        if leakage:
            raise ValueError(f"geographic split leakage detected: {leakage}")

        train_sensors = {s.sensor for s in self.scenes if s.split == "train"}
        for scene in self.scenes:
            if scene.split == "cross_sensor_test" and scene.sensor in train_sensors:
                raise ValueError(
                    f"cross_sensor_test scene {scene.scene_id} uses training sensor {scene.sensor}"
                )

    def terrain_coverage(self) -> dict[str, set[str]]:
        coverage: dict[str, set[str]] = defaultdict(set)
        for scene in self.scenes:
            coverage[scene.split].add(scene.terrain)
        return dict(coverage)


def load_registry(path: str | Path) -> DatasetRegistry:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    registry = DatasetRegistry.model_validate(payload)
    registry.assert_integrity()
    return registry
