from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from scripts.freeze_final_science_manifest import PredictionFreezeError, freeze_prediction_manifest


def _registry(path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "scenes": [
            {
                "scene_id": "train-urban",
                "dataset": "development",
                "split": "train",
                "terrain": "urban",
                "geographic_group": "train-city",
                "sensor": "sensor-a",
                "rgb_path": "train-rgb.tif",
            },
            {
                "scene_id": "urban-test",
                "dataset": "urban",
                "split": "test",
                "terrain": "urban",
                "geographic_group": "urban-city",
                "sensor": "sensor-a",
                "rgb_path": "urban-rgb.tif",
                "reference_path": "urban-ref.tif",
            },
            {
                "scene_id": "sparse-test",
                "dataset": "sparse",
                "split": "test",
                "terrain": "sparse",
                "geographic_group": "sparse-site",
                "sensor": "sensor-a",
                "rgb_path": "sparse-rgb.tif",
                "reference_path": "sparse-ref.tif",
            },
            {
                "scene_id": "hilly-test",
                "dataset": "hilly",
                "split": "test",
                "terrain": "hilly",
                "geographic_group": "hilly-site",
                "sensor": "sensor-a",
                "rgb_path": "hilly-rgb.tif",
                "reference_path": "hilly-ref.tif",
            },
            {
                "scene_id": "forested-test",
                "dataset": "forest",
                "split": "test",
                "terrain": "forested",
                "geographic_group": "forest-site",
                "sensor": "sensor-a",
                "rgb_path": "forest-rgb.tif",
                "reference_path": "forest-ref.tif",
            },
            {
                "scene_id": "cross-test",
                "dataset": "cross",
                "split": "cross_sensor_test",
                "terrain": "forested",
                "geographic_group": "cross-site",
                "sensor": "sensor-b",
                "rgb_path": "cross-rgb.tif",
                "reference_path": "cross-ref.tif",
            },
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _touch_inputs(root: Path) -> None:
    for name in (
        "train-rgb.tif",
        "urban-rgb.tif",
        "urban-ref.tif",
        "sparse-rgb.tif",
        "sparse-ref.tif",
        "hilly-rgb.tif",
        "hilly-ref.tif",
        "forest-rgb.tif",
        "forest-ref.tif",
        "cross-rgb.tif",
        "cross-ref.tif",
        "checkpoint.bin",
    ):
        (root / name).write_bytes(f"fixture:{name}\n".encode())
    for scene in ("urban-test", "sparse-test", "hilly-test", "forested-test", "cross-test"):
        (root / f"{scene}-prediction.tif").write_bytes(f"prediction:{scene}\n".encode())
        (root / f"{scene}-dem.tif").write_bytes(f"calibration:{scene}\n".encode())


def _draft(path: Path, scene_ids: list[str]) -> Path:
    payload = {
        "schema_version": 2,
        "model_id": "calibrated-da3-production",
        "checkpoint_path": "checkpoint.bin",
        "predictions": [
            {
                "scene_id": scene_id,
                "prediction_path": f"{scene_id}-prediction.tif",
                "calibration_evidence_paths": [f"{scene_id}-dem.tif"],
                "prediction_vertical_datum": "EGM96 geoid",
                "reference_vertical_datum": "EGM96 geoid",
                "prediction_elevation_reference": "orthometric",
                "reference_elevation_reference": "orthometric",
            }
            for scene_id in scene_ids
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_freeze_hashes_predictions_without_opening_reference_rasters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _touch_inputs(tmp_path)
    registry = _registry(tmp_path / "registry.yaml")
    scene_ids = ["urban-test", "sparse-test", "hilly-test", "forested-test", "cross-test"]
    draft = _draft(tmp_path / "draft.yaml", scene_ids)
    output = tmp_path / "frozen-predictions.yaml"
    monkeypatch.setattr("scripts.freeze_final_science_manifest._git_head", lambda: "a" * 40)

    report = freeze_prediction_manifest(registry, draft, output)

    assert report["status"] == "PASS_FINAL_SCIENCE_PREDICTION_FREEZE"
    assert report["reference_rasters_opened_or_hashed"] is False
    assert report["evaluation_scene_count"] == 5
    frozen = yaml.safe_load(output.read_text(encoding="utf-8"))
    assert len(frozen["checkpoint_sha256"]) == 64
    assert {entry["scene_id"] for entry in frozen["predictions"]} == set(scene_ids)
    assert all(len(entry["prediction_sha256"]) == 64 for entry in frozen["predictions"])
    freeze_report = output.with_name("frozen-predictions-freeze-report.json")
    stored = json.loads(freeze_report.read_text(encoding="utf-8"))
    assert stored["frozen_manifest_sha256"] == report["frozen_manifest_sha256"]


def test_freeze_refuses_missing_or_extra_evaluation_predictions(tmp_path: Path) -> None:
    _touch_inputs(tmp_path)
    registry = _registry(tmp_path / "registry.yaml")
    draft = _draft(
        tmp_path / "draft.yaml",
        ["urban-test", "sparse-test", "hilly-test", "forested-test"],
    )

    with pytest.raises(PredictionFreezeError, match="must exactly match"):
        freeze_prediction_manifest(registry, draft, tmp_path / "frozen.yaml")


def test_freeze_refuses_placeholder_vertical_datum(tmp_path: Path) -> None:
    _touch_inputs(tmp_path)
    registry = _registry(tmp_path / "registry.yaml")
    scene_ids = ["urban-test", "sparse-test", "hilly-test", "forested-test", "cross-test"]
    draft = _draft(tmp_path / "draft.yaml", scene_ids)
    payload = yaml.safe_load(draft.read_text(encoding="utf-8"))
    payload["predictions"][0]["prediction_vertical_datum"] = "unspecified"
    payload["predictions"][0]["reference_vertical_datum"] = "unspecified"
    draft.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(PredictionFreezeError, match="must be explicit"):
        freeze_prediction_manifest(registry, draft, tmp_path / "frozen.yaml")
