from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from depthwizard.evaluation.final_campaign import evaluate_final_science_campaign
from depthwizard.provenance.manifest import sha256_file


def _write_surface(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _setup_campaign(
    tmp_path: Path,
    *,
    missing_terrain: str | None = None,
    reference_as_evidence: bool = False,
    identical_prediction: bool = False,
) -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    prediction_dir = tmp_path / "preds"
    data_dir.mkdir()
    prediction_dir.mkdir()

    train_rgb = data_dir / "train.rgb"
    train_rgb.write_bytes(b"train")
    scenes: list[dict[str, object]] = [
        {
            "scene_id": "train",
            "dataset": "d",
            "split": "train",
            "terrain": "urban",
            "geographic_group": "train-g",
            "sensor": "sensor-a",
            "rgb_path": "data/train.rgb",
        }
    ]
    predictions: list[dict[str, object]] = []
    y, x = np.mgrid[:8, :8]
    truth = 100.0 + 0.5 * x + 0.25 * y
    offsets = {"urban": 1.0, "sparse": 2.0, "hilly": 3.0, "forested": 4.0}

    for terrain, offset in offsets.items():
        if terrain == missing_terrain:
            continue
        scene_id = f"test-{terrain}"
        (data_dir / f"{scene_id}.rgb").write_bytes(scene_id.encode("utf-8"))
        reference = data_dir / f"{scene_id}-ref.tif"
        prediction = prediction_dir / f"{scene_id}-pred.tif"
        evidence = data_dir / f"{scene_id}-dem.tif"
        _write_surface(reference, truth)
        _write_surface(
            prediction,
            truth if identical_prediction and terrain == "urban" else truth + offset,
        )
        _write_surface(evidence, truth - 10.0)
        scenes.append(
            {
                "scene_id": scene_id,
                "dataset": "d",
                "split": "test",
                "terrain": terrain,
                "geographic_group": f"g-{terrain}",
                "sensor": "sensor-a",
                "rgb_path": f"data/{scene_id}.rgb",
                "reference_path": f"data/{scene_id}-ref.tif",
                "nominal_gsd_m": 1.0,
            }
        )
        evidence_path = (
            f"../data/{scene_id}-ref.tif"
            if reference_as_evidence and terrain == "urban"
            else f"../data/{scene_id}-dem.tif"
        )
        predictions.append(
            {
                "scene_id": scene_id,
                "prediction_path": f"../preds/{scene_id}-pred.tif",
                "prediction_sha256": sha256_file(prediction),
                "calibration_evidence_paths": [evidence_path],
                "prediction_vertical_datum": "EGM96 geoid",
                "reference_vertical_datum": "EGM96 geoid",
                "prediction_elevation_reference": "orthometric",
                "reference_elevation_reference": "orthometric",
            }
        )

    cross_scene_id = "cross-urban"
    (data_dir / f"{cross_scene_id}.rgb").write_bytes(b"cross")
    cross_reference = data_dir / f"{cross_scene_id}-ref.tif"
    cross_prediction = prediction_dir / f"{cross_scene_id}-pred.tif"
    cross_evidence = data_dir / f"{cross_scene_id}-dem.tif"
    _write_surface(cross_reference, truth)
    _write_surface(cross_prediction, truth + 0.5)
    _write_surface(cross_evidence, truth - 20.0)
    scenes.append(
        {
            "scene_id": cross_scene_id,
            "dataset": "d2",
            "split": "cross_sensor_test",
            "terrain": "urban",
            "geographic_group": "cross-g",
            "sensor": "sensor-b",
            "rgb_path": f"data/{cross_scene_id}.rgb",
            "reference_path": f"data/{cross_scene_id}-ref.tif",
            "nominal_gsd_m": 1.0,
        }
    )
    predictions.append(
        {
            "scene_id": cross_scene_id,
            "prediction_path": f"../preds/{cross_scene_id}-pred.tif",
            "prediction_sha256": sha256_file(cross_prediction),
            "calibration_evidence_paths": [f"../data/{cross_scene_id}-dem.tif"],
            "prediction_vertical_datum": "EGM96 geoid",
            "reference_vertical_datum": "EGM96 geoid",
            "prediction_elevation_reference": "orthometric",
            "reference_elevation_reference": "orthometric",
        }
    )

    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump({"schema_version": 1, "scenes": scenes}, sort_keys=False),
        encoding="utf-8",
    )
    campaign_dir = tmp_path / "campaign"
    campaign_dir.mkdir()
    manifest = campaign_dir / "predictions.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "model_id": "DA3MONO-LARGE",
                "checkpoint_sha256": "a" * 64,
                "predictions": predictions,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return registry, manifest


def test_campaign_emits_required_four_terrain_and_cross_sensor_evidence(
    tmp_path: Path,
) -> None:
    registry, manifest = _setup_campaign(tmp_path)
    output_dir = tmp_path / "out"

    report = evaluate_final_science_campaign(
        registry,
        manifest,
        output_dir,
        min_valid_pixels=4,
    )

    assert report["requirements"]["test_terrain_coverage"] == [
        "forested",
        "hilly",
        "sparse",
        "urban",
    ]
    assert report["requirements"]["cross_sensor_scene_count"] == 1
    assert report["requirements"]["checkpoint_identity_frozen"] == "passed"
    assert report["requirements"]["prediction_identity_freeze"] == "passed"
    assert report["terrain"]["urban"]["rmse_m"] == pytest.approx(1.0, abs=1e-5)
    assert report["terrain"]["forested"]["mae_m"] == pytest.approx(4.0, abs=1e-5)
    assert report["test_overall"]["rmse_m"] == pytest.approx(np.sqrt(7.5), abs=1e-5)
    assert report["cross_sensor_overall"]["rmse_m"] == pytest.approx(0.5, abs=1e-5)

    rows = list(csv.DictReader((output_dir / "terrain_breakdown.csv").open()))
    assert [row["terrain"] for row in rows] == ["urban", "sparse", "hilly", "forested"]
    persisted = json.loads(
        (output_dir / "domain_generalization_report.json").read_text(encoding="utf-8")
    )
    assert persisted["protocol"] == "depthwizard_final_science_campaign_v2"


def test_campaign_requires_all_four_test_terrains(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path, missing_terrain="forested")

    with pytest.raises(ValueError, match=r"missing=.*forested"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )


def test_campaign_rejects_reference_reused_for_calibration(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path, reference_as_evidence=True)

    with pytest.raises(ValueError, match="reference is reused as calibration evidence"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )


def test_campaign_rejects_prediction_identical_to_reference(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path, identical_prediction=True)

    with pytest.raises(ValueError, match="prediction is byte-identical"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )


def test_campaign_rejects_prediction_mutated_after_manifest_freeze(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    prediction_entry = payload["predictions"][0]
    prediction_path = (manifest.parent / prediction_entry["prediction_path"]).resolve()
    prediction_path.write_bytes(prediction_path.read_bytes() + b"post-freeze-mutation")

    with pytest.raises(ValueError, match="no longer matches frozen manifest"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )


def test_campaign_rejects_vertical_datum_mismatch_before_scoring(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["predictions"][0]["reference_vertical_datum"] = "NAVD88"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="vertical datums must match"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )


def test_campaign_rejects_placeholder_vertical_datum_before_scoring(tmp_path: Path) -> None:
    registry, manifest = _setup_campaign(tmp_path)
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    payload["predictions"][0]["prediction_vertical_datum"] = "unknown"
    payload["predictions"][0]["reference_vertical_datum"] = "unknown"
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must be explicit"):
        evaluate_final_science_campaign(
            registry,
            manifest,
            tmp_path / "out",
            min_valid_pixels=4,
        )
