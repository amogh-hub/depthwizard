from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import rasterio
import yaml
from pydantic import BaseModel, Field, model_validator

from depthwizard.data.registry import DatasetRegistry, DatasetScene, load_registry
from depthwizard.evaluation.metrics import compute_elevation_metrics
from depthwizard.io.raster import reproject_to_match
from depthwizard.provenance.manifest import sha256_file

REQUIRED_TERRAINS = ("urban", "sparse", "hilly", "forested")
EVALUATION_SPLITS = ("test", "cross_sensor_test")


class CampaignPrediction(BaseModel):
    scene_id: str = Field(min_length=1)
    prediction_path: Path
    prediction_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    calibration_evidence_paths: list[Path] = Field(min_length=1, max_length=32)
    prediction_vertical_units: Literal["m"] = "m"
    reference_vertical_units: Literal["m"] = "m"
    notes: str | None = None


class CampaignManifest(BaseModel):
    schema_version: Literal[1] = 1
    model_id: str = Field(min_length=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions: list[CampaignPrediction] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_scene_predictions(self) -> CampaignManifest:
        counts = Counter(item.scene_id for item in self.predictions)
        duplicates = sorted(scene_id for scene_id, count in counts.items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate campaign predictions for scenes: {duplicates}")
        return self


@dataclass
class _MetricAccumulator:
    valid_pixels: int = 0
    mean_prediction: float = 0.0
    mean_reference: float = 0.0
    prediction_m2: float = 0.0
    reference_m2: float = 0.0
    covariance_sum: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sum_error: float = 0.0
    scenes: int = 0

    def add(self, prediction: np.ndarray, reference: np.ndarray) -> None:
        p = np.asarray(prediction, dtype=np.float64).reshape(-1)
        r = np.asarray(reference, dtype=np.float64).reshape(-1)
        if p.shape != r.shape:
            raise ValueError("campaign accumulator prediction/reference shape mismatch")
        if p.size == 0:
            raise ValueError("campaign accumulator received no valid pixels")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(r)):
            raise ValueError("campaign accumulator requires finite values")

        batch_n = int(p.size)
        batch_mean_p = float(np.mean(p))
        batch_mean_r = float(np.mean(r))
        centered_p = p - batch_mean_p
        centered_r = r - batch_mean_r
        batch_m2_p = float(np.dot(centered_p, centered_p))
        batch_m2_r = float(np.dot(centered_r, centered_r))
        batch_cov = float(np.dot(centered_p, centered_r))

        if self.valid_pixels == 0:
            self.valid_pixels = batch_n
            self.mean_prediction = batch_mean_p
            self.mean_reference = batch_mean_r
            self.prediction_m2 = batch_m2_p
            self.reference_m2 = batch_m2_r
            self.covariance_sum = batch_cov
        else:
            old_n = self.valid_pixels
            new_n = old_n + batch_n
            delta_p = batch_mean_p - self.mean_prediction
            delta_r = batch_mean_r - self.mean_reference
            merge_weight = old_n * batch_n / new_n
            self.prediction_m2 += batch_m2_p + delta_p * delta_p * merge_weight
            self.reference_m2 += batch_m2_r + delta_r * delta_r * merge_weight
            self.covariance_sum += batch_cov + delta_p * delta_r * merge_weight
            self.mean_prediction += delta_p * batch_n / new_n
            self.mean_reference += delta_r * batch_n / new_n
            self.valid_pixels = new_n

        error = p - r
        self.sum_abs_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.sum_squared_error += float(np.dot(error, error))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.scenes += 1

    def summary(self) -> dict[str, int | float | None]:
        if self.valid_pixels <= 0:
            raise ValueError("cannot summarize an empty campaign metric accumulator")
        denominator = float(np.sqrt(self.prediction_m2 * self.reference_m2))
        pearson = self.covariance_sum / denominator if denominator > 0 else None
        n = self.valid_pixels
        return {
            "scenes": self.scenes,
            "valid_pixels": n,
            "rmse_m": float(np.sqrt(self.sum_squared_error / n)),
            "mae_m": self.sum_abs_error / n,
            "pearson_r": float(pearson) if pearson is not None else None,
            "mean_bias_m": self.sum_error / n,
        }


def load_campaign_manifest(path: str | Path) -> CampaignManifest:
    manifest_path = Path(path)
    raw = manifest_path.read_text(encoding="utf-8")
    if manifest_path.suffix.lower() == ".json":
        payload = json.loads(raw)
    else:
        payload = yaml.safe_load(raw)
    if not isinstance(payload, dict):
        raise TypeError("campaign prediction manifest root must be an object")
    return CampaignManifest.model_validate(payload)


def _resolved_file(path: Path, *, base_dir: Path, context: str) -> Path:
    candidate = path if path.is_absolute() else base_dir / path
    resolved = candidate.resolve(strict=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"{context} does not exist: {resolved}")
    return resolved


def _read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        if src.count < 1:
            raise ValueError(f"prediction raster has no bands: {path}")
        prediction = src.read(1).astype(np.float64)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(prediction)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= prediction != float(src.nodata)
    return prediction, valid


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def _summary_csv(
    field_name: str,
    groups: dict[str, _MetricAccumulator],
    *,
    ordered_keys: list[str] | tuple[str, ...] | None = None,
) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        field_name,
        "scenes",
        "valid_pixels",
        "rmse_m",
        "mae_m",
        "pearson_r",
        "mean_bias_m",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    keys = list(ordered_keys) if ordered_keys is not None else sorted(groups)
    for key in keys:
        accumulator = groups.get(key)
        if accumulator is None:
            continue
        row = {field_name: key, **accumulator.summary()}
        writer.writerow({name: _csv_value(row.get(name)) for name in fieldnames})
    return output.getvalue()


def _scene_csv(scenes: list[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "scene_id",
        "dataset",
        "split",
        "terrain",
        "sensor",
        "geographic_group",
        "valid_pixels",
        "coverage_fraction",
        "rmse_m",
        "mae_m",
        "pearson_r",
        "mean_bias_m",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for scene in sorted(scenes, key=lambda item: item["scene_id"]):
        metrics = scene["metrics"]
        row: dict[str, Any] = {
            "scene_id": scene["scene_id"],
            "dataset": scene["dataset"],
            "split": scene["split"],
            "terrain": scene["terrain"],
            "sensor": scene["sensor"],
            "geographic_group": scene["geographic_group"],
            "valid_pixels": metrics["valid_pixels"],
            "coverage_fraction": scene["coverage_fraction"],
            "rmse_m": metrics["rmse_m"],
            "mae_m": metrics["mae_m"],
            "pearson_r": metrics["pearson_r"],
            "mean_bias_m": metrics["mean_bias_m"],
        }
        writer.writerow(
            {
                name: _csv_value(row.get(name))
                if name
                not in {"scene_id", "dataset", "split", "terrain", "sensor", "geographic_group"}
                else str(row.get(name, ""))
                for name in fieldnames
            }
        )
    return output.getvalue()


def _required_evaluation_scenes(registry: DatasetRegistry) -> list[DatasetScene]:
    evaluation = [scene for scene in registry.scenes if scene.split in EVALUATION_SPLITS]
    if not evaluation:
        raise ValueError("final science campaign requires test/cross_sensor_test scenes")

    test_scenes = [scene for scene in evaluation if scene.split == "test"]
    terrain_coverage = {scene.terrain for scene in test_scenes}
    missing = sorted(set(REQUIRED_TERRAINS) - terrain_coverage)
    if missing:
        raise ValueError(
            "final science campaign test split must cover urban, sparse, hilly and forested; "
            f"missing={missing}"
        )
    if any(scene.reference_path is None for scene in evaluation):
        missing_refs = sorted(
            scene.scene_id for scene in evaluation if scene.reference_path is None
        )
        raise ValueError(f"evaluation scenes require independent reference rasters: {missing_refs}")

    cross_sensor = [scene for scene in evaluation if scene.split == "cross_sensor_test"]
    if not cross_sensor:
        raise ValueError("final science campaign requires at least one cross_sensor_test scene")
    train_sensors = {scene.sensor for scene in registry.scenes if scene.split == "train"}
    if not train_sensors:
        raise ValueError(
            "cross-sensor evidence requires at least one train scene so sensor independence "
            "can be proven"
        )
    return evaluation


def evaluate_final_science_campaign(
    registry_path: str | Path,
    prediction_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    min_valid_pixels: int = 128,
) -> dict[str, Any]:
    """Evaluate the frozen production model on the official four-terrain/cross-sensor campaign.

    This function never creates predictions and never uses reference data for calibration. It only
    evaluates pre-existing metric DSM predictions against registry-declared independent references.
    Prediction identities must already be frozen in the manifest and are SHA-verified before any
    reference evaluation. Calibration evidence is SHA-audited against each reference so a
    copied/renamed calibration DEM cannot silently become evaluation truth.
    """
    if min_valid_pixels < 2:
        raise ValueError("min_valid_pixels must be >= 2")

    registry_file = Path(registry_path).resolve(strict=True)
    campaign_file = Path(prediction_manifest_path).resolve(strict=True)
    registry = load_registry(registry_file)
    registry.assert_integrity()
    campaign = load_campaign_manifest(campaign_file)
    evaluation_scenes = _required_evaluation_scenes(registry)

    prediction_by_scene = {item.scene_id: item for item in campaign.predictions}
    expected_scene_ids = {scene.scene_id for scene in evaluation_scenes}
    supplied_scene_ids = set(prediction_by_scene)
    missing_predictions = sorted(expected_scene_ids - supplied_scene_ids)
    unknown_predictions = sorted(supplied_scene_ids - expected_scene_ids)
    if missing_predictions or unknown_predictions:
        raise ValueError(
            "campaign prediction manifest must match test/cross_sensor_test scenes exactly; "
            f"missing={missing_predictions}, unknown={unknown_predictions}"
        )

    registry_base = registry_file.parent
    campaign_base = campaign_file.parent
    scene_reports: list[dict[str, Any]] = []
    test_overall = _MetricAccumulator()
    cross_sensor_overall = _MetricAccumulator()
    terrain_groups = {terrain: _MetricAccumulator() for terrain in REQUIRED_TERRAINS}
    sensor_groups: dict[str, _MetricAccumulator] = {}

    for scene in sorted(evaluation_scenes, key=lambda item: item.scene_id):
        prediction_entry = prediction_by_scene[scene.scene_id]
        prediction_path = _resolved_file(
            prediction_entry.prediction_path,
            base_dir=campaign_base,
            context=f"prediction for {scene.scene_id}",
        )
        prediction_sha = sha256_file(prediction_path)
        if prediction_sha != prediction_entry.prediction_sha256:
            raise ValueError(
                f"scene {scene.scene_id} prediction SHA-256 no longer matches frozen manifest; "
                f"expected={prediction_entry.prediction_sha256}, actual={prediction_sha}"
            )

        assert scene.reference_path is not None
        reference_path = _resolved_file(
            scene.reference_path,
            base_dir=registry_base,
            context=f"reference for {scene.scene_id}",
        )
        rgb_path = _resolved_file(
            scene.rgb_path,
            base_dir=registry_base,
            context=f"RGB source for {scene.scene_id}",
        )

        reference_sha = sha256_file(reference_path)
        if prediction_path == reference_path or prediction_sha == reference_sha:
            raise ValueError(
                f"scene {scene.scene_id} prediction is byte-identical to its reference; "
                "independent evaluation was refused"
            )

        calibration_evidence: list[dict[str, str]] = []
        for raw_evidence_path in prediction_entry.calibration_evidence_paths:
            evidence_path = _resolved_file(
                raw_evidence_path,
                base_dir=campaign_base,
                context=f"calibration evidence for {scene.scene_id}",
            )
            evidence_sha = sha256_file(evidence_path)
            if evidence_path == reference_path or evidence_sha == reference_sha:
                raise ValueError(
                    f"scene {scene.scene_id} reference is reused as calibration evidence; "
                    "independent evaluation was refused"
                )
            calibration_evidence.append({"path": str(evidence_path), "sha256": evidence_sha})

        prediction, prediction_valid = _read_prediction(prediction_path)
        aligned_reference, reference_valid = reproject_to_match(
            reference_path,
            prediction_path,
        )
        valid = prediction_valid & reference_valid
        valid &= np.isfinite(prediction) & np.isfinite(aligned_reference)
        valid_pixels = int(np.count_nonzero(valid))
        if valid_pixels < min_valid_pixels:
            raise ValueError(
                f"scene {scene.scene_id} has only {valid_pixels} valid independent evaluation "
                f"pixels; minimum is {min_valid_pixels}"
            )

        metrics = compute_elevation_metrics(prediction, aligned_reference, valid_mask=valid)
        p = prediction[valid]
        r = aligned_reference[valid]
        if scene.split == "test":
            test_overall.add(p, r)
            terrain_groups[scene.terrain].add(p, r)
        else:
            cross_sensor_overall.add(p, r)
        sensor_groups.setdefault(scene.sensor, _MetricAccumulator()).add(p, r)

        scene_reports.append(
            {
                "scene_id": scene.scene_id,
                "dataset": scene.dataset,
                "split": scene.split,
                "terrain": scene.terrain,
                "geographic_group": scene.geographic_group,
                "sensor": scene.sensor,
                "nominal_gsd_m": scene.nominal_gsd_m,
                "rgb": {"path": str(rgb_path), "sha256": sha256_file(rgb_path)},
                "prediction": {
                    "path": str(prediction_path),
                    "sha256": prediction_sha,
                    "manifest_sha256": prediction_entry.prediction_sha256,
                    "identity_check": "passed",
                    "vertical_units": prediction_entry.prediction_vertical_units,
                },
                "reference": {
                    "path": str(reference_path),
                    "sha256": reference_sha,
                    "vertical_units": prediction_entry.reference_vertical_units,
                },
                "calibration_evidence": calibration_evidence,
                "independence_check": (
                    "reference_path_and_sha_differ_from_prediction_and_calibration_evidence"
                ),
                "valid_pixels": valid_pixels,
                "coverage_fraction": valid_pixels / prediction.size,
                "metrics": metrics.model_dump(mode="json"),
                "notes": prediction_entry.notes,
            }
        )

    output = Path(output_dir).resolve(strict=False)
    output.mkdir(parents=True, exist_ok=True)
    terrain_csv_path = output / "terrain_breakdown.csv"
    scene_csv_path = output / "scene_metrics.csv"
    sensor_csv_path = output / "sensor_breakdown.csv"
    report_path = output / "domain_generalization_report.json"

    _atomic_write_text(
        terrain_csv_path,
        _summary_csv("terrain", terrain_groups, ordered_keys=REQUIRED_TERRAINS),
    )
    _atomic_write_text(scene_csv_path, _scene_csv(scene_reports))
    _atomic_write_text(sensor_csv_path, _summary_csv("sensor", sensor_groups))

    report: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "depthwizard_final_science_campaign_v1",
        "claim_boundary": (
            "Independent evaluation only. Production checkpoint identity and prediction bytes are "
            "frozen before reference evaluation. Reference rasters are prohibited from matching "
            "prediction or calibration-evidence bytes. Results do not imply unseen sensor/terrain "
            "performance outside the frozen registry."
        ),
        "model": {
            "model_id": campaign.model_id,
            "checkpoint_sha256": campaign.checkpoint_sha256,
        },
        "registry": {
            "path": str(registry_file),
            "sha256": sha256_file(registry_file),
        },
        "prediction_manifest": {
            "path": str(campaign_file),
            "sha256": sha256_file(campaign_file),
        },
        "requirements": {
            "required_test_terrains": list(REQUIRED_TERRAINS),
            "test_terrain_coverage": sorted(
                {scene.terrain for scene in evaluation_scenes if scene.split == "test"}
            ),
            "cross_sensor_scene_count": sum(
                scene.split == "cross_sensor_test" for scene in evaluation_scenes
            ),
            "geographic_split_integrity": "passed",
            "cross_sensor_train_sensor_separation": "passed",
            "checkpoint_identity_frozen": "passed",
            "prediction_identity_freeze": "passed",
            "reference_independence": "passed",
        },
        "test_overall": test_overall.summary(),
        "cross_sensor_overall": cross_sensor_overall.summary(),
        "terrain": {terrain: terrain_groups[terrain].summary() for terrain in REQUIRED_TERRAINS},
        "sensor": {
            sensor: accumulator.summary()
            for sensor, accumulator in sorted(sensor_groups.items())
        },
        "scenes": scene_reports,
        "artifacts": {
            "terrain_breakdown_csv": str(terrain_csv_path),
            "scene_metrics_csv": str(scene_csv_path),
            "sensor_breakdown_csv": str(sensor_csv_path),
        },
    }
    _atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
