from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.evaluation.building_height import evaluate_building_height_instances
from qualification.compare_urban_building_height import _validate_report_integrity
from qualification.evaluate_urban_building_height import _prepare_evaluation_inputs
from qualification.run_exposed_potsdam_2_14_diagnosis import _resolve_semantic_label


def _two_building_scene() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (160, 160)
    reference = np.full(shape, 100.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    first = (slice(25, 45), slice(25, 45))
    second = (slice(95, 115), slice(95, 115))
    buildings[first] = True
    buildings[second] = True
    reference[first] += 8.0
    reference[second] += 12.0
    return reference, buildings, ~buildings


def _evaluate_valid_report():
    reference, buildings, strict_ground = _two_building_scene()
    return evaluate_building_height_instances(
        reference.copy(),
        reference,
        buildings,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        ground_candidate_mask=strict_ground,
        min_building_area_m2=20.0,
        min_reference_height_m=2.0,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=32,
    )


def _write_reference_and_label(root: Path, *, label_name: str) -> tuple[Path, Path]:
    transform = from_origin(500000.0, 5800000.0, 0.05, 0.05)
    reference = root / "dsm_potsdam_02_14.tif"
    label = root / label_name
    profile = {
        "driver": "GTiff",
        "height": 12,
        "width": 12,
        "crs": "EPSG:32633",
        "transform": transform,
    }
    with rasterio.open(reference, "w", count=1, dtype="float32", **profile) as dst:
        dst.write(np.full((12, 12), 100.0, dtype=np.float32), 1)

    rgb = np.full((3, 12, 12), 255, dtype=np.uint8)
    rgb[:, 3:9, 3:9] = np.asarray([0, 0, 255], dtype=np.uint8)[:, None, None]
    with rasterio.open(label, "w", count=3, dtype="uint8", **profile) as dst:
        dst.write(rgb)
    return reference, label


def test_candidate_nodata_cannot_remove_reference_eligible_building() -> None:
    reference, buildings, strict_ground = _two_building_scene()
    prediction = reference.copy()
    second = (slice(95, 115), slice(95, 115))

    reference_valid = np.ones(reference.shape, dtype=bool)
    prediction_valid = np.ones(reference.shape, dtype=bool)
    prediction_valid[second] = False

    prepared_prediction, prepared_reference, prepared_buildings, prepared_ground = (
        _prepare_evaluation_inputs(
            prediction,
            prediction_valid,
            reference,
            reference_valid,
            buildings,
            strict_ground,
        )
    )

    assert np.all(np.isfinite(prepared_reference[second]))
    assert np.all(~np.isfinite(prepared_prediction[second]))
    assert np.all(prepared_buildings[second])
    assert prepared_ground is not None

    report = evaluate_building_height_instances(
        prepared_prediction,
        prepared_reference,
        prepared_buildings,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        ground_candidate_mask=prepared_ground,
        min_building_area_m2=20.0,
        min_reference_height_m=2.0,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=32,
    )

    assert len(report.eligible_instance_ids) == 2
    assert len(report.evaluated_instance_ids) == 1
    assert len(report.prediction_failure_ids) == 1
    assert report.prediction_failure_ids[0] in report.eligible_instance_ids


def test_persisted_promotion_evidence_rejects_nonfinite_metric() -> None:
    report = _evaluate_valid_report()
    corrupted = replace(report, height_mae_m=float("nan"))

    with pytest.raises(ValueError, match="non-finite height_mae_m"):
        _validate_report_integrity(corrupted, role="candidate")


def test_persisted_promotion_evidence_rejects_inconsistent_instance_support() -> None:
    report = _evaluate_valid_report()
    corrupted = replace(report, prediction_failure_ids=(999,))

    with pytest.raises(ValueError, match="eligibility is inconsistent"):
        _validate_report_integrity(corrupted, role="candidate")


def test_exposed_semantic_label_auto_resolution_accepts_one_valid_candidate(tmp_path: Path) -> None:
    reference, expected = _write_reference_and_label(
        tmp_path,
        label_name="top_potsdam_2_14_label.tif",
    )
    (tmp_path / "top_potsdam_2_14_RGB.tif").write_bytes(b"not-a-label")

    resolved, method = _resolve_semantic_label(
        explicit=None,
        dataset_root=tmp_path,
        reference=reference,
    )

    assert resolved == expected
    assert method == "dataset_root_unique_palette_exact_grid"


def test_exposed_semantic_label_auto_resolution_fails_closed_on_ambiguity(tmp_path: Path) -> None:
    reference, _ = _write_reference_and_label(
        tmp_path,
        label_name="top_potsdam_2_14_label.tif",
    )
    _, second = _write_reference_and_label(
        tmp_path,
        label_name="top_potsdam_2_14_label_copy.tif",
    )
    assert second.is_file()

    with pytest.raises(RuntimeError, match="ambiguous exposed semantic labels"):
        _resolve_semantic_label(
            explicit=None,
            dataset_root=tmp_path,
            reference=reference,
        )
