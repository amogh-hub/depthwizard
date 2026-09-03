from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from depthwizard.evaluation.building_height import (
    BuildingHeightBenchmarkReport,
    BuildingHeightInstance,
)
from qualification.qualify_tsd_dev_candidate import (
    _blend_weight,
    _qualification_checks,
    _window_starts,
)


def _report(error_m: float, *, count: int = 60) -> BuildingHeightBenchmarkReport:
    instances = tuple(
        BuildingHeightInstance(
            instance_id=index + 1,
            area_m2=100.0,
            reference_height_m=10.0,
            predicted_height_m=10.0 + error_m,
            height_error_m=error_m,
            reference_top_m=50.0,
            predicted_top_m=50.0 + error_m,
            top_error_m=error_m,
            reference_ground_m=40.0,
            predicted_ground_m=40.0,
            ground_error_m=0.0,
            reference_roof_dispersion_m=0.5,
            predicted_roof_dispersion_m=0.5,
            reference_local_height_dispersion_m=0.5,
            predicted_local_height_dispersion_m=0.5,
        )
        for index in range(count)
    )
    errors = np.asarray([item.height_error_m for item in instances], dtype=np.float64)
    absolute = np.abs(errors)
    ids = tuple(item.instance_id for item in instances)
    return BuildingHeightBenchmarkReport(
        eligible_instance_ids=ids,
        evaluated_instance_ids=ids,
        prediction_failure_ids=(),
        skipped_reference_instances=0,
        instances=instances,
        height_mae_m=float(np.mean(absolute)),
        height_rmse_m=float(np.sqrt(np.mean(errors**2))),
        height_bias_m=float(np.mean(errors)),
        height_median_abs_error_m=float(np.median(absolute)),
        height_p90_abs_error_m=float(np.percentile(absolute, 90)),
        height_p95_abs_error_m=float(np.percentile(absolute, 95)),
        top_mae_m=float(np.mean(absolute)),
        ground_mae_m=0.0,
        within_1m_fraction=float(np.mean(absolute <= 1.0)),
        within_2m_fraction=float(np.mean(absolute <= 2.0)),
        catastrophic_over_3m_fraction=float(np.mean(absolute > 3.0)),
        mean_reference_height_m=10.0,
        mean_predicted_height_m=10.0 + error_m,
    )


def test_window_starts_cover_native_scene_edges() -> None:
    starts = _window_starts(6000, 512, 64)
    assert starts[0] == 0
    assert starts[-1] == 6000 - 512
    assert all(later > earlier for earlier, later in zip(starts, starts[1:], strict=False))


def test_blend_weight_never_leaves_zero_coverage() -> None:
    weight = _blend_weight(512)
    assert weight.shape == (512, 512)
    assert np.all(np.isfinite(weight))
    assert float(np.min(weight)) > 0.0


def test_frozen_dev_gate_requires_material_tall_improvement() -> None:
    baseline = _report(4.0)
    candidate = _report(2.0)
    checks, tall, promotion = _qualification_checks(
        baseline_buildings=baseline,
        candidate_buildings=candidate,
        baseline_valid={"pixels": 1000, "mae_m": 3.0, "rmse_m": 4.0},
        candidate_valid={"pixels": 1000, "mae_m": 2.8, "rmse_m": 3.8},
        baseline_building_surface={"pixels": 500, "mae_m": 4.0, "rmse_m": 4.5},
        candidate_building_surface={"pixels": 500, "mae_m": 3.5, "rmse_m": 4.0},
        per_tile_mae_degradation=[-0.5] * 5,
    )
    assert promotion.passed is True
    assert tall["count"] == 60
    assert tall["mae_reduction_fraction"] == 0.5
    assert all(bool(item["passed"]) for item in checks)


def test_frozen_dev_gate_fails_without_tall_improvement() -> None:
    baseline = _report(4.0)
    candidate = _report(3.8)
    checks, tall, _ = _qualification_checks(
        baseline_buildings=baseline,
        candidate_buildings=candidate,
        baseline_valid={"pixels": 1000, "mae_m": 3.0, "rmse_m": 4.0},
        candidate_valid={"pixels": 1000, "mae_m": 3.0, "rmse_m": 4.0},
        baseline_building_surface={"pixels": 500, "mae_m": 4.0, "rmse_m": 4.5},
        candidate_building_surface={"pixels": 500, "mae_m": 4.0, "rmse_m": 4.5},
        per_tile_mae_degradation=[-0.05] * 5,
    )
    tall_check = next(item for item in checks if item["name"] == "tall_building_mae_reduction")
    assert float(tall["mae_reduction_fraction"]) < 0.10
    assert tall_check["passed"] is False


def test_dev_qualifier_can_be_launched_as_a_file() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification" / "qualify_tsd_dev_candidate.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Qualify the frozen epoch-2 TSD candidate" in result.stdout
