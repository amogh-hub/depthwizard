from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.check_sih26175_completion as completion

HEAD = "a" * 40


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _metric() -> dict[str, object]:
    return {
        "scenes": 1,
        "valid_pixels": 512,
        "rmse_m": 2.0,
        "mae_m": 1.5,
        "pearson_r": 0.9,
        "mean_bias_m": 0.1,
    }


def _extended_metric() -> dict[str, object]:
    return {
        **_metric(),
        "spearman_r": 0.88,
        "median_abs_error_m": 1.2,
        "nmad_m": 0.8,
        "p90_abs_error_m": 2.8,
        "p95_abs_error_m": 3.2,
    }


def _science_scene(terrain: str, split: str = "test") -> dict[str, object]:
    return {
        "terrain": terrain,
        "split": split,
        "metrics": _extended_metric(),
        "slope_metrics": {
            "valid_pixels": 512,
            "mae_degrees": 1.0,
            "rmse_degrees": 1.5,
            "p95_abs_error_degrees": 2.5,
        },
        "height_range_performance": [
            {"band": name, "metrics": _extended_metric()} for name in ("lower", "middle", "upper")
        ],
    }


def _rt5() -> dict[str, object]:
    return {
        "status": "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE",
        "git_head": HEAD,
        "clean_application_launch": True,
        "user_visible_terminal_required": False,
        "offline_first_reconstruction": True,
        "bundled_model_payload_verified": True,
        "mesh": {"lod_count": 4},
        "export": {"zip_integrity": "PASS"},
        "lifecycle": {"sidecar_terminated_with_app": True},
    }


def _inputs() -> dict[str, object]:
    return {
        "status": "PASS_SIH26175_INPUT_FORMAT_CONTRACT",
        "git_head": HEAD,
        "formats": {
            "png": {"status": "PASS", "elevation_contract": "relative_only_no_metric_claim"},
            "jpg": {"status": "PASS", "elevation_contract": "relative_only_no_metric_claim"},
            "tiff": {
                "status": "PASS",
                "elevation_contract": "georeferenced_metric_calibration_eligible",
            },
        },
    }


def _science() -> dict[str, object]:
    return {
        "protocol": "depthwizard_final_science_campaign_v2",
        "git_head": HEAD,
        "model": {"model_id": "DA3MONO-LARGE", "checkpoint_sha256": "1" * 64},
        "registry": {"sha256": "2" * 64},
        "prediction_manifest": {"sha256": "3" * 64},
        "requirements": {
            "geographic_split_integrity": "passed",
            "reference_independence": "passed",
            "vertical_reference_compatibility": "passed",
            "checkpoint_identity_frozen": "passed",
            "prediction_identity_freeze": "passed",
            "cross_sensor_train_sensor_separation": "passed",
            "cross_sensor_scene_count": 1,
            "test_terrain_coverage": list(completion.REQUIRED_TERRAINS),
        },
        "test_overall": _metric(),
        "cross_sensor_overall": _metric(),
        "terrain": {name: _metric() for name in completion.REQUIRED_TERRAINS},
        "scenes": [
            *[_science_scene(name) for name in completion.REQUIRED_TERRAINS],
            _science_scene("urban", "cross_sensor_test"),
        ],
    }


def _soak() -> dict[str, object]:
    return {
        "status": "PASS_TWO_HOUR_PACKAGED_SOAK",
        "git_head": HEAD,
        "monitored_seconds": 7200.1,
    }


def _baselines() -> dict[str, object]:
    identity = "c" * 64
    mask = "d" * 64
    return {
        "status": "PASS_SAME_INPUT_BASELINES",
        "git_head": HEAD,
        "input_manifest_sha256": identity,
        "evaluation_mask_manifest_sha256": mask,
        "methods": {
            name: {
                "input_manifest_sha256": identity,
                "evaluation_mask_manifest_sha256": mask,
                "metrics": _metric(),
            }
            for name in completion.REQUIRED_BASELINES
        },
        "published_alternatives": [
            {"name": "Metric3D", "status": "not_feasible", "reason": "fixture"}
        ],
    }


def _ablations() -> dict[str, object]:
    identity = "c" * 64
    mask = "d" * 64
    return {
        "status": "PASS_REQUIRED_ABLATIONS",
        "git_head": HEAD,
        "input_manifest_sha256": identity,
        "evaluation_mask_manifest_sha256": mask,
        "experiments": {
            name: {
                "status": "measured",
                "input_manifest_sha256": identity,
                "evaluation_mask_manifest_sha256": mask,
                "baseline": _metric(),
                "ablated": _metric(),
            }
            for name in completion.REQUIRED_ABLATIONS
        },
    }


def _operator() -> dict[str, object]:
    return {
        "status": "PASS_SIH26175_OPERATOR_ACCEPTANCE",
        "git_head": HEAD,
        "checks": {
            name: {"status": "PASS", "evidence": [f"evidence/{name}.png"]}
            for name in completion.REQUIRED_OPERATOR_CHECKS
        },
    }


def _performance() -> dict[str, object]:
    return {
        "status": "PASS_SUSTAINED_3D_PERFORMANCE",
        "git_head": HEAD,
        "duration_seconds": 65.0,
        "sample_count": 65,
        "mean_fps": 34.0,
        "p05_fps": 31.0,
        "navigation_exercised": True,
        "camera_positions": ["aerial", "low"],
        "rendering_modes": ["texture", "analytical_overlay"],
    }


def _clean_machine() -> dict[str, object]:
    return {
        "status": "PASS_CLEAN_MACHINE_STANDALONE",
        "git_head": HEAD,
        "application_sha256": "b" * 64,
        "packaged_app_launch": True,
        "no_user_visible_terminal": True,
        "owned_sidecar_boot": True,
        "offline_first_reconstruction": True,
        "bundled_model_payload_verified": True,
        "end_to_end_reconstruction": True,
        "metric_calibration": True,
        "terrain_3d": True,
        "export_and_reopen": True,
    }


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "rt5_path": _write(tmp_path / "rt5.json", _rt5()),
        "input_path": _write(tmp_path / "inputs.json", _inputs()),
        "science_path": _write(tmp_path / "science.json", _science()),
        "baseline_path": _write(tmp_path / "baselines.json", _baselines()),
        "ablation_path": _write(tmp_path / "ablations.json", _ablations()),
        "soak_path": _write(tmp_path / "soak.json", _soak()),
        "operator_path": _write(tmp_path / "operator.json", _operator()),
        "performance_path": _write(tmp_path / "performance.json", _performance()),
        "clean_machine_path": _write(tmp_path / "clean-machine.json", _clean_machine()),
    }


def test_completion_gate_passes_only_when_every_ps_gate_has_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_docs = []
    for name in ("README.md", "qualification.md", "science.md", "traceability.md"):
        path = tmp_path / name
        path.write_text("evidence\n", encoding="utf-8")
        required_docs.append(path)
    monkeypatch.setattr(completion, "REQUIRED_DOCS", tuple(required_docs))
    monkeypatch.setattr(completion, "ROOT", tmp_path)

    report = completion.evaluate_completion(
        head=HEAD,
        repo_clean=True,
        **_paths(tmp_path),
    )

    assert report["status"] == "PASS_SIH26175_PROBLEM_STATEMENT_COMPLETE"
    assert report["blocking_gates"] == []


def test_completion_gate_rejects_missing_operator_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = []
    for name in ("README.md", "qualification.md", "science.md", "traceability.md"):
        path = tmp_path / name
        path.write_text("evidence\n", encoding="utf-8")
        docs.append(path)
    monkeypatch.setattr(completion, "REQUIRED_DOCS", tuple(docs))
    monkeypatch.setattr(completion, "ROOT", tmp_path)
    paths = _paths(tmp_path)
    operator = _operator()
    checks = operator["checks"]
    assert isinstance(checks, dict)
    checks["structure_height_urban"] = {"status": "PENDING", "evidence": []}
    _write(paths["operator_path"], operator)

    report = completion.evaluate_completion(head=HEAD, repo_clean=True, **paths)

    assert report["status"] == "INCOMPLETE_SIH26175_PROBLEM_STATEMENT"
    blocking_gates = report["blocking_gates"]
    assert isinstance(blocking_gates, list)
    assert "operator_workstation" in blocking_gates


def test_completion_gate_rejects_unreported_correlation(tmp_path: Path) -> None:
    science = _science()
    terrain = science["terrain"]
    assert isinstance(terrain, dict)
    urban = terrain["urban"]
    assert isinstance(urban, dict)
    urban["pearson_r"] = None

    with pytest.raises(completion.CompletionEvidenceError, match="pearson_r must be numeric"):
        completion.check_science(science, HEAD)


def test_completion_gate_rejects_science_from_another_commit() -> None:
    science = _science()
    science["git_head"] = "b" * 40

    with pytest.raises(completion.CompletionEvidenceError, match="current exact head"):
        completion.check_science(science, HEAD)


def test_completion_gate_rejects_baseline_mask_drift() -> None:
    baselines = _baselines()
    methods = baselines["methods"]
    assert isinstance(methods, dict)
    coarse = methods["coarse_dem"]
    assert isinstance(coarse, dict)
    coarse["evaluation_mask_manifest_sha256"] = "e" * 64

    with pytest.raises(completion.CompletionEvidenceError, match="different mask"):
        completion.check_baselines(baselines, HEAD)


def test_completion_gate_requires_explanation_for_unavailable_ablation() -> None:
    ablations = _ablations()
    experiments = ablations["experiments"]
    assert isinstance(experiments, dict)
    experiments["confidence_weighting"] = {"status": "not_applicable", "reason": ""}

    with pytest.raises(completion.CompletionEvidenceError, match="no not-applicable reason"):
        completion.check_ablations(ablations, HEAD)


def test_completion_gate_rejects_ablation_input_drift() -> None:
    ablations = _ablations()
    experiments = ablations["experiments"]
    assert isinstance(experiments, dict)
    experiment = experiments["dem_calibration"]
    assert isinstance(experiment, dict)
    experiment["input_manifest_sha256"] = "e" * 64

    with pytest.raises(completion.CompletionEvidenceError, match="different inputs"):
        completion.check_ablations(ablations, HEAD)


def test_completion_gate_rejects_single_toolbar_fps_sample() -> None:
    performance = _performance()
    performance["duration_seconds"] = 1.0
    performance["sample_count"] = 1

    with pytest.raises(completion.CompletionEvidenceError, match="duration_seconds"):
        completion.check_performance(performance, HEAD)
