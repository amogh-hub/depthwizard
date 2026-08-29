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


def _rt5() -> dict[str, object]:
    return {
        "status": "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE",
        "git_head": HEAD,
        "clean_application_launch": True,
        "user_visible_terminal_required": False,
        "offline_after_model_install": True,
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
        "protocol": "depthwizard_final_science_campaign_v1",
        "requirements": {
            "geographic_split_integrity": "passed",
            "reference_independence": "passed",
            "test_terrain_coverage": list(completion.REQUIRED_TERRAINS),
        },
        "test_overall": _metric(),
        "terrain": {name: _metric() for name in completion.REQUIRED_TERRAINS},
    }


def _soak() -> dict[str, object]:
    return {
        "status": "PASS_TWO_HOUR_PACKAGED_SOAK",
        "git_head": HEAD,
        "monitored_seconds": 7200.1,
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
        "offline_after_model_install": True,
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
        completion.check_science(science)


def test_completion_gate_rejects_single_toolbar_fps_sample() -> None:
    performance = _performance()
    performance["duration_seconds"] = 1.0
    performance["sample_count"] = 1

    with pytest.raises(completion.CompletionEvidenceError, match="duration_seconds"):
        completion.check_performance(performance, HEAD)