from __future__ import annotations

import argparse
import json
import math
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "acceptance" / "sih26175"
DEFAULT_RT5 = (
    ROOT
    / "artifacts"
    / "acceptance"
    / "release-train-5-full"
    / "release-train-5-full-acceptance.json"
)
DEFAULT_INPUTS = (
    ROOT / "artifacts" / "acceptance" / "sih26175-input-formats" / "input-format-contract.json"
)
DEFAULT_SCIENCE = ROOT / "artifacts" / "final-science" / "domain_generalization_report.json"
DEFAULT_BASELINES = ROOT / "artifacts" / "final-science" / "baseline-comparison.json"
DEFAULT_ABLATIONS = ROOT / "artifacts" / "final-science" / "ablation-report.json"
DEFAULT_SOAK = (
    ROOT / "artifacts" / "acceptance" / "release-train-7-soak" / "software_stability_report.json"
)
DEFAULT_OPERATOR = DEFAULT_OUT / "operator-acceptance.json"
DEFAULT_PERFORMANCE = DEFAULT_OUT / "rendering-performance.json"
DEFAULT_CLEAN_MACHINE = DEFAULT_OUT / "clean-machine-standalone.json"
DEFAULT_REPORT = DEFAULT_OUT / "problem-statement-completion.json"

REQUIRED_TERRAINS = ("urban", "sparse", "hilly", "forested")
REQUIRED_BASELINES = ("coarse_dem", "raw_monocular", "simple_affine", "depthwizard_complete")
REQUIRED_ABLATIONS = (
    "dem_calibration",
    "gcp_calibration",
    "confidence_weighting",
    "bias_correction",
    "global_vs_per_tile_normalization",
    "semantic_priors",
    "seam_harmonization",
    "learned_refinement",
    "backbone_substitution",
)
REQUIRED_OPERATOR_CHECKS = (
    "geo_tiff_metric_dsm",
    "nongeo_png_rdsm",
    "nongeo_jpg_rdsm",
    "optical_source_view",
    "dsm_view",
    "texture_projection_3d",
    "dsm_overlay_3d",
    "slope_overlay_3d",
    "hillshade_3d",
    "contours_3d",
    "orbit_navigation",
    "fly_navigation",
    "first_person_navigation",
    "top_down_navigation",
    "deterministic_flythrough",
    "auto_and_manual_lod",
    "vertical_exaggeration",
    "probe_3d",
    "measure_3d",
    "profile_3d",
    "structure_height_urban",
    "reference_validation_independent",
    "reference_view",
    "residual_view",
    "projection_accuracy_visual",
    "export_bundle",
    "relaunch_reopen",
)
REQUIRED_DOCS = (
    ROOT / "README.md",
    ROOT / "docs" / "final-sih-qualification.md",
    ROOT / "docs" / "final-science-campaign.md",
    ROOT / "docs" / "sih26175-problem-statement-traceability.md",
)


class CompletionEvidenceError(ValueError):
    """One evidence artifact exists but violates the SIH26175 completion contract."""


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return not result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CompletionEvidenceError(f"JSON root must be an object: {path}")
    return payload


def _finite_number(value: object, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompletionEvidenceError(f"{context} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise CompletionEvidenceError(f"{context} must be finite")
    if minimum is not None and number < minimum:
        raise CompletionEvidenceError(f"{context} must be >= {minimum}; got {number}")
    return number


def _require_head(payload: dict[str, Any], head: str, *, context: str) -> None:
    if payload.get("git_head") != head:
        raise CompletionEvidenceError(
            f"{context} was not produced by current exact head; "
            f"evidence={payload.get('git_head')!r}, current={head!r}"
        )


def _metric_summary(
    payload: object,
    *,
    context: str,
    require_scene_count: bool = True,
    minimum_valid_pixels: int = 128,
) -> None:
    if not isinstance(payload, dict):
        raise CompletionEvidenceError(f"{context} must be an object")
    scenes = payload.get("scenes")
    valid_pixels = payload.get("valid_pixels")
    if require_scene_count and (not isinstance(scenes, int) or scenes < 1):
        raise CompletionEvidenceError(f"{context}.scenes must be >= 1")
    if not isinstance(valid_pixels, int) or valid_pixels < minimum_valid_pixels:
        raise CompletionEvidenceError(f"{context}.valid_pixels must be >= {minimum_valid_pixels}")
    _finite_number(payload.get("rmse_m"), context=f"{context}.rmse_m", minimum=0.0)
    _finite_number(payload.get("mae_m"), context=f"{context}.mae_m", minimum=0.0)
    pearson = _finite_number(payload.get("pearson_r"), context=f"{context}.pearson_r")
    if not -1.0 <= pearson <= 1.0:
        raise CompletionEvidenceError(f"{context}.pearson_r must be within [-1, 1]")


def _extended_scene_metrics(
    payload: object,
    *,
    context: str,
    minimum_valid_pixels: int = 128,
) -> None:
    _metric_summary(
        payload,
        context=context,
        require_scene_count=False,
        minimum_valid_pixels=minimum_valid_pixels,
    )
    assert isinstance(payload, dict)
    spearman = _finite_number(payload.get("spearman_r"), context=f"{context}.spearman_r")
    if not -1.0 <= spearman <= 1.0:
        raise CompletionEvidenceError(f"{context}.spearman_r must be within [-1, 1]")
    for name in (
        "median_abs_error_m",
        "nmad_m",
        "p90_abs_error_m",
        "p95_abs_error_m",
    ):
        _finite_number(payload.get(name), context=f"{context}.{name}", minimum=0.0)
    _finite_number(payload.get("mean_bias_m"), context=f"{context}.mean_bias_m")


def _slope_metric_summary(payload: object, *, context: str) -> None:
    if not isinstance(payload, dict):
        raise CompletionEvidenceError(f"{context} must be an object")
    valid_pixels = payload.get("valid_pixels")
    if not isinstance(valid_pixels, int) or valid_pixels < 128:
        raise CompletionEvidenceError(f"{context}.valid_pixels must be >= 128")
    for name in ("mae_degrees", "rmse_degrees", "p95_abs_error_degrees"):
        _finite_number(payload.get(name), context=f"{context}.{name}", minimum=0.0)


def check_rt5(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE":
        raise CompletionEvidenceError(
            "RT5 exact-head standalone engineering acceptance is not PASS"
        )
    _require_head(payload, head, context="RT5 acceptance")
    if payload.get("clean_application_launch") is not True:
        raise CompletionEvidenceError("RT5 did not prove a clean application launch")
    if payload.get("user_visible_terminal_required") is not False:
        raise CompletionEvidenceError("RT5 still requires a user-visible terminal")
    if payload.get("offline_first_reconstruction") is not True:
        raise CompletionEvidenceError("RT5 did not prove offline-first reconstruction")
    if payload.get("bundled_model_payload_verified") is not True:
        raise CompletionEvidenceError("RT5 did not verify the bundled model payload")
    mesh = payload.get("mesh")
    if (
        not isinstance(mesh, dict)
        or not isinstance(mesh.get("lod_count"), int)
        or mesh["lod_count"] < 1
    ):
        raise CompletionEvidenceError("RT5 did not prove packaged mesh/LOD generation")
    export = payload.get("export")
    if not isinstance(export, dict) or export.get("zip_integrity") != "PASS":
        raise CompletionEvidenceError("RT5 did not prove packaged export ZIP integrity")
    lifecycle = payload.get("lifecycle")
    if not isinstance(lifecycle, dict) or lifecycle.get("sidecar_terminated_with_app") is not True:
        raise CompletionEvidenceError("RT5 did not prove desktop-owned sidecar shutdown")
    return {"status": "PASS", "claim": "exact-head packaged end-to-end standalone engineering"}


def check_input_formats(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_SIH26175_INPUT_FORMAT_CONTRACT":
        raise CompletionEvidenceError("literal PNG/JPG/TIFF contract is not PASS")
    _require_head(payload, head, context="input-format qualification")
    formats = payload.get("formats")
    if not isinstance(formats, dict):
        raise CompletionEvidenceError("input-format report is missing formats")
    for key in ("png", "jpg", "tiff"):
        item = formats.get(key)
        if not isinstance(item, dict) or item.get("status") != "PASS":
            raise CompletionEvidenceError(f"literal {key.upper()} input is not PASS")
    if formats["png"].get("elevation_contract") != "relative_only_no_metric_claim":
        raise CompletionEvidenceError("PNG path no longer preserves relative-only semantics")
    if formats["jpg"].get("elevation_contract") != "relative_only_no_metric_claim":
        raise CompletionEvidenceError("JPG path no longer preserves relative-only semantics")
    if formats["tiff"].get("elevation_contract") != "georeferenced_metric_calibration_eligible":
        raise CompletionEvidenceError("GeoTIFF path is no longer metric-calibration eligible")
    return {
        "status": "PASS",
        "claim": "literal PNG/JPG/TIFF ingestion and truthful metadata semantics",
    }


def check_science(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("protocol") != "depthwizard_final_science_campaign_v2":
        raise CompletionEvidenceError("final science report uses an unknown protocol")
    _require_head(payload, head, context="final science report")
    model = payload.get("model")
    if not isinstance(model, dict) or not str(model.get("model_id") or "").strip():
        raise CompletionEvidenceError(
            "final science report is missing the evaluated model identity"
        )
    _sha256_identity(model.get("checkpoint_sha256"), context="science.model.checkpoint_sha256")
    registry = payload.get("registry")
    prediction_manifest = payload.get("prediction_manifest")
    if not isinstance(registry, dict) or not isinstance(prediction_manifest, dict):
        raise CompletionEvidenceError("final science report is missing frozen manifest identities")
    _sha256_identity(registry.get("sha256"), context="science.registry.sha256")
    _sha256_identity(
        prediction_manifest.get("sha256"), context="science.prediction_manifest.sha256"
    )
    requirements = payload.get("requirements")
    if not isinstance(requirements, dict):
        raise CompletionEvidenceError("final science report is missing requirements")
    if requirements.get("geographic_split_integrity") != "passed":
        raise CompletionEvidenceError("final science geographic split integrity is not passed")
    if requirements.get("reference_independence") != "passed":
        raise CompletionEvidenceError("final science reference independence is not passed")
    if requirements.get("vertical_reference_compatibility") != "passed":
        raise CompletionEvidenceError(
            "final science vertical-reference compatibility is not passed"
        )
    if requirements.get("checkpoint_identity_frozen") != "passed":
        raise CompletionEvidenceError("final science checkpoint identity was not frozen")
    if requirements.get("prediction_identity_freeze") != "passed":
        raise CompletionEvidenceError("final science prediction identity was not frozen")
    if requirements.get("cross_sensor_train_sensor_separation") != "passed":
        raise CompletionEvidenceError("final science cross-sensor separation is not passed")
    cross_sensor_scene_count = requirements.get("cross_sensor_scene_count")
    if not isinstance(cross_sensor_scene_count, int) or cross_sensor_scene_count < 1:
        raise CompletionEvidenceError("final science requires at least one cross-sensor scene")
    coverage = set(requirements.get("test_terrain_coverage") or [])
    if coverage != set(REQUIRED_TERRAINS):
        raise CompletionEvidenceError(
            f"final science must cover exactly the four required terrains; coverage={sorted(coverage)}"
        )
    terrain = payload.get("terrain")
    if not isinstance(terrain, dict):
        raise CompletionEvidenceError("final science report is missing terrain summaries")
    for terrain_name in REQUIRED_TERRAINS:
        _metric_summary(terrain.get(terrain_name), context=f"science.terrain.{terrain_name}")
    _metric_summary(payload.get("test_overall"), context="science.test_overall")
    _metric_summary(payload.get("cross_sensor_overall"), context="science.cross_sensor_overall")
    scenes = payload.get("scenes")
    if not isinstance(scenes, list) or not scenes:
        raise CompletionEvidenceError("final science report is missing scene-level distributions")
    test_terrains: set[str] = set()
    cross_sensor_scenes = 0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict):
            raise CompletionEvidenceError(f"science.scenes[{index}] must be an object")
        _extended_scene_metrics(scene.get("metrics"), context=f"science.scenes[{index}].metrics")
        _slope_metric_summary(
            scene.get("slope_metrics"), context=f"science.scenes[{index}].slope_metrics"
        )
        height_ranges = scene.get("height_range_performance")
        if not isinstance(height_ranges, list) or {
            item.get("band") for item in height_ranges if isinstance(item, dict)
        } != {
            "lower",
            "middle",
            "upper",
        }:
            raise CompletionEvidenceError(
                f"science.scenes[{index}] must report lower/middle/upper height-range performance"
            )
        for band_index, item in enumerate(height_ranges):
            assert isinstance(item, dict)
            _extended_scene_metrics(
                item.get("metrics"),
                context=f"science.scenes[{index}].height_range_performance[{band_index}].metrics",
                minimum_valid_pixels=1,
            )
        if scene.get("split") == "test" and isinstance(scene.get("terrain"), str):
            test_terrains.add(scene["terrain"])
        if scene.get("split") == "cross_sensor_test":
            cross_sensor_scenes += 1
    if test_terrains != set(REQUIRED_TERRAINS) or cross_sensor_scenes < 1:
        raise CompletionEvidenceError(
            "scene-level science evidence must cover all test terrains and a cross-sensor scene"
        )
    return {
        "status": "PASS",
        "claim": "independent RMSE/MAE/correlation across urban, sparse, hilly and forested landscapes",
    }


def _sha256_identity(value: object, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value.casefold())
    ):
        raise CompletionEvidenceError(f"{context} must be a SHA-256 identity")
    return value.casefold()


def check_baselines(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_SAME_INPUT_BASELINES":
        raise CompletionEvidenceError("same-input baseline comparison is not PASS")
    _require_head(payload, head, context="baseline comparison")
    input_identity = _sha256_identity(
        payload.get("input_manifest_sha256"), context="baselines.input_manifest_sha256"
    )
    mask_identity = _sha256_identity(
        payload.get("evaluation_mask_manifest_sha256"),
        context="baselines.evaluation_mask_manifest_sha256",
    )
    methods = payload.get("methods")
    if not isinstance(methods, dict):
        raise CompletionEvidenceError("baseline comparison is missing methods")
    for method_name in REQUIRED_BASELINES:
        method = methods.get(method_name)
        if not isinstance(method, dict):
            raise CompletionEvidenceError(f"required baseline {method_name!r} is missing")
        if method.get("input_manifest_sha256") != input_identity:
            raise CompletionEvidenceError(f"baseline {method_name!r} used different inputs")
        if method.get("evaluation_mask_manifest_sha256") != mask_identity:
            raise CompletionEvidenceError(f"baseline {method_name!r} used a different mask")
        _metric_summary(method.get("metrics"), context=f"baselines.{method_name}.metrics")
    alternatives = payload.get("published_alternatives")
    if not isinstance(alternatives, list):
        raise CompletionEvidenceError(
            "published alternative baselines must be explicitly documented"
        )
    for index, alternative in enumerate(alternatives):
        if not isinstance(alternative, dict) or alternative.get("status") not in {
            "measured",
            "not_feasible",
        }:
            raise CompletionEvidenceError(
                f"published_alternatives[{index}] must be measured or not_feasible"
            )
        if (
            alternative.get("status") == "not_feasible"
            and not str(alternative.get("reason") or "").strip()
        ):
            raise CompletionEvidenceError(
                f"published_alternatives[{index}] has no feasibility explanation"
            )
        if alternative.get("status") == "measured":
            if alternative.get("input_manifest_sha256") != input_identity:
                raise CompletionEvidenceError(
                    f"published_alternatives[{index}] used different inputs"
                )
            if alternative.get("evaluation_mask_manifest_sha256") != mask_identity:
                raise CompletionEvidenceError(
                    f"published_alternatives[{index}] used a different mask"
                )
            _metric_summary(
                alternative.get("metrics"),
                context=f"published_alternatives[{index}].metrics",
            )
    return {
        "status": "PASS",
        "claim": "coarse DEM, raw monocular, affine and full-system same-input baselines",
    }


def check_ablations(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_REQUIRED_ABLATIONS":
        raise CompletionEvidenceError("required ablation report is not PASS")
    _require_head(payload, head, context="ablation report")
    input_identity = _sha256_identity(
        payload.get("input_manifest_sha256"), context="ablations.input_manifest_sha256"
    )
    mask_identity = _sha256_identity(
        payload.get("evaluation_mask_manifest_sha256"),
        context="ablations.evaluation_mask_manifest_sha256",
    )
    experiments = payload.get("experiments")
    if not isinstance(experiments, dict):
        raise CompletionEvidenceError("ablation report is missing experiments")
    for name in REQUIRED_ABLATIONS:
        experiment = experiments.get(name)
        if not isinstance(experiment, dict):
            raise CompletionEvidenceError(f"required ablation {name!r} is missing")
        status = experiment.get("status")
        if status == "measured":
            if experiment.get("input_manifest_sha256") != input_identity:
                raise CompletionEvidenceError(f"ablation {name!r} used different inputs")
            if experiment.get("evaluation_mask_manifest_sha256") != mask_identity:
                raise CompletionEvidenceError(f"ablation {name!r} used a different mask")
            _metric_summary(experiment.get("baseline"), context=f"ablations.{name}.baseline")
            _metric_summary(experiment.get("ablated"), context=f"ablations.{name}.ablated")
        elif status == "not_applicable":
            if not str(experiment.get("reason") or "").strip():
                raise CompletionEvidenceError(f"ablation {name!r} has no not-applicable reason")
        else:
            raise CompletionEvidenceError(f"ablation {name!r} must be measured or not_applicable")
    return {"status": "PASS", "claim": "required production-path ablations reported"}


def check_soak(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_TWO_HOUR_PACKAGED_SOAK":
        raise CompletionEvidenceError("two-hour packaged stability soak is not PASS")
    _require_head(payload, head, context="RT7 soak")
    _finite_number(
        payload.get("monitored_seconds"), context="soak.monitored_seconds", minimum=7200.0
    )
    return {"status": "PASS", "claim": "two-hour packaged stability"}


def check_operator(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_SIH26175_OPERATOR_ACCEPTANCE":
        raise CompletionEvidenceError("real operator acceptance is not PASS")
    _require_head(payload, head, context="operator acceptance")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise CompletionEvidenceError("operator acceptance is missing checks")
    for name in REQUIRED_OPERATOR_CHECKS:
        item = checks.get(name)
        if not isinstance(item, dict) or item.get("status") != "PASS":
            raise CompletionEvidenceError(f"operator check {name!r} is not PASS")
        evidence = item.get("evidence")
        if (
            not isinstance(evidence, list)
            or not evidence
            or not all(isinstance(entry, str) and entry.strip() for entry in evidence)
        ):
            raise CompletionEvidenceError(
                f"operator check {name!r} has no explicit evidence reference"
            )
    return {"status": "PASS", "claim": "complete human-visible SIH26175 workstation workflow"}


def check_performance(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_SUSTAINED_3D_PERFORMANCE":
        raise CompletionEvidenceError("sustained 3D rendering performance is not PASS")
    _require_head(payload, head, context="rendering performance")
    _finite_number(
        payload.get("duration_seconds"), context="performance.duration_seconds", minimum=60.0
    )
    _finite_number(payload.get("mean_fps"), context="performance.mean_fps", minimum=30.0)
    _finite_number(payload.get("p05_fps"), context="performance.p05_fps", minimum=30.0)
    samples = payload.get("sample_count")
    if not isinstance(samples, int) or samples < 55:
        raise CompletionEvidenceError(
            "sustained FPS evidence requires at least 55 one-second samples"
        )
    if payload.get("navigation_exercised") is not True:
        raise CompletionEvidenceError("FPS evidence did not exercise navigation")
    camera_positions = set(payload.get("camera_positions") or [])
    if not {"aerial", "low"}.issubset(camera_positions):
        raise CompletionEvidenceError("FPS evidence requires both aerial and low-camera positions")
    layers = set(payload.get("rendering_modes") or [])
    if not {"texture", "analytical_overlay"}.issubset(layers):
        raise CompletionEvidenceError(
            "FPS evidence requires texture and analytical-overlay rendering"
        )
    return {"status": "PASS", "claim": ">=30 FPS sustained representative 3D navigation"}


def check_clean_machine(payload: dict[str, Any], head: str) -> dict[str, object]:
    if payload.get("status") != "PASS_CLEAN_MACHINE_STANDALONE":
        raise CompletionEvidenceError("clean-machine standalone qualification is not PASS")
    _require_head(payload, head, context="clean-machine qualification")
    required_true = (
        "packaged_app_launch",
        "no_user_visible_terminal",
        "owned_sidecar_boot",
        "offline_first_reconstruction",
        "bundled_model_payload_verified",
        "end_to_end_reconstruction",
        "metric_calibration",
        "terrain_3d",
        "export_and_reopen",
    )
    for field in required_true:
        if payload.get(field) is not True:
            raise CompletionEvidenceError(f"clean-machine field {field!r} is not true")
    bundle_sha = payload.get("application_sha256")
    if not isinstance(bundle_sha, str) or len(bundle_sha) != 64:
        raise CompletionEvidenceError("clean-machine evidence is missing application SHA-256")
    return {"status": "PASS", "claim": "standalone clean-machine deployment"}


def check_documentation() -> dict[str, object]:
    missing = [str(path.relative_to(ROOT)) for path in REQUIRED_DOCS if not path.is_file()]
    if missing:
        raise CompletionEvidenceError(f"required source/technical documentation missing: {missing}")
    return {
        "status": "PASS",
        "claim": "complete source/technical documentation surface",
        "files": [str(path.relative_to(ROOT)) for path in REQUIRED_DOCS],
    }


def _gate_from_file(
    name: str,
    path: Path,
    checker: Callable[[dict[str, Any]], dict[str, object]],
) -> dict[str, object]:
    if not path.is_file():
        return {"status": "PENDING", "path": str(path), "reason": "evidence file does not exist"}
    try:
        result = checker(_read_json(path))
    except (OSError, json.JSONDecodeError, CompletionEvidenceError) as exc:
        return {"status": "FAIL", "path": str(path), "reason": str(exc)}
    return {**result, "path": str(path)}


def evaluate_completion(
    *,
    head: str,
    repo_clean: bool,
    rt5_path: Path,
    input_path: Path,
    science_path: Path,
    baseline_path: Path,
    ablation_path: Path,
    soak_path: Path,
    operator_path: Path,
    performance_path: Path,
    clean_machine_path: Path,
) -> dict[str, object]:
    gates: dict[str, dict[str, object]] = {
        "repository_clean": {
            "status": "PASS" if repo_clean else "FAIL",
            "claim": "exact source head has no tracked/untracked worktree drift",
            **({} if repo_clean else {"reason": "git status --porcelain is not empty"}),
        },
        "exact_head_packaged_acceptance": _gate_from_file(
            "rt5",
            rt5_path,
            lambda payload: check_rt5(payload, head),
        ),
        "literal_input_formats": _gate_from_file(
            "input_formats",
            input_path,
            lambda payload: check_input_formats(payload, head),
        ),
        "four_terrain_science": _gate_from_file(
            "science", science_path, lambda payload: check_science(payload, head)
        ),
        "same_input_baselines": _gate_from_file(
            "baselines", baseline_path, lambda payload: check_baselines(payload, head)
        ),
        "required_ablations": _gate_from_file(
            "ablations", ablation_path, lambda payload: check_ablations(payload, head)
        ),
        "two_hour_stability": _gate_from_file(
            "soak",
            soak_path,
            lambda payload: check_soak(payload, head),
        ),
        "operator_workstation": _gate_from_file(
            "operator",
            operator_path,
            lambda payload: check_operator(payload, head),
        ),
        "sustained_rendering": _gate_from_file(
            "performance",
            performance_path,
            lambda payload: check_performance(payload, head),
        ),
        "clean_machine_standalone": _gate_from_file(
            "clean_machine",
            clean_machine_path,
            lambda payload: check_clean_machine(payload, head),
        ),
    }
    try:
        gates["source_and_documentation"] = check_documentation()
    except CompletionEvidenceError as exc:
        gates["source_and_documentation"] = {"status": "FAIL", "reason": str(exc)}

    blocking = [name for name, gate in gates.items() if gate.get("status") != "PASS"]
    status = (
        "PASS_SIH26175_PROBLEM_STATEMENT_COMPLETE"
        if not blocking
        else "INCOMPLETE_SIH26175_PROBLEM_STATEMENT"
    )
    return {
        "schema_version": 1,
        "status": status,
        "git_head": head,
        "problem_statement": "SIH26175 DepthWizard - Single-View Height Estimation and 3D Flythrough",
        "completion_rule": (
            "PASS is emitted only when every functional, scientific-validation, rendering/UX, "
            "standalone-stability and documentation gate required by the official problem statement "
            "has explicit evidence. Feature presence alone is never sufficient."
        ),
        "gates": gates,
        "blocking_gates": blocking,
    }


def _operator_template(head: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "PENDING_SIH26175_OPERATOR_ACCEPTANCE",
        "git_head": head,
        "finale_hardware": "FILL_ME",
        "checks": {
            name: {"status": "PENDING", "evidence": [], "notes": ""}
            for name in REQUIRED_OPERATOR_CHECKS
        },
    }


def _performance_template(head: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "PENDING_SUSTAINED_3D_PERFORMANCE",
        "git_head": head,
        "duration_seconds": 0.0,
        "sample_count": 0,
        "mean_fps": 0.0,
        "p05_fps": 0.0,
        "navigation_exercised": False,
        "camera_positions": [],
        "rendering_modes": [],
        "evidence": [],
    }


def _clean_machine_template(head: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "PENDING_CLEAN_MACHINE_STANDALONE",
        "git_head": head,
        "application_sha256": "",
        "packaged_app_launch": False,
        "no_user_visible_terminal": False,
        "owned_sidecar_boot": False,
        "offline_first_reconstruction": False,
        "bundled_model_payload_verified": False,
        "end_to_end_reconstruction": False,
        "metric_calibration": False,
        "terrain_3d": False,
        "export_and_reopen": False,
        "evidence": [],
    }


def initialize_templates(output_dir: Path, *, head: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = {
        output_dir / "operator-acceptance.json": _operator_template(head),
        output_dir / "rendering-performance.json": _performance_template(head),
        output_dir / "clean-machine-standalone.json": _clean_machine_template(head),
    }
    written: list[Path] = []
    for path, payload in templates.items():
        if path.exists():
            continue
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed SIH26175 problem-statement completion gate."
    )
    parser.add_argument("--rt5", type=Path, default=DEFAULT_RT5)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--science", type=Path, default=DEFAULT_SCIENCE)
    parser.add_argument("--baselines", type=Path, default=DEFAULT_BASELINES)
    parser.add_argument("--ablations", type=Path, default=DEFAULT_ABLATIONS)
    parser.add_argument("--soak", type=Path, default=DEFAULT_SOAK)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--performance", type=Path, default=DEFAULT_PERFORMANCE)
    parser.add_argument("--clean-machine", type=Path, default=DEFAULT_CLEAN_MACHINE)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--init-templates",
        action="store_true",
        help="Create pending operator/performance/clean-machine evidence templates without overwriting existing files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless every SIH26175 problem-statement gate passes.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    head = _git_head()
    if args.init_templates:
        written = initialize_templates(DEFAULT_OUT, head=head)
        for path in written:
            print(f"Initialized: {path}")

    report = evaluate_completion(
        head=head,
        repo_clean=_git_clean(),
        rt5_path=args.rt5,
        input_path=args.inputs,
        science_path=args.science,
        baseline_path=args.baselines,
        ablation_path=args.ablations,
        soak_path=args.soak,
        operator_path=args.operator,
        performance_path=args.performance,
        clean_machine_path=args.clean_machine,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and report["status"] != "PASS_SIH26175_PROBLEM_STATEMENT_COMPLETE":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
