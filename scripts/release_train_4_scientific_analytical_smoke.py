from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter

from depthwizard.analysis.project_structure import estimate_project_structure_height
from depthwizard.calibration.gcp_io import inspect_ground_control_point_file
from depthwizard.contracts import (
    GroundControlPoint,
    GroundControlPointEvidence,
    NormalizedPoint,
    ProcessingRequest,
    ProjectRunStatus,
    ProjectStructureHeightRequest,
)
from depthwizard.geometry_prior.base import GeometryPrior, GeometryPriorOutput
from depthwizard.pipeline.policy import EstimatorPath, current_production_estimator_decision
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.raster_preview import render_project_layer_preview

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-4-analytical"
SOURCE_PATH = OUTPUT_DIR / "inputs" / "synthetic_rgb.tif"
DEM_PATH = OUTPUT_DIR / "inputs" / "coarse_dem.tif"
GCP_PATH = OUTPUT_DIR / "inputs" / "control_points.csv"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _relative_surface(shape: tuple[int, int]) -> np.ndarray:
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    terrain = 0.003 * cols + 0.002 * rows
    building = np.zeros(shape, dtype=np.float64)
    building[24:40, 24:40] = 0.4
    return (terrain + building).astype(np.float32)


def _metric_truth(shape: tuple[int, int]) -> np.ndarray:
    return (120.0 + 40.0 * _relative_surface(shape)).astype(np.float32)


class AnalyticPrior(GeometryPrior):
    """Deterministic acceptance surrogate; it is never production/promotion evidence."""

    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        shape = (int(rgb_normalized.shape[0]), int(rgb_normalized.shape[1]))
        return GeometryPriorOutput(
            relative_height=_relative_surface(shape),
            confidence=None,
            model_id="RT4-ANALYTIC-PRIOR-NOT-PRODUCTION",
            metadata={"purpose": "release_train_4_engineering_acceptance"},
        )


class FlatPrior(GeometryPrior):
    def infer(self, rgb_normalized: np.ndarray) -> GeometryPriorOutput:
        shape = (int(rgb_normalized.shape[0]), int(rgb_normalized.shape[1]))
        return GeometryPriorOutput(
            relative_height=np.zeros(shape, dtype=np.float32),
            confidence=None,
            model_id="RT4-FLAT-NEGATIVE-CONTROL",
            metadata={"purpose": "weak_evidence_abort_test"},
        )


def _write_inputs() -> None:
    inputs = SOURCE_PATH.parent
    inputs.mkdir(parents=True, exist_ok=True)
    shape = (64, 64)
    rows, cols = np.mgrid[: shape[0], : shape[1]]
    rgb = np.stack(
        [
            np.clip(30 + 2 * cols, 0, 255),
            np.clip(50 + 2 * rows, 0, 255),
            np.clip(70 + cols + rows, 0, 255),
        ],
        axis=0,
    ).astype(np.uint8)
    with rasterio.open(
        SOURCE_PATH,
        "w",
        driver="GTiff",
        height=shape[0],
        width=shape[1],
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
    ) as dst:
        dst.write(rgb)

    truth = _metric_truth(shape)
    coarse = truth.reshape(8, 8, 8, 8).mean(axis=(1, 3)).astype(np.float32)
    with rasterio.open(
        DEM_PATH,
        "w",
        driver="GTiff",
        height=8,
        width=8,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 8.0, 8.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(coarse, 1)

    transform = from_origin(500000.0, 1400000.0, 1.0, 1.0)
    points = [(5, 5), (8, 52), (52, 8), (56, 56), (18, 45), (46, 20)]
    lines = ["x,y,elevation_m,weight"]
    for row, col in points:
        x, y = transform * (col + 0.5, row + 0.5)
        lines.append(f"{x:.6f},{y:.6f},{truth[row, col]:.6f},1.0")
    GCP_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read_surface(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        if src.nodata is not None:
            values[values == np.float32(src.nodata)] = np.nan
        return values


def _load_calibration(project_dir: Path) -> dict[str, object]:
    path = project_dir / "calibration.json"
    if not path.is_file():
        raise RuntimeError(f"calibration evidence is missing: {project_dir}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("calibration evidence root must be an object")
    return payload


def _run_project(name: str, request: ProcessingRequest) -> tuple[ProjectManifest, dict[str, object]]:
    result = ProductionElevationRuntime(prior=AnalyticPrior()).run(request, job_id=f"rt4-{name}")
    if result.status is not ProjectRunStatus.COMPLETE:
        raise RuntimeError(f"{name} project did not complete: {result.status.value}")
    manifest = ProjectManifest.load(request.output_dir)
    calibration = _load_calibration(request.output_dir)
    if manifest.artifact_path("confidence") is not None:
        raise RuntimeError("acceptance surrogate emitted no confidence, but a confidence artifact was fabricated")
    return manifest, calibration


def _metric_error(manifest: ProjectManifest) -> dict[str, float]:
    dsm_path = manifest.artifact_path("dsm")
    if dsm_path is None or not dsm_path.is_file():
        raise RuntimeError("metric project has no DSM artifact")
    prediction = _read_surface(dsm_path)
    truth = _metric_truth(prediction.shape)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    residual = prediction[valid] - truth[valid]
    return {
        "mae_m": float(np.mean(np.abs(residual))),
        "rmse_m": float(np.sqrt(np.mean(residual**2))),
        "max_abs_error_m": float(np.max(np.abs(residual))),
    }


def _high_frequency_preservation(manifest: ProjectManifest) -> dict[str, float | bool]:
    dsm_path = manifest.artifact_path("dsm")
    if dsm_path is None:
        raise RuntimeError("DSM missing for high-frequency preservation check")
    prediction = _read_surface(dsm_path).astype(np.float64)
    raw_metric_structure = 40.0 * _relative_surface(prediction.shape).astype(np.float64)
    pred_high = prediction - gaussian_filter(prediction, sigma=3.0)
    raw_high = raw_metric_structure - gaussian_filter(raw_metric_structure, sigma=3.0)
    valid = np.isfinite(pred_high) & np.isfinite(raw_high)
    correlation = float(np.corrcoef(pred_high[valid], raw_high[valid])[0, 1])
    return {
        "high_frequency_pearson_r": correlation,
        "preserved": bool(np.isfinite(correlation) and correlation >= 0.98),
    }


def _preview_report(project_dir: Path) -> list[dict[str, object]]:
    reports: list[dict[str, object]] = []
    for layer in ("dsm", "slope", "hillshade", "contours"):
        payload = render_project_layer_preview(project_dir, layer, max_side=512)
        if not payload.startswith(PNG_SIGNATURE):
            raise RuntimeError(f"{layer} preview did not render as PNG")
        reports.append(
            {
                "layer": layer,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return reports


def _structure_report(project_dir: Path) -> dict[str, object]:
    width = height = 64
    polygon = [
        NormalizedPoint(x=24 / (width - 1), y=24 / (height - 1)),
        NormalizedPoint(x=39 / (width - 1), y=24 / (height - 1)),
        NormalizedPoint(x=39 / (width - 1), y=39 / (height - 1)),
        NormalizedPoint(x=24 / (width - 1), y=39 / (height - 1)),
    ]
    result = estimate_project_structure_height(
        ProjectStructureHeightRequest(
            project_dir=project_dir,
            polygon=polygon,
            ring_pixels=6,
            min_structure_pixels=16,
            min_ground_pixels=16,
        )
    )
    if not 14.0 <= result.structure_height_m <= 18.0:
        raise RuntimeError(
            "structural-height acceptance escaped the expected synthetic building range: "
            f"{result.structure_height_m:.3f} m"
        )
    return result.model_dump(mode="json")


def _weak_evidence_abort(gcps: list[GroundControlPoint]) -> dict[str, object]:
    project_dir = OUTPUT_DIR / "weak-gcp-project"
    try:
        ProductionElevationRuntime(prior=FlatPrior()).run(
            ProcessingRequest(
                source=SOURCE_PATH,
                output_dir=project_dir,
                gcps=gcps,
                requested_output="dsm",
            ),
            job_id="rt4-weak-gcp",
        )
    except ValueError as exc:
        manifest = ProjectManifest.load(project_dir)
        if manifest.status != ProjectRunStatus.FAILED.value:
            raise RuntimeError("weak GCP evidence raised but project manifest was not marked failed") from exc
        return {
            "aborted": True,
            "error": str(exc),
            "project_status": manifest.status,
        }
    raise RuntimeError("weak/underdetermined GCP evidence unexpectedly produced a metric DSM")


def _mutation_rejection(report_points: list[GroundControlPoint], inspected_sha: str) -> dict[str, object]:
    mutation_path = OUTPUT_DIR / "inputs" / "mutated_control_points.csv"
    mutation_path.write_bytes(GCP_PATH.read_bytes())
    expected_sha = sha256_file(mutation_path)
    if expected_sha != inspected_sha:
        raise RuntimeError("mutation test copy does not match the inspected GCP source")
    mutation_path.write_text(
        mutation_path.read_text(encoding="utf-8") + "# post-inspection mutation\n",
        encoding="utf-8",
    )
    project_dir = OUTPUT_DIR / "mutated-gcp-project"
    try:
        ProductionElevationRuntime(prior=AnalyticPrior()).run(
            ProcessingRequest(
                source=SOURCE_PATH,
                output_dir=project_dir,
                gcps=report_points,
                gcp_evidence=GroundControlPointEvidence(
                    source_path=mutation_path,
                    sha256=inspected_sha,
                ),
                requested_output="dsm",
            ),
            job_id="rt4-mutated-gcp",
        )
    except RuntimeError as exc:
        if "bytes changed after inspection" not in str(exc):
            raise
        return {"rejected": True, "error": str(exc)}
    raise RuntimeError("post-inspection GCP mutation unexpectedly passed calibration")


def _write_report(name: str, payload: dict[str, object]) -> Path:
    path = OUTPUT_DIR / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _verified_gcp_source(calibration: dict[str, object]) -> dict[str, object]:
    evidence = calibration.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("GCP calibration evidence is not an object")
    gcp = evidence.get("gcp")
    if not isinstance(gcp, dict):
        raise RuntimeError("GCP-only calibration evidence is missing its gcp record")
    source = gcp.get("source_evidence")
    if not isinstance(source, dict):
        raise RuntimeError("GCP-only calibration evidence is missing source identity")
    return source


def main() -> None:
    """Exercise RT4 calibration/analysis contracts without rerunning any consumed benchmark."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_inputs()
    gcp_report = inspect_ground_control_point_file(GCP_PATH)
    gcp_evidence = GroundControlPointEvidence(
        source_path=gcp_report.source_path,
        sha256=gcp_report.sha256,
    )

    print("DepthWizard Release Train 4 scientific/analytical acceptance")
    print("Purpose: calibration + analyst-tool engineering closure; NOT model promotion or accuracy evidence")

    dem_dir = OUTPUT_DIR / "dem-project"
    dem_manifest, dem_calibration = _run_project(
        "dem",
        ProcessingRequest(
            source=SOURCE_PATH,
            output_dir=dem_dir,
            dem_path=DEM_PATH,
            requested_output="dsm",
        ),
    )
    if dem_calibration.get("mode") != "dem":
        raise RuntimeError("DEM-only acceptance did not persist calibration mode=dem")
    dem_error = _metric_error(dem_manifest)
    dem_high_frequency = _high_frequency_preservation(dem_manifest)
    if not bool(dem_high_frequency["preserved"]):
        raise RuntimeError("DEM calibration no longer preserves the synthetic high-frequency structure")

    gcp_dir = OUTPUT_DIR / "gcp-project"
    gcp_manifest, gcp_calibration = _run_project(
        "gcp",
        ProcessingRequest(
            source=SOURCE_PATH,
            output_dir=gcp_dir,
            gcps=gcp_report.points,
            gcp_evidence=gcp_evidence,
            requested_output="dsm",
        ),
    )
    if gcp_calibration.get("mode") != "gcp":
        raise RuntimeError("GCP-only acceptance did not persist calibration mode=gcp")
    gcp_source = _verified_gcp_source(gcp_calibration)
    if gcp_source.get("identity_verified") is not True:
        raise RuntimeError("GCP-only acceptance did not preserve verified CSV source identity")
    gcp_error = _metric_error(gcp_manifest)

    fusion_dir = OUTPUT_DIR / "dem-gcp-project"
    fusion_manifest, fusion_calibration = _run_project(
        "dem-gcp",
        ProcessingRequest(
            source=SOURCE_PATH,
            output_dir=fusion_dir,
            dem_path=DEM_PATH,
            gcps=gcp_report.points,
            gcp_evidence=gcp_evidence,
            requested_output="dsm",
        ),
    )
    if fusion_calibration.get("mode") != "dem_gcp":
        raise RuntimeError("DEM+GCP acceptance did not persist calibration mode=dem_gcp")
    evidence = fusion_calibration.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("fusion_method") != "dem_then_gcp_high_reliability_refinement":
        raise RuntimeError("DEM+GCP fusion method contract changed")
    fusion_error = _metric_error(fusion_manifest)

    weak_abort = _weak_evidence_abort(gcp_report.points)
    mutation = _mutation_rejection(gcp_report.points, gcp_report.sha256)
    structure = _structure_report(gcp_dir)
    structure_height = structure.get("structure_height_m")
    if not isinstance(structure_height, (int, float)):
        raise RuntimeError("structure acceptance report is missing numeric structure_height_m")
    previews = _preview_report(gcp_dir)

    policy = current_production_estimator_decision()
    if policy.selected_path is not EstimatorPath.CALIBRATED_DA3:
        raise RuntimeError("RT4 acceptance detected an unapproved learned-refiner production promotion")

    dem_report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_ENGINEERING_CALIBRATION_DEM",
        "mode": "dem",
        "project_id": dem_manifest.project_id,
        "dem_sha256": sha256_file(DEM_PATH),
        "metric_error_against_synthetic_truth": dem_error,
        "high_frequency_preservation": dem_high_frequency,
        "calibration_evidence": dem_calibration,
        "scientific_boundary": "Synthetic deterministic calibration engineering acceptance only; not an external accuracy benchmark.",
    }
    gcp_only_report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_ENGINEERING_CALIBRATION_GCP",
        "mode": "gcp",
        "project_id": gcp_manifest.project_id,
        "gcp_file_sha256": gcp_report.sha256,
        "gcp_count": gcp_report.point_count,
        "metric_error_against_synthetic_truth": gcp_error,
        "weak_evidence_abort": weak_abort,
        "post_inspection_mutation_rejection": mutation,
        "calibration_evidence": gcp_calibration,
        "scientific_boundary": "Synthetic deterministic calibration engineering acceptance only; not an external accuracy benchmark.",
    }
    fusion_report: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_ENGINEERING_CALIBRATION_DEM_GCP",
        "mode": "dem_gcp",
        "project_id": fusion_manifest.project_id,
        "metric_error_against_synthetic_truth": fusion_error,
        "calibration_evidence": fusion_calibration,
        "scientific_boundary": "Synthetic deterministic calibration engineering acceptance only; not an external accuracy benchmark.",
    }
    _write_report("calibration_dem_report.json", dem_report)
    _write_report("calibration_gcp_report.json", gcp_only_report)
    _write_report("calibration_fusion_report.json", fusion_report)
    _write_report(
        "structural_height_validation.json",
        {
            "schema_version": 1,
            "status": "PASS_ENGINEERING_STRUCTURE_HEIGHT",
            "project_id": gcp_manifest.project_id,
            "result": structure,
            "scientific_boundary": "Analyst-selected synthetic structure engineering acceptance; no automatic footprint-classification claim.",
        },
    )

    integrated: dict[str, object] = {
        "schema_version": 1,
        "status": "PASS_RT4_SCIENTIFIC_ANALYTICAL_ENGINEERING_PATH",
        "production_estimator_policy": policy.as_dict(),
        "production_estimator_remains_calibrated_da3": True,
        "consumed_benchmark_rerun": False,
        "model_promotion_claim": False,
        "reference_data_used_for_calibration": False,
        "confidence_fabricated": False,
        "gcp_csv_identity_verified": True,
        "gcp_post_inspection_mutation_rejected": True,
        "weak_evidence_aborted": True,
        "dem_only": dem_report,
        "gcp_only": gcp_only_report,
        "dem_gcp": fusion_report,
        "structural_height": structure,
        "derived_previews": previews,
        "scientific_boundary": (
            "This train acceptance exercises production calibration, provenance, derived visualization, "
            "and analyst-selected structural-height contracts on deterministic synthetic data. It does "
            "not establish external DSM accuracy, terrain generalization, cross-sensor robustness, "
            "learned-model promotion, or Gate B scientific validation."
        ),
    }
    report_path = _write_report("release-train-4-analytical-acceptance.json", integrated)

    print("RT4 DEM-only calibration path: PASS")
    print("RT4 GCP-only calibration path + CSV identity: PASS")
    print("RT4 DEM+GCP high-reliability refinement path: PASS")
    print("Weak/underdetermined metric evidence abort: PASS")
    print("Post-inspection GCP mutation rejection: PASS")
    print(f"Analyst-selected structural height: {float(structure_height):.3f} m")
    print(f"Derived persisted-surface previews rendered: {len(previews)}")
    print("Confidence fabricated: NO")
    print("Production learned refiner promoted: NO")
    print("Consumed benchmark/model-promotion protocol rerun: NO")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
