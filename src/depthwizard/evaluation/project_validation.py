from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling

from depthwizard.contracts import (
    ErrorConfidenceBin,
    ReferenceValidationReport,
    ReferenceValidationRequest,
    ReliabilityDiagnostics,
)
from depthwizard.evaluation.metrics import compute_elevation_metrics, compute_slope_metrics
from depthwizard.io.raster import (
    ground_pixel_jacobian_m,
    ground_sample_distance_m,
    reproject_to_match,
    write_float_geotiff,
)
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import sha256_file


def _read_prediction(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        valid = src.read_masks(1) > 0
        valid &= np.isfinite(values)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= values != np.float32(src.nodata)
    values[~valid] = np.nan
    return values, valid


def _alignment_description(reference_path: Path, prediction_path: Path) -> str:
    with rasterio.open(reference_path) as reference, rasterio.open(prediction_path) as prediction:
        exact = (
            reference.width == prediction.width
            and reference.height == prediction.height
            and reference.crs == prediction.crs
            and reference.transform.almost_equals(prediction.transform)
        )
        if exact:
            return "exact_grid_no_resampling"
        if reference.crs is None or prediction.crs is None:
            return "strict_crs_free_exact_grid_required"
        return "reference_reprojected_to_prediction_grid_bilinear"


def _calibration_dem_sha256(manifest: ProjectManifest) -> str | None:
    calibration = manifest.stages.get(ProcessingStage.CALIBRATION.value, {})
    details = calibration.get("details")
    if not isinstance(details, dict):
        return None
    evidence = details.get("evidence")
    if not isinstance(evidence, dict):
        return None
    dem = evidence.get("dem")
    if not isinstance(dem, dict):
        return None
    value = dem.get("sha256")
    return value if isinstance(value, str) else None


def _confidence_reliability(
    manifest: ProjectManifest,
    *,
    prediction_path: Path,
    absolute_error: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[ReliabilityDiagnostics, list[str]]:
    artifact = manifest.artifacts.get("confidence")
    if not artifact:
        return (
            ReliabilityDiagnostics(
                available=False,
                semantics="unavailable_no_native_confidence_artifact",
            ),
            [
                (
                    "No confidence artifact is available for this estimator path; "
                    "error-confidence reliability was not fabricated."
                )
            ],
        )

    raw_path = artifact.get("path")
    if not isinstance(raw_path, str):
        raise TypeError("confidence artifact path is malformed in project manifest")
    confidence, confidence_valid = reproject_to_match(
        Path(raw_path),
        prediction_path,
        resampling=Resampling.bilinear,
    )
    mask = valid_mask & confidence_valid & np.isfinite(confidence) & np.isfinite(absolute_error)
    count = int(mask.sum())
    semantics = str(artifact.get("semantics") or "model_native_confidence_not_probability_calibrated")
    if count < 2:
        return (
            ReliabilityDiagnostics(
                available=False,
                semantics=f"{semantics}; insufficient_overlap",
                valid_pixels=count,
            ),
            ["Confidence artifact has insufficient overlap with valid reference pixels."],
        )

    conf = np.asarray(confidence[mask], dtype=np.float64)
    error = np.asarray(absolute_error[mask], dtype=np.float64)
    correlation: float | None
    if np.std(conf) <= 1e-12 or np.std(error) <= 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(conf, error)[0, 1])

    order = np.argsort(conf, kind="stable")
    bins: list[ErrorConfidenceBin] = []
    for indices in np.array_split(order, min(5, count)):
        if indices.size == 0:
            continue
        bin_conf = conf[indices]
        bin_error = error[indices]
        bins.append(
            ErrorConfidenceBin(
                lower_confidence=float(np.min(bin_conf)),
                upper_confidence=float(np.max(bin_conf)),
                valid_pixels=int(indices.size),
                mean_confidence=float(np.mean(bin_conf)),
                mae_m=float(np.mean(bin_error)),
                rmse_m=float(np.sqrt(np.mean(bin_error**2))),
            )
        )

    return (
        ReliabilityDiagnostics(
            available=True,
            semantics=semantics,
            valid_pixels=count,
            confidence_abs_error_pearson_r=correlation,
            bins=bins,
        ),
        [],
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _write_human_report(path: Path, report: ReferenceValidationReport) -> None:
    pearson = "—" if report.elevation.pearson_r is None else f"{report.elevation.pearson_r:.4f}"
    reliability = (
        "Unavailable; no values were synthesized."
        if not report.reliability.available
        else (
            "Model-native confidence vs absolute error Pearson r: "
            f"{report.reliability.confidence_abs_error_pearson_r}"
        )
    )
    lines = [
        "# DepthWizard project reference validation",
        "",
        f"- Project: `{report.project_id}`",
        f"- Reference: `{report.reference_path}`",
        f"- Reference SHA-256: `{report.reference_sha256}`",
        f"- Prediction SHA-256: `{report.prediction_sha256}`",
        f"- Alignment: `{report.alignment}`",
        f"- Exact-file calibration independence check: `{report.independence_check}`",
        f"- Valid evaluation pixels: **{report.valid_pixels:,}**",
        f"- Prediction-grid coverage: **{100.0 * report.coverage_fraction:.2f}%**",
        "",
        "## Elevation metrics",
        "",
        f"- RMSE: **{report.elevation.rmse_m:.4f} m**",
        f"- MAE: **{report.elevation.mae_m:.4f} m**",
        f"- Mean bias: **{report.elevation.mean_bias_m:.4f} m**",
        f"- Median absolute error: **{report.elevation.median_abs_error_m:.4f} m**",
        f"- P90 absolute error: **{report.elevation.p90_abs_error_m:.4f} m**",
        f"- P95 absolute error: **{report.elevation.p95_abs_error_m:.4f} m**",
        f"- Pearson r: **{pearson}**",
        "",
        "## Slope metrics",
        "",
        f"- RMSE: **{report.slope.rmse_degrees:.4f}°**",
        f"- MAE: **{report.slope.mae_degrees:.4f}°**",
        f"- P95 absolute error: **{report.slope.p95_abs_error_degrees:.4f}°**",
        "",
        "## Reliability",
        "",
        reliability,
        "",
        "## Claim boundary",
        "",
        (
            "This report evaluates the supplied reference on its valid overlap after deterministic "
            "alignment to the prediction grid. Passing the exact-file hash check only proves that "
            "the reference is not byte-identical to the DEM used for calibration; it does not by "
            "itself prove geographic, temporal, or sensor independence."
        ),
    ]
    if report.warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report.warnings)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_project_reference(request: ReferenceValidationRequest) -> ReferenceValidationReport:
    """Evaluate one completed metric project against an explicit reference surface.

    Validation is deliberately downstream of reconstruction/calibration. It never feeds reference
    values back into the estimator, calibration fit, or promotion policy. A reference that is
    byte-identical to the calibration DEM is rejected so one file cannot serve as both scale evidence
    and claimed independent evaluation evidence.
    """
    project_dir = request.project_dir
    reference_path = request.reference_path
    if not (project_dir / "project-manifest.json").is_file():
        raise FileNotFoundError("DepthWizard project manifest does not exist")
    if not reference_path.is_file():
        raise FileNotFoundError("reference DSM does not exist")

    manifest = ProjectManifest.load(project_dir)
    prediction_path = manifest.artifact_path("dsm")
    if prediction_path is None or not prediction_path.is_file():
        raise ValueError("reference validation requires a completed metric DSM artifact")

    prediction_sha = sha256_file(prediction_path)
    reference_sha = sha256_file(reference_path)
    if reference_sha == prediction_sha:
        raise ValueError("reference DSM cannot be the prediction artifact itself")

    calibration_dem_sha = _calibration_dem_sha256(manifest)
    if calibration_dem_sha is not None and reference_sha == calibration_dem_sha:
        raise ValueError(
            "reference DSM is byte-identical to the calibration DEM; independent validation "
            "requires separate evidence"
        )
    independence_check = (
        "different_sha_from_calibration_dem"
        if calibration_dem_sha is not None
        else "no_calibration_dem_hash_available_for_exact_file_comparison"
    )

    existing = manifest.stages.get(ProcessingStage.VALIDATION.value, {})
    if existing.get("status") == "completed":
        metrics_path = manifest.artifact_path("metrics")
        if metrics_path is not None and metrics_path.is_file():
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            previous_sha = payload.get("reference_sha256")
            if previous_sha == reference_sha:
                return ReferenceValidationReport.model_validate(payload)
        raise RuntimeError(
            "project already contains completed validation for a different reference; preserve the "
            "existing evidence and validate the alternate reference in a separate project copy"
        )

    started = time.perf_counter()
    manifest.record_stage(
        ProcessingStage.VALIDATION,
        status="running",
        details={
            "reference_path": str(reference_path),
            "reference_sha256": reference_sha,
            "independence_check": independence_check,
        },
    )

    try:
        prediction, prediction_valid = _read_prediction(prediction_path)
        alignment = _alignment_description(reference_path, prediction_path)
        reference, reference_valid = reproject_to_match(
            reference_path,
            prediction_path,
            resampling=Resampling.bilinear,
        )
        valid = prediction_valid & reference_valid & np.isfinite(reference)
        valid_pixels = int(valid.sum())
        if valid_pixels < request.min_valid_pixels:
            raise ValueError(
                f"insufficient reference overlap: {valid_pixels} valid pixels; "
                f"need at least {request.min_valid_pixels}"
            )

        prediction_support = int(prediction_valid.sum())
        coverage = valid_pixels / max(prediction_support, 1)
        gsd = ground_sample_distance_m(prediction_path)
        ground_jacobian = ground_pixel_jacobian_m(prediction_path)
        if gsd is None or ground_jacobian is None:
            raise ValueError(
                "metric validation requires a georeferenced prediction with full local "
                "ground-pixel geometry"
            )

        elevation = compute_elevation_metrics(prediction, reference, valid_mask=valid)
        slope = compute_slope_metrics(
            prediction,
            reference,
            gsd_x=gsd[0],
            gsd_y=gsd[1],
            ground_jacobian_m=ground_jacobian,
            valid_mask=valid,
        )
        residual = np.full(prediction.shape, np.nan, dtype=np.float32)
        residual[valid] = prediction[valid] - reference[valid]
        aligned_reference = np.full(prediction.shape, np.nan, dtype=np.float32)
        aligned_reference[valid] = reference[valid]
        absolute_error = np.abs(residual).astype(np.float32)

        reliability, warnings = _confidence_reliability(
            manifest,
            prediction_path=prediction_path,
            absolute_error=absolute_error,
            valid_mask=valid,
        )

        products_dir = project_dir / "products"
        residual_path = write_float_geotiff(
            products_dir / "residual.tif",
            residual,
            template_path=prediction_path,
            description="DepthWizard prediction minus reference DSM residual",
            tags={
                "DEPTHWIZARD_PRODUCT": "VALIDATION_RESIDUAL",
                "ELEVATION_UNITS": "m",
                "REFERENCE_SHA256": reference_sha,
                "RESIDUAL_SIGN": "prediction_minus_reference",
            },
        )
        reference_aligned_path = write_float_geotiff(
            products_dir / "reference-aligned.tif",
            aligned_reference,
            template_path=prediction_path,
            description="DepthWizard aligned evaluation reference DSM",
            tags={
                "DEPTHWIZARD_PRODUCT": "ALIGNED_REFERENCE_DSM",
                "ELEVATION_UNITS": "m",
                "SOURCE_REFERENCE_SHA256": reference_sha,
            },
        )

        metrics_path = project_dir / "metrics.json"
        human_report_path = project_dir / "validation-report.md"
        report = ReferenceValidationReport(
            project_id=manifest.project_id,
            prediction_sha256=prediction_sha,
            reference_path=reference_path,
            reference_sha256=reference_sha,
            reference_label=request.reference_label,
            independence_check=independence_check,
            alignment=alignment,
            valid_pixels=valid_pixels,
            coverage_fraction=float(coverage),
            elevation=elevation,
            slope=slope,
            reliability=reliability,
            artifacts={
                "reference": str(reference_aligned_path.resolve(strict=False)),
                "residual": str(residual_path.resolve(strict=False)),
                "metrics": str(metrics_path.resolve(strict=False)),
                "validation_report": str(human_report_path.resolve(strict=False)),
            },
            warnings=warnings,
        )
        _write_json(metrics_path, report.model_dump(mode="json"))
        _write_human_report(human_report_path, report)

        manifest.register_artifact(
            "reference",
            reference_aligned_path,
            semantics="aligned_evaluation_reference_dsm",
            units="m",
            sha256=sha256_file(reference_aligned_path),
        )
        manifest.register_artifact(
            "residual",
            residual_path,
            semantics="prediction_minus_reference_residual",
            units="m",
            sha256=sha256_file(residual_path),
        )
        manifest.register_artifact(
            "metrics",
            metrics_path,
            semantics="reference_validation_metrics",
            units=None,
            sha256=sha256_file(metrics_path),
        )
        manifest.register_artifact(
            "validation_report",
            human_report_path,
            semantics="human_readable_reference_validation_report",
            units=None,
            sha256=sha256_file(human_report_path),
        )
        for warning in warnings:
            manifest.add_warning(warning)
        manifest.record_stage(
            ProcessingStage.VALIDATION,
            status="completed",
            artifacts=report.artifacts,
            details={
                "reference_path": str(reference_path),
                "reference_sha256": reference_sha,
                "prediction_sha256": prediction_sha,
                "independence_check": independence_check,
                "alignment": alignment,
                "coverage_fraction": float(coverage),
                "elevation": elevation.model_dump(mode="json"),
                "slope": slope.model_dump(mode="json"),
                "reliability": reliability.model_dump(mode="json"),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        return report
    except Exception as exc:
        manifest.record_stage(
            ProcessingStage.VALIDATION,
            status="failed",
            details={
                "reference_path": str(reference_path),
                "reference_sha256": reference_sha,
                "error": str(exc),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        manifest.add_error(str(exc), stage=ProcessingStage.VALIDATION)
        raise
