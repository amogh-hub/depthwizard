from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import rasterio

from depthwizard.contracts import (
    ProcessingRequest,
    ProjectRunStatus,
    ReferenceValidationRequest,
)
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.io.raster import ground_sample_distance_m
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file
from scripts.release_train_2_validation_smoke import (
    DATA_DIR,
    DSM_URL,
    _artifact_evidence,
    _build_coarse_calibration_dem,
    _download,
    _prepare_source_and_calibration_source,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-3-ortholoc-spatial"
REPORT_PATH = PROJECT_DIR / "release-train-3-spatial-foundation-acceptance.json"


def _direct_affine_spacing_m(path: Path) -> tuple[float, float]:
    with rasterio.open(path) as src:
        if src.transform.is_identity:
            raise ValueError("RT3 spatial foundation requires a meaningful OrthoLoC affine grid")
        gsd_x = float(np.hypot(src.transform.a, src.transform.d))
        gsd_y = float(np.hypot(src.transform.b, src.transform.e))
    if not np.isfinite(gsd_x) or not np.isfinite(gsd_y) or gsd_x <= 0 or gsd_y <= 0:
        raise ValueError("RT3 OrthoLoC affine grid does not encode positive metric pixel spacing")
    return gsd_x, gsd_y


def _assert_same_spacing(
    observed: tuple[float, float] | None,
    expected: tuple[float, float],
    *,
    label: str,
) -> None:
    if observed is None:
        raise RuntimeError(f"{label} did not expose metric GSD under the OrthoLoC contract")
    if not np.allclose(observed, expected, rtol=1e-7, atol=1e-9):
        raise RuntimeError(
            f"{label} GSD {observed} disagrees with direct OrthoLoC affine metric spacing {expected}"
        )


def main() -> None:
    """Rebuild one clean integration project after closing the OrthoLoC spatial-scale ambiguity.

    The official OrthoLoC dataset contract defines DOP/DSM pixel scale in metres. This dedicated
    acceptance therefore opts into DepthWizard's OrthoLoC local-metric affine mode before any
    reconstruction, calibration, slope generation, validation, mesh generation or analyst distance
    is produced. It creates a new project directory so previously accepted RT2 evidence is preserved
    rather than silently rewritten after the spatial-scale bug was discovered.
    """
    os.environ["DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE"] = "1"
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print("DepthWizard Release Train 3 spatial foundation acceptance")
    print("Purpose: clean OrthoLoC metric-affine product rebuild; not model promotion or unseen Gate B evidence")

    source_path, calibration_source_path = _prepare_source_and_calibration_source()
    reference_path = DATA_DIR / "urban_residential_DSM.tif"
    reference_cached_before_runtime = reference_path.is_file() and reference_path.stat().st_size > 0
    dem_path = _build_coarse_calibration_dem(calibration_source_path)

    direct_source_gsd = _direct_affine_spacing_m(source_path)
    source_gsd = ground_sample_distance_m(source_path)
    _assert_same_spacing(source_gsd, direct_source_gsd, label="source")

    source_sha = sha256_file(source_path)
    calibration_source_sha = sha256_file(calibration_source_path)
    dem_sha = sha256_file(dem_path)

    print(
        "OrthoLoC local metric affine: "
        f"{direct_source_gsd[0]:.6f} m/px × {direct_source_gsd[1]:.6f} m/px"
    )
    print("Running real production elevation runtime into a new spatial-foundation project...")
    runtime_result = ProductionElevationRuntime().run(
        ProcessingRequest(
            source=source_path,
            output_dir=PROJECT_DIR,
            dem_path=dem_path,
            requested_output="dsm",
            tile_size=768,
            overlap=128,
        ),
        job_id="release-train-3-spatial-foundation-smoke",
    )
    if runtime_result.status is not ProjectRunStatus.COMPLETE:
        raise RuntimeError(f"production runtime did not complete: {runtime_result.status.value}")

    # The reference may already be present in the shared dataset cache from RT2. The scientific
    # boundary is that it is not supplied to reconstruction/calibration and validation is invoked
    # only after the runtime is complete; we never delete a cached file to manufacture chronology.
    print("Production DSM complete. Supplying OrthoLoC DSM to downstream validation only...")
    _download(DSM_URL, reference_path)
    reference_sha = sha256_file(reference_path)
    if dem_sha == reference_sha:
        raise RuntimeError("calibration DEM and evaluation DSM unexpectedly have identical SHA-256")
    if calibration_source_sha == reference_sha:
        raise RuntimeError("calibration source and evaluation DSM unexpectedly have identical SHA-256")

    validation = validate_project_reference(
        ReferenceValidationRequest(
            project_dir=PROJECT_DIR,
            reference_path=reference_path,
            reference_label=(
                "TUM OrthoLoC urban_residential DSM / downstream integration reference under "
                "dataset-local metric-affine spatial contract"
            ),
            min_valid_pixels=1000,
        )
    )
    manifest = ProjectManifest.load(PROJECT_DIR)
    dsm_path = manifest.artifact_path("dsm")
    slope_path = manifest.artifact_path("slope")
    if dsm_path is None or not dsm_path.is_file():
        raise RuntimeError("spatial-foundation runtime did not persist a DSM")
    if slope_path is None or not slope_path.is_file():
        raise RuntimeError("corrected metric-affine runtime did not persist a slope raster")
    dsm_gsd = ground_sample_distance_m(dsm_path)
    _assert_same_spacing(dsm_gsd, direct_source_gsd, label="persisted DSM")

    acceptance = {
        "schema_version": 1,
        "status": "PASS_RT3_SPATIAL_FOUNDATION",
        "purpose": (
            "Release Train 3 clean product-integration rebuild under the official OrthoLoC "
            "dataset-local metric pixel-scale contract; not independent accuracy or model-promotion evidence"
        ),
        "project_id": manifest.project_id,
        "project_dir": str(PROJECT_DIR.resolve()),
        "spatial_scale": {
            "policy": "ortholoc_dataset_local_metric_affine",
            "official_contract": "DOP/DSM scale is metres per pixel",
            "global_lon_lat_claim": False,
            "source_affine_gsd_x_m": direct_source_gsd[0],
            "source_affine_gsd_y_m": direct_source_gsd[1],
            "runtime_source_gsd_x_m": source_gsd[0] if source_gsd is not None else None,
            "runtime_source_gsd_y_m": source_gsd[1] if source_gsd is not None else None,
            "persisted_dsm_gsd_x_m": dsm_gsd[0] if dsm_gsd is not None else None,
            "persisted_dsm_gsd_y_m": dsm_gsd[1] if dsm_gsd is not None else None,
        },
        "scientific_separation": {
            "reference_boundary_enforced": True,
            "reference_supplied_to_reconstruction_or_calibration": False,
            "validation_invoked_after_runtime_completion": True,
            "evaluation_reference_cached_before_runtime": reference_cached_before_runtime,
            "evaluation_reference_downloaded_after_runtime": not reference_cached_before_runtime,
            "lineage_independent": False,
            "calibration_source_sha256": calibration_source_sha,
            "calibration_dem_sha256": dem_sha,
            "evaluation_reference_sha256": reference_sha,
            "distinct_file_sha256": (
                dem_sha != reference_sha and calibration_source_sha != reference_sha
            ),
            "note": (
                "Calibration surrogate and evaluation DSM are distinct files but share the same "
                "OrthoLoC scene/dataset lineage. The cache state is recorded truthfully; a cached "
                "reference is never deleted merely to claim it was downloaded later."
            ),
        },
        "source": {
            "path": str(source_path.resolve()),
            "sha256": source_sha,
            "dataset": "TUM OrthoLoC demo / urban_residential",
            "dataset_license": "CC BY-NC-SA 4.0",
        },
        "calibration_evidence": {
            "path": str(dem_path.resolve()),
            "sha256": dem_sha,
            "source_path": str(calibration_source_path.resolve()),
            "source_sha256": calibration_source_sha,
            "lineage_independent": False,
        },
        "evaluation_reference": {
            "path": str(reference_path.resolve()),
            "sha256": reference_sha,
            "cached_before_runtime": reference_cached_before_runtime,
            "supplied_to_validation_after_runtime": True,
        },
        "runtime": runtime_result.as_dict(),
        "validation": validation.model_dump(mode="json"),
        "artifacts": _artifact_evidence(manifest),
        "manifest_warnings": manifest.warnings,
        "manifest_errors": manifest.errors,
    }
    temporary = REPORT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(REPORT_PATH)

    print("DepthWizard Release Train 3 spatial foundation: PASS")
    print(f"Project: {manifest.project_id}")
    print(f"Project directory: {PROJECT_DIR}")
    print(f"Reference cached before runtime: {'YES' if reference_cached_before_runtime else 'NO'}")
    print(f"Valid evaluation pixels: {validation.valid_pixels:,}")
    print(
        f"Elevation: RMSE {validation.elevation.rmse_m:.3f} m | "
        f"MAE {validation.elevation.mae_m:.3f} m | "
        f"P95 {validation.elevation.p95_abs_error_m:.3f} m | "
        f"r {validation.elevation.pearson_r}"
    )
    print(
        f"Slope: RMSE {validation.slope.rmse_degrees:.3f}° | "
        f"MAE {validation.slope.mae_degrees:.3f}° | "
        f"P95 {validation.slope.p95_abs_error_degrees:.3f}°"
    )
    print("Global lon/lat claim from demo TIFF CRS: NO")
    print("Consumed benchmark/model-promotion protocols rerun: NO")
    print(f"Acceptance report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
