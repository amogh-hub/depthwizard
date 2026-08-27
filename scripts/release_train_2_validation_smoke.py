from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

from depthwizard.contracts import (
    ProcessingRequest,
    ProjectRunStatus,
    ReferenceValidationRequest,
)
from depthwizard.evaluation.project_validation import validate_project_reference
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.provenance.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "acceptance" / "release-train-2-ortholoc"
PROJECT_DIR = ROOT / "artifacts" / "acceptance" / "release-train-2-ortholoc"
ORTHOLOC_BASE_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC/demo"
DOP_URL = f"{ORTHOLOC_BASE_URL}/urban_residential_DOP.tif"
DSM_URL = f"{ORTHOLOC_BASE_URL}/urban_residential_DSM.tif"
XDSM_URL = f"{ORTHOLOC_BASE_URL}/urban_residential_xDSM.tif"
CALIBRATION_DOWNSAMPLE_FACTOR = 16
USER_AGENT = "DepthWizard-SIH26175/0.2 release-train-2-acceptance"


def _download(url: str, path: Path) -> bytes:
    if path.is_file() and path.stat().st_size > 0:
        return path.read_bytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = response.read()
    if not payload:
        raise RuntimeError(f"empty response from {url}")
    path.write_bytes(payload)
    return payload


def _prepare_source_and_calibration_source() -> tuple[Path, Path]:
    """Cache only the production source and calibration lineage before reconstruction.

    The evaluation DSM is not requested by this helper. It may already exist in the shared local
    cache from an earlier run; the scientifically enforced boundary is that it is never supplied to
    reconstruction/calibration and reference validation is invoked only after production completes.
    """
    dop_path = DATA_DIR / "urban_residential_DOP.tif"
    xdsm_path = DATA_DIR / "urban_residential_xDSM.tif"
    _download(DOP_URL, dop_path)
    _download(XDSM_URL, xdsm_path)
    return dop_path, xdsm_path


def _build_coarse_calibration_dem(calibration_source_path: Path) -> Path:
    """Create a deliberately coarse, same-scene calibration surrogate for RT2 integration.

    OrthoLoC's demo geodata is suitable for local metric geometry, but the acceptance gate must not
    invent a global web-tile location when the source georeferencing cannot be safely interpreted as
    a global slippy-map footprint. Instead, RT2 uses the public cross-domain xDSM as a *calibration
    surrogate*, downsamples it by a fixed factor, and records that its lineage is not independent of
    the downstream OrthoLoC reference DSM.

    This artifact exists only to exercise the production metric-calibration path. It is not external
    validation evidence and must never be used for model promotion.
    """
    with rasterio.open(calibration_source_path) as src:
        if src.crs is None or src.transform.is_identity:
            raise ValueError("RT2 calibration surrogate requires georeferenced xDSM evidence")
        if src.count < 1:
            raise ValueError("RT2 calibration surrogate requires at least one elevation band")

        out_width = max(8, int(np.ceil(src.width / CALIBRATION_DOWNSAMPLE_FACTOR)))
        out_height = max(8, int(np.ceil(src.height / CALIBRATION_DOWNSAMPLE_FACTOR)))
        coarse = src.read(
            1,
            out_shape=(out_height, out_width),
            masked=True,
            resampling=Resampling.average,
        )
        nodata = float(src.nodata) if src.nodata is not None and np.isfinite(src.nodata) else -9999.0
        data = np.asarray(coarse.filled(nodata), dtype=np.float32)
        data[~np.isfinite(data)] = nodata
        transform = src.transform * Affine.scale(
            src.width / out_width,
            src.height / out_height,
        )
        crs = src.crs

    output = DATA_DIR / "urban_residential_xDSM_coarse_calibration.tif"
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        height=out_height,
        width=out_width,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
    ) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, "Coarse OrthoLoC xDSM calibration surrogate")
        dst.update_tags(
            DEPTHWIZARD_ROLE="CALIBRATION_ONLY_INTEGRATION_SURROGATE",
            CALIBRATION_SOURCE="TUM OrthoLoC urban_residential_xDSM.tif",
            CALIBRATION_LINEAGE_INDEPENDENT="false",
            DOWNSAMPLE_FACTOR=str(CALIBRATION_DOWNSAMPLE_FACTOR),
            CLAIM_BOUNDARY=(
                "Release Train 2 product integration only; not unseen validation or model-promotion evidence"
            ),
        )
    return output


def _artifact_evidence(manifest: ProjectManifest) -> dict[str, dict[str, object]]:
    evidence: dict[str, dict[str, object]] = {}
    for name in ("rdsm", "dsm", "slope", "reference", "residual", "metrics", "validation_report"):
        artifact = manifest.artifacts.get(name)
        if artifact is None:
            continue
        evidence[name] = {
            "path": artifact.get("path"),
            "sha256": artifact.get("sha256"),
            "semantics": artifact.get("semantics"),
            "units": artifact.get("units"),
        }
    return evidence


def main() -> None:
    """Run the historical RT2 local integration gate with a downstream reference boundary.

    This smoke intentionally does not create scientific model-promotion evidence. The public
    OrthoLoC cross-domain xDSM is downsampled into a coarse calibration surrogate and the regular
    OrthoLoC DSM is supplied to validation only after the production runtime has completed. Because
    both belong to the same scene/dataset lineage, the acceptance report explicitly records that
    calibration and evaluation are not lineage-independent.

    Reference cache chronology is recorded rather than manufactured: if the DSM already exists from
    a prior run, the report says so instead of claiming that its bytes were downloaded later.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    reference_path = DATA_DIR / "urban_residential_DSM.tif"
    reference_cached_before_runtime = reference_path.is_file() and reference_path.stat().st_size > 0

    print("DepthWizard Release Train 2 acceptance")
    print("Purpose: integrated production validation smoke; NOT model promotion or unseen Gate B evidence")
    print("Preparing/caching OrthoLoC optical source + cross-domain calibration source...")
    source_path, calibration_source_path = _prepare_source_and_calibration_source()
    print("Building deliberately coarse OrthoLoC xDSM calibration surrogate...")
    dem_path = _build_coarse_calibration_dem(calibration_source_path)

    source_sha = sha256_file(source_path)
    calibration_source_sha = sha256_file(calibration_source_path)
    dem_sha = sha256_file(dem_path)

    print("Running real production elevation runtime with coarse calibration evidence...")
    runtime_result = ProductionElevationRuntime().run(
        ProcessingRequest(
            source=source_path,
            output_dir=PROJECT_DIR,
            dem_path=dem_path,
            requested_output="dsm",
            tile_size=768,
            overlap=128,
        ),
        job_id="release-train-2-validation-smoke",
    )
    if runtime_result.status is not ProjectRunStatus.COMPLETE:
        raise RuntimeError(f"production runtime did not complete: {runtime_result.status.value}")

    print("Production DSM complete. Supplying TUM OrthoLoC DSM as downstream evaluation-only evidence...")
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
            reference_label="TUM OrthoLoC urban_residential DSM / evaluation-only integration reference",
            min_valid_pixels=1000,
        )
    )
    manifest = ProjectManifest.load(PROJECT_DIR)

    acceptance = {
        "schema_version": 3,
        "status": "PASS_INTEGRATED_VALIDATION_PATH",
        "purpose": (
            "Release Train 2 production integration acceptance; not an unseen benchmark and not "
            "model-promotion evidence"
        ),
        "scientific_separation": {
            "reference_boundary_enforced": True,
            "reference_supplied_to_reconstruction_or_calibration": False,
            "validation_invoked_after_runtime_completion": True,
            "evaluation_reference_cached_before_runtime": reference_cached_before_runtime,
            "evaluation_reference_downloaded_after_runtime": not reference_cached_before_runtime,
            "calibration": (
                "coarse surrogate downsampled from TUM OrthoLoC urban_residential_xDSM.tif"
            ),
            "evaluation": (
                "TUM OrthoLoC urban_residential_DSM.tif supplied only to downstream validation"
            ),
            "lineage_independent": False,
            "calibration_source_sha256": calibration_source_sha,
            "calibration_dem_sha256": dem_sha,
            "evaluation_reference_sha256": reference_sha,
            "distinct_file_sha256": dem_sha != reference_sha and calibration_source_sha != reference_sha,
            "note": (
                "The calibration surrogate and evaluation DSM are distinct files but share the same "
                "OrthoLoC scene/dataset lineage. Cache state is recorded truthfully; this smoke "
                "validates software integration and the downstream reference boundary only."
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
            "source": "TUM OrthoLoC urban_residential_xDSM.tif",
            "downsample_factor": CALIBRATION_DOWNSAMPLE_FACTOR,
            "lineage_independent": False,
        },
        "evaluation_reference": {
            "path": str(reference_path.resolve()),
            "sha256": reference_sha,
            "source": "TUM OrthoLoC urban_residential_DSM.tif",
            "cached_before_runtime": reference_cached_before_runtime,
            "supplied_to_validation_after_runtime": True,
        },
        "runtime": runtime_result.as_dict(),
        "validation": validation.model_dump(mode="json"),
        "artifacts": _artifact_evidence(manifest),
        "manifest_warnings": manifest.warnings,
        "manifest_errors": manifest.errors,
    }
    report_path = PROJECT_DIR / "release-train-2-acceptance.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(acceptance, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(report_path)

    print("DepthWizard Release Train 2 integrated validation path: PASS")
    print(f"Project: {PROJECT_DIR}")
    print(f"Calibration source SHA-256: {calibration_source_sha}")
    print(f"Coarse calibration DEM SHA-256: {dem_sha}")
    print(f"Evaluation DSM SHA-256: {reference_sha}")
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
        f"MAE {validation.slope.mae_degrees:.3f}°"
    )
    print("Claim boundary: engineering/product acceptance only; calibration/reference lineage is not independent.")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
