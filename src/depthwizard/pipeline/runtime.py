from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import rasterio

from depthwizard.calibration.evidence import (
    EvidenceCalibrationOutput,
    calibrate_relative_height_with_dem,
)
from depthwizard.calibration.gcp import calibrate_relative_height_with_gcps
from depthwizard.contracts import (
    CalibrationMode,
    InputKind,
    ProcessingRequest,
    ProjectRunStatus,
)
from depthwizard.evaluation.metrics import slope_degrees
from depthwizard.geometry_prior.base import GeometryPrior
from depthwizard.geometry_prior.da3 import DA3MonocularPrior
from depthwizard.io.products import write_unreferenced_float_tiff
from depthwizard.io.raster import (
    ground_sample_distance_m,
    inspect_raster,
    reproject_to_match,
    write_float_geotiff,
    write_relative_tiff,
)
from depthwizard.pipeline.geometry import infer_geometry_scene
from depthwizard.pipeline.policy import (
    EstimatorDecision,
    EstimatorPath,
    current_production_estimator_decision,
)
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.stages import ProcessingStage
from depthwizard.provenance.manifest import build_provenance, canonical_json_hash, sha256_file


class SceneRefiner(Protocol):
    """Permanent adapter boundary for a future independently promoted learned refiner."""

    model_id: str

    def refine(
        self,
        source_path: Path,
        geometry: np.ndarray,
        *,
        band_indices: tuple[int, int, int],
        gsd_m: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None]: ...


@dataclass(frozen=True)
class ProjectRunResult:
    project_id: str
    job_id: str | None
    status: ProjectRunStatus
    manifest_path: Path
    primary_product: Path | None
    artifacts: dict[str, str]
    resumed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "job_id": self.job_id,
            "status": self.status.value,
            "manifest_path": str(self.manifest_path.resolve(strict=False)),
            "primary_product": (
                str(self.primary_product.resolve(strict=False))
                if self.primary_product is not None
                else None
            ),
            "artifacts": self.artifacts,
            "resumed": self.resumed,
        }


@dataclass(frozen=True)
class _GeometryState:
    relative_height: np.ndarray
    confidence: np.ndarray | None
    model_id: str
    tile_count: int
    harmonized_tiles: int


@dataclass(frozen=True)
class _CalibrationOutcome:
    dsm: np.ndarray
    evidence: dict[str, object]
    mode: CalibrationMode


def _write_json_atomic(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def _mean_gsd(gsd: tuple[float, float] | None) -> float | None:
    if gsd is None:
        return None
    return float((gsd[0] + gsd[1]) / 2.0)


def _request_config(request: ProcessingRequest) -> dict[str, object]:
    return request.model_dump(mode="json")


def _geometry_config(
    request: ProcessingRequest,
    estimator_decision: EstimatorDecision,
) -> dict[str, object]:
    """Hash every decision capable of changing the persisted relative geometry artifact."""
    return {
        "source": str(request.source.resolve(strict=False)),
        "band_indices": list(request.band_indices),
        "tile_size": request.tile_size,
        "overlap": request.overlap,
        "harmonize_overlaps": request.harmonize_overlaps,
        "estimator": estimator_decision.as_dict(),
    }


def _artifact_paths(manifest: ProjectManifest) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, payload in manifest.artifacts.items():
        raw = payload.get("path")
        if isinstance(raw, str):
            result[name] = raw
    return result


def _result_from_manifest(manifest: ProjectManifest, *, resumed: bool) -> ProjectRunResult:
    status = ProjectRunStatus(manifest.status)
    primary = manifest.artifact_path("dsm") or manifest.artifact_path("rdsm")
    return ProjectRunResult(
        project_id=manifest.project_id,
        job_id=manifest.job_id,
        status=status,
        manifest_path=manifest.path,
        primary_product=primary,
        artifacts=_artifact_paths(manifest),
        resumed=resumed,
    )


def _register_artifact(
    manifest: ProjectManifest,
    name: str,
    path: Path,
    *,
    semantics: str,
    units: str | None,
) -> None:
    manifest.register_artifact(
        name,
        path,
        semantics=semantics,
        units=units,
        sha256=sha256_file(path),
    )


def _read_float_product(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        array = src.read(1).astype(np.float32)
        valid = np.isfinite(array)
        if src.nodata is not None and np.isfinite(src.nodata):
            valid &= array != np.float32(src.nodata)
        return np.where(valid, array, np.nan).astype(np.float32)


def _dem_evidence_payload(dem_path: Path, result: EvidenceCalibrationOutput) -> dict[str, object]:
    return {
        "source": str(dem_path.resolve()),
        "sha256": sha256_file(dem_path),
        "calibration": result.calibration.model_dump(),
        "orientation_flipped": result.orientation_flipped,
        "anchor_correlation_before": result.anchor_correlation_before,
        "anchor_correlation_after": result.anchor_correlation_after,
        "frequency_match_sigma_px": result.frequency_match_sigma_px,
        "anchor_stride_px": result.anchor_stride_px,
        "anchors": int(result.anchor_mask.sum()),
    }


class ProductionElevationRuntime:
    """Unified production runtime for truthful rDSM/metric DSM project processing.

    Georeferenced imagery may reconstruct once and pause for DEM/GCP evidence, but cannot become a
    metric DSM without that evidence. The geometry stage is resumable under a hash containing every
    geometry-affecting parameter and estimator decision. Completed projects are immutable evidence.
    """

    def __init__(
        self,
        *,
        prior: GeometryPrior | None = None,
        estimator_decision: EstimatorDecision | None = None,
        learned_refiner: SceneRefiner | None = None,
    ) -> None:
        self.prior = prior or DA3MonocularPrior(device="auto")
        self.estimator_decision = estimator_decision or current_production_estimator_decision()
        self.learned_refiner = learned_refiner

    def _write_relative_product(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        relative_height: np.ndarray,
        *,
        georeferenced: bool,
        model_id: str,
    ) -> Path:
        path = request.output_dir / "products" / "rdsm.tif"
        if georeferenced:
            write_float_geotiff(
                path,
                relative_height,
                template_path=request.source,
                description="DepthWizard relative DSM (dimensionless)",
                tags={
                    "DEPTHWIZARD_PRODUCT": "RELATIVE_DSM_DIMENSIONLESS",
                    "ELEVATION_UNITS": "relative",
                    "MODEL_ID": model_id,
                    "ABSOLUTE_ELEVATION_STATUS": "not_calibrated",
                },
            )
        else:
            write_relative_tiff(path, relative_height)
        _register_artifact(
            manifest,
            "rdsm",
            path,
            semantics="dimensionless_relative_surface_height",
            units="relative",
        )
        return path

    def _write_confidence_product(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        confidence: np.ndarray | None,
        *,
        georeferenced: bool,
    ) -> Path | None:
        if confidence is None:
            manifest.add_warning(
                "selected geometry path emitted no native confidence raster; no confidence values "
                "were fabricated"
            )
            return None
        path = request.output_dir / "products" / "confidence.tif"
        tags = {
            "DEPTHWIZARD_PRODUCT": "MODEL_NATIVE_CONFIDENCE",
            "CONFIDENCE_SEMANTICS": "model_native_not_probability_calibrated",
        }
        if georeferenced:
            write_float_geotiff(
                path,
                confidence,
                template_path=request.source,
                description="DepthWizard model-native confidence",
                tags=tags,
            )
        else:
            write_unreferenced_float_tiff(
                path,
                confidence,
                description="DepthWizard model-native confidence",
                tags=tags,
            )
        _register_artifact(
            manifest,
            "confidence",
            path,
            semantics="model_native_confidence_not_probability_calibrated",
            units=None,
        )
        return path

    def _load_or_run_geometry(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        *,
        georeferenced: bool,
    ) -> tuple[_GeometryState, bool]:
        existing = manifest.artifact_path("rdsm")
        reusable = (
            manifest.stage_completed(ProcessingStage.GEOMETRY)
            and existing is not None
            and existing.is_file()
        )
        if reusable:
            assert existing is not None
            details = manifest.stages[ProcessingStage.GEOMETRY.value].get("details", {})
            confidence_path = manifest.artifact_path("confidence")
            confidence = (
                _read_float_product(confidence_path)
                if confidence_path is not None and confidence_path.is_file()
                else None
            )
            return (
                _GeometryState(
                    relative_height=_read_float_product(existing),
                    confidence=confidence,
                    model_id=str(details.get("model_id", "DA3MONO-LARGE")),
                    tile_count=int(details.get("tile_count", 0)),
                    harmonized_tiles=int(details.get("harmonized_tiles", 0)),
                ),
                True,
            )

        started = time.perf_counter()
        manifest.record_stage(ProcessingStage.GEOMETRY, status="running")
        scene = infer_geometry_scene(
            request.source,
            self.prior,
            band_indices=request.band_indices,
            tile_size=request.tile_size,
            overlap=request.overlap,
            harmonize_overlaps=request.harmonize_overlaps,
        )
        selected_relative = scene.relative_height
        selected_confidence = scene.confidence
        selected_model_id = scene.model_id

        if self.estimator_decision.selected_path is EstimatorPath.PROMOTED_LEARNED_REFINER:
            if self.learned_refiner is None:
                raise RuntimeError(
                    "production policy selected a promoted learned refiner but no production refiner "
                    "adapter is installed"
                )
            gsd = ground_sample_distance_m(request.source)
            selected_relative, learned_confidence = self.learned_refiner.refine(
                request.source,
                scene.relative_height,
                band_indices=request.band_indices,
                gsd_m=_mean_gsd(gsd),
            )
            if selected_relative.shape != scene.relative_height.shape:
                raise ValueError("learned refiner output shape does not match geometry prior")
            selected_confidence = learned_confidence
            selected_model_id = self.learned_refiner.model_id

        rdsm_path = self._write_relative_product(
            manifest,
            request,
            selected_relative,
            georeferenced=georeferenced,
            model_id=selected_model_id,
        )
        confidence_path = self._write_confidence_product(
            manifest,
            request,
            selected_confidence,
            georeferenced=georeferenced,
        )
        artifacts = {"rdsm": str(rdsm_path.resolve())}
        if confidence_path is not None:
            artifacts["confidence"] = str(confidence_path.resolve())
        manifest.record_stage(
            ProcessingStage.GEOMETRY,
            status="completed",
            artifacts=artifacts,
            details={
                "model_id": selected_model_id,
                "foundation_model_id": scene.model_id,
                "estimator_path": self.estimator_decision.selected_path.value,
                "tile_count": scene.tile_count,
                "harmonized_tiles": scene.harmonized_tiles,
                "normalization": asdict(scene.normalization),
            },
            elapsed_seconds=time.perf_counter() - started,
        )
        return (
            _GeometryState(
                relative_height=selected_relative,
                confidence=selected_confidence,
                model_id=selected_model_id,
                tile_count=scene.tile_count,
                harmonized_tiles=scene.harmonized_tiles,
            ),
            False,
        )

    def _dem_calibration(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
        dem_path: Path,
    ) -> EvidenceCalibrationOutput:
        aligned_dem, dem_valid = reproject_to_match(dem_path, request.source)
        target_gsd_m = _mean_gsd(ground_sample_distance_m(request.source))
        dem_effective_gsd_m = _mean_gsd(ground_sample_distance_m(dem_path))
        common = {
            "dem_valid": dem_valid & np.isfinite(geometry.relative_height),
            "low_frequency_sigma_px": request.low_frequency_sigma_px,
        }
        if target_gsd_m is not None and dem_effective_gsd_m is not None:
            return calibrate_relative_height_with_dem(
                geometry.relative_height,
                aligned_dem,
                target_gsd_m=target_gsd_m,
                dem_effective_gsd_m=dem_effective_gsd_m,
                **common,
            )
        manifest.add_warning(
            "DEM/source physical GSD could not both be derived; calibration frequency matching "
            "was not applied rather than guessed"
        )
        return calibrate_relative_height_with_dem(
            geometry.relative_height,
            aligned_dem,
            dem_valid=dem_valid & np.isfinite(geometry.relative_height),
            low_frequency_sigma_px=request.low_frequency_sigma_px,
        )

    def _calibrate(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
    ) -> _CalibrationOutcome:
        dem_path = request.metric_dem_path
        gcps = request.gcps
        with rasterio.open(request.source) as source:
            if source.crs is None or source.transform.is_identity:
                raise ValueError("metric calibration requires a georeferenced source raster")
            transform = source.transform

        if dem_path is None and not gcps:
            raise ValueError("metric calibration requires DEM and/or GCP evidence")

        if dem_path is not None:
            if not dem_path.is_file():
                raise FileNotFoundError(f"metric DEM does not exist: {dem_path}")
            dem_result = self._dem_calibration(manifest, request, geometry, dem_path)
            dem_payload = _dem_evidence_payload(dem_path, dem_result)
            if not gcps:
                return _CalibrationOutcome(
                    dsm=dem_result.dsm,
                    evidence={"dem": dem_payload},
                    mode=CalibrationMode.DEM,
                )

            # DEM establishes broad spatial support. GCPs then receive highest reliability by
            # refining the already metric field. A negative relation at this stage is contradictory
            # evidence and is rejected instead of flipping an already metric surface.
            gcp_result = calibrate_relative_height_with_gcps(
                dem_result.dsm,
                transform=transform,
                gcps=gcps,
                low_frequency_sigma_px=request.low_frequency_sigma_px,
                resolve_orientation=False,
            )
            return _CalibrationOutcome(
                dsm=gcp_result.dsm,
                evidence={
                    "fusion_method": "dem_then_gcp_high_reliability_refinement",
                    "dem": dem_payload,
                    "gcp_refinement": {
                        "calibration": gcp_result.calibration.model_dump(),
                        "gcp_count_supplied": len(gcps),
                        "gcp_residuals_m": gcp_result.gcp_residuals_m.tolist(),
                        "anchor_correlation_before": gcp_result.anchor_correlation_before,
                        "orientation_flipped": gcp_result.orientation_flipped,
                    },
                },
                mode=CalibrationMode.DEM_GCP,
            )

        gcp_result = calibrate_relative_height_with_gcps(
            geometry.relative_height,
            transform=transform,
            gcps=gcps,
            low_frequency_sigma_px=request.low_frequency_sigma_px,
            resolve_orientation=True,
        )
        return _CalibrationOutcome(
            dsm=gcp_result.dsm,
            evidence={
                "gcp": {
                    "calibration": gcp_result.calibration.model_dump(),
                    "gcp_count_supplied": len(gcps),
                    "gcp_residuals_m": gcp_result.gcp_residuals_m.tolist(),
                    "anchor_correlation_before": gcp_result.anchor_correlation_before,
                    "orientation_flipped": gcp_result.orientation_flipped,
                }
            },
            mode=CalibrationMode.GCP,
        )

    def _write_provenance(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
        *,
        calibration_mode: CalibrationMode,
    ) -> Path:
        source_hash = manifest.source_sha256
        if source_hash is None:
            raise RuntimeError("project source hash is missing before provenance generation")
        provenance = build_provenance(
            source_path=request.source,
            source_sha256=source_hash,
            config={
                "processing_request": _request_config(request),
                "estimator_decision": self.estimator_decision.as_dict(),
                "calibration_mode": calibration_mode.value,
            },
            model_manifest={
                "selected_model_id": geometry.model_id,
                "selected_path": self.estimator_decision.selected_path.value,
                "promotion_reason": self.estimator_decision.reason,
                "promotion_evidence": [
                    asdict(item) for item in self.estimator_decision.evidence
                ],
            },
            warnings=manifest.warnings,
        )
        provenance["project"] = {
            "project_id": manifest.project_id,
            "job_id": manifest.job_id,
            "input_kind": manifest.input_kind,
            "geometry_config_sha256": manifest.geometry_config_sha256,
            "run_config_sha256": manifest.run_config_sha256,
        }
        provenance["products"] = manifest.artifacts
        path = request.output_dir / "provenance.json"
        _write_json_atomic(path, provenance)
        _register_artifact(
            manifest,
            "provenance",
            path,
            semantics="project_processing_provenance",
            units=None,
        )
        return path

    def _complete_relative_project(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
        *,
        job_id: str | None,
        resumed: bool,
        reason: str,
    ) -> ProjectRunResult:
        manifest.record_stage(
            ProcessingStage.CALIBRATION,
            status="skipped",
            details={"reason": reason, "metric_claim": False},
        )
        current_rdsm = manifest.artifact_path("rdsm")
        if current_rdsm is None or not current_rdsm.is_file():
            raise RuntimeError("relative project completed geometry without a durable rDSM artifact")
        provenance = self._write_provenance(
            manifest,
            request,
            geometry,
            calibration_mode=CalibrationMode.NONE,
        )
        export_artifacts = {
            "rdsm": str(current_rdsm.resolve()),
            "provenance": str(provenance.resolve()),
        }
        confidence_path = manifest.artifact_path("confidence")
        if confidence_path is not None:
            export_artifacts["confidence"] = str(confidence_path.resolve(strict=False))
        manifest.record_stage(
            ProcessingStage.EXPORT,
            status="completed",
            artifacts=export_artifacts,
            details={
                "elevation_units": "relative",
                "metric_claim": False,
                "source_files_overwritten": False,
            },
        )
        manifest.record_stage(ProcessingStage.COMPLETE, status="completed")
        manifest.mark_status(ProjectRunStatus.COMPLETE, job_id=job_id)
        return _result_from_manifest(manifest, resumed=resumed)

    def run(self, request: ProcessingRequest, *, job_id: str | None = None) -> ProjectRunResult:
        if not request.source.is_file():
            raise FileNotFoundError(f"source raster does not exist: {request.source}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = ProjectManifest.create_or_load(request.output_dir, request.source)
        current_stage: ProcessingStage | None = None

        source_hash = sha256_file(request.source)
        geometry_hash = canonical_json_hash(_geometry_config(request, self.estimator_decision))
        run_hash = canonical_json_hash(_request_config(request))

        if manifest.source_sha256 is not None and manifest.source_sha256 != source_hash:
            raise RuntimeError("source raster bytes changed after this project was created")
        if (
            manifest.geometry_config_sha256 is not None
            and manifest.geometry_config_sha256 != geometry_hash
            and manifest.stage_completed(ProcessingStage.GEOMETRY)
        ):
            raise RuntimeError(
                "geometry-affecting configuration or estimator policy changed after reconstruction; "
                "create a new project directory rather than mixing incompatible artifacts"
            )
        if manifest.status == ProjectRunStatus.COMPLETE.value:
            if manifest.run_config_sha256 != run_hash:
                raise RuntimeError(
                    "completed DepthWizard projects are immutable under their processing config; "
                    "create a new project directory for a changed calibration/export request"
                )
            return _result_from_manifest(manifest, resumed=True)

        try:
            manifest.mark_status(ProjectRunStatus.RUNNING, job_id=job_id)
            metadata = inspect_raster(request.source)
            georeferenced = metadata.input_kind is InputKind.GEOREFERENCED
            manifest.set_identity(
                source_sha256=source_hash,
                input_kind=metadata.input_kind.value,
                geometry_config_sha256=geometry_hash,
                run_config_sha256=run_hash,
            )
            manifest.set_estimator(self.estimator_decision.as_dict())

            current_stage = ProcessingStage.INGEST
            manifest.record_stage(ProcessingStage.INGEST, status="running")
            if max(request.band_indices) > metadata.count:
                raise ValueError(
                    f"RGB band mapping {request.band_indices} exceeds source band count "
                    f"{metadata.count}"
                )
            manifest.record_stage(
                ProcessingStage.INGEST,
                status="completed",
                details={"raster": metadata.model_dump(mode="json")},
            )

            current_stage = ProcessingStage.GEOMETRY
            geometry, geometry_resumed = self._load_or_run_geometry(
                manifest,
                request,
                georeferenced=georeferenced,
            )

            current_stage = ProcessingStage.CALIBRATION
            if not georeferenced:
                if request.requested_output == "dsm":
                    raise ValueError(
                        "non-georeferenced imagery cannot produce an absolute DSM without spatial "
                        "metadata; request rDSM or provide georeferenced imagery"
                    )
                current_stage = ProcessingStage.EXPORT
                return self._complete_relative_project(
                    manifest,
                    request,
                    geometry,
                    job_id=job_id,
                    resumed=geometry_resumed,
                    reason="non_georeferenced_input",
                )

            if request.requested_output == "rdsm":
                current_stage = ProcessingStage.EXPORT
                return self._complete_relative_project(
                    manifest,
                    request,
                    geometry,
                    job_id=job_id,
                    resumed=geometry_resumed,
                    reason="explicit_relative_output_requested",
                )

            if request.metric_dem_path is None and not request.gcps:
                manifest.record_stage(
                    ProcessingStage.CALIBRATION,
                    status="waiting",
                    details={
                        "reason": "metric_evidence_required",
                        "accepted_evidence": ["DEM", "GCP", "DEM+GCP"],
                        "metric_claim": False,
                    },
                )
                self._write_provenance(
                    manifest,
                    request,
                    geometry,
                    calibration_mode=CalibrationMode.NONE,
                )
                manifest.mark_status(ProjectRunStatus.WAITING_FOR_CALIBRATION, job_id=job_id)
                return _result_from_manifest(manifest, resumed=geometry_resumed)

            started = time.perf_counter()
            manifest.record_stage(ProcessingStage.CALIBRATION, status="running")
            calibration = self._calibrate(manifest, request, geometry)
            dsm_path = request.output_dir / "products" / "dsm.tif"
            write_float_geotiff(
                dsm_path,
                calibration.dsm,
                template_path=request.source,
                description="DepthWizard absolute Digital Surface Model (metres)",
                tags={
                    "DEPTHWIZARD_PRODUCT": "ABSOLUTE_DSM_METRES",
                    "ELEVATION_UNITS": "metres",
                    "CALIBRATION_MODE": calibration.mode.value,
                    "ESTIMATOR_PATH": self.estimator_decision.selected_path.value,
                },
            )
            _register_artifact(
                manifest,
                "dsm",
                dsm_path,
                semantics="absolute_digital_surface_model",
                units="m",
            )
            calibration_document: dict[str, object] = {
                "schema": "depthwizard.calibration.v1",
                "mode": calibration.mode.value,
                "metric_claim": True,
                "evidence": calibration.evidence,
            }
            calibration_path = request.output_dir / "calibration.json"
            _write_json_atomic(calibration_path, calibration_document)
            _register_artifact(
                manifest,
                "calibration",
                calibration_path,
                semantics="metric_calibration_evidence",
                units=None,
            )
            manifest.record_stage(
                ProcessingStage.CALIBRATION,
                status="completed",
                artifacts={
                    "dsm": str(dsm_path.resolve()),
                    "calibration": str(calibration_path.resolve()),
                },
                details={
                    "mode": calibration.mode.value,
                    "metric_claim": True,
                    "evidence": calibration.evidence,
                },
                elapsed_seconds=time.perf_counter() - started,
            )

            current_stage = ProcessingStage.EXPORT
            started = time.perf_counter()
            manifest.record_stage(ProcessingStage.EXPORT, status="running")
            source_gsd = ground_sample_distance_m(request.source)
            slope_path: Path | None = None
            if source_gsd is not None:
                slope = slope_degrees(
                    calibration.dsm,
                    gsd_x=source_gsd[0],
                    gsd_y=source_gsd[1],
                )
                slope_path = request.output_dir / "products" / "slope.tif"
                write_float_geotiff(
                    slope_path,
                    slope,
                    template_path=request.source,
                    description="DepthWizard DSM slope (degrees)",
                    tags={
                        "DEPTHWIZARD_PRODUCT": "SLOPE_DEGREES",
                        "ANGLE_UNITS": "degrees",
                    },
                )
                _register_artifact(
                    manifest,
                    "slope",
                    slope_path,
                    semantics="surface_slope",
                    units="degrees",
                )
            else:
                manifest.add_warning(
                    "physical GSD unavailable; slope raster omitted rather than deriving slope in "
                    "unknown coordinate units"
                )

            provenance = self._write_provenance(
                manifest,
                request,
                geometry,
                calibration_mode=calibration.mode,
            )
            export_artifacts = {
                "dsm": str(dsm_path.resolve()),
                "calibration": str(calibration_path.resolve()),
                "provenance": str(provenance.resolve()),
            }
            if slope_path is not None:
                export_artifacts["slope"] = str(slope_path.resolve())
            confidence_path = manifest.artifact_path("confidence")
            if confidence_path is not None:
                export_artifacts["confidence"] = str(confidence_path.resolve(strict=False))
            manifest.record_stage(
                ProcessingStage.EXPORT,
                status="completed",
                artifacts=export_artifacts,
                details={
                    "metric_claim": True,
                    "elevation_units": "m",
                    "source_files_overwritten": False,
                },
                elapsed_seconds=time.perf_counter() - started,
            )
            manifest.record_stage(ProcessingStage.COMPLETE, status="completed")
            manifest.mark_status(ProjectRunStatus.COMPLETE, job_id=job_id)
            return _result_from_manifest(manifest, resumed=geometry_resumed)
        except Exception as exc:
            if current_stage is not None:
                manifest.record_stage(
                    current_stage,
                    status="failed",
                    details={"error": str(exc)},
                )
            manifest.add_error(str(exc), stage=current_stage)
            manifest.mark_status(ProjectRunStatus.FAILED, job_id=job_id)
            raise
