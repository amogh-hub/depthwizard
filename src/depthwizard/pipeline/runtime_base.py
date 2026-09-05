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
from depthwizard.calibration.gcp import (
    calibrate_relative_height_with_gcps,
    validate_metric_dsm_with_gcps,
)
from depthwizard.cancellation import (
    CancellationProbe,
    CancellationRequested,
    raise_if_cancelled,
)
from depthwizard.contracts import (
    CalibrationMode,
    InputKind,
    ProcessingRequest,
    ProjectRunStatus,
)
from depthwizard.evaluation.metrics import slope_degrees
from depthwizard.geometry_prior.base import GeometryPrior
from depthwizard.geometry_prior.da3 import (
    DA3_ADAPTER_CONTRACT,
    DA3_CHECKPOINT_SHA256,
    DA3_HF_REVISION,
    DA3_MODEL_SOURCE,
    DA3MonocularPrior,
)
from depthwizard.io.products import write_unreferenced_float_tiff
from depthwizard.io.raster import (
    ground_pixel_jacobian_m,
    ground_sample_distance_m,
    inspect_raster,
    reproject_to_match,
    write_float_geotiff,
    write_relative_tiff,
)
from depthwizard.pipeline.geometry import GEOMETRY_PIPELINE_CONTRACT, infer_geometry_scene
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
    valid_pixel_fraction: float = 1.0


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
    *,
    prior: GeometryPrior,
    learned_refiner: SceneRefiner | None,
) -> dict[str, object]:
    """Hash every decision capable of changing the persisted relative geometry artifact."""
    prior_class = f"{type(prior).__module__}.{type(prior).__qualname__}"
    prior_identity: dict[str, object] = {"class": prior_class}
    model_source = getattr(prior, "model_source", None)
    if model_source is not None:
        prior_identity["model_source"] = str(model_source)
    if isinstance(prior, DA3MonocularPrior) and str(prior.model_source) == DA3_MODEL_SOURCE:
        prior_identity.update(
            {
                "adapter_contract": DA3_ADAPTER_CONTRACT,
                "model_revision": DA3_HF_REVISION,
                "checkpoint_sha256": DA3_CHECKPOINT_SHA256,
            }
        )

    refiner_identity: dict[str, object] | None = None
    if learned_refiner is not None:
        refiner_identity = {
            "class": f"{type(learned_refiner).__module__}.{type(learned_refiner).__qualname__}",
            "model_id": learned_refiner.model_id,
        }
    return {
        "geometry_pipeline_contract": GEOMETRY_PIPELINE_CONTRACT,
        "source": str(request.source.resolve(strict=False)),
        "band_indices": list(request.band_indices),
        "tile_size": request.tile_size,
        "overlap": request.overlap,
        "harmonize_overlaps": request.harmonize_overlaps,
        "prior": prior_identity,
        "learned_refiner": refiner_identity,
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
        "bias_sigma_px": result.bias_sigma_px,
        "anchor_spatial_coverage_fraction": result.anchor_spatial_coverage_fraction,
        "metric_relief_span_m": result.metric_relief_span_m,
        "anchors": int(result.anchor_mask.sum()),
    }


def _resolved_bias_sigma_px(
    request: ProcessingRequest,
    *,
    target_gsd_m: float | None,
) -> float | None:
    """Resolve an operator smoothing request without silently mixing pixels and metres."""
    if request.low_frequency_sigma_m is not None:
        if target_gsd_m is None:
            raise ValueError(
                "low_frequency_sigma_m requires trustworthy physical source GSD; "
                "the requested smoothing distance cannot be converted to pixels"
            )
        return float(request.low_frequency_sigma_m / target_gsd_m)
    return request.low_frequency_sigma_px


def _gcp_source_evidence_payload(request: ProcessingRequest) -> dict[str, object]:
    """Return auditable GCP source identity and reject post-inspection file mutation."""
    evidence = request.gcp_evidence
    if evidence is None:
        return {
            "kind": "inline_points",
            "source": None,
            "sha256": None,
            "identity_verified": False,
        }
    source = evidence.source_path
    if not source.is_file():
        raise FileNotFoundError(f"GCP evidence file does not exist: {source}")
    actual_sha256 = sha256_file(source)
    if actual_sha256 != evidence.sha256:
        raise RuntimeError(
            "GCP evidence file bytes changed after inspection; re-import the GCP file before calibration"
        )
    return {
        "kind": "csv_file",
        "source": str(source.resolve()),
        "sha256": actual_sha256,
        "identity_verified": True,
    }


def _vertical_reference_payload(request: ProcessingRequest) -> dict[str, object]:
    """Resolve explicit vertical semantics while keeping unknown metadata visibly unknown."""
    vertical_crs = request.vertical_crs
    vertical_datum = request.vertical_datum
    elevation_reference = request.elevation_reference
    source = "processing_request" if any(
        value is not None for value in (vertical_crs, vertical_datum)
    ) or elevation_reference != "unknown" else "unspecified"

    dem_path = request.metric_dem_path
    if dem_path is not None and dem_path.is_file():
        dem_metadata = inspect_raster(dem_path)
        if (
            vertical_crs is not None
            and dem_metadata.vertical_crs is not None
            and vertical_crs.casefold() != dem_metadata.vertical_crs.casefold()
        ):
            raise ValueError(
                "processing-request vertical CRS conflicts with calibration DEM metadata"
            )
        if (
            vertical_datum is not None
            and dem_metadata.vertical_datum is not None
            and vertical_datum.casefold() != dem_metadata.vertical_datum.casefold()
        ):
            raise ValueError(
                "processing-request vertical datum conflicts with calibration DEM metadata"
            )
        if (
            elevation_reference != "unknown"
            and dem_metadata.elevation_reference != "unknown"
            and elevation_reference != dem_metadata.elevation_reference
        ):
            raise ValueError(
                "processing-request elevation reference conflicts with calibration DEM metadata"
            )
        if vertical_crs is None and dem_metadata.vertical_crs is not None:
            vertical_crs = dem_metadata.vertical_crs
            source = "calibration_dem_metadata"
        if vertical_datum is None and dem_metadata.vertical_datum is not None:
            vertical_datum = dem_metadata.vertical_datum
            source = "calibration_dem_metadata"
        if elevation_reference == "unknown" and dem_metadata.elevation_reference != "unknown":
            elevation_reference = dem_metadata.elevation_reference
            source = "calibration_dem_metadata"

    datum_resolved = bool(vertical_crs or vertical_datum) and elevation_reference != "unknown"
    return {
        "vertical_crs": vertical_crs,
        "vertical_datum": vertical_datum,
        "elevation_reference": elevation_reference,
        "metadata_source": source,
        "datum_resolved": datum_resolved,
        "absolute_elevation_claim": datum_resolved,
        "surface_product": "dsm",
        "calibration_dem_surface_type": request.dem_surface_type if dem_path is not None else None,
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
        cancellation_probe: CancellationProbe | None = None,
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
                    valid_pixel_fraction=float(details.get("valid_pixel_fraction", 1.0)),
                ),
                True,
            )

        started = time.perf_counter()
        raise_if_cancelled(cancellation_probe)
        manifest.record_stage(ProcessingStage.GEOMETRY, status="running")
        scene = infer_geometry_scene(
            request.source,
            self.prior,
            band_indices=request.band_indices,
            tile_size=request.tile_size,
            overlap=request.overlap,
            harmonize_overlaps=request.harmonize_overlaps,
            cancellation_probe=cancellation_probe,
        )
        raise_if_cancelled(cancellation_probe)
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
                "valid_pixel_fraction": scene.valid_pixel_fraction,
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
                valid_pixel_fraction=scene.valid_pixel_fraction,
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
        bias_sigma_px = _resolved_bias_sigma_px(request, target_gsd_m=target_gsd_m)
        common = {
            "dem_valid": dem_valid & np.isfinite(geometry.relative_height),
            "low_frequency_sigma_px": bias_sigma_px,
            "min_abs_anchor_correlation": request.min_dem_anchor_correlation,
            "max_anchor_rmse_m": request.max_dem_anchor_rmse_m,
            "max_normalized_rmse": request.max_dem_normalized_rmse,
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
            low_frequency_sigma_px=bias_sigma_px,
            min_abs_anchor_correlation=request.min_dem_anchor_correlation,
            max_anchor_rmse_m=request.max_dem_anchor_rmse_m,
            max_normalized_rmse=request.max_dem_normalized_rmse,
        )

    def _calibrate(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
    ) -> _CalibrationOutcome:
        dem_path = request.metric_dem_path
        gcps = request.gcps
        gcp_source_evidence = _gcp_source_evidence_payload(request) if gcps else None
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

            # DEM establishes broad terrain support and relief scale. Sparse GCPs are permitted to
            # correct only one global vertical-datum offset; re-fitting scale here would distort all
            # image-derived roofs, trees, and slopes from a handful of points.
            gcp_validation = validate_metric_dsm_with_gcps(
                dem_result.dsm,
                transform=transform,
                gcps=gcps,
                min_gcps=request.min_gcp_count,
                max_rmse_m=request.max_gcp_anchor_rmse_m,
                max_cross_validation_rmse_m=request.max_gcp_cross_validation_rmse_m,
            )
            return _CalibrationOutcome(
                dsm=gcp_validation.dsm,
                evidence={
                    "fusion_method": "dem_scale_then_gcp_robust_global_datum_offset",
                    "relief_rescaled_by_gcps": False,
                    "dem": dem_payload,
                    "gcp_validation": {
                        "source_evidence": gcp_source_evidence,
                        "gcp_count_supplied": len(gcps),
                        "offset_applied_m": gcp_validation.offset_applied_m,
                        "rmse_before_m": gcp_validation.rmse_before_m,
                        "rmse_after_m": gcp_validation.rmse_after_m,
                        "cross_validation_rmse_m": (
                            gcp_validation.cross_validation_rmse_m
                        ),
                        "gcp_residuals_before_m": (
                            gcp_validation.gcp_residuals_before_m.tolist()
                        ),
                        "gcp_residuals_after_m": (
                            gcp_validation.gcp_residuals_after_m.tolist()
                        ),
                        "spatial_coverage_fraction": (
                            gcp_validation.spatial_coverage_fraction
                        ),
                        "spatial_rank_ratio": gcp_validation.spatial_rank_ratio,
                        "semantics": "validation_and_global_vertical_datum_offset_only",
                    },
                },
                mode=CalibrationMode.DEM_GCP,
            )

        target_gsd_m = _mean_gsd(ground_sample_distance_m(request.source))
        gcp_bias_sigma_px = _resolved_bias_sigma_px(request, target_gsd_m=target_gsd_m)
        gcp_result = calibrate_relative_height_with_gcps(
            geometry.relative_height,
            transform=transform,
            gcps=gcps,
            low_frequency_sigma_px=gcp_bias_sigma_px,
            resolve_orientation=True,
            min_gcps=request.min_gcp_count,
            max_anchor_rmse_m=request.max_gcp_anchor_rmse_m,
            max_cross_validation_rmse_m=request.max_gcp_cross_validation_rmse_m,
        )
        return _CalibrationOutcome(
            dsm=gcp_result.dsm,
            evidence={
                "gcp": {
                    "source_evidence": gcp_source_evidence,
                    "calibration": gcp_result.calibration.model_dump(),
                    "gcp_count_supplied": len(gcps),
                    "gcp_residuals_m": gcp_result.gcp_residuals_m.tolist(),
                    "anchor_correlation_before": gcp_result.anchor_correlation_before,
                    "orientation_flipped": gcp_result.orientation_flipped,
                    "cross_validation_rmse_m": gcp_result.cross_validation_rmse_m,
                    "spatial_coverage_fraction": gcp_result.spatial_coverage_fraction,
                    "spatial_rank_ratio": gcp_result.spatial_rank_ratio,
                    "bias_sigma_px": gcp_bias_sigma_px,
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

    def run(
        self,
        request: ProcessingRequest,
        *,
        job_id: str | None = None,
        cancellation_probe: CancellationProbe | None = None,
    ) -> ProjectRunResult:
        if not request.source.is_file():
            raise FileNotFoundError(f"source raster does not exist: {request.source}")
        request.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = ProjectManifest.create_or_load(request.output_dir, request.source)
        current_stage: ProcessingStage | None = None

        source_hash = sha256_file(request.source)
        geometry_hash = canonical_json_hash(
            _geometry_config(
                request,
                self.estimator_decision,
                prior=self.prior,
                learned_refiner=self.learned_refiner,
            )
        )
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
            raise_if_cancelled(cancellation_probe)
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
            raise_if_cancelled(cancellation_probe)

            current_stage = ProcessingStage.GEOMETRY
            geometry, geometry_resumed = self._load_or_run_geometry(
                manifest,
                request,
                georeferenced=georeferenced,
                cancellation_probe=cancellation_probe,
            )
            raise_if_cancelled(cancellation_probe)

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
            raise_if_cancelled(cancellation_probe)
            vertical_reference = _vertical_reference_payload(request)
            absolute_elevation_claim = bool(vertical_reference["absolute_elevation_claim"])
            if not absolute_elevation_claim:
                manifest.add_warning(
                    "metric scale/offset calibration passed, but the vertical CRS/datum is not "
                    "fully declared; output is a metric calibrated DSM, not a datum-resolved "
                    "absolute-elevation claim"
                )
            dsm_path = request.output_dir / "products" / "dsm.tif"
            vertical_crs_tag = str(vertical_reference["vertical_crs"] or "unspecified")
            vertical_datum_tag = str(vertical_reference["vertical_datum"] or "unspecified")
            write_float_geotiff(
                dsm_path,
                calibration.dsm,
                template_path=request.source,
                description=(
                    "DepthWizard datum-resolved absolute Digital Surface Model (metres)"
                    if absolute_elevation_claim
                    else "DepthWizard metric calibrated Digital Surface Model; vertical datum unspecified"
                ),
                tags={
                    "DEPTHWIZARD_PRODUCT": (
                        "ABSOLUTE_DSM_METRES"
                        if absolute_elevation_claim
                        else "METRIC_DSM_VERTICAL_DATUM_UNSPECIFIED"
                    ),
                    "ELEVATION_UNITS": "metres",
                    "ABSOLUTE_ELEVATION_STATUS": (
                        "datum_resolved" if absolute_elevation_claim else "vertical_datum_unspecified"
                    ),
                    "VERTICAL_CRS": vertical_crs_tag,
                    "VERTICAL_DATUM": vertical_datum_tag,
                    "ELEVATION_REFERENCE": str(vertical_reference["elevation_reference"]),
                    "SURFACE_SEMANTICS": "digital_surface_model",
                    "CALIBRATION_MODE": calibration.mode.value,
                    "ESTIMATOR_PATH": self.estimator_decision.selected_path.value,
                },
            )
            _register_artifact(
                manifest,
                "dsm",
                dsm_path,
                semantics=(
                    "datum_resolved_absolute_digital_surface_model"
                    if absolute_elevation_claim
                    else "metric_calibrated_digital_surface_model_vertical_datum_unspecified"
                ),
                units="m",
            )
            calibration_document: dict[str, object] = {
                "schema": "depthwizard.calibration.v2",
                "mode": calibration.mode.value,
                "metric_claim": True,
                "absolute_elevation_claim": absolute_elevation_claim,
                "vertical_reference": vertical_reference,
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
                    "absolute_elevation_claim": absolute_elevation_claim,
                    "vertical_reference": vertical_reference,
                    "evidence": calibration.evidence,
                },
                elapsed_seconds=time.perf_counter() - started,
            )

            current_stage = ProcessingStage.EXPORT
            raise_if_cancelled(cancellation_probe)
            started = time.perf_counter()
            manifest.record_stage(ProcessingStage.EXPORT, status="running")
            source_gsd = ground_sample_distance_m(request.source)
            source_ground_jacobian = ground_pixel_jacobian_m(request.source)
            slope_path: Path | None = None
            if source_gsd is not None and source_ground_jacobian is not None:
                slope = slope_degrees(
                    calibration.dsm,
                    gsd_x=source_gsd[0],
                    gsd_y=source_gsd[1],
                    ground_jacobian_m=source_ground_jacobian,
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
                        "GROUND_GRADIENT_GEOMETRY": "local_east_north_2x2_jacobian",
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
                    "full local ground-pixel geometry unavailable; slope raster omitted rather "
                    "than assuming orthogonal pixel axes"
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
                    "absolute_elevation_claim": absolute_elevation_claim,
                    "vertical_reference": vertical_reference,
                    "elevation_units": "m",
                    "source_files_overwritten": False,
                },
                elapsed_seconds=time.perf_counter() - started,
            )
            manifest.record_stage(ProcessingStage.COMPLETE, status="completed")
            manifest.mark_status(ProjectRunStatus.COMPLETE, job_id=job_id)
            return _result_from_manifest(manifest, resumed=geometry_resumed)
        except CancellationRequested:
            if current_stage is not None:
                manifest.record_stage(
                    current_stage,
                    status="cancelled",
                    details={"reason": "operator_requested_cancellation"},
                )
            manifest.mark_status(ProjectRunStatus.CANCELLED, job_id=job_id)
            raise
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
