from __future__ import annotations

from pathlib import Path

import numpy as np

from depthwizard.calibration.confidence import model_confidence_to_uncertainty
from depthwizard.calibration.evidence import EvidenceCalibrationOutput, calibrate_relative_height_with_dem
from depthwizard.contracts import ProcessingRequest
from depthwizard.io.raster import ground_sample_distance_m, reproject_to_match
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime_base import (
    ProjectRunResult,
    SceneRefiner,
    _CalibrationOutcome,
    _GeometryState,
    _mean_gsd,
)
from depthwizard.pipeline.runtime_base import ProductionElevationRuntime as _BaseProductionElevationRuntime

__all__ = ["ProductionElevationRuntime", "ProjectRunResult", "SceneRefiner"]


class ProductionElevationRuntime(_BaseProductionElevationRuntime):
    """Production runtime with confidence-aware, physically scaled DEM calibration.

    Model-native confidence is never presented as a calibrated probability. When it has usable
    spatial variation, DepthWizard converts it monotonically to a conservative uncertainty weight
    that can only down-weight DEM anchors. Missing or degenerate confidence falls back to the
    established DEM-only weighting path rather than fabricating reliability.

    DEM calibration itself is stricter: both the optical target and DEM must expose trustworthy
    physical ground spacing. Without that pair, DepthWizard cannot know the DEM's effective support
    relative to the optical grid, so it fails closed instead of treating upsampled DEM pixels as
    independent metric evidence.
    """

    _last_dem_confidence_weighting: dict[str, object] | None = None

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
        if target_gsd_m is None or dem_effective_gsd_m is None:
            raise ValueError(
                "DEM calibration requires trustworthy physical ground sample distance for both "
                "the optical source and DEM; metric scale was not guessed"
            )

        weighting = model_confidence_to_uncertainty(geometry.confidence)
        self._last_dem_confidence_weighting = weighting.evidence()
        if geometry.confidence is not None and not weighting.active:
            manifest.add_warning(
                "model-native confidence could not support defensible calibration weighting "
                f"({weighting.reason}); DEM calibration continued without confidence weighting"
            )

        return calibrate_relative_height_with_dem(
            geometry.relative_height,
            aligned_dem,
            dem_valid=dem_valid & np.isfinite(geometry.relative_height),
            uncertainty=weighting.uncertainty if weighting.active else None,
            low_frequency_sigma_px=request.low_frequency_sigma_px,
            target_gsd_m=target_gsd_m,
            dem_effective_gsd_m=dem_effective_gsd_m,
        )

    def _calibrate(
        self,
        manifest: ProjectManifest,
        request: ProcessingRequest,
        geometry: _GeometryState,
    ) -> _CalibrationOutcome:
        self._last_dem_confidence_weighting = None
        outcome = super()._calibrate(manifest, request, geometry)
        if request.metric_dem_path is None or self._last_dem_confidence_weighting is None:
            return outcome

        evidence = dict(outcome.evidence)
        dem_payload = evidence.get("dem")
        if isinstance(dem_payload, dict):
            dem_payload = dict(dem_payload)
            dem_payload["confidence_weighting"] = dict(self._last_dem_confidence_weighting)
            evidence["dem"] = dem_payload
        return _CalibrationOutcome(dsm=outcome.dsm, evidence=evidence, mode=outcome.mode)
