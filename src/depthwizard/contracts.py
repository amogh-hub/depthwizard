from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class InputKind(str, Enum):
    NON_GEOREFERENCED = "non_georeferenced"
    GEOREFERENCED = "georeferenced"


class CalibrationMode(str, Enum):
    NONE = "none"
    DEM = "dem"
    GCP = "gcp"
    DEM_GCP = "dem_gcp"


class ProjectRunStatus(str, Enum):
    CREATED = "created"
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_FOR_CALIBRATION = "waiting_for_calibration"
    COMPLETE = "complete"
    FAILED = "failed"


class RasterMetadata(BaseModel):
    path: Path
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    count: int = Field(gt=0)
    dtype: str
    crs: str | None = None
    transform: tuple[float, float, float, float, float, float] | None = None
    nodata: float | None = None
    ground_sample_distance_x: float | None = None
    ground_sample_distance_y: float | None = None

    @property
    def input_kind(self) -> InputKind:
        return InputKind.GEOREFERENCED if self.crs and self.transform else InputKind.NON_GEOREFERENCED


class GroundControlPoint(BaseModel):
    x: float
    y: float
    elevation_m: float
    weight: float = Field(default=1.0, gt=0.0)


class EvaluationMetrics(BaseModel):
    valid_pixels: int = Field(ge=0)
    mae_m: float
    rmse_m: float
    pearson_r: float | None
    mean_bias_m: float
    median_abs_error_m: float
    p90_abs_error_m: float
    p95_abs_error_m: float


class SlopeMetrics(BaseModel):
    valid_pixels: int = Field(ge=0)
    mae_degrees: float
    rmse_degrees: float
    p95_abs_error_degrees: float


class ErrorConfidenceBin(BaseModel):
    lower_confidence: float
    upper_confidence: float
    valid_pixels: int = Field(ge=0)
    mean_confidence: float
    mae_m: float
    rmse_m: float


class ReliabilityDiagnostics(BaseModel):
    available: bool
    semantics: str
    valid_pixels: int = Field(default=0, ge=0)
    confidence_abs_error_pearson_r: float | None = None
    bins: list[ErrorConfidenceBin] = Field(default_factory=list)


class ReferenceValidationRequest(BaseModel):
    project_dir: Path
    reference_path: Path
    reference_label: str | None = None
    min_valid_pixels: int = Field(default=128, ge=2)


class ReferenceValidationReport(BaseModel):
    schema_version: int = 1
    project_id: str
    prediction_sha256: str
    reference_path: Path
    reference_sha256: str
    reference_label: str | None = None
    independence_check: str
    alignment: str
    valid_pixels: int = Field(ge=0)
    coverage_fraction: float = Field(ge=0.0, le=1.0)
    elevation: EvaluationMetrics
    slope: SlopeMetrics
    reliability: ReliabilityDiagnostics
    artifacts: dict[str, str]
    warnings: list[str] = Field(default_factory=list)


class NormalizedPoint(BaseModel):
    """Image-relative coordinate independent of desktop preview resolution."""

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)


class RasterSample(BaseModel):
    available: bool
    value: float | None = None
    units: str | None = None
    semantics: str


class ProjectProbeRequest(BaseModel):
    project_dir: Path
    point: NormalizedPoint


class ProjectProbeResult(BaseModel):
    project_id: str
    point: NormalizedPoint
    pixel_col: int = Field(ge=0)
    pixel_row: int = Field(ge=0)
    map_x: float | None = None
    map_y: float | None = None
    longitude: float | None = None
    latitude: float | None = None
    surface_product: Literal["dsm", "rdsm"]
    surface: RasterSample
    slope: RasterSample
    reference: RasterSample
    residual: RasterSample
    confidence: RasterSample


class ProjectProfileRequest(BaseModel):
    project_dir: Path
    start: NormalizedPoint
    end: NormalizedPoint
    samples: int = Field(default=128, ge=2, le=512)


class ProfileSample(BaseModel):
    fraction: float = Field(ge=0.0, le=1.0)
    point: NormalizedPoint
    distance_pixels: float = Field(ge=0.0)
    distance_m: float | None = Field(default=None, ge=0.0)
    surface: RasterSample
    slope: RasterSample
    reference: RasterSample
    residual: RasterSample
    confidence: RasterSample


class ProjectProfileResult(BaseModel):
    project_id: str
    surface_product: Literal["dsm", "rdsm"]
    start: NormalizedPoint
    end: NormalizedPoint
    sample_count: int = Field(ge=2)
    horizontal_distance_pixels: float = Field(ge=0.0)
    horizontal_distance_m: float | None = Field(default=None, ge=0.0)
    vertical_delta: float | None = None
    vertical_units: str | None = None
    minimum_surface: float | None = None
    maximum_surface: float | None = None
    elevation_gain: float | None = Field(default=None, ge=0.0)
    elevation_loss: float | None = Field(default=None, ge=0.0)
    samples: list[ProfileSample]
    semantics: str


class CalibrationResult(BaseModel):
    scale: float
    offset: float
    rmse_anchor: float
    median_abs_residual: float = 0.0
    anchors_used: int = 0
    iterations: int
    converged: bool
    method: str = "robust_affine_huber_irls"


class ProcessingRequest(BaseModel):
    """Permanent end-to-end project processing contract.

    ``srtm_path`` is retained for compatibility with early callers. New code should use
    ``dem_path`` because the calibration source may be SRTM, Copernicus DEM, or another explicit
    geospatial DEM product. Supplying both names is rejected rather than guessed.
    """

    source: Path
    output_dir: Path
    dem_path: Path | None = None
    srtm_path: Path | None = None
    gcps: list[GroundControlPoint] = Field(default_factory=list)
    requested_output: Literal["rdsm", "dsm"] | None = None
    band_indices: tuple[int, int, int] = (1, 2, 3)
    tile_size: int = Field(default=1024, ge=256)
    overlap: int = Field(default=128, ge=0)
    harmonize_overlaps: bool = True
    low_frequency_sigma_px: float = Field(default=24.0, ge=0.0)

    @property
    def metric_dem_path(self) -> Path | None:
        return self.dem_path or self.srtm_path

    @model_validator(mode="after")
    def validate_requested_output(self) -> ProcessingRequest:
        suffix = self.source.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            raise ValueError("DepthWizard accepts PNG, JPG/JPEG, TIFF, and GeoTIFF inputs")
        if self.dem_path is not None and self.srtm_path is not None:
            raise ValueError("supply either dem_path or legacy srtm_path, not both")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")
        if len(set(self.band_indices)) != 3 or any(index < 1 for index in self.band_indices):
            raise ValueError("band_indices must contain three distinct positive 1-based bands")
        return self
