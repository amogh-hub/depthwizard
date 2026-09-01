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


class GroundControlPointEvidence(BaseModel):
    """Identity of an analyst-supplied GCP evidence file used to build ``gcps``."""

    source_path: Path
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class GroundControlPointFileReport(BaseModel):
    source_path: Path
    sha256: str
    point_count: int = Field(ge=2)
    minimum_elevation_m: float
    maximum_elevation_m: float
    points: list[GroundControlPoint] = Field(min_length=2)
    semantics: str


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


class ProjectStructureHeightRequest(BaseModel):
    project_dir: Path
    polygon: list[NormalizedPoint] = Field(min_length=3, max_length=64)
    ring_pixels: int = Field(
        default=8,
        ge=1,
        le=128,
        description=(
            "Deprecated compatibility field. Project-level metric measurement derives physical "
            "ground support from trustworthy GSD rather than this pixel count."
        ),
    )
    min_structure_pixels: int = Field(default=4, ge=1)
    min_ground_pixels: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_polygon(self) -> ProjectStructureHeightRequest:
        unique = {(round(point.x, 9), round(point.y, 9)) for point in self.polygon}
        if len(unique) < 3:
            raise ValueError("structure polygon requires at least three distinct points")
        return self


class ProjectStructureHeightResult(BaseModel):
    project_id: str
    polygon: list[NormalizedPoint]
    ring_pixels: int = Field(
        ge=1,
        description="Compatibility display of the effective outer support radius in raster pixels.",
    )
    top_elevation_m: float
    ground_elevation_m: float
    structure_height_m: float
    structure_pixels: int = Field(ge=1)
    ground_pixels: int = Field(ge=1)
    ground_candidate_pixels: int = Field(ge=1)
    roof_inset_m: float = Field(ge=0.0)
    ground_inner_buffer_m: float = Field(ge=0.0)
    ground_outer_buffer_m: float = Field(gt=0.0)
    ground_inlier_fraction: float = Field(ge=0.0, le=1.0)
    ground_sector_coverage: float = Field(ge=0.0, le=1.0)
    roof_dispersion_m: float = Field(ge=0.0)
    ground_residual_sigma_m: float = Field(ge=0.0)
    local_height_dispersion_m: float = Field(ge=0.0)
    measurement_quality: Literal["high", "moderate", "low"]
    warnings: list[str] = Field(default_factory=list)
    semantics: str


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


class ProjectMeshBuildRequest(BaseModel):
    project_dir: Path
    max_finest_samples: int = Field(default=512, ge=128, le=1024)
    lod_levels: int = Field(default=4, ge=2, le=6)


class TerrainLodArtifact(BaseModel):
    level: int = Field(ge=0)
    stride: int = Field(ge=1)
    path: Path
    sha256: str
    vertices: int = Field(ge=3)
    faces: int = Field(ge=1)
    width_samples: int = Field(ge=2)
    height_samples: int = Field(ge=2)


class ProjectMeshReport(BaseModel):
    schema_version: int = 1
    project_id: str
    surface_product: Literal["dsm", "rdsm"]
    surface_sha256: str
    texture_sha256: str
    build_config_sha256: str
    horizontal_units: Literal["m", "px"]
    vertical_units: Literal["m", "relative"]
    gsd_x: float = Field(gt=0.0)
    gsd_y: float = Field(gt=0.0)
    raster_width: int = Field(ge=2)
    raster_height: int = Field(ge=2)
    valid_pixels: int = Field(ge=4)
    minimum_elevation: float
    maximum_elevation: float
    relief: float = Field(ge=0.0)
    lods: list[TerrainLodArtifact] = Field(min_length=1)
    mesh_manifest_path: Path
    semantics: str


class ProjectExportRequest(BaseModel):
    project_dir: Path
    include_source: bool = False
    include_mesh: bool = True
    include_validation: bool = True


class ProjectExportFile(BaseModel):
    arcname: str
    source_path: Path
    sha256: str
    bytes: int = Field(ge=0)
    semantics: str
    units: str | None = None


class ProjectExportReport(BaseModel):
    schema_version: int = 1
    project_id: str
    bundle_path: Path
    bundle_sha256: str
    bundle_bytes: int = Field(ge=1)
    project_manifest_sha256: str
    export_manifest_path: Path
    include_source: bool
    include_mesh: bool
    include_validation: bool
    files: list[ProjectExportFile] = Field(min_length=1)
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

    Resource-affecting inputs are explicitly bounded. DepthWizard is a local scientific workstation,
    but authenticated local callers still must not be able to request arbitrarily large inference
    tiles, evidence arrays, or smoothing kernels that bypass the runtime's bounded-memory design.
    """

    source: Path
    output_dir: Path
    dem_path: Path | None = None
    srtm_path: Path | None = None
    gcps: list[GroundControlPoint] = Field(default_factory=list, max_length=10_000)
    gcp_evidence: GroundControlPointEvidence | None = None
    requested_output: Literal["rdsm", "dsm"] | None = None
    band_indices: tuple[int, int, int] = (1, 2, 3)
    tile_size: int = Field(default=1024, ge=256, le=4096)
    overlap: int = Field(default=128, ge=0, le=4095)
    harmonize_overlaps: bool = True
    low_frequency_sigma_px: float = Field(default=24.0, ge=0.0, le=4096.0)

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
        if self.gcp_evidence is not None and not self.gcps:
            raise ValueError("gcp_evidence cannot be supplied without GCP points")
        if self.overlap >= self.tile_size:
            raise ValueError("overlap must be smaller than tile_size")
        if len(set(self.band_indices)) != 3 or any(index < 1 for index in self.band_indices):
            raise ValueError("band_indices must contain three distinct positive 1-based bands")
        return self
