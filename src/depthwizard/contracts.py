from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class InputKind(str, Enum):
    NON_GEOREFERENCED = "non_georeferenced"
    GEOREFERENCED = "georeferenced"


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
    source: Path
    output_dir: Path
    srtm_path: Path | None = None
    gcps: list[GroundControlPoint] = Field(default_factory=list)
    requested_output: Literal["rdsm", "dsm"] | None = None

    @model_validator(mode="after")
    def validate_requested_output(self) -> ProcessingRequest:
        suffix = self.source.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            raise ValueError("DepthWizard accepts PNG, JPG/JPEG, TIFF, and GeoTIFF inputs")
        return self
