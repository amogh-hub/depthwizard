from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from depthwizard.contracts import ProcessingRequest
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.pipeline.runtime import ProductionElevationRuntime
from depthwizard.pipeline.runtime_base import _GeometryState


def _write_raster(path: Path, values: np.ndarray, *, count: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = values.shape[-2:]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        if count == 1:
            dst.write(values.astype(np.float32), 1)
        else:
            for band in range(1, count + 1):
                dst.write(values.astype(np.float32), band)


def test_runtime_records_active_model_native_confidence_weighting_for_dem_calibration(
    tmp_path: Path,
) -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.01 * x + 0.02 * y).astype(np.float32)
    confidence = (0.2 + 0.8 * (x + y) / float(2 * 63)).astype(np.float32)
    dem_values = (120.0 + 4.0 * relative).astype(np.float32)

    source = tmp_path / "rgb.tif"
    dem = tmp_path / "dem.tif"
    project_dir = tmp_path / "project"
    _write_raster(source, np.ones((64, 64), dtype=np.float32), count=3)
    _write_raster(dem, dem_values)

    request = ProcessingRequest(
        source=source,
        output_dir=project_dir,
        dem_path=dem,
        requested_output="dsm",
    )
    manifest = ProjectManifest.create_or_load(project_dir, source)
    geometry = _GeometryState(
        relative_height=relative,
        confidence=confidence,
        model_id="TEST-PRIOR",
        tile_count=1,
        harmonized_tiles=0,
    )

    outcome = ProductionElevationRuntime()._calibrate(manifest, request, geometry)
    dem_evidence = outcome.evidence["dem"]
    assert isinstance(dem_evidence, dict)
    weighting = dem_evidence["confidence_weighting"]
    assert isinstance(weighting, dict)
    assert weighting["active"] is True
    assert weighting["source"] == "model_native_confidence"
    assert weighting["probability_calibrated"] is False
    assert "not_probability_calibrated" in str(weighting["semantics"])


def test_runtime_falls_back_truthfully_when_native_confidence_is_degenerate(tmp_path: Path) -> None:
    y, x = np.mgrid[:64, :64]
    relative = (0.01 * x + 0.02 * y).astype(np.float32)
    dem_values = (80.0 + 3.0 * relative).astype(np.float32)
    source = tmp_path / "rgb.tif"
    dem = tmp_path / "dem.tif"
    project_dir = tmp_path / "project"
    _write_raster(source, np.ones((64, 64), dtype=np.float32), count=3)
    _write_raster(dem, dem_values)

    request = ProcessingRequest(source=source, output_dir=project_dir, dem_path=dem)
    manifest = ProjectManifest.create_or_load(project_dir, source)
    geometry = _GeometryState(
        relative_height=relative,
        confidence=np.full_like(relative, 0.8),
        model_id="TEST-PRIOR",
        tile_count=1,
        harmonized_tiles=0,
    )

    outcome = ProductionElevationRuntime()._calibrate(manifest, request, geometry)
    dem_evidence = outcome.evidence["dem"]
    assert isinstance(dem_evidence, dict)
    weighting = dem_evidence["confidence_weighting"]
    assert isinstance(weighting, dict)
    assert weighting["active"] is False
    assert weighting["reason"] == "degenerate_model_native_confidence_range"
    assert any("continued without confidence weighting" in warning for warning in manifest.warnings)
