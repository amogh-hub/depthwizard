from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.analysis.project_structure import estimate_project_structure_height
from depthwizard.contracts import NormalizedPoint, ProjectStructureHeightRequest
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file


def _write_surface(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _project(tmp_path: Path, *, metric: bool = True) -> Path:
    source = tmp_path / "rgb.tif"
    source.touch()
    project = tmp_path / "project"
    values = np.full((64, 64), 100.0, dtype=np.float32)
    values[24:40, 24:40] = 112.0
    product_name = "dsm" if metric else "rdsm"
    surface = project / "products" / f"{product_name}.tif"
    _write_surface(surface, values)
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        product_name,
        surface,
        semantics="absolute_digital_surface_model" if metric else "relative_digital_surface_model",
        units="m" if metric else "relative",
        sha256=sha256_file(surface),
    )
    return project


def test_structure_height_uses_explicit_footprint_and_local_ground(tmp_path: Path) -> None:
    project = _project(tmp_path)
    polygon = [
        NormalizedPoint(x=24 / 63, y=24 / 63),
        NormalizedPoint(x=39 / 63, y=24 / 63),
        NormalizedPoint(x=39 / 63, y=39 / 63),
        NormalizedPoint(x=24 / 63, y=39 / 63),
    ]
    result = estimate_project_structure_height(
        ProjectStructureHeightRequest(project_dir=project, polygon=polygon, ring_pixels=6)
    )

    assert abs(result.top_elevation_m - 112.0) < 1e-6
    assert abs(result.ground_elevation_m - 100.0) < 1e-6
    assert abs(result.structure_height_m - 12.0) < 1e-6
    assert result.structure_pixels >= 200
    assert result.ground_pixels >= 8
    assert result.warnings == []
    assert "not an automatic building classification" in result.semantics


def test_structure_height_rejects_relative_only_project(tmp_path: Path) -> None:
    project = _project(tmp_path, metric=False)
    polygon = [
        NormalizedPoint(x=0.3, y=0.3),
        NormalizedPoint(x=0.7, y=0.3),
        NormalizedPoint(x=0.7, y=0.7),
    ]
    with pytest.raises(ValueError, match="absolute metric DSM"):
        estimate_project_structure_height(
            ProjectStructureHeightRequest(project_dir=project, polygon=polygon)
        )
