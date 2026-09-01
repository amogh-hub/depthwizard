from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt

from depthwizard.analysis.project_structure import estimate_project_structure_height
from depthwizard.contracts import NormalizedPoint, ProjectStructureHeightRequest
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file


def _write_surface(path: Path, values: np.ndarray, *, gsd_m: float = 1.0) -> None:
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
        transform=from_origin(500000, 1400000, gsd_m, gsd_m),
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


def _sloping_project(tmp_path: Path) -> Path:
    source = tmp_path / "rgb-slope.tif"
    source.touch()
    project = tmp_path / "project-slope"
    rows, cols = np.mgrid[:80, :80]
    ground = 300.0 + 0.45 * cols + 0.75 * rows
    values = ground.astype(np.float32)
    values[28:52, 28:52] += 12.0
    values[20:23, 35:38] += 20.0
    values[56:59, 42:45] += 16.0
    surface = project / "products" / "dsm.tif"
    _write_surface(surface, values)
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        surface,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(surface),
    )
    return project


def _high_resolution_project(tmp_path: Path) -> Path:
    source = tmp_path / "rgb-high-resolution.tif"
    source.touch()
    project = tmp_path / "project-high-resolution"
    gsd_m = 0.05
    values = np.full((500, 500), 100.0, dtype=np.float32)
    structure_mask = np.zeros_like(values, dtype=bool)
    structure_mask[170:330, 170:330] = True
    values[structure_mask] = 110.0
    distance_m = distance_transform_edt(~structure_mask, sampling=(gsd_m, gsd_m))
    edge_contamination = (~structure_mask) & (distance_m <= 1.0)
    values[edge_contamination] = 108.0

    surface = project / "products" / "dsm.tif"
    _write_surface(surface, values, gsd_m=gsd_m)
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        surface,
        semantics="absolute_digital_surface_model",
        units="m",
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
    assert result.ground_candidate_pixels >= result.ground_pixels
    assert result.roof_inset_m == pytest.approx(0.5)
    assert result.ground_inner_buffer_m == pytest.approx(1.5)
    assert result.ground_outer_buffer_m == pytest.approx(8.0)
    assert result.ground_inlier_fraction >= 0.99
    assert result.ground_sector_coverage == 1.0
    assert result.measurement_quality == "high"
    assert result.warnings == []
    assert "not an automatic building classification" in result.semantics
    assert "ground plane" in result.semantics
    assert "physical" in result.semantics


def test_structure_height_uses_local_ground_plane_on_hillside(tmp_path: Path) -> None:
    project = _sloping_project(tmp_path)
    polygon = [
        NormalizedPoint(x=28 / 79, y=28 / 79),
        NormalizedPoint(x=51 / 79, y=28 / 79),
        NormalizedPoint(x=51 / 79, y=51 / 79),
        NormalizedPoint(x=28 / 79, y=51 / 79),
    ]
    result = estimate_project_structure_height(
        ProjectStructureHeightRequest(project_dir=project, polygon=polygon, ring_pixels=10)
    )

    assert result.structure_height_m == pytest.approx(12.0, abs=0.15)
    assert result.structure_pixels > 400
    assert result.ground_pixels > 100
    assert result.measurement_quality == "high"
    assert result.warnings == []


def test_project_structure_height_uses_metric_support_at_five_centimetres(
    tmp_path: Path,
) -> None:
    project = _high_resolution_project(tmp_path)
    polygon = [
        NormalizedPoint(x=170 / 499, y=170 / 499),
        NormalizedPoint(x=329 / 499, y=170 / 499),
        NormalizedPoint(x=329 / 499, y=329 / 499),
        NormalizedPoint(x=170 / 499, y=329 / 499),
    ]
    result = estimate_project_structure_height(
        ProjectStructureHeightRequest(project_dir=project, polygon=polygon, ring_pixels=8)
    )

    assert result.structure_height_m == pytest.approx(10.0, abs=1e-4)
    assert result.ground_elevation_m == pytest.approx(100.0, abs=1e-4)
    assert result.ring_pixels >= 150
    assert result.structure_pixels > 10_000
    assert result.ground_pixels > 10_000
    assert result.ground_candidate_pixels > result.ground_pixels
    assert result.ground_inlier_fraction > 0.90
    assert result.ground_sector_coverage == 1.0
    assert result.measurement_quality == "high"
    assert result.warnings == []
    assert "1.50-8.00 m annulus" in result.semantics
    assert "ring_pixels field does not control" in result.semantics


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
