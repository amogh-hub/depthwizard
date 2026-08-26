from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine, from_origin

from depthwizard.contracts import (
    NormalizedPoint,
    ProjectProbeRequest,
    ProjectProfileRequest,
)
from depthwizard.evaluation.project_analysis import probe_project, sample_project_profile
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file


def _write_surface(
    path: Path,
    values: np.ndarray,
    *,
    units: str = "m",
    crs: str = "EPSG:32643",
    transform: Affine | None = None,
) -> None:
    del units
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform or from_origin(500000, 1400000, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values.astype(np.float32), 1)


def _project_with_analytical_products(tmp_path: Path) -> tuple[Path, np.ndarray]:
    project = tmp_path / "project"
    source = tmp_path / "rgb.tif"
    source.touch()
    y, x = np.mgrid[:11, :11]
    dsm_values = (100.0 + x + 2.0 * y).astype(np.float32)
    dsm = project / "products" / "dsm.tif"
    slope = project / "products" / "slope.tif"
    reference = project / "products" / "reference-aligned.tif"
    residual = project / "products" / "residual.tif"
    _write_surface(dsm, dsm_values)
    _write_surface(slope, np.full_like(dsm_values, 12.5))
    _write_surface(reference, dsm_values - 1.0)
    _write_surface(residual, np.ones_like(dsm_values))

    manifest = ProjectManifest.create_or_load(project, source)
    for name, path, semantics, units in (
        ("dsm", dsm, "absolute_digital_surface_model", "m"),
        ("slope", slope, "surface_slope", "degrees"),
        ("reference", reference, "aligned_evaluation_reference_dsm", "m"),
        ("residual", residual, "prediction_minus_reference_residual", "m"),
    ):
        manifest.register_artifact(
            name,
            path,
            semantics=semantics,
            units=units,
            sha256=sha256_file(path),
        )
    return project, dsm_values


def test_probe_samples_synchronized_project_products(tmp_path: Path) -> None:
    project, dsm_values = _project_with_analytical_products(tmp_path)
    result = probe_project(
        ProjectProbeRequest(
            project_dir=project,
            point=NormalizedPoint(x=0.5, y=0.5),
        )
    )

    assert result.pixel_col == 5
    assert result.pixel_row == 5
    assert result.surface.available is True
    assert result.surface.value == float(dsm_values[5, 5])
    assert result.slope.value == 12.5
    assert result.reference.value == float(dsm_values[5, 5] - 1.0)
    assert result.residual.value == 1.0
    assert result.confidence.available is False
    assert result.map_x is not None
    assert result.longitude is not None


def test_profile_reports_metric_distance_and_surface_delta(tmp_path: Path) -> None:
    project, _ = _project_with_analytical_products(tmp_path)
    result = sample_project_profile(
        ProjectProfileRequest(
            project_dir=project,
            start=NormalizedPoint(x=0.0, y=0.5),
            end=NormalizedPoint(x=1.0, y=0.5),
            samples=11,
        )
    )

    assert result.sample_count == 11
    assert result.horizontal_distance_pixels == 10.0
    assert result.horizontal_distance_m is not None
    assert 9.9 < result.horizontal_distance_m < 10.1
    assert result.vertical_delta == 10.0
    assert result.vertical_units == "m"
    assert result.elevation_gain == 10.0
    assert result.elevation_loss == 0.0
    assert result.minimum_surface == 110.0
    assert result.maximum_surface == 120.0
    assert result.samples[-1].reference.value == 119.0


def test_projected_profile_uses_declared_linear_units_without_global_roundtrip(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = tmp_path / "rgb.tif"
    source.touch()
    dsm = project / "products" / "dsm.tif"
    values = np.arange(121, dtype=np.float32).reshape(11, 11)
    _write_surface(
        dsm,
        values,
        crs="EPSG:32632",
        transform=from_origin(100_000_000, 100_000_000, 1.0, 1.0),
    )

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )

    profile = sample_project_profile(
        ProjectProfileRequest(
            project_dir=project,
            start=NormalizedPoint(x=0.0, y=0.5),
            end=NormalizedPoint(x=1.0, y=0.5),
            samples=11,
        )
    )
    probe = probe_project(
        ProjectProbeRequest(
            project_dir=project,
            point=NormalizedPoint(x=0.5, y=0.5),
        )
    )

    assert profile.horizontal_distance_m == 10.0
    assert probe.map_x is not None
    assert probe.map_y is not None
    assert probe.longitude is None
    assert probe.latitude is None


def test_relative_project_never_invents_metric_distance(tmp_path: Path) -> None:
    project = tmp_path / "project"
    source = tmp_path / "rgb.png"
    source.touch()
    rdsm = project / "products" / "rdsm.tif"
    rdsm.parent.mkdir(parents=True, exist_ok=True)
    values = np.arange(25, dtype=np.float32).reshape(5, 5)
    with rasterio.open(
        rdsm,
        "w",
        driver="GTiff",
        height=5,
        width=5,
        count=1,
        dtype="float32",
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "rdsm",
        rdsm,
        semantics="relative_digital_surface_model",
        units="relative",
        sha256=sha256_file(rdsm),
    )
    result = sample_project_profile(
        ProjectProfileRequest(
            project_dir=project,
            start=NormalizedPoint(x=0.0, y=0.0),
            end=NormalizedPoint(x=1.0, y=1.0),
            samples=5,
        )
    )

    assert result.horizontal_distance_m is None
    assert result.vertical_units == "relative"
    assert result.horizontal_distance_pixels > 0
