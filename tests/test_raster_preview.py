from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.raster_preview import render_project_layer_preview


def _write_rgb(path: Path) -> None:
    y, x = np.mgrid[:64, :64]
    data = np.stack(
        (
            20 + x,
            40 + y,
            60 + (x + y) // 2,
        ),
        axis=0,
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def _write_scalar(path: Path) -> None:
    y, x = np.mgrid[:64, :64]
    values = (100.0 + 0.5 * x + 0.25 * y).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=1,
        dtype="float32",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
        nodata=-9999.0,
    ) as dst:
        dst.write(values, 1)


def test_project_preview_renders_optical_and_metric_dsm(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    _write_rgb(source)
    _write_scalar(dsm)
    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )

    optical = render_project_layer_preview(project, "optical", max_side=256)
    metric = render_project_layer_preview(project, "dsm", max_side=256)

    assert optical.startswith(b"\x89PNG\r\n\x1a\n")
    assert metric.startswith(b"\x89PNG\r\n\x1a\n")
    assert source.is_file()
    assert dsm.is_file()


def test_project_preview_rejects_missing_scientific_layer(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    _write_rgb(source)
    ProjectManifest.create_or_load(project, source)

    with pytest.raises(FileNotFoundError, match="residual"):
        render_project_layer_preview(project, "residual")


def test_hillshade_preview_rejects_all_nodata_surface(tmp_path: Path) -> None:
    source = tmp_path / "rgb.tif"
    project = tmp_path / "project"
    dsm = project / "products" / "dsm.tif"
    _write_rgb(source)
    _write_scalar(dsm)
    with rasterio.open(dsm, "r+") as dataset:
        dataset.write(np.full((64, 64), -9999.0, dtype=np.float32), 1)

    manifest = ProjectManifest.create_or_load(project, source)
    manifest.register_artifact(
        "dsm",
        dsm,
        semantics="absolute_digital_surface_model",
        units="m",
        sha256=sha256_file(dsm),
    )

    with pytest.raises(ValueError, match="no valid surface pixels"):
        render_project_layer_preview(project, "hillshade", max_side=256)
