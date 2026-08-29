from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import rasterio
import trimesh
from PIL import Image

from depthwizard.contracts import ProjectMeshBuildRequest
from depthwizard.mesh.project_mesh import build_project_mesh
from depthwizard.pipeline.project import ProjectManifest
from depthwizard.provenance.manifest import sha256_file
from depthwizard.visualization.raster_preview import _hillshade_rgb
from depthwizard.visualization.relative_scale import (
    RELATIVE_DISPLAY_RELIEF_FRACTION,
    relative_display_vertical_scale,
)


def _fixture_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_png(path: Path, *, height: int = 96, width: int = 144) -> None:
    yy, xx = np.mgrid[0:height, 0:width]
    rgb = np.stack(
        [
            (40 + xx) % 255,
            (70 + yy * 2) % 255,
            (100 + xx // 2 + yy // 3) % 255,
        ],
        axis=-1,
    ).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(path)


def _write_rdsm(path: Path, *, height: int = 96, width: int = 144) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    # Deliberately tiny dimensionless relief compared with the pixel-space XY footprint. Without
    # display normalization this reproduces the flat-sheet defect seen with PNG/JPG input.
    rdsm = (
        0.40
        + 0.34 * np.sin(xx / 17.0)
        + 0.22 * np.cos(yy / 14.0)
        + 0.16 * yy / max(height - 1, 1)
    ).astype(np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
    ) as dst:
        dst.write(rdsm, 1)
        dst.update_tags(
            DEPTHWIZARD_PRODUCT="RELATIVE_DSM_DIMENSIONLESS",
            ELEVATION_UNITS="relative",
            ABSOLUTE_ELEVATION_STATUS="not_calibrated",
        )
    return rdsm


def _relative_project(tmp_path: Path) -> tuple[Path, np.ndarray]:
    source = tmp_path / "source.png"
    project_dir = tmp_path / "project"
    products = project_dir / "products"
    products.mkdir(parents=True)
    rdsm_path = products / "rdsm.tif"
    _write_png(source)
    rdsm = _write_rdsm(rdsm_path)

    manifest = ProjectManifest.create_or_load(project_dir, source)
    manifest.set_identity(
        source_sha256=sha256_file(source),
        input_kind="non_georeferenced",
        geometry_config_sha256=_fixture_sha("relative-display-geometry"),
        run_config_sha256=_fixture_sha("relative-display-run"),
    )
    manifest.register_artifact(
        "rdsm",
        rdsm_path,
        semantics="dimensionless_relative_surface_height",
        units="relative",
        sha256=sha256_file(rdsm_path),
    )
    return project_dir, rdsm


def test_relative_display_scale_targets_visible_robust_relief() -> None:
    yy, xx = np.mgrid[0:100, 0:160]
    values = (0.4 + 0.3 * np.sin(xx / 19.0) + 0.2 * np.cos(yy / 13.0)).astype(np.float32)
    valid = np.ones(values.shape, dtype=bool)
    scale = relative_display_vertical_scale(values, valid, span_x=159.0, span_y=99.0)
    low, high = np.percentile(values, [2.0, 98.0])
    displayed_robust_relief = float(high - low) * scale

    assert scale > 1.0
    assert np.isclose(
        displayed_robust_relief,
        99.0 * RELATIVE_DISPLAY_RELIEF_FRACTION,
        rtol=1e-6,
    )


def test_rdsm_mesh_has_visible_relief_without_mutating_raw_semantics(tmp_path: Path) -> None:
    project_dir, raw_rdsm = _relative_project(tmp_path)
    raw_relief = float(np.max(raw_rdsm) - np.min(raw_rdsm))

    report = build_project_mesh(
        ProjectMeshBuildRequest(project_dir=project_dir, max_finest_samples=128, lod_levels=2)
    )
    assert report.surface_product == "rdsm"
    assert report.horizontal_units == "px"
    assert report.vertical_units == "relative"
    assert report.relief == raw_relief
    assert "display_normalized_z" in report.semantics

    manifest = ProjectManifest.load(project_dir)
    mesh_details = manifest.stages["mesh"]["details"]
    assert mesh_details["display_only_normalization"] is True
    assert float(mesh_details["display_vertical_scale"]) > 1.0

    loaded = trimesh.load(report.lods[0].path, force="scene")
    bounds = np.asarray(loaded.bounds, dtype=np.float64)
    spans = bounds[1] - bounds[0]
    displayed_relief = float(spans[1])
    shorter_horizontal = float(min(spans[0], spans[2]))

    # The exact full-range relief can exceed the P02-P98 target slightly, but it must now be
    # unambiguously visible rather than ~0.1% of the scene width.
    assert displayed_relief / shorter_horizontal >= 0.08
    assert displayed_relief / shorter_horizontal <= 0.25


def test_relative_hillshade_uses_display_scale_without_claiming_slope(tmp_path: Path) -> None:
    project_dir, _raw_rdsm = _relative_project(tmp_path)
    rdsm_path = ProjectManifest.load(project_dir).artifact_path("rdsm")
    assert rdsm_path is not None

    raw = _hillshade_rgb(rdsm_path, max_side=512, relative_surface=False)
    normalized = _hillshade_rgb(rdsm_path, max_side=512, relative_surface=True)
    raw_dynamic_range = int(np.ptp(raw[..., 0].astype(np.int16)))
    normalized_dynamic_range = int(np.ptp(normalized[..., 0].astype(np.int16)))

    assert normalized_dynamic_range > raw_dynamic_range
    assert normalized_dynamic_range >= 20
