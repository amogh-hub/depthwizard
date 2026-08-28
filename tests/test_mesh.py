from pathlib import Path

import numpy as np

from depthwizard.mesh.terrain import export_terrain_glb, terrain_mesh_from_dsm


def test_metric_mesh_coordinates_and_export(tmp_path: Path) -> None:
    y, x = np.mgrid[:5, :6]
    elevation = (100 + x + y).astype(np.float32)
    rgb = np.zeros((5, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 120
    rgb[..., 1] = 160
    rgb[..., 2] = 90

    mesh = terrain_mesh_from_dsm(elevation, rgb, gsd_x=2.0, gsd_y=3.0)
    assert mesh.vertices.shape[0] == 30
    assert np.isclose(mesh.vertices[:, 0].max(), 10.0)
    assert np.isclose(mesh.vertices[:, 2].min(), -12.0)
    # Terrain front faces must point upward. RGB rendering may be double-sided for robustness, but
    # analytical overlays must never depend on that material setting to hide reversed geometry.
    assert np.all(mesh.face_normals[:, 1] > 0.0)

    output = tmp_path / "terrain.glb"
    result = export_terrain_glb(output, elevation, rgb, gsd_x=2.0, gsd_y=3.0)
    assert output.exists()
    assert output.stat().st_size > 100
    assert result.faces == 40


def test_flat_terrain_faces_are_upward() -> None:
    elevation = np.full((3, 3), 100.0, dtype=np.float32)
    rgb = np.full((3, 3, 3), 128, dtype=np.uint8)
    mesh = terrain_mesh_from_dsm(elevation, rgb)
    np.testing.assert_allclose(mesh.face_normals[:, 0], 0.0, atol=1e-7)
    np.testing.assert_allclose(mesh.face_normals[:, 1], 1.0, atol=1e-7)
    np.testing.assert_allclose(mesh.face_normals[:, 2], 0.0, atol=1e-7)


def test_invalid_source_pixels_remove_terrain_faces() -> None:
    elevation = np.arange(16, dtype=np.float32).reshape(4, 4)
    rgb = np.full((4, 4, 3), 140, dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    valid[:2, :2] = False

    full = terrain_mesh_from_dsm(elevation, rgb)
    masked = terrain_mesh_from_dsm(elevation, rgb, valid_mask=valid)

    assert len(full.faces) == 18
    assert 0 < len(masked.faces) < len(full.faces)
    assert np.all(masked.face_normals[:, 1] > 0.0)
