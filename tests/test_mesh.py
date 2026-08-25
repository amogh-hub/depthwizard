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

    output = tmp_path / "terrain.glb"
    result = export_terrain_glb(output, elevation, rgb, gsd_x=2.0, gsd_y=3.0)
    assert output.exists()
    assert output.stat().st_size > 100
    assert result.faces == 40
