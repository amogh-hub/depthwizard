from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

from depthwizard.service import app


def test_inspect_endpoint_reports_georeferenced_raster(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    data = np.zeros((3, 16, 16), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=16,
        width=16,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)

    client = TestClient(app)
    response = client.post("/v1/inspect", json={"path": str(path)})
    assert response.status_code == 200
    payload = response.json()
    assert payload["crs"] == "EPSG:32643"
    # Reported GSD is physical ground distance, so projected CRS scale factor can differ slightly
    # from the affine coordinate-unit spacing.
    assert abs(payload["ground_sample_distance_x"] - 1.0) < 0.01
