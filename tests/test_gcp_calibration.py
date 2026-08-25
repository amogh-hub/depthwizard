import numpy as np
from rasterio.transform import from_origin

from depthwizard.calibration.gcp import calibrate_relative_height_with_gcps
from depthwizard.contracts import GroundControlPoint


def test_gcp_calibration_recovers_scale_and_offset() -> None:
    y, x = np.mgrid[:20, :20]
    relative = (x + 0.5 * y).astype(np.float32)
    transform = from_origin(1000, 2000, 2.0, 2.0)

    def gcp_for(row: int, col: int) -> GroundControlPoint:
        px, py = transform * (col + 0.5, row + 0.5)
        rel = float(relative[row, col])
        return GroundControlPoint(x=px, y=py, elevation_m=4.0 * rel + 50.0)

    gcps = [gcp_for(2, 2), gcp_for(5, 12), gcp_for(15, 4), gcp_for(17, 17)]
    result = calibrate_relative_height_with_gcps(relative, transform=transform, gcps=gcps)
    assert abs(result.calibration.scale - 4.0) < 1e-4
    assert abs(result.calibration.offset - 50.0) < 1e-3
    assert np.nanmean(np.abs(result.dsm - (4.0 * relative + 50.0))) < 1e-3
