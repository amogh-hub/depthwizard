import numpy as np
from rasterio.transform import Affine, from_origin

from depthwizard.calibration.gcp import calibrate_relative_height_with_gcps
from depthwizard.contracts import GroundControlPoint


def _gcps_for_field(
    relative: np.ndarray,
    transform: Affine,
    *,
    scale: float,
    offset: float,
) -> list[GroundControlPoint]:
    points: list[GroundControlPoint] = []
    for row, col in ((2, 2), (5, 12), (15, 4), (17, 17)):
        px, py = transform * (col + 0.5, row + 0.5)
        points.append(
            GroundControlPoint(
                x=px,
                y=py,
                elevation_m=scale * float(relative[row, col]) + offset,
            )
        )
    return points


def test_gcp_calibration_recovers_scale_and_offset() -> None:
    y, x = np.mgrid[:20, :20]
    relative = (x + 0.5 * y).astype(np.float32)
    transform = from_origin(1000, 2000, 2.0, 2.0)
    gcps = _gcps_for_field(relative, transform, scale=4.0, offset=50.0)

    result = calibrate_relative_height_with_gcps(relative, transform=transform, gcps=gcps)

    assert result.orientation_flipped is False
    assert result.anchor_correlation_before > 0.99
    assert abs(result.calibration.scale - 4.0) < 1e-4
    assert abs(result.calibration.offset - 50.0) < 1e-3
    assert np.nanmean(np.abs(result.dsm - (4.0 * relative + 50.0))) < 1e-3


def test_gcp_calibration_resolves_inverted_relative_height_orientation() -> None:
    y, x = np.mgrid[:20, :20]
    physical_relative = (x + 0.5 * y).astype(np.float32)
    inverted_relative = -physical_relative
    transform = from_origin(1000, 2000, 2.0, 2.0)
    gcps = _gcps_for_field(physical_relative, transform, scale=3.0, offset=80.0)

    result = calibrate_relative_height_with_gcps(
        inverted_relative,
        transform=transform,
        gcps=gcps,
    )

    assert result.orientation_flipped is True
    assert result.anchor_correlation_before < -0.99
    assert result.calibration.scale > 0.0
    assert np.nanmean(np.abs(result.dsm - (3.0 * physical_relative + 80.0))) < 1e-3
