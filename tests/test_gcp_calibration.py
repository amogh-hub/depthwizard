import numpy as np
import pytest
from rasterio.transform import Affine, from_origin

from depthwizard.calibration.gcp import (
    calibrate_relative_height_with_gcps,
    validate_metric_dsm_with_gcps,
)
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


def test_gcp_calibration_requires_independent_quality_evidence() -> None:
    y, x = np.mgrid[:20, :20]
    relative = (x + 0.5 * y).astype(np.float32)
    transform = from_origin(1000, 2000, 2.0, 2.0)
    gcps = _gcps_for_field(relative, transform, scale=4.0, offset=50.0)

    with pytest.raises(ValueError, match="at least 4 GCPs"):
        calibrate_relative_height_with_gcps(relative, transform=transform, gcps=gcps[:3])


def test_gcp_calibration_rejects_clustered_control() -> None:
    y, x = np.mgrid[:20, :20]
    relative = (x + 0.5 * y).astype(np.float32)
    transform = from_origin(1000, 2000, 2.0, 2.0)
    gcps: list[GroundControlPoint] = []
    for row, col in ((1, 1), (1, 2), (2, 1), (2, 2)):
        px, py = transform * (col + 0.5, row + 0.5)
        gcps.append(
            GroundControlPoint(
                x=px,
                y=py,
                elevation_m=4.0 * float(relative[row, col]) + 50.0,
            )
        )

    with pytest.raises(ValueError, match="spatially clustered"):
        calibrate_relative_height_with_gcps(relative, transform=transform, gcps=gcps)


def test_dem_plus_gcp_applies_only_global_vertical_datum_offset() -> None:
    y, x = np.mgrid[:20, :20]
    metric_dsm = (100.0 + 0.8 * x + 0.3 * y).astype(np.float32)
    transform = from_origin(1000, 2000, 2.0, 2.0)
    gcps: list[GroundControlPoint] = []
    for row, col in ((2, 2), (5, 12), (15, 4), (17, 17)):
        px, py = transform * (col + 0.5, row + 0.5)
        gcps.append(
            GroundControlPoint(
                x=px,
                y=py,
                elevation_m=float(metric_dsm[row, col]) + 7.0,
            )
        )

    result = validate_metric_dsm_with_gcps(metric_dsm, transform=transform, gcps=gcps)

    assert abs(result.offset_applied_m - 7.0) < 1e-6
    assert np.allclose(result.dsm, metric_dsm + 7.0, atol=1e-6)
    original_relief = float(metric_dsm[17, 17] - metric_dsm[2, 2])
    corrected_relief = float(result.dsm[17, 17] - result.dsm[2, 2])
    assert abs(corrected_relief - original_relief) < 1e-6
