import numpy as np

from depthwizard.analysis.profile import sample_elevation_profile
from depthwizard.analysis.structures import estimate_structure_height


def test_structure_height_uses_local_ground_ring() -> None:
    dsm = np.full((40, 40), 100.0, dtype=np.float32)
    mask = np.zeros_like(dsm, dtype=bool)
    mask[15:25, 15:25] = True
    dsm[mask] = 112.5
    estimate = estimate_structure_height(dsm, mask, ring_pixels=5)
    assert np.isclose(estimate.ground_elevation_m, 100.0)
    assert np.isclose(estimate.structure_height_m, 12.5)


def test_elevation_profile_respects_metric_gsd() -> None:
    y, x = np.mgrid[:20, :20]
    elevation = (x + y).astype(np.float32)
    profile = sample_elevation_profile(
        elevation,
        start_pixel=(0, 0),
        end_pixel=(10, 10),
        gsd_x=2.0,
        gsd_y=2.0,
        samples=11,
    )
    assert np.isclose(profile.distance_m[-1], np.sqrt(800.0))
    assert np.isclose(profile.elevation[-1], 20.0)
