import numpy as np
from scipy.ndimage import distance_transform_edt

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


def test_metric_structure_support_rejects_high_resolution_edge_contamination() -> None:
    gsd_m = 0.05
    dsm = np.full((500, 500), 100.0, dtype=np.float32)
    mask = np.zeros_like(dsm, dtype=bool)
    mask[170:330, 170:330] = True
    dsm[mask] = 110.0

    distance_from_structure_m = distance_transform_edt(~mask, sampling=(gsd_m, gsd_m))
    contaminated_edge = (~mask) & (distance_from_structure_m <= 1.0)
    dsm[contaminated_edge] = 108.0

    legacy = estimate_structure_height(dsm, mask, ring_pixels=8)
    metric = estimate_structure_height(
        dsm,
        mask,
        ring_pixels=None,
        pixel_size_x_m=gsd_m,
        pixel_size_y_m=gsd_m,
        roof_inset_m=0.5,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=6.0,
        min_structure_pixels=16,
        min_ground_pixels=16,
    )

    assert legacy.structure_height_m < 3.0
    assert metric.structure_height_m == 10.0
    assert metric.ground_elevation_m == 100.0
    assert metric.ground_sector_coverage == 1.0


def test_metric_structure_support_is_physical_scale_invariant() -> None:
    results: list[float] = []
    for gsd_m in (1.0, 0.10):
        side_m = 30.0
        size = int(round(side_m / gsd_m))
        dsm = np.full((size, size), 250.0, dtype=np.float32)
        mask = np.zeros_like(dsm, dtype=bool)
        start = int(round(11.0 / gsd_m))
        stop = int(round(19.0 / gsd_m))
        mask[start:stop, start:stop] = True
        dsm[mask] = 262.0
        estimate = estimate_structure_height(
            dsm,
            mask,
            ring_pixels=None,
            pixel_size_x_m=gsd_m,
            pixel_size_y_m=gsd_m,
            roof_inset_m=0.5,
            ground_inner_buffer_m=1.5,
            ground_outer_buffer_m=6.0,
            min_structure_pixels=4,
            min_ground_pixels=8,
        )
        results.append(estimate.structure_height_m)

    assert np.allclose(results, [12.0, 12.0], atol=1e-6)


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
