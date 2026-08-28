import numpy as np

from depthwizard.evaluation.seams import evaluate_tiling_seams


def test_smooth_surface_has_no_tile_specific_discontinuity_amplification() -> None:
    y, x = np.mgrid[:96, :112]
    surface = (0.2 * x + 0.35 * y).astype(np.float32)
    metrics = evaluate_tiling_seams(surface, tile_size=40, overlap=8)

    assert metrics.vertical_seams >= 1
    assert metrics.horizontal_seams >= 1
    assert metrics.seam_samples > 0
    assert metrics.interior_samples > metrics.seam_samples
    assert metrics.p95_seam_to_interior_ratio is not None
    assert metrics.p95_seam_to_interior_ratio < 1.05


def test_injected_boundary_offset_is_detected_as_large_seam_artifact() -> None:
    y, x = np.mgrid[:96, :112]
    surface = (0.2 * x + 0.35 * y).astype(np.float32)
    # stride = 32. Inject a persistent mismatch beginning exactly at the first vertical seam.
    surface[:, 32:] += 12.0
    metrics = evaluate_tiling_seams(surface, tile_size=40, overlap=8)

    assert metrics.p95_seam_to_interior_ratio is not None
    assert metrics.p95_seam_to_interior_ratio > 10.0
    assert metrics.seam_p95_abs_jump > metrics.interior_p95_abs_gradient


def test_seam_evaluator_rejects_invalid_contracts_and_missing_boundaries() -> None:
    surface = np.ones((8, 8), dtype=np.float32)
    for tile_size, overlap in ((1, 0), (8, 8), (8, -1)):
        try:
            evaluate_tiling_seams(surface, tile_size=tile_size, overlap=overlap)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid tiling contract should be rejected")

    try:
        evaluate_tiling_seams(surface, tile_size=16, overlap=4)
    except ValueError as exc:
        assert "too small" in str(exc)
    else:
        raise AssertionError("a seam metric without any seam must not be fabricated")
