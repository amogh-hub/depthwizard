from __future__ import annotations

import numpy as np
import pytest

from depthwizard.height_model.terrain_structure_targets import (
    TerrainStructureTargetConfig,
    canonicalize_metric_terrain_structure_targets,
    prepare_metric_terrain_structure_targets,
)
from depthwizard.height_model.training import PriorReferenceFit


def test_metric_target_builder_recovers_flat_ground_beneath_building() -> None:
    shape = (96, 96)
    reference = np.full(shape, 100.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    buildings[36:60, 36:60] = True
    reference[buildings] = 110.0
    ground = ~buildings

    config = TerrainStructureTargetConfig(
        min_building_area_m2=10.0,
        min_reference_height_m=2.0,
        ground_inner_buffer_m=1.5,
        ground_outer_buffer_m=8.0,
        min_ground_pixels=64,
        boundary_width_m=0.5,
    )
    targets = prepare_metric_terrain_structure_targets(
        reference,
        buildings,
        ground,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        config=config,
    )

    assert targets.accepted_instance_ids == (1,)
    assert targets.rejected_instance_ids == ()
    assert np.allclose(targets.terrain_m[targets.building_mask], 100.0, atol=1e-6)
    assert np.allclose(targets.above_ground_m[targets.building_mask], 10.0, atol=1e-6)
    assert np.allclose(targets.dsm_m[targets.building_mask], 110.0, atol=1e-6)
    assert np.allclose(targets.above_ground_m[targets.ground_mask], 0.0, atol=1e-6)
    assert np.array_equal(targets.valid_mask, targets.building_mask | targets.ground_mask)
    assert np.all(targets.boundary_mask <= targets.building_mask)
    assert np.count_nonzero(targets.boundary_mask) > 0


def test_target_builder_rejects_building_without_strict_ground_support() -> None:
    shape = (64, 64)
    reference = np.full(shape, 50.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    buildings[20:44, 20:44] = True
    reference[buildings] = 60.0
    ground = np.zeros(shape, dtype=bool)

    with pytest.raises(ValueError, match="no building instance"):
        prepare_metric_terrain_structure_targets(
            reference,
            buildings,
            ground,
            gsd_x_m=0.5,
            gsd_y_m=0.5,
            config=TerrainStructureTargetConfig(
                min_building_area_m2=10.0,
                min_ground_pixels=16,
            ),
        )


def test_target_builder_robustly_ignores_high_edge_collar() -> None:
    shape = (128, 128)
    reference = np.full(shape, 100.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    buildings[48:80, 48:80] = True
    reference[buildings] = 112.0
    ground = ~buildings

    # Contaminate the first metre around the building. The physical 1.5 m inner exclusion buffer
    # must keep this edge artefact out of the target terrain fit.
    reference[46:82, 46:82] = np.maximum(reference[46:82, 46:82], 108.0)
    reference[buildings] = 112.0

    targets = prepare_metric_terrain_structure_targets(
        reference,
        buildings,
        ground,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        config=TerrainStructureTargetConfig(
            min_building_area_m2=10.0,
            min_reference_height_m=2.0,
            ground_inner_buffer_m=1.5,
            ground_outer_buffer_m=8.0,
            min_ground_pixels=64,
        ),
    )

    assert targets.accepted_instance_ids == (1,)
    assert np.median(targets.terrain_m[targets.building_mask]) == pytest.approx(100.0, abs=1e-6)
    assert np.median(targets.above_ground_m[targets.building_mask]) == pytest.approx(12.0, abs=1e-6)


def test_metric_targets_canonicalize_terrain_and_agl_differently() -> None:
    shape = (96, 96)
    reference = np.full(shape, 100.0, dtype=np.float64)
    buildings = np.zeros(shape, dtype=bool)
    buildings[36:60, 36:60] = True
    reference[buildings] = 110.0
    ground = ~buildings
    metric = prepare_metric_terrain_structure_targets(
        reference,
        buildings,
        ground,
        gsd_x_m=0.5,
        gsd_y_m=0.5,
        config=TerrainStructureTargetConfig(
            min_building_area_m2=10.0,
            min_ground_pixels=64,
        ),
    )
    fit = PriorReferenceFit(
        scale_m_per_prior_unit=20.0,
        offset_m=80.0,
        rmse_m=1.0,
        median_abs_residual_m=0.5,
        samples=1000,
    )
    relative = canonicalize_metric_terrain_structure_targets(metric, fit)

    assert np.median(relative.terrain_relative[relative.building_mask]) == pytest.approx(1.0)
    assert np.median(relative.above_ground_relative[relative.building_mask]) == pytest.approx(0.5)
    assert np.median(relative.relative_dsm[relative.building_mask]) == pytest.approx(1.5)
    assert np.allclose(
        relative.relative_dsm[relative.valid_mask],
        relative.terrain_relative[relative.valid_mask]
        + relative.above_ground_relative[relative.valid_mask],
        atol=1e-6,
    )
