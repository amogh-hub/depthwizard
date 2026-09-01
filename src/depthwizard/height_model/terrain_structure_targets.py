from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import cast

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    find_objects,
    generate_binary_structure,
    label,
)

from depthwizard.analysis.structures import (
    _ground_sector_coverage,
    _robust_ground_plane,
    estimate_structure_height,
)
from depthwizard.height_model.training import PriorReferenceFit


@dataclass(frozen=True)
class TerrainStructureTargetConfig:
    min_building_area_m2: float = 20.0
    min_reference_height_m: float = 2.0
    ground_inner_buffer_m: float = 1.50
    ground_outer_buffer_m: float = 8.00
    min_ground_pixels: int = 64
    min_ground_inlier_fraction: float = 0.60
    min_ground_sector_coverage: float = 0.75
    max_negative_agl_fraction: float = 0.05
    negative_agl_tolerance_m: float = 0.50
    boundary_width_m: float = 0.50

    def __post_init__(self) -> None:
        if self.min_building_area_m2 <= 0:
            raise ValueError("min_building_area_m2 must be positive")
        if self.min_reference_height_m < 0:
            raise ValueError("min_reference_height_m cannot be negative")
        if self.ground_inner_buffer_m < 0:
            raise ValueError("ground_inner_buffer_m cannot be negative")
        if self.ground_outer_buffer_m <= self.ground_inner_buffer_m:
            raise ValueError("ground buffers must satisfy inner < outer")
        if self.min_ground_pixels < 3:
            raise ValueError("min_ground_pixels must be at least 3")
        if not 0.0 <= self.min_ground_inlier_fraction <= 1.0:
            raise ValueError("min_ground_inlier_fraction must be in [0, 1]")
        if not 0.0 <= self.min_ground_sector_coverage <= 1.0:
            raise ValueError("min_ground_sector_coverage must be in [0, 1]")
        if not 0.0 <= self.max_negative_agl_fraction <= 1.0:
            raise ValueError("max_negative_agl_fraction must be in [0, 1]")
        if self.negative_agl_tolerance_m < 0:
            raise ValueError("negative_agl_tolerance_m cannot be negative")
        if self.boundary_width_m <= 0:
            raise ValueError("boundary_width_m must be positive")


@dataclass(frozen=True)
class TerrainStructureTargetInstance:
    instance_id: int
    accepted: bool
    reason: str
    area_m2: float
    reference_height_m: float | None
    ground_candidate_pixels: int
    ground_inlier_pixels: int
    ground_inlier_fraction: float | None
    ground_sector_coverage: float | None
    negative_agl_fraction: float | None


@dataclass(frozen=True)
class MetricTerrainStructureTargets:
    dsm_m: np.ndarray
    terrain_m: np.ndarray
    above_ground_m: np.ndarray
    valid_mask: np.ndarray
    building_mask: np.ndarray
    ground_mask: np.ndarray
    boundary_mask: np.ndarray
    instances: tuple[TerrainStructureTargetInstance, ...]

    @property
    def accepted_instance_ids(self) -> tuple[int, ...]:
        return tuple(item.instance_id for item in self.instances if item.accepted)

    @property
    def rejected_instance_ids(self) -> tuple[int, ...]:
        return tuple(item.instance_id for item in self.instances if not item.accepted)


@dataclass(frozen=True)
class RelativeTerrainStructureTargets:
    relative_dsm: np.ndarray
    terrain_relative: np.ndarray
    above_ground_relative: np.ndarray
    valid_mask: np.ndarray
    building_mask: np.ndarray
    ground_mask: np.ndarray
    boundary_mask: np.ndarray


def _validate_inputs(
    reference_dsm_m: np.ndarray,
    building_mask: np.ndarray,
    ground_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference = np.asarray(reference_dsm_m, dtype=np.float64)
    buildings = np.asarray(building_mask, dtype=bool)
    ground = np.asarray(ground_mask, dtype=bool)
    if reference.ndim != 2 or buildings.ndim != 2 or ground.ndim != 2:
        raise ValueError("reference_dsm_m, building_mask, and ground_mask must be 2D")
    if reference.shape != buildings.shape or reference.shape != ground.shape:
        raise ValueError("reference_dsm_m, building_mask, and ground_mask must share shape")
    if np.any(buildings & ground):
        raise ValueError("building_mask and ground_mask must be disjoint")
    return reference, buildings, ground


def _crop_for_instance(
    instance_slice: tuple[slice, slice],
    *,
    shape: tuple[int, int],
    gsd_x_m: float,
    gsd_y_m: float,
    outer_buffer_m: float,
) -> tuple[slice, slice]:
    rows, cols = instance_slice
    row_margin = ceil(outer_buffer_m / gsd_y_m) + 2
    col_margin = ceil(outer_buffer_m / gsd_x_m) + 2
    row_start = max(0, int(rows.start or 0) - row_margin)
    row_stop = min(shape[0], int(rows.stop or shape[0]) + row_margin)
    col_start = max(0, int(cols.start or 0) - col_margin)
    col_stop = min(shape[1], int(cols.stop or shape[1]) + col_margin)
    return slice(row_start, row_stop), slice(col_start, col_stop)


def _rejection(
    *,
    instance_id: int,
    reason: str,
    area_m2: float,
    reference_height_m: float | None = None,
    ground_candidate_pixels: int = 0,
    ground_inlier_pixels: int = 0,
    ground_inlier_fraction: float | None = None,
    ground_sector_coverage: float | None = None,
    negative_agl_fraction: float | None = None,
) -> TerrainStructureTargetInstance:
    return TerrainStructureTargetInstance(
        instance_id=instance_id,
        accepted=False,
        reason=reason,
        area_m2=area_m2,
        reference_height_m=reference_height_m,
        ground_candidate_pixels=ground_candidate_pixels,
        ground_inlier_pixels=ground_inlier_pixels,
        ground_inlier_fraction=ground_inlier_fraction,
        ground_sector_coverage=ground_sector_coverage,
        negative_agl_fraction=negative_agl_fraction,
    )


def prepare_metric_terrain_structure_targets(
    reference_dsm_m: np.ndarray,
    building_mask: np.ndarray,
    ground_mask: np.ndarray,
    *,
    gsd_x_m: float,
    gsd_y_m: float,
    config: TerrainStructureTargetConfig | None = None,
) -> MetricTerrainStructureTargets:
    """Build explicit bare-earth + AGL supervision from training-only semantic/reference evidence.

    Strict ground pixels directly supervise terrain. Terrain beneath each eligible building is obtained
    from a robust local plane fitted only to surrounding semantic-ground candidates. Unsupported or
    geometrically inconsistent buildings are recorded and excluded; this function never interpolates an
    unverified terrain target silently.
    """

    cfg = config or TerrainStructureTargetConfig()
    reference, buildings, ground = _validate_inputs(reference_dsm_m, building_mask, ground_mask)
    if not np.isfinite(gsd_x_m) or gsd_x_m <= 0:
        raise ValueError("gsd_x_m must be positive and finite")
    if not np.isfinite(gsd_y_m) or gsd_y_m <= 0:
        raise ValueError("gsd_y_m must be positive and finite")

    finite = np.isfinite(reference)
    strict_ground = ground & finite
    terrain = np.full(reference.shape, np.nan, dtype=np.float64)
    above_ground = np.full(reference.shape, np.nan, dtype=np.float64)
    valid = np.zeros(reference.shape, dtype=bool)
    accepted_buildings = np.zeros(reference.shape, dtype=bool)

    terrain[strict_ground] = reference[strict_ground]
    above_ground[strict_ground] = 0.0
    valid[strict_ground] = True

    connected, count = cast(
        tuple[np.ndarray, int],
        label(buildings, structure=generate_binary_structure(2, 1)),
    )
    slices = find_objects(connected)
    pixel_area_m2 = float(gsd_x_m * gsd_y_m)
    instances: list[TerrainStructureTargetInstance] = []

    for zero_based_id in range(count):
        instance_id = zero_based_id + 1
        object_slice = slices[zero_based_id]
        if object_slice is None:
            continue
        native_instance = connected[object_slice] == instance_id
        area_m2 = float(np.count_nonzero(native_instance) * pixel_area_m2)
        if area_m2 < cfg.min_building_area_m2:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="building area below minimum target-support threshold",
                    area_m2=area_m2,
                )
            )
            continue

        crop = _crop_for_instance(
            object_slice,
            shape=(int(reference.shape[0]), int(reference.shape[1])),
            gsd_x_m=gsd_x_m,
            gsd_y_m=gsd_y_m,
            outer_buffer_m=cfg.ground_outer_buffer_m,
        )
        local_labels = connected[crop]
        local_structure = local_labels == instance_id
        local_reference = reference[crop]
        local_ground = strict_ground[crop] & (local_labels == 0)

        # Reuse the exact scientific measurement contract for reference eligibility before building
        # a dense target beneath the structure.
        try:
            estimate = estimate_structure_height(
                local_reference,
                local_structure,
                ring_pixels=None,
                min_structure_pixels=16,
                min_ground_pixels=cfg.min_ground_pixels,
                pixel_size_x_m=gsd_x_m,
                pixel_size_y_m=gsd_y_m,
                roof_inset_m=0.50,
                ground_inner_buffer_m=cfg.ground_inner_buffer_m,
                ground_outer_buffer_m=cfg.ground_outer_buffer_m,
                ground_candidate_mask=local_ground,
            )
        except ValueError as exc:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason=f"reference structure support rejected: {exc}",
                    area_m2=area_m2,
                )
            )
            continue

        if estimate.structure_height_m < cfg.min_reference_height_m:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="reference height below minimum structure threshold",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=estimate.ground_pixels,
                    ground_inlier_pixels=estimate.ground_inlier_pixels,
                    ground_inlier_fraction=estimate.ground_inlier_fraction,
                    ground_sector_coverage=estimate.ground_sector_coverage,
                )
            )
            continue
        if estimate.ground_inlier_fraction < cfg.min_ground_inlier_fraction:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="reference ground inlier fraction below target threshold",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=estimate.ground_pixels,
                    ground_inlier_pixels=estimate.ground_inlier_pixels,
                    ground_inlier_fraction=estimate.ground_inlier_fraction,
                    ground_sector_coverage=estimate.ground_sector_coverage,
                )
            )
            continue
        if estimate.ground_sector_coverage < cfg.min_ground_sector_coverage:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="reference ground sector coverage below target threshold",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=estimate.ground_pixels,
                    ground_inlier_pixels=estimate.ground_inlier_pixels,
                    ground_inlier_fraction=estimate.ground_inlier_fraction,
                    ground_sector_coverage=estimate.ground_sector_coverage,
                )
            )
            continue

        distance_from_structure_m = cast(
            np.ndarray,
            distance_transform_edt(
                ~local_structure,
                sampling=(gsd_y_m, gsd_x_m),
            ),
        )
        ring = (
            local_ground
            & (distance_from_structure_m >= cfg.ground_inner_buffer_m)
            & (distance_from_structure_m <= cfg.ground_outer_buffer_m)
        )
        candidate_pixels = int(np.count_nonzero(ring))
        if candidate_pixels < cfg.min_ground_pixels:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="insufficient strict-ground candidates for dense target plane",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=candidate_pixels,
                )
            )
            continue

        ground_fit = _robust_ground_plane(local_reference, ring)
        inlier_pixels = int(np.count_nonzero(ground_fit.inliers))
        inlier_fraction = float(inlier_pixels / max(candidate_pixels, 1))
        sector_coverage = _ground_sector_coverage(
            local_structure,
            ground_fit.inliers,
            pixel_size_x_m=gsd_x_m,
            pixel_size_y_m=gsd_y_m,
        )
        if inlier_pixels < cfg.min_ground_pixels:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="insufficient robust ground inliers for dense target plane",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=candidate_pixels,
                    ground_inlier_pixels=inlier_pixels,
                    ground_inlier_fraction=inlier_fraction,
                    ground_sector_coverage=sector_coverage,
                )
            )
            continue
        if inlier_fraction < cfg.min_ground_inlier_fraction:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="dense target plane inlier fraction below threshold",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=candidate_pixels,
                    ground_inlier_pixels=inlier_pixels,
                    ground_inlier_fraction=inlier_fraction,
                    ground_sector_coverage=sector_coverage,
                )
            )
            continue
        if sector_coverage < cfg.min_ground_sector_coverage:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="dense target plane lacks surrounding sector coverage",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=candidate_pixels,
                    ground_inlier_pixels=inlier_pixels,
                    ground_inlier_fraction=inlier_fraction,
                    ground_sector_coverage=sector_coverage,
                )
            )
            continue

        local_terrain = ground_fit.surface[local_structure]
        local_roof = local_reference[local_structure]
        raw_agl = local_roof - local_terrain
        strongly_negative = raw_agl < -cfg.negative_agl_tolerance_m
        negative_fraction = float(np.mean(strongly_negative)) if raw_agl.size else 1.0
        if negative_fraction > cfg.max_negative_agl_fraction:
            instances.append(
                _rejection(
                    instance_id=instance_id,
                    reason="too much roof support lies below the fitted local terrain",
                    area_m2=area_m2,
                    reference_height_m=estimate.structure_height_m,
                    ground_candidate_pixels=candidate_pixels,
                    ground_inlier_pixels=inlier_pixels,
                    ground_inlier_fraction=inlier_fraction,
                    ground_sector_coverage=sector_coverage,
                    negative_agl_fraction=negative_fraction,
                )
            )
            continue

        local_valid = local_structure & np.isfinite(local_reference) & np.isfinite(ground_fit.surface)
        global_valid_view = valid[crop]
        global_terrain_view = terrain[crop]
        global_agl_view = above_ground[crop]
        global_building_view = accepted_buildings[crop]
        global_valid_view[local_valid] = True
        global_terrain_view[local_valid] = ground_fit.surface[local_valid]
        global_agl_view[local_valid] = np.maximum(
            local_reference[local_valid] - ground_fit.surface[local_valid],
            0.0,
        )
        global_building_view[local_valid] = True

        instances.append(
            TerrainStructureTargetInstance(
                instance_id=instance_id,
                accepted=True,
                reason="accepted",
                area_m2=area_m2,
                reference_height_m=estimate.structure_height_m,
                ground_candidate_pixels=candidate_pixels,
                ground_inlier_pixels=inlier_pixels,
                ground_inlier_fraction=inlier_fraction,
                ground_sector_coverage=sector_coverage,
                negative_agl_fraction=negative_fraction,
            )
        )

    if not np.any(accepted_buildings):
        raise ValueError("no building instance has sufficient evidence for terrain-structure targets")

    # Physical boundary width avoids a one-pixel definition that changes meaning across GSDs.
    distance_inside_m = cast(
        np.ndarray,
        distance_transform_edt(
            accepted_buildings,
            sampling=(gsd_y_m, gsd_x_m),
        ),
    )
    boundary = accepted_buildings & (distance_inside_m <= cfg.boundary_width_m)
    target_dsm = terrain + above_ground
    target_dsm[~valid] = np.nan
    terrain[~valid] = np.nan
    above_ground[~valid] = np.nan

    return MetricTerrainStructureTargets(
        dsm_m=target_dsm.astype(np.float32),
        terrain_m=terrain.astype(np.float32),
        above_ground_m=above_ground.astype(np.float32),
        valid_mask=valid,
        building_mask=accepted_buildings,
        ground_mask=strict_ground,
        boundary_mask=boundary,
        instances=tuple(instances),
    )


def canonicalize_metric_terrain_structure_targets(
    targets: MetricTerrainStructureTargets,
    fit: PriorReferenceFit,
) -> RelativeTerrainStructureTargets:
    """Express metric T + H_AGL targets in one training scene's DA3-relative coordinates."""

    if not np.isfinite(fit.scale_m_per_prior_unit) or fit.scale_m_per_prior_unit <= 1e-8:
        raise ValueError("fit must contain a finite positive scale")
    if not np.isfinite(fit.offset_m):
        raise ValueError("fit offset must be finite")
    valid = np.asarray(targets.valid_mask, dtype=bool)
    terrain = np.asarray(targets.terrain_m, dtype=np.float32)
    agl = np.asarray(targets.above_ground_m, dtype=np.float32)
    if terrain.shape != valid.shape or agl.shape != valid.shape:
        raise ValueError("metric target arrays and valid mask must share shape")
    if not np.all(np.isfinite(terrain[valid])) or not np.all(np.isfinite(agl[valid])):
        raise ValueError("metric targets must be finite on valid pixels")

    scale = np.float32(fit.scale_m_per_prior_unit)
    offset = np.float32(fit.offset_m)
    terrain_relative = (terrain - offset) / scale
    above_ground_relative = agl / scale
    relative_dsm = terrain_relative + above_ground_relative
    terrain_relative[~valid] = np.nan
    above_ground_relative[~valid] = np.nan
    relative_dsm[~valid] = np.nan

    return RelativeTerrainStructureTargets(
        relative_dsm=relative_dsm.astype(np.float32, copy=False),
        terrain_relative=terrain_relative.astype(np.float32, copy=False),
        above_ground_relative=above_ground_relative.astype(np.float32, copy=False),
        valid_mask=valid.copy(),
        building_mask=np.asarray(targets.building_mask, dtype=bool).copy(),
        ground_mask=np.asarray(targets.ground_mask, dtype=bool).copy(),
        boundary_mask=np.asarray(targets.boundary_mask, dtype=bool).copy(),
    )
