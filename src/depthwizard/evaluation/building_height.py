from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
from scipy.ndimage import find_objects, generate_binary_structure, label

from depthwizard.analysis.structures import StructureHeightEstimate, estimate_structure_height


@dataclass(frozen=True)
class BuildingHeightInstance:
    instance_id: int
    area_m2: float
    reference_height_m: float
    predicted_height_m: float
    height_error_m: float
    reference_top_m: float
    predicted_top_m: float
    top_error_m: float
    reference_ground_m: float
    predicted_ground_m: float
    ground_error_m: float
    reference_roof_dispersion_m: float
    predicted_roof_dispersion_m: float
    reference_local_height_dispersion_m: float
    predicted_local_height_dispersion_m: float


@dataclass(frozen=True)
class BuildingHeightBenchmarkReport:
    eligible_instance_ids: tuple[int, ...]
    evaluated_instance_ids: tuple[int, ...]
    prediction_failure_ids: tuple[int, ...]
    skipped_reference_instances: int
    instances: tuple[BuildingHeightInstance, ...]
    height_mae_m: float
    height_rmse_m: float
    height_bias_m: float
    height_median_abs_error_m: float
    height_p90_abs_error_m: float
    height_p95_abs_error_m: float
    top_mae_m: float
    ground_mae_m: float
    within_1m_fraction: float
    within_2m_fraction: float
    catastrophic_over_3m_fraction: float
    mean_reference_height_m: float
    mean_predicted_height_m: float


@dataclass(frozen=True)
class BuildingHeightPromotionThresholds:
    min_instances: int = 20
    min_mae_reduction_fraction: float = 0.15
    min_rmse_reduction_fraction: float = 0.10
    min_p90_reduction_fraction: float = 0.05
    max_within_2m_fraction_drop: float = 0.0
    max_catastrophic_fraction_increase: float = 0.0


@dataclass(frozen=True)
class BuildingHeightPromotionDecision:
    passed: bool
    reasons: tuple[str, ...]
    mae_reduction_fraction: float
    rmse_reduction_fraction: float
    p90_reduction_fraction: float


def _validate_inputs(
    prediction: np.ndarray,
    reference: np.ndarray,
    building_mask: np.ndarray,
    ground_candidate_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    pred = np.asarray(prediction, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    buildings = np.asarray(building_mask, dtype=bool)
    if pred.shape != ref.shape or pred.shape != buildings.shape:
        raise ValueError("prediction, reference, and building_mask must have identical shape")
    candidates: np.ndarray | None = None
    if ground_candidate_mask is not None:
        candidates = np.asarray(ground_candidate_mask, dtype=bool)
        if candidates.shape != pred.shape:
            raise ValueError("ground_candidate_mask must match prediction shape")
    return pred, ref, buildings, candidates


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


def _estimate_instance(
    elevation: np.ndarray,
    structure_mask: np.ndarray,
    *,
    gsd_x_m: float,
    gsd_y_m: float,
    roof_inset_m: float,
    ground_inner_buffer_m: float,
    ground_outer_buffer_m: float,
    min_structure_pixels: int,
    min_ground_pixels: int,
    ground_candidate_mask: np.ndarray,
) -> StructureHeightEstimate:
    return estimate_structure_height(
        elevation,
        structure_mask,
        ring_pixels=None,
        min_structure_pixels=min_structure_pixels,
        min_ground_pixels=min_ground_pixels,
        pixel_size_x_m=gsd_x_m,
        pixel_size_y_m=gsd_y_m,
        roof_inset_m=roof_inset_m,
        ground_inner_buffer_m=ground_inner_buffer_m,
        ground_outer_buffer_m=ground_outer_buffer_m,
        ground_candidate_mask=ground_candidate_mask,
    )


def evaluate_building_height_instances(
    prediction: np.ndarray,
    reference: np.ndarray,
    building_mask: np.ndarray,
    *,
    gsd_x_m: float,
    gsd_y_m: float,
    ground_candidate_mask: np.ndarray | None = None,
    min_building_area_m2: float = 20.0,
    min_reference_height_m: float = 2.0,
    roof_inset_m: float = 0.50,
    ground_inner_buffer_m: float = 1.50,
    ground_outer_buffer_m: float = 8.00,
    min_structure_pixels: int = 16,
    min_ground_pixels: int = 64,
    min_reference_ground_inlier_fraction: float = 0.60,
    min_reference_ground_sector_coverage: float = 0.75,
) -> BuildingHeightBenchmarkReport:
    """Evaluate explicit building instances using reference-selected physical support.

    Instance eligibility depends only on the building mask and reference DSM, never on the candidate
    prediction. That prevents a weak candidate from improving its score by causing hard buildings to
    disappear from evaluation. If an eligible reference building cannot be measured in a candidate,
    its id is recorded as a prediction failure and any promotion gate must fail closed.
    """
    pred, ref, buildings, explicit_ground = _validate_inputs(
        prediction,
        reference,
        building_mask,
        ground_candidate_mask,
    )
    if not np.isfinite(gsd_x_m) or gsd_x_m <= 0.0:
        raise ValueError("gsd_x_m must be a positive finite value")
    if not np.isfinite(gsd_y_m) or gsd_y_m <= 0.0:
        raise ValueError("gsd_y_m must be a positive finite value")
    if min_building_area_m2 <= 0.0:
        raise ValueError("min_building_area_m2 must be positive")
    if ground_inner_buffer_m < 0.0 or ground_outer_buffer_m <= ground_inner_buffer_m:
        raise ValueError("ground buffers must satisfy 0 <= inner < outer")

    connected, count = label(buildings, structure=generate_binary_structure(2, 1))
    slices = find_objects(connected)
    pixel_area_m2 = float(gsd_x_m * gsd_y_m)

    eligible_ids: list[int] = []
    evaluated_ids: list[int] = []
    failure_ids: list[int] = []
    instances: list[BuildingHeightInstance] = []
    skipped_reference = 0

    for zero_based_id in range(count):
        instance_id = zero_based_id + 1
        object_slice = slices[zero_based_id]
        if object_slice is None:
            continue
        native_instance = connected[object_slice] == instance_id
        area_m2 = float(np.count_nonzero(native_instance) * pixel_area_m2)
        if area_m2 < min_building_area_m2:
            skipped_reference += 1
            continue

        crop = _crop_for_instance(
            object_slice,
            shape=pred.shape,
            gsd_x_m=gsd_x_m,
            gsd_y_m=gsd_y_m,
            outer_buffer_m=ground_outer_buffer_m,
        )
        local_labels = connected[crop]
        local_structure = local_labels == instance_id
        local_all_buildings = local_labels > 0
        local_ground_candidates = ~local_all_buildings
        if explicit_ground is not None:
            local_ground_candidates &= explicit_ground[crop]

        try:
            reference_estimate = _estimate_instance(
                ref[crop],
                local_structure,
                gsd_x_m=gsd_x_m,
                gsd_y_m=gsd_y_m,
                roof_inset_m=roof_inset_m,
                ground_inner_buffer_m=ground_inner_buffer_m,
                ground_outer_buffer_m=ground_outer_buffer_m,
                min_structure_pixels=min_structure_pixels,
                min_ground_pixels=min_ground_pixels,
                ground_candidate_mask=local_ground_candidates,
            )
        except ValueError:
            skipped_reference += 1
            continue

        if reference_estimate.structure_height_m < min_reference_height_m:
            skipped_reference += 1
            continue
        if reference_estimate.ground_inlier_fraction < min_reference_ground_inlier_fraction:
            skipped_reference += 1
            continue
        if reference_estimate.ground_sector_coverage < min_reference_ground_sector_coverage:
            skipped_reference += 1
            continue

        eligible_ids.append(instance_id)
        try:
            prediction_estimate = _estimate_instance(
                pred[crop],
                local_structure,
                gsd_x_m=gsd_x_m,
                gsd_y_m=gsd_y_m,
                roof_inset_m=roof_inset_m,
                ground_inner_buffer_m=ground_inner_buffer_m,
                ground_outer_buffer_m=ground_outer_buffer_m,
                min_structure_pixels=min_structure_pixels,
                min_ground_pixels=min_ground_pixels,
                ground_candidate_mask=local_ground_candidates,
            )
        except ValueError:
            failure_ids.append(instance_id)
            continue

        evaluated_ids.append(instance_id)
        height_error = prediction_estimate.structure_height_m - reference_estimate.structure_height_m
        top_error = prediction_estimate.top_elevation_m - reference_estimate.top_elevation_m
        ground_error = prediction_estimate.ground_elevation_m - reference_estimate.ground_elevation_m
        instances.append(
            BuildingHeightInstance(
                instance_id=instance_id,
                area_m2=area_m2,
                reference_height_m=reference_estimate.structure_height_m,
                predicted_height_m=prediction_estimate.structure_height_m,
                height_error_m=float(height_error),
                reference_top_m=reference_estimate.top_elevation_m,
                predicted_top_m=prediction_estimate.top_elevation_m,
                top_error_m=float(top_error),
                reference_ground_m=reference_estimate.ground_elevation_m,
                predicted_ground_m=prediction_estimate.ground_elevation_m,
                ground_error_m=float(ground_error),
                reference_roof_dispersion_m=reference_estimate.roof_dispersion_m,
                predicted_roof_dispersion_m=prediction_estimate.roof_dispersion_m,
                reference_local_height_dispersion_m=reference_estimate.local_height_dispersion_m,
                predicted_local_height_dispersion_m=prediction_estimate.local_height_dispersion_m,
            )
        )

    if not instances:
        raise ValueError("no measurable building instances remain after reference-only eligibility checks")

    height_error = np.asarray([item.height_error_m for item in instances], dtype=np.float64)
    height_abs_error = np.abs(height_error)
    top_abs_error = np.abs(np.asarray([item.top_error_m for item in instances], dtype=np.float64))
    ground_abs_error = np.abs(
        np.asarray([item.ground_error_m for item in instances], dtype=np.float64)
    )
    reference_heights = np.asarray(
        [item.reference_height_m for item in instances], dtype=np.float64
    )
    predicted_heights = np.asarray(
        [item.predicted_height_m for item in instances], dtype=np.float64
    )

    return BuildingHeightBenchmarkReport(
        eligible_instance_ids=tuple(eligible_ids),
        evaluated_instance_ids=tuple(evaluated_ids),
        prediction_failure_ids=tuple(failure_ids),
        skipped_reference_instances=skipped_reference,
        instances=tuple(instances),
        height_mae_m=float(np.mean(height_abs_error)),
        height_rmse_m=float(np.sqrt(np.mean(height_error**2))),
        height_bias_m=float(np.mean(height_error)),
        height_median_abs_error_m=float(np.median(height_abs_error)),
        height_p90_abs_error_m=float(np.percentile(height_abs_error, 90)),
        height_p95_abs_error_m=float(np.percentile(height_abs_error, 95)),
        top_mae_m=float(np.mean(top_abs_error)),
        ground_mae_m=float(np.mean(ground_abs_error)),
        within_1m_fraction=float(np.mean(height_abs_error <= 1.0)),
        within_2m_fraction=float(np.mean(height_abs_error <= 2.0)),
        catastrophic_over_3m_fraction=float(np.mean(height_abs_error > 3.0)),
        mean_reference_height_m=float(np.mean(reference_heights)),
        mean_predicted_height_m=float(np.mean(predicted_heights)),
    )


def _fractional_reduction(baseline: float, candidate: float) -> float:
    if baseline <= 1e-12:
        return 0.0 if candidate <= baseline + 1e-12 else -float("inf")
    return float((baseline - candidate) / baseline)


def building_height_promotion_gate(
    baseline: BuildingHeightBenchmarkReport,
    candidate: BuildingHeightBenchmarkReport,
    *,
    thresholds: BuildingHeightPromotionThresholds = BuildingHeightPromotionThresholds(),
) -> BuildingHeightPromotionDecision:
    """Require material per-building improvement before an urban model may be promoted.

    This gate is necessary but not sufficient for production promotion. Whole-scene DSM metrics,
    natural-terrain non-degradation, human-visible operator validation, and reserved blind evidence
    remain separate mandatory gates.
    """
    if thresholds.min_instances < 1:
        raise ValueError("min_instances must be positive")
    reasons: list[str] = []

    if baseline.eligible_instance_ids != candidate.eligible_instance_ids:
        reasons.append("candidate and baseline do not share the same reference-selected instances")
    if baseline.prediction_failure_ids:
        reasons.append("baseline contains prediction failures on reference-eligible buildings")
    if candidate.prediction_failure_ids:
        reasons.append("candidate contains prediction failures on reference-eligible buildings")
    if len(candidate.evaluated_instance_ids) < thresholds.min_instances:
        reasons.append(
            f"only {len(candidate.evaluated_instance_ids)} buildings evaluated; "
            f"need at least {thresholds.min_instances}"
        )

    mae_reduction = _fractional_reduction(baseline.height_mae_m, candidate.height_mae_m)
    rmse_reduction = _fractional_reduction(baseline.height_rmse_m, candidate.height_rmse_m)
    p90_reduction = _fractional_reduction(
        baseline.height_p90_abs_error_m,
        candidate.height_p90_abs_error_m,
    )

    if mae_reduction < thresholds.min_mae_reduction_fraction:
        reasons.append(
            f"building-height MAE reduction {mae_reduction:.3%} is below required "
            f"{thresholds.min_mae_reduction_fraction:.3%}"
        )
    if rmse_reduction < thresholds.min_rmse_reduction_fraction:
        reasons.append(
            f"building-height RMSE reduction {rmse_reduction:.3%} is below required "
            f"{thresholds.min_rmse_reduction_fraction:.3%}"
        )
    if p90_reduction < thresholds.min_p90_reduction_fraction:
        reasons.append(
            f"building-height P90 reduction {p90_reduction:.3%} is below required "
            f"{thresholds.min_p90_reduction_fraction:.3%}"
        )
    if (
        candidate.within_2m_fraction
        < baseline.within_2m_fraction - thresholds.max_within_2m_fraction_drop
    ):
        reasons.append("fraction of buildings within 2 m regressed beyond the allowed tolerance")
    if (
        candidate.catastrophic_over_3m_fraction
        > baseline.catastrophic_over_3m_fraction
        + thresholds.max_catastrophic_fraction_increase
    ):
        reasons.append("catastrophic >3 m building-error rate increased")

    return BuildingHeightPromotionDecision(
        passed=not reasons,
        reasons=tuple(reasons),
        mae_reduction_fraction=mae_reduction,
        rmse_reduction_fraction=rmse_reduction,
        p90_reduction_fraction=p90_reduction,
    )
