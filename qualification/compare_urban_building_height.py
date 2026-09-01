from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any, cast

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.building_height import (
    BuildingHeightBenchmarkReport,
    BuildingHeightPromotionThresholds,
    building_height_promotion_gate,
    building_height_report_from_dict,
)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"building-height report must contain a JSON object: {path}")
    typed = cast(dict[str, Any], payload)
    if typed.get("schema_version") != 1:
        raise ValueError(f"unsupported building-height report schema: {path}")
    return typed


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"building-height report field {key!r} must be a JSON object")
    return cast(dict[str, Any], value)


def _shared_contract(baseline: dict[str, Any], candidate: dict[str, Any]) -> None:
    baseline_inputs = _mapping(baseline, "inputs")
    candidate_inputs = _mapping(candidate, "inputs")
    for key in (
        "reference_sha256",
        "building_mask_sha256",
        "ground_mask_sha256",
        "gsd_x_m",
        "gsd_y_m",
    ):
        if baseline_inputs.get(key) != candidate_inputs.get(key):
            raise ValueError(f"baseline/candidate benchmark contract differs at {key}")
    if _mapping(baseline, "config") != _mapping(candidate, "config"):
        raise ValueError("baseline/candidate benchmark configuration differs")


def _validate_report_integrity(report: BuildingHeightBenchmarkReport, *, role: str) -> None:
    """Fail closed on corrupted, non-finite, or internally inconsistent persisted evidence."""
    finite_fields = (
        "height_mae_m",
        "height_rmse_m",
        "height_bias_m",
        "height_median_abs_error_m",
        "height_p90_abs_error_m",
        "height_p95_abs_error_m",
        "top_mae_m",
        "ground_mae_m",
        "within_1m_fraction",
        "within_2m_fraction",
        "catastrophic_over_3m_fraction",
        "mean_reference_height_m",
        "mean_predicted_height_m",
    )
    for field in finite_fields:
        value = float(getattr(report, field))
        if not isfinite(value):
            raise ValueError(f"{role} report contains non-finite {field}: {value!r}")

    nonnegative_fields = (
        "height_mae_m",
        "height_rmse_m",
        "height_median_abs_error_m",
        "height_p90_abs_error_m",
        "height_p95_abs_error_m",
        "top_mae_m",
        "ground_mae_m",
    )
    for field in nonnegative_fields:
        if float(getattr(report, field)) < 0.0:
            raise ValueError(f"{role} report contains negative error metric {field}")

    for field in ("within_1m_fraction", "within_2m_fraction", "catastrophic_over_3m_fraction"):
        value = float(getattr(report, field))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{role} report fraction {field} must lie in [0, 1]")

    eligible = report.eligible_instance_ids
    evaluated = report.evaluated_instance_ids
    failures = report.prediction_failure_ids
    if len(set(eligible)) != len(eligible):
        raise ValueError(f"{role} report contains duplicate eligible instance ids")
    if len(set(evaluated)) != len(evaluated):
        raise ValueError(f"{role} report contains duplicate evaluated instance ids")
    if len(set(failures)) != len(failures):
        raise ValueError(f"{role} report contains duplicate prediction-failure instance ids")
    evaluated_set = set(evaluated)
    failure_set = set(failures)
    if evaluated_set & failure_set:
        raise ValueError(f"{role} report marks an instance as both evaluated and failed")
    if evaluated_set | failure_set != set(eligible):
        raise ValueError(
            f"{role} report eligibility is inconsistent with evaluated/failure instance support"
        )
    instance_ids = tuple(item.instance_id for item in report.instances)
    if instance_ids != evaluated:
        raise ValueError(f"{role} report instance rows do not match evaluated instance ids")
    if report.skipped_reference_instances < 0:
        raise ValueError(f"{role} report contains a negative skipped-reference count")

    for instance in report.instances:
        for field, value in asdict(instance).items():
            if isinstance(value, float) and not isfinite(value):
                raise ValueError(
                    f"{role} report instance {instance.instance_id} contains non-finite {field}"
                )


def _validate_threshold_args(args: argparse.Namespace) -> None:
    for name in ("min_mae_reduction", "min_rmse_reduction", "min_p90_reduction"):
        value = float(getattr(args, name))
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be a finite fraction in [0, 1]")
    if args.min_instances < 1:
        raise ValueError("--min-instances must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two exposed urban building-height reports using material-effect promotion gates. "
            "Passing this gate is necessary but never sufficient for production promotion."
        )
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-instances", type=int, default=20)
    parser.add_argument("--min-mae-reduction", type=float, default=0.15)
    parser.add_argument("--min-rmse-reduction", type=float, default=0.10)
    parser.add_argument("--min-p90-reduction", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _validate_threshold_args(args)
    baseline_payload = _load(args.baseline)
    candidate_payload = _load(args.candidate)
    _shared_contract(baseline_payload, candidate_payload)

    baseline = building_height_report_from_dict(_mapping(baseline_payload, "report"))
    candidate = building_height_report_from_dict(_mapping(candidate_payload, "report"))
    _validate_report_integrity(baseline, role="baseline")
    _validate_report_integrity(candidate, role="candidate")
    thresholds = BuildingHeightPromotionThresholds(
        min_instances=args.min_instances,
        min_mae_reduction_fraction=args.min_mae_reduction,
        min_rmse_reduction_fraction=args.min_rmse_reduction,
        min_p90_reduction_fraction=args.min_p90_reduction,
    )
    decision = building_height_promotion_gate(baseline, candidate, thresholds=thresholds)

    output = {
        "schema_version": 1,
        "status": "PASS_BUILDING_HEIGHT_GATE" if decision.passed else "FAIL_BUILDING_HEIGHT_GATE",
        "claim_boundary": (
            "This building-instance gate is necessary but not sufficient for model promotion. "
            "Whole-scene DSM safety, terrain non-degradation, operator validation, reserved blind "
            "evaluation, and production policy review remain mandatory."
        ),
        "baseline_report": str(args.baseline.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "thresholds": asdict(thresholds),
        "decision": asdict(decision),
        "baseline_summary": {
            "evaluated_instances": len(baseline.evaluated_instance_ids),
            "height_mae_m": baseline.height_mae_m,
            "height_rmse_m": baseline.height_rmse_m,
            "height_p90_abs_error_m": baseline.height_p90_abs_error_m,
            "top_mae_m": baseline.top_mae_m,
            "ground_mae_m": baseline.ground_mae_m,
            "within_2m_fraction": baseline.within_2m_fraction,
            "catastrophic_over_3m_fraction": baseline.catastrophic_over_3m_fraction,
        },
        "candidate_summary": {
            "evaluated_instances": len(candidate.evaluated_instance_ids),
            "prediction_failures": len(candidate.prediction_failure_ids),
            "height_mae_m": candidate.height_mae_m,
            "height_rmse_m": candidate.height_rmse_m,
            "height_p90_abs_error_m": candidate.height_p90_abs_error_m,
            "top_mae_m": candidate.top_mae_m,
            "ground_mae_m": candidate.ground_mae_m,
            "within_2m_fraction": candidate.within_2m_fraction,
            "catastrophic_over_3m_fraction": candidate.catastrophic_over_3m_fraction,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(args.output)

    print(f"status={output['status']}")
    print(f"mae_reduction_fraction={decision.mae_reduction_fraction:.6f}")
    print(f"rmse_reduction_fraction={decision.rmse_reduction_fraction:.6f}")
    print(f"p90_reduction_fraction={decision.p90_reduction_fraction:.6f}")
    for reason in decision.reasons:
        print(f"reason={reason}")
    return 0 if decision.passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
