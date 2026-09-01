from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.evaluation.building_height import (
    BuildingHeightPromotionThresholds,
    building_height_promotion_gate,
    building_height_report_from_dict,
)


def _load(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported building-height report schema: {path}")
    return payload


def _shared_contract(baseline: dict[str, object], candidate: dict[str, object]) -> None:
    baseline_inputs = dict(baseline["inputs"])
    candidate_inputs = dict(candidate["inputs"])
    for key in ("reference_sha256", "building_mask_sha256", "ground_mask_sha256", "gsd_x_m", "gsd_y_m"):
        if baseline_inputs.get(key) != candidate_inputs.get(key):
            raise ValueError(f"baseline/candidate benchmark contract differs at {key}")
    if baseline.get("config") != candidate.get("config"):
        raise ValueError("baseline/candidate benchmark configuration differs")


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
    baseline_payload = _load(args.baseline)
    candidate_payload = _load(args.candidate)
    _shared_contract(baseline_payload, candidate_payload)

    baseline = building_height_report_from_dict(dict(baseline_payload["report"]))
    candidate = building_height_report_from_dict(dict(candidate_payload["report"]))
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
            "within_2m_fraction": baseline.within_2m_fraction,
            "catastrophic_over_3m_fraction": baseline.catastrophic_over_3m_fraction,
        },
        "candidate_summary": {
            "evaluated_instances": len(candidate.evaluated_instance_ids),
            "prediction_failures": len(candidate.prediction_failure_ids),
            "height_mae_m": candidate.height_mae_m,
            "height_rmse_m": candidate.height_rmse_m,
            "height_p90_abs_error_m": candidate.height_p90_abs_error_m,
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
