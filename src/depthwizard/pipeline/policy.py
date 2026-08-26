from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class EstimatorPath(str, Enum):
    """Production estimator paths exposed by the permanent runtime contract."""

    CALIBRATED_DA3 = "calibrated_da3"
    PROMOTED_LEARNED_REFINER = "promoted_learned_refiner"


@dataclass(frozen=True)
class PromotionEvidence:
    """Independent evidence attached to a learned-estimator promotion decision."""

    branch_id: str
    evidence_id: str
    independently_evaluated: bool
    promotion_passed: bool
    summary: str


@dataclass(frozen=True)
class EstimatorDecision:
    selected_path: EstimatorPath
    selected_model_id: str
    reason: str
    evidence: tuple[PromotionEvidence, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_path": self.selected_path.value,
            "selected_model_id": self.selected_model_id,
            "reason": self.reason,
            "evidence": [asdict(item) for item in self.evidence],
        }


def select_production_estimator(
    evidence: tuple[PromotionEvidence, ...] = (),
) -> EstimatorDecision:
    """Choose the safest production estimator from independently reviewed evidence.

    A learned branch is eligible only when it has both independent evaluation and an explicit
    promotion PASS. Development, same-geography or failed external evidence can never silently
    replace the calibrated DA3 baseline.
    """
    eligible = [
        item
        for item in evidence
        if item.independently_evaluated and item.promotion_passed
    ]
    if len(eligible) > 1:
        raise RuntimeError(
            "multiple learned estimator branches are independently promoted; production policy "
            "requires an explicit deterministic tie-break before deployment"
        )
    if eligible:
        promoted = eligible[0]
        return EstimatorDecision(
            selected_path=EstimatorPath.PROMOTED_LEARNED_REFINER,
            selected_model_id=promoted.branch_id,
            reason=(
                "learned refinement selected because its independently frozen promotion contract "
                f"passed: {promoted.evidence_id}"
            ),
            evidence=evidence,
        )
    return EstimatorDecision(
        selected_path=EstimatorPath.CALIBRATED_DA3,
        selected_model_id="DA3MONO-LARGE",
        reason=(
            "calibrated DA3 retained because no learned branch has independently satisfied the "
            "production promotion contract"
        ),
        evidence=evidence,
    )


def current_production_estimator_decision() -> EstimatorDecision:
    """Return the evidence-locked estimator decision for the current production line."""
    return select_production_estimator(
        (
            PromotionEvidence(
                branch_id="DepthWizard-V4-confidence-gated-v2",
                evidence_id=(
                    "potsdam-external-v2:"
                    "3b1ecb2ae9b559d84cc2233af25887e05575889bdbd7b57bd2568b7e98bc7118"
                ),
                independently_evaluated=True,
                promotion_passed=False,
                summary=(
                    "Frozen V4 was slightly worse than independently calibrated DA3 in aggregate "
                    "and violated the per-tile non-degradation criterion."
                ),
            ),
        )
    )
