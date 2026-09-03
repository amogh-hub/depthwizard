from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TerrainStructureTrainingGate:
    """Predeclared data-adequacy gate for the first Potsdam TSD campaign.

    This gate controls only whether the already generated train/dev supervision is adequate to begin
    optimization. It is not a model-performance threshold and it does not authorize any reserved,
    challenge-test, exposed-corrective, external-evaluation, or sealed-blind evidence.
    """

    min_overall_building_pixel_support: float = 0.90
    min_train_building_pixel_support: float = 0.90
    min_dev_building_pixel_support: float = 0.85
    max_overall_ge_8m_rejection_fraction: float = 0.12
    max_split_ge_8m_rejection_fraction: float = 0.15
    max_overall_ge_12m_rejection_fraction: float = 0.10
    max_split_ge_12m_rejection_fraction: float = 0.15
    min_train_accepted_ge_8m: int = 80
    min_dev_accepted_ge_8m: int = 50
    min_train_accepted_ge_12m: int = 50
    min_dev_accepted_ge_12m: int = 25
    max_rejected_measured_height_m: float = 20.0

    def __post_init__(self) -> None:
        fraction_fields = (
            self.min_overall_building_pixel_support,
            self.min_train_building_pixel_support,
            self.min_dev_building_pixel_support,
            self.max_overall_ge_8m_rejection_fraction,
            self.max_split_ge_8m_rejection_fraction,
            self.max_overall_ge_12m_rejection_fraction,
            self.max_split_ge_12m_rejection_fraction,
        )
        if any(value < 0.0 or value > 1.0 for value in fraction_fields):
            raise ValueError("TSD training-gate fractions must be in [0, 1]")
        count_fields = (
            self.min_train_accepted_ge_8m,
            self.min_dev_accepted_ge_8m,
            self.min_train_accepted_ge_12m,
            self.min_dev_accepted_ge_12m,
        )
        if any(value < 1 for value in count_fields):
            raise ValueError("TSD training-gate sample-count floors must be positive")
        if self.max_rejected_measured_height_m <= 0:
            raise ValueError("max_rejected_measured_height_m must be positive")


def _required_mapping(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"audit section {key!r} must be an object")
    return value


def _required_float(mapping: dict[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"audit field {key!r} must be numeric")
    return float(value)


def _required_int(mapping: dict[str, Any], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int):
        raise TypeError(f"audit field {key!r} must be an integer")
    return value


def assess_tsd_training_authorization(
    audit: dict[str, Any],
    *,
    gate: TerrainStructureTrainingGate | None = None,
) -> dict[str, Any]:
    """Assess whether frozen supervision is adequate to begin the first TSD optimization run.

    The audit is expected to have been produced by ``potsdam-tsd-target-supervision-audit-v1``.
    The gate intentionally focuses on retained building-pixel support and measured tall-structure
    support rather than raw connected-component acceptance, because tiny components dominate the
    rejection count while contributing little useful roof-area supervision.
    """

    cfg = gate or TerrainStructureTrainingGate()
    if audit.get("status") != "AUDITED_TSD_TARGET_SUPERVISION":
        raise ValueError("training authorization requires an audited TSD supervision payload")
    if audit.get("protocol_version") != "potsdam-tsd-target-supervision-audit-v1":
        raise ValueError("unexpected TSD supervision audit protocol")
    if audit.get("training_authorized") is not False:
        raise ValueError("diagnostic target audit must remain fail-closed before authorization")

    train = _required_mapping(audit, "train")
    dev = _required_mapping(audit, "dev")
    overall = _required_mapping(audit, "overall")
    tall_rejections_raw = audit.get("rejected_measured_structures_ge_8m")
    if not isinstance(tall_rejections_raw, list):
        raise TypeError("audit tall-rejection evidence must be a list")
    tall_rejections = [item for item in tall_rejections_raw if isinstance(item, dict)]
    if len(tall_rejections) != len(tall_rejections_raw):
        raise TypeError("every tall-rejection evidence item must be an object")

    checks: list[dict[str, Any]] = []

    def require_min(name: str, actual: float, minimum: float) -> None:
        checks.append(
            {
                "name": name,
                "passed": actual >= minimum,
                "actual": actual,
                "operator": ">=",
                "threshold": minimum,
            }
        )

    def require_max(name: str, actual: float, maximum: float) -> None:
        checks.append(
            {
                "name": name,
                "passed": actual <= maximum,
                "actual": actual,
                "operator": "<=",
                "threshold": maximum,
            }
        )

    require_min(
        "overall_building_pixel_support",
        _required_float(overall, "accepted_building_pixel_fraction"),
        cfg.min_overall_building_pixel_support,
    )
    require_min(
        "train_building_pixel_support",
        _required_float(train, "accepted_building_pixel_fraction"),
        cfg.min_train_building_pixel_support,
    )
    require_min(
        "dev_building_pixel_support",
        _required_float(dev, "accepted_building_pixel_fraction"),
        cfg.min_dev_building_pixel_support,
    )
    require_max(
        "overall_measured_ge_8m_rejection_fraction",
        _required_float(overall, "measured_ge_8m_rejection_fraction"),
        cfg.max_overall_ge_8m_rejection_fraction,
    )
    require_max(
        "train_measured_ge_8m_rejection_fraction",
        _required_float(train, "measured_ge_8m_rejection_fraction"),
        cfg.max_split_ge_8m_rejection_fraction,
    )
    require_max(
        "dev_measured_ge_8m_rejection_fraction",
        _required_float(dev, "measured_ge_8m_rejection_fraction"),
        cfg.max_split_ge_8m_rejection_fraction,
    )
    require_max(
        "overall_measured_ge_12m_rejection_fraction",
        _required_float(overall, "measured_ge_12m_rejection_fraction"),
        cfg.max_overall_ge_12m_rejection_fraction,
    )
    require_max(
        "train_measured_ge_12m_rejection_fraction",
        _required_float(train, "measured_ge_12m_rejection_fraction"),
        cfg.max_split_ge_12m_rejection_fraction,
    )
    require_max(
        "dev_measured_ge_12m_rejection_fraction",
        _required_float(dev, "measured_ge_12m_rejection_fraction"),
        cfg.max_split_ge_12m_rejection_fraction,
    )
    require_min(
        "train_accepted_ge_8m",
        float(_required_int(train, "accepted_height_ge_8m")),
        float(cfg.min_train_accepted_ge_8m),
    )
    require_min(
        "dev_accepted_ge_8m",
        float(_required_int(dev, "accepted_height_ge_8m")),
        float(cfg.min_dev_accepted_ge_8m),
    )
    require_min(
        "train_accepted_ge_12m",
        float(_required_int(train, "accepted_height_ge_12m")),
        float(cfg.min_train_accepted_ge_12m),
    )
    require_min(
        "dev_accepted_ge_12m",
        float(_required_int(dev, "accepted_height_ge_12m")),
        float(cfg.min_dev_accepted_ge_12m),
    )

    rejected_heights = [
        float(item["reference_height_m"])
        for item in tall_rejections
        if isinstance(item.get("reference_height_m"), (int, float))
    ]
    max_rejected = max(rejected_heights) if rejected_heights else 0.0
    checks.append(
        {
            "name": "no_rejected_measured_structure_at_or_above_20m",
            "passed": max_rejected < cfg.max_rejected_measured_height_m,
            "actual": max_rejected,
            "operator": "<",
            "threshold": cfg.max_rejected_measured_height_m,
        }
    )

    failures = [item for item in checks if item["passed"] is not True]
    return {
        "schema_version": 1,
        "status": "AUTHORIZED_TSD_TRAINING" if not failures else "REJECTED_TSD_TRAINING",
        "protocol_version": "potsdam-tsd-training-authorization-v1",
        "training_authorized": not failures,
        "gate": asdict(cfg),
        "checks": checks,
        "failed_checks": failures,
        "claim_boundary": (
            "Authorization applies only to optimization on the frozen potsdam-tsd-urban-spatial-v1 "
            "train/dev supervision. It is not a model-performance pass and does not authorize any "
            "reserved, challenge-test, exposed-corrective, external-evaluation, or sealed-blind tile."
        ),
    }
