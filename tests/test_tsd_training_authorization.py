from __future__ import annotations

from copy import deepcopy

from depthwizard.height_model.terrain_structure_campaign import (
    assess_tsd_training_authorization,
)


def _observed_audit() -> dict[str, object]:
    return {
        "status": "AUDITED_TSD_TARGET_SUPERVISION",
        "protocol_version": "potsdam-tsd-target-supervision-audit-v1",
        "training_authorized": False,
        "train": {
            "accepted_building_pixel_fraction": 0.951,
            "accepted_height_ge_8m": 119,
            "rejected_height_ge_8m": 11,
            "measured_ge_8m_rejection_fraction": 11 / 130,
            "accepted_height_ge_12m": 86,
            "rejected_height_ge_12m": 4,
            "measured_ge_12m_rejection_fraction": 4 / 90,
        },
        "dev": {
            "accepted_building_pixel_fraction": 0.897,
            "accepted_height_ge_8m": 102,
            "rejected_height_ge_8m": 12,
            "measured_ge_8m_rejection_fraction": 12 / 114,
            "accepted_height_ge_12m": 36,
            "rejected_height_ge_12m": 4,
            "measured_ge_12m_rejection_fraction": 4 / 40,
        },
        "overall": {
            "accepted_building_pixel_fraction": 0.937,
            "accepted_height_ge_8m": 221,
            "rejected_height_ge_8m": 23,
            "measured_ge_8m_rejection_fraction": 23 / 244,
            "accepted_height_ge_12m": 122,
            "rejected_height_ge_12m": 8,
            "measured_ge_12m_rejection_fraction": 8 / 130,
        },
        "rejected_measured_structures_ge_8m": [
            {"reference_height_m": 17.019},
            {"reference_height_m": 16.623},
            {"reference_height_m": 13.870},
        ],
    }


def test_observed_target_distribution_authorizes_training() -> None:
    result = assess_tsd_training_authorization(_observed_audit())
    assert result["training_authorized"] is True
    assert result["status"] == "AUTHORIZED_TSD_TRAINING"
    assert result["failed_checks"] == []


def test_gate_rejects_low_dev_building_pixel_support() -> None:
    audit = deepcopy(_observed_audit())
    dev = audit["dev"]
    assert isinstance(dev, dict)
    dev["accepted_building_pixel_fraction"] = 0.80
    result = assess_tsd_training_authorization(audit)
    assert result["training_authorized"] is False
    assert any(
        item["name"] == "dev_building_pixel_support" for item in result["failed_checks"]
    )


def test_gate_rejects_lost_extreme_structure() -> None:
    audit = deepcopy(_observed_audit())
    tall = audit["rejected_measured_structures_ge_8m"]
    assert isinstance(tall, list)
    tall.append({"reference_height_m": 24.0})
    result = assess_tsd_training_authorization(audit)
    assert result["training_authorized"] is False
    assert any(
        item["name"] == "no_rejected_measured_structure_at_or_above_20m"
        for item in result["failed_checks"]
    )
