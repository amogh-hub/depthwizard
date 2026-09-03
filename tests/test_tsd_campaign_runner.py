from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from qualification.audit_tsd_target_supervision import (
    EXPECTED_DEV_TILE_IDS,
    EXPECTED_TRAIN_TILE_IDS,
    audit_target_supervision,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_instances(path: Path, tile_id: str, role: str, tall: bool) -> None:
    payload = {
        "tile_id": tile_id,
        "role": role,
        "instances": [
            {
                "instance_id": 1,
                "accepted": True,
                "reason": "accepted",
                "area_m2": 100.0,
                "reference_height_m": 10.0,
                "ground_candidate_pixels": 100,
                "ground_inlier_pixels": 90,
                "ground_inlier_fraction": 0.9,
                "ground_sector_coverage": 1.0,
                "negative_agl_fraction": 0.0,
            },
            {
                "instance_id": 2,
                "accepted": False,
                "reason": "insufficient strict-ground candidates for dense target plane",
                "area_m2": 40.0,
                "reference_height_m": 12.5 if tall else None,
                "ground_candidate_pixels": 10,
                "ground_inlier_pixels": 0,
                "ground_inlier_fraction": None,
                "ground_sector_coverage": None,
                "negative_agl_fraction": None,
            },
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_manifest(tmp_path: Path) -> Path:
    records = []
    all_tiles = sorted(EXPECTED_TRAIN_TILE_IDS | EXPECTED_DEV_TILE_IDS)
    for index, tile_id in enumerate(all_tiles):
        role = "train" if tile_id in EXPECTED_TRAIN_TILE_IDS else "dev"
        instances = tmp_path / f"{tile_id}.json"
        _write_instances(instances, tile_id, role, tall=index == 0)
        records.append(
            {
                "tile_id": tile_id,
                "role": role,
                "instances": str(instances),
                "instances_sha256": _digest(instances),
                "accepted_instance_count": 1,
                "rejected_instance_count": 1,
                "source_semantic_building_pixels": 1000,
                "accepted_building_pixels": 700,
            }
        )
    path = tmp_path / "targets.json"
    path.write_text(
        json.dumps(
            {
                "status": "GENERATED_TSD_METRIC_TARGETS",
                "protocol_version": "potsdam-tsd-metric-targets-v1",
                "tiles": records,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_target_audit_reports_tall_rejections_and_stays_fail_closed(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    audit = audit_target_supervision(manifest, expected_manifest_sha256=_digest(manifest))

    assert audit["status"] == "AUDITED_TSD_TARGET_SUPERVISION"
    assert audit["training_authorized"] is False
    assert audit["overall"]["accepted_instances"] == 13
    assert audit["overall"]["rejected_instances"] == 13
    assert audit["overall"]["accepted_building_pixel_fraction"] == pytest.approx(0.7)
    assert len(audit["rejected_measured_structures_ge_8m"]) == 1


def test_target_audit_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    manifest = _make_manifest(tmp_path)
    with pytest.raises(ValueError, match="target manifest SHA-256 mismatch"):
        audit_target_supervision(manifest, expected_manifest_sha256="0" * 64)


def test_campaign_runner_can_be_launched_as_a_file() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "qualification" / "run_tsd_campaign.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Advance the frozen Potsdam TSD campaign" in result.stdout
