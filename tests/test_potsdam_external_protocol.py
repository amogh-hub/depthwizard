from __future__ import annotations

import json
from pathlib import Path

from depthwizard.evaluation.potsdam import (
    FROZEN_POTSDAM_TILE_IDS,
    POTSDAM_BENCHMARK_GSD_M,
    POTSDAM_NATIVE_GSD_M,
    protocol_sha256,
    resolve_potsdam_tile_paths,
    write_or_verify_protocol_seal,
)


def test_potsdam_external_contract_is_frozen_and_downsampled() -> None:
    assert FROZEN_POTSDAM_TILE_IDS == ("2_10", "3_13", "5_11", "6_14")
    assert POTSDAM_NATIVE_GSD_M == 0.05
    assert POTSDAM_BENCHMARK_GSD_M == 0.25


def test_potsdam_resolver_finds_official_rgb_and_dsm_names(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb" / "top_potsdam_2_10_RGB.tif"
    dsm = tmp_path / "dsm" / "dsm_potsdam_02_10.tif"
    rgb.parent.mkdir()
    dsm.parent.mkdir()
    rgb.touch()
    dsm.touch()

    resolved = resolve_potsdam_tile_paths(tmp_path, "2_10")

    assert resolved.rgb == rgb
    assert resolved.reference_dsm == dsm


def test_protocol_seal_is_deterministic_and_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "protocol_seal.json"
    payload: dict[str, object] = {
        "protocol_version": "potsdam-external-v1",
        "tiles": ["2_10", "3_13"],
        "benchmark_gsd_m": 0.25,
    }

    first = write_or_verify_protocol_seal(path, payload)
    second = write_or_verify_protocol_seal(path, payload)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert first == second == protocol_sha256(payload)
    assert document["protocol_sha256"] == first
    assert document["protocol"] == payload


def test_protocol_seal_rejects_post_hoc_mutation(tmp_path: Path) -> None:
    path = tmp_path / "protocol_seal.json"
    write_or_verify_protocol_seal(path, {"tiles": ["2_10"]})

    try:
        write_or_verify_protocol_seal(path, {"tiles": ["2_10", "3_13"]})
    except RuntimeError as exc:
        assert "different contents" in str(exc)
    else:
        raise AssertionError("a consumed external protocol must not be mutable")
