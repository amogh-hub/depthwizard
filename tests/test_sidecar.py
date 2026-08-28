from __future__ import annotations

import json
from pathlib import Path

import pytest

from depthwizard.sidecar import (
    _startup_trace,
    packaged_geospatial_self_check,
    validate_launch_environment,
)


def test_packaged_sidecar_rejects_non_loopback_host(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "a" * 64)
    with pytest.raises(ValueError, match="loopback"):
        validate_launch_environment(host="0.0.0.0", port=8765)


def test_packaged_sidecar_requires_session_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.delenv("DEPTHWIZARD_SESSION_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="per-session authentication token"):
        validate_launch_environment(host="127.0.0.1", port=8765)


@pytest.mark.parametrize("token", ["a" * 63, "a" * 65, "g" * 64])
def test_packaged_sidecar_rejects_non_256_bit_hex_session_token(
    monkeypatch: pytest.MonkeyPatch,
    token: str,
) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", token)
    with pytest.raises(RuntimeError, match="256-bit hexadecimal"):
        validate_launch_environment(host="127.0.0.1", port=8765)


def test_packaged_sidecar_requires_boot_identity_nonce(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "a" * 64)
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_BOOT_NONCE", "1")
    monkeypatch.delenv("DEPTHWIZARD_BOOT_NONCE", raising=False)
    with pytest.raises(RuntimeError, match="per-process boot identity nonce"):
        validate_launch_environment(host="127.0.0.1", port=8765)


@pytest.mark.parametrize("nonce", ["b" * 63, "b" * 65, "z" * 64])
def test_packaged_sidecar_rejects_non_256_bit_hex_boot_nonce(
    monkeypatch: pytest.MonkeyPatch,
    nonce: str,
) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "a" * 64)
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_BOOT_NONCE", "1")
    monkeypatch.setenv("DEPTHWIZARD_BOOT_NONCE", nonce)
    with pytest.raises(RuntimeError, match="256-bit hexadecimal"):
        validate_launch_environment(host="127.0.0.1", port=8765)


def test_packaged_sidecar_accepts_secured_loopback_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "b" * 64)
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_BOOT_NONCE", "1")
    monkeypatch.setenv("DEPTHWIZARD_BOOT_NONCE", "c" * 64)
    validate_launch_environment(host="127.0.0.1", port=49152)


def test_packaged_geospatial_self_check_exercises_epsg_runtime_without_forcing_da3() -> None:
    report = packaged_geospatial_self_check()
    assert report["status"] == "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK"
    assert report["rasterio_serde_imported"] is True
    assert report["epsg_roundtrip"] == 32643
    assert report["da3_probe_required"] is False
    assert report["da3_api_imported"] is False
    assert report["da3_runtime_modules_imported"] == []
    assert report["da3_runtime_execution_gate"] == "release_train_5_full_acceptance"
    assert report["network_used"] is False
    assert report["model_loaded"] is False


def test_startup_trace_is_machine_readable_and_secret_free(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace = tmp_path / "startup.jsonl"
    monkeypatch.setenv("DEPTHWIZARD_STARTUP_TRACE", str(trace))
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "secret-token-that-must-not-be-recorded")
    monkeypatch.setenv("DEPTHWIZARD_BOOT_NONCE", "secret-boot-nonce-that-must-not-be-recorded")

    _startup_trace("test_phase", port=43210)

    text = trace.read_text(encoding="utf-8")
    payload = json.loads(text.strip())
    assert payload["phase"] == "test_phase"
    assert payload["port"] == 43210
    assert "secret-token-that-must-not-be-recorded" not in text
    assert "secret-boot-nonce-that-must-not-be-recorded" not in text
