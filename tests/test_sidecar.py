from __future__ import annotations

import pytest

from depthwizard.sidecar import validate_launch_environment


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


def test_packaged_sidecar_accepts_secured_loopback_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_REQUIRE_SESSION_TOKEN", "1")
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "b" * 64)
    validate_launch_environment(host="127.0.0.1", port=49152)
