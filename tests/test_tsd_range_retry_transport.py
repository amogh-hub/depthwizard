from __future__ import annotations

import io
import urllib.error
from email.message import Message
from typing import Any, cast

import pytest

from qualification import run_tsd_potsdam_official_ranges as retry_transport


class _Response(io.BytesIO):
    def __init__(self, payload: bytes, *, start: int, end: int, total: int = 1000) -> None:
        super().__init__(payload)
        self.status = 206
        self.headers = Message()
        self.headers["Content-Range"] = f"bytes {start}-{end}/{total}"

    def getcode(self) -> int:
        return self.status


class _FlakyOpener:
    def __init__(self) -> None:
        self.calls = 0

    def open(self, _request: Any, timeout: int):
        assert timeout == retry_transport.RANGE_OPEN_TIMEOUT_SECONDS
        self.calls += 1
        if self.calls == 1:
            raise urllib.error.URLError(TimeoutError("TLS handshake timed out"))
        return _Response(b"x", start=10, end=10)


class _WrongRangeOpener:
    def __init__(self) -> None:
        self.calls = 0

    def open(self, _request: Any, timeout: int):
        assert timeout == retry_transport.RANGE_OPEN_TIMEOUT_SECONDS
        self.calls += 1
        return _Response(b"x", start=11, end=11)


def test_transient_tls_failure_retries_then_preserves_range_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_transport.time, "sleep", lambda _seconds: None)
    retry_transport.ResilientHttpRangeSource.retry_attempts_total = 0
    opener = _FlakyOpener()
    source = retry_transport.ResilientHttpRangeSource(cast(Any, opener))

    with source._open_range(10, 10) as response:
        assert response.read() == b"x"

    assert opener.calls == 2
    assert source.requests == 1
    assert retry_transport.ResilientHttpRangeSource.retry_attempts_total == 1


def test_archive_range_mismatch_fails_immediately_without_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry_transport.time, "sleep", lambda _seconds: None)
    retry_transport.ResilientHttpRangeSource.retry_attempts_total = 0
    opener = _WrongRangeOpener()
    source = retry_transport.ResilientHttpRangeSource(cast(Any, opener))

    with pytest.raises(RuntimeError, match="different range"):
        source._open_range(10, 10)

    assert opener.calls == 1
    assert retry_transport.ResilientHttpRangeSource.retry_attempts_total == 0
