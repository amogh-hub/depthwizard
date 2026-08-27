from __future__ import annotations

import pytest

from depthwizard.network_guard import assert_loopback_destination, is_loopback_host


@pytest.mark.parametrize(
    "host",
    ["127.0.0.1", "127.12.34.56", "::1", "localhost", "LOCALHOST."],
)
def test_loopback_hosts_are_accepted(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize(
    "host",
    ["8.8.8.8", "1.1.1.1", "example.com", "huggingface.co", "0.0.0.0"],
)
def test_non_loopback_hosts_are_rejected(host: str) -> None:
    assert not is_loopback_host(host)
    with pytest.raises(OSError, match="blocks non-loopback network egress"):
        assert_loopback_destination((host, 443))


def test_unrecognized_destination_fails_closed() -> None:
    with pytest.raises(OSError, match="unrecognized network destination"):
        assert_loopback_destination("not-an-inet-address")
