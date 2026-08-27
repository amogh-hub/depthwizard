from __future__ import annotations

import ipaddress
import socket
from typing import Any, cast

_ORIGINAL_CONNECT = socket.socket.connect
_ORIGINAL_CONNECT_EX = socket.socket.connect_ex
_INSTALLED = False


def is_loopback_host(host: str | bytes) -> bool:
    """Return whether an INET host is unambiguously loopback without DNS resolution."""
    if isinstance(host, bytes):
        try:
            host = host.decode("ascii")
        except UnicodeDecodeError:
            return False
    normalized = host.strip().lower().rstrip(".")
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        # Hostnames other than localhost are deliberately not resolved in strict offline mode.
        return False


def assert_loopback_destination(address: object) -> None:
    """Reject a network destination unless it is explicitly loopback."""
    if not isinstance(address, tuple) or not address:
        raise OSError("DepthWizard offline core rejected an unrecognized network destination")
    host = address[0]
    if not isinstance(host, (str, bytes)) or not is_loopback_host(host):
        raise OSError(
            "DepthWizard offline core blocks non-loopback network egress; "
            f"destination={host!r}"
        )


def install_strict_offline_network_guard() -> bool:
    """Install a process-wide Python INET connect guard. Idempotent."""
    global _INSTALLED
    if _INSTALLED:
        return False

    def guarded_connect(sock: socket.socket, address: Any) -> Any:
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            assert_loopback_destination(address)
        return _ORIGINAL_CONNECT(sock, address)

    def guarded_connect_ex(sock: socket.socket, address: Any) -> int:
        if sock.family in {socket.AF_INET, socket.AF_INET6}:
            assert_loopback_destination(address)
        return int(_ORIGINAL_CONNECT_EX(sock, address))

    socket_type = cast(Any, socket.socket)
    socket_type.connect = guarded_connect
    socket_type.connect_ex = guarded_connect_ex
    _INSTALLED = True
    return True


def strict_offline_network_guard_installed() -> bool:
    return _INSTALLED
