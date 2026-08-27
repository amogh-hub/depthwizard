from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthwizard-core",
        description="DepthWizard packaged local scientific sidecar",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--log-level", default="warning")
    return parser


def validate_launch_environment(*, host: str, port: int) -> None:
    """Fail closed for packaged standalone launches before starting the HTTP server."""
    if host not in _LOOPBACK_HOSTS:
        raise ValueError("DepthWizard packaged core may bind only to loopback")
    if not 1 <= port <= 65535:
        raise ValueError("DepthWizard packaged core port must be between 1 and 65535")
    if os.environ.get("DEPTHWIZARD_REQUIRE_SESSION_TOKEN") == "1":
        token = os.environ.get("DEPTHWIZARD_SESSION_TOKEN", "")
        if len(token) < 32:
            raise RuntimeError(
                "DepthWizard standalone launch requires a per-session authentication token"
            )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged local core without requiring a user-visible terminal."""
    args = _parser().parse_args(argv)
    validate_launch_environment(host=args.host, port=args.port)

    # Standalone mode is deliberately offline after model installation. Hugging Face-compatible
    # model loaders must use the local cache instead of silently reaching the network. The Python
    # socket guard adds a second fail-closed boundary: INET connections are allowed only to loopback.
    if os.environ.get("DEPTHWIZARD_OFFLINE_CORE") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from depthwizard.network_guard import install_strict_offline_network_guard

        install_strict_offline_network_guard()

    import uvicorn

    from depthwizard.service import app

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
