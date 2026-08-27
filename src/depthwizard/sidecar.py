from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def _startup_trace(phase: str, **details: object) -> None:
    """Best-effort startup tracing for packaged-runtime diagnostics.

    The trace is disabled during normal launches unless DEPTHWIZARD_STARTUP_TRACE is set. It never
    records the session token or other secrets, and tracing failures must never block the sidecar.
    """
    raw_path = os.environ.get("DEPTHWIZARD_STARTUP_TRACE")
    if not raw_path:
        return
    payload = {"phase": phase, "pid": os.getpid(), **details}
    try:
        path = Path(raw_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except (OSError, TypeError, ValueError):
        return


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="depthwizard-core",
        description="DepthWizard packaged local scientific sidecar",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-level", default="warning")
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="validate the frozen geospatial runtime and exit without starting the server",
    )
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


def packaged_geospatial_self_check() -> dict[str, object]:
    """Exercise frozen Rasterio/GDAL/PROJ wiring without touching model or network state."""
    _startup_trace("self_check_import_start")
    import numpy as np
    import pyproj
    import rasterio
    import rasterio.serde as rasterio_serde
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    _startup_trace("self_check_import_complete")

    pyproj_crs = pyproj.CRS.from_epsg(32643)
    if pyproj_crs.to_epsg() != 32643:
        raise RuntimeError("PyProj failed to resolve EPSG:32643 from bundled PROJ data")

    data = np.zeros((1, 2, 2), dtype=np.uint8)
    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
            crs="EPSG:32643",
            transform=from_origin(500000, 1400000, 1.0, 1.0),
        ) as dataset:
            dataset.write(data)
        with memory_file.open() as dataset:
            rasterio_epsg = dataset.crs.to_epsg() if dataset.crs is not None else None
            if rasterio_epsg != 32643:
                raise RuntimeError(
                    "Rasterio/GDAL failed to round-trip EPSG:32643 from bundled geospatial data"
                )

    report = {
        "schema_version": 1,
        "status": "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK",
        "rasterio_version": rasterio.__version__,
        "gdal_version": rasterio.__gdal_version__,
        "pyproj_version": pyproj.__version__,
        "rasterio_serde_imported": rasterio_serde.__name__ == "rasterio.serde",
        "epsg_roundtrip": 32643,
        "network_used": False,
        "model_loaded": False,
    }
    _startup_trace("self_check_complete", status=report["status"])
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged local core without requiring a user-visible terminal."""
    _startup_trace("python_entry")
    parser = _parser()
    args = parser.parse_args(argv)
    _startup_trace("arguments_parsed", self_check=bool(args.self_check))

    if args.self_check:
        report = packaged_geospatial_self_check()
        print(json.dumps(report, sort_keys=True), flush=True)
        return

    if args.port is None:
        parser.error("--port is required unless --self-check is used")

    validate_launch_environment(host=args.host, port=args.port)
    _startup_trace("launch_environment_validated", port=args.port)

    # Standalone mode is deliberately offline after model installation. Hugging Face-compatible
    # model loaders must use the local cache instead of silently reaching the network. The Python
    # socket guard adds a second fail-closed boundary: INET connections are allowed only to loopback.
    if os.environ.get("DEPTHWIZARD_OFFLINE_CORE") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from depthwizard.network_guard import install_strict_offline_network_guard

        install_strict_offline_network_guard()
        _startup_trace("offline_network_guard_installed")

    _startup_trace("uvicorn_import_start")
    import uvicorn

    _startup_trace("uvicorn_import_complete")
    _startup_trace("service_import_start")
    from depthwizard.service import app

    _startup_trace("service_import_complete")
    _startup_trace("server_start")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


if __name__ == "__main__":
    main()
