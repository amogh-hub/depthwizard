from __future__ import annotations

import argparse
import importlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_DA3_PRODUCTION_RUNTIME_MODULES = (
    "depth_anything_3.api",
    "depth_anything_3.model.da3",
    "depth_anything_3.model.dinov2.dinov2",
    "depth_anything_3.model.dpt",
)


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
        help="validate the frozen geospatial runtime wiring, then exit",
    )
    parser.add_argument(
        "--self-check-da3",
        action="store_true",
        help=(
            "diagnostically import the frozen DA3 runtime closure and geometry helper after the "
            "geospatial self-check; final RT5 acceptance still requires real offline DA3 inference"
        ),
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


def packaged_geospatial_self_check(*, require_da3: bool = False) -> dict[str, object]:
    """Exercise the frozen geospatial runtime without forcing heavy ML initialization.

    The build-time qualification intentionally proves Rasterio/GDAL/PROJ correctness and exact
    frozen-process startup only. A full DA3 import can take substantially longer on a cold macOS
    process because it initializes the PyTorch/vision/model stack; using that import as the build
    watchdog conflates package correctness with ML cold-start cost. The final RT5 acceptance is the
    stronger gate: it launches the real packaged application offline and performs an actual DA3
    reconstruction before validation, mesh generation, export, and lifecycle acceptance.

    ``require_da3=True`` remains available as a targeted diagnostic probe. It is not the build gate.
    """
    _startup_trace("self_check_import_start")

    _startup_trace("self_check_numpy_import_start")
    import numpy as np

    _startup_trace("self_check_numpy_import_complete")
    _startup_trace("self_check_pyproj_import_start")
    import pyproj

    _startup_trace("self_check_pyproj_import_complete")
    _startup_trace("self_check_rasterio_import_start")
    import rasterio

    _startup_trace("self_check_rasterio_import_complete")
    _startup_trace("self_check_rasterio_serde_import_start")
    import rasterio.serde as rasterio_serde

    _startup_trace("self_check_rasterio_serde_import_complete")
    _startup_trace("self_check_rasterio_io_import_start")
    from rasterio.io import MemoryFile
    from rasterio.transform import from_origin

    _startup_trace("self_check_rasterio_io_import_complete")
    _startup_trace("self_check_import_complete")

    _startup_trace("self_check_pyproj_epsg_start")
    pyproj_crs = pyproj.CRS.from_epsg(32643)
    if pyproj_crs.to_epsg() != 32643:
        raise RuntimeError("PyProj failed to resolve EPSG:32643 from bundled PROJ data")
    _startup_trace("self_check_pyproj_epsg_complete")

    _startup_trace("self_check_rasterio_roundtrip_start")
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
    _startup_trace("self_check_rasterio_roundtrip_complete")

    da3_api_imported = False
    da3_runtime_modules_imported: list[str] = []
    da3_geometry_imported = False
    da3_affine_inverse_probe = "DEFERRED_TO_RT5_FULL_PACKAGED_INFERENCE"
    torch_version: str | None = None
    if require_da3:
        _startup_trace("self_check_da3_api_import_start")
        try:
            for module_name in _DA3_PRODUCTION_RUNTIME_MODULES:
                _startup_trace("self_check_da3_module_import_start", module=module_name)
                importlib.import_module(module_name)
                da3_runtime_modules_imported.append(module_name)
                _startup_trace("self_check_da3_module_import_complete", module=module_name)
            da3_api: Any = importlib.import_module("depth_anything_3.api")
        except ImportError as exc:
            missing = getattr(exc, "name", None)
            raise RuntimeError(
                "Frozen DA3 runtime import closure is incomplete: "
                f"{type(exc).__name__}: {exc}; missing_module={missing!r}"
            ) from exc
        if getattr(da3_api, "DepthAnything3", None) is None:
            raise RuntimeError("Frozen depth_anything_3.api does not expose DepthAnything3")
        da3_api_imported = True
        _startup_trace("self_check_da3_api_import_complete")
        _startup_trace(
            "self_check_da3_runtime_closure_complete",
            module_count=len(da3_runtime_modules_imported),
        )

        _startup_trace("self_check_da3_geometry_import_start")
        import torch

        geometry_module: Any = importlib.import_module("depth_anything_3.utils.geometry")
        affine_inverse = geometry_module.affine_inverse
        _startup_trace("self_check_da3_geometry_import_complete")
        _startup_trace("self_check_da3_geometry_probe_start")
        transform = torch.eye(4, dtype=torch.float32)
        transform[:3, 3] = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float32)
        actual_inverse = affine_inverse(transform)
        expected_inverse = torch.linalg.inv(transform)
        if not torch.allclose(actual_inverse, expected_inverse, atol=1e-6, rtol=1e-6):
            raise RuntimeError(
                "DA3 affine_inverse frozen-runtime probe disagrees with torch.linalg.inv"
            )
        _startup_trace("self_check_da3_geometry_probe_complete")
        da3_geometry_imported = True
        da3_affine_inverse_probe = "PASS"
        torch_version = torch.__version__

    report = {
        "schema_version": 4,
        "status": "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK",
        "rasterio_version": rasterio.__version__,
        "gdal_version": rasterio.__gdal_version__,
        "pyproj_version": pyproj.__version__,
        "torch_version": torch_version,
        "rasterio_serde_imported": rasterio_serde.__name__ == "rasterio.serde",
        "epsg_roundtrip": 32643,
        "da3_probe_required": require_da3,
        "da3_api_imported": da3_api_imported,
        "da3_runtime_modules_required": list(_DA3_PRODUCTION_RUNTIME_MODULES),
        "da3_runtime_modules_imported": da3_runtime_modules_imported,
        "da3_geometry_imported": da3_geometry_imported,
        "da3_affine_inverse_probe": da3_affine_inverse_probe,
        "da3_runtime_execution_gate": "release_train_5_full_acceptance",
        "network_used": False,
        "model_loaded": False,
        "model_weights_loaded": False,
    }
    _startup_trace("self_check_complete", status=report["status"])
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the packaged local core without requiring a user-visible terminal."""
    _startup_trace("python_entry")
    parser = _parser()
    args = parser.parse_args(argv)
    _startup_trace(
        "arguments_parsed",
        self_check=bool(args.self_check),
        self_check_da3=bool(args.self_check_da3),
    )

    if args.self_check or args.self_check_da3:
        report = packaged_geospatial_self_check(require_da3=bool(args.self_check_da3))
        print(json.dumps(report, sort_keys=True), flush=True)
        return

    if args.port is None:
        parser.error("--port is required unless --self-check is used")

    validate_launch_environment(host=args.host, port=args.port)
    _startup_trace("launch_environment_validated", port=args.port)

    if os.environ.get("DEPTHWIZARD_OFFLINE_CORE") == "1":
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("PROJ_NETWORK", "OFF")
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
