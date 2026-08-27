from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from secrets import token_hex

import numpy as np
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "acceptance" / "release-train-5-sidecar"
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _host_triple() -> str:
    result = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    verbose = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True).stdout
    for line in verbose.splitlines():
        if line.startswith("host: "):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("unable to resolve Rust host target triple")


def _default_binary() -> Path:
    extension = ".exe" if os.name == "nt" else ""
    return (
        ROOT
        / "apps"
        / "desktop"
        / "src-tauri"
        / "binaries"
        / f"depthwizard-core-{_host_triple()}{extension}"
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _request(
    url: str,
    *,
    token: str | None = None,
    payload: dict[str, object] | None = None,
) -> tuple[int, bytes]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
        method = "POST"
    if token is not None:
        headers["x-depthwizard-token"] = token
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with _DIRECT_OPENER.open(request, timeout=2.0) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def _wait_for_health(
    base: str,
    process: subprocess.Popen[bytes],
    timeout_s: float = 90.0,
) -> float:
    started = time.monotonic()
    deadline = started + timeout_s
    last_error: str | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            raise RuntimeError(
                "packaged sidecar exited before readiness\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        try:
            status, body = _request(f"{base}/health")
            if status == 200 and json.loads(body)["status"] == "ok":
                return time.monotonic() - started
        except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = repr(exc)
        time.sleep(0.1)
    raise RuntimeError(
        "packaged sidecar did not become healthy before timeout; "
        f"timeout_s={timeout_s:.1f}, process_alive={process.poll() is None}, "
        f"last_health_error={last_error}"
    )


def _stop_process(process: subprocess.Popen[bytes]) -> tuple[bytes, bytes]:
    if process.poll() is None:
        process.terminate()
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    return process.communicate(timeout=5)


def _write_rgb(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((3, 8, 8), dtype=np.uint8)
    data[0] = 32
    data[1] = 64
    data[2] = 96
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=None)
    args = parser.parse_args()
    binary = (args.binary or _default_binary()).resolve()
    if not binary.is_file():
        raise FileNotFoundError(
            f"packaged sidecar not found: {binary}. Run `make sidecar-build` first."
        )

    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "inputs" / "rgb.tif"
    _write_rgb(source)
    port = _free_port()
    token = token_hex(32)
    base = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env.update(
        {
            "DEPTHWIZARD_REQUIRE_SESSION_TOKEN": "1",
            "DEPTHWIZARD_SESSION_TOKEN": token,
            "DEPTHWIZARD_OFFLINE_CORE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    process = subprocess.Popen(
        [str(binary), "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    try:
        startup_elapsed = _wait_for_health(base, process)

        health_status, health_body = _request(f"{base}/health")
        unauth_status, _ = _request(f"{base}/v1/inspect", payload={"path": str(source)})
        wrong_status, _ = _request(
            f"{base}/v1/inspect",
            token="wrong-token",
            payload={"path": str(source)},
        )
        auth_status, auth_body = _request(
            f"{base}/v1/inspect",
            token=token,
            payload={"path": str(source)},
        )
        if health_status != 200:
            raise RuntimeError(f"health readiness failed with HTTP {health_status}")
        if unauth_status != 401 or wrong_status != 401:
            raise RuntimeError(
                f"session guard failed: missing={unauth_status}, wrong={wrong_status}"
            )
        if auth_status != 200:
            raise RuntimeError(f"authorized inspect failed with HTTP {auth_status}")
        inspect_payload = json.loads(auth_body)
        if inspect_payload.get("crs") != "EPSG:32643":
            raise RuntimeError("authorized inspect returned unexpected raster metadata")
        health_payload = json.loads(health_body)
    except Exception as exc:
        stdout, stderr = _stop_process(process)
        raise RuntimeError(
            f"{exc}\n"
            f"binary={binary}\n"
            f"binary_bytes={binary.stat().st_size}\n"
            f"stdout={stdout.decode(errors='replace')}\n"
            f"stderr={stderr.decode(errors='replace')}"
        ) from exc
    else:
        _stop_process(process)

    report = {
        "schema_version": 3,
        "status": "PASS_RT5_PACKAGED_SIDECAR_SECURITY_SMOKE",
        "binary": str(binary),
        "binary_bytes": binary.stat().st_size,
        "binary_sha256": _sha256(binary),
        "sidecar_port": port,
        "startup_elapsed_seconds": round(startup_elapsed, 3),
        "session_token_bits": len(token) * 4,
        "health": health_payload,
        "missing_token_http_status": unauth_status,
        "wrong_token_http_status": wrong_status,
        "authorized_inspect_http_status": auth_status,
        "authorized_inspect_crs": inspect_payload["crs"],
        "offline_environment_forced": True,
        "strict_python_egress_guard": True,
        "consumed_benchmark_rerun": False,
        "model_promotion_claim": False,
        "scientific_boundary": (
            "Packaged sidecar lifecycle/security acceptance only. Offline mode installs a Python "
            "INET connect guard that permits loopback only. This smoke does not run DA3 inference "
            "and is not scientific accuracy, model-promotion, clean-machine, FPS, or soak evidence."
        ),
    }
    report_path = OUT / "release-train-5-sidecar-acceptance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DepthWizard RT5 packaged sidecar security path: PASS")
    print(f"Packaged core cold-start readiness: {startup_elapsed:.3f} s")
    print(f"Packaged core size: {binary.stat().st_size / (1024 * 1024):.2f} MiB")
    print("Loopback health readiness: PASS")
    print("Missing token rejected: PASS")
    print("Wrong token rejected: PASS")
    print("Authorized raster inspection: PASS")
    print("Offline environment forced: YES")
    print("Strict Python non-loopback egress guard: YES")
    print("Consumed benchmark/model-promotion protocol rerun: NO")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
