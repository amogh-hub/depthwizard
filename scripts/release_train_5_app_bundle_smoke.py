from __future__ import annotations

import hashlib
import json
import os
import platform
import plistlib
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAURI_TARGET = ROOT / "apps" / "desktop" / "src-tauri" / "target" / "release" / "bundle"
OUT = ROOT / "artifacts" / "acceptance" / "release-train-5-standalone"
SIDECAR_RESOURCE = Path("Contents/Resources/depthwizard-core-runtime/depthwizard-core")
RUNTIME_MANIFEST_RESOURCE = Path(
    "Contents/Resources/depthwizard-core-runtime/runtime-manifest.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=2)
            raise RuntimeError(
                "DepthWizard app exited before standalone boot acceptance was recorded\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        time.sleep(0.1)
    raise RuntimeError("DepthWizard app did not record standalone boot acceptance before timeout")


def _health(api_base: str) -> dict[str, object]:
    request = urllib.request.Request(f"{api_base}/health", method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            if response.status != 200:
                raise RuntimeError(f"standalone health returned HTTP {response.status}")
            payload = json.loads(response.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"unable to reach packaged standalone core: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise RuntimeError("standalone health payload is invalid")
    return payload


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_pid_exit(pid: int, timeout_s: float = 8.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    raise RuntimeError(f"packaged sidecar process {pid} survived after desktop exit")


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _macos_bundle() -> tuple[Path, Path, Path, dict[str, object]]:
    bundles = sorted((TAURI_TARGET / "macos").glob("*.app"))
    if len(bundles) != 1:
        raise RuntimeError(
            "expected exactly one macOS application bundle under "
            f"{TAURI_TARGET / 'macos'}, found {len(bundles)}"
        )
    bundle = bundles[0]
    info_path = bundle / "Contents" / "Info.plist"
    if not info_path.is_file():
        raise RuntimeError("DepthWizard application bundle is missing Info.plist")
    with info_path.open("rb") as handle:
        info = plistlib.load(handle)
    executable_name = info.get("CFBundleExecutable")
    if not isinstance(executable_name, str) or not executable_name:
        raise RuntimeError("DepthWizard Info.plist does not declare CFBundleExecutable")
    executable = bundle / "Contents" / "MacOS" / executable_name
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise RuntimeError(f"DepthWizard application executable is missing: {executable}")

    sidecar = bundle / SIDECAR_RESOURCE
    if not sidecar.is_file() or not os.access(sidecar, os.X_OK):
        raise RuntimeError(f"qualified packaged scientific runtime is missing: {sidecar}")
    internal = sidecar.parent / "_internal"
    if not internal.is_dir():
        raise RuntimeError(f"PyInstaller onedir support tree is missing: {internal}")

    runtime_manifest_path = bundle / RUNTIME_MANIFEST_RESOURCE
    if not runtime_manifest_path.is_file():
        raise RuntimeError(f"packaged runtime manifest is missing: {runtime_manifest_path}")
    runtime_manifest = _read_json(runtime_manifest_path)
    if runtime_manifest.get("status") != "QUALIFIED_DEPTHWIZARD_CORE_RUNTIME":
        raise RuntimeError("packaged runtime manifest is not qualified")
    if runtime_manifest.get("packaging_mode") != "pyinstaller_onedir":
        raise RuntimeError("packaged runtime is not the approved PyInstaller onedir layout")
    if runtime_manifest.get("executable_sha256") != _sha256(sidecar):
        raise RuntimeError("packaged sidecar executable does not match its qualification manifest")
    self_check = runtime_manifest.get("frozen_self_check")
    if not isinstance(self_check, dict):
        raise TypeError("packaged runtime manifest is missing frozen self-check evidence")
    if self_check.get("status") != "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK":
        raise RuntimeError("packaged runtime frozen geospatial self-check is not passing")
    return bundle, executable, sidecar, runtime_manifest


def main() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("RT5 real application-bundle acceptance must run on the finale macOS host")

    bundle, executable, sidecar, runtime_manifest = _macos_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    boot_report = OUT / "tauri-boot-report.json"
    if boot_report.exists():
        boot_report.unlink()

    env = os.environ.copy()
    env["DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT"] = str(boot_report)
    env["DEPTHWIZARD_ACCEPTANCE_AUTO_EXIT_MS"] = "3500"

    process = subprocess.Popen(
        [str(executable)],
        cwd=executable.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    started_at = time.monotonic()
    try:
        _wait_for_file(boot_report, process, timeout_s=35.0)
        payload = _read_json(boot_report)
        if payload.get("status") != "PASS_TAURI_SIDECAR_BOOT":
            raise RuntimeError("Tauri boot report did not record a passing sidecar startup")
        if payload.get("sessionTokenBits") != 256 or payload.get("sessionTokenExported") is not False:
            raise RuntimeError("Tauri boot report violates the session-token secrecy contract")
        if payload.get("offlineCore") is not True:
            raise RuntimeError("Tauri boot report did not enforce offline-after-install mode")
        if payload.get("strictPythonEgressGuard") is not True:
            raise RuntimeError("Tauri boot report did not enforce the strict Python egress guard")
        api_base = payload.get("apiBase")
        sidecar_pid = payload.get("sidecarPid")
        if not isinstance(api_base, str) or not api_base.startswith("http://127.0.0.1:"):
            raise RuntimeError("Tauri boot report contains a non-loopback API endpoint")
        if not isinstance(sidecar_pid, int) or sidecar_pid <= 0 or not _pid_alive(sidecar_pid):
            raise RuntimeError("Tauri-owned sidecar process is not alive after application startup")
        health = _health(api_base)

        process.wait(timeout=12.0)
        if process.returncode != 0:
            stdout, stderr = process.communicate(timeout=2)
            raise RuntimeError(
                f"DepthWizard acceptance auto-exit returned {process.returncode}\n"
                f"stdout={stdout.decode(errors='replace')}\n"
                f"stderr={stderr.decode(errors='replace')}"
            )
        _wait_for_pid_exit(sidecar_pid)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    report = {
        "schema_version": 2,
        "status": "PASS_RT5_REAL_TAURI_BUNDLE_LIFECYCLE",
        "platform": platform.platform(),
        "bundle": str(bundle.resolve()),
        "application_executable": str(executable.resolve()),
        "application_sha256": _sha256(executable),
        "packaged_sidecar": str(sidecar.resolve()),
        "packaged_sidecar_sha256": _sha256(sidecar),
        "runtime_manifest": runtime_manifest,
        "packaging_mode": "pyinstaller_onedir_tauri_resource",
        "startup_elapsed_seconds": round(time.monotonic() - started_at, 3),
        "sidecar_pid": sidecar_pid,
        "loopback_api_base": api_base,
        "health": health,
        "session_token_bits": 256,
        "session_token_exported": False,
        "offline_after_model_install": True,
        "strict_python_egress_guard": True,
        "sidecar_terminated_with_app": True,
        "user_visible_terminal_required": False,
        "consumed_benchmark_rerun": False,
        "model_promotion_claim": False,
        "scientific_boundary": (
            "Real packaged Tauri lifecycle/security acceptance only. The runtime must already carry "
            "its passing frozen geospatial self-check. This does not run DA3 inference and is not "
            "clean-machine, DSM-accuracy, generalization, FPS, soak, or model-promotion evidence."
        ),
    }
    report_path = OUT / "release-train-5-app-bundle-acceptance.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("DepthWizard RT5 real Tauri application-bundle lifecycle: PASS")
    print(f"Application bundle: {bundle}")
    print(f"Packaged scientific runtime: {sidecar}")
    print("Packaging mode: qualified PyInstaller onedir Tauri resource")
    print("Frozen geospatial self-check carried into bundle: PASS")
    print("Loopback scientific core readiness: PASS")
    print("256-bit session token exported to evidence: NO")
    print("Offline-after-install mode: YES")
    print("Strict Python non-loopback egress guard: YES")
    print("Sidecar terminated with desktop: PASS")
    print("User-visible terminal required by packaged app: NO")
    print("Consumed benchmark/model-promotion protocol rerun: NO")
    print(f"Acceptance report: {report_path}")


if __name__ == "__main__":
    main()
