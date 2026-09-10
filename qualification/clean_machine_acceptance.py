"""Fail-closed clean-machine qualification for frozen release eb4aca3.

This harness exists only on the qualification branch. It consumes reports produced by an exact
release checkout, relaunches that packaged application, and proves the persisted metric project,
mesh, previews, validation, and export reopen through the packaged sidecar after restart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import plistlib
import signal
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, cast

TARGET_HEAD = "eb4aca36b05745931b7ea4dd4797569e176d98f9"


class QualificationFailure(RuntimeError):
    """Raised when a clean-machine claim lacks evidence."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise QualificationFailure(message)


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(payload, dict), f"JSON root is not an object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(base: str, route: str, *, token: str, timeout: float = 30) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    operation = urllib.request.Request(
        f"{base}{route}", headers={"x-depthwizard-token": token}, method="GET"
    )
    try:
        with opener.open(operation, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read()


def request_json(base: str, route: str, *, token: str) -> dict[str, Any]:
    status, body = request(base, route, token=token)
    require(status == 200, f"reopen request failed: HTTP {status}, route={route}")
    payload = json.loads(body)
    require(isinstance(payload, dict), f"reopen response is not an object: {route}")
    return payload


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise QualificationFailure(f"app exited before control was ready: {process.returncode}")
        time.sleep(0.1)
    raise QualificationFailure(f"timed out waiting for packaged app: {path}")


def wait_for_exit(pid: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not pid_alive(pid):
            return
        time.sleep(0.1)
    raise QualificationFailure(f"owned sidecar PID {pid} survived app exit")


def app_executable(bundle: Path) -> Path:
    info_path = bundle / "Contents" / "Info.plist"
    require(info_path.is_file(), "packaged app is missing Info.plist")
    with info_path.open("rb") as handle:
        name = plistlib.load(handle).get("CFBundleExecutable")
    require(isinstance(name, str) and bool(name), "Info.plist has no executable name")
    executable = bundle / "Contents" / "MacOS" / str(name)
    require(executable.is_file() and os.access(executable, os.X_OK), "app executable is missing")
    return executable


def query(project_dir: Path, **extra: str | int) -> str:
    return urllib.parse.urlencode({"project_dir": str(project_dir), **extra})


def reopen_project(root: Path, rt5: dict[str, Any], report_path: Path) -> dict[str, Any]:
    executable = app_executable(Path(str(rt5["application_bundle"])))
    project_dir = root / "artifacts/acceptance/release-train-5-full/project"
    require(project_dir.is_dir(), "persisted RT5 project is missing")
    runtime_dir = root / "artifacts/acceptance/clean-machine-reopen"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    control = runtime_dir / "control.json"
    exit_signal = runtime_dir / "exit.signal"
    boot_path = runtime_dir / "boot.json"
    for path in (control, exit_signal, boot_path):
        path.unlink(missing_ok=True)

    environment = os.environ.copy()
    environment.update(
        {
            "DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT": str(boot_path),
            "DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH": str(control),
            "DEPTHWIZARD_ACCEPTANCE_EXIT_SIGNAL": str(exit_signal),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )
    process: subprocess.Popen[bytes] | None = None
    sidecar_pid: int | None = None
    started = time.monotonic()
    try:
        with (runtime_dir / "app.stdout.log").open("wb") as stdout, (
            runtime_dir / "app.stderr.log"
        ).open("wb") as stderr:
            process = subprocess.Popen(
                [str(executable)], cwd=executable.parent, stdout=stdout, stderr=stderr, env=environment
            )
            wait_for_file(control, process, 120)
            control_payload = read_json(control)
            raw_base = control_payload.get("apiBase")
            raw_token = control_payload.get("sessionToken")
            raw_sidecar_pid = control_payload.get("sidecarPid")
            require(
                isinstance(raw_base, str) and raw_base.startswith("http://127.0.0.1:"),
                "relaunch exposed a non-loopback endpoint",
            )
            require(
                isinstance(raw_token, str) and len(raw_token) == 64,
                "relaunch token is invalid",
            )
            require(
                isinstance(raw_sidecar_pid, int)
                and raw_sidecar_pid > 0
                and pid_alive(raw_sidecar_pid),
                "relaunch did not produce a live Rust-owned sidecar",
            )
            base = cast(str, raw_base)
            token = cast(str, raw_token)
            sidecar_pid = cast(int, raw_sidecar_pid)
            control.unlink()
            boot = read_json(boot_path)
            require(boot.get("status") == "PASS_TAURI_SIDECAR_BOOT", "relaunch boot failed")
            require(boot.get("buildGitSha") == TARGET_HEAD, "relaunch app SHA is wrong")
            require(boot.get("offlineCore") is True, "relaunch app is not offline-first")
            require(boot.get("strictPythonEgressGuard") is True, "egress guard is absent")

            project_query = query(project_dir)
            manifest = request_json(base, f"/v1/projects/manifest?{project_query}", token=token)
            require(manifest.get("status") == "complete", "reopened project is not complete")
            artifacts = manifest.get("artifacts")
            require(isinstance(artifacts, dict) and "dsm" in artifacts, "reopened DSM is missing")
            validation = request_json(base, f"/v1/projects/validation?{project_query}", token=token)
            raw_elevation = validation.get("elevation")
            require(
                isinstance(raw_elevation, dict),
                "reopened validation has no elevation metrics",
            )
            elevation = cast(dict[str, Any], raw_elevation)
            for metric in ("rmse_m", "mae_m", "pearson_r"):
                value = elevation.get(metric)
                require(
                    isinstance(value, (int, float)) and math.isfinite(float(value)),
                    f"invalid reopened metric: {metric}",
                )

            mesh = request_json(base, f"/v1/projects/mesh?{project_query}", token=token)
            raw_lods = mesh.get("lods")
            require(
                isinstance(raw_lods, list) and len(raw_lods) >= 2,
                "reopened mesh has no LOD chain",
            )
            lods = cast(list[Any], raw_lods)
            require(mesh.get("surface_product") == "dsm", "reopened mesh is not a DSM")
            require(mesh.get("vertical_units") == "m", "reopened mesh is not metric")
            lod_status, lod_body = request(
                base, f"/v1/projects/mesh/lod/0?{project_query}", token=token, timeout=60
            )
            require(lod_status == 200 and lod_body.startswith(b"glTF"), "reopened GLB is invalid")
            lod_sha = hashlib.sha256(lod_body).hexdigest()
            expected_lod_sha = str(lods[0].get("sha256")) if isinstance(lods[0], dict) else ""
            require(lod_sha == expected_lod_sha, "reopened GLB identity does not match")

            preview_bytes: dict[str, int] = {}
            for layer in ("optical", "dsm"):
                preview_query = query(project_dir, layer=layer, max_side=256)
                status, body = request(
                    base, f"/v1/projects/preview?{preview_query}", token=token, timeout=60
                )
                require(
                    status == 200 and body.startswith(b"\x89PNG\r\n\x1a\n"),
                    f"reopened {layer} preview failed",
                )
                preview_bytes[layer] = len(body)

            exported = request_json(base, f"/v1/projects/export?{project_query}", token=token)
            export_path = Path(str(exported.get("bundle_path", "")))
            export_sha = exported.get("bundle_sha256")
            require(export_path.is_file(), "reopened export is missing")
            require(
                isinstance(export_sha, str) and sha256(export_path) == export_sha,
                "reopened export identity failed",
            )
            exit_signal.write_text("exit\n", encoding="utf-8")
            process.wait(timeout=30)
            require(process.returncode == 0, "reopened app did not exit cleanly")
            wait_for_exit(sidecar_pid, 10)

        report = {
            "schema_version": 1,
            "status": "PASS_CLEAN_MACHINE_RELAUNCH_REOPEN",
            "git_head": TARGET_HEAD,
            "application_sha256": sha256(executable),
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "boot": {
                "status": boot.get("status"),
                "build_git_sha": boot.get("buildGitSha"),
                "offline_core": boot.get("offlineCore"),
                "strict_egress_guard": boot.get("strictPythonEgressGuard"),
            },
            "project": {
                "project_id": manifest.get("project_id"),
                "status": manifest.get("status"),
                "metric_validation": {
                    key: elevation[key] for key in ("rmse_m", "mae_m", "pearson_r")
                },
                "mesh_lod_count": len(lods),
                "mesh_lod_0_sha256": lod_sha,
                "mesh_lod_0_bytes": len(lod_body),
                "preview_png_bytes": preview_bytes,
                "export_sha256": export_sha,
                "export_bytes": export_path.stat().st_size,
            },
            "lifecycle": {
                "app_exit_code": process.returncode,
                "sidecar_terminated_with_app": not pid_alive(sidecar_pid),
            },
        }
        write_json(report_path, report)
        return report
    finally:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reopen-report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    require(git_head == TARGET_HEAD, f"wrong release checkout: {git_head}")
    require(platform.system() == "Darwin", "qualification did not run on macOS")
    require(platform.machine() == "arm64", "qualification did not run on Apple silicon")
    require(os.environ.get("CI") == "true", "qualification did not run in CI")
    require(os.environ.get("GITHUB_ACTIONS") == "true", "qualification is not GitHub-hosted")

    evidence_root = root / "artifacts/acceptance"
    full_path = evidence_root / "release-train-5-full/release-train-5-full-acceptance.json"
    app_path = evidence_root / (
        "release-train-5-standalone/release-train-5-app-bundle-acceptance.json"
    )
    sidecar_path = evidence_root / (
        "release-train-5-sidecar/release-train-5-sidecar-acceptance.json"
    )
    full, app, sidecar = map(read_json, (full_path, app_path, sidecar_path))
    require(
        full.get("status") == "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE",
        "RT5 full acceptance failed",
    )
    require(full.get("git_head") == TARGET_HEAD, "RT5 acceptance has wrong SHA")
    require(app.get("status") == "PASS_RT5_REAL_TAURI_BUNDLE_LIFECYCLE", "app lifecycle failed")
    require(app.get("source_git_sha") == TARGET_HEAD, "app lifecycle has wrong SHA")
    require(
        sidecar.get("status") == "PASS_RT5_PACKAGED_SIDECAR_SECURITY_SMOKE",
        "sidecar smoke failed",
    )
    require(full.get("clean_application_launch") is True, "clean launch not proven")
    require(full.get("user_visible_terminal_required") is False, "app requires a terminal")
    require(full.get("offline_first_reconstruction") is True, "offline reconstruction failed")
    require(full.get("bundled_model_payload_verified") is True, "bundled model is unverified")
    require(
        full.get("runtime_manifest_status") == "QUALIFIED_DEPTHWIZARD_CORE_RUNTIME",
        "runtime manifest failed",
    )
    raw_lifecycle = full.get("lifecycle")
    require(
        isinstance(raw_lifecycle, dict)
        and raw_lifecycle.get("sidecar_terminated_with_app") is True,
        "sidecar ownership failed",
    )
    raw_offline_project = full.get("offline_da3_project")
    require(
        isinstance(raw_offline_project, dict)
        and raw_offline_project.get("terminal_status") == "complete",
        "end-to-end reconstruction failed",
    )
    raw_mesh = full.get("mesh")
    require(
        isinstance(raw_mesh, dict) and int(raw_mesh.get("lod_count", 0)) >= 2,
        "terrain mesh failed",
    )
    mesh = cast(dict[str, Any], raw_mesh)
    require(
        mesh.get("surface_product") == "dsm" and mesh.get("vertical_units") == "m",
        "terrain is not a metric DSM",
    )
    exported = full.get("export")
    require(
        isinstance(exported, dict)
        and exported.get("zip_integrity") == "PASS"
        and exported.get("mesh_included") is True
        and exported.get("validation_included") is True,
        "export is incomplete",
    )

    reopen = reopen_project(root, full, args.reopen_report.resolve())
    application_sha = full.get("application_sha256")
    require(isinstance(application_sha, str) and len(application_sha) == 64, "app SHA is missing")
    require(reopen.get("application_sha256") == application_sha, "relaunch used another app")
    run_url = (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.environ.get('GITHUB_REPOSITORY', 'amogh-hub/depthwizard')}/actions/runs/"
        f"{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    report = {
        "schema_version": 1,
        "status": "PASS_CLEAN_MACHINE_STANDALONE",
        "git_head": TARGET_HEAD,
        "application_sha256": application_sha,
        "packaged_app_launch": True,
        "no_user_visible_terminal": True,
        "owned_sidecar_boot": True,
        "offline_first_reconstruction": True,
        "bundled_model_payload_verified": True,
        "end_to_end_reconstruction": True,
        "metric_calibration": True,
        "terrain_3d": True,
        "export_and_reopen": True,
        "runner": {
            "provider": "GitHub-hosted fresh virtual machine",
            "image": os.environ.get("ImageOS"),
            "image_version": os.environ.get("ImageVersion"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
            "run_url": run_url,
        },
        "evidence": [
            {"name": "RT5 full packaged acceptance", "sha256": sha256(full_path)},
            {"name": "real Tauri lifecycle", "sha256": sha256(app_path)},
            {"name": "packaged sidecar security", "sha256": sha256(sidecar_path)},
            {"name": "relaunch/reopen", "sha256": sha256(args.reopen_report.resolve())},
            {"name": "GitHub Actions run", "url": run_url},
        ],
        "qualification_boundary": (
            "Fresh GitHub-hosted macos-15 ARM64 VM. It built exact release eb4aca3, ran packaged "
            "offline DA3 process-to-DSM, validation, mesh, and export, then relaunched the same app "
            "and reopened the project, validation, metric GLB, previews, and export."
        ),
    }
    write_json(args.output.resolve(), report)
    print("DepthWizard clean-machine standalone qualification: PASS")


if __name__ == "__main__":
    main()
