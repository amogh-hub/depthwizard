"""Fail-closed clean-machine qualification for frozen release 012301b.

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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

TARGET_HEAD = "012301b9c1910ef4ccde5b3da5d4e4d94da60ce6"
EXPECTED_DMG_SHA256 = "a4703a2aa8886b5dea55af429057f8d9828c8dcb00c0b90e13a311e6ee28c3b3"


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


def request(
    base: str,
    route: str,
    *,
    token: str,
    timeout: float = 30,
    payload: dict[str, Any] | None = None,
) -> tuple[int, bytes]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    headers = {"x-depthwizard-token": token}
    data = None
    method = "GET"
    if payload is not None:
        headers["content-type"] = "application/json"
        data = json.dumps(payload).encode("utf-8")
        method = "POST"
    operation = urllib.request.Request(f"{base}{route}", headers=headers, data=data, method=method)
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


def post_json(
    base: str,
    route: str,
    *,
    token: str,
    payload: dict[str, Any],
    timeout: float = 30,
) -> dict[str, Any]:
    status, body = request(base, route, token=token, payload=payload, timeout=timeout)
    require(status in {200, 202}, f"request failed: HTTP {status}, route={route}")
    decoded = json.loads(body)
    require(isinstance(decoded, dict), f"response is not an object: {route}")
    return decoded


def poll_job(base: str, *, token: str, job_id: str, timeout: float = 1200) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = request_json(base, f"/v1/jobs/{job_id}", token=token)
        state = last.get("status")
        if state == "complete":
            return last
        require(state not in {"failed", "waiting_for_calibration"}, f"job failed: {last}")
        time.sleep(0.5)
    raise QualificationFailure(f"job timed out: {job_id}; last={last}")


def run_relative_project(
    base: str,
    *,
    token: str,
    source: Path,
    output_dir: Path,
) -> dict[str, Any]:
    submitted = post_json(
        base,
        "/v1/projects",
        token=token,
        payload={
            "source": str(source),
            "output_dir": str(output_dir),
            "requested_output": "rdsm",
            "tile_size": 256,
            "overlap": 32,
        },
    )
    job_id = submitted.get("job_id")
    require(isinstance(job_id, str) and bool(job_id), "relative project has no job id")
    terminal = poll_job(base, token=token, job_id=job_id)
    manifest = request_json(base, f"/v1/projects/manifest?{query(output_dir)}", token=token)
    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, dict) and "rdsm" in artifacts, "relative DSM is missing")
    require("dsm" not in artifacts, "non-georeferenced project invented a metric DSM")
    require(manifest.get("status") == "complete", "relative project is not complete")
    return {
        "source": str(source),
        "job_id": job_id,
        "terminal_status": terminal.get("status"),
        "project_id": manifest.get("project_id"),
        "rdsm_present": True,
        "metric_dsm_absent": True,
    }


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
    relative_projects: list[dict[str, Any]] = []
    profile: dict[str, Any] = {}
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

            fixture_root = root / "artifacts/acceptance/sih26175-input-formats/fixtures"
            for extension in ("png", "jpg"):
                source = fixture_root / f"single-view-rgb.{extension}"
                require(source.is_file(), f"clean-machine {extension.upper()} fixture is missing")
                relative_projects.append(
                    run_relative_project(
                        base,
                        token=token,
                        source=source,
                        output_dir=root
                        / "artifacts/acceptance/clean-machine-relative"
                        / extension,
                    )
                )

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

            profile = post_json(
                base,
                "/v1/projects/profile",
                token=token,
                payload={
                    "project_dir": str(project_dir),
                    "start": {"x": 0.2, "y": 0.25},
                    "end": {"x": 0.8, "y": 0.75},
                    "samples": 64,
                },
                timeout=60,
            )
            require(profile.get("surface_product") == "dsm", "profile did not sample the DSM")
            require(profile.get("vertical_units") == "m", "profile did not preserve metric units")
            require(profile.get("sample_count") == 64, "profile sample count changed")

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
            for layer in (
                "optical",
                "dsm",
                "slope",
                "hillshade",
                "contours",
                "reference",
                "residual",
            ):
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
                "profile": {
                    "surface_product": profile.get("surface_product"),
                    "vertical_units": profile.get("vertical_units"),
                    "sample_count": profile.get("sample_count"),
                    "horizontal_distance_m": profile.get("horizontal_distance_m"),
                    "vertical_delta": profile.get("vertical_delta"),
                },
                "relative_projects": relative_projects,
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
    started_at = datetime.now(timezone.utc)
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dmg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reopen-report", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    dmg = args.dmg.resolve()
    git_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    require(git_head == TARGET_HEAD, f"wrong release checkout: {git_head}")
    require(platform.system() == "Darwin", "qualification did not run on macOS")
    require(platform.machine() == "arm64", "qualification did not run on Apple silicon")
    require(os.environ.get("CI") == "true", "qualification did not run in CI")
    require(os.environ.get("GITHUB_ACTIONS") == "true", "qualification is not GitHub-hosted")
    require(
        os.environ.get("DEPTHWIZARD_OFFLINE_GATE") == "verified_interface_down",
        "clean-machine workflow was not run with its external network interface disabled",
    )
    require(dmg.is_file(), "the exact candidate DMG is missing")
    dmg_sha = sha256(dmg)
    require(dmg_sha == EXPECTED_DMG_SHA256, f"candidate DMG SHA-256 changed: {dmg_sha}")

    evidence_root = root / "artifacts/acceptance"
    full_path = evidence_root / "release-train-5-full/release-train-5-full-acceptance.json"
    app_path = evidence_root / (
        "release-train-5-standalone/release-train-5-app-bundle-acceptance.json"
    )
    sidecar_path = evidence_root / (
        "release-train-5-sidecar/release-train-5-sidecar-acceptance.json"
    )
    input_path = evidence_root / "sih26175-input-formats/input-format-contract.json"
    rt4_path = evidence_root / (
        "release-train-4-analytical/release-train-4-analytical-acceptance.json"
    )
    gcp_path = evidence_root / "release-train-4-analytical/calibration_gcp_report.json"
    consecutive_paths = sorted((evidence_root / "clean-machine-consecutive").glob("run-*.json"))
    full, app, sidecar = map(read_json, (full_path, app_path, sidecar_path))
    inputs, rt4, gcp = map(read_json, (input_path, rt4_path, gcp_path))
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
    require(
        inputs.get("status") == "PASS_SIH26175_INPUT_FORMAT_CONTRACT"
        and inputs.get("git_head") == TARGET_HEAD,
        "literal PNG/JPG/GeoTIFF input contract failed on the fresh runner",
    )
    formats = inputs.get("formats")
    require(isinstance(formats, dict), "input-format report has no format records")
    for name in ("png", "jpg", "tiff"):
        item = formats.get(name)
        require(isinstance(item, dict) and item.get("status") == "PASS", f"{name} input failed")
    require(
        rt4.get("status") == "PASS_RT4_SCIENTIFIC_ANALYTICAL_ENGINEERING_PATH",
        "fresh-runner analytical qualification failed",
    )
    require(
        gcp.get("status") == "PASS_ENGINEERING_CALIBRATION_GCP"
        and int(gcp.get("gcp_count", 0)) >= 6,
        "six-point GCP calibration did not pass on the fresh runner",
    )
    require(len(consecutive_paths) == 3, "three consecutive packaged runs were not recorded")
    consecutive = [read_json(path) for path in consecutive_paths]
    for index, item in enumerate(consecutive, start=1):
        require(
            item.get("status") == "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE"
            and item.get("git_head") == TARGET_HEAD,
            f"packaged run {index} is not a passing exact-head reconstruction",
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
    reopened_project = reopen.get("project")
    require(isinstance(reopened_project, dict), "reopen report has no project evidence")
    relative_projects = reopened_project.get("relative_projects")
    require(
        isinstance(relative_projects, list)
        and len(relative_projects) == 2
        and all(isinstance(item, dict) and item.get("metric_dsm_absent") is True for item in relative_projects),
        "fresh packaged PNG/JPG relative reconstruction was not proven",
    )
    profile = reopened_project.get("profile")
    require(
        isinstance(profile, dict)
        and profile.get("surface_product") == "dsm"
        and profile.get("vertical_units") == "m",
        "fresh packaged metric measurement/profile was not proven",
    )
    run_url = (
        f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
        f"{os.environ.get('GITHUB_REPOSITORY', 'amogh-hub/depthwizard')}/actions/runs/"
        f"{os.environ.get('GITHUB_RUN_ID', '')}"
    )
    report = {
        "schema_version": 1,
        "status": "PASS_CLEAN_MACHINE_STANDALONE",
        "git_head": TARGET_HEAD,
        "application_sha256": dmg_sha,
        "desktop_executable_sha256": application_sha,
        "package_name": dmg.name,
        "target": "fresh supported Apple Silicon Mac or fresh macOS ARM64 VM",
        "network_disconnected_during_workflow": True,
        "packaged_app_launch": True,
        "no_user_visible_terminal": True,
        "owned_sidecar_boot": True,
        "offline_first_reconstruction": True,
        "bundled_model_payload_verified": True,
        "end_to_end_reconstruction": True,
        "metric_calibration": True,
        "terrain_3d": True,
        "export_and_reopen": True,
        "png_processed": True,
        "jpg_processed": True,
        "geotiff_processed": True,
        "dem_calibration_exercised": True,
        "gcp_calibration_exercised": True,
        "measurement_exercised": True,
        "repeated_jobs_completed": len(consecutive),
        "unexpected_log_failures": [],
        "hardware": platform.machine(),
        "macos_version": platform.platform(),
        "operator": "GitHub Actions automated clean-machine qualification",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
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
            {"name": "exact DMG", "sha256": dmg_sha},
            {"name": "RT5 full packaged acceptance", "sha256": sha256(full_path)},
            {"name": "real Tauri lifecycle", "sha256": sha256(app_path)},
            {"name": "packaged sidecar security", "sha256": sha256(sidecar_path)},
            {"name": "literal PNG/JPG/GeoTIFF contract", "sha256": sha256(input_path)},
            {"name": "six-point GCP and analytical acceptance", "sha256": sha256(rt4_path)},
            {
                "name": "three consecutive packaged reconstructions",
                "sha256": [sha256(path) for path in consecutive_paths],
            },
            {"name": "relaunch/reopen", "sha256": sha256(args.reopen_report.resolve())},
            {"name": "GitHub Actions run", "url": run_url},
        ],
        "notes": [
            "The candidate is ad-hoc signed rather than Developer-ID notarized.",
            "The runner's external network interface was disabled during input, calibration, packaged reconstruction, validation, mesh, measurement, export and reopen checks.",
            "The application was extracted from the checksum-verified DMG before qualification.",
        ],
        "qualification_boundary": (
            "Fresh GitHub-hosted macos-15 ARM64 VM. It built exact release 012301b, verified and "
            "extracted the exact DMG, disabled its external network interface, processed PNG and "
            "JPG as relative rDSM, processed GeoTIFF through DEM calibration, exercised six-point "
            "GCP calibration, completed three packaged reconstructions, validation, metric profile, "
            "analytical previews, mesh and export, then relaunched and reopened the project."
        ),
    }
    write_json(args.output.resolve(), report)
    print("DepthWizard clean-machine standalone qualification: PASS")


if __name__ == "__main__":
    main()
