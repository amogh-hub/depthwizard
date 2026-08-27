from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.release_train_2_validation_smoke import (
    DATA_DIR,
    DSM_URL,
    _build_coarse_calibration_dem,
    _download,
    _prepare_source_and_calibration_source,
)
from scripts.release_train_5_app_bundle_smoke import (
    _macos_bundle,
    _pid_alive,
    _sha256,
    _wait_for_pid_exit,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "acceptance" / "release-train-5-full"
PROJECT_DIR = OUT / "project"
FAILURE_PROJECT_DIR = OUT / "failure-project"
CONTROL_PATH = OUT / ".acceptance-control.json"
EXIT_SIGNAL = OUT / ".acceptance-exit"
BOOT_REPORT = OUT / "tauri-boot-report.json"
REPORT_PATH = OUT / "release-train-5-full-acceptance.json"
APP_STDOUT = OUT / "depthwizard-app.stdout.log"
APP_STDERR = OUT / "depthwizard-app.stderr.log"
DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _json_request(
    base: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> tuple[int, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
        method = "POST"
    if token is not None:
        headers["x-depthwizard-token"] = token
    request = urllib.request.Request(
        f"{base}{path}", data=data, headers=headers, method=method
    )
    try:
        with DIRECT_OPENER.open(request, timeout=timeout_s) as response:
            raw = response.read()
            status = int(response.status)
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, raw.decode("utf-8", errors="replace")


def _wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise RuntimeError(
                f"DepthWizard app exited before acceptance control became available: {process.returncode}"
            )
        time.sleep(0.1)
    raise RuntimeError(f"timed out waiting for acceptance control: {path}")


def _poll_job(
    base: str,
    token: str,
    job_id: str,
    *,
    expected_terminal: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        status, payload = _json_request(base, f"/v1/jobs/{job_id}", token=token)
        if status != 200 or not isinstance(payload, dict):
            raise RuntimeError(f"job polling failed: HTTP {status}, payload={payload!r}")
        last = payload
        state = str(payload.get("status"))
        if state in {"complete", "failed", "waiting_for_calibration"}:
            if state != expected_terminal:
                raise RuntimeError(
                    f"job {job_id} terminated as {state!r}; expected {expected_terminal!r}; "
                    f"error={payload.get('error')!r}"
                )
            return payload
        time.sleep(0.5)
    raise RuntimeError(f"job {job_id} did not reach a terminal state; last={last!r}")


def _write_corrupt_raster(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-raster\n")


def _write_failure_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.full((1, 32, 32), 100, dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=32,
        height=32,
        count=1,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def _prepare_assets() -> tuple[Path, Path, Path, dict[str, bool]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dop = DATA_DIR / "urban_residential_DOP.tif"
    xdsm = DATA_DIR / "urban_residential_xDSM.tif"
    reference = DATA_DIR / "urban_residential_DSM.tif"
    cached = {
        "source": dop.is_file() and dop.stat().st_size > 0,
        "calibration_source": xdsm.is_file() and xdsm.stat().st_size > 0,
        "reference": reference.is_file() and reference.stat().st_size > 0,
    }
    source, calibration_source = _prepare_source_and_calibration_source()
    _download(DSM_URL, reference)
    dem = _build_coarse_calibration_dem(calibration_source)
    return source, dem, reference, cached


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _tail(path: Path, max_chars: int = 12000) -> str:
    if not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _assert_project_outputs(project_dir: Path) -> dict[str, str]:
    manifest_path = project_dir / "project-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("packaged workflow did not persist project-manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise RuntimeError(f"project manifest is not complete: {manifest.get('status')!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("project manifest artifacts must be an object")
    required = ("rdsm", "dsm", "slope")
    evidence: dict[str, str] = {}
    for name in required:
        item = artifacts.get(name)
        if not isinstance(item, dict):
            raise RuntimeError(f"project manifest is missing required artifact: {name}")
        raw_path = item.get("path")
        raw_sha = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_sha, str):
            raise RuntimeError(f"project artifact {name} has incomplete identity evidence")
        artifact_path = Path(raw_path)
        if not artifact_path.is_file():
            raise RuntimeError(f"project artifact {name} does not exist: {artifact_path}")
        if _sha256(artifact_path) != raw_sha:
            raise RuntimeError(f"project artifact {name} SHA-256 does not match manifest")
        evidence[name] = raw_sha
    return evidence


def main() -> None:
    if platform.system() != "Darwin":
        raise RuntimeError("RT5 full standalone acceptance must run on the finale macOS host")

    bundle, executable, sidecar, runtime_manifest = _macos_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (CONTROL_PATH, EXIT_SIGNAL, BOOT_REPORT, REPORT_PATH, APP_STDOUT, APP_STDERR):
        if path.exists():
            path.unlink()
    for directory in (PROJECT_DIR, FAILURE_PROJECT_DIR):
        if directory.exists():
            shutil.rmtree(directory)

    source, dem, reference, cached_before = _prepare_assets()
    corrupt = OUT / "inputs" / "corrupt.tif"
    failure_source = OUT / "inputs" / "single-band-runtime-failure.tif"
    _write_corrupt_raster(corrupt)
    _write_failure_fixture(failure_source)

    env = os.environ.copy()
    env.update(
        {
            "DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT": str(BOOT_REPORT),
            "DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH": str(CONTROL_PATH),
            "DEPTHWIZARD_ACCEPTANCE_EXIT_SIGNAL": str(EXIT_SIGNAL),
            "DEPTHWIZARD_ORTHOLOC_METRIC_AFFINE": "1",
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )

    started_at = time.monotonic()
    sidecar_pid: int | None = None
    control_deleted = False
    process: subprocess.Popen[bytes] | None = None
    try:
        with APP_STDOUT.open("wb") as stdout_handle, APP_STDERR.open("wb") as stderr_handle:
            process = subprocess.Popen(
                [str(executable)],
                cwd=executable.parent,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            _wait_for_file(CONTROL_PATH, process, timeout_s=95.0)
            control = _read_json(CONTROL_PATH)
            api_base = control.get("apiBase")
            token = control.get("sessionToken")
            raw_pid = control.get("sidecarPid")
            if not isinstance(api_base, str) or not api_base.startswith("http://127.0.0.1:"):
                raise RuntimeError("acceptance control exposed a non-loopback API base")
            if not isinstance(token, str) or len(token) != 64 or any(
                character not in "0123456789abcdefABCDEF" for character in token
            ):
                raise RuntimeError("acceptance control did not provide a 256-bit hexadecimal token")
            if not isinstance(raw_pid, int) or raw_pid <= 0 or not _pid_alive(raw_pid):
                raise RuntimeError("acceptance control sidecar PID is not live")
            sidecar_pid = raw_pid
            CONTROL_PATH.unlink()
            control_deleted = not CONTROL_PATH.exists()
            if not control_deleted:
                raise RuntimeError("ephemeral acceptance control file could not be deleted")

            boot = _read_json(BOOT_REPORT)
            if boot.get("status") != "PASS_TAURI_SIDECAR_BOOT":
                raise RuntimeError("clean app launch did not record passing Tauri sidecar boot")
            if boot.get("offlineCore") is not True or boot.get("strictPythonEgressGuard") is not True:
                raise RuntimeError("clean app launch did not enforce packaged offline policy")
            if boot.get("sessionTokenExported") is not False:
                raise RuntimeError("boot evidence unexpectedly exported the session token")
            if boot.get("ephemeralAcceptanceControlEnabled") is not True:
                raise RuntimeError("full acceptance control channel was not explicitly marked test-only")

            health_status, health = _json_request(api_base, "/health")
            if health_status != 200 or not isinstance(health, dict) or health.get("status") != "ok":
                raise RuntimeError(f"packaged app health failed: HTTP {health_status}, {health!r}")

            missing_status, _ = _json_request(
                api_base, "/v1/inspect", payload={"path": str(source)}
            )
            wrong_status, _ = _json_request(
                api_base,
                "/v1/inspect",
                token="wrong-token",
                payload={"path": str(source)},
            )
            if missing_status != 401 or wrong_status != 401:
                raise RuntimeError(
                    f"packaged app session guard failed: missing={missing_status}, wrong={wrong_status}"
                )

            corrupt_status, corrupt_payload = _json_request(
                api_base,
                "/v1/inspect",
                token=token,
                payload={"path": str(corrupt)},
            )
            if corrupt_status != 422:
                raise RuntimeError(
                    f"malformed raster was not rejected recoverably: HTTP {corrupt_status}, "
                    f"payload={corrupt_payload!r}"
                )

            failure_submit_status, failure_submit = _json_request(
                api_base,
                "/v1/projects",
                token=token,
                payload={
                    "source": str(failure_source),
                    "output_dir": str(FAILURE_PROJECT_DIR),
                    "requested_output": "rdsm",
                    "tile_size": 256,
                    "overlap": 32,
                },
            )
            if failure_submit_status != 202 or not isinstance(failure_submit, dict):
                raise RuntimeError(
                    f"runtime-failure fixture was not queued: HTTP {failure_submit_status}, "
                    f"payload={failure_submit!r}"
                )
            failure_job_id = failure_submit.get("job_id")
            if not isinstance(failure_job_id, str):
                raise RuntimeError("runtime-failure fixture did not return a job id")
            failure_terminal = _poll_job(
                api_base,
                token,
                failure_job_id,
                expected_terminal="failed",
                timeout_s=180.0,
            )
            recovery_health_status, recovery_health = _json_request(api_base, "/health")
            if (
                recovery_health_status != 200
                or not isinstance(recovery_health, dict)
                or recovery_health.get("status") != "ok"
            ):
                raise RuntimeError("scientific core did not remain healthy after a runtime job failure")

            submit_status, submit = _json_request(
                api_base,
                "/v1/projects",
                token=token,
                payload={
                    "source": str(source),
                    "output_dir": str(PROJECT_DIR),
                    "dem_path": str(dem),
                    "requested_output": "dsm",
                    "tile_size": 768,
                    "overlap": 128,
                    "harmonize_overlaps": True,
                },
            )
            if submit_status != 202 or not isinstance(submit, dict):
                raise RuntimeError(
                    f"fresh packaged project was not queued: HTTP {submit_status}, payload={submit!r}"
                )
            job_id = submit.get("job_id")
            if not isinstance(job_id, str):
                raise RuntimeError("fresh packaged project did not return a job id")
            job_terminal = _poll_job(
                api_base,
                token,
                job_id,
                expected_terminal="complete",
                timeout_s=1200.0,
            )
            artifact_hashes = _assert_project_outputs(PROJECT_DIR)

            validation_status, validation = _json_request(
                api_base,
                "/v1/projects/validate",
                token=token,
                payload={
                    "project_dir": str(PROJECT_DIR),
                    "reference_path": str(reference),
                    "reference_label": (
                        "TUM OrthoLoC urban_residential DSM / RT5 downstream integration reference"
                    ),
                    "min_valid_pixels": 1000,
                },
                timeout_s=120.0,
            )
            if validation_status != 200 or not isinstance(validation, dict):
                raise RuntimeError(
                    f"packaged validation failed: HTTP {validation_status}, payload={validation!r}"
                )

            mesh_status, mesh = _json_request(
                api_base,
                "/v1/projects/mesh",
                token=token,
                payload={
                    "project_dir": str(PROJECT_DIR),
                    "max_finest_samples": 256,
                    "lod_levels": 3,
                },
                timeout_s=120.0,
            )
            if mesh_status != 200 or not isinstance(mesh, dict):
                raise RuntimeError(
                    f"packaged mesh build failed: HTTP {mesh_status}, payload={mesh!r}"
                )
            lods = mesh.get("lods")
            if not isinstance(lods, list) or len(lods) < 2:
                raise RuntimeError("packaged mesh build did not produce the required LOD chain")

            export_status, export = _json_request(
                api_base,
                "/v1/projects/export",
                token=token,
                payload={
                    "project_dir": str(PROJECT_DIR),
                    "include_source": False,
                    "include_mesh": True,
                    "include_validation": True,
                },
                timeout_s=120.0,
            )
            if export_status != 200 or not isinstance(export, dict):
                raise RuntimeError(
                    f"packaged export failed: HTTP {export_status}, payload={export!r}"
                )
            bundle_path_raw = export.get("bundle_path")
            bundle_sha = export.get("bundle_sha256")
            if not isinstance(bundle_path_raw, str) or not isinstance(bundle_sha, str):
                raise RuntimeError("packaged export response is missing bundle identity")
            export_bundle = Path(bundle_path_raw)
            if not export_bundle.is_file() or _sha256(export_bundle) != bundle_sha:
                raise RuntimeError("packaged export bundle identity verification failed")
            with zipfile.ZipFile(export_bundle, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise RuntimeError(f"packaged export ZIP integrity failed at {bad_member}")
                archive_names = set(archive.namelist())
            if "project-manifest.json" not in archive_names:
                raise RuntimeError("packaged export is missing project-manifest.json")

            final_health_status, final_health = _json_request(api_base, "/health")
            if (
                final_health_status != 200
                or not isinstance(final_health, dict)
                or final_health.get("status") != "ok"
            ):
                raise RuntimeError("scientific core was not healthy after process→validate→mesh→export")

            EXIT_SIGNAL.write_text("exit\n", encoding="utf-8")
            process.wait(timeout=30.0)
            if process.returncode != 0:
                raise RuntimeError(f"DepthWizard app exited with code {process.returncode}")
            _wait_for_pid_exit(sidecar_pid, timeout_s=10.0)

        elapsed = time.monotonic() - started_at
        validation_elevation = validation.get("elevation")
        report = {
            "schema_version": 1,
            "status": "PASS_RT5_FULL_STANDALONE_ENGINEERING_ACCEPTANCE",
            "git_head": _git_head(),
            "platform": platform.platform(),
            "application_bundle": str(bundle.resolve()),
            "application_sha256": _sha256(executable),
            "packaged_sidecar": str(sidecar.resolve()),
            "packaged_sidecar_sha256": _sha256(sidecar),
            "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256"),
            "runtime_manifest_status": runtime_manifest.get("status"),
            "clean_application_launch": True,
            "user_visible_terminal_required": False,
            "offline_after_model_install": True,
            "strict_python_non_loopback_egress_guard": True,
            "session_guard": {
                "missing_token_http_status": missing_status,
                "wrong_token_http_status": wrong_status,
                "token_recorded_in_evidence": False,
                "ephemeral_acceptance_control_used": True,
                "ephemeral_control_deleted_before_scientific_requests": control_deleted,
                "ephemeral_control_removed_on_app_exit": not CONTROL_PATH.exists(),
            },
            "recovery": {
                "malformed_raster_http_status": corrupt_status,
                "runtime_failure_job_id": failure_job_id,
                "runtime_failure_terminal_status": failure_terminal.get("status"),
                "runtime_failure_error_present": bool(failure_terminal.get("error")),
                "health_after_runtime_failure": recovery_health,
                "subsequent_valid_job_completed": True,
            },
            "offline_da3_project": {
                "job_id": job_id,
                "terminal_status": job_terminal.get("status"),
                "source": str(source.resolve()),
                "source_sha256": _sha256(source),
                "calibration_dem": str(dem.resolve()),
                "calibration_dem_sha256": _sha256(dem),
                "reference": str(reference.resolve()),
                "reference_sha256": _sha256(reference),
                "asset_cache_state_before_acceptance_prep": cached_before,
                "project_dir": str(PROJECT_DIR.resolve()),
                "artifact_sha256": artifact_hashes,
            },
            "validation": validation,
            "validation_elevation": validation_elevation,
            "mesh": {
                "lod_count": len(lods),
                "surface_product": mesh.get("surface_product"),
                "horizontal_units": mesh.get("horizontal_units"),
                "vertical_units": mesh.get("vertical_units"),
            },
            "export": {
                "bundle_path": str(export_bundle.resolve()),
                "bundle_sha256": bundle_sha,
                "bundle_bytes": export.get("bundle_bytes"),
                "zip_integrity": "PASS",
                "source_included": export.get("include_source"),
                "mesh_included": export.get("include_mesh"),
                "validation_included": export.get("include_validation"),
            },
            "lifecycle": {
                "sidecar_pid": sidecar_pid,
                "sidecar_terminated_with_app": not _pid_alive(sidecar_pid),
                "app_exit_code": process.returncode if process is not None else None,
                "elapsed_seconds": round(elapsed, 3),
            },
            "consumed_benchmark_rerun": False,
            "model_promotion_claim": False,
            "scientific_boundary": (
                "RT5 standalone engineering acceptance only. The OrthoLoC source/calibration/reference "
                "are reused as an integration fixture and are not independent scientific evidence. "
                "This run proves packaged offline DA3 execution, failure recovery, validation, mesh, "
                "export, session security and clean Tauri-owned lifecycle. RT6 owns final scientific "
                "evidence; RT7 owns finale-Mac FPS, two-hour soak and clean-machine qualification."
            ),
        }
        REPORT_PATH.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        print("DepthWizard Release Train 5 full standalone engineering acceptance: PASS")
        print("Clean packaged DepthWizard.app launch: PASS")
        print("Offline DA3 reconstruction from installed model assets: PASS")
        print("Malformed-input rejection without service loss: PASS")
        print("Runtime job failure → service recovery → subsequent valid job: PASS")
        print("Fresh-image process → downstream validation: PASS")
        print("Packaged 3D mesh + LOD generation: PASS")
        print("Packaged project export + ZIP integrity: PASS")
        print("Session token persisted in final evidence: NO")
        print("Sidecar terminated with desktop: PASS")
        print("Consumed benchmark/model-promotion protocol rerun: NO")
        print("RT7 FPS/soak/clean-machine evidence consumed here: NO")
        print(f"Acceptance report: {REPORT_PATH}")
    except Exception as exc:
        if process is not None and process.poll() is None:
            try:
                EXIT_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
                EXIT_SIGNAL.write_text("exit\n", encoding="utf-8")
                process.wait(timeout=15.0)
            except Exception:
                process.kill()
                process.wait(timeout=5.0)
        if CONTROL_PATH.exists():
            CONTROL_PATH.unlink()
        if sidecar_pid is not None and _pid_alive(sidecar_pid):
            time.sleep(1.0)
        raise RuntimeError(
            f"{exc}\n\nDepthWizard app stdout tail:\n{_tail(APP_STDOUT)}\n\n"
            f"DepthWizard app stderr tail:\n{_tail(APP_STDERR)}"
        ) from exc


if __name__ == "__main__":
    main()
