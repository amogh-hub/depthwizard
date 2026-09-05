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


class AcceptanceFailure(RuntimeError):
    """A fail-closed RT5 acceptance contract violation."""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _require_dict(payload: object, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be a JSON object")
    return payload


def _require_str(payload: dict[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"{context}.{key} must be a non-empty string")
    return value


def _require_int(payload: dict[str, Any], key: str, *, context: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer")
    return value


def _json_request(
    base: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 5.0,
) -> tuple[int, object]:
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
            raise AcceptanceFailure(
                "DepthWizard app exited before acceptance control became available: "
                f"returncode={process.returncode}"
            )
        time.sleep(0.1)
    raise AcceptanceFailure(f"timed out waiting for acceptance control: {path}")


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
        status, raw_payload = _json_request(base, f"/v1/jobs/{job_id}", token=token)
        if status != 200:
            raise AcceptanceFailure(
                f"job polling failed: HTTP {status}, payload={raw_payload!r}"
            )
        payload = _require_dict(raw_payload, context="job status response")
        last = payload
        state = str(payload.get("status"))
        if state in {"complete", "failed", "waiting_for_calibration"}:
            if state != expected_terminal:
                raise AcceptanceFailure(
                    f"job {job_id} terminated as {state!r}; expected {expected_terminal!r}; "
                    f"error={payload.get('error')!r}"
                )
            return payload
        time.sleep(0.5)
    raise AcceptanceFailure(
        f"job {job_id} did not reach a terminal state before {timeout_s:.1f}s; last={last!r}"
    )


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
    """Prepare RT5 integration fixtures before the packaged core is launched offline."""
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
        raise AcceptanceFailure("packaged workflow did not persist project-manifest.json")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise AcceptanceFailure(
            f"project manifest is not complete: {manifest.get('status')!r}"
        )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise TypeError("project manifest artifacts must be an object")

    evidence: dict[str, str] = {}
    for name in ("rdsm", "dsm", "slope"):
        item = artifacts.get(name)
        if not isinstance(item, dict):
            raise TypeError(f"project manifest artifact {name} must be an object")
        raw_path = item.get("path")
        raw_sha = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(raw_sha, str):
            raise TypeError(f"project artifact {name} identity fields must be strings")
        artifact_path = Path(raw_path)
        if not artifact_path.is_file():
            raise AcceptanceFailure(
                f"project artifact {name} does not exist: {artifact_path}"
            )
        if _sha256(artifact_path) != raw_sha:
            raise AcceptanceFailure(
                f"project artifact {name} SHA-256 does not match manifest"
            )
        evidence[name] = raw_sha
    return evidence


def _cleanup_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        EXIT_SIGNAL.parent.mkdir(parents=True, exist_ok=True)
        EXIT_SIGNAL.write_text("exit\n", encoding="utf-8")
        process.wait(timeout=15.0)
    except (OSError, subprocess.SubprocessError):
        process.kill()
        process.wait(timeout=5.0)


def _validate_control(control: dict[str, Any]) -> tuple[str, str, int]:
    api_base = _require_str(control, "apiBase", context="acceptance control")
    token = _require_str(control, "sessionToken", context="acceptance control")
    sidecar_pid = _require_int(control, "sidecarPid", context="acceptance control")
    if not api_base.startswith("http://127.0.0.1:"):
        raise AcceptanceFailure("acceptance control exposed a non-loopback API base")
    if len(token) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in token
    ):
        raise AcceptanceFailure(
            "acceptance control did not provide a 256-bit hexadecimal token"
        )
    if sidecar_pid <= 0 or not _pid_alive(sidecar_pid):
        raise AcceptanceFailure("acceptance control sidecar PID is not live")
    return api_base, token, sidecar_pid


def _run_full_acceptance(
    executable: Path,
    *,
    source: Path,
    dem: Path,
    reference: Path,
) -> dict[str, Any]:
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

    corrupt = OUT / "inputs" / "corrupt.tif"
    failure_source = OUT / "inputs" / "single-band-runtime-failure.tif"
    _write_corrupt_raster(corrupt)
    _write_failure_fixture(failure_source)

    process: subprocess.Popen[bytes] | None = None
    sidecar_pid: int | None = None
    started_at = time.monotonic()
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
            api_base, token, sidecar_pid = _validate_control(_read_json(CONTROL_PATH))
            CONTROL_PATH.unlink()
            if CONTROL_PATH.exists():
                raise AcceptanceFailure(
                    "ephemeral acceptance control file could not be deleted"
                )

            boot = _read_json(BOOT_REPORT)
            if boot.get("status") != "PASS_TAURI_SIDECAR_BOOT":
                raise AcceptanceFailure(
                    "clean app launch did not record passing Tauri sidecar boot"
                )
            if (
                boot.get("offlineCore") is not True
                or boot.get("strictPythonEgressGuard") is not True
            ):
                raise AcceptanceFailure(
                    "clean app launch did not enforce packaged offline policy"
                )
            if boot.get("sessionTokenExported") is not False:
                raise AcceptanceFailure(
                    "boot evidence unexpectedly exported the session token"
                )
            if boot.get("ephemeralAcceptanceControlEnabled") is not True:
                raise AcceptanceFailure(
                    "full acceptance control channel was not explicitly marked test-only"
                )

            health_status, raw_health = _json_request(api_base, "/health")
            health = _require_dict(raw_health, context="packaged app health response")
            if health_status != 200 or health.get("status") != "ok":
                raise AcceptanceFailure(
                    f"packaged app health failed: HTTP {health_status}, {health!r}"
                )

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
                raise AcceptanceFailure(
                    "packaged app session guard failed: "
                    f"missing={missing_status}, wrong={wrong_status}"
                )

            corrupt_status, corrupt_payload = _json_request(
                api_base,
                "/v1/inspect",
                token=token,
                payload={"path": str(corrupt)},
            )
            if corrupt_status != 422:
                raise AcceptanceFailure(
                    f"malformed raster was not rejected recoverably: HTTP {corrupt_status}, "
                    f"payload={corrupt_payload!r}"
                )

            failure_submit_status, raw_failure_submit = _json_request(
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
            if failure_submit_status != 202:
                raise AcceptanceFailure(
                    "runtime-failure fixture was not queued: "
                    f"HTTP {failure_submit_status}, payload={raw_failure_submit!r}"
                )
            failure_submit = _require_dict(
                raw_failure_submit, context="runtime-failure project submission"
            )
            failure_job_id = _require_str(
                failure_submit, "job_id", context="runtime-failure project submission"
            )
            failure_terminal = _poll_job(
                api_base,
                token,
                failure_job_id,
                expected_terminal="failed",
                timeout_s=180.0,
            )

            recovery_health_status, raw_recovery_health = _json_request(
                api_base, "/health"
            )
            recovery_health = _require_dict(
                raw_recovery_health, context="post-failure health response"
            )
            if recovery_health_status != 200 or recovery_health.get("status") != "ok":
                raise AcceptanceFailure(
                    "scientific core did not remain healthy after a runtime job failure"
                )

            submit_status, raw_submit = _json_request(
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
            if submit_status != 202:
                raise AcceptanceFailure(
                    f"fresh packaged project was not queued: HTTP {submit_status}, "
                    f"payload={raw_submit!r}"
                )
            submit = _require_dict(raw_submit, context="fresh packaged project submission")
            job_id = _require_str(submit, "job_id", context="fresh packaged project submission")
            job_terminal = _poll_job(
                api_base,
                token,
                job_id,
                expected_terminal="complete",
                timeout_s=1200.0,
            )
            artifact_hashes = _assert_project_outputs(PROJECT_DIR)

            validation_status, raw_validation = _json_request(
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
            if validation_status != 200:
                raise AcceptanceFailure(
                    f"packaged validation failed: HTTP {validation_status}, "
                    f"payload={raw_validation!r}"
                )
            validation = _require_dict(raw_validation, context="packaged validation response")

            mesh_status, raw_mesh = _json_request(
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
            if mesh_status != 200:
                raise AcceptanceFailure(
                    f"packaged mesh build failed: HTTP {mesh_status}, payload={raw_mesh!r}"
                )
            mesh = _require_dict(raw_mesh, context="packaged mesh response")
            lods = mesh.get("lods")
            if not isinstance(lods, list):
                raise TypeError("packaged mesh response.lods must be a list")
            if len(lods) < 2:
                raise AcceptanceFailure(
                    "packaged mesh build did not produce the required LOD chain"
                )

            export_status, raw_export = _json_request(
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
            if export_status != 200:
                raise AcceptanceFailure(
                    f"packaged export failed: HTTP {export_status}, payload={raw_export!r}"
                )
            export = _require_dict(raw_export, context="packaged export response")
            bundle_path_raw = _require_str(
                export, "bundle_path", context="packaged export response"
            )
            bundle_sha = _require_str(
                export, "bundle_sha256", context="packaged export response"
            )
            export_bundle = Path(bundle_path_raw)
            if not export_bundle.is_file() or _sha256(export_bundle) != bundle_sha:
                raise AcceptanceFailure(
                    "packaged export bundle identity verification failed"
                )
            with zipfile.ZipFile(export_bundle, "r") as archive:
                bad_member = archive.testzip()
                if bad_member is not None:
                    raise AcceptanceFailure(
                        f"packaged export ZIP integrity failed at {bad_member}"
                    )
                archive_names = set(archive.namelist())
            if "project-manifest.json" not in archive_names:
                raise AcceptanceFailure(
                    "packaged export is missing project-manifest.json"
                )

            final_health_status, raw_final_health = _json_request(api_base, "/health")
            final_health = _require_dict(
                raw_final_health, context="final packaged health response"
            )
            if final_health_status != 200 or final_health.get("status") != "ok":
                raise AcceptanceFailure(
                    "scientific core was not healthy after process→validate→mesh→export"
                )

            EXIT_SIGNAL.write_text("exit\n", encoding="utf-8")
            process.wait(timeout=30.0)
            if process.returncode != 0:
                raise AcceptanceFailure(
                    f"DepthWizard app exited with code {process.returncode}"
                )
            _wait_for_pid_exit(sidecar_pid, timeout_s=10.0)

        elapsed = time.monotonic() - started_at
        if _pid_alive(sidecar_pid):
            raise AcceptanceFailure("Rust-owned scientific sidecar survived desktop exit")

        return {
            "elapsed_seconds": round(elapsed, 3),
            "sidecar_pid": sidecar_pid,
            "app_exit_code": process.returncode,
            "missing_status": missing_status,
            "wrong_status": wrong_status,
            "corrupt_status": corrupt_status,
            "failure_job_id": failure_job_id,
            "failure_terminal": failure_terminal,
            "recovery_health": recovery_health,
            "job_id": job_id,
            "job_terminal": job_terminal,
            "artifact_hashes": artifact_hashes,
            "validation": validation,
            "mesh": mesh,
            "lod_count": len(lods),
            "export": export,
            "export_bundle": export_bundle,
            "bundle_sha": bundle_sha,
            "final_health": final_health,
        }
    except (
        AcceptanceFailure,
        TypeError,
        ValueError,
        OSError,
        TimeoutError,
        subprocess.SubprocessError,
        urllib.error.URLError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        _cleanup_process(process)
        if CONTROL_PATH.exists():
            CONTROL_PATH.unlink()
        raise AcceptanceFailure(
            f"{exc}\n\nDepthWizard app stdout tail:\n{_tail(APP_STDOUT)}\n\n"
            f"DepthWizard app stderr tail:\n{_tail(APP_STDERR)}"
        ) from exc


def main() -> None:
    if platform.system() != "Darwin":
        raise AcceptanceFailure(
            "RT5 full standalone acceptance must run on the finale macOS host"
        )

    bundle, executable, sidecar, runtime_manifest = _macos_bundle()
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (
        CONTROL_PATH,
        EXIT_SIGNAL,
        BOOT_REPORT,
        REPORT_PATH,
        APP_STDOUT,
        APP_STDERR,
    ):
        if path.exists():
            path.unlink()
    for directory in (PROJECT_DIR, FAILURE_PROJECT_DIR):
        if directory.exists():
            shutil.rmtree(directory)

    source, dem, reference, cached_before = _prepare_assets()
    evidence = _run_full_acceptance(
        executable, source=source, dem=dem, reference=reference
    )

    failure_terminal = _require_dict(
        evidence["failure_terminal"], context="stored runtime-failure evidence"
    )
    job_terminal = _require_dict(
        evidence["job_terminal"], context="stored project evidence"
    )
    recovery_health = _require_dict(
        evidence["recovery_health"], context="stored recovery health evidence"
    )
    validation = _require_dict(evidence["validation"], context="stored validation evidence")
    mesh = _require_dict(evidence["mesh"], context="stored mesh evidence")
    export = _require_dict(evidence["export"], context="stored export evidence")
    artifact_hashes = _require_dict(
        evidence["artifact_hashes"], context="stored artifact identity evidence"
    )
    export_bundle = evidence["export_bundle"]
    if not isinstance(export_bundle, Path):
        raise TypeError("stored export bundle must be a Path")
    bundle_sha = evidence["bundle_sha"]
    if not isinstance(bundle_sha, str):
        raise TypeError("stored export bundle SHA-256 must be a string")
    sidecar_pid = evidence["sidecar_pid"]
    if not isinstance(sidecar_pid, int):
        raise TypeError("stored sidecar PID must be an integer")

    report = {
        "schema_version": 3,
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
        "offline_first_reconstruction": True,
        "bundled_model_payload_verified": True,
        "strict_python_non_loopback_egress_guard": True,
        "session_guard": {
            "missing_token_http_status": evidence["missing_status"],
            "wrong_token_http_status": evidence["wrong_status"],
            "token_recorded_in_evidence": False,
            "ephemeral_acceptance_control_used": True,
            "ephemeral_control_deleted_before_scientific_requests": True,
            "ephemeral_control_removed_on_app_exit": not CONTROL_PATH.exists(),
        },
        "recovery": {
            "malformed_raster_http_status": evidence["corrupt_status"],
            "runtime_failure_job_id": evidence["failure_job_id"],
            "runtime_failure_terminal_status": failure_terminal.get("status"),
            "runtime_failure_error_present": bool(failure_terminal.get("error")),
            "health_after_runtime_failure": recovery_health,
            "subsequent_valid_job_completed": True,
        },
        "offline_da3_project": {
            "job_id": evidence["job_id"],
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
        "validation_elevation": validation.get("elevation"),
        "mesh": {
            "lod_count": evidence["lod_count"],
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
            "app_exit_code": evidence["app_exit_code"],
            "elapsed_seconds": evidence["elapsed_seconds"],
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


if __name__ == "__main__":
    main()
