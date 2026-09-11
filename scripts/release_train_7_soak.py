from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts.release_train_5_app_bundle_smoke import (
    _macos_bundle,
    _pid_alive,
    _sha256,
    _wait_for_pid_exit,
)
from scripts.release_train_5_full_acceptance import _json_request, _validate_control

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "artifacts" / "acceptance" / "release-train-7-soak"
QUALIFYING_DURATION_SECONDS = 2 * 60 * 60
DEFAULT_INTERVAL_SECONDS = 30.0


class SoakFailure(RuntimeError):
    """A packaged-app stability-soak contract violation."""


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return payload


def _wait_for_file(path: Path, process: subprocess.Popen[bytes], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.is_file():
            return
        if process.poll() is not None:
            raise SoakFailure(
                "DepthWizard desktop exited before packaged soak control became available: "
                f"returncode={process.returncode}"
            )
        time.sleep(0.1)
    raise SoakFailure(f"timed out waiting for packaged soak control: {path}")


def _rss_kib(pid: int) -> int | None:
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    if not text:
        return None
    try:
        value = int(text.splitlines()[-1].strip())
    except ValueError:
        return None
    return value if value >= 0 else None


def _write_inspection_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y, x = np.mgrid[:64, :64]
    rgb = np.stack(
        [
            (x * 3 + y) % 255,
            (y * 4 + 32) % 255,
            ((x + y) * 2 + 64) % 255,
        ],
        axis=0,
    ).astype(np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=64,
        width=64,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000.0, 1400000.0, 1.0, 1.0),
    ) as dst:
        dst.write(rgb)


def qualification_status(elapsed_seconds: float) -> str:
    if elapsed_seconds >= QUALIFYING_DURATION_SECONDS:
        return "PASS_TWO_HOUR_PACKAGED_SOAK"
    return "NON_QUALIFYING_SHORT_SOAK"


def _sample_memory(samples: list[dict[str, object]], key: str) -> dict[str, int | None]:
    values: list[int] = []
    for sample in samples:
        value = sample.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            values.append(value)
    if not values:
        return {"start_kib": None, "end_kib": None, "peak_kib": None}
    return {
        "start_kib": values[0],
        "end_kib": values[-1],
        "peak_kib": max(values),
    }


def _sample_float(sample: dict[str, object], key: str) -> float:
    value = sample.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SoakFailure(f"packaged soak sample field {key!r} is not numeric")
    return float(value)


def _resolved_output_dir(output_dir: Path) -> Path:
    """Return a stable absolute directory for paths consumed by the app subprocess."""
    return output_dir.resolve(strict=False)


def run_soak(
    *,
    duration_seconds: float,
    interval_seconds: float,
    output_dir: Path,
) -> dict[str, Any]:
    if platform.system() != "Darwin":
        raise SoakFailure("the final packaged stability soak must run on the finale macOS host")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")

    git_head = _git_head()
    # The desktop process starts with the executable's directory as its working directory. Resolve
    # operator-supplied relative output paths before exporting control/report locations so the
    # child never interprets them relative to the application bundle.
    output_dir = _resolved_output_dir(output_dir)
    bundle, executable, sidecar, runtime_manifest = _macos_bundle()
    output_dir.mkdir(parents=True, exist_ok=True)
    control_path = output_dir / ".soak-control.json"
    exit_signal = output_dir / ".soak-exit"
    boot_report_path = output_dir / "tauri-boot-report.json"
    report_path = output_dir / "software_stability_report.json"
    app_stdout = output_dir / "depthwizard-app.stdout.log"
    app_stderr = output_dir / "depthwizard-app.stderr.log"
    fixture = output_dir / "inputs" / "inspection-fixture.tif"

    for path in (control_path, exit_signal, boot_report_path):
        path.unlink(missing_ok=True)
    _write_inspection_fixture(fixture)

    env = os.environ.copy()
    env.update(
        {
            "DEPTHWIZARD_ACCEPTANCE_BOOT_REPORT": str(boot_report_path),
            "DEPTHWIZARD_ACCEPTANCE_CONTROL_PATH": str(control_path),
            "DEPTHWIZARD_ACCEPTANCE_EXIT_SIGNAL": str(exit_signal),
            "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        }
    )

    process: subprocess.Popen[bytes] | None = None
    sidecar_pid: int | None = None
    samples: list[dict[str, object]] = []
    process_started_monotonic = time.monotonic()
    process_started_at_utc = datetime.now(UTC).isoformat()
    soak_started_monotonic: float | None = None
    soak_started_at_utc: str | None = None
    monitored_seconds = 0.0
    try:
        with app_stdout.open("wb") as stdout_handle, app_stderr.open("wb") as stderr_handle:
            process = subprocess.Popen(
                [str(executable)],
                cwd=executable.parent,
                stdout=stdout_handle,
                stderr=stderr_handle,
                env=env,
            )
            _wait_for_file(control_path, process, timeout_s=95.0)
            api_base, token, sidecar_pid = _validate_control(_read_json(control_path))
            control_path.unlink()
            if control_path.exists():
                raise SoakFailure("ephemeral soak control file could not be deleted")

            boot = _read_json(boot_report_path)
            if boot.get("status") != "PASS_TAURI_SIDECAR_BOOT":
                raise SoakFailure("packaged app did not record identity-bound sidecar boot")
            if boot.get("buildGitSha") != git_head:
                raise SoakFailure(
                    "packaged app was not built from the checked-out Git head: "
                    f"app={boot.get('buildGitSha')!r}, checkout={git_head}"
                )

            soak_started_monotonic = time.monotonic()
            soak_started_at_utc = datetime.now(UTC).isoformat()
            deadline = soak_started_monotonic + duration_seconds
            iteration = 0
            while True:
                now = time.monotonic()
                if now >= deadline:
                    break
                if process.poll() is not None:
                    raise SoakFailure(
                        f"DepthWizard desktop exited during soak: returncode={process.returncode}"
                    )
                if sidecar_pid is None or not _pid_alive(sidecar_pid):
                    raise SoakFailure("DepthWizard scientific sidecar exited during soak")

                iteration_started = time.monotonic()
                health_status, raw_health = _json_request(api_base, "/health", timeout_s=5.0)
                if (
                    health_status != 200
                    or not isinstance(raw_health, dict)
                    or raw_health.get("status") != "ok"
                ):
                    raise SoakFailure(
                        f"packaged health failed during soak: HTTP {health_status}, "
                        f"payload={raw_health!r}"
                    )

                inspect_status, raw_inspect = _json_request(
                    api_base,
                    "/v1/inspect",
                    token=token,
                    payload={"path": str(fixture)},
                    timeout_s=10.0,
                )
                if inspect_status != 200 or not isinstance(raw_inspect, dict):
                    raise SoakFailure(
                        "authenticated raster inspection failed during soak: "
                        f"HTTP {inspect_status}, payload={raw_inspect!r}"
                    )

                iteration += 1
                samples.append(
                    {
                        "iteration": iteration,
                        "elapsed_seconds": time.monotonic() - soak_started_monotonic,
                        "request_cycle_seconds": time.monotonic() - iteration_started,
                        "desktop_rss_kib": _rss_kib(process.pid),
                        "sidecar_rss_kib": _rss_kib(sidecar_pid),
                    }
                )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(interval_seconds, remaining))

            monitored_seconds = time.monotonic() - soak_started_monotonic
            exit_signal.write_text("exit\n", encoding="utf-8")
            try:
                process.wait(timeout=20.0)
            except subprocess.TimeoutExpired as exc:
                raise SoakFailure(
                    "DepthWizard desktop did not exit after soak exit signal"
                ) from exc
            if process.returncode != 0:
                raise SoakFailure(
                    f"DepthWizard desktop exited non-zero after soak: {process.returncode}"
                )
            assert sidecar_pid is not None
            _wait_for_pid_exit(sidecar_pid, timeout_s=10.0)
    except Exception:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5.0)
        raise
    finally:
        control_path.unlink(missing_ok=True)
        exit_signal.unlink(missing_ok=True)

    process_elapsed_seconds = time.monotonic() - process_started_monotonic
    if soak_started_monotonic is None or soak_started_at_utc is None:
        raise SoakFailure("packaged soak never reached the monitored healthy state")
    if not samples:
        raise SoakFailure("packaged soak completed without any liveness samples")
    cycle_latencies = [_sample_float(sample, "request_cycle_seconds") for sample in samples]
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": qualification_status(monitored_seconds),
        "claim_boundary": (
            "This soak proves packaged desktop/owned-sidecar liveness, authenticated raster "
            "inspection, process ownership and clean shutdown for the measured duration. It does "
            "not replace GPU/FPS or visual operator acceptance."
        ),
        "git_head": git_head,
        "process_started_at_utc": process_started_at_utc,
        "soak_started_at_utc": soak_started_at_utc,
        "startup_seconds": soak_started_monotonic - process_started_monotonic,
        "monitored_seconds": monitored_seconds,
        "process_elapsed_seconds": process_elapsed_seconds,
        "required_qualifying_seconds": QUALIFYING_DURATION_SECONDS,
        "interval_seconds": interval_seconds,
        "iterations": len(samples),
        "bundle": str(bundle),
        "desktop_executable": str(executable),
        "desktop_executable_sha256": _sha256(executable),
        "sidecar_executable": str(sidecar),
        "sidecar_executable_sha256": _sha256(sidecar),
        "runtime_manifest_status": runtime_manifest.get("status"),
        "boot_identity_bound": True,
        "request_cycle_seconds": {
            "mean": float(np.mean(cycle_latencies)),
            "max": max(cycle_latencies),
        },
        "desktop_memory": _sample_memory(samples, "desktop_rss_kib"),
        "sidecar_memory": _sample_memory(samples, "sidecar_rss_kib"),
        "samples": samples,
        "artifacts": {
            "boot_report": str(boot_report_path),
            "stdout_log": str(app_stdout),
            "stderr_log": str(app_stderr),
        },
    }
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the DepthWizard packaged desktop stability soak on macOS."
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=float(QUALIFYING_DURATION_SECONDS),
        help="Soak duration. Only >=7200 s earns PASS_TWO_HOUR_PACKAGED_SOAK.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Interval between authenticated liveness/inspection cycles.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        report = run_soak(
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / "software_stability_report.json"
        failure = {
            "schema_version": 1,
            "status": "FAIL_PACKAGED_SOAK",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "claim_boundary": (
                "The packaged stability soak failed before earning the required two-hour PASS. "
                "This record must not be presented as successful stability evidence."
            ),
        }
        temporary = failure_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(failure_path)
        raise

    print(
        json.dumps(
            {
                "status": report["status"],
                "monitored_seconds": report["monitored_seconds"],
                "iterations": report["iterations"],
                "desktop_memory": report["desktop_memory"],
                "sidecar_memory": report["sidecar_memory"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
