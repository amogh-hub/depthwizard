from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = ROOT / "apps" / "desktop" / "src-tauri"
RUNTIME_DIR = TAURI_DIR / "resources" / "depthwizard-core-runtime"
BUILD_ROOT = ROOT / "artifacts" / "standalone" / "pyinstaller"
ENTRY = ROOT / "scripts" / "depthwizard_sidecar_entry.py"
DA3_VENDOR = ROOT / ".vendor" / "depth-anything-3" / "src"
SELF_CHECK_TIMEOUT_SECONDS = 45.0


def _host_triple() -> str:
    direct = subprocess.run(
        ["rustc", "--print", "host-tuple"],
        check=False,
        capture_output=True,
        text=True,
    )
    if direct.returncode == 0 and direct.stdout.strip():
        return direct.stdout.strip()
    verbose = subprocess.run(
        ["rustc", "-vV"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in verbose.splitlines():
        if line.startswith("host: "):
            return line.split(":", 1)[1].strip()
    raise RuntimeError("unable to resolve Rust host target triple")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_identity(root: Path) -> tuple[str, int, int, int]:
    """Return deterministic SHA, regular-file count, symlink count and logical bytes."""
    digest = hashlib.sha256()
    file_count = 0
    symlink_count = 0
    logical_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            target = os.readlink(path).encode("utf-8")
            digest.update(b"L\0" + relative + b"\0" + str(mode).encode() + b"\0" + target + b"\n")
            symlink_count += 1
            continue
        if path.is_file():
            digest.update(b"F\0" + relative + b"\0" + str(mode).encode() + b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
                    logical_bytes += len(chunk)
            digest.update(b"\n")
            file_count += 1
    return digest.hexdigest(), file_count, symlink_count, logical_bytes


def _read_trace(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    events: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _qualify_frozen_runtime(executable: Path) -> tuple[dict[str, object], float, list[str]]:
    trace_path = BUILD_ROOT.parent / "frozen-self-check-startup-trace.jsonl"
    if trace_path.exists():
        trace_path.unlink()
    environment = os.environ.copy()
    environment.update(
        {
            "DEPTHWIZARD_STARTUP_TRACE": str(trace_path),
            "DEPTHWIZARD_OFFLINE_CORE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(executable), "--self-check"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=SELF_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        events = _read_trace(trace_path)
        phases = [str(event.get("phase", "unknown")) for event in events]
        location = phases[-1] if phases else "before Python entrypoint"
        raise RuntimeError(
            "frozen sidecar startup qualification timed out after "
            f"{SELF_CHECK_TIMEOUT_SECONDS:.0f}s; last_phase={location}. "
            "The build is rejected rather than publishing an unqualified runtime."
        ) from exc
    elapsed = time.monotonic() - started
    events = _read_trace(trace_path)
    phases = [str(event.get("phase", "unknown")) for event in events]
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen sidecar geospatial self-check failed\n"
            f"returncode={completed.returncode}\n"
            f"startup_phases={phases}\n"
            f"stdout={completed.stdout}\n"
            f"stderr={completed.stderr}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError("frozen sidecar self-check produced no machine-readable result")
    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "frozen sidecar self-check did not end with valid JSON\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("frozen sidecar self-check JSON root must be an object")
    if payload.get("status") != "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK":
        raise RuntimeError(f"frozen sidecar self-check returned non-passing status: {payload}")
    required_phases = {"python_entry", "self_check_import_complete", "self_check_complete"}
    if not required_phases.issubset(phases):
        raise RuntimeError(
            "frozen sidecar startup trace is incomplete; "
            f"required={sorted(required_phases)}, observed={phases}"
        )
    return payload, elapsed, phases


def main() -> None:
    """Build and qualify the no-Python-required scientific runtime staged for Tauri."""
    if not DA3_VENDOR.is_dir():
        raise RuntimeError(
            "pinned Depth Anything 3 source is missing; run `make da3-setup` before sidecar packaging"
        )
    if importlib.util.find_spec("PyInstaller") is None:
        raise RuntimeError(
            "PyInstaller is missing; install the standalone extra with "
            "`python -m pip install -e '.[standalone]'`"
        )

    triple = _host_triple()
    extension = ".exe" if os.name == "nt" else ""
    dist_dir = BUILD_ROOT / "dist"
    work_dir = BUILD_ROOT / "work"
    spec_dir = BUILD_ROOT / "spec"
    for path in (dist_dir, work_dir, spec_dir):
        path.mkdir(parents=True, exist_ok=True)

    # A scientific runtime of this size must not use PyInstaller onefile mode on macOS. Onefile
    # extracts its complete native runtime at every launch and obscures bootstrap failures. Onedir
    # keeps the frozen environment directly inspectable, starts deterministically, and is staged as
    # a Tauri resource tree. The resulting application remains a normal single .app for the user.
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--contents-directory",
        "_internal",
        "--name",
        "depthwizard-core",
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(work_dir),
        "--specpath",
        str(spec_dir),
        "--paths",
        str(ROOT / "src"),
        "--paths",
        str(DA3_VENDOR),
        "--hidden-import",
        "torch",
        "--hidden-import",
        "depth_anything_3.api",
        "--hidden-import",
        "rasterio.serde",
        "--collect-data",
        "rasterio",
        "--collect-binaries",
        "rasterio",
        "--collect-data",
        "pyproj",
        "--collect-binaries",
        "pyproj",
        "--collect-data",
        "depth_anything_3",
        str(ENTRY),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    built_runtime = dist_dir / "depthwizard-core"
    built_executable = built_runtime / f"depthwizard-core{extension}"
    if not built_executable.is_file():
        raise RuntimeError(
            f"PyInstaller did not produce the expected onedir executable: {built_executable}"
        )

    if RUNTIME_DIR.exists():
        shutil.rmtree(RUNTIME_DIR)
    RUNTIME_DIR.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(built_runtime, RUNTIME_DIR, symlinks=True)
    staged_executable = RUNTIME_DIR / f"depthwizard-core{extension}"
    if not staged_executable.is_file():
        raise RuntimeError(f"staged Tauri scientific runtime is missing: {staged_executable}")
    if os.name != "nt":
        staged_executable.chmod(staged_executable.stat().st_mode | 0o111)

    self_check, self_check_elapsed, startup_phases = _qualify_frozen_runtime(staged_executable)
    executable_sha = _sha256(staged_executable)
    runtime_manifest = {
        "schema_version": 1,
        "status": "QUALIFIED_DEPTHWIZARD_CORE_RUNTIME",
        "target_triple": triple,
        "packaging_mode": "pyinstaller_onedir",
        "executable": staged_executable.name,
        "executable_sha256": executable_sha,
        "frozen_self_check": self_check,
        "frozen_self_check_elapsed_seconds": round(self_check_elapsed, 3),
        "startup_phases": startup_phases,
        "model_loaded_during_packaging_check": False,
        "network_used_during_packaging_check": False,
    }
    runtime_manifest_path = RUNTIME_DIR / "runtime-manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_sha, file_count, symlink_count, logical_bytes = _tree_identity(RUNTIME_DIR)
    report = {
        "schema_version": 3,
        "status": "PASS_QUALIFIED_SIDECAR_BUILD",
        "target_triple": triple,
        "platform": platform.platform(),
        "python": sys.version,
        "packaging_mode": "pyinstaller_onedir",
        "tauri_runtime_dir": str(RUNTIME_DIR.resolve()),
        "binary": str(staged_executable.resolve()),
        "binary_bytes": staged_executable.stat().st_size,
        "binary_sha256": executable_sha,
        "runtime_tree_sha256": runtime_sha,
        "runtime_regular_files": file_count,
        "runtime_symlinks": symlink_count,
        "runtime_logical_bytes": logical_bytes,
        "da3_vendor_source": str(DA3_VENDOR.resolve()),
        "geospatial_packaging": {
            "rasterio_serde_hidden_import": True,
            "rasterio_data_collected": True,
            "rasterio_binaries_collected": True,
            "pyproj_data_collected": True,
            "pyproj_binaries_collected": True,
            "custom_proj_environment_override": False,
        },
        "frozen_self_check": self_check,
        "frozen_self_check_elapsed_seconds": round(self_check_elapsed, 3),
        "startup_phases": startup_phases,
        "offline_after_model_install": True,
        "scientific_boundary": (
            "Packaging and frozen-runtime integrity evidence only. The self-check exercises bundled "
            "Rasterio/GDAL/PROJ without loading DA3 or using the network. It does not establish DSM "
            "accuracy, model promotion, offline DA3 inference, clean-machine success, FPS, or soak."
        ),
    }
    report_path = BUILD_ROOT.parent / "sidecar-build-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("DepthWizard packaged scientific runtime build + qualification: PASS")
    print(f"Target: {triple}")
    print("Packaging mode: PyInstaller onedir staged as a Tauri resource tree")
    print(f"Runtime: {RUNTIME_DIR}")
    print(f"Runtime logical size: {logical_bytes / (1024 * 1024):.2f} MiB")
    print(f"Runtime regular files: {file_count}; symlinks: {symlink_count}")
    print(f"Executable SHA-256: {executable_sha}")
    print(f"Runtime tree SHA-256: {runtime_sha}")
    print(f"Frozen geospatial self-check: PASS in {self_check_elapsed:.3f} s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
