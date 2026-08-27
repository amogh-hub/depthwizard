from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

from scripts.da3_frozen_compat import apply_da3_frozen_compat
from scripts.standalone_integrity import sha256_file, tree_identity

ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = ROOT / "apps" / "desktop" / "src-tauri"
RUNTIME_DIR = TAURI_DIR / "resources" / "depthwizard-core-runtime"
BUILD_ROOT = ROOT / "artifacts" / "standalone" / "pyinstaller"
ENTRY = ROOT / "scripts" / "depthwizard_sidecar_entry.py"
DA3_VENDOR = ROOT / ".vendor" / "depth-anything-3" / "src"
RUNTIME_MANIFEST_NAME = "runtime-manifest.json"
# Frozen native scientific runtimes can incur a one-time macOS cold-start/dyld validation cost.
# This watchdog is deliberately a correctness timeout, not a performance acceptance threshold.
# Startup time is recorded as evidence and is evaluated separately by RT7 performance gates.
SELF_CHECK_TIMEOUT_SECONDS = 120.0


def _source_git_sha() -> str:
    override = os.environ.get("DEPTHWIZARD_BUILD_GIT_SHA", "").strip()
    if override:
        value = override
    else:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
    if len(value) != 40 or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise RuntimeError(f"invalid DepthWizard source Git SHA for standalone build: {value!r}")
    return value.lower()


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
            "PROJ_NETWORK": "OFF",
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
            f"{SELF_CHECK_TIMEOUT_SECONDS:.0f}s; last_phase={location}; "
            f"startup_phases={phases}. The build is rejected rather than publishing an "
            "unqualified runtime."
        ) from exc
    elapsed = time.monotonic() - started
    events = _read_trace(trace_path)
    phases = [str(event.get("phase", "unknown")) for event in events]
    if completed.returncode != 0:
        raise RuntimeError(
            "frozen sidecar runtime self-check failed\n"
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
        raise TypeError("frozen sidecar self-check JSON root must be an object")
    if payload.get("status") != "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK":
        raise RuntimeError(f"frozen sidecar self-check returned non-passing status: {payload}")
    required_phases = {
        "python_entry",
        "self_check_import_complete",
        "self_check_da3_api_import_complete",
        "self_check_da3_runtime_closure_complete",
        "self_check_da3_geometry_probe_complete",
        "self_check_complete",
    }
    if not required_phases.issubset(phases):
        raise RuntimeError(
            "frozen sidecar startup trace is incomplete; "
            f"required={sorted(required_phases)}, observed={phases}"
        )
    if payload.get("da3_api_imported") is not True:
        raise RuntimeError("frozen sidecar did not import the DA3 public API")
    required_modules = payload.get("da3_runtime_modules_required")
    imported_modules = payload.get("da3_runtime_modules_imported")
    if not isinstance(required_modules, list) or imported_modules != required_modules:
        raise RuntimeError(
            "frozen sidecar did not import the complete DA3MONO-LARGE runtime module closure"
        )
    if payload.get("da3_affine_inverse_probe") != "PASS":
        raise RuntimeError("frozen sidecar did not pass the DA3 affine_inverse compatibility probe")
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

    # The pinned upstream DA3 geometry helper uses import-time torch.jit.script, which cannot
    # compile inside a normal PyInstaller frozen loader because source retrieval is intentionally
    # unavailable. Apply the exact audited compatibility patch before analysis. The helper's tensor
    # math is unchanged; script_if_tracing preserves scripting when tracing and eager inference stays
    # eager. Any unexpected upstream source shape fails closed.
    da3_compatibility = apply_da3_frozen_compat(DA3_VENDOR)

    source_git_sha = _source_git_sha()
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
        # DA3 builds the production network from YAML object paths through importlib. PyInstaller
        # cannot infer those modules from static imports, so freeze the exact DA3MONO-LARGE closure.
        "--hidden-import",
        "depth_anything_3.model.da3",
        "--hidden-import",
        "depth_anything_3.model.dinov2.dinov2",
        "--hidden-import",
        "depth_anything_3.model.dpt",
        # Hugging Face exposes PyTorchModelHubMixin through a lazy module. Bundle its implementation
        # explicitly so depth_anything_3.api can import in a frozen process. Safetensors is the
        # production local-weight path used by the pinned DA3 checkpoint.
        "--hidden-import",
        "huggingface_hub.hub_mixin",
        "--hidden-import",
        "safetensors.torch",
        "--hidden-import",
        "rasterio.serde",
        # Rasterio's C extensions dynamically import Python helpers such as rasterio.sample.
        # Data/binary collection alone is therefore insufficient for a frozen runtime. Collect the
        # package's Python submodules explicitly so the packaged import graph matches normal Python.
        "--collect-submodules",
        "rasterio",
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
    executable_sha = sha256_file(staged_executable)
    payload_sha, payload_files, payload_symlinks, payload_bytes = tree_identity(RUNTIME_DIR)
    runtime_manifest = {
        "schema_version": 4,
        "status": "QUALIFIED_DEPTHWIZARD_CORE_RUNTIME",
        "source_git_sha": source_git_sha,
        "target_triple": triple,
        "packaging_mode": "pyinstaller_onedir",
        "executable": staged_executable.name,
        "executable_sha256": executable_sha,
        "payload_tree_sha256": payload_sha,
        "payload_regular_files": payload_files,
        "payload_symlinks": payload_symlinks,
        "payload_logical_bytes": payload_bytes,
        "da3_frozen_compatibility": da3_compatibility,
        "frozen_self_check": self_check,
        "frozen_self_check_elapsed_seconds": round(self_check_elapsed, 3),
        "startup_phases": startup_phases,
        "model_weights_loaded_during_packaging_check": False,
        "network_used_during_packaging_check": False,
    }
    runtime_manifest_path = RUNTIME_DIR / RUNTIME_MANIFEST_NAME
    runtime_manifest_path.write_text(
        json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    runtime_sha, file_count, symlink_count, logical_bytes = tree_identity(RUNTIME_DIR)
    verified_payload_sha, _, _, _ = tree_identity(
        RUNTIME_DIR, excluded_relative_paths={RUNTIME_MANIFEST_NAME}
    )
    if verified_payload_sha != payload_sha:
        raise RuntimeError("runtime payload identity changed while writing qualification manifest")

    report = {
        "schema_version": 6,
        "status": "PASS_QUALIFIED_SIDECAR_BUILD",
        "source_git_sha": source_git_sha,
        "target_triple": triple,
        "platform": platform.platform(),
        "python": sys.version,
        "packaging_mode": "pyinstaller_onedir",
        "tauri_runtime_dir": str(RUNTIME_DIR.resolve()),
        "binary": str(staged_executable.resolve()),
        "binary_bytes": staged_executable.stat().st_size,
        "binary_sha256": executable_sha,
        "runtime_payload_tree_sha256": payload_sha,
        "runtime_tree_sha256": runtime_sha,
        "runtime_regular_files": file_count,
        "runtime_symlinks": symlink_count,
        "runtime_logical_bytes": logical_bytes,
        "da3_vendor_source": str(DA3_VENDOR.resolve()),
        "da3_frozen_compatibility": da3_compatibility,
        "da3_packaging": {
            "public_api_hidden_import": True,
            "da3mono_large_config_modules_hidden": True,
            "huggingface_hub_mixin_hidden_import": True,
            "safetensors_torch_hidden_import": True,
            "package_data_collected": True,
        },
        "geospatial_packaging": {
            "rasterio_serde_hidden_import": True,
            "rasterio_python_submodules_collected": True,
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
            "Rasterio/GDAL/PROJ, imports the full production DA3MONO-LARGE runtime closure, and "
            "numerically exercises the pinned affine_inverse compatibility path without loading model "
            "weights or using the network. It does not establish DSM accuracy, model promotion, "
            "clean-machine success, FPS, or soak."
        ),
    }
    report_path = BUILD_ROOT.parent / "sidecar-build-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("DepthWizard packaged scientific runtime build + qualification: PASS")
    print(f"Source Git SHA: {source_git_sha}")
    print(f"Target: {triple}")
    print("Packaging mode: PyInstaller onedir staged as a Tauri resource tree")
    print(f"Runtime: {RUNTIME_DIR}")
    print(f"Runtime logical size: {logical_bytes / (1024 * 1024):.2f} MiB")
    print(f"Runtime regular files: {file_count}; symlinks: {symlink_count}")
    print(f"Executable SHA-256: {executable_sha}")
    print(f"Runtime payload SHA-256: {payload_sha}")
    print(f"Runtime tree SHA-256: {runtime_sha}")
    print("DA3 frozen TorchScript compatibility: PASS (torch.jit.script_if_tracing)")
    print("DA3 production API + config-driven runtime closure: PASS")
    print(f"Frozen runtime self-check: PASS in {self_check_elapsed:.3f} s")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
