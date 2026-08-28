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

from depthwizard.geometry_prior.da3_runtime_contract import (
    DA3_RUNTIME_DEPENDENCY_MODULES,
    verify_da3_runtime_dependencies,
)
from scripts.da3_frozen_compat import apply_da3_frozen_compat
from scripts.standalone_integrity import sha256_file, tree_identity

ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = ROOT / "apps" / "desktop" / "src-tauri"
RUNTIME_DIR = TAURI_DIR / "resources" / "depthwizard-core-runtime"
BUILD_ROOT = ROOT / "artifacts" / "standalone" / "pyinstaller"
ENTRY = ROOT / "scripts" / "depthwizard_sidecar_entry.py"
DA3_VENDOR = ROOT / ".vendor" / "depth-anything-3" / "src"
RUNTIME_MANIFEST_NAME = "runtime-manifest.json"
SELF_CHECK_TIMEOUT_SECONDS = 120.0
# Importing the complete frozen DA3/PyTorch closure is intentionally slower than the geospatial
# bootstrap probe, but it must still complete before a runtime can be staged into the desktop app.
DA3_IMPORT_CHECK_TIMEOUT_SECONDS = 300.0


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


def _offline_environment(trace_path: Path) -> dict[str, str]:
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
    return environment


def _parse_self_check_result(
    completed: subprocess.CompletedProcess[str],
    *,
    context: str,
) -> dict[str, object]:
    if completed.returncode != 0:
        raise RuntimeError(
            f"{context} failed\nreturncode={completed.returncode}\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise RuntimeError(f"{context} produced no machine-readable result")
    try:
        payload = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{context} did not end with valid JSON\n"
            f"stdout={completed.stdout}\nstderr={completed.stderr}"
        ) from exc
    if not isinstance(payload, dict):
        raise TypeError(f"{context} JSON root must be an object")
    if payload.get("status") != "PASS_PACKAGED_GEOSPATIAL_SELF_CHECK":
        raise RuntimeError(f"{context} returned non-passing status: {payload}")
    return payload


def _qualify_frozen_runtime(executable: Path) -> tuple[dict[str, object], float, list[str]]:
    """Qualify deterministic frozen geospatial startup before publishing the runtime tree."""
    trace_path = BUILD_ROOT.parent / "frozen-self-check-startup-trace.jsonl"
    trace_path.unlink(missing_ok=True)
    environment = _offline_environment(trace_path)
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
            "frozen sidecar geospatial qualification timed out after "
            f"{SELF_CHECK_TIMEOUT_SECONDS:.0f}s; last_phase={location}; "
            f"startup_phases={phases}. The build is rejected rather than publishing an "
            "unqualified geospatial runtime."
        ) from exc
    elapsed = time.monotonic() - started
    phases = [str(event.get("phase", "unknown")) for event in _read_trace(trace_path)]
    payload = _parse_self_check_result(completed, context="frozen sidecar geospatial self-check")
    required_phases = {
        "python_entry",
        "self_check_import_complete",
        "self_check_pyproj_epsg_complete",
        "self_check_rasterio_roundtrip_complete",
        "self_check_complete",
    }
    if not required_phases.issubset(phases):
        raise RuntimeError(
            "frozen sidecar startup trace is incomplete; "
            f"required={sorted(required_phases)}, observed={phases}"
        )
    if payload.get("rasterio_serde_imported") is not True:
        raise RuntimeError("frozen sidecar did not import rasterio.serde")
    if payload.get("epsg_roundtrip") != 32643:
        raise RuntimeError("frozen sidecar did not preserve the EPSG:32643 CRS round-trip")
    if payload.get("da3_probe_required") is not False:
        raise RuntimeError("lightweight build self-check unexpectedly forced DA3 initialization")
    if payload.get("network_used") is not False:
        raise RuntimeError("frozen geospatial qualification unexpectedly used the network")
    if payload.get("model_loaded") is not False or payload.get("model_weights_loaded") is not False:
        raise RuntimeError("build-time geospatial qualification unexpectedly loaded model weights")
    return payload, elapsed, phases


def _qualify_frozen_da3_imports(
    executable: Path,
) -> tuple[dict[str, object], float, list[str]]:
    """Prove the packaged DA3 dependency/module closure before Tauri can consume the runtime."""
    trace_path = BUILD_ROOT.parent / "frozen-da3-import-startup-trace.jsonl"
    trace_path.unlink(missing_ok=True)
    environment = _offline_environment(trace_path)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(executable), "--self-check-da3"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
            timeout=DA3_IMPORT_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        phases = [str(event.get("phase", "unknown")) for event in _read_trace(trace_path)]
        location = phases[-1] if phases else "before Python entrypoint"
        raise RuntimeError(
            "frozen DA3 import qualification timed out after "
            f"{DA3_IMPORT_CHECK_TIMEOUT_SECONDS:.0f}s; last_phase={location}; "
            f"startup_phases={phases}. The runtime is rejected before Tauri packaging."
        ) from exc
    elapsed = time.monotonic() - started
    phases = [str(event.get("phase", "unknown")) for event in _read_trace(trace_path)]
    payload = _parse_self_check_result(completed, context="frozen DA3 import self-check")
    if payload.get("da3_probe_required") is not True:
        raise RuntimeError("frozen DA3 import self-check did not enable the DA3 closure probe")
    required_dependencies = list(DA3_RUNTIME_DEPENDENCY_MODULES)
    if payload.get("da3_dependency_modules_required") != required_dependencies:
        raise RuntimeError("frozen DA3 import self-check reported the wrong dependency contract")
    if payload.get("da3_dependency_modules_imported") != required_dependencies:
        raise RuntimeError("frozen DA3 dependency closure did not import completely")
    required_modules = payload.get("da3_runtime_modules_required")
    if not isinstance(required_modules, list) or payload.get("da3_runtime_modules_imported") != required_modules:
        raise RuntimeError("frozen DA3 production module closure did not import completely")
    if payload.get("da3_api_imported") is not True:
        raise RuntimeError("frozen DA3 API import was not proven")
    if payload.get("da3_geometry_imported") is not True:
        raise RuntimeError("frozen DA3 geometry helper import was not proven")
    if payload.get("da3_affine_inverse_probe") != "PASS":
        raise RuntimeError("frozen DA3 affine_inverse compatibility probe did not pass")
    if payload.get("network_used") is not False:
        raise RuntimeError("frozen DA3 import qualification unexpectedly used the network")
    if payload.get("model_loaded") is not False or payload.get("model_weights_loaded") is not False:
        raise RuntimeError("frozen DA3 import qualification unexpectedly loaded model weights")
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

    # Fail before a 90-second PyInstaller build if the locked Python environment does not contain
    # the complete curated DA3 monocular dependency closure.
    build_dependency_modules = verify_da3_runtime_dependencies()

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
        "depth_anything_3.api",
        # DA3 builds the production network from YAML object paths through importlib. PyInstaller
        # cannot infer those modules from static imports, so freeze the exact DA3MONO-LARGE closure.
        "--hidden-import",
        "depth_anything_3.model.da3",
        "--hidden-import",
        "depth_anything_3.model.dinov2.dinov2",
        "--hidden-import",
        "depth_anything_3.model.dpt",
        # Hugging Face and safetensors both contain lazy/dynamic import surfaces. Collect their
        # Python submodules in addition to the exact runtime imports below.
        "--collect-submodules",
        "huggingface_hub",
        "--collect-submodules",
        "safetensors",
        "--hidden-import",
        "rasterio.serde",
        # Rasterio's C extensions dynamically import Python helpers such as rasterio.sample.
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
    ]
    for module_name in DA3_RUNTIME_DEPENDENCY_MODULES:
        command.extend(("--hidden-import", module_name))
    command.append(str(ENTRY))
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
    da3_import_check, da3_import_elapsed, da3_import_phases = _qualify_frozen_da3_imports(
        staged_executable
    )
    executable_sha = sha256_file(staged_executable)
    payload_sha, payload_files, payload_symlinks, payload_bytes = tree_identity(RUNTIME_DIR)
    runtime_manifest = {
        "schema_version": 6,
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
        "build_da3_dependency_modules": list(build_dependency_modules),
        "frozen_self_check": self_check,
        "frozen_self_check_elapsed_seconds": round(self_check_elapsed, 3),
        "startup_phases": startup_phases,
        "frozen_da3_import_check": da3_import_check,
        "frozen_da3_import_check_elapsed_seconds": round(da3_import_elapsed, 3),
        "da3_import_startup_phases": da3_import_phases,
        "da3_runtime_execution_gate": "release_train_5_full_acceptance",
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
        "schema_version": 8,
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
            "dependency_contract": list(DA3_RUNTIME_DEPENDENCY_MODULES),
            "build_dependency_import_check": "passed",
            "frozen_dependency_import_check": "passed",
            "public_api_hidden_import": True,
            "da3mono_large_config_modules_hidden": True,
            "huggingface_hub_submodules_collected": True,
            "safetensors_submodules_collected": True,
            "package_data_collected": True,
            "runtime_execution_gate": "release_train_5_full_acceptance",
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
        "frozen_da3_import_check": da3_import_check,
        "frozen_da3_import_check_elapsed_seconds": round(da3_import_elapsed, 3),
        "da3_import_startup_phases": da3_import_phases,
        "offline_after_model_install": True,
        "scientific_boundary": (
            "Packaging and frozen-runtime integrity evidence only. The build proves bundled "
            "Rasterio/GDAL/PROJ plus the complete curated DA3 monocular dependency/module closure "
            "inside the actual frozen executable, without loading model weights or using network "
            "access. Actual packaged DA3 correctness is still gated by "
            "release_train_5_full_acceptance, which must launch the real application offline and "
            "complete an end-to-end DA3 reconstruction. This build report does not establish DSM "
            "accuracy, model promotion, clean-machine success, FPS, or soak."
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
    print("DA3 frozen TorchScript compatibility patch: APPLIED AND AUDITED")
    print(f"Frozen geospatial self-check: PASS in {self_check_elapsed:.3f} s")
    print(f"Frozen DA3 dependency/import self-check: PASS in {da3_import_elapsed:.3f} s")
    print("DA3 model-weight execution remains gated by full RT5 packaged inference")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
