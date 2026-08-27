from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = ROOT / "apps" / "desktop" / "src-tauri"
BIN_DIR = TAURI_DIR / "binaries"
BUILD_ROOT = ROOT / "artifacts" / "standalone" / "pyinstaller"
ENTRY = ROOT / "scripts" / "depthwizard_sidecar_entry.py"
RUNTIME_HOOK = ROOT / "scripts" / "pyinstaller_geospatial_runtime_hook.py"
DA3_VENDOR = ROOT / ".vendor" / "depth-anything-3" / "src"


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


def main() -> None:
    """Build the no-Python-required scientific sidecar for Tauri packaging."""
    if not DA3_VENDOR.is_dir():
        raise RuntimeError(
            "pinned Depth Anything 3 source is missing; run `make da3-setup` before sidecar packaging"
        )
    if not RUNTIME_HOOK.is_file():
        raise RuntimeError(f"PyInstaller geospatial runtime hook is missing: {RUNTIME_HOOK}")
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
    for path in (dist_dir, work_dir, spec_dir, BIN_DIR):
        path.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
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
        "--runtime-hook",
        str(RUNTIME_HOOK),
        "--hidden-import",
        "torch",
        "--hidden-import",
        "depth_anything_3.api",
        "--hidden-import",
        "rasterio.serde",
        "--collect-all",
        "rasterio",
        "--collect-all",
        "pyproj",
        "--collect-data",
        "depth_anything_3",
        str(ENTRY),
    ]
    subprocess.run(command, cwd=ROOT, check=True)

    built = dist_dir / f"depthwizard-core{extension}"
    if not built.is_file():
        raise RuntimeError(f"PyInstaller did not produce the expected executable: {built}")

    target = BIN_DIR / f"depthwizard-core-{triple}{extension}"
    shutil.copy2(built, target)
    if os.name != "nt":
        target.chmod(target.stat().st_mode | 0o111)

    report = {
        "schema_version": 2,
        "status": "PASS_SIDECAR_BUILD",
        "target_triple": triple,
        "platform": platform.platform(),
        "python": sys.version,
        "binary": str(target.resolve()),
        "bytes": target.stat().st_size,
        "sha256": _sha256(target),
        "da3_vendor_source": str(DA3_VENDOR.resolve()),
        "geospatial_packaging": {
            "rasterio_collect_all": True,
            "pyproj_collect_all": True,
            "rasterio_serde_hidden_import": True,
            "runtime_data_hook": str(RUNTIME_HOOK.resolve()),
        },
        "offline_after_model_install": True,
        "scientific_boundary": (
            "Packaging evidence only. This build does not establish DSM accuracy, model promotion, "
            "clean-machine success, or offline inference until the dedicated RT5 acceptance runs."
        ),
    }
    report_path = BUILD_ROOT.parent / "sidecar-build-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("DepthWizard packaged scientific sidecar build: PASS")
    print(f"Target: {triple}")
    print(f"Binary: {target}")
    print(f"Size: {report['bytes'] / (1024 * 1024):.2f} MiB")
    print("Geospatial bundle policy: rasterio + pyproj collected, runtime data hook enabled")
    print(f"SHA-256: {report['sha256']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
