from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "binaries"


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


def main() -> None:
    """Create a compile-only Tauri external-binary stub when no packaged core exists yet."""
    if os.name == "nt":
        raise RuntimeError("compile-only sidecar stub helper currently supports Unix Tauri hosts")

    BIN_DIR.mkdir(parents=True, exist_ok=True)
    target = BIN_DIR / f"depthwizard-core-{_host_triple()}"
    if target.exists():
        target.chmod(target.stat().st_mode | 0o111)
        print(f"Tauri sidecar compile target already exists: {target}")
        return

    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    print(f"Created compile-only Tauri sidecar stub: {target}")


if __name__ == "__main__":
    main()
