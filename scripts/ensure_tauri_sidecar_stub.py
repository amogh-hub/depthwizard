from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "apps" / "desktop" / "src-tauri" / "resources" / "depthwizard-core-runtime"


def main() -> None:
    """Create a compile-only Tauri resource stub when no qualified runtime is staged yet."""
    if os.name == "nt":
        raise RuntimeError("compile-only sidecar resource stub currently supports Unix Tauri hosts")

    target = RUNTIME_DIR / "depthwizard-core"
    if target.exists():
        target.chmod(target.stat().st_mode | 0o111)
        print(f"Tauri scientific runtime resource already exists: {target}")
        return

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    target.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    target.chmod(0o755)
    manifest = {
        "schema_version": 1,
        "status": "COMPILE_ONLY_RUNTIME_STUB",
        "scientific_runtime": False,
        "purpose": "Tauri compile/test configuration only; never release or acceptance evidence.",
    }
    (RUNTIME_DIR / "runtime-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Created compile-only Tauri scientific runtime resource stub: {target}")


if __name__ == "__main__":
    main()
