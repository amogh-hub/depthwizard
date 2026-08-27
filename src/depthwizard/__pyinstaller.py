from __future__ import annotations

from pathlib import Path


def get_hook_dirs() -> list[str]:
    """Return DepthWizard-owned PyInstaller analysis hooks.

    These hooks are registered through the standard ``pyinstaller40.hook-dirs`` entry point so
    frozen-runtime packaging requirements remain versioned with DepthWizard instead of depending on
    machine-local PyInstaller configuration.
    """
    return [str(Path(__file__).resolve().with_name("_pyinstaller_hooks"))]
