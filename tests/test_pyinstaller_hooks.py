from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from depthwizard.__pyinstaller import get_hook_dirs


def test_pyinstaller_hook_directory_contains_imageio_metadata_hook() -> None:
    hook_dirs = get_hook_dirs()
    assert len(hook_dirs) == 1
    hook_dir = Path(hook_dirs[0])
    hook_path = hook_dir / "hook-imageio.py"
    assert hook_dir.is_dir()
    assert hook_path.is_file()
    source = hook_path.read_text(encoding="utf-8")
    assert 'copy_metadata("imageio")' in source


def test_pyinstaller40_entry_point_registers_depthwizard_hook_directory() -> None:
    registered = [
        item
        for item in entry_points(group="pyinstaller40")
        if item.name == "hook-dirs" and item.value == "depthwizard.__pyinstaller:get_hook_dirs"
    ]
    assert len(registered) == 1
    loaded = registered[0].load()
    assert loaded() == get_hook_dirs()
