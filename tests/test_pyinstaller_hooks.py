from __future__ import annotations

import tomllib
from pathlib import Path

from depthwizard.__pyinstaller import get_hook_dirs

ROOT = Path(__file__).resolve().parents[1]


def test_pyinstaller_hook_directory_contains_imageio_metadata_hook() -> None:
    hook_dirs = get_hook_dirs()
    assert len(hook_dirs) == 1
    hook_dir = Path(hook_dirs[0])
    hook_path = hook_dir / "hook-imageio.py"
    assert hook_dir.is_dir()
    assert hook_path.is_file()
    source = hook_path.read_text(encoding="utf-8")
    assert 'copy_metadata("imageio")' in source


def test_pyinstaller40_entry_point_is_declared_in_source_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyinstaller40 = project["project"]["entry-points"]["pyinstaller40"]
    assert pyinstaller40["hook-dirs"] == "depthwizard.__pyinstaller:get_hook_dirs"
