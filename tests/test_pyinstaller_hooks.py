from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from depthwizard.__pyinstaller import get_hook_dirs

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("hook_name", "distribution"),
    [
        ("hook-imageio.py", "imageio"),
        ("hook-safetensors.py", "safetensors"),
    ],
)
def test_pyinstaller_hook_directory_preserves_required_distribution_metadata(
    hook_name: str,
    distribution: str,
) -> None:
    hook_dirs = get_hook_dirs()
    assert len(hook_dirs) == 1
    hook_dir = Path(hook_dirs[0])
    hook_path = hook_dir / hook_name
    assert hook_dir.is_dir()
    assert hook_path.is_file()
    source = hook_path.read_text(encoding="utf-8")
    assert f'copy_metadata("{distribution}")' in source


def test_pyinstaller40_entry_point_is_declared_in_source_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    pyinstaller40 = project["project"]["entry-points"]["pyinstaller40"]
    assert pyinstaller40["hook-dirs"] == "depthwizard.__pyinstaller:get_hook_dirs"
