from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY = ROOT / "apps" / "desktop" / "src-tauri" / "capabilities" / "default.json"


def test_main_window_can_open_native_file_dialog_without_broad_plugin_permissions() -> None:
    capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))

    assert capability["identifier"] == "main-capability"
    assert capability["local"] is True
    assert capability["windows"] == ["main"]
    assert "core:default" in capability["permissions"]
    assert "dialog:allow-open" in capability["permissions"]
    assert "dialog:default" not in capability["permissions"]
