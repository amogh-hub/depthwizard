from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPABILITY = ROOT / "apps" / "desktop" / "src-tauri" / "capabilities" / "default.json"
TAURI_CONFIG = ROOT / "apps" / "desktop" / "src-tauri" / "tauri.conf.json"


def test_main_window_can_open_native_file_dialog_without_broad_plugin_permissions() -> None:
    capability = json.loads(CAPABILITY.read_text(encoding="utf-8"))

    assert capability["identifier"] == "main-capability"
    assert capability["local"] is True
    assert capability["windows"] == ["main"]
    assert "core:default" in capability["permissions"]
    assert "dialog:allow-open" in capability["permissions"]
    assert "dialog:default" not in capability["permissions"]


def test_csp_allows_authenticated_blob_backed_glb_assets() -> None:
    config = json.loads(TAURI_CONFIG.read_text(encoding="utf-8"))
    csp = config["app"]["security"]["csp"]
    connect_src = next(
        directive.strip()
        for directive in csp.split(";")
        if directive.strip().startswith("connect-src ")
    )

    assert "http://127.0.0.1:*" in connect_src
    assert "blob:" in connect_src
