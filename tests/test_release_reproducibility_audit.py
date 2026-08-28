import json
from pathlib import Path

from scripts.check_release_reproducibility import build_report


def _write_minimal_project(root: Path, *, npm_lock: bool, cargo_lock: bool, moving: bool) -> None:
    desktop = root / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    tauri.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='depthwizard-test'\n", encoding="utf-8")
    (tauri / "Cargo.toml").write_text("[package]\nname='depthwizard-test'\n", encoding="utf-8")
    (desktop / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"react": "latest" if moving else "19.2.0"},
                "devDependencies": {"vite": "7.1.0"},
            }
        ),
        encoding="utf-8",
    )
    if npm_lock:
        (desktop / "package-lock.json").write_text("{}\n", encoding="utf-8")
    if cargo_lock:
        (tauri / "Cargo.lock").write_text("# lock\n", encoding="utf-8")


def test_release_reproducibility_audit_passes_only_with_frozen_resolution(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path, npm_lock=True, cargo_lock=True, moving=False)
    report = build_report(tmp_path)
    assert report["ready_for_final_reproducibility_qualification"] is True
    assert report["findings"] == []


def test_release_reproducibility_audit_reports_missing_locks_and_moving_tags(tmp_path: Path) -> None:
    _write_minimal_project(tmp_path, npm_lock=False, cargo_lock=False, moving=True)
    report = build_report(tmp_path)
    assert report["ready_for_final_reproducibility_qualification"] is False
    findings = report["findings"]
    assert isinstance(findings, list)
    codes = {str(item["code"]) for item in findings if isinstance(item, dict)}
    assert codes == {"missing_npm_lockfile", "missing_cargo_lockfile", "moving_npm_dependency_spec"}
