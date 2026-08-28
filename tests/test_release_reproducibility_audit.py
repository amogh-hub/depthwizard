import json
import tomllib
from pathlib import Path

from scripts.check_release_reproducibility import build_report


def _write_minimal_project(
    root: Path,
    *,
    npm_lock: bool,
    cargo_lock: bool,
    python_lock: bool,
    non_exact_npm: bool,
) -> None:
    desktop = root / "apps" / "desktop"
    tauri = desktop / "src-tauri"
    tauri.mkdir(parents=True)
    (root / "pyproject.toml").write_text("[project]\nname='depthwizard-test'\n", encoding="utf-8")
    (tauri / "Cargo.toml").write_text("[package]\nname='depthwizard-test'\n", encoding="utf-8")
    (desktop / "package.json").write_text(
        json.dumps(
            {
                "dependencies": {"react": "^19.2.0" if non_exact_npm else "19.2.0"},
                "devDependencies": {"vite": "8.2.2"},
            }
        ),
        encoding="utf-8",
    )
    if npm_lock:
        (desktop / "package-lock.json").write_text("{}\n", encoding="utf-8")
    if cargo_lock:
        (tauri / "Cargo.lock").write_text("# lock\n", encoding="utf-8")
    if python_lock:
        (root / "uv.lock").write_text("version = 1\n", encoding="utf-8")


def test_release_reproducibility_audit_passes_only_with_frozen_resolution(tmp_path: Path) -> None:
    _write_minimal_project(
        tmp_path,
        npm_lock=True,
        cargo_lock=True,
        python_lock=True,
        non_exact_npm=False,
    )
    report = build_report(tmp_path)
    assert report["ready_for_final_reproducibility_qualification"] is True
    assert report["findings"] == []
    assert report["non_exact_npm_specs"] == []


def test_release_reproducibility_audit_reports_every_unfrozen_surface(tmp_path: Path) -> None:
    _write_minimal_project(
        tmp_path,
        npm_lock=False,
        cargo_lock=False,
        python_lock=False,
        non_exact_npm=True,
    )
    report = build_report(tmp_path)
    assert report["ready_for_final_reproducibility_qualification"] is False
    findings = report["findings"]
    assert isinstance(findings, list)
    codes = {str(item["code"]) for item in findings if isinstance(item, dict)}
    assert codes == {
        "missing_npm_lockfile",
        "missing_cargo_lockfile",
        "missing_python_lockfile",
        "non_exact_npm_dependency_spec",
    }


def test_final_qualification_target_is_lock_enforcing_and_fail_closed() -> None:
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")
    target = makefile.split("release-train-5-final-qualification:\n", 1)[1].split(
        "\nstandalone-build:", 1
    )[0]

    assert "git ls-files --error-unmatch uv.lock" in target
    assert 'test -z "$$(git status --porcelain)"' in target
    assert "uv lock --check" in target
    assert (
        "uv sync --frozen --python 3.12 --extra ml --extra dev --extra standalone"
        in target
    )
    assert "verify_da3_runtime_dependencies" in target
    assert "DA3 runtime dependency imports PASS:" in target
    assert "import torch, torchvision; from torch import nn" in target
    assert "Scientific runtime imports PASS:" in target
    assert "npm ci --no-audit --no-fund" in target
    assert "cargo clippy --locked" in target
    assert "cargo test --locked" in target
    assert (
        "git diff --exit-code -- uv.lock apps/desktop/package-lock.json "
        "apps/desktop/src-tauri/Cargo.lock"
    ) in target
    assert 'pip install -e ".[dev,standalone]"' not in target


def test_ml_extra_declares_complete_curated_da3_runtime() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ml = payload["project"]["optional-dependencies"]["ml"]
    normalized = {str(spec).lower().replace("_", "-") for spec in ml}
    required_prefixes = {
        "torch",
        "torchvision",
        "opencv-python",
        "addict",
        "einops",
        "huggingface-hub",
        "imageio",
        "omegaconf",
        "requests",
        "safetensors",
        "tqdm",
        "evo",
        "e3nn",
    }
    for package in required_prefixes:
        assert any(spec.startswith(package) for spec in normalized), package


def test_standalone_build_requires_frozen_da3_import_probe() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "scripts" / "build_standalone_sidecar.py").read_text(encoding="utf-8")

    assert "verify_da3_runtime_dependencies()" in build_script
    assert '"--self-check-da3"' in build_script
    assert "_qualify_frozen_da3_imports" in build_script
    assert '"--collect-submodules",\n        "huggingface_hub"' in build_script
    assert '"--collect-submodules",\n        "safetensors"' in build_script


def test_verify_pins_pyright_to_invoking_python() -> None:
    root = Path(__file__).resolve().parents[1]
    verify = (root / "scripts" / "verify.py").read_text(encoding="utf-8")

    assert '"pyright",\n        "--pythonpath",\n        sys.executable,' in verify


def test_all_scientific_sidecar_build_targets_install_ml_extra() -> None:
    root = Path(__file__).resolve().parents[1]
    makefile = (root / "Makefile").read_text(encoding="utf-8")

    sidecar = makefile.split("sidecar-build:\n", 1)[1].split(
        "\nrelease-train-5-sidecar-smoke:", 1
    )[0]
    workstation = makefile.split("release-train-5-workstation-build:\n", 1)[1].split(
        "\n# Final qualification", 1
    )[0]

    assert 'pip install -e ".[ml,standalone]"' in sidecar
    assert 'pip install -e ".[ml,dev,standalone]"' in workstation
