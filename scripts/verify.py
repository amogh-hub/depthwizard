from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    # Verification is intentionally scoped to DepthWizard-owned tests. Packaged PyInstaller
    # resource trees can contain third-party files named test_*.py (for example inside torch),
    # and those are runtime payload, not repository tests. Never let generated scientific runtime
    # contents alter the verification surface.
    run(sys.executable, "-m", "pytest", "tests")
    run(sys.executable, "-m", "ruff", "check", "src", "tests", "scripts")
    # Pyright is a Node-backed checker even when launched through `python -m pyright`; without an
    # explicit interpreter it may inspect a system Python instead of the environment that actually
    # ran pytest. Pin it to this exact interpreter so imports and typing are checked against the
    # same locked runtime used by qualification.
    run(
        sys.executable,
        "-m",
        "pyright",
        "--pythonpath",
        sys.executable,
        "src",
        "tests",
        "scripts/train_ortholoc_height_acceptance.py",
        "scripts/train_ortholoc_multiscene.py",
        "scripts/train_ortholoc_multiscene_v2.py",
        "scripts/train_ortholoc_multiscene_v3.py",
        "scripts/train_ortholoc_multiscene_v4.py",
        "scripts/train_ortholoc_structure_band_v5.py",
        "scripts/evaluate_ortholoc_adaptive_refinement.py",
        "scripts/evaluate_ortholoc_frozen_holdout.py",
        "scripts/evaluate_ortholoc_frozen_location_v2.py",
        "scripts/audit_potsdam_contract.py",
        "scripts/evaluate_potsdam_external.py",
        "scripts/evaluate_potsdam_external_v2.py",
        "scripts/evaluate_final_science_campaign.py",
        "scripts/freeze_final_science_manifest.py",
        "scripts/preflight_potsdam_external_v2_execution.py",
        "scripts/production_runtime_smoke.py",
        "scripts/release_train_2_validation_smoke.py",
        "scripts/release_train_3_spatial_foundation_smoke.py",
        "scripts/release_train_3_mesh_smoke.py",
        "scripts/release_train_3_export_smoke.py",
        "scripts/release_train_3_workstation_smoke.py",
        "scripts/release_train_4_scientific_analytical_smoke.py",
        "scripts/da3_frozen_compat.py",
        "scripts/build_standalone_sidecar.py",
        "scripts/ensure_tauri_sidecar_stub.py",
        "scripts/release_train_5_sidecar_smoke.py",
        "scripts/release_train_5_app_bundle_smoke.py",
        "scripts/release_train_5_full_acceptance.py",
        "scripts/release_train_7_soak.py",
        "scripts/verify_problem_statement_inputs.py",
        "scripts/check_sih26175_completion.py",
        "scripts/benchmark_ortholoc_demo.py",
        "scripts/demo_india_absolute.py",
        "scripts/smoke_da3.py",
        "scripts/smoke_height_model.py",
    )
    print("Python verification passed: tests, lint, and static typing are green.")
