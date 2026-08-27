from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*command: str) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    run(sys.executable, "-m", "pytest")
    run(sys.executable, "-m", "ruff", "check", "src", "tests", "scripts")
    run(
        sys.executable,
        "-m",
        "pyright",
        "src",
        "tests",
        "scripts/train_ortholoc_height_acceptance.py",
        "scripts/train_ortholoc_multiscene.py",
        "scripts/train_ortholoc_multiscene_v2.py",
        "scripts/train_ortholoc_multiscene_v3.py",
        "scripts/train_ortholoc_multiscene_v4.py",
        "scripts/evaluate_ortholoc_adaptive_refinement.py",
        "scripts/evaluate_ortholoc_frozen_holdout.py",
        "scripts/evaluate_ortholoc_frozen_location_v2.py",
        "scripts/audit_potsdam_contract.py",
        "scripts/evaluate_potsdam_external.py",
        "scripts/evaluate_potsdam_external_v2.py",
        "scripts/preflight_potsdam_external_v2_execution.py",
        "scripts/production_runtime_smoke.py",
        "scripts/release_train_2_validation_smoke.py",
        "scripts/release_train_3_spatial_foundation_smoke.py",
        "scripts/release_train_3_mesh_smoke.py",
        "scripts/release_train_3_export_smoke.py",
        "scripts/release_train_3_workstation_smoke.py",
        "scripts/release_train_4_scientific_analytical_smoke.py",
        "scripts/build_standalone_sidecar.py",
        "scripts/ensure_tauri_sidecar_stub.py",
        "scripts/pyinstaller_geospatial_runtime_hook.py",
        "scripts/release_train_5_sidecar_smoke.py",
        "scripts/release_train_5_app_bundle_smoke.py",
        "scripts/benchmark_ortholoc_demo.py",
        "scripts/demo_india_absolute.py",
        "scripts/smoke_height_model.py",
    )
    print("Python verification passed: tests, lint, and static typing are green.")
