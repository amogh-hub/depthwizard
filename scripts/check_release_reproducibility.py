from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "apps" / "desktop"
_EXACT_NPM_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _non_exact_npm_specs(dependencies: dict[str, object]) -> list[str]:
    return sorted(
        name
        for name, spec in dependencies.items()
        if not isinstance(spec, str) or _EXACT_NPM_VERSION.fullmatch(spec.strip()) is None
    )


def build_report(root: Path = ROOT) -> dict[str, object]:
    desktop = root / "apps" / "desktop"
    package_json = desktop / "package.json"
    package_lock = desktop / "package-lock.json"
    cargo_toml = desktop / "src-tauri" / "Cargo.toml"
    cargo_lock = desktop / "src-tauri" / "Cargo.lock"
    python_manifest = root / "pyproject.toml"
    python_lock = root / "uv.lock"

    package = _json(package_json)
    dependencies = {
        **dict(package.get("dependencies") or {}),
        **dict(package.get("devDependencies") or {}),
    }
    non_exact_npm_specs = _non_exact_npm_specs(dependencies)

    findings: list[dict[str, object]] = []
    if not package_lock.is_file():
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "missing_npm_lockfile",
                "path": str(package_lock.relative_to(root)),
                "message": "Desktop JavaScript transitive dependency resolution is not frozen.",
            }
        )
    if not cargo_lock.is_file():
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "missing_cargo_lockfile",
                "path": str(cargo_lock.relative_to(root)),
                "message": "Desktop Rust dependency resolution is not frozen.",
            }
        )
    if not python_lock.is_file():
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "missing_python_lockfile",
                "path": str(python_lock.relative_to(root)),
                "message": "Python transitive dependency resolution is not frozen.",
            }
        )
    if non_exact_npm_specs:
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "non_exact_npm_dependency_spec",
                "path": str(package_json.relative_to(root)),
                "packages": non_exact_npm_specs,
                "message": (
                    "Release package.json must use exact top-level versions; lockfiles then freeze "
                    "the full transitive graph."
                ),
            }
        )

    return {
        "schema": "depthwizard.release-reproducibility-audit.v2",
        "npm_lockfile_present": package_lock.is_file(),
        "cargo_lockfile_present": cargo_lock.is_file(),
        "python_lockfile_present": python_lock.is_file(),
        "non_exact_npm_specs": non_exact_npm_specs,
        "python_manifest_present": python_manifest.is_file(),
        "cargo_manifest_present": cargo_toml.is_file(),
        "findings": findings,
        "ready_for_final_reproducibility_qualification": not findings,
        "claim_boundary": (
            "This source audit checks deterministic dependency-resolution prerequisites only. A "
            "passing result does not replace the final clean-machine build/install/process "
            "qualification or prove accelerator compatibility."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit DepthWizard source prerequisites for deterministic release reproduction."
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero when final reproducibility prerequisites are not yet satisfied",
    )
    args = parser.parse_args()

    report = build_report()
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(encoded, encoding="utf-8")
        temporary.replace(args.output)
    else:
        print(encoded, end="")

    if args.strict and not bool(report["ready_for_final_reproducibility_qualification"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
