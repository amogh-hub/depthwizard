from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "apps" / "desktop"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(root: Path = ROOT) -> dict[str, object]:
    desktop = root / "apps" / "desktop"
    package_json = desktop / "package.json"
    package_lock = desktop / "package-lock.json"
    cargo_toml = desktop / "src-tauri" / "Cargo.toml"
    cargo_lock = desktop / "src-tauri" / "Cargo.lock"

    package = _json(package_json)
    dependencies = {
        **dict(package.get("dependencies") or {}),
        **dict(package.get("devDependencies") or {}),
    }
    moving_npm_specs = sorted(
        name
        for name, spec in dependencies.items()
        if isinstance(spec, str) and spec.strip().lower() in {"latest", "next", "*"}
    )

    findings: list[dict[str, object]] = []
    if not package_lock.is_file():
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "missing_npm_lockfile",
                "path": str(package_lock.relative_to(root)),
                "message": "Desktop JavaScript dependency resolution is not frozen.",
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
    if moving_npm_specs:
        findings.append(
            {
                "severity": "blocker_for_final_reproducibility",
                "code": "moving_npm_dependency_spec",
                "path": str(package_json.relative_to(root)),
                "packages": moving_npm_specs,
                "message": "Release manifests must not depend on moving npm tags.",
            }
        )

    return {
        "schema": "depthwizard.release-reproducibility-audit.v1",
        "npm_lockfile_present": package_lock.is_file(),
        "cargo_lockfile_present": cargo_lock.is_file(),
        "moving_npm_specs": moving_npm_specs,
        "python_manifest_present": (root / "pyproject.toml").is_file(),
        "cargo_manifest_present": cargo_toml.is_file(),
        "findings": findings,
        "ready_for_final_reproducibility_qualification": not findings,
        "claim_boundary": (
            "This source audit checks release dependency-resolution prerequisites only. A passing "
            "result does not replace the final clean-machine build/install/process qualification."
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
