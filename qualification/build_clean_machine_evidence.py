#!/usr/bin/env python3
"""Build clean-machine standalone evidence from real observations and the installed app bundle.

This helper is designed to run on the clean Mac itself. It does not require a source checkout.
It computes a deterministic SHA-256 tree identity for the installed `.app` bundle and emits PASS
only when every frozen clean-machine observation is explicitly true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml

FROZEN_HEAD = "339bdf485149f552db846543b9e09377b567c19c"
REQUIRED_TRUE = (
    "packaged_app_launch",
    "no_user_visible_terminal",
    "owned_sidecar_boot",
    "offline_after_model_install",
    "end_to_end_reconstruction",
    "metric_calibration",
    "terrain_3d",
    "export_and_reopen",
)


def _load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    payload = json.loads(text) if path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise SystemExit("clean-machine observation root must be an object")
    return payload


def _tree_sha256(root: Path) -> str:
    """Hash relative paths, file types, symlink targets and regular-file bytes deterministically."""
    digest = hashlib.sha256()
    root = root.resolve(strict=True)
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        if path.is_symlink():
            digest.update(b"L\0" + rel + b"\0" + os.readlink(path).encode("utf-8") + b"\0")
            continue
        if path.is_dir():
            digest.update(b"D\0" + rel + b"\0")
            continue
        if path.is_file():
            digest.update(b"F\0" + rel + b"\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build DepthWizard clean-machine PASS evidence.")
    parser.add_argument("observation", type=Path, help="YAML/JSON populated during real clean-Mac use")
    parser.add_argument("--application", type=Path, required=True, help="Installed DepthWizard.app path")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    observation_path = args.observation.resolve(strict=True)
    payload = _load(observation_path)
    declared_head = payload.get("source_build_git_head")
    if declared_head != FROZEN_HEAD:
        raise SystemExit(
            f"observation must identify frozen source build {FROZEN_HEAD}; got {declared_head!r}"
        )
    evidence = payload.get("evidence")
    if not isinstance(evidence, list) or not evidence or not all(
        isinstance(item, str) and item.strip() for item in evidence
    ):
        raise SystemExit("clean-machine observation requires explicit evidence references")
    for field in REQUIRED_TRUE:
        if payload.get(field) is not True:
            raise SystemExit(f"clean-machine field {field!r} must be explicitly true after observation")

    application = args.application.expanduser().resolve(strict=True)
    if application.suffix != ".app" or not application.is_dir():
        raise SystemExit(f"--application must point to an installed .app bundle: {application}")
    app_sha = _tree_sha256(application)
    hardware = payload.get("clean_machine_hardware")
    if not isinstance(hardware, str) or not hardware.strip():
        raise SystemExit("clean_machine_hardware must be recorded")

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_CLEAN_MACHINE_STANDALONE",
        "git_head": FROZEN_HEAD,
        "application_path": str(application),
        "application_sha256": app_sha,
        "application_sha256_semantics": "deterministic installed .app bundle tree SHA-256",
        "clean_machine_hardware": hardware.strip(),
        "evidence": [item.strip() for item in evidence],
    }
    for field in REQUIRED_TRUE:
        report[field] = True
    report["claim_boundary"] = (
        "PASS is emitted only from explicit clean-machine observations and a deterministic identity "
        "hash of the installed DepthWizard.app bundle."
    )

    output = args.output.resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "application_sha256": app_sha, "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
