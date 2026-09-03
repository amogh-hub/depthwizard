from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from qualification.audit_tsd_target_supervision import (
    audit_target_supervision,
    print_audit_summary,
    sha256_file,
)

EXPECTED_TARGET_MANIFEST_SHA256 = (
    "cffe83e74d6a58adf5ea4919f70de35fd0a2b7a3532d1d272a9d2e927f20716c"
)
TARGET_MANIFEST_REL = Path("qualification/evidence/tsd-metric-targets-v1.json")
AUDIT_REL = Path("qualification/evidence/tsd-metric-target-supervision-audit-v1.json")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def _require_clean_tracked_worktree(root: Path) -> tuple[str, str]:
    branch = _git_output(root, "branch", "--show-current")
    if branch != "engineering/terrain-structure-vnext":
        raise RuntimeError(f"wrong branch: {branch}")
    dirty = _git_output(root, "status", "--short", "--untracked-files=no")
    if dirty:
        raise RuntimeError("tracked worktree is not clean")
    return _git_output(root, "rev-parse", "HEAD"), branch


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advance the frozen Potsdam TSD campaign through guarded local stages."
    )
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    source_sha, branch = _require_clean_tracked_worktree(root)
    target_manifest = (root / TARGET_MANIFEST_REL).resolve()
    audit_path = (root / AUDIT_REL).resolve()

    if not target_manifest.is_file():
        raise FileNotFoundError(target_manifest)

    actual_target_sha = sha256_file(target_manifest)
    if actual_target_sha != EXPECTED_TARGET_MANIFEST_SHA256:
        raise RuntimeError(
            "frozen target manifest identity changed: "
            f"expected {EXPECTED_TARGET_MANIFEST_SHA256}, got {actual_target_sha}"
        )

    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("target_manifest_sha256") != EXPECTED_TARGET_MANIFEST_SHA256:
            raise RuntimeError("existing TSD audit is bound to a different target manifest")
        print("campaign_stage=TARGET_SUPERVISION_AUDIT_ALREADY_COMPLETE")
    else:
        audit = audit_target_supervision(
            target_manifest,
            expected_manifest_sha256=EXPECTED_TARGET_MANIFEST_SHA256,
        )
        audit["qualification_git_sha"] = source_sha
        audit["qualification_git_branch"] = branch
        _write_json_atomic(audit_path, audit)
        print("campaign_stage=TARGET_SUPERVISION_AUDIT_CREATED")

    print_audit_summary(audit)
    print(f"target_manifest_sha256={EXPECTED_TARGET_MANIFEST_SHA256}")
    print(f"audit={audit_path}")
    print(f"audit_sha256={sha256_file(audit_path)}")
    print("campaign_state=AWAITING_COMMITTED_TRAINING_GATE")
    print("training_started=false")
    print("sealed_blind_tile_payloads_consumed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
