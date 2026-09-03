from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from depthwizard.height_model.terrain_structure_campaign import (
    assess_tsd_training_authorization,
)
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
AUTHORIZATION_REL = Path("qualification/evidence/tsd-training-authorization-v1.json")
DEV_QUALIFICATION_REL = Path("qualification/evidence/tsd-dev-qualification-v1.json")
TRAINING_DIR_REL = Path("artifacts/training/tsd-urban-v1")
TRAINER_REL = Path("scripts/train_tsd_potsdam_v1.py")
DEV_QUALIFIER_REL = Path("qualification/qualify_tsd_dev_candidate.py")


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


def _print_authorization_summary(authorization: dict[str, object]) -> None:
    print("===== TSD TRAINING AUTHORIZATION =====")
    checks = authorization.get("checks")
    if not isinstance(checks, list):
        raise TypeError("training authorization checks must be a list")
    for check in checks:
        if not isinstance(check, dict):
            raise TypeError("training authorization check must be an object")
        print(
            f"{'PASS' if check['passed'] else 'FAIL'}: {check['name']} "
            f"actual={float(check['actual']):.6f} "
            f"required {check['operator']} {float(check['threshold']):.6f}"
        )
    print(f"training_authorized={str(bool(authorization['training_authorized'])).lower()}")


def _run_training(
    root: Path,
    *,
    target_manifest: Path,
    audit_path: Path,
    authorization_path: Path,
) -> None:
    trainer = (root / TRAINER_REL).resolve()
    if not trainer.is_file():
        raise FileNotFoundError(trainer)
    output_dir = (root / TRAINING_DIR_REL).resolve()
    print("campaign_stage=TSD_TRAINING_START_OR_RESUME", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(trainer),
            "--target-manifest",
            str(target_manifest),
            "--audit",
            str(audit_path),
            "--authorization",
            str(authorization_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=root,
        check=True,
    )
    report_path = output_dir / "training_report.json"
    if not report_path.is_file():
        raise RuntimeError("TSD trainer returned successfully without a training report")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "TRAINED_TSD_CANDIDATE_NOT_PROMOTED":
        raise RuntimeError("TSD trainer report is not in completed research-candidate state")
    if report.get("sealed_blind_tile_payloads_consumed") is not False:
        raise RuntimeError("TSD training report violated sealed-blind boundary")
    if report.get("production_promoted") is not False:
        raise RuntimeError("TSD training unexpectedly claims production promotion")
    print(f"training_report={report_path}")
    print(f"training_report_sha256={sha256_file(report_path)}")


def _run_dev_qualification(root: Path, *, target_manifest: Path) -> bool:
    qualifier = (root / DEV_QUALIFIER_REL).resolve()
    if not qualifier.is_file():
        raise FileNotFoundError(qualifier)
    training_dir = (root / TRAINING_DIR_REL).resolve()
    output = (root / DEV_QUALIFICATION_REL).resolve()
    print("campaign_stage=TSD_FROZEN_DEV_QUALIFICATION", flush=True)
    subprocess.run(
        [
            sys.executable,
            str(qualifier),
            "--target-manifest",
            str(target_manifest),
            "--training-dir",
            str(training_dir),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
    )
    if not output.is_file():
        raise RuntimeError("TSD dev qualifier returned without a qualification report")
    report = json.loads(output.read_text(encoding="utf-8"))
    if report.get("status") not in {"TSD_DEV_QUALIFICATION_PASS", "TSD_DEV_QUALIFICATION_FAIL"}:
        raise RuntimeError("unexpected TSD dev qualification status")
    if report.get("production_promoted") is not False:
        raise RuntimeError("dev qualification unexpectedly claims production promotion")
    if report.get("exposed_corrective_2_14_consumed") is not False:
        raise RuntimeError("dev qualification consumed exposed corrective 2_14")
    if report.get("external_evaluation_3_14_consumed") is not False:
        raise RuntimeError("dev qualification consumed external evaluation 3_14")
    if report.get("sealed_blind_tile_payloads_consumed") is not False:
        raise RuntimeError("dev qualification violated sealed-blind boundary")
    print(f"dev_qualification_report={output}")
    print(f"dev_qualification_report_sha256={sha256_file(output)}")
    return report.get("qualification_passed") is True


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
    authorization_path = (root / AUTHORIZATION_REL).resolve()

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
    audit_sha = sha256_file(audit_path)
    print(f"target_manifest_sha256={EXPECTED_TARGET_MANIFEST_SHA256}")
    print(f"audit={audit_path}")
    print(f"audit_sha256={audit_sha}")

    if authorization_path.exists():
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if authorization.get("target_manifest_sha256") != EXPECTED_TARGET_MANIFEST_SHA256:
            raise RuntimeError("existing TSD authorization is bound to a different target manifest")
        if authorization.get("audit_sha256") != audit_sha:
            raise RuntimeError("existing TSD authorization is bound to a different supervision audit")
        print("campaign_stage=TRAINING_AUTHORIZATION_ALREADY_COMPLETE")
    else:
        authorization = assess_tsd_training_authorization(audit)
        authorization["target_manifest_sha256"] = EXPECTED_TARGET_MANIFEST_SHA256
        authorization["audit"] = str(audit_path)
        authorization["audit_sha256"] = audit_sha
        authorization["qualification_git_sha"] = source_sha
        authorization["qualification_git_branch"] = branch
        _write_json_atomic(authorization_path, authorization)
        print("campaign_stage=TRAINING_AUTHORIZATION_CREATED")

    _print_authorization_summary(authorization)
    print(f"authorization={authorization_path}")
    print(f"authorization_sha256={sha256_file(authorization_path)}")
    if authorization.get("training_authorized") is not True:
        print("campaign_state=TRAINING_REJECTED_BY_DATA_GATE")
        print("training_started=false")
        print("sealed_blind_tile_payloads_consumed=false")
        return 2

    print("campaign_state=TRAINING_AUTHORIZED")
    print("sealed_blind_tile_payloads_consumed=false")
    _run_training(
        root,
        target_manifest=target_manifest,
        audit_path=audit_path,
        authorization_path=authorization_path,
    )
    print("campaign_state=TSD_TRAINED_CANDIDATE_AWAITING_QUALIFICATION")
    print("production_promoted=false")
    print("sealed_blind_tile_payloads_consumed=false")

    qualified = _run_dev_qualification(root, target_manifest=target_manifest)
    if not qualified:
        print("campaign_state=TSD_DEV_QUALIFICATION_FAILED")
        print("exposed_corrective_2_14_consumed=false")
        print("production_promoted=false")
        print("sealed_blind_tile_payloads_consumed=false")
        return 0

    print("campaign_state=TSD_DEV_QUALIFIED_AWAITING_EXPOSED_2_14")
    print("exposed_corrective_2_14_consumed=false")
    print("production_promoted=false")
    print("sealed_blind_tile_payloads_consumed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
