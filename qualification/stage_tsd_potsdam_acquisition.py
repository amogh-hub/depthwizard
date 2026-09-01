from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from depthwizard.height_model.terrain_structure_split import (
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    initial_tsd_campaign_split,
)

ISPRS_DATASET_AUTHORITY_PAGE: Final = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"
)
ISPRS_DOWNLOAD_INDEX_PAGE: Final = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx"
)
ISPRS_POTSDAM_SHARE: Final = "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/"

CANONICAL_ARCHIVE_BASENAMES: Final[dict[str, str]] = {
    "rgb": "2_Ortho_RGB.zip",
    "dsm": "1_DSM.zip",
    "label": "5_Labels_for_participants.zip",
}
DESTINATION_SUBDIRECTORIES: Final[dict[str, str]] = {
    "rgb": "2_Ortho_RGB",
    "dsm": "1_DSM",
    "label": "5_Labels_for_participants",
}
CHUNK_BYTES: Final = 1024 * 1024
MAX_SELECTED_MEMBER_BYTES: Final = 1024 * 1024 * 1024
MAX_SELECTED_TOTAL_BYTES: Final = 8 * 1024 * 1024 * 1024
MIN_FREE_SPACE_MARGIN_BYTES: Final = 512 * 1024 * 1024


@dataclass(frozen=True)
class PlannedFile:
    tile_id: str
    role: str
    component: str
    expected_filename: str


@dataclass(frozen=True)
class SelectedArchiveMember:
    component: str
    tile_id: str
    role: str
    archive_path: str
    archive_member: str
    expected_filename: str
    compressed_bytes: int
    uncompressed_bytes: int
    crc32_hex: str


@dataclass(frozen=True)
class StagedFile:
    component: str
    tile_id: str
    role: str
    source_archive: str
    source_member: str
    destination: str
    bytes_written: int
    sha256: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and selectively stage only the missing files for the frozen first Potsdam TSD "
            "campaign. ZIP metadata is inspected for all supplied archives, but only active planned "
            "members may be copied. Raster pixels are never decoded by this command."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--rgb-archive", type=Path, required=True)
    parser.add_argument("--dsm-archive", type=Path, required=True)
    parser.add_argument("--label-archive", type=Path, required=True)
    parser.add_argument("--provenance-output", type=Path, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "Actually copy the selected active members. Without --execute the command performs a "
            "metadata-only archive validation/dry-run and writes a validation provenance record."
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _expected_filename(tile_id: str, component: str) -> str:
    row_text, col_text = tile_id.split("_")
    row = int(row_text)
    col = int(col_text)
    if component == "rgb":
        return f"top_potsdam_{row}_{col}_RGB.tif"
    if component == "dsm":
        return f"dsm_potsdam_{row:02d}_{col:02d}.tif"
    if component == "label":
        return f"top_potsdam_{row}_{col}_label.tif"
    raise ValueError(f"unsupported component: {component}")


def _load_plan(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("TSD acquisition plan must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported acquisition-plan schema: {payload.get('schema_version')!r}")
    if payload.get("split_protocol_version") != TSD_SPLIT_PROTOCOL_VERSION:
        raise ValueError("acquisition-plan split protocol does not match current TSD split protocol")
    if payload.get("campaign_protocol_version") != INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION:
        raise ValueError("acquisition plan is not for the frozen first urban TSD campaign")
    if payload.get("train_tile_ids") != list(INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS):
        raise ValueError("acquisition-plan training tiles do not match the frozen campaign")
    if payload.get("dev_tile_ids") != list(INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS):
        raise ValueError("acquisition-plan development tiles do not match the frozen campaign")
    if payload.get("active_tile_count") != len(initial_tsd_campaign_split().supervised_tile_ids):
        raise ValueError("acquisition-plan active tile count is inconsistent with the frozen campaign")
    _planned_files(payload)
    return payload


def _planned_files(payload: dict[str, Any]) -> tuple[PlannedFile, ...]:
    raw_files = payload.get("required_files")
    if not isinstance(raw_files, list):
        raise TypeError("acquisition plan required_files must be a JSON array")

    split = initial_tsd_campaign_split()
    active = set(split.supervised_tile_ids)
    train = set(split.train_tile_ids)
    seen: set[tuple[str, str]] = set()
    files: list[PlannedFile] = []

    for raw in raw_files:
        if not isinstance(raw, dict):
            raise TypeError("acquisition plan contains a malformed required-file record")
        tile_id = raw.get("tile_id")
        role = raw.get("role")
        component = raw.get("component")
        expected = raw.get("expected_filename")
        if not all(isinstance(value, str) for value in (tile_id, role, component, expected)):
            raise TypeError("acquisition plan required-file fields must be strings")
        assert isinstance(tile_id, str)
        assert isinstance(role, str)
        assert isinstance(component, str)
        assert isinstance(expected, str)
        if tile_id not in active:
            raise ValueError(f"acquisition plan contains non-active tile {tile_id}")
        expected_role = "train" if tile_id in train else "dev"
        if role != expected_role:
            raise ValueError(
                f"acquisition plan role mismatch for {tile_id}: expected {expected_role}, got {role}"
            )
        if component not in CANONICAL_ARCHIVE_BASENAMES:
            raise ValueError(f"unsupported acquisition component {component!r}")
        canonical = _expected_filename(tile_id, component)
        if expected != canonical:
            raise ValueError(
                f"acquisition-plan filename mismatch for {tile_id}/{component}: "
                f"expected {canonical}, got {expected}"
            )
        key = (tile_id, component)
        if key in seen:
            raise ValueError(f"duplicate acquisition requirement for {tile_id}/{component}")
        seen.add(key)
        files.append(
            PlannedFile(
                tile_id=tile_id,
                role=role,
                component=component,
                expected_filename=expected,
            )
        )

    for component in CANONICAL_ARCHIVE_BASENAMES:
        expected_tiles = payload.get(f"missing_{component}_tile_ids")
        expected_count = payload.get(f"missing_{component}_count")
        actual_tiles = [item.tile_id for item in files if item.component == component]
        if expected_tiles != actual_tiles or expected_count != len(actual_tiles):
            raise ValueError(
                f"acquisition-plan {component} summary does not match required_files records"
            )
    return tuple(files)


def _tracked_source_identity() -> tuple[str, str]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short", "--untracked-files=no"],
        cwd=CODE_ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if status:
        raise RuntimeError(
            "refusing acquisition staging from a tracked-dirty worktree; commit or restore changes first"
        )
    if not head:
        raise RuntimeError("could not resolve qualification source commit")
    return head, branch


def _validate_archive_basename(path: Path, component: str) -> None:
    canonical = CANONICAL_ARCHIVE_BASENAMES[component]
    if path.name.casefold() != canonical.casefold():
        raise ValueError(
            f"{component} archive must be the canonical ISPRS package {canonical!r}; got {path.name!r}"
        )
    if not path.is_file():
        raise FileNotFoundError(path)
    if not zipfile.is_zipfile(path):
        raise ValueError(f"not a readable ZIP archive: {path}")


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    raw_name = info.filename.replace("\\", "/")
    if "\x00" in raw_name:
        raise ValueError("ZIP member contains NUL byte")
    pure = PurePosixPath(raw_name)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(f"refusing symbolic-link ZIP member: {info.filename!r}")
    return raw_name


def _archive_index(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    index: dict[str, list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        normalized = _safe_member_name(info)
        if info.is_dir():
            continue
        basename = PurePosixPath(normalized).name.casefold()
        index.setdefault(basename, []).append(info)
    return index


def _assert_participant_label_archive_only(index: dict[str, list[zipfile.ZipInfo]]) -> None:
    forbidden_present: list[str] = []
    for tile_id in sorted(HISTORICAL_CHALLENGE_TEST_TILE_IDS):
        forbidden_name = _expected_filename(tile_id, "label").casefold()
        if forbidden_name in index:
            forbidden_present.append(tile_id)
    if forbidden_present:
        raise RuntimeError(
            "refusing label archive containing historical challenge-test labels; use the historical "
            "5_Labels_for_participants.zip package instead. forbidden_tiles="
            + ",".join(forbidden_present)
        )


def _select_archive_members(
    *,
    archive_path: Path,
    component: str,
    planned_files: tuple[PlannedFile, ...],
) -> tuple[SelectedArchiveMember, ...]:
    _validate_archive_basename(archive_path, component)
    selected: list[SelectedArchiveMember] = []
    with zipfile.ZipFile(archive_path, "r") as archive:
        index = _archive_index(archive)
        if component == "label":
            _assert_participant_label_archive_only(index)
        for planned in planned_files:
            if planned.component != component:
                continue
            matches = index.get(planned.expected_filename.casefold(), [])
            if not matches:
                raise FileNotFoundError(
                    f"{archive_path.name} does not contain required {planned.expected_filename}"
                )
            if len(matches) != 1:
                raise RuntimeError(
                    f"{archive_path.name} contains ambiguous duplicate member "
                    f"{planned.expected_filename}: {len(matches)} copies"
                )
            info = matches[0]
            if info.file_size <= 0:
                raise ValueError(
                    f"required archive member has non-positive size: {archive_path.name}:{info.filename}"
                )
            if info.file_size > MAX_SELECTED_MEMBER_BYTES:
                raise ValueError(
                    f"required archive member exceeds safety limit: {archive_path.name}:{info.filename}"
                )
            selected.append(
                SelectedArchiveMember(
                    component=component,
                    tile_id=planned.tile_id,
                    role=planned.role,
                    archive_path=str(archive_path.resolve()),
                    archive_member=info.filename,
                    expected_filename=planned.expected_filename,
                    compressed_bytes=info.compress_size,
                    uncompressed_bytes=info.file_size,
                    crc32_hex=f"{info.CRC:08x}",
                )
            )
    return tuple(selected)


def validate_acquisition_archives(
    *,
    payload: dict[str, Any],
    rgb_archive: Path,
    dsm_archive: Path,
    label_archive: Path,
) -> tuple[SelectedArchiveMember, ...]:
    planned_files = _planned_files(payload)
    archives = {
        "rgb": rgb_archive,
        "dsm": dsm_archive,
        "label": label_archive,
    }
    selected: list[SelectedArchiveMember] = []
    for component, path in archives.items():
        selected.extend(
            _select_archive_members(
                archive_path=path,
                component=component,
                planned_files=planned_files,
            )
        )

    expected_keys = {(item.tile_id, item.component) for item in planned_files}
    selected_keys = {(item.tile_id, item.component) for item in selected}
    if selected_keys != expected_keys or len(selected) != len(planned_files):
        raise RuntimeError("selected archive members do not exactly match the frozen acquisition plan")
    total_uncompressed = sum(item.uncompressed_bytes for item in selected)
    if total_uncompressed > MAX_SELECTED_TOTAL_BYTES:
        raise ValueError(
            f"selected acquisition payload exceeds safety limit: {total_uncompressed} bytes"
        )
    return tuple(selected)


def _dataset_filename_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name.casefold(), []).append(path)
    return index


def _assert_plan_not_stale(dataset_root: Path, selected: tuple[SelectedArchiveMember, ...]) -> None:
    index = _dataset_filename_index(dataset_root)
    stale: list[str] = []
    for item in selected:
        if index.get(item.expected_filename.casefold()):
            stale.append(item.expected_filename)
    if stale:
        raise RuntimeError(
            "refusing to stage from stale acquisition plan because files now exist locally: "
            + ",".join(sorted(stale))
        )


def _destination(dataset_root: Path, item: SelectedArchiveMember) -> Path:
    return dataset_root / DESTINATION_SUBDIRECTORIES[item.component] / item.expected_filename


def _assert_free_space(dataset_root: Path, selected: tuple[SelectedArchiveMember, ...]) -> None:
    required = sum(item.uncompressed_bytes for item in selected) + MIN_FREE_SPACE_MARGIN_BYTES
    free = shutil.disk_usage(dataset_root).free
    if free < required:
        raise RuntimeError(
            f"insufficient free disk space for staged acquisition: required>={required}, free={free}"
        )


def _copy_selected_member(
    *,
    archive_path: Path,
    member_name: str,
    destination: Path,
) -> tuple[int, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".partial",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(fd, "wb") as output, zipfile.ZipFile(archive_path, "r") as archive:
            with archive.open(member_name, "r") as source:
                while True:
                    chunk = source.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    output.write(chunk)
                    digest.update(chunk)
                    count += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return count, digest.hexdigest()


def stage_selected_members(
    *,
    dataset_root: Path,
    selected: tuple[SelectedArchiveMember, ...],
    archive_paths: dict[str, Path],
) -> tuple[StagedFile, ...]:
    _assert_plan_not_stale(dataset_root, selected)
    _assert_free_space(dataset_root, selected)
    staged: list[StagedFile] = []
    created: list[Path] = []
    try:
        for item in selected:
            destination = _destination(dataset_root, item)
            count, sha256 = _copy_selected_member(
                archive_path=archive_paths[item.component],
                member_name=item.archive_member,
                destination=destination,
            )
            if count != item.uncompressed_bytes:
                raise RuntimeError(
                    f"extracted size mismatch for {item.expected_filename}: "
                    f"expected {item.uncompressed_bytes}, got {count}"
                )
            created.append(destination)
            staged.append(
                StagedFile(
                    component=item.component,
                    tile_id=item.tile_id,
                    role=item.role,
                    source_archive=item.archive_path,
                    source_member=item.archive_member,
                    destination=str(destination.resolve()),
                    bytes_written=count,
                    sha256=sha256,
                )
            )
    except BaseException:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return tuple(staged)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    payload = _load_plan(args.plan)
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)

    plan_root = payload.get("source_inventory_dataset_root")
    if not isinstance(plan_root, str):
        raise ValueError("acquisition plan does not contain a valid source_inventory_dataset_root")
    if Path(plan_root).resolve() != args.dataset_root.resolve():
        raise ValueError(
            "dataset root differs from the root used to create the frozen acquisition plan: "
            f"plan={Path(plan_root).resolve()}, requested={args.dataset_root.resolve()}"
        )

    source_sha, source_branch = _tracked_source_identity()
    plan_sha256 = _sha256_file(args.plan)
    archive_paths = {
        "rgb": args.rgb_archive,
        "dsm": args.dsm_archive,
        "label": args.label_archive,
    }
    selected = validate_acquisition_archives(
        payload=payload,
        rgb_archive=args.rgb_archive,
        dsm_archive=args.dsm_archive,
        label_archive=args.label_archive,
    )
    _assert_plan_not_stale(args.dataset_root, selected)

    archive_records = {
        component: {
            "path": str(path.resolve()),
            "basename": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for component, path in archive_paths.items()
    }

    staged: tuple[StagedFile, ...] = ()
    if args.execute:
        staged = stage_selected_members(
            dataset_root=args.dataset_root,
            selected=selected,
            archive_paths=archive_paths,
        )

    provenance = {
        "schema_version": 1,
        "status": "STAGED_ACTIVE_COMPONENTS" if args.execute else "VALIDATED_DRY_RUN",
        "split_protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "campaign_protocol_version": INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
        "qualification_git_sha": source_sha,
        "qualification_git_branch": source_branch,
        "dataset_root": str(args.dataset_root.resolve()),
        "acquisition_plan": str(args.plan.resolve()),
        "acquisition_plan_sha256": plan_sha256,
        "official_source": {
            "dataset_authority_page": ISPRS_DATASET_AUTHORITY_PAGE,
            "download_index_page": ISPRS_DOWNLOAD_INDEX_PAGE,
            "potsdam_share": ISPRS_POTSDAM_SHARE,
            "origin_verification_boundary": (
                "The command records exact local archive SHA-256 values and requires canonical package "
                "names/content structure, but ISPRS does not provide a remote SHA-256 through this "
                "workflow. The operator remains responsible for obtaining the archives from the "
                "recorded official ISPRS/Leibniz Hannover share."
            ),
        },
        "archives": archive_records,
        "selected_members": [asdict(item) for item in selected],
        "selected_member_count": len(selected),
        "selected_uncompressed_bytes": sum(item.uncompressed_bytes for item in selected),
        "staged_files": [asdict(item) for item in staged],
        "staged_file_count": len(staged),
        "historical_challenge_test_label_archive_guard": "PASS",
        "raster_pixels_decoded": False,
        "nonactive_raster_payload_members_copied": False,
        "claim_boundary": (
            "Only files already listed as missing by the frozen potsdam-tsd-urban-spatial-v1 "
            "acquisition plan may be staged. No buffer, exposed, external-evaluation, sealed-blind, "
            "or historical challenge-test raster payload is copied. ZIP central-directory metadata "
            "may enumerate package members, and selected active raster bytes are copied and hashed "
            "when --execute is used, but raster pixels are never decoded or inspected by this command."
        ),
    }
    _write_json_atomic(args.provenance_output, provenance)

    print(f"status={provenance['status']}")
    print(f"qualification_git_sha={source_sha}")
    print(f"campaign_protocol_version={INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION}")
    print(f"acquisition_plan_sha256={plan_sha256}")
    for component in ("rgb", "dsm", "label"):
        record = archive_records[component]
        print(f"{component}_archive={record['path']}")
        print(f"{component}_archive_sha256={record['sha256']}")
    print(f"selected_member_count={len(selected)}")
    print(f"selected_uncompressed_bytes={sum(item.uncompressed_bytes for item in selected)}")
    print(f"staged_file_count={len(staged)}")
    print("historical_challenge_test_label_archive_guard=PASS")
    print("nonactive_raster_payload_members_copied=false")
    print("raster_pixels_decoded=false")
    print(f"provenance={args.provenance_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
