#!/usr/bin/env python3
"""Acquire two fresh official ISPRS Potsdam RGB/DSM pairs from the public ISPRS Seafile share.

The helper deliberately avoids the 13.3 GB full-share download when the server exposes the
standard product archives. It authenticates to the public ISPRS folder-share, discovers the
archive layout, downloads only the RGB and absolute-DSM archives, and extracts only the requested
unused tile pairs plus georeferencing sidecars.

Reference-safety boundary:
- ZIP directory metadata may be inspected.
- Reference DSM bytes may be downloaded/extracted.
- Reference DSM raster values are never decoded here.
- Reference DSM SHA-256 is intentionally not computed here.

The share URL and password are public dataset access credentials published by ISPRS at:
https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx
"""

from __future__ import annotations

import argparse
import http.cookiejar
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

SHARE_URL = "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/"
SHARE_TOKEN = "429be50cc79d423ab6c4"
# Public password rendered on the official ISPRS benchmark download page.
SHARE_PASSWORD = "CjwcipT4-P8g"
OFFICIAL_PAGE = "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx"
OFFICIAL_POTSDAM_PAGE = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"
)
CONSUMED = {"2_10", "3_13", "5_11", "6_14"}
DEFAULT_TILES = ("2_14", "3_14")
USER_AGENT = "DepthWizard-SIH26175-Potsdam-Acquisition/1"


class AcquisitionError(RuntimeError):
    """Official Potsdam acquisition failure."""


def _open(opener: urllib.request.OpenerDirector, request: urllib.request.Request, timeout: float = 120.0):
    try:
        return opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise AcquisitionError(f"HTTP {exc.code} for {request.full_url}: {body}") from exc
    except urllib.error.URLError as exc:
        raise AcquisitionError(f"request failed for {request.full_url}: {exc}") from exc


def _cookie_value(jar: http.cookiejar.CookieJar, name: str) -> str | None:
    for cookie in jar:
        if cookie.name == name:
            return cookie.value
    return None


def _authenticated_opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    opener.addheaders = [("User-Agent", USER_AGENT)]

    request = urllib.request.Request(SHARE_URL, headers={"User-Agent": USER_AGENT})
    with _open(opener, request) as response:
        response.read(4096)
    csrf = _cookie_value(jar, "sfcsrftoken")
    if not csrf:
        raise AcquisitionError("ISPRS Seafile share did not provide sfcsrftoken")

    form = urllib.parse.urlencode(
        {
            "csrfmiddlewaretoken": csrf,
            "token": SHARE_TOKEN,
            "password": SHARE_PASSWORD,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        SHARE_URL,
        data=form,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": SHARE_URL,
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with _open(opener, request) as response:
        response.read(4096)
    if not _cookie_value(jar, "sessionid"):
        raise AcquisitionError("ISPRS Seafile share authentication failed")
    return opener


def _list_dir(opener: urllib.request.OpenerDirector, path: str) -> list[dict[str, Any]]:
    api = (
        f"https://seafile.projekt.uni-hannover.de/api/v2.1/share-links/{SHARE_TOKEN}/dirents/?"
        + urllib.parse.urlencode({"path": path})
    )
    request = urllib.request.Request(
        api,
        headers={"User-Agent": USER_AGENT, "Referer": SHARE_URL, "Accept": "application/json"},
    )
    with _open(opener, request) as response:
        raw = response.read()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise AcquisitionError(f"Seafile directory API returned non-JSON for {path!r}") from exc
    if not isinstance(payload, dict):
        raise AcquisitionError(f"Seafile directory API returned invalid root for {path!r}")
    entries = payload.get("dirent_list")
    if not isinstance(entries, list):
        raise AcquisitionError(f"Seafile directory API is missing dirent_list for {path!r}")
    return [entry for entry in entries if isinstance(entry, dict)]


def _entry_name(entry: dict[str, Any]) -> str:
    for key in ("name", "obj_name"):
        value = entry.get(key)
        if isinstance(value, str) and value:
            return value
    file_path = entry.get("file_path")
    if isinstance(file_path, str) and file_path:
        return PurePosixPath(file_path).name
    return ""


def _entry_path(entry: dict[str, Any], parent: str) -> str:
    value = entry.get("file_path")
    if isinstance(value, str) and value:
        return value if value.startswith("/") else "/" + value
    name = _entry_name(entry)
    if not name:
        raise AcquisitionError(f"Seafile entry has no path/name: {entry!r}")
    if parent == "/":
        return "/" + name
    return parent.rstrip("/") + "/" + name


def _is_dir(entry: dict[str, Any]) -> bool:
    value = entry.get("type")
    if isinstance(value, str):
        return value.lower() in {"dir", "directory"}
    return bool(entry.get("is_dir") is True)


def _walk_files(opener: urllib.request.OpenerDirector, path: str = "/") -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    stack = [path]
    visited: set[str] = set()
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        for entry in _list_dir(opener, current):
            resolved = _entry_path(entry, current)
            if _is_dir(entry):
                stack.append(resolved)
            else:
                copied = dict(entry)
                copied["_resolved_path"] = resolved
                files.append(copied)
    return files


def _archive_size(entry: dict[str, Any]) -> int | None:
    for key in ("size", "file_size"):
        value = entry.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _select_archives(files: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    zips = [entry for entry in files if _entry_name(entry).lower().endswith(".zip")]
    rgb = [entry for entry in zips if _entry_name(entry).casefold() == "2_ortho_rgb.zip"]
    if len(rgb) != 1:
        names = sorted(_entry_name(entry) for entry in zips)
        raise AcquisitionError(
            "expected exactly one official 2_Ortho_RGB.zip in the ISPRS share; "
            f"found={len(rgb)}; zip_files={names}"
        )

    def dsm_rank(entry: dict[str, Any]) -> tuple[int, str]:
        name = _entry_name(entry).casefold()
        if "dsm" not in name:
            return (99, name)
        if any(term in name for term in ("normal", "ndsm", "label")):
            return (98, name)
        if name in {"1_dsm.zip", "dsm.zip", "1-dsm.zip"}:
            return (0, name)
        if name.startswith("1_dsm"):
            return (1, name)
        return (2, name)

    dsm_candidates = sorted((entry for entry in zips if dsm_rank(entry)[0] < 98), key=dsm_rank)
    if not dsm_candidates:
        names = sorted(_entry_name(entry) for entry in zips)
        raise AcquisitionError(f"could not locate an absolute DSM archive; zip_files={names}")
    return rgb[0], dsm_candidates[0]


def _download_share_file(
    opener: urllib.request.OpenerDirector,
    entry: dict[str, Any],
    destination: Path,
) -> Path:
    remote_path = str(entry.get("_resolved_path") or "")
    if not remote_path:
        raise AcquisitionError("share file entry is missing resolved path")
    expected_size = _archive_size(entry)
    if destination.is_file() and destination.stat().st_size > 0:
        if expected_size is None or destination.stat().st_size == expected_size:
            print(f"Reusing cached archive: {destination.name}")
            return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    url = SHARE_URL + "files/?" + urllib.parse.urlencode({"p": remote_path, "dl": 1})
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Referer": SHARE_URL},
    )
    print(
        f"Downloading official ISPRS archive: {destination.name}"
        + (f" ({expected_size / (1024**3):.2f} GiB)" if expected_size else "")
    )
    try:
        with _open(opener, request, timeout=300.0) as response, temporary.open("wb") as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if not temporary.is_file() or temporary.stat().st_size <= 0:
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(f"empty download for {remote_path}")
    if expected_size is not None and temporary.stat().st_size != expected_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(
            f"archive size mismatch for {remote_path}: expected={expected_size}, actual={actual}"
        )
    temporary.replace(destination)
    return destination


def _tile_names(tile_id: str) -> dict[str, set[str]]:
    row, col = tile_id.split("_")
    rgb_stem = f"top_potsdam_{int(row)}_{int(col)}_RGB"
    dsm_stems = {
        f"dsm_potsdam_{int(row):02d}_{int(col):02d}",
        f"dsm_potsdam_{int(row)}_{int(col)}",
    }
    return {
        "rgb": {f"{rgb_stem}.tif", f"{rgb_stem}.tiff", f"{rgb_stem}.tfw"},
        "dsm": {
            *(f"{stem}.tif" for stem in dsm_stems),
            *(f"{stem}.tiff" for stem in dsm_stems),
            *(f"{stem}.tfw" for stem in dsm_stems),
        },
    }


def _extract_members(archive: Path, wanted: set[str], destination: Path, role: str) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        by_basename: dict[str, list[str]] = {}
        for name in zf.namelist():
            base = PurePosixPath(name).name
            if base in wanted:
                by_basename.setdefault(base, []).append(name)
        for base, members in sorted(by_basename.items()):
            if len(members) != 1:
                raise AcquisitionError(f"{role}: ambiguous archive member {base}: {members}")
            target = destination / base
            with zf.open(members[0]) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted.append(str(target.resolve()))
    tif_count = sum(Path(path).suffix.lower() in {".tif", ".tiff"} for path in extracted)
    if tif_count != 1:
        raise AcquisitionError(
            f"{role}: expected exactly one TIFF/TIFF member for requested tile, extracted={extracted}"
        )
    return extracted


def _validate_tile_id(tile_id: str) -> str:
    parts = tile_id.split("_")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"invalid Potsdam tile id: {tile_id!r}")
    normalized = f"{int(parts[0])}_{int(parts[1])}"
    if normalized in CONSUMED:
        raise argparse.ArgumentTypeError(f"tile {normalized} is already consumed and cannot be reused")
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download only the official Potsdam RGB/DSM archives needed for two fresh SIH26175 tiles."
    )
    parser.add_argument(
        "--tile",
        action="append",
        type=_validate_tile_id,
        help="Fresh tile id; repeat twice. Defaults to 2_14 and 3_14.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/external/isprs-potsdam"),
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("workspace/final-science-data/potsdam-official-cache"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("workspace/final-science-data/potsdam-acquisition.json"),
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Authenticate and report selected official archives without downloading them.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tiles = tuple(dict.fromkeys(args.tile or DEFAULT_TILES))
    if len(tiles) != 2:
        raise SystemExit("exactly two distinct fresh Potsdam tile ids are required")

    opener = _authenticated_opener()
    files = _walk_files(opener)
    rgb_entry, dsm_entry = _select_archives(files)
    inventory = {
        "rgb_archive": {
            "name": _entry_name(rgb_entry),
            "remote_path": rgb_entry.get("_resolved_path"),
            "bytes": _archive_size(rgb_entry),
        },
        "dsm_archive": {
            "name": _entry_name(dsm_entry),
            "remote_path": dsm_entry.get("_resolved_path"),
            "bytes": _archive_size(dsm_entry),
        },
    }
    if args.inventory_only:
        print(json.dumps({"status": "PASS_POTSDAM_OFFICIAL_ARCHIVE_INVENTORY", "archives": inventory}, indent=2))
        return 0

    rgb_archive = _download_share_file(
        opener,
        rgb_entry,
        args.cache_root / _entry_name(rgb_entry),
    )
    dsm_archive = _download_share_file(
        opener,
        dsm_entry,
        args.cache_root / _entry_name(dsm_entry),
    )

    extracted: dict[str, Any] = {}
    for tile_id in tiles:
        wanted = _tile_names(tile_id)
        rgb_paths = _extract_members(rgb_archive, wanted["rgb"], args.dataset_root, f"RGB {tile_id}")
        dsm_paths = _extract_members(dsm_archive, wanted["dsm"], args.dataset_root, f"DSM {tile_id}")
        extracted[tile_id] = {
            "rgb_files": rgb_paths,
            "dsm_files": dsm_paths,
            "reference_values_decoded_or_hashed": False,
        }

    report = {
        "schema_version": 1,
        "status": "PASS_POTSDAM_OFFICIAL_FRESH_PAIR_ACQUISITION",
        "official_page": OFFICIAL_PAGE,
        "official_potsdam_page": OFFICIAL_POTSDAM_PAGE,
        "share_url": SHARE_URL,
        "selected_tiles": list(tiles),
        "consumed_tiles_excluded": sorted(CONSUMED),
        "archives": inventory,
        "extracted": extracted,
        "reference_values_decoded_or_hashed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
