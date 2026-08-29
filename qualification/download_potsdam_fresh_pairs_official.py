#!/usr/bin/env python3
"""Acquire two fresh official ISPRS Potsdam RGB/DSM pairs from the public file share.

ISPRS publishes Potsdam as a password-protected Seafile *file* share (~13.3 GB), not a folder
share. This helper therefore authenticates to the file share, probes byte-range support, and when
possible treats the remote object as a seekable ZIP using HTTP Range requests. That lets us inspect
the ZIP central directory without downloading the whole archive and then either:

1. extract the requested RGB/DSM members directly from the remote ZIP, or
2. if the outer ZIP contains product ZIPs (for example 2_Ortho_RGB.zip), download only the
   relevant RGB + absolute-DSM inner archives and extract the requested tiles locally.

Reference-safety boundary:
- ZIP directory metadata may be inspected.
- Reference DSM bytes may be downloaded/extracted.
- Reference DSM raster values are never decoded here.
- Reference DSM SHA-256 is intentionally not computed here.

The share URL and public password are published by ISPRS at:
https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx
"""

from __future__ import annotations

import argparse
import http.cookiejar
import io
import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SHARE_URL = "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/"
DOWNLOAD_URL = SHARE_URL + "?dl=1"
SHARE_TOKEN = "429be50cc79d423ab6c4"
SHARE_PASSWORD = "CjwcipT4-P8g"  # public password shown by the official ISPRS page
OFFICIAL_PAGE = "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx"
OFFICIAL_POTSDAM_PAGE = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"
)
CONSUMED = {"2_10", "3_13", "5_11", "6_14"}
DEFAULT_TILES = ("2_14", "3_14")
USER_AGENT = "DepthWizard-SIH26175-Potsdam-Acquisition/2"
BLOCK_SIZE = 8 * 1024 * 1024
MAX_CACHE_BLOCKS = 4


class AcquisitionError(RuntimeError):
    """Official Potsdam acquisition failure."""


def _open(
    opener: urllib.request.OpenerDirector,
    request: urllib.request.Request,
    timeout: float = 120.0,
):
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


def _filename_from_disposition(value: str | None) -> str | None:
    if not value:
        return None
    match = re.search(r"filename\*=UTF-8''([^;]+)", value, flags=re.I)
    if match:
        return PurePosixPath(urllib.parse.unquote(match.group(1).strip())).name
    match = re.search(r'filename="?([^";]+)"?', value, flags=re.I)
    if match:
        return PurePosixPath(match.group(1).strip()).name
    return None


def _total_from_content_range(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+|\*)", value.strip(), flags=re.I)
    if not match or match.group(1) == "*":
        return None
    return int(match.group(1))


def _probe_file_share(opener: urllib.request.OpenerDirector) -> dict[str, Any]:
    request = urllib.request.Request(
        DOWNLOAD_URL,
        headers={
            "User-Agent": USER_AGENT,
            "Referer": SHARE_URL,
            "Range": "bytes=0-0",
            "Accept-Encoding": "identity",
        },
    )
    with _open(opener, request) as response:
        status = int(response.getcode())
        headers = response.headers
        final_url = response.geturl()
        # Never consume a 13.3 GB body when the server ignores Range. One byte is enough for 206.
        if status == 206:
            response.read(1)
        content_range = headers.get("Content-Range")
        total = _total_from_content_range(content_range)
        content_length = headers.get("Content-Length")
        if total is None and content_length and status == 200:
            try:
                total = int(content_length)
            except ValueError:
                total = None
        filename = _filename_from_disposition(headers.get("Content-Disposition"))
        if filename is None:
            filename = PurePosixPath(urllib.parse.urlparse(final_url).path).name or None
        range_supported = status == 206 and total is not None
        return {
            "share_kind": "file",
            "status_code": status,
            "filename": filename,
            "bytes": total,
            "range_supported": range_supported,
            "content_range": content_range,
            "accept_ranges": headers.get("Accept-Ranges"),
            "content_type": headers.get("Content-Type"),
            "content_disposition": headers.get("Content-Disposition"),
            "final_host": urllib.parse.urlparse(final_url).netloc,
        }


class HTTPRangeReader(io.RawIOBase):
    """Seekable read-only view over the authenticated Seafile file share using byte ranges."""

    def __init__(
        self,
        opener: urllib.request.OpenerDirector,
        size: int,
        *,
        block_size: int = BLOCK_SIZE,
        max_cache_blocks: int = MAX_CACHE_BLOCKS,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("remote size must be positive")
        self._opener = opener
        self._size = size
        self._pos = 0
        self._block_size = block_size
        self._max_cache_blocks = max_cache_blocks
        self._cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError("negative seek position")
        self._pos = min(new_pos, self._size)
        return self._pos

    def _fetch_block(self, block_index: int) -> bytes:
        cached = self._cache.get(block_index)
        if cached is not None:
            self._cache.move_to_end(block_index)
            return cached
        start = block_index * self._block_size
        if start >= self._size:
            return b""
        end = min(self._size - 1, start + self._block_size - 1)
        request = urllib.request.Request(
            DOWNLOAD_URL,
            headers={
                "User-Agent": USER_AGENT,
                "Referer": SHARE_URL,
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
        )
        with _open(self._opener, request, timeout=300.0) as response:
            if int(response.getcode()) != 206:
                raise AcquisitionError(
                    "ISPRS fileserver stopped honoring HTTP byte ranges; refusing full-share transfer"
                )
            content_range = response.headers.get("Content-Range")
            expected_prefix = f"bytes {start}-{end}/"
            if not content_range or not content_range.lower().startswith(expected_prefix.lower()):
                raise AcquisitionError(
                    f"unexpected Content-Range for block {block_index}: {content_range!r}"
                )
            data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise AcquisitionError(
                f"short ranged read: start={start} end={end} expected={expected} actual={len(data)}"
            )
        self._cache[block_index] = data
        self._cache.move_to_end(block_index)
        while len(self._cache) > self._max_cache_blocks:
            self._cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self._pos >= self._size:
            return b""
        if size is None or size < 0:
            size = self._size - self._pos
        size = min(size, self._size - self._pos)
        if size <= 0:
            return b""
        output = bytearray()
        remaining = size
        while remaining > 0:
            block_index = self._pos // self._block_size
            block_offset = self._pos % self._block_size
            block = self._fetch_block(block_index)
            take = min(remaining, len(block) - block_offset)
            if take <= 0:
                break
            output.extend(block[block_offset : block_offset + take])
            self._pos += take
            remaining -= take
        return bytes(output)

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)



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


def _member_index(zf: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    index: dict[str, list[zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = PurePosixPath(info.filename).name.casefold()
        index.setdefault(base, []).append(info)
    return index


def _find_direct_members(
    zf: zipfile.ZipFile,
    tiles: tuple[str, ...],
) -> dict[str, dict[str, list[zipfile.ZipInfo]]]:
    index = _member_index(zf)
    result: dict[str, dict[str, list[zipfile.ZipInfo]]] = {}
    for tile_id in tiles:
        wanted = _tile_names(tile_id)
        roles: dict[str, list[zipfile.ZipInfo]] = {}
        for role in ("rgb", "dsm"):
            matches: list[zipfile.ZipInfo] = []
            for basename in wanted[role]:
                matches.extend(index.get(basename.casefold(), []))
            roles[role] = matches
        result[tile_id] = roles
    return result


def _nested_archive_candidates(zf: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    rgb: list[zipfile.ZipInfo] = []
    dsm: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if info.is_dir() or not info.filename.casefold().endswith(".zip"):
            continue
        base = PurePosixPath(info.filename).name.casefold()
        if base == "2_ortho_rgb.zip" or ("ortho" in base and "rgb" in base):
            rgb.append(info)
        if "dsm" in base and not any(term in base for term in ("ndsm", "normal", "label")):
            dsm.append(info)
    rgb.sort(key=lambda info: (0 if PurePosixPath(info.filename).name.casefold() == "2_ortho_rgb.zip" else 1, info.filename))
    dsm.sort(key=lambda info: (0 if PurePosixPath(info.filename).name.casefold() in {"1_dsm.zip", "dsm.zip"} else 1, info.filename))
    return {"rgb": rgb, "dsm": dsm}


def _info_summary(info: zipfile.ZipInfo) -> dict[str, Any]:
    return {
        "name": info.filename,
        "uncompressed_bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "compress_type": info.compress_type,
    }


def _extract_selected_from_zip(
    zf: zipfile.ZipFile,
    wanted: set[str],
    destination: Path,
    role: str,
) -> list[str]:
    destination.mkdir(parents=True, exist_ok=True)
    wanted_cf = {name.casefold() for name in wanted}
    matches: dict[str, list[zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        base = PurePosixPath(info.filename).name
        if base.casefold() in wanted_cf:
            matches.setdefault(base.casefold(), []).append(info)
    extracted: list[str] = []
    for _, infos in sorted(matches.items()):
        if len(infos) != 1:
            raise AcquisitionError(f"{role}: ambiguous archive member(s): {[i.filename for i in infos]}")
        info = infos[0]
        base = PurePosixPath(info.filename).name
        target = destination / base
        with zf.open(info) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
        extracted.append(str(target.resolve()))
    tif_count = sum(Path(path).suffix.lower() in {".tif", ".tiff"} for path in extracted)
    if tif_count != 1:
        raise AcquisitionError(
            f"{role}: expected exactly one TIFF/TIFF for requested tile, extracted={extracted}"
        )
    return extracted


def _download_zip_member(
    outer: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> Path:
    if destination.is_file() and destination.stat().st_size == info.file_size:
        print(f"Reusing cached inner archive: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    print(
        f"Downloading only inner archive {PurePosixPath(info.filename).name}: "
        f"{info.file_size / (1024**3):.2f} GiB uncompressed"
    )
    try:
        with outer.open(info) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if temporary.stat().st_size != info.file_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(
            f"inner archive size mismatch for {info.filename}: expected={info.file_size}, actual={actual}"
        )
    temporary.replace(destination)
    return destination


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
        description="Acquire only two fresh official Potsdam RGB/DSM pairs from the ISPRS file share."
    )
    parser.add_argument(
        "--tile",
        action="append",
        type=_validate_tile_id,
        help="Fresh tile id; repeat twice. Defaults to 2_14 and 3_14.",
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/external/isprs-potsdam"))
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
        help="Probe file/range/ZIP metadata only. Does not download the full shared file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tiles = tuple(dict.fromkeys(args.tile or DEFAULT_TILES))
    if len(tiles) != 2:
        raise SystemExit("exactly two distinct fresh Potsdam tile ids are required")

    opener = _authenticated_opener()
    probe = _probe_file_share(opener)
    inventory: dict[str, Any] = {
        "official_page": OFFICIAL_PAGE,
        "share_url": SHARE_URL,
        "file_share": probe,
        "selected_tiles": list(tiles),
        "consumed_tiles_excluded": sorted(CONSUMED),
    }

    if not probe.get("range_supported"):
        payload = {
            "status": "BLOCKED_POTSDAM_FILE_SHARE_NO_RANGE_SUPPORT",
            **inventory,
            "reason": "Server did not honor Range: bytes=0-0; refusing accidental 13.3 GB transfer.",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 3

    total = probe.get("bytes")
    if not isinstance(total, int) or total <= 0:
        raise AcquisitionError(f"could not determine shared file size from ranged probe: {probe}")

    reader = HTTPRangeReader(opener, total)
    try:
        outer = zipfile.ZipFile(reader)
    except zipfile.BadZipFile as exc:
        payload = {
            "status": "BLOCKED_POTSDAM_SHARED_FILE_NOT_ZIP",
            **inventory,
            "reason": "The official 13.3 GB shared file is not a ZIP; selective ZIP extraction is unavailable.",
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 4

    with outer:
        direct = _find_direct_members(outer, tiles)
        nested = _nested_archive_candidates(outer)
        inventory["outer_zip"] = {
            "member_count": len(outer.infolist()),
            "direct_tile_members": {
                tile: {
                    role: [_info_summary(info) for info in infos]
                    for role, infos in roles.items()
                }
                for tile, roles in direct.items()
            },
            "nested_archive_candidates": {
                role: [_info_summary(info) for info in infos[:10]] for role, infos in nested.items()
            },
        }

        direct_ready = all(
            any(Path(PurePosixPath(info.filename).name).suffix.lower() in {".tif", ".tiff"} for info in direct[tile][role])
            for tile in tiles
            for role in ("rgb", "dsm")
        )
        nested_ready = bool(nested["rgb"] and nested["dsm"])
        inventory["selection_mode"] = (
            "direct_remote_members" if direct_ready else "nested_product_archives" if nested_ready else None
        )

        if args.inventory_only:
            status = (
                "PASS_POTSDAM_REMOTE_ZIP_INVENTORY"
                if direct_ready or nested_ready
                else "BLOCKED_POTSDAM_REMOTE_ZIP_LAYOUT_UNRECOGNIZED"
            )
            print(json.dumps({"status": status, **inventory}, indent=2, sort_keys=True))
            return 0 if status.startswith("PASS_") else 5

        extracted: dict[str, Any] = {}
        if direct_ready:
            for tile_id in tiles:
                wanted = _tile_names(tile_id)
                extracted[tile_id] = {
                    "rgb_files": _extract_selected_from_zip(
                        outer, wanted["rgb"], args.dataset_root, f"RGB {tile_id}"
                    ),
                    "dsm_files": _extract_selected_from_zip(
                        outer, wanted["dsm"], args.dataset_root, f"DSM {tile_id}"
                    ),
                    "reference_values_decoded_or_hashed": False,
                }
        elif nested_ready:
            rgb_info = nested["rgb"][0]
            dsm_info = nested["dsm"][0]
            rgb_archive = _download_zip_member(
                outer,
                rgb_info,
                args.cache_root / PurePosixPath(rgb_info.filename).name,
            )
            dsm_archive = _download_zip_member(
                outer,
                dsm_info,
                args.cache_root / PurePosixPath(dsm_info.filename).name,
            )
            with zipfile.ZipFile(rgb_archive) as rgb_zip, zipfile.ZipFile(dsm_archive) as dsm_zip:
                for tile_id in tiles:
                    wanted = _tile_names(tile_id)
                    extracted[tile_id] = {
                        "rgb_files": _extract_selected_from_zip(
                            rgb_zip, wanted["rgb"], args.dataset_root, f"RGB {tile_id}"
                        ),
                        "dsm_files": _extract_selected_from_zip(
                            dsm_zip, wanted["dsm"], args.dataset_root, f"DSM {tile_id}"
                        ),
                        "reference_values_decoded_or_hashed": False,
                    }
        else:
            raise AcquisitionError(
                "remote ZIP contains neither direct requested tile members nor recognizable RGB/DSM product ZIPs"
            )

    report = {
        "schema_version": 2,
        "status": "PASS_POTSDAM_OFFICIAL_FRESH_PAIR_ACQUISITION",
        **inventory,
        "extracted": extracted,
        "reference_values_decoded_or_hashed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
