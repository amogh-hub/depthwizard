from __future__ import annotations

import argparse
import binascii
import hashlib
import http.cookiejar
import io
import json
import os
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
import zlib
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Final

from depthwizard.height_model.terrain_structure_split import (
    HISTORICAL_CHALLENGE_TEST_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS,
    INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
    INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS,
    SEALED_BLIND_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
    initial_tsd_campaign_split,
)

CODE_ROOT = Path(__file__).resolve().parents[1]

SHARE_URL: Final = "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/"
DOWNLOAD_URL: Final = SHARE_URL + "?dl=1"
SHARE_TOKEN: Final = "429be50cc79d423ab6c4"
PASSWORD_ENV_DEFAULT: Final = "DW_SEAFILE_PASSWORD"
USER_AGENT: Final = "DepthWizard-TSD-Official-Range-Acquisition/1"

OUTER_RGB_MEMBER: Final = "Potsdam/2_Ortho_RGB.zip"
OUTER_DSM_MEMBER: Final = "Potsdam/1_DSM.rar"
OUTER_LABEL_MEMBER: Final = "Potsdam/5_Labels_for_participants.zip"

DESTINATION_SUBDIRECTORIES: Final[dict[str, str]] = {
    "rgb": "2_Ortho_RGB",
    "dsm": "1_DSM",
    "label": "5_Labels_for_participants",
}

RAR4_SIGNATURE: Final = b"Rar!\x1a\x07\x00"
RAR_BLOCK_MAIN: Final = 0x73
RAR_BLOCK_FILE: Final = 0x74
RAR_BLOCK_END: Final = 0x7B
RAR_MAIN_SOLID: Final = 0x0008
RAR_FILE_SPLIT_BEFORE: Final = 0x0001
RAR_FILE_SPLIT_AFTER: Final = 0x0002
RAR_FILE_PASSWORD: Final = 0x0004
RAR_FILE_SOLID: Final = 0x0010
RAR_FILE_LARGE: Final = 0x0100
RAR_FILE_UNICODE: Final = 0x0200
RAR_LONG_BLOCK: Final = 0x8000

NETWORK_CHUNK_BYTES: Final = 4 * 1024 * 1024
MAX_METADATA_READ_BYTES: Final = 64 * 1024 * 1024
MIN_EXPECTED_OUTER_BYTES: Final = 10_000_000_000
MAX_LABEL_OUTER_PACKED_BYTES: Final = 64 * 1024 * 1024
MAX_SELECTED_PACKED_BYTES: Final = 256 * 1024 * 1024
MIN_FREE_SPACE_MARGIN_BYTES: Final = 1024 * 1024 * 1024


@dataclass(frozen=True)
class PlannedFile:
    tile_id: str
    role: str
    component: str
    expected_filename: str


@dataclass(frozen=True)
class SelectedRemoteMember:
    component: str
    tile_id: str
    role: str
    filename: str
    archive_member: str
    packed_bytes: int
    unpacked_bytes: int
    crc32_hex: str
    transport: str
    companion: bool


@dataclass(frozen=True)
class AcquiredFile:
    component: str
    tile_id: str
    role: str
    filename: str
    destination: str
    bytes_written: int
    sha256: str
    crc32_hex: str
    companion: bool


@dataclass(frozen=True)
class RarMember:
    name: str
    header_offset: int
    data_offset: int
    header_bytes: bytes
    packed_bytes: int
    unpacked_bytes: int
    crc32: int
    unpack_version: int
    method: int
    flags: int


class _CsrfParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.token: str | None = None

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.casefold() != "input":
            return
        values = {key.casefold(): value for key, value in attrs}
        if values.get("name") == "csrfmiddlewaretoken" and values.get("value"):
            self.token = values["value"]


class HttpRangeSource:
    def __init__(self, opener: urllib.request.OpenerDirector) -> None:
        self.opener = opener
        self.total_size: int | None = None
        self.requests = 0
        self.network_bytes = 0

    def _open_range(self, start: int, end: int):
        if start < 0 or end < start:
            raise ValueError(f"invalid HTTP range {start}-{end}")
        request = urllib.request.Request(
            DOWNLOAD_URL,
            headers={
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
                "Referer": SHARE_URL,
                "User-Agent": USER_AGENT,
            },
        )
        response = self.opener.open(request, timeout=180)
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")
        if status != 206:
            response.close()
            raise RuntimeError(
                "official Potsdam server did not honor the requested byte range; refusing any "
                f"fallback that could download the complete archive (status={status})"
            )
        match = re.fullmatch(
            r"bytes (\d+)-(\d+)/(\d+|\*)",
            content_range.strip(),
            flags=re.IGNORECASE,
        )
        if match is None:
            response.close()
            raise RuntimeError(f"unexpected Content-Range header: {content_range!r}")
        actual_start = int(match.group(1))
        actual_end = int(match.group(2))
        if (actual_start, actual_end) != (start, end):
            response.close()
            raise RuntimeError(
                "official server returned a different range than requested: "
                f"requested={start}-{end}, returned={actual_start}-{actual_end}"
            )
        if match.group(3) != "*":
            total_size = int(match.group(3))
            if self.total_size is None:
                self.total_size = total_size
            elif self.total_size != total_size:
                response.close()
                raise RuntimeError("remote Potsdam.zip size changed during acquisition")
        self.requests += 1
        return response

    def fetch(self, start: int, end: int) -> bytes:
        if end - start + 1 > MAX_METADATA_READ_BYTES:
            raise RuntimeError(
                "single metadata fetch exceeds the fail-closed read limit; use chunked payload transfer"
            )
        with self._open_range(start, end) as response:
            data = response.read()
        expected = end - start + 1
        if len(data) != expected:
            raise RuntimeError(f"short HTTP range read: expected={expected}, got={len(data)}")
        self.network_bytes += len(data)
        return data

    def probe(self) -> int:
        data = self.fetch(0, 0)
        if data == b"":
            raise RuntimeError("one-byte range probe returned no data")
        if self.total_size is None:
            raise RuntimeError("official server did not report the Potsdam.zip size")
        if self.total_size < MIN_EXPECTED_OUTER_BYTES:
            raise RuntimeError(
                f"unexpected Potsdam.zip size {self.total_size}; refusing acquisition"
            )
        return self.total_size

    def download_absolute_range(
        self,
        *,
        start: int,
        length: int,
        destination: Path,
        resume: bool = True,
    ) -> None:
        if length <= 0:
            raise ValueError("range payload length must be positive")
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = destination.stat().st_size if destination.exists() and resume else 0
        if existing > length:
            raise RuntimeError(
                f"partial payload is larger than the expected range: {destination}"
            )
        if not resume and destination.exists():
            destination.unlink()
            existing = 0
        mode = "ab" if existing else "wb"
        with destination.open(mode) as output:
            cursor = existing
            while cursor < length:
                take = min(NETWORK_CHUNK_BYTES, length - cursor)
                absolute_start = start + cursor
                absolute_end = absolute_start + take - 1
                last_error: Exception | None = None
                for _attempt in range(3):
                    try:
                        with self._open_range(absolute_start, absolute_end) as response:
                            data = response.read()
                        if len(data) != take:
                            raise RuntimeError(
                                f"short payload range read: expected={take}, got={len(data)}"
                            )
                        output.write(data)
                        output.flush()
                        self.network_bytes += len(data)
                        cursor += len(data)
                        last_error = None
                        break
                    except Exception as exc:  # noqa: BLE001 - retry transport errors, then re-raise.
                        last_error = exc
                if last_error is not None:
                    raise last_error
        if destination.stat().st_size != length:
            raise RuntimeError(
                f"packed payload size mismatch: expected={length}, got={destination.stat().st_size}"
            )


class RemoteView:
    def __init__(self, source: HttpRangeSource, absolute_start: int, size: int) -> None:
        if absolute_start < 0 or size <= 0:
            raise ValueError("invalid remote view")
        self.source = source
        self.absolute_start = absolute_start
        self.size = size

    def fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start or end >= self.size:
            raise ValueError(
                f"view range outside bounds: {start}-{end} of {self.size}"
            )
        return self.source.fetch(self.absolute_start + start, self.absolute_start + end)

    def download(self, *, start: int, length: int, destination: Path) -> None:
        if start < 0 or length <= 0 or start + length > self.size:
            raise ValueError("payload range outside remote view")
        self.source.download_absolute_range(
            start=self.absolute_start + start,
            length=length,
            destination=destination,
        )


class SeekableRemote(io.RawIOBase):
    def __init__(self, view: RemoteView) -> None:
        super().__init__()
        self.view = view
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_position = offset
        elif whence == io.SEEK_CUR:
            new_position = self.position + offset
        elif whence == io.SEEK_END:
            new_position = self.view.size + offset
        else:
            raise ValueError(f"unsupported whence {whence}")
        if new_position < 0:
            raise ValueError("negative seek")
        self.position = new_position
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.view.size:
            return b""
        if size is None or size < 0:
            size = self.view.size - self.position
        if size == 0:
            return b""
        if size > MAX_METADATA_READ_BYTES:
            raise RuntimeError(
                "zip metadata reader attempted an unexpectedly large remote read; refusing"
            )
        end = min(self.view.size - 1, self.position + size - 1)
        data = self.view.fetch(self.position, end)
        self.position += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        view = memoryview(buffer)
        data = self.read(len(view))
        view[: len(data)] = data
        return len(data)


class _NoCloseBufferedReader(io.BufferedReader):
    def close(self) -> None:
        # ZipFile attempts to close user-provided streams. The owner controls the underlying view.
        pass


def _make_opener() -> urllib.request.OpenerDirector:
    cookie_jar = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cookie_jar))
    opener.addheaders = [
        ("User-Agent", USER_AGENT),
        ("Accept-Encoding", "identity"),
    ]
    return opener


def _authenticate(opener: urllib.request.OpenerDirector, password: str) -> None:
    if not password:
        raise RuntimeError("Seafile password is empty")
    request = urllib.request.Request(SHARE_URL, headers={"Accept": "text/html"})
    with opener.open(request, timeout=30) as response:
        page = response.read(2_000_000).decode("utf-8", "replace")
    parser = _CsrfParser()
    parser.feed(page)
    csrf = parser.token
    if csrf is None:
        match = re.search(
            r'name=["\']csrfmiddlewaretoken["\'][^>]*value=["\']([^"\']+)',
            page,
            flags=re.IGNORECASE,
        )
        csrf = match.group(1) if match is not None else None
    if csrf is None:
        raise RuntimeError("could not locate Seafile CSRF token")
    payload = urllib.parse.urlencode(
        {
            "csrfmiddlewaretoken": csrf,
            "token": SHARE_TOKEN,
            "password": password,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        SHARE_URL,
        data=payload,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": SHARE_URL,
            "Accept": "text/html",
        },
        method="POST",
    )
    with opener.open(request, timeout=30) as response:
        body = response.read(2_000_000).decode("utf-8", "replace")
    if "Please enter a correct password" in body:
        raise RuntimeError("Seafile rejected DW_SEAFILE_PASSWORD")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            crc = binascii.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


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
            "refusing official acquisition from a tracked-dirty worktree; restore or commit first"
        )
    return head, branch


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
    raise ValueError(f"unsupported component {component!r}")


def _load_plan(path: Path, dataset_root: Path) -> tuple[dict[str, Any], tuple[PlannedFile, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("TSD acquisition plan must be a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported TSD acquisition-plan schema")
    if payload.get("split_protocol_version") != TSD_SPLIT_PROTOCOL_VERSION:
        raise ValueError("acquisition plan split protocol differs from current TSD protocol")
    if payload.get("campaign_protocol_version") != INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION:
        raise ValueError("acquisition plan is not the frozen first TSD campaign")
    if payload.get("train_tile_ids") != list(INITIAL_TSD_CAMPAIGN_TRAIN_TILE_IDS):
        raise ValueError("acquisition-plan training tiles differ from the frozen campaign")
    if payload.get("dev_tile_ids") != list(INITIAL_TSD_CAMPAIGN_DEV_TILE_IDS):
        raise ValueError("acquisition-plan development tiles differ from the frozen campaign")
    plan_root = payload.get("source_inventory_dataset_root")
    if not isinstance(plan_root, str):
        raise TypeError("acquisition plan has no valid source inventory dataset root")
    if Path(plan_root).resolve() != dataset_root.resolve():
        raise ValueError("requested dataset root differs from the frozen acquisition plan")

    split = initial_tsd_campaign_split()
    active = set(split.supervised_tile_ids)
    train = set(split.train_tile_ids)
    raw_required = payload.get("required_files")
    if not isinstance(raw_required, list):
        raise TypeError("acquisition plan required_files must be an array")
    planned: list[PlannedFile] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_required:
        if not isinstance(raw, dict):
            raise TypeError("malformed acquisition-plan required-file record")
        tile_id = raw.get("tile_id")
        role = raw.get("role")
        component = raw.get("component")
        filename = raw.get("expected_filename")
        if not all(isinstance(value, str) for value in (tile_id, role, component, filename)):
            raise TypeError("acquisition-plan required-file fields must be strings")
        assert isinstance(tile_id, str)
        assert isinstance(role, str)
        assert isinstance(component, str)
        assert isinstance(filename, str)
        if tile_id not in active:
            raise ValueError(f"acquisition plan contains non-active tile {tile_id}")
        if tile_id in SEALED_BLIND_TILE_IDS:
            raise RuntimeError(f"sealed blind tile appeared in acquisition plan: {tile_id}")
        expected_role = "train" if tile_id in train else "dev"
        if role != expected_role:
            raise ValueError(f"role mismatch for active tile {tile_id}")
        if component not in DESTINATION_SUBDIRECTORIES:
            raise ValueError(f"unsupported acquisition component {component!r}")
        if filename != _expected_filename(tile_id, component):
            raise ValueError(f"filename mismatch for {tile_id}/{component}")
        key = (tile_id, component)
        if key in seen:
            raise ValueError(f"duplicate acquisition requirement {tile_id}/{component}")
        seen.add(key)
        planned.append(
            PlannedFile(
                tile_id=tile_id,
                role=role,
                component=component,
                expected_filename=filename,
            )
        )
    for component in DESTINATION_SUBDIRECTORIES:
        actual = [item.tile_id for item in planned if item.component == component]
        if payload.get(f"missing_{component}_tile_ids") != actual:
            raise ValueError(f"acquisition-plan {component} summary disagrees with required_files")
        if payload.get(f"missing_{component}_count") != len(actual):
            raise ValueError(f"acquisition-plan {component} count disagrees with required_files")
    return payload, tuple(planned)


def _companion_world_filename(filename: str) -> str:
    lower = filename.casefold()
    if not (lower.endswith(".tif") or lower.endswith(".tiff")):
        raise ValueError(f"cannot derive world file from {filename}")
    return str(Path(filename).with_suffix(".tfw"))


def required_physical_files(planned: tuple[PlannedFile, ...]) -> tuple[PlannedFile, ...]:
    files: list[PlannedFile] = []
    for item in planned:
        files.append(item)
        if item.component in {"rgb", "dsm"}:
            files.append(
                PlannedFile(
                    tile_id=item.tile_id,
                    role=item.role,
                    component=item.component,
                    expected_filename=_companion_world_filename(item.expected_filename),
                )
            )
    return tuple(files)


def _zip_member_data_start(view: RemoteView, info: zipfile.ZipInfo) -> int:
    header = view.fetch(info.header_offset, info.header_offset + 29)
    if header[:4] != b"PK\x03\x04" or len(header) != 30:
        raise RuntimeError(f"invalid ZIP local header for {info.filename}")
    fields = struct.unpack("<4s5H3L2H", header)
    filename_length = fields[-2]
    extra_length = fields[-1]
    return info.header_offset + 30 + filename_length + extra_length


def _safe_zip_name(info: zipfile.ZipInfo) -> str:
    normalized = info.filename.replace("\\", "/")
    if "\x00" in normalized:
        raise ValueError("ZIP member contains NUL")
    pure = PurePosixPath(normalized)
    if pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
    return normalized


def _zip_basename_index(archive: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    index: dict[str, list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        normalized = _safe_zip_name(info)
        if info.is_dir():
            continue
        index.setdefault(PurePosixPath(normalized).name.casefold(), []).append(info)
    return index


def _unique_outer_member(archive: zipfile.ZipFile, expected: str) -> zipfile.ZipInfo:
    matches = [info for info in archive.infolist() if _safe_zip_name(info) == expected]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one outer member {expected!r}; found {len(matches)}")
    return matches[0]


def _extract_remote_zip_member(
    *,
    view: RemoteView,
    info: zipfile.ZipInfo,
    packed_cache: Path,
    destination: Path,
) -> tuple[int, str, str]:
    if info.compress_size <= 0 or info.compress_size > MAX_SELECTED_PACKED_BYTES:
        raise RuntimeError(f"invalid selected ZIP packed size for {info.filename}")
    data_start = _zip_member_data_start(view, info)
    view.download(start=data_start, length=info.compress_size, destination=packed_cache)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    crc = 0
    count = 0
    digest = hashlib.sha256()
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_STORED:
        decompressor = None
    else:
        raise RuntimeError(
            f"unsupported selected ZIP compression method {info.compress_type}: {info.filename}"
        )
    try:
        with packed_cache.open("rb") as source, partial.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                decoded = decompressor.decompress(chunk) if decompressor is not None else chunk
                if decoded:
                    output.write(decoded)
                    crc = binascii.crc32(decoded, crc)
                    digest.update(decoded)
                    count += len(decoded)
            if decompressor is not None:
                tail = decompressor.flush()
                if tail:
                    output.write(tail)
                    crc = binascii.crc32(tail, crc)
                    digest.update(tail)
                    count += len(tail)
        crc &= 0xFFFFFFFF
        if count != info.file_size:
            raise RuntimeError(
                f"ZIP output size mismatch for {info.filename}: expected={info.file_size}, got={count}"
            )
        if crc != info.CRC:
            raise RuntimeError(
                f"ZIP CRC mismatch for {info.filename}: expected={info.CRC:08x}, got={crc:08x}"
            )
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    packed_cache.unlink(missing_ok=True)
    return count, digest.hexdigest(), f"{crc:08x}"


def _rar_header_crc_valid(raw: bytes) -> bool:
    if len(raw) < 7:
        return False
    expected = struct.unpack_from("<H", raw, 0)[0]
    actual = binascii.crc32(raw[2:]) & 0xFFFF
    return expected == actual


def _rar_name(raw_name: bytes, unicode_flag: bool) -> str:
    ascii_part = raw_name.split(b"\x00", 1)[0] if unicode_flag else raw_name
    return ascii_part.decode("latin-1").replace("\\", "/")


def parse_rar4_members(view: RemoteView) -> tuple[bytes, bytes, tuple[RarMember, ...]]:
    if view.fetch(0, len(RAR4_SIGNATURE) - 1) != RAR4_SIGNATURE:
        raise RuntimeError("official DSM member is not the expected RAR4 archive")
    offset = len(RAR4_SIGNATURE)
    main_header: bytes | None = None
    end_header: bytes | None = None
    members: list[RarMember] = []
    archive_solid = False
    while offset < view.size:
        base = view.fetch(offset, offset + 6)
        head_type = base[2]
        flags = struct.unpack_from("<H", base, 3)[0]
        head_size = struct.unpack_from("<H", base, 5)[0]
        if head_size < 7 or head_size > 65535 or offset + head_size > view.size:
            raise RuntimeError(f"invalid RAR4 header size at offset {offset}: {head_size}")
        raw = view.fetch(offset, offset + head_size - 1)
        if not _rar_header_crc_valid(raw):
            raise RuntimeError(f"RAR4 header CRC mismatch at offset {offset}")
        add_size = 0
        if head_type == RAR_BLOCK_MAIN:
            main_header = raw
            archive_solid = bool(flags & RAR_MAIN_SOLID)
        elif head_type == RAR_BLOCK_FILE:
            if len(raw) < 32:
                raise RuntimeError(f"truncated RAR4 file header at {offset}")
            pack_low = struct.unpack_from("<I", raw, 7)[0]
            unpack_low = struct.unpack_from("<I", raw, 11)[0]
            file_crc = struct.unpack_from("<I", raw, 16)[0]
            unpack_version = raw[24]
            method = raw[25]
            name_size = struct.unpack_from("<H", raw, 26)[0]
            name_offset = 32
            pack_high = 0
            unpack_high = 0
            if flags & RAR_FILE_LARGE:
                if len(raw) < 40:
                    raise RuntimeError(f"truncated large RAR4 file header at {offset}")
                pack_high = struct.unpack_from("<I", raw, 32)[0]
                unpack_high = struct.unpack_from("<I", raw, 36)[0]
                name_offset = 40
            if name_offset + name_size > len(raw):
                raise RuntimeError(f"RAR4 filename exceeds header at {offset}")
            packed = pack_low | (pack_high << 32)
            unpacked = unpack_low | (unpack_high << 32)
            name = _rar_name(
                raw[name_offset : name_offset + name_size],
                bool(flags & RAR_FILE_UNICODE),
            )
            if flags & (RAR_FILE_SPLIT_BEFORE | RAR_FILE_SPLIT_AFTER | RAR_FILE_PASSWORD):
                raise RuntimeError(f"unsupported split/encrypted DSM RAR member: {name}")
            members.append(
                RarMember(
                    name=name,
                    header_offset=offset,
                    data_offset=offset + head_size,
                    header_bytes=raw,
                    packed_bytes=packed,
                    unpacked_bytes=unpacked,
                    crc32=file_crc,
                    unpack_version=unpack_version,
                    method=method,
                    flags=flags,
                )
            )
            add_size = packed
        elif head_type == RAR_BLOCK_END:
            end_header = raw
            break
        elif flags & RAR_LONG_BLOCK:
            if len(raw) < 11:
                raise RuntimeError(f"truncated long RAR4 block at {offset}")
            add_size = struct.unpack_from("<I", raw, 7)[0]
        offset += head_size + add_size
    if main_header is None or end_header is None:
        raise RuntimeError("RAR4 main/end block was not found")
    if archive_solid:
        raise RuntimeError("official DSM RAR unexpectedly became solid; selective extraction refused")
    return main_header, end_header, tuple(members)


def _rar_basename_index(members: tuple[RarMember, ...]) -> dict[str, list[RarMember]]:
    index: dict[str, list[RarMember]] = {}
    for member in members:
        basename = PurePosixPath(member.name).name.casefold()
        if basename:
            index.setdefault(basename, []).append(member)
    return index


def _extract_remote_rar_member(
    *,
    view: RemoteView,
    member: RarMember,
    main_header: bytes,
    end_header: bytes,
    packed_cache: Path,
    destination: Path,
    bsdtar: str,
) -> tuple[int, str, str]:
    if member.flags & RAR_FILE_SOLID:
        raise RuntimeError(f"selected DSM RAR member is solid: {member.name}")
    if member.packed_bytes <= 0 or member.packed_bytes > MAX_SELECTED_PACKED_BYTES:
        raise RuntimeError(f"invalid selected DSM packed size: {member.name}")
    view.download(start=member.data_offset, length=member.packed_bytes, destination=packed_cache)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    with tempfile.TemporaryDirectory(prefix="depthwizard-tsd-rar-") as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        mini_rar = temp_dir / "selected.rar"
        with mini_rar.open("wb") as output, packed_cache.open("rb") as packed:
            output.write(RAR4_SIGNATURE)
            output.write(main_header)
            output.write(member.header_bytes)
            shutil.copyfileobj(packed, output, length=1024 * 1024)
            output.write(end_header)
        extracted_root = temp_dir / "out"
        extracted_root.mkdir()
        result = subprocess.run(
            [bsdtar, "-xf", str(mini_rar), "-C", str(extracted_root)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "bsdtar failed to unpack the selectively reconstructed DSM member: "
                f"{result.stderr.strip()}"
            )
        matches = [
            path
            for path in extracted_root.rglob("*")
            if path.is_file() and path.name.casefold() == destination.name.casefold()
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"selective DSM reconstruction yielded {len(matches)} matches for {destination.name}"
            )
        extracted = matches[0]
        if extracted.stat().st_size != member.unpacked_bytes:
            raise RuntimeError(
                f"RAR output size mismatch for {member.name}: "
                f"expected={member.unpacked_bytes}, got={extracted.stat().st_size}"
            )
        crc = _crc32_file(extracted)
        if crc != member.crc32:
            raise RuntimeError(
                f"RAR CRC mismatch for {member.name}: expected={member.crc32:08x}, got={crc:08x}"
            )
        sha256 = _sha256_file(extracted)
        os.replace(extracted, destination)
    packed_cache.unlink(missing_ok=True)
    return member.unpacked_bytes, sha256, f"{member.crc32:08x}"


def _reconstruct_outer_zip_member(
    *,
    outer_view: RemoteView,
    info: zipfile.ZipInfo,
    packed_cache: Path,
    destination: Path,
) -> None:
    if info.compress_size <= 0 or info.compress_size > MAX_LABEL_OUTER_PACKED_BYTES:
        raise RuntimeError(
            f"participant-label archive packed size outside safety bound: {info.compress_size}"
        )
    data_start = _zip_member_data_start(outer_view, info)
    outer_view.download(start=data_start, length=info.compress_size, destination=packed_cache)
    crc = 0
    count = 0
    if info.compress_type == zipfile.ZIP_DEFLATED:
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    elif info.compress_type == zipfile.ZIP_STORED:
        decompressor = None
    else:
        raise RuntimeError(
            f"unsupported outer compression for participant labels: {info.compress_type}"
        )
    with packed_cache.open("rb") as source, destination.open("wb") as output:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            decoded = decompressor.decompress(chunk) if decompressor is not None else chunk
            if decoded:
                output.write(decoded)
                count += len(decoded)
                crc = binascii.crc32(decoded, crc)
        if decompressor is not None:
            tail = decompressor.flush()
            if tail:
                output.write(tail)
                count += len(tail)
                crc = binascii.crc32(tail, crc)
    crc &= 0xFFFFFFFF
    if count != info.file_size or crc != info.CRC:
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            "participant-label nested ZIP failed outer member size/CRC verification"
        )
    packed_cache.unlink(missing_ok=True)


def _extract_local_label(
    *,
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> tuple[int, str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.unlink(missing_ok=True)
    crc = 0
    count = 0
    digest = hashlib.sha256()
    try:
        with archive.open(info, "r") as source, partial.open("wb") as output:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                output.write(chunk)
                count += len(chunk)
                crc = binascii.crc32(chunk, crc)
                digest.update(chunk)
        crc &= 0xFFFFFFFF
        if count != info.file_size or crc != info.CRC:
            raise RuntimeError(f"participant label size/CRC mismatch: {info.filename}")
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return count, digest.hexdigest(), f"{crc:08x}"


def _assert_no_local_conflicts(dataset_root: Path, files: tuple[PlannedFile, ...]) -> None:
    index: dict[str, list[Path]] = {}
    for path in dataset_root.rglob("*"):
        if path.is_file():
            index.setdefault(path.name.casefold(), []).append(path)
    conflicts = sorted(
        item.expected_filename
        for item in files
        if index.get(item.expected_filename.casefold())
    )
    if conflicts:
        raise RuntimeError(
            "frozen acquisition plan is stale because planned physical files already exist locally: "
            + ",".join(conflicts)
        )


def _destination(dataset_root: Path, item: PlannedFile) -> Path:
    return dataset_root / DESTINATION_SUBDIRECTORIES[item.component] / item.expected_filename


def _select_zip_files(
    *,
    archive: zipfile.ZipFile,
    planned: tuple[PlannedFile, ...],
    component: str,
    transport: str,
    companions: set[str],
) -> tuple[tuple[PlannedFile, zipfile.ZipInfo], ...]:
    index = _zip_basename_index(archive)
    selected: list[tuple[PlannedFile, zipfile.ZipInfo]] = []
    for item in planned:
        if item.component != component:
            continue
        matches = index.get(item.expected_filename.casefold(), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"official {transport} package must contain exactly one {item.expected_filename}; "
                f"found {len(matches)}"
            )
        selected.append((item, matches[0]))
    return tuple(selected)


def _selected_manifest_entry(
    item: PlannedFile,
    info: zipfile.ZipInfo,
    *,
    transport: str,
    companions: set[str],
) -> SelectedRemoteMember:
    return SelectedRemoteMember(
        component=item.component,
        tile_id=item.tile_id,
        role=item.role,
        filename=item.expected_filename,
        archive_member=info.filename,
        packed_bytes=info.compress_size,
        unpacked_bytes=info.file_size,
        crc32_hex=f"{info.CRC:08x}",
        transport=transport,
        companion=item.expected_filename in companions,
    )


def _rar_selected_manifest_entry(
    item: PlannedFile,
    member: RarMember,
    *,
    companions: set[str],
) -> SelectedRemoteMember:
    return SelectedRemoteMember(
        component=item.component,
        tile_id=item.tile_id,
        role=item.role,
        filename=item.expected_filename,
        archive_member=member.name,
        packed_bytes=member.packed_bytes,
        unpacked_bytes=member.unpacked_bytes,
        crc32_hex=f"{member.crc32:08x}",
        transport="outer-store/rar4-nonsolid/member-range",
        companion=item.expected_filename in companions,
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire only the frozen first-campaign Potsdam inputs from the official 13.3 GB "
            "Seafile Potsdam.zip using authenticated HTTP byte ranges. The complete outer archive, "
            "RGB inner archive, and DSM RAR are never downloaded."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--password-env", default=PASSWORD_ENV_DEFAULT)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="write selected TIFF/TFW/label outputs after all remote metadata gates pass",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    password = os.environ.get(args.password_env, "")
    if not password:
        raise RuntimeError(
            f"set the Seafile password in environment variable {args.password_env!r}"
        )
    _, planned_logical = _load_plan(args.plan, args.dataset_root)
    planned_physical = required_physical_files(planned_logical)
    companion_names = {
        item.expected_filename
        for item in planned_physical
        if item.component in {"rgb", "dsm"}
        and item.expected_filename.casefold().endswith(".tfw")
    }
    _assert_no_local_conflicts(args.dataset_root, planned_physical)
    source_sha, source_branch = _tracked_source_identity()

    bsdtar = shutil.which("bsdtar")
    if bsdtar is None:
        raise RuntimeError("bsdtar is required for selective RAR4 DSM extraction")

    opener = _make_opener()
    print("[1/7] Authenticating to official ISPRS/Leibniz Hannover Seafile ...")
    _authenticate(opener, password)
    source = HttpRangeSource(opener)
    outer_size = source.probe()
    print(f"PASS: HTTP 206 byte ranges; Potsdam.zip={outer_size:,} bytes")

    outer_view = RemoteView(source, 0, outer_size)
    print("[2/7] Reading outer ZIP metadata only ...")
    outer_stream = _NoCloseBufferedReader(SeekableRemote(outer_view))
    with zipfile.ZipFile(outer_stream, "r", allowZip64=True) as outer_zip:
        rgb_outer = _unique_outer_member(outer_zip, OUTER_RGB_MEMBER)
        dsm_outer = _unique_outer_member(outer_zip, OUTER_DSM_MEMBER)
        label_outer = _unique_outer_member(outer_zip, OUTER_LABEL_MEMBER)
    if rgb_outer.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError("RGB inner ZIP is no longer STORE-compressed in outer archive")
    if dsm_outer.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError("DSM RAR is no longer STORE-compressed in outer archive")

    rgb_start = _zip_member_data_start(outer_view, rgb_outer)
    dsm_start = _zip_member_data_start(outer_view, dsm_outer)
    rgb_view = RemoteView(source, rgb_start, rgb_outer.file_size)
    dsm_view = RemoteView(source, dsm_start, dsm_outer.file_size)

    print("[3/7] Resolving only planned RGB TIFF + TFW metadata ...")
    rgb_stream = _NoCloseBufferedReader(SeekableRemote(rgb_view))
    with zipfile.ZipFile(rgb_stream, "r", allowZip64=True) as rgb_zip:
        rgb_selected = _select_zip_files(
            archive=rgb_zip,
            planned=planned_physical,
            component="rgb",
            transport="outer-store/inner-zip/member-range",
            companions=companion_names,
        )
    if len(rgb_selected) != 24:
        raise RuntimeError(f"expected 24 planned RGB physical files, found {len(rgb_selected)}")

    print("[4/7] Walking DSM RAR4 headers only; packed payloads are skipped mathematically ...")
    main_header, end_header, rar_members = parse_rar4_members(dsm_view)
    rar_index = _rar_basename_index(rar_members)
    dsm_selected: list[tuple[PlannedFile, RarMember]] = []
    for item in planned_physical:
        if item.component != "dsm":
            continue
        matches = rar_index.get(item.expected_filename.casefold(), [])
        if len(matches) != 1:
            raise RuntimeError(
                f"official DSM RAR must contain exactly one {item.expected_filename}; "
                f"found {len(matches)}"
            )
        member = matches[0]
        if member.flags & RAR_FILE_SOLID:
            raise RuntimeError(f"selected DSM member became solid: {member.name}")
        dsm_selected.append((item, member))
    if len(dsm_selected) != 24:
        raise RuntimeError(f"expected 24 planned DSM physical files, found {len(dsm_selected)}")

    print("[5/7] Reconstructing only the small participant-label nested ZIP ...")
    acquired: list[AcquiredFile] = []
    selected_manifest: list[SelectedRemoteMember] = []
    selected_manifest.extend(
        _selected_manifest_entry(
            item,
            info,
            transport="outer-store/inner-zip/member-range",
            companions=companion_names,
        )
        for item, info in rgb_selected
    )
    selected_manifest.extend(
        _rar_selected_manifest_entry(item, member, companions=companion_names)
        for item, member in dsm_selected
    )

    with tempfile.TemporaryDirectory(prefix="depthwizard-tsd-labels-") as label_temp_text:
        label_temp = Path(label_temp_text)
        label_packed = label_temp / "participant-labels.outer-packed.partial"
        label_zip_path = label_temp / "5_Labels_for_participants.zip"
        _reconstruct_outer_zip_member(
            outer_view=outer_view,
            info=label_outer,
            packed_cache=label_packed,
            destination=label_zip_path,
        )
        if not zipfile.is_zipfile(label_zip_path):
            raise RuntimeError("reconstructed participant-label member is not a readable ZIP")
        with zipfile.ZipFile(label_zip_path, "r") as labels_zip:
            label_index = _zip_basename_index(labels_zip)
            forbidden = [
                tile_id
                for tile_id in HISTORICAL_CHALLENGE_TEST_TILE_IDS
                if _expected_filename(tile_id, "label").casefold() in label_index
            ]
            if forbidden:
                raise RuntimeError(
                    "participant-label package unexpectedly contains historical challenge-test labels; "
                    "refusing supervision acquisition"
                )
            label_selected = _select_zip_files(
                archive=labels_zip,
                planned=planned_physical,
                component="label",
                transport="outer-deflate/full-small-inner-zip/selected-entry",
                companions=companion_names,
            )
            if len(label_selected) != 13:
                raise RuntimeError(
                    f"expected 13 planned participant labels, found {len(label_selected)}"
                )
            selected_manifest.extend(
                _selected_manifest_entry(
                    item,
                    info,
                    transport="outer-deflate/full-small-inner-zip/selected-entry",
                    companions=companion_names,
                )
                for item, info in label_selected
            )

            if args.execute:
                required_unpacked = sum(item.unpacked_bytes for item in selected_manifest)
                free = shutil.disk_usage(args.dataset_root).free
                if free < required_unpacked + MIN_FREE_SPACE_MARGIN_BYTES:
                    raise RuntimeError(
                        "insufficient free disk space for frozen TSD acquisition: "
                        f"required>={required_unpacked + MIN_FREE_SPACE_MARGIN_BYTES}, free={free}"
                    )

                cache_root = args.dataset_root / ".tsd-range-cache"
                cache_root.mkdir(parents=True, exist_ok=True)
                try:
                    print("[6/7] Downloading exact planned RGB/DSM member ranges ...")
                    for item, info in rgb_selected:
                        destination = _destination(args.dataset_root, item)
                        cache = cache_root / f"rgb-{item.tile_id}-{item.expected_filename}.packed.partial"
                        count, sha256, crc_hex = _extract_remote_zip_member(
                            view=rgb_view,
                            info=info,
                            packed_cache=cache,
                            destination=destination,
                        )
                        acquired.append(
                            AcquiredFile(
                                component=item.component,
                                tile_id=item.tile_id,
                                role=item.role,
                                filename=item.expected_filename,
                                destination=str(destination.resolve()),
                                bytes_written=count,
                                sha256=sha256,
                                crc32_hex=crc_hex,
                                companion=item.expected_filename in companion_names,
                            )
                        )
                        print(f"RGB PASS: {item.expected_filename} | {count:,} bytes")

                    for item, member in dsm_selected:
                        destination = _destination(args.dataset_root, item)
                        cache = cache_root / f"dsm-{item.tile_id}-{item.expected_filename}.packed.partial"
                        count, sha256, crc_hex = _extract_remote_rar_member(
                            view=dsm_view,
                            member=member,
                            main_header=main_header,
                            end_header=end_header,
                            packed_cache=cache,
                            destination=destination,
                            bsdtar=bsdtar,
                        )
                        acquired.append(
                            AcquiredFile(
                                component=item.component,
                                tile_id=item.tile_id,
                                role=item.role,
                                filename=item.expected_filename,
                                destination=str(destination.resolve()),
                                bytes_written=count,
                                sha256=sha256,
                                crc32_hex=crc_hex,
                                companion=item.expected_filename in companion_names,
                            )
                        )
                        print(f"DSM PASS: {item.expected_filename} | {count:,} bytes")

                    print("[7/7] Extracting only the 13 planned participant labels ...")
                    for item, info in label_selected:
                        destination = _destination(args.dataset_root, item)
                        count, sha256, crc_hex = _extract_local_label(
                            archive=labels_zip,
                            info=info,
                            destination=destination,
                        )
                        acquired.append(
                            AcquiredFile(
                                component=item.component,
                                tile_id=item.tile_id,
                                role=item.role,
                                filename=item.expected_filename,
                                destination=str(destination.resolve()),
                                bytes_written=count,
                                sha256=sha256,
                                crc32_hex=crc_hex,
                                companion=False,
                            )
                        )
                        print(f"LABEL PASS: {item.expected_filename} | {count:,} bytes")
                except BaseException:
                    for record in reversed(acquired):
                        Path(record.destination).unlink(missing_ok=True)
                    raise
                finally:
                    if cache_root.is_dir() and not any(cache_root.iterdir()):
                        cache_root.rmdir()

    expected_physical_count = 61
    if len(selected_manifest) != expected_physical_count:
        raise RuntimeError(
            f"selected physical-file count mismatch: expected={expected_physical_count}, "
            f"got={len(selected_manifest)}"
        )
    if args.execute and len(acquired) != expected_physical_count:
        raise RuntimeError(
            f"acquired physical-file count mismatch: expected={expected_physical_count}, "
            f"got={len(acquired)}"
        )

    status = "PASS_TSD_OFFICIAL_RANGE_ACQUISITION" if args.execute else "PASS_TSD_RANGE_PREFLIGHT"
    report = {
        "schema_version": 1,
        "status": status,
        "qualification_git_sha": source_sha,
        "qualification_git_branch": source_branch,
        "split_protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "campaign_protocol_version": INITIAL_TSD_CAMPAIGN_PROTOCOL_VERSION,
        "acquisition_plan": str(args.plan.resolve()),
        "acquisition_plan_sha256": _sha256_file(args.plan),
        "dataset_root": str(args.dataset_root.resolve()),
        "official_share": SHARE_URL,
        "download_button_url": DOWNLOAD_URL,
        "file_share": {
            "filename": "Potsdam.zip",
            "bytes": outer_size,
            "range_supported": True,
            "status_code": 206,
        },
        "outer_members": {
            "rgb": {
                "name": rgb_outer.filename,
                "bytes": rgb_outer.file_size,
                "compress_type": rgb_outer.compress_type,
                "payload_absolute_start": rgb_start,
            },
            "dsm": {
                "name": dsm_outer.filename,
                "bytes": dsm_outer.file_size,
                "compress_type": dsm_outer.compress_type,
                "payload_absolute_start": dsm_start,
                "rar_format": "RAR4",
                "rar_solid": False,
                "rar_member_count": len(rar_members),
            },
            "participant_labels": {
                "name": label_outer.filename,
                "compressed_bytes": label_outer.compress_size,
                "uncompressed_bytes": label_outer.file_size,
                "compress_type": label_outer.compress_type,
            },
        },
        "logical_missing_component_count": len(planned_logical),
        "mandatory_georeference_sidecar_count": len(companion_names),
        "selected_physical_file_count": len(selected_manifest),
        "selected_members": [asdict(item) for item in selected_manifest],
        "acquired_file_count": len(acquired),
        "acquired_files": [asdict(item) for item in acquired],
        "range_requests": source.requests,
        "network_bytes_transferred": source.network_bytes,
        "full_outer_archive_downloaded": False,
        "full_rgb_inner_archive_downloaded": False,
        "full_dsm_rar_downloaded": False,
        "participant_label_outer_member_transferred": True,
        "nonselected_participant_label_entries_decompressed": False,
        "raster_pixels_decoded_by_downloader": False,
        "sealed_blind_tile_payloads_extracted": False,
        "sealed_blind_tile_ids": list(SEALED_BLIND_TILE_IDS),
        "claim_boundary": (
            "The downloader uses authenticated HTTP 206 byte ranges against the official 13.3 GB "
            "Potsdam.zip and writes only frozen active-campaign TIFFs plus mandatory TFW companions "
            "and selected participant labels. The complete outer ZIP, RGB nested ZIP, and DSM RAR "
            "are never transferred. Because the small participant-label nested ZIP is DEFLATE-"
            "compressed by the outer ZIP, its complete ~19 MB nested ZIP representation is "
            "reconstructed transiently; only the 13 planned label entries are decompressed to "
            "dataset outputs. No raster pixels are decoded by this acquisition command, and sealed "
            "4_12/6_12 entries are never extracted."
        ),
    }
    _write_json_atomic(args.report, report)

    print(f"status={status}")
    print(f"qualification_git_sha={source_sha}")
    print(f"logical_missing_component_count={len(planned_logical)}")
    print(f"mandatory_georeference_sidecar_count={len(companion_names)}")
    print(f"selected_physical_file_count={len(selected_manifest)}")
    print(f"acquired_file_count={len(acquired)}")
    print(f"range_requests={source.requests}")
    print(f"network_bytes_transferred={source.network_bytes}")
    print("full_outer_archive_downloaded=false")
    print("full_rgb_inner_archive_downloaded=false")
    print("full_dsm_rar_downloaded=false")
    print("nonselected_participant_label_entries_decompressed=false")
    print("raster_pixels_decoded_by_downloader=false")
    print("sealed_blind_tile_payloads_extracted=false")
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"official Potsdam HTTP error: {exc.code} {exc.reason}") from exc
