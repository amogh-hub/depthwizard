#!/usr/bin/env python3
"""Selectively acquire two fresh official ISPRS Potsdam RGB/DSM pairs.

This intentionally reuses the proven DepthWizard acquisition strategy from the earlier
external benchmark:

1. authenticate to the official ISPRS Seafile *file-share*;
2. use the copied Download-button endpoint with HTTP byte ranges;
3. open the outer Potsdam.zip remotely;
4. jump into the STORE-compressed ``Potsdam/2_Ortho_RGB.zip`` member and extract only
   the requested RGB TIFF/world-file members;
5. walk the STORE-compressed, non-solid RAR4 ``Potsdam/1_DSM.rar`` headers remotely;
6. fetch only the packed bytes for the requested DSM TIFF/world-file members and let
   macOS ``bsdtar`` decompress one synthetic single-member RAR at a time.

The full ~12.4 GiB Potsdam archive, full ~2.75 GiB RGB ZIP, and full ~1.93 GiB DSM RAR
are never downloaded.

Reference-safety boundary:
- archive metadata/headers may be inspected;
- DSM archive bytes may be selectively downloaded and decompressed to TIFF bytes;
- DSM raster bands/elevation values are never opened here;
- no DSM TIFF SHA/hash is computed here.
"""

from __future__ import annotations

import argparse
import binascii
import http.cookiejar
import io
import json
import re
import shutil
import struct
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

SHARE_URL = "https://seafile.projekt.uni-hannover.de/f/429be50cc79d423ab6c4/"
DOWNLOAD_URL = SHARE_URL + "?dl=1"
SHARE_TOKEN = "429be50cc79d423ab6c4"
# Public dataset password published on the official ISPRS benchmark download page.
SHARE_PASSWORD = "CjwcipT4-P8g"
OFFICIAL_PAGE = "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/Default.aspx"
OFFICIAL_POTSDAM_PAGE = (
    "https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx"
)
CONSUMED = {"2_10", "3_13", "5_11", "6_14"}
DEFAULT_TILES = ("2_14", "3_14")
RGB_OUT_DIR = "2_Ortho_RGB"
DSM_OUT_DIR = "1_DSM"
OUTER_RGB_BASENAME = "2_Ortho_RGB.zip"
OUTER_DSM_BASENAME = "1_DSM.rar"
USER_AGENT = "DepthWizard-SIH26175-Potsdam-Selective/3"
ZIP_BLOCK_SIZE = 4 * 1024 * 1024
RAR_HEADER_BLOCK_SIZE = 64 * 1024
TRANSFER_CHUNK = 8 * 1024 * 1024


class AcquisitionError(RuntimeError):
    """Official Potsdam selective-acquisition failure."""


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
        raise AcquisitionError("official ISPRS Seafile share did not provide sfcsrftoken")

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
        raise AcquisitionError("official ISPRS Seafile share authentication failed")
    return opener


def _total_from_content_range(value: str | None) -> int | None:
    if not value:
        return None
    match = re.fullmatch(r"bytes\s+\d+-\d+/(\d+|\*)", value.strip(), flags=re.I)
    if not match or match.group(1) == "*":
        return None
    return int(match.group(1))


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
        if status == 206:
            response.read(1)
        total = _total_from_content_range(headers.get("Content-Range"))
        filename = _filename_from_disposition(headers.get("Content-Disposition"))
        return {
            "status_code": status,
            "range_supported": status == 206 and isinstance(total, int) and total > 0,
            "bytes": total,
            "filename": filename,
            "content_range": headers.get("Content-Range"),
            "accept_ranges": headers.get("Accept-Ranges"),
            "content_type": headers.get("Content-Type"),
            "final_host": urllib.parse.urlparse(final_url).netloc,
        }


class RangeTransport:
    """Authenticated HTTP Range transport with strict Content-Range validation."""

    def __init__(self, opener: urllib.request.OpenerDirector, total_size: int) -> None:
        self.opener = opener
        self.total_size = total_size
        self.bytes_transferred = 0
        self.range_requests = 0

    def fetch(self, start: int, end: int, *, timeout: float = 300.0) -> bytes:
        if start < 0 or end < start or end >= self.total_size:
            raise AcquisitionError(
                f"invalid HTTP range {start}-{end} for remote size {self.total_size}"
            )
        request = urllib.request.Request(
            DOWNLOAD_URL,
            headers={
                "User-Agent": USER_AGENT,
                "Referer": SHARE_URL,
                "Range": f"bytes={start}-{end}",
                "Accept-Encoding": "identity",
            },
        )
        with _open(self.opener, request, timeout=timeout) as response:
            if int(response.getcode()) != 206:
                raise AcquisitionError(
                    "official fileserver stopped honoring HTTP ranges; "
                    "refusing accidental full-share transfer"
                )
            content_range = response.headers.get("Content-Range")
            expected = f"bytes {start}-{end}/{self.total_size}"
            if content_range is None or content_range.strip().casefold() != expected.casefold():
                raise AcquisitionError(
                    f"unexpected Content-Range: expected={expected!r}, actual={content_range!r}"
                )
            data = response.read()
        expected_len = end - start + 1
        if len(data) != expected_len:
            raise AcquisitionError(
                f"short ranged read {start}-{end}: expected={expected_len}, actual={len(data)}"
            )
        self.bytes_transferred += len(data)
        self.range_requests += 1
        return data

    def copy(
        self,
        start: int,
        length: int,
        output: BinaryIO,
        *,
        chunk_size: int = TRANSFER_CHUNK,
    ) -> None:
        if length < 0:
            raise AcquisitionError(f"negative transfer length: {length}")
        remaining = length
        cursor = start
        while remaining:
            take = min(chunk_size, remaining)
            output.write(self.fetch(cursor, cursor + take - 1))
            cursor += take
            remaining -= take


class HTTPRangeReader(io.RawIOBase):
    """Seekable slice of the official remote file backed by byte ranges."""

    def __init__(
        self,
        transport: RangeTransport,
        size: int,
        *,
        base_offset: int = 0,
        block_size: int = ZIP_BLOCK_SIZE,
        max_cache_blocks: int = 4,
    ) -> None:
        super().__init__()
        if size <= 0:
            raise ValueError("slice size must be positive")
        if base_offset < 0 or base_offset + size > transport.total_size:
            raise ValueError("slice lies outside remote object")
        self.transport = transport
        self.size = size
        self.base_offset = base_offset
        self.block_size = block_size
        self.max_cache_blocks = max_cache_blocks
        self.pos = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self.pos + offset
        elif whence == io.SEEK_END:
            new_pos = self.size + offset
        else:
            raise ValueError(f"unsupported whence: {whence}")
        if new_pos < 0:
            raise ValueError("negative seek")
        self.pos = min(new_pos, self.size)
        return self.pos

    def _block(self, index: int) -> bytes:
        cached = self.cache.get(index)
        if cached is not None:
            self.cache.move_to_end(index)
            return cached
        rel_start = index * self.block_size
        if rel_start >= self.size:
            return b""
        rel_end = min(self.size - 1, rel_start + self.block_size - 1)
        data = self.transport.fetch(
            self.base_offset + rel_start,
            self.base_offset + rel_end,
        )
        self.cache[index] = data
        self.cache.move_to_end(index)
        while len(self.cache) > self.max_cache_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, size: int = -1) -> bytes:
        if self.pos >= self.size:
            return b""
        if size is None or size < 0:
            size = self.size - self.pos
        size = min(size, self.size - self.pos)
        out = bytearray()
        remaining = size
        while remaining:
            index = self.pos // self.block_size
            offset = self.pos % self.block_size
            block = self._block(index)
            take = min(remaining, len(block) - offset)
            if take <= 0:
                break
            out.extend(block[offset : offset + take])
            self.pos += take
            remaining -= take
        return bytes(out)

    def readinto(self, b: bytearray | memoryview) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def _find_outer_member(zf: zipfile.ZipFile, basename: str) -> zipfile.ZipInfo:
    matches = [
        info
        for info in zf.infolist()
        if not info.is_dir() and PurePosixPath(info.filename).name.casefold() == basename.casefold()
    ]
    if len(matches) != 1:
        raise AcquisitionError(
            f"expected exactly one outer archive member {basename!r}; "
            f"found={[info.filename for info in matches]}"
        )
    info = matches[0]
    if info.compress_type != zipfile.ZIP_STORED:
        raise AcquisitionError(
            f"outer member {info.filename} is no longer STORE-compressed; "
            "selective inner random access is unsafe"
        )
    return info


def _zip_payload_start(reader: HTTPRangeReader, info: zipfile.ZipInfo) -> int:
    reader.seek(info.header_offset)
    fixed = reader.read(30)
    if len(fixed) != 30:
        raise AcquisitionError(f"short ZIP local header for {info.filename}")
    (
        signature,
        _version,
        _flags,
        _method,
        _mtime,
        _mdate,
        _crc,
        _compressed,
        _uncompressed,
        name_len,
        extra_len,
    ) = struct.unpack("<IHHHHHIIIHH", fixed)
    if signature != 0x04034B50:
        raise AcquisitionError(
            f"invalid ZIP local-header signature for {info.filename}: 0x{signature:08x}"
        )
    return info.header_offset + 30 + name_len + extra_len


def _tile_names(tile_id: str) -> dict[str, tuple[str, str]]:
    row, col = tile_id.split("_", 1)
    r = int(row)
    c = int(col)
    return {
        "rgb": (
            f"top_potsdam_{r}_{c}_RGB.tif",
            f"top_potsdam_{r}_{c}_RGB.tfw",
        ),
        "dsm": (
            f"dsm_potsdam_{r:02d}_{c:02d}.tif",
            f"dsm_potsdam_{r:02d}_{c:02d}.tfw",
        ),
    }


def _zip_index_by_basename(zf: zipfile.ZipFile) -> dict[str, list[zipfile.ZipInfo]]:
    out: dict[str, list[zipfile.ZipInfo]] = {}
    for info in zf.infolist():
        if info.is_dir():
            continue
        out.setdefault(PurePosixPath(info.filename).name.casefold(), []).append(info)
    return out


def _require_zip_member(
    index: dict[str, list[zipfile.ZipInfo]],
    basename: str,
) -> zipfile.ZipInfo:
    matches = index.get(basename.casefold(), [])
    if len(matches) != 1:
        raise AcquisitionError(
            f"expected exactly one RGB inner member {basename!r}; "
            f"found={[info.filename for info in matches]}"
        )
    return matches[0]


def _extract_rgb_member(
    zf: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == info.file_size:
        return {
            "path": str(destination.resolve()),
            "bytes": info.file_size,
            "compressed_bytes": info.compress_size,
            "reused": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with zf.open(info) as source, temporary.open("wb") as output:
            shutil.copyfileobj(source, output, length=TRANSFER_CHUNK)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if temporary.stat().st_size != info.file_size:
        actual = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(
            f"RGB member size mismatch for {info.filename}: "
            f"expected={info.file_size}, actual={actual}"
        )
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "bytes": info.file_size,
        "compressed_bytes": info.compress_size,
        "reused": False,
    }


@dataclass(frozen=True)
class RarMember:
    name: str
    header_offset: int
    data_offset: int
    pack_size: int
    unpack_size: int
    crc32: int
    flags: int
    method: int
    unpack_version: int
    header_bytes: bytes

    @property
    def basename(self) -> str:
        return self.name.replace("\\", "/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class RarArchive:
    signature: bytes
    main_header: bytes
    endarc_header: bytes
    members: tuple[RarMember, ...]


def _read_exact(reader: HTTPRangeReader, offset: int, size: int) -> bytes:
    reader.seek(offset)
    data = reader.read(size)
    if len(data) != size:
        raise AcquisitionError(
            f"short archive read at {offset}: expected={size}, actual={len(data)}"
        )
    return data


def _decode_rar_name(raw: bytes) -> str:
    # Potsdam RAR filenames are plain ASCII. If the RAR4 Unicode flag is present,
    # the ANSI name precedes a NUL and remains sufficient for these official names.
    ansi = raw.split(b"\x00", 1)[0]
    return ansi.decode("cp437", errors="replace")


def _parse_rar4(reader: HTTPRangeReader) -> RarArchive:
    signature = _read_exact(reader, 0, 7)
    if signature != b"Rar!\x1a\x07\x00":
        raise AcquisitionError(f"expected RAR4 signature, got {signature!r}")

    pos = 7
    main_header: bytes | None = None
    endarc_header: bytes | None = None
    members: list[RarMember] = []

    for _ in range(2048):
        if pos + 7 > reader.size:
            break
        base = _read_exact(reader, pos, 7)
        _head_crc, head_type, flags, head_size = struct.unpack("<HBHH", base)
        if head_size < 7 or pos + head_size > reader.size:
            raise AcquisitionError(
                f"invalid RAR4 header at {pos}: type=0x{head_type:02x} size={head_size}"
            )
        header = _read_exact(reader, pos, head_size)

        if head_type == 0x73:  # MAIN_HEAD
            if main_header is not None:
                raise AcquisitionError("RAR4 archive contains multiple MAIN headers")
            main_header = header
            # MHD_SOLID = 0x0008. Selective extraction is valid only for non-solid RARs.
            if flags & 0x0008:
                raise AcquisitionError("official DSM RAR became solid; selective extraction refused")
            pos += head_size
            continue

        if head_type == 0x74:  # FILE_HEAD
            if head_size < 32:
                raise AcquisitionError(f"short RAR4 FILE header at {pos}: {head_size}")
            pack_low = struct.unpack_from("<I", header, 7)[0]
            unpack_low = struct.unpack_from("<I", header, 11)[0]
            file_crc = struct.unpack_from("<I", header, 16)[0]
            unpack_version = header[24]
            method = header[25]
            name_size = struct.unpack_from("<H", header, 26)[0]
            cursor = 32
            pack_high = 0
            unpack_high = 0
            if flags & 0x0100:  # LHD_LARGE
                if head_size < 40:
                    raise AcquisitionError(f"truncated RAR4 large-file header at {pos}")
                pack_high = struct.unpack_from("<I", header, 32)[0]
                unpack_high = struct.unpack_from("<I", header, 36)[0]
                cursor = 40
            if cursor + name_size > len(header):
                raise AcquisitionError(f"RAR4 filename extends beyond header at {pos}")
            name = _decode_rar_name(header[cursor : cursor + name_size])
            pack_size = pack_low | (pack_high << 32)
            unpack_size = unpack_low | (unpack_high << 32)
            data_offset = pos + head_size

            # LHD_SPLIT_BEFORE=0x0001, LHD_SPLIT_AFTER=0x0002,
            # LHD_PASSWORD=0x0004, LHD_SOLID=0x0010.
            if flags & 0x0017:
                raise AcquisitionError(
                    f"RAR member {name!r} uses split/encrypted/solid flags 0x{flags:04x}"
                )
            members.append(
                RarMember(
                    name=name,
                    header_offset=pos,
                    data_offset=data_offset,
                    pack_size=pack_size,
                    unpack_size=unpack_size,
                    crc32=file_crc,
                    flags=flags,
                    method=method,
                    unpack_version=unpack_version,
                    header_bytes=header,
                )
            )
            pos = data_offset + pack_size
            continue

        if head_type == 0x7B:  # ENDARC_HEAD
            endarc_header = header
            break

        # Generic RAR4 long block. ADD_SIZE begins at byte 7 when LONG_BLOCK is set.
        add_size = 0
        if flags & 0x8000:
            if head_size < 11:
                raise AcquisitionError(f"truncated RAR4 LONG_BLOCK header at {pos}")
            add_size = struct.unpack_from("<I", header, 7)[0]
        pos += head_size + add_size
    else:
        raise AcquisitionError("RAR4 header walk exceeded safety limit")

    if main_header is None:
        raise AcquisitionError("RAR4 archive is missing MAIN header")
    if endarc_header is None:
        raise AcquisitionError("RAR4 archive is missing ENDARC header")
    if not members:
        raise AcquisitionError("RAR4 archive contains no file members")

    return RarArchive(
        signature=signature,
        main_header=main_header,
        endarc_header=endarc_header,
        members=tuple(members),
    )


def _rar_index_by_basename(archive: RarArchive) -> dict[str, list[RarMember]]:
    out: dict[str, list[RarMember]] = {}
    for member in archive.members:
        out.setdefault(member.basename.casefold(), []).append(member)
    return out


def _require_rar_member(
    index: dict[str, list[RarMember]],
    basename: str,
) -> RarMember:
    matches = index.get(basename.casefold(), [])
    if len(matches) != 1:
        raise AcquisitionError(
            f"expected exactly one DSM RAR member {basename!r}; "
            f"found={[member.name for member in matches]}"
        )
    return matches[0]


def _minimal_endarc() -> bytes:
    body = struct.pack("<BHH", 0x7B, 0, 7)
    crc16 = binascii.crc32(body) & 0xFFFF
    return struct.pack("<H", crc16) + body


def _extract_rar_member(
    transport: RangeTransport,
    rar_absolute_start: int,
    archive: RarArchive,
    member: RarMember,
    destination: Path,
    *,
    bsdtar: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size == member.unpack_size:
        return {
            "path": str(destination.resolve()),
            "bytes": member.unpack_size,
            "packed_bytes": member.pack_size,
            "reused": True,
        }

    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)

    with tempfile.TemporaryDirectory(prefix="depthwizard-rar-member-") as temp_root:
        synthetic = Path(temp_root) / "single-member.rar"
        with synthetic.open("wb") as output:
            output.write(archive.signature)
            output.write(archive.main_header)
            output.write(member.header_bytes)
            transport.copy(
                rar_absolute_start + member.data_offset,
                member.pack_size,
                output,
            )
            # Use a minimal valid ENDARC so no archive-wide data CRC from the original
            # full RAR can incorrectly describe this synthetic one-member archive.
            output.write(_minimal_endarc())

        try:
            with temporary.open("wb") as output:
                proc = subprocess.run(
                    [bsdtar, "-xOf", str(synthetic)],
                    stdout=output,
                    stderr=subprocess.PIPE,
                    check=False,
                )
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        if proc.returncode != 0:
            temporary.unlink(missing_ok=True)
            stderr = proc.stderr.decode("utf-8", errors="replace")[-4000:]
            raise AcquisitionError(
                f"bsdtar failed extracting {member.name!r} from synthetic RAR: {stderr}"
            )

    if not temporary.is_file() or temporary.stat().st_size != member.unpack_size:
        actual = temporary.stat().st_size if temporary.exists() else -1
        temporary.unlink(missing_ok=True)
        raise AcquisitionError(
            f"DSM extracted size mismatch for {member.name}: "
            f"expected={member.unpack_size}, actual={actual}"
        )
    temporary.replace(destination)
    return {
        "path": str(destination.resolve()),
        "bytes": member.unpack_size,
        "packed_bytes": member.pack_size,
        "reused": False,
    }


def _validate_tile_id(tile_id: str) -> str:
    parts = tile_id.split("_")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise argparse.ArgumentTypeError(f"invalid Potsdam tile id: {tile_id!r}")
    normalized = f"{int(parts[0])}_{int(parts[1])}"
    if normalized in CONSUMED:
        raise argparse.ArgumentTypeError(
            f"tile {normalized} is already consumed and cannot be reused"
        )
    return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Selectively download exactly two fresh official Potsdam RGB/DSM pairs "
            "using the proven HTTP-range archive path."
        )
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
        "--report",
        type=Path,
        default=Path("workspace/final-science-data/potsdam-acquisition.json"),
    )
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help=(
            "Authenticate, inspect outer/inner RGB ZIP and DSM RAR4 headers, and report "
            "exact target metadata without downloading raster payloads."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tiles = tuple(dict.fromkeys(args.tile or DEFAULT_TILES))
    if len(tiles) != 2:
        raise SystemExit("exactly two distinct fresh Potsdam tile ids are required")

    opener = _authenticated_opener()
    probe = _probe_file_share(opener)
    if not probe.get("range_supported"):
        print(
            json.dumps(
                {
                    "status": "BLOCKED_POTSDAM_FILE_SHARE_NO_RANGE_SUPPORT",
                    "file_share": probe,
                    "reason": (
                        "Official Download-button endpoint did not honor Range: bytes=0-0; "
                        "refusing accidental full archive transfer."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 3

    total = probe.get("bytes")
    if not isinstance(total, int) or total <= 0:
        raise AcquisitionError(f"could not determine official archive size: {probe}")

    transport = RangeTransport(opener, total)
    outer_reader = HTTPRangeReader(
        transport,
        total,
        block_size=256 * 1024,
        max_cache_blocks=4,
    )
    try:
        outer_zip = zipfile.ZipFile(outer_reader)
    except zipfile.BadZipFile as exc:
        raise AcquisitionError("official Download-button object is not a valid ZIP") from exc

    with outer_zip:
        rgb_outer = _find_outer_member(outer_zip, OUTER_RGB_BASENAME)
        dsm_outer = _find_outer_member(outer_zip, OUTER_DSM_BASENAME)

        rgb_payload_start = _zip_payload_start(outer_reader, rgb_outer)
        dsm_payload_start = _zip_payload_start(outer_reader, dsm_outer)

        rgb_reader = HTTPRangeReader(
            transport,
            rgb_outer.file_size,
            base_offset=rgb_payload_start,
            block_size=ZIP_BLOCK_SIZE,
            max_cache_blocks=4,
        )
        try:
            rgb_zip = zipfile.ZipFile(rgb_reader)
        except zipfile.BadZipFile as exc:
            raise AcquisitionError("stored official 2_Ortho_RGB.zip is not a valid ZIP") from exc

        rar_reader = HTTPRangeReader(
            transport,
            dsm_outer.file_size,
            base_offset=dsm_payload_start,
            block_size=RAR_HEADER_BLOCK_SIZE,
            max_cache_blocks=8,
        )
        rar = _parse_rar4(rar_reader)

        with rgb_zip:
            rgb_index = _zip_index_by_basename(rgb_zip)
            rar_index = _rar_index_by_basename(rar)

            selected: dict[str, dict[str, Any]] = {}
            selected_rgb: dict[str, dict[str, zipfile.ZipInfo]] = {}
            selected_dsm: dict[str, dict[str, RarMember]] = {}

            for tile_id in tiles:
                names = _tile_names(tile_id)
                rgb_tif = _require_zip_member(rgb_index, names["rgb"][0])
                rgb_tfw = _require_zip_member(rgb_index, names["rgb"][1])
                dsm_tif = _require_rar_member(rar_index, names["dsm"][0])
                dsm_tfw = _require_rar_member(rar_index, names["dsm"][1])

                selected_rgb[tile_id] = {"tif": rgb_tif, "tfw": rgb_tfw}
                selected_dsm[tile_id] = {"tif": dsm_tif, "tfw": dsm_tfw}
                selected[tile_id] = {
                    "rgb": {
                        "tif": {
                            "name": rgb_tif.filename,
                            "compressed_bytes": rgb_tif.compress_size,
                            "uncompressed_bytes": rgb_tif.file_size,
                            "crc32_archive_metadata": f"{rgb_tif.CRC:08x}",
                        },
                        "tfw": {
                            "name": rgb_tfw.filename,
                            "compressed_bytes": rgb_tfw.compress_size,
                            "uncompressed_bytes": rgb_tfw.file_size,
                            "crc32_archive_metadata": f"{rgb_tfw.CRC:08x}",
                        },
                    },
                    "dsm": {
                        "tif": {
                            "name": dsm_tif.name,
                            "packed_bytes": dsm_tif.pack_size,
                            "unpacked_bytes": dsm_tif.unpack_size,
                            "crc32_archive_metadata": f"{dsm_tif.crc32:08x}",
                            "rar_header_offset": dsm_tif.header_offset,
                            "rar_data_offset": dsm_tif.data_offset,
                            "method": dsm_tif.method,
                            "unpack_version": dsm_tif.unpack_version,
                        },
                        "tfw": {
                            "name": dsm_tfw.name,
                            "packed_bytes": dsm_tfw.pack_size,
                            "unpacked_bytes": dsm_tfw.unpack_size,
                            "crc32_archive_metadata": f"{dsm_tfw.crc32:08x}",
                            "rar_header_offset": dsm_tfw.header_offset,
                            "rar_data_offset": dsm_tfw.data_offset,
                            "method": dsm_tfw.method,
                            "unpack_version": dsm_tfw.unpack_version,
                        },
                    },
                }

            inventory = {
                "schema_version": 3,
                "official_page": OFFICIAL_PAGE,
                "official_potsdam_page": OFFICIAL_POTSDAM_PAGE,
                "share_url": SHARE_URL,
                "download_button_url": DOWNLOAD_URL,
                "file_share": probe,
                "outer_archive": {
                    "bytes": total,
                    "rgb_container": {
                        "name": rgb_outer.filename,
                        "bytes": rgb_outer.file_size,
                        "compress_type": rgb_outer.compress_type,
                        "payload_absolute_start": rgb_payload_start,
                    },
                    "dsm_container": {
                        "name": dsm_outer.filename,
                        "bytes": dsm_outer.file_size,
                        "compress_type": dsm_outer.compress_type,
                        "payload_absolute_start": dsm_payload_start,
                        "rar_format": "RAR4",
                        "rar_solid": False,
                        "rar_member_count": len(rar.members),
                    },
                },
                "selected_tiles": list(tiles),
                "consumed_tiles_excluded": sorted(CONSUMED),
                "targets": selected,
                "reference_raster_values_opened_or_hashed": False,
            }

            if args.inventory_only:
                print(
                    json.dumps(
                        {
                            "status": "PASS_POTSDAM_PROVEN_SELECTIVE_INVENTORY",
                            **inventory,
                            "network_bytes_transferred_for_inventory": transport.bytes_transferred,
                            "range_requests_for_inventory": transport.range_requests,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                )
                return 0

            bsdtar = shutil.which("bsdtar")
            if not bsdtar:
                raise AcquisitionError(
                    "bsdtar is required for selective non-solid RAR4 decompression"
                )

            rgb_dir = args.dataset_root / RGB_OUT_DIR
            dsm_dir = args.dataset_root / DSM_OUT_DIR
            outputs: dict[str, Any] = {}

            for tile_id in tiles:
                names = _tile_names(tile_id)
                rgb_info = selected_rgb[tile_id]
                dsm_info = selected_dsm[tile_id]

                print(f"Acquiring fresh official Potsdam {tile_id}: RGB ...")
                rgb_tif_result = _extract_rgb_member(
                    rgb_zip, rgb_info["tif"], rgb_dir / names["rgb"][0]
                )
                rgb_tfw_result = _extract_rgb_member(
                    rgb_zip, rgb_info["tfw"], rgb_dir / names["rgb"][1]
                )

                print(f"Acquiring fresh official Potsdam {tile_id}: sealed DSM bytes ...")
                dsm_tif_result = _extract_rar_member(
                    transport,
                    dsm_payload_start,
                    rar,
                    dsm_info["tif"],
                    dsm_dir / names["dsm"][0],
                    bsdtar=bsdtar,
                )
                dsm_tfw_result = _extract_rar_member(
                    transport,
                    dsm_payload_start,
                    rar,
                    dsm_info["tfw"],
                    dsm_dir / names["dsm"][1],
                    bsdtar=bsdtar,
                )

                outputs[tile_id] = {
                    "rgb_tif": rgb_tif_result,
                    "rgb_tfw": rgb_tfw_result,
                    "reference_dsm_tif": {
                        **dsm_tif_result,
                        "raster_values_opened": False,
                        "sha256_computed": False,
                    },
                    "reference_dsm_tfw": {
                        **dsm_tfw_result,
                        "sha256_computed": False,
                    },
                }

    report = {
        "status": "PASS_POTSDAM_OFFICIAL_FRESH_PAIR_ACQUISITION",
        **inventory,
        "outputs": outputs,
        "network_bytes_transferred": transport.bytes_transferred,
        "range_requests": transport.range_requests,
        "full_outer_archive_downloaded": False,
        "full_rgb_inner_archive_downloaded": False,
        "full_dsm_rar_downloaded": False,
        "reference_raster_values_opened_or_hashed": False,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
