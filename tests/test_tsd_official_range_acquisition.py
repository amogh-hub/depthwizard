from __future__ import annotations

import binascii
import io
import struct
import zipfile
from pathlib import Path
from typing import Any, cast

from depthwizard.height_model.terrain_structure_split import (
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
)
from qualification.download_tsd_potsdam_official_ranges import (
    RAR4_SIGNATURE,
    RAR_BLOCK_END,
    RAR_BLOCK_FILE,
    RAR_BLOCK_MAIN,
    RAR_LONG_BLOCK,
    RemoteView,
    SeekableRemote,
    _NoCloseBufferedReader,
    parse_rar4_members,
    required_physical_files,
)
from qualification.plan_tsd_potsdam_acquisition import build_acquisition_plan


def _inventory_payload(dataset_root: Path) -> dict[str, object]:
    records = [
        {
            "tile_id": tile_id,
            "rgb_count": 0,
            "dsm_count": 0,
            "label_count": 0,
            "status": "MISSING_RGB_DSM_LABEL",
        }
        for tile_id in sorted(SUPERVISION_ELIGIBLE_TILE_IDS)
    ]
    for record in records:
        if record["tile_id"] == "2_10":
            record["rgb_count"] = 1
            record["dsm_count"] = 1
            record["status"] = "MISSING_LABEL"
        if record["tile_id"] == "5_11":
            record["rgb_count"] = 1
            record["dsm_count"] = 1
            record["status"] = "MISSING_LABEL"
    return {
        "schema_version": 3,
        "mode": "filename_metadata_only",
        "protocol_version": TSD_SPLIT_PROTOCOL_VERSION,
        "dataset_root": str(dataset_root),
        "supervision_eligible_tile_ids": sorted(SUPERVISION_ELIGIBLE_TILE_IDS),
        "tiles": records,
    }


def test_physical_transport_adds_mandatory_world_files_without_expanding_tile_scope(
    tmp_path: Path,
) -> None:
    plan = build_acquisition_plan(_inventory_payload(tmp_path))
    logical = plan["required_files"]
    assert isinstance(logical, list)

    from qualification.download_tsd_potsdam_official_ranges import PlannedFile

    planned = tuple(PlannedFile(**cast(dict[str, Any], item)) for item in logical)
    physical = required_physical_files(planned)

    assert len(planned) == 37
    assert len(physical) == 61
    assert sum(item.expected_filename.endswith(".tfw") for item in physical) == 24
    assert sum(item.component == "rgb" for item in physical) == 24
    assert sum(item.component == "dsm" for item in physical) == 24
    assert sum(item.component == "label" for item in physical) == 13
    assert not any(item.tile_id in {"2_14", "3_14", "4_12", "6_12"} for item in physical)


class _MemoryRangeSource:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def fetch(self, start: int, end: int) -> bytes:
        return self.payload[start : end + 1]


def test_remote_seekable_zip_reader_reads_only_from_view() -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("official/top_potsdam_6_7_RGB.tfw", b"world-file")
    payload = buffer.getvalue()
    source = cast(Any, _MemoryRangeSource(payload))
    view = RemoteView(source, 0, len(payload))
    stream = _NoCloseBufferedReader(SeekableRemote(view))

    with zipfile.ZipFile(stream, "r") as archive:
        assert archive.namelist() == ["official/top_potsdam_6_7_RGB.tfw"]
        # Acquisition metadata access must not require decoding the raster payload itself.
        info = archive.getinfo("official/top_potsdam_6_7_RGB.tfw")
        assert info.file_size == len(b"world-file")


def _rar_header(head_type: int, flags: int, body: bytes) -> bytes:
    head_size = 7 + len(body)
    without_crc = bytes([head_type]) + struct.pack("<HH", flags, head_size) + body
    crc16 = binascii.crc32(without_crc) & 0xFFFF
    return struct.pack("<H", crc16) + without_crc


def test_rar4_header_walk_skips_packed_payload_without_decoding() -> None:
    main = _rar_header(RAR_BLOCK_MAIN, 0, b"\x00" * 6)
    name = b"1_DSM\\dsm_potsdam_06_07.tif"
    packed = b"PACKED-DATA-NOT-A-RASTER"
    file_body = (
        struct.pack("<I", len(packed))
        + struct.pack("<I", 145_000_000)
        + b"\x03"
        + struct.pack("<I", 0x12345678)
        + b"\x00" * 4
        + bytes([29, 0x33])
        + struct.pack("<H", len(name))
        + struct.pack("<I", 0x20)
        + name
    )
    file_header = _rar_header(RAR_BLOCK_FILE, RAR_LONG_BLOCK, file_body)
    end = _rar_header(RAR_BLOCK_END, 0, b"")
    payload = RAR4_SIGNATURE + main + file_header + packed + end
    source = cast(Any, _MemoryRangeSource(payload))
    view = RemoteView(source, 0, len(payload))

    parsed_main, parsed_end, members = parse_rar4_members(view)

    assert parsed_main == main
    assert parsed_end == end
    assert len(members) == 1
    member = members[0]
    assert member.name == "1_DSM/dsm_potsdam_06_07.tif"
    assert member.packed_bytes == len(packed)
    assert member.unpacked_bytes == 145_000_000
    assert member.method == 0x33
    assert member.unpack_version == 29
    assert member.data_offset == len(RAR4_SIGNATURE) + len(main) + len(file_header)
