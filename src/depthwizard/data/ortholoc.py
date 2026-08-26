from __future__ import annotations

import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

ORTHOLOC_BASE_URL = "https://cvg.cit.tum.de/webshare/g/papers/Dhaouadi/OrthoLoC"
USER_AGENT = "DepthWizard-SIH26175/0.3"

_LOCATION_RE = re.compile(r"^(L\d+)_.*\.tif$", re.IGNORECASE)


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


@dataclass(frozen=True)
class OrthoLoCRemoteScene:
    scene_id: str
    split: str
    location_id: str
    dop_url: str
    dsm_url: str


def parse_apache_tif_listing(html: str) -> list[str]:
    """Extract safe TIFF basenames from the OrthoLoC Apache directory listing."""
    parser = _HrefParser()
    parser.feed(html)
    names: set[str] = set()
    for href in parser.hrefs:
        parsed = urllib.parse.urlparse(href)
        name = Path(urllib.parse.unquote(parsed.path)).name
        if not name or name in {".", ".."}:
            continue
        if not name.lower().endswith((".tif", ".tiff")):
            continue
        if Path(name).name != name:
            continue
        names.add(name)
    return sorted(names)


def location_id_from_filename(filename: str) -> str:
    match = _LOCATION_RE.match(Path(filename).name)
    if match is None:
        raise ValueError(f"unrecognized OrthoLoC scene filename: {filename}")
    return match.group(1).upper()


def pair_scene_names(dop_names: list[str], dsm_names: list[str]) -> list[str]:
    """Return exact DOP/DSM filename intersections; mismatches are intentionally excluded."""
    return sorted(set(dop_names).intersection(dsm_names))


def select_geographic_scenes(
    scene_names: list[str],
    *,
    max_locations: int,
    samples_per_location: int = 1,
    excluded_locations: set[str] | None = None,
) -> list[str]:
    """Select deterministic samples from distinct location IDs without geographic leakage."""
    if max_locations <= 0 or samples_per_location <= 0:
        raise ValueError("max_locations and samples_per_location must be positive")
    excluded = {value.upper() for value in (excluded_locations or set())}
    grouped: dict[str, list[str]] = {}
    for name in sorted(scene_names):
        location = location_id_from_filename(name)
        if location in excluded:
            continue
        grouped.setdefault(location, []).append(name)

    selected: list[str] = []
    for location in sorted(grouped)[:max_locations]:
        selected.extend(grouped[location][:samples_per_location])
    if len({location_id_from_filename(name) for name in selected}) < max_locations:
        raise ValueError(
            f"requested {max_locations} geographic groups but only "
            f"{len({location_id_from_filename(name) for name in selected})} were available"
        )
    return selected


def _fetch_text(url: str, *, timeout_s: float = 60.0) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read().decode("utf-8", errors="strict")


def discover_remote_scenes(split: str) -> list[OrthoLoCRemoteScene]:
    """Discover matched unpacked DOP/DSM GeoTIFF pairs from an official OrthoLoC split."""
    if split not in {"train", "val", "test_inPlace", "test_outPlace"}:
        raise ValueError(f"unsupported OrthoLoC split: {split}")
    split_root = f"{ORTHOLOC_BASE_URL}/unpacked/{split}"
    dop_root = f"{split_root}/DOPs"
    dsm_root = f"{split_root}/DSMs"
    names = pair_scene_names(
        parse_apache_tif_listing(_fetch_text(f"{dop_root}/")),
        parse_apache_tif_listing(_fetch_text(f"{dsm_root}/")),
    )
    return [
        OrthoLoCRemoteScene(
            scene_id=Path(name).stem,
            split=split,
            location_id=location_id_from_filename(name),
            dop_url=f"{dop_root}/{urllib.parse.quote(name)}",
            dsm_url=f"{dsm_root}/{urllib.parse.quote(name)}",
        )
        for name in names
    ]


def download_file(url: str, destination: Path, *, timeout_s: float = 180.0) -> Path:
    """Atomically download one official dataset file and reuse valid cached content."""
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        if temporary.stat().st_size <= 0:
            raise RuntimeError(f"empty download: {url}")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
