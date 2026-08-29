#!/usr/bin/env python3
"""Download one deterministic, co-acquired NEON RGB/DSM tile per final-science site.

This helper is deliberately reference-safe. It uses only NEON metadata and filenames to
select the paired tile, downloads the reference DSM bytes, but never opens or hashes the DSM.
The final reference evaluator must remain the first code path that decodes reference heights.

Required environment:
  NEON_API_TOKEN   NEON Data API token (never written to disk or stdout)

Default sites:
  CPER -> sparse test
  NIWO -> hilly test
  HARV -> forested test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pyproj import Transformer

API_BASE = "https://data.neonscience.org/api/v0"
GRAPHQL_URL = "https://data.neonscience.org/graphql"
RGB_PRODUCT = "DP3.30010.001"
DSM_PRODUCT = "DP3.30024.001"
DEFAULT_RELEASE = "RELEASE-2026"
DEFAULT_SITES = {
    "CPER": "sparse",
    "NIWO": "hilly",
    "HARV": "forested",
}
RGB_RE = re.compile(r"_(?P<easting>\d{5,8})_(?P<northing>\d{5,9})_image\.tif$", re.I)
DSM_RE = re.compile(r"_(?P<easting>\d{5,8})_(?P<northing>\d{5,9})_DSM\.tif$", re.I)
DOWNLOAD_ATTEMPTS = 6
DOWNLOAD_CHUNK = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 300

GRAPHQL_QUERY = r"""
query Site($siteCode: String!) {
  site(siteCode: $siteCode) {
    siteCode
    siteName
    siteLatitude
    siteLongitude
    dataProducts {
      dataProductCode
      availableMonths
      availableReleases {
        release
        availableMonths
      }
    }
  }
}
"""


class NeonEvidenceError(RuntimeError):
    """NEON evidence acquisition contract violation."""


def _request_json(url: str, *, token: str | None = None, payload: bytes | None = None) -> dict[str, Any]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "DepthWizard-SIH26175-evidence-kit/1",
    }
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["X-API-Token"] = token
    request = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:2000]
        raise NeonEvidenceError(f"NEON HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise NeonEvidenceError(f"NEON request failed for {url}: {exc}") from exc
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise NeonEvidenceError(f"NEON response root is not an object: {url}")
    if decoded.get("errors"):
        raise NeonEvidenceError(f"NEON API returned errors for {url}: {decoded['errors']!r}")
    return decoded


def _site_metadata(site: str) -> dict[str, Any]:
    payload = json.dumps(
        {"query": GRAPHQL_QUERY, "variables": {"siteCode": site}},
        separators=(",", ":"),
    ).encode("utf-8")
    decoded = _request_json(GRAPHQL_URL, payload=payload)
    data = decoded.get("data")
    site_payload = data.get("site") if isinstance(data, dict) else None
    if not isinstance(site_payload, dict):
        raise NeonEvidenceError(f"{site}: NEON site metadata is unavailable")
    return site_payload


def _product_info(site_payload: dict[str, Any], code: str) -> dict[str, Any]:
    products = site_payload.get("dataProducts")
    if not isinstance(products, list):
        raise NeonEvidenceError("NEON site metadata is missing dataProducts")
    for product in products:
        if isinstance(product, dict) and product.get("dataProductCode") == code:
            return product
    raise NeonEvidenceError(f"site has no availability for {code}")


def _released_months(product: dict[str, Any], release: str) -> set[str]:
    released = product.get("availableReleases")
    if not isinstance(released, list):
        return set()
    for item in released:
        if not isinstance(item, dict) or item.get("release") != release:
            continue
        months = item.get("availableMonths")
        if isinstance(months, list):
            return {str(month) for month in months if isinstance(month, str)}
    return set()


def _choose_month(site_payload: dict[str, Any], release: str) -> str:
    rgb = _product_info(site_payload, RGB_PRODUCT)
    dsm = _product_info(site_payload, DSM_PRODUCT)
    rgb_release = _released_months(rgb, release)
    dsm_release = _released_months(dsm, release)
    common_release = sorted(rgb_release & dsm_release)
    if common_release:
        return common_release[-1]

    def all_months(product: dict[str, Any]) -> set[str]:
        value = product.get("availableMonths")
        if not isinstance(value, list):
            return set()
        return {str(month) for month in value if isinstance(month, str)}

    common = sorted(all_months(rgb) & all_months(dsm))
    if not common:
        raise NeonEvidenceError("no common RGB/LiDAR acquisition month exists")
    return common[-1]


def _file_list(product: str, site: str, month: str, release: str, token: str) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"package": "basic", "release": release})
    url = f"{API_BASE}/data/{product}/{site}/{month}?{query}"
    payload = _request_json(url, token=token)
    data = payload.get("data")
    if not isinstance(data, dict):
        raise NeonEvidenceError(f"{product} {site} {month}: response is missing data")
    files = data.get("files")
    if not isinstance(files, list):
        raise NeonEvidenceError(f"{product} {site} {month}: response is missing files")
    return [item for item in files if isinstance(item, dict)]


def _file_name(item: dict[str, Any]) -> str | None:
    for key in ("name", "fileName", "filename"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return Path(value).name
    return None


def _file_url(item: dict[str, Any]) -> str | None:
    for key in ("url", "fileUrl", "downloadUrl"):
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("https://", "http://")):
            return value
    return None


def _file_size(item: dict[str, Any]) -> int | None:
    for key in ("size", "fileSize", "sizeBytes"):
        value = item.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def _coord_index(files: list[dict[str, Any]], regex: re.Pattern[str]) -> dict[tuple[int, int], dict[str, Any]]:
    result: dict[tuple[int, int], dict[str, Any]] = {}
    for item in files:
        name = _file_name(item)
        if name is None:
            continue
        match = regex.search(name)
        if match is None:
            continue
        key = (int(match.group("easting")), int(match.group("northing")))
        if key in result:
            raise NeonEvidenceError(f"duplicate NEON raster coordinate in response: {key}")
        result[key] = item
    return result


def _utm_epsg(latitude: float, longitude: float) -> int:
    zone = int(math.floor((longitude + 180.0) / 6.0)) + 1
    zone = max(1, min(60, zone))
    return (32600 if latitude >= 0.0 else 32700) + zone


def _select_pair(
    *,
    rgb_files: list[dict[str, Any]],
    dsm_files: list[dict[str, Any]],
    latitude: float,
    longitude: float,
) -> tuple[tuple[int, int], dict[str, Any], dict[str, Any], int]:
    rgb = _coord_index(rgb_files, RGB_RE)
    dsm = _coord_index(dsm_files, DSM_RE)
    common = sorted(set(rgb) & set(dsm))
    if not common:
        raise NeonEvidenceError("NEON RGB and DSM responses contain no common 1 km tile coordinate")
    epsg = _utm_epsg(latitude, longitude)
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    site_e, site_n = transformer.transform(longitude, latitude)
    selected = min(
        common,
        key=lambda coord: (coord[0] + 500.0 - site_e) ** 2 + (coord[1] + 500.0 - site_n) ** 2,
    )
    return selected, rgb[selected], dsm[selected], epsg


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_headers(url: str, token: str, *, start: int) -> dict[str, str]:
    headers = {
        "User-Agent": "DepthWizard-SIH26175-evidence-kit/2",
        "Accept-Encoding": "identity",
    }
    if urllib.parse.urlparse(url).netloc.endswith("neonscience.org"):
        headers["X-API-Token"] = token
    if start > 0:
        headers["Range"] = f"bytes={start}-"
    return headers


def _retryable_http(code: int) -> bool:
    return code in {408, 425, 429, 500, 502, 503, 504}


def _download(item: dict[str, Any], destination: Path, token: str) -> dict[str, Any]:
    """Download one NEON file with bounded retries and byte-range resume.

    Completed destination files are reused only when their size matches NEON metadata when
    available. Interrupted transfers remain as ``.part`` files. A retry requests the remaining
    byte range; if the server ignores Range and returns HTTP 200, the partial file is safely
    truncated and that response is treated as a clean restart rather than appended corruption.
    Reference DSM bytes are still never opened or hashed by this function.
    """
    name = _file_name(item)
    url = _file_url(item)
    if name is None or url is None:
        raise NeonEvidenceError("NEON file entry is missing name/url")
    expected = _file_size(item)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        actual = destination.stat().st_size
        if actual > 0 and (expected is None or actual == expected):
            return {"path": str(destination.resolve()), "bytes": actual, "reused": True}
        destination.unlink(missing_ok=True)

    tmp = destination.with_suffix(destination.suffix + ".part")
    if tmp.is_file() and expected is not None and tmp.stat().st_size > expected:
        tmp.unlink(missing_ok=True)

    last_error: BaseException | None = None
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        start = tmp.stat().st_size if tmp.is_file() else 0
        if expected is not None and start == expected and start > 0:
            tmp.replace(destination)
            return {
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "reused": False,
                "resumed": True,
                "attempts": attempt - 1,
            }

        request = urllib.request.Request(
            url,
            headers=_download_headers(url, token, start=start),
        )
        try:
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
                status = int(response.getcode())
                mode = "ab" if start > 0 and status == 206 else "wb"
                if start > 0 and status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {start}-"):
                        raise NeonEvidenceError(
                            f"invalid NEON resume Content-Range for {name}: {content_range!r}"
                        )
                elif start > 0 and status == 200:
                    print(
                        f"NEON server ignored resume range for {name}; restarting this file from byte 0.",
                        file=sys.stderr,
                    )
                    start = 0
                    mode = "wb"
                elif status not in {200, 206}:
                    raise NeonEvidenceError(f"unexpected NEON download HTTP {status} for {name}")

                with tmp.open(mode) as output:
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK)
                        if not chunk:
                            break
                        output.write(chunk)

            actual = tmp.stat().st_size if tmp.is_file() else 0
            if actual <= 0:
                raise NeonEvidenceError(f"download produced an empty file: {name}")
            if expected is not None and actual != expected:
                raise NeonEvidenceError(
                    f"download size mismatch for {name}: expected={expected}, actual={actual}"
                )
            tmp.replace(destination)
            return {
                "path": str(destination.resolve()),
                "bytes": destination.stat().st_size,
                "reused": False,
                "resumed": start > 0,
                "attempts": attempt,
            }

        except urllib.error.HTTPError as exc:
            last_error = exc
            if not _retryable_http(exc.code):
                detail = exc.read().decode("utf-8", errors="replace")[:2000]
                raise NeonEvidenceError(
                    f"NEON HTTP {exc.code} downloading {name}: {detail}"
                ) from exc
        except NeonEvidenceError:
            raise
        except (BrokenPipeError, ConnectionResetError, TimeoutError, urllib.error.URLError, OSError) as exc:
            last_error = exc

        if attempt >= DOWNLOAD_ATTEMPTS:
            break
        preserved = tmp.stat().st_size if tmp.is_file() else 0
        delay = min(30, 2 ** (attempt - 1))
        print(
            f"Transient NEON transfer interruption for {name}; preserving {preserved} bytes and "
            f"retrying in {delay}s ({attempt}/{DOWNLOAD_ATTEMPTS}).",
            file=sys.stderr,
        )
        time.sleep(delay)

    preserved = tmp.stat().st_size if tmp.is_file() else 0
    raise NeonEvidenceError(
        f"NEON download failed after {DOWNLOAD_ATTEMPTS} attempts for {name}; "
        f"preserved_partial_bytes={preserved}; last_error={last_error!r}"
    ) from last_error


def _number(value: object, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeonEvidenceError(f"{context} is missing or non-numeric")
    return float(value)


def acquire_site(site: str, terrain: str, release: str, root: Path, token: str) -> dict[str, Any]:
    metadata = _site_metadata(site)
    month = _choose_month(metadata, release)
    latitude = _number(metadata.get("siteLatitude"), f"{site}.siteLatitude")
    longitude = _number(metadata.get("siteLongitude"), f"{site}.siteLongitude")
    rgb_files = _file_list(RGB_PRODUCT, site, month, release, token)
    dsm_files = _file_list(DSM_PRODUCT, site, month, release, token)
    coord, rgb_item, dsm_item, epsg = _select_pair(
        rgb_files=rgb_files,
        dsm_files=dsm_files,
        latitude=latitude,
        longitude=longitude,
    )
    easting, northing = coord
    scene_dir = root / site / month / f"{easting}_{northing}"
    rgb_name = _file_name(rgb_item)
    dsm_name = _file_name(dsm_item)
    assert rgb_name is not None and dsm_name is not None
    rgb_download = _download(rgb_item, scene_dir / "rgb" / rgb_name, token)
    dsm_download = _download(dsm_item, scene_dir / "reference-sealed" / dsm_name, token)
    rgb_path = Path(str(rgb_download["path"]))
    return {
        "site": site,
        "site_name": metadata.get("siteName"),
        "terrain": terrain,
        "release": release,
        "month": month,
        "site_latitude": latitude,
        "site_longitude": longitude,
        "selected_tile": {"easting": easting, "northing": northing, "utm_epsg": epsg},
        "rgb_product": RGB_PRODUCT,
        "reference_product": DSM_PRODUCT,
        "rgb": {
            **rgb_download,
            "filename": rgb_name,
            "sha256": _sha256(rgb_path),
        },
        "reference": {
            **dsm_download,
            "filename": dsm_name,
            "sha256": None,
            "opened_or_hashed": False,
            "claim_boundary": "Reference bytes downloaded only; elevation values were not opened and SHA-256 was intentionally not computed before prediction freeze.",
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Acquire deterministic NEON RGB/DSM final-science tiles.")
    parser.add_argument("--output-root", type=Path, default=Path("workspace/final-science-data/neon"))
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument(
        "--site",
        action="append",
        help="Optional four-letter site code. Repeatable. Defaults to CPER, NIWO, HARV.",
    )
    parser.add_argument("--report", type=Path, default=Path("workspace/final-science-data/neon-acquisition.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    token = os.environ.get("NEON_API_TOKEN", "").strip()
    if not token:
        raise SystemExit("NEON_API_TOKEN is required. Create it in the NEON portal and export it only in your shell; never commit it.")
    sites = args.site or list(DEFAULT_SITES)
    unique_sites = list(dict.fromkeys(site.strip().upper() for site in sites))
    output: list[dict[str, Any]] = []
    for site in unique_sites:
        if len(site) != 4 or not site.isalnum():
            raise SystemExit(f"invalid NEON site code: {site!r}")
        terrain = DEFAULT_SITES.get(site)
        if terrain is None:
            raise SystemExit(f"site {site!r} has no frozen terrain assignment; allowed={sorted(DEFAULT_SITES)}")
        print(f"Acquiring {site} ({terrain}) ...", file=sys.stderr)
        output.append(acquire_site(site, terrain, args.release, args.output_root, token))
    report = {
        "schema_version": 1,
        "status": "PASS_NEON_FINAL_SCIENCE_ACQUISITION",
        "reference_values_opened_or_hashed": False,
        "release": args.release,
        "scenes": output,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report), "scene_count": len(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())