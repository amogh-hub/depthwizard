#!/usr/bin/env python3
"""Resolve common NEON RGB/LiDAR acquisition months without downloading data.

This is an operations helper for the frozen DepthWizard final-science campaign.
It queries only public NEON metadata through GraphQL. Data-file download itself may
require a NEON API token and is intentionally a separate step.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass

GRAPHQL_URL = "https://data.neonscience.org/graphql"
RGB_PRODUCT = "DP3.30010.001"
DSM_PRODUCT = "DP3.30024.001"
DEFAULT_SITES = ("CPER", "NIWO", "HARV")

QUERY = r"""
query Site($siteCode: String!) {
  site(siteCode: $siteCode) {
    siteCode
    siteName
    dataProducts {
      dataProductCode
      dataProductTitle
      availableMonths
      availableDataUrls
      availableReleases {
        release
        availableMonths
      }
    }
  }
}
"""


@dataclass(frozen=True)
class ProductAvailability:
    code: str
    months: tuple[str, ...]
    urls: tuple[str, ...]
    releases: tuple[dict[str, object], ...]

    def url_for_month(self, month: str) -> str | None:
        try:
            index = self.months.index(month)
        except ValueError:
            return None
        return self.urls[index] if index < len(self.urls) else None

    def releases_for_month(self, month: str) -> list[str]:
        matches: list[str] = []
        for item in self.releases:
            release = item.get("release")
            months = item.get("availableMonths")
            if isinstance(release, str) and isinstance(months, list) and month in months:
                matches.append(release)
        return sorted(set(matches))


def post_graphql(site: str, timeout: float) -> dict[str, object]:
    payload = json.dumps({"query": QUERY, "variables": {"siteCode": site}}).encode("utf-8")
    request = urllib.request.Request(
        GRAPHQL_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "DepthWizard-SIH26175-evidence-kit/1",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"NEON GraphQL HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"NEON GraphQL request failed: {exc}") from exc
    decoded = json.loads(raw)
    if not isinstance(decoded, dict):
        raise RuntimeError("NEON GraphQL response root is not an object")
    if decoded.get("errors"):
        raise RuntimeError(f"NEON GraphQL returned errors: {decoded['errors']!r}")
    return decoded


def availability(product: object, expected_code: str) -> ProductAvailability:
    if not isinstance(product, dict) or product.get("dataProductCode") != expected_code:
        raise ValueError(f"invalid availability payload for {expected_code}")
    months = product.get("availableMonths")
    urls = product.get("availableDataUrls")
    releases = product.get("availableReleases")
    if not isinstance(months, list) or not all(isinstance(x, str) for x in months):
        raise ValueError(f"{expected_code}: availableMonths is invalid")
    if not isinstance(urls, list) or not all(isinstance(x, str) for x in urls):
        raise ValueError(f"{expected_code}: availableDataUrls is invalid")
    if releases is None:
        releases = []
    if not isinstance(releases, list) or not all(isinstance(x, dict) for x in releases):
        raise ValueError(f"{expected_code}: availableReleases is invalid")
    return ProductAvailability(
        code=expected_code,
        months=tuple(months),
        urls=tuple(urls),
        releases=tuple(releases),
    )


def resolve_site(site: str, timeout: float) -> dict[str, object]:
    response = post_graphql(site, timeout)
    data = response.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"{site}: GraphQL response is missing data")
    site_payload = data.get("site")
    if not isinstance(site_payload, dict):
        raise RuntimeError(f"{site}: site metadata was not returned")
    products = site_payload.get("dataProducts")
    if not isinstance(products, list):
        raise RuntimeError(f"{site}: dataProducts is missing")

    by_code = {
        item.get("dataProductCode"): item
        for item in products
        if isinstance(item, dict) and isinstance(item.get("dataProductCode"), str)
    }
    if RGB_PRODUCT not in by_code:
        raise RuntimeError(f"{site}: RGB product {RGB_PRODUCT} is unavailable")
    if DSM_PRODUCT not in by_code:
        raise RuntimeError(f"{site}: LiDAR DSM product {DSM_PRODUCT} is unavailable")

    rgb = availability(by_code[RGB_PRODUCT], RGB_PRODUCT)
    dsm = availability(by_code[DSM_PRODUCT], DSM_PRODUCT)
    common = sorted(set(rgb.months) & set(dsm.months))
    if not common:
        raise RuntimeError(f"{site}: no common RGB/LiDAR acquisition month exists")

    candidates: list[dict[str, object]] = []
    for month in reversed(common):
        rgb_releases = rgb.releases_for_month(month)
        dsm_releases = dsm.releases_for_month(month)
        common_releases = sorted(set(rgb_releases) & set(dsm_releases), reverse=True)
        candidates.append(
            {
                "month": month,
                "rgb_metadata_url": rgb.url_for_month(month),
                "dsm_metadata_url": dsm.url_for_month(month),
                "rgb_releases": rgb_releases,
                "dsm_releases": dsm_releases,
                "common_releases": common_releases,
            }
        )

    return {
        "site_code": site_payload.get("siteCode", site),
        "site_name": site_payload.get("siteName"),
        "rgb_product": RGB_PRODUCT,
        "dsm_product": DSM_PRODUCT,
        "common_month_count": len(common),
        "latest_common_month": common[-1],
        "candidates_newest_first": candidates,
    }


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="List common NEON RGB/LiDAR months for DepthWizard final science."
    )
    p.add_argument(
        "--site",
        dest="sites",
        action="append",
        help="NEON site code; repeatable. Defaults to CPER, NIWO and HARV.",
    )
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument(
        "--all-candidates",
        action="store_true",
        help="Print all common months; default output keeps only the newest five per site.",
    )
    return p


def main() -> int:
    args = parser().parse_args()
    sites = tuple(dict.fromkeys((args.sites or list(DEFAULT_SITES))))
    report: dict[str, object] = {
        "schema_version": 1,
        "purpose": "DepthWizard final-science NEON RGB/LiDAR acquisition matching",
        "rgb_product": RGB_PRODUCT,
        "dsm_product": DSM_PRODUCT,
        "sites": [],
    }
    output_sites: list[dict[str, object]] = []
    for raw_site in sites:
        site = raw_site.strip().upper()
        if len(site) != 4 or not site.isalnum():
            raise SystemExit(f"invalid NEON site code: {raw_site!r}")
        resolved = resolve_site(site, args.timeout)
        if not args.all_candidates:
            candidates = resolved.get("candidates_newest_first")
            if isinstance(candidates, list):
                resolved["candidates_newest_first"] = candidates[:5]
        output_sites.append(resolved)
    report["sites"] = output_sites
    json.dump(report, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
