from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import tomllib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, order=True)
class Component:
    ecosystem: str
    name: str
    version: str
    license: str
    source: str
    integrity: str | None = None

    @property
    def bom_ref(self) -> str:
        # Package URLs retain npm scopes/namespaces while escaping reserved characters. Using the
        # same standards-based identity for ``bom-ref`` also guarantees stable cross-report joins.
        package_name = quote(self.name, safe="/")
        package_version = quote(self.version, safe=".+-~")
        return f"pkg:{self.ecosystem}/{package_name}@{package_version}"


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"TOML root must be an object: {path}")
    return payload


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _python_components() -> list[Component]:
    packages = _read_toml(ROOT / "uv.lock").get("package")
    if not isinstance(packages, list):
        raise TypeError("uv.lock has no package array")
    result: list[Component] = []
    for package in packages:
        if not isinstance(package, dict):
            continue
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        source = package.get("source")
        source_label = json.dumps(source, sort_keys=True) if isinstance(source, dict) else "unknown"
        result.append(Component("pypi", name, version, "NOASSERTION", source_label))
    return result


def _npm_name(path: str, package: dict[str, Any]) -> str | None:
    explicit = package.get("name")
    if isinstance(explicit, str) and explicit:
        return explicit
    marker = "node_modules/"
    if marker not in path:
        return None
    tail = path.rsplit(marker, 1)[1]
    if tail.startswith("@") and "/" in tail:
        scope, name = tail.split("/", 1)
        return f"{scope}/{name.split('/', 1)[0]}"
    return tail.split("/", 1)[0]


def _npm_components() -> list[Component]:
    path = ROOT / "apps" / "desktop" / "package-lock.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    packages = payload.get("packages")
    if not isinstance(packages, dict):
        raise TypeError("package-lock.json has no packages object")
    result: list[Component] = []
    for package_path, raw in packages.items():
        if not isinstance(package_path, str) or not isinstance(raw, dict):
            continue
        name = _npm_name(package_path, raw)
        version = raw.get("version")
        if name is None or not isinstance(version, str):
            continue
        license_name = raw.get("license")
        resolved = raw.get("resolved")
        integrity = raw.get("integrity")
        result.append(
            Component(
                "npm",
                name,
                version,
                license_name if isinstance(license_name, str) else "NOASSERTION",
                resolved if isinstance(resolved, str) else "package-lock",
                integrity if isinstance(integrity, str) else None,
            )
        )
    return result


def _cargo_components() -> list[Component]:
    packages = _read_toml(ROOT / "apps" / "desktop" / "src-tauri" / "Cargo.lock").get("package")
    if not isinstance(packages, list):
        raise TypeError("Cargo.lock has no package array")
    result: list[Component] = []
    for package in packages:
        if not isinstance(package, dict):
            continue
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        source = package.get("source")
        checksum = package.get("checksum")
        result.append(
            Component(
                "cargo",
                name,
                version,
                "NOASSERTION",
                source if isinstance(source, str) else "workspace/path",
                checksum if isinstance(checksum, str) else None,
            )
        )
    return result


def collect_components() -> list[Component]:
    components = _python_components() + _npm_components() + _cargo_components()
    return sorted(set(components))


def _timestamp() -> str:
    raw_epoch = os.environ.get("SOURCE_DATE_EPOCH")
    if raw_epoch is None:
        raw_epoch = _git_value("show", "-s", "--format=%ct", "HEAD")
    return datetime.fromtimestamp(int(raw_epoch), tz=UTC).isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cyclonedx_document(components: list[Component], git_sha: str) -> dict[str, Any]:
    serial = uuid.uuid5(uuid.NAMESPACE_URL, f"depthwizard:{git_sha}")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{serial}",
        "version": 1,
        "metadata": {
            "timestamp": _timestamp(),
            "component": {
                "type": "application",
                "bom-ref": f"depthwizard@{git_sha}",
                "name": "DepthWizard",
                "version": git_sha,
            },
            "tools": [{"vendor": "DepthWizard", "name": "generate_supply_chain_reports.py"}],
            "properties": [{"name": "depthwizard:git_sha", "value": git_sha}],
        },
        "components": [
            {
                "type": "library",
                "bom-ref": item.bom_ref,
                "group": item.ecosystem,
                "name": item.name,
                "version": item.version,
                "purl": item.bom_ref,
                "licenses": (
                    [{"expression": item.license}] if item.license != "NOASSERTION" else []
                ),
                "externalReferences": (
                    [{"type": "distribution", "url": item.source}]
                    if item.source.startswith(("http://", "https://"))
                    else []
                ),
                "properties": [
                    {"name": "depthwizard:source", "value": item.source},
                    {"name": "depthwizard:integrity", "value": item.integrity or "unavailable"},
                    {"name": "depthwizard:license", "value": item.license},
                ],
            }
            for item in components
        ],
    }


def generate(output_dir: Path) -> dict[str, Any]:
    git_sha = _git_value("rev-parse", "HEAD")
    components = collect_components()
    output_dir.mkdir(parents=True, exist_ok=True)
    sbom_path = output_dir / "depthwizard-sbom.cdx.json"
    licenses_path = output_dir / "dependency-license-report.csv"
    manifest_path = output_dir / "supply-chain-manifest.json"

    sbom_path.write_text(
        json.dumps(_cyclonedx_document(components, git_sha), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with licenses_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("ecosystem", "name", "version", "license", "source", "integrity"))
        for item in components:
            writer.writerow(
                (
                    item.ecosystem,
                    item.name,
                    item.version,
                    item.license,
                    item.source,
                    item.integrity or "",
                )
            )

    counts: dict[str, int] = {}
    for item in components:
        counts[item.ecosystem] = counts.get(item.ecosystem, 0) + 1
    unresolved = sum(item.license == "NOASSERTION" for item in components)
    manifest = {
        "schema_version": 1,
        "status": "PASS_SUPPLY_CHAIN_INVENTORY_GENERATED",
        "git_head": git_sha,
        "generated_at_utc": _timestamp(),
        "component_count": len(components),
        "component_count_by_ecosystem": dict(sorted(counts.items())),
        "license_identified_count": len(components) - unresolved,
        "license_unresolved_count": unresolved,
        "claim_boundary": (
            "The SBOM inventories every package pinned by the three release lockfiles. License "
            "values absent from a lockfile remain NOASSERTION and require human/legal review."
        ),
        "files": {
            sbom_path.name: _sha256(sbom_path),
            licenses_path.name: _sha256(licenses_path),
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate deterministic CycloneDX and dependency-license reports from locks"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "release" / "supply-chain",
    )
    arguments = parser.parse_args()
    print(json.dumps(generate(arguments.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
