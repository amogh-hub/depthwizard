from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.generate_supply_chain_reports import collect_components, generate


def test_lockfiles_generate_a_deterministic_supply_chain_inventory(tmp_path: Path) -> None:
    components = collect_components()
    ecosystems = {component.ecosystem for component in components}

    assert ecosystems == {"cargo", "npm", "pypi"}
    assert len(components) > 100
    assert len({component.bom_ref for component in components}) == len(components)
    assert any(component.bom_ref.startswith("pkg:npm/%40") for component in components)

    first = generate(tmp_path)
    first_sbom = (tmp_path / "depthwizard-sbom.cdx.json").read_bytes()
    first_licenses = (tmp_path / "dependency-license-report.csv").read_bytes()
    second = generate(tmp_path)

    assert first == second
    assert (tmp_path / "depthwizard-sbom.cdx.json").read_bytes() == first_sbom
    assert (tmp_path / "dependency-license-report.csv").read_bytes() == first_licenses
    assert first["status"] == "PASS_SUPPLY_CHAIN_INVENTORY_GENERATED"

    sbom = json.loads(first_sbom)
    assert sbom["bomFormat"] == "CycloneDX"
    assert sbom["metadata"]["properties"][0]["value"] == first["git_head"]
    with (tmp_path / "dependency-license-report.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == first["component_count"]
    assert all(row["license"] for row in rows)
