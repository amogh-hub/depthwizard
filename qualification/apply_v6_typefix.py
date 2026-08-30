from __future__ import annotations

from pathlib import Path

TARGET = Path("scripts/train_urban_structure_v6.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        """import json\nimport os\nimport time\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\n\nimport numpy as np\nimport rasterio\nimport torch\nfrom rasterio.enums import Resampling\n""",
        """import json\nimport os\nimport time\nfrom collections.abc import Mapping\nfrom dataclasses import asdict, dataclass\nfrom pathlib import Path\n\nimport numpy as np\nimport rasterio\nimport torch\nfrom affine import Affine\nfrom rasterio.enums import Resampling\n""",
        "imports",
    )

    text = replace_once(
        text,
        "def _target_grid(rgb_path: Path) -> tuple[int, int, object]:",
        "def _target_grid(rgb_path: Path) -> tuple[int, int, Affine]:",
        "target-grid return type",
    )
    text = replace_once(
        text,
        "    transform: object,\n) -> tuple[np.ndarray, np.ndarray]:",
        "    transform: Affine,\n) -> tuple[np.ndarray, np.ndarray]:",
        "reference transform type",
    )

    text = replace_once(
        text,
        """def _metric(report: dict[str, object], section: str, field: str) -> float:\n    mapping = report.get(section)\n    if not isinstance(mapping, dict):\n        raise TypeError(f\"missing metric section {section}\")\n    value = mapping.get(field)\n    if not isinstance(value, (int, float)):\n        raise TypeError(f\"metric {section}.{field} is not numeric\")\n    return float(value)\n\n\ndef _natural_validation_summary(\n""",
        """def _metric(report: dict[str, object], section: str, field: str) -> float:\n    mapping = report.get(section)\n    if not isinstance(mapping, dict):\n        raise TypeError(f\"missing metric section {section}\")\n    value = mapping.get(field)\n    if not isinstance(value, (int, float)):\n        raise TypeError(f\"metric {section}.{field} is not numeric\")\n    return float(value)\n\n\ndef _required_number(mapping: Mapping[str, object], key: str) -> float:\n    value = mapping.get(key)\n    if not isinstance(value, (int, float)):\n        raise TypeError(f\"metric {key} is not numeric\")\n    return float(value)\n\n\ndef _required_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:\n    value = mapping.get(key)\n    if not isinstance(value, dict):\n        raise TypeError(f\"report section {key} is missing or invalid\")\n    return value\n\n\ndef _natural_validation_summary(\n""",
        "metric helpers",
    )

    text = replace_once(
        text,
        """    structure4 = urban.get(\"structure_4m\")\n    structure8 = urban.get(\"structure_8m\")\n    base_slope = urban.get(\"base_slope\")\n    refined_slope = urban.get(\"refined_slope\")\n    if not all(\n        isinstance(value, dict)\n        for value in (structure4, structure8, base_slope, refined_slope)\n    ):\n        raise TypeError(\"urban validation report is incomplete\")\n    s4_base = float(structure4[\"base_rmse_m\"])\n    s4_refined = float(structure4[\"refined_rmse_m\"])\n    s8_base = float(structure8[\"base_rmse_m\"])\n    s8_refined = float(structure8[\"refined_rmse_m\"])\n    s4_base_r = structure4[\"base_pearson_r\"]\n    s4_refined_r = structure4[\"refined_pearson_r\"]\n    s8_base_r = structure8[\"base_pearson_r\"]\n    s8_refined_r = structure8[\"refined_pearson_r\"]\n    if not all(\n        isinstance(value, (int, float))\n        for value in (s4_base_r, s4_refined_r, s8_base_r, s8_refined_r)\n    ):\n        return False, float(\"-inf\")\n    slope_ok = float(refined_slope[\"rmse_degrees\"]) <= float(\n        base_slope[\"rmse_degrees\"]\n    ) + 1e-9\n""",
        """    structure4 = _required_mapping(urban, \"structure_4m\")\n    structure8 = _required_mapping(urban, \"structure_8m\")\n    base_slope = _required_mapping(urban, \"base_slope\")\n    refined_slope = _required_mapping(urban, \"refined_slope\")\n    s4_base = _required_number(structure4, \"base_rmse_m\")\n    s4_refined = _required_number(structure4, \"refined_rmse_m\")\n    s8_base = _required_number(structure8, \"base_rmse_m\")\n    s8_refined = _required_number(structure8, \"refined_rmse_m\")\n    s4_base_r = structure4.get(\"base_pearson_r\")\n    s4_refined_r = structure4.get(\"refined_pearson_r\")\n    s8_base_r = structure8.get(\"base_pearson_r\")\n    s8_refined_r = structure8.get(\"refined_pearson_r\")\n    if not all(\n        isinstance(value, (int, float))\n        for value in (s4_base_r, s4_refined_r, s8_base_r, s8_refined_r)\n    ):\n        return False, float(\"-inf\")\n    slope_ok = _required_number(refined_slope, \"rmse_degrees\") <= (\n        _required_number(base_slope, \"rmse_degrees\") + 1e-9\n    )\n""",
        "urban eligibility narrowing",
    )

    text = replace_once(
        text,
        """    pass_transfer = bool(\n        refined_metrics.rmse_m < base_metrics.rmse_m\n        and refined_metrics.mae_m <= base_metrics.mae_m\n        and refined_metrics.pearson_r is not None\n        and base_metrics.pearson_r is not None\n        and refined_metrics.pearson_r >= base_metrics.pearson_r\n        and refined_slope.rmse_degrees <= base_slope.rmse_degrees\n        and float(structure4[\"refined_rmse_m\"]) < float(structure4[\"base_rmse_m\"])\n        and float(structure8[\"refined_rmse_m\"]) < float(structure8[\"base_rmse_m\"])\n        and isinstance(structure4[\"refined_pearson_r\"], (int, float))\n        and isinstance(structure4[\"base_pearson_r\"], (int, float))\n        and float(structure4[\"refined_pearson_r\"]) >= float(structure4[\"base_pearson_r\"])\n        and isinstance(structure8[\"refined_pearson_r\"], (int, float))\n        and isinstance(structure8[\"base_pearson_r\"], (int, float))\n        and float(structure8[\"refined_pearson_r\"]) >= float(structure8[\"base_pearson_r\"])\n    )\n""",
        """    s4_refined_r = structure4.get(\"refined_pearson_r\")\n    s4_base_r = structure4.get(\"base_pearson_r\")\n    s8_refined_r = structure8.get(\"refined_pearson_r\")\n    s8_base_r = structure8.get(\"base_pearson_r\")\n    structure_correlation_ok = False\n    if (\n        isinstance(s4_refined_r, (int, float))\n        and isinstance(s4_base_r, (int, float))\n        and isinstance(s8_refined_r, (int, float))\n        and isinstance(s8_base_r, (int, float))\n    ):\n        structure_correlation_ok = (\n            float(s4_refined_r) >= float(s4_base_r)\n            and float(s8_refined_r) >= float(s8_base_r)\n        )\n    pass_transfer = bool(\n        refined_metrics.rmse_m < base_metrics.rmse_m\n        and refined_metrics.mae_m <= base_metrics.mae_m\n        and refined_metrics.pearson_r is not None\n        and base_metrics.pearson_r is not None\n        and refined_metrics.pearson_r >= base_metrics.pearson_r\n        and refined_slope.rmse_degrees <= base_slope.rmse_degrees\n        and _required_number(structure4, \"refined_rmse_m\")\n        < _required_number(structure4, \"base_rmse_m\")\n        and _required_number(structure8, \"refined_rmse_m\")\n        < _required_number(structure8, \"base_rmse_m\")\n        and structure_correlation_ok\n    )\n""",
        "development transfer narrowing",
    )

    text = replace_once(
        text,
        """    if exposed_2_14 is not None:\n        base = exposed_2_14[\"base\"]\n        refined = exposed_2_14[\"refined\"]\n        s4 = exposed_2_14[\"structure_4m\"]\n        s8 = exposed_2_14[\"structure_8m\"]\n        print(\n            f\"Potsdam 2_14 RMSE: {base['rmse_m']:.4f} -> \"\n            f\"{refined['rmse_m']:.4f} m\"\n        )\n        print(\n            f\"Potsdam 2_14 structure 4m: {s4['base_rmse_m']:.4f} -> \"\n            f\"{s4['refined_rmse_m']:.4f} m\"\n        )\n        print(\n            f\"Potsdam 2_14 structure 8m: {s8['base_rmse_m']:.4f} -> \"\n            f\"{s8['refined_rmse_m']:.4f} m\"\n        )\n        print(\n            \"Potsdam 2_14 production-transfer gate: \"\n            f\"{'PASS' if exposed_2_14['pass'] else 'REJECT'}\"\n        )\n""",
        """    if exposed_2_14 is not None:\n        base = _required_mapping(exposed_2_14, \"base\")\n        refined = _required_mapping(exposed_2_14, \"refined\")\n        s4 = _required_mapping(exposed_2_14, \"structure_4m\")\n        s8 = _required_mapping(exposed_2_14, \"structure_8m\")\n        transfer_pass = exposed_2_14.get(\"pass\")\n        if not isinstance(transfer_pass, bool):\n            raise TypeError(\"Potsdam 2_14 transfer gate is not boolean\")\n        print(\n            f\"Potsdam 2_14 RMSE: {_required_number(base, 'rmse_m'):.4f} -> \"\n            f\"{_required_number(refined, 'rmse_m'):.4f} m\"\n        )\n        print(\n            f\"Potsdam 2_14 structure 4m: {_required_number(s4, 'base_rmse_m'):.4f} -> \"\n            f\"{_required_number(s4, 'refined_rmse_m'):.4f} m\"\n        )\n        print(\n            f\"Potsdam 2_14 structure 8m: {_required_number(s8, 'base_rmse_m'):.4f} -> \"\n            f\"{_required_number(s8, 'refined_rmse_m'):.4f} m\"\n        )\n        print(\n            \"Potsdam 2_14 production-transfer gate: \"\n            f\"{'PASS' if transfer_pass else 'REJECT'}\"\n        )\n""",
        "final report narrowing",
    )

    TARGET.write_text(text, encoding="utf-8")
    print("PASS_V6_TYPEFIX_SOURCE_TRANSFORM")


if __name__ == "__main__":
    main()
