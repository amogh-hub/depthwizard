from __future__ import annotations

from pathlib import Path

TARGET = Path("scripts/train_urban_structure_v6.py")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly one source match, found {count}")
    return text.replace(old, new, 1)


def replace_between(
    text: str,
    start_marker: str,
    end_marker: str,
    replacement: str,
    label: str,
    *,
    search_from: int = 0,
) -> str:
    start = text.find(start_marker, search_from)
    if start < 0:
        raise RuntimeError(f"{label}: start marker not found")
    end = text.find(end_marker, start + len(start_marker))
    if end < 0:
        raise RuntimeError(f"{label}: end marker not found")
    return text[:start] + replacement + text[end:]


def main() -> None:
    text = TARGET.read_text(encoding="utf-8")

    text = replace_once(
        text,
        "import time\nfrom dataclasses import asdict, dataclass\n",
        "import time\nfrom collections.abc import Mapping\nfrom dataclasses import asdict, dataclass\n",
        "Mapping import",
    )
    text = replace_once(
        text,
        "from pathlib import Path\n\nimport numpy as np\nimport rasterio\nimport torch\n",
        "from pathlib import Path\n\nimport numpy as np\nimport rasterio\nimport torch\nfrom affine import Affine\n",
        "Affine import",
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

    helper_marker = "\n\ndef _natural_validation_summary(reports: list[dict[str, object]]) -> dict[str, object]:"
    helper_insert = """


def _required_number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"metric {key} is not numeric")
    return float(value)


def _required_mapping(mapping: Mapping[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise TypeError(f"report section {key} is missing or invalid")
    return value
"""
    if helper_marker not in text:
        raise RuntimeError("metric helper insertion marker not found")
    text = text.replace(helper_marker, helper_insert + helper_marker, 1)

    urban_function = '''def _urban_epoch_eligible(
    urban: dict[str, object],
    natural: dict[str, object],
) -> tuple[bool, float]:
    refined = urban.get("refined")
    if not isinstance(refined, dict):
        return False, float("-inf")
    base_rmse = _metric(urban, "base", "rmse_m")
    refined_rmse = _metric(urban, "refined", "rmse_m")
    structure4 = _required_mapping(urban, "structure_4m")
    structure8 = _required_mapping(urban, "structure_8m")
    base_slope = _required_mapping(urban, "base_slope")
    refined_slope = _required_mapping(urban, "refined_slope")

    s4_base = _required_number(structure4, "base_rmse_m")
    s4_refined = _required_number(structure4, "refined_rmse_m")
    s8_base = _required_number(structure8, "base_rmse_m")
    s8_refined = _required_number(structure8, "refined_rmse_m")

    s4_base_r = structure4.get("base_pearson_r")
    s4_refined_r = structure4.get("refined_pearson_r")
    s8_base_r = structure8.get("base_pearson_r")
    s8_refined_r = structure8.get("refined_pearson_r")
    if not isinstance(s4_base_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s4_refined_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s8_base_r, (int, float)):
        return False, float("-inf")
    if not isinstance(s8_refined_r, (int, float)):
        return False, float("-inf")

    natural_non_degrading = natural.get("non_degrading")
    if not isinstance(natural_non_degrading, bool):
        raise TypeError("natural validation non_degrading flag is not boolean")

    slope_ok = _required_number(refined_slope, "rmse_degrees") <= (
        _required_number(base_slope, "rmse_degrees") + 1e-9
    )
    eligible = bool(
        natural_non_degrading
        and refined_rmse < base_rmse - 1e-3
        and s4_refined < s4_base - 1e-3
        and s8_refined < s8_base - 1e-3
        and float(s4_refined_r) >= float(s4_base_r)
        and float(s8_refined_r) >= float(s8_base_r)
        and slope_ok
    )
    if not eligible:
        return False, float("-inf")
    score = (
        (base_rmse - refined_rmse) / base_rmse
        + (s4_base - s4_refined) / s4_base
        + 1.5 * (s8_base - s8_refined) / s8_base
    )
    return True, float(score)
'''
    text = replace_between(
        text,
        "def _urban_epoch_eligible(",
        "\n\ndef _warm_start_v5",
        urban_function,
        "urban eligibility function",
    )

    exposed_start = text.find("def _evaluate_exposed_2_14(")
    if exposed_start < 0:
        raise RuntimeError("exposed 2_14 evaluator not found")
    pass_block = '''    s4_refined_r = structure4.get("refined_pearson_r")
    s4_base_r = structure4.get("base_pearson_r")
    s8_refined_r = structure8.get("refined_pearson_r")
    s8_base_r = structure8.get("base_pearson_r")
    structure_correlation_ok = False
    if (
        isinstance(s4_refined_r, (int, float))
        and isinstance(s4_base_r, (int, float))
        and isinstance(s8_refined_r, (int, float))
        and isinstance(s8_base_r, (int, float))
    ):
        structure_correlation_ok = (
            float(s4_refined_r) >= float(s4_base_r)
            and float(s8_refined_r) >= float(s8_base_r)
        )
    pass_transfer = bool(
        refined_metrics.rmse_m < base_metrics.rmse_m
        and refined_metrics.mae_m <= base_metrics.mae_m
        and refined_metrics.pearson_r is not None
        and base_metrics.pearson_r is not None
        and refined_metrics.pearson_r >= base_metrics.pearson_r
        and refined_slope.rmse_degrees <= base_slope.rmse_degrees
        and _required_number(structure4, "refined_rmse_m")
        < _required_number(structure4, "base_rmse_m")
        and _required_number(structure8, "refined_rmse_m")
        < _required_number(structure8, "base_rmse_m")
        and structure_correlation_ok
    )
'''
    text = replace_between(
        text,
        "    pass_transfer = bool(",
        "    return {",
        pass_block,
        "exposed transfer gate",
        search_from=exposed_start,
    )

    main_start = text.find("def main() -> None:")
    if main_start < 0:
        raise RuntimeError("main function not found")
    report_block = '''    if exposed_2_14 is not None:
        base = _required_mapping(exposed_2_14, "base")
        refined = _required_mapping(exposed_2_14, "refined")
        s4 = _required_mapping(exposed_2_14, "structure_4m")
        s8 = _required_mapping(exposed_2_14, "structure_8m")
        transfer_pass = exposed_2_14.get("pass")
        if not isinstance(transfer_pass, bool):
            raise TypeError("Potsdam 2_14 transfer gate is not boolean")
        print(
            f"Potsdam 2_14 RMSE: {_required_number(base, 'rmse_m'):.4f} -> "
            f"{_required_number(refined, 'rmse_m'):.4f} m"
        )
        print(
            f"Potsdam 2_14 structure 4m: {_required_number(s4, 'base_rmse_m'):.4f} -> "
            f"{_required_number(s4, 'refined_rmse_m'):.4f} m"
        )
        print(
            f"Potsdam 2_14 structure 8m: {_required_number(s8, 'base_rmse_m'):.4f} -> "
            f"{_required_number(s8, 'refined_rmse_m'):.4f} m"
        )
        print(
            "Potsdam 2_14 production-transfer gate: "
            f"{'PASS' if transfer_pass else 'REJECT'}"
        )
    else:
        print("Potsdam 2_14 was not opened because no validation-eligible learned epoch existed.")
'''
    text = replace_between(
        text,
        "    if exposed_2_14 is not None:",
        "    print(\n        \"V6 candidate: \"",
        report_block,
        "final report block",
        search_from=main_start,
    )

    TARGET.write_text(text, encoding="utf-8")
    print("PASS_V6_TYPEFIX_SOURCE_TRANSFORM")


if __name__ == "__main__":
    main()
