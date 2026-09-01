from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from depthwizard.height_model.terrain_structure_split import (
    SUPERVISION_ELIGIBLE_TILE_IDS,
    TSD_SPLIT_PROTOCOL_VERSION,
)
from qualification.plan_tsd_potsdam_acquisition import build_acquisition_plan
from qualification.stage_tsd_potsdam_acquisition import (
    DESTINATION_SUBDIRECTORIES,
    stage_selected_members,
    validate_acquisition_archives,
)


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


def _write_archive(path: Path, names: list[str]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, name in enumerate(names):
            archive.writestr(name, f"payload-{index}".encode())


def _plan_and_archives(
    tmp_path: Path,
    *,
    extra_label_names: list[str] | None = None,
) -> tuple[Path, dict[str, object], Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    plan = build_acquisition_plan(_inventory_payload(dataset_root))
    required = plan["required_files"]
    assert isinstance(required, list)

    names_by_component: dict[str, list[str]] = {"rgb": [], "dsm": [], "label": []}
    for item in required:
        assert isinstance(item, dict)
        component = item["component"]
        expected_filename = item["expected_filename"]
        assert isinstance(component, str)
        assert isinstance(expected_filename, str)
        names_by_component[component].append(f"official/{expected_filename}")

    # Participant labels for the two DepthWizard sealed blind tiles may exist in the participant
    # package. They are deliberately not selected or copied by the frozen campaign.
    names_by_component["label"].extend(
        [
            "official/top_potsdam_4_12_label.tif",
            "official/top_potsdam_6_12_label.tif",
        ]
    )
    if extra_label_names:
        names_by_component["label"].extend(extra_label_names)

    rgb_archive = tmp_path / "2_Ortho_RGB.zip"
    dsm_archive = tmp_path / "1_DSM.zip"
    label_archive = tmp_path / "5_Labels_for_participants.zip"
    _write_archive(rgb_archive, names_by_component["rgb"])
    _write_archive(dsm_archive, names_by_component["dsm"])
    _write_archive(label_archive, names_by_component["label"])
    return dataset_root, plan, rgb_archive, dsm_archive, label_archive


def test_acquisition_stage_selects_only_frozen_missing_members(tmp_path: Path) -> None:
    dataset_root, plan, rgb_archive, dsm_archive, label_archive = _plan_and_archives(tmp_path)

    selected = validate_acquisition_archives(
        payload=plan,
        rgb_archive=rgb_archive,
        dsm_archive=dsm_archive,
        label_archive=label_archive,
    )

    assert len(selected) == 37
    assert sum(item.component == "rgb" for item in selected) == 12
    assert sum(item.component == "dsm" for item in selected) == 12
    assert sum(item.component == "label" for item in selected) == 13
    assert not any(item.tile_id in {"4_12", "6_12"} for item in selected)

    staged = stage_selected_members(
        dataset_root=dataset_root,
        selected=selected,
        archive_paths={
            "rgb": rgb_archive,
            "dsm": dsm_archive,
            "label": label_archive,
        },
    )

    assert len(staged) == 37
    for item in staged:
        path = Path(item.destination)
        assert path.is_file()
        assert path.parent.name == DESTINATION_SUBDIRECTORIES[item.component]
        assert item.bytes_written > 0
        assert len(item.sha256) == 64
    assert not list(dataset_root.rglob("top_potsdam_4_12_label.tif"))
    assert not list(dataset_root.rglob("top_potsdam_6_12_label.tif"))


def test_acquisition_stage_rejects_all_labels_archive_content(tmp_path: Path) -> None:
    _, plan, rgb_archive, dsm_archive, label_archive = _plan_and_archives(
        tmp_path,
        extra_label_names=["official/top_potsdam_2_14_label.tif"],
    )

    with pytest.raises(RuntimeError, match="historical challenge-test labels"):
        validate_acquisition_archives(
            payload=plan,
            rgb_archive=rgb_archive,
            dsm_archive=dsm_archive,
            label_archive=label_archive,
        )


def test_acquisition_stage_rejects_unsafe_zip_member_path(tmp_path: Path) -> None:
    dataset_root, plan, rgb_archive, dsm_archive, label_archive = _plan_and_archives(tmp_path)
    del dataset_root
    with zipfile.ZipFile(rgb_archive, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("../unexpected.bin", b"unsafe")

    with pytest.raises(ValueError, match="unsafe ZIP member path"):
        validate_acquisition_archives(
            payload=plan,
            rgb_archive=rgb_archive,
            dsm_archive=dsm_archive,
            label_archive=label_archive,
        )


def test_acquisition_stage_refuses_stale_plan_overwrite(tmp_path: Path) -> None:
    dataset_root, plan, rgb_archive, dsm_archive, label_archive = _plan_and_archives(tmp_path)
    selected = validate_acquisition_archives(
        payload=plan,
        rgb_archive=rgb_archive,
        dsm_archive=dsm_archive,
        label_archive=label_archive,
    )
    first = selected[0]
    conflicting = (
        dataset_root / DESTINATION_SUBDIRECTORIES[first.component] / first.expected_filename
    )
    conflicting.parent.mkdir(parents=True)
    conflicting.write_bytes(b"unexpected-new-local-file")

    with pytest.raises(RuntimeError, match="stale acquisition plan"):
        stage_selected_members(
            dataset_root=dataset_root,
            selected=selected,
            archive_paths={
                "rgb": rgb_archive,
                "dsm": dsm_archive,
                "label": label_archive,
            },
        )
