import numpy as np
import pytest

from depthwizard.evaluation.potsdam_semantics import (
    ISPRS_BUILDING_RGB,
    ISPRS_CAR_RGB,
    ISPRS_CLUTTER_RGB,
    ISPRS_IMPERVIOUS_RGB,
    ISPRS_LOW_VEGETATION_RGB,
    ISPRS_TREE_RGB,
    decode_potsdam_semantic_labels,
)


def test_decode_potsdam_semantic_palette_and_ground_policies() -> None:
    labels = np.array(
        [
            [ISPRS_IMPERVIOUS_RGB, ISPRS_BUILDING_RGB, ISPRS_LOW_VEGETATION_RGB],
            [ISPRS_TREE_RGB, ISPRS_CAR_RGB, ISPRS_CLUTTER_RGB],
            [(0, 0, 0), ISPRS_BUILDING_RGB, ISPRS_IMPERVIOUS_RGB],
        ],
        dtype=np.uint8,
    )
    decoded = decode_potsdam_semantic_labels(labels)

    assert decoded.building.tolist() == [
        [False, True, False],
        [False, False, False],
        [False, True, False],
    ]
    assert decoded.strict_ground.tolist() == [
        [True, False, False],
        [False, False, False],
        [False, False, True],
    ]
    assert decoded.expanded_ground.tolist() == [
        [True, False, True],
        [False, False, False],
        [False, False, True],
    ]
    assert decoded.unlabeled_pixels == 1
    assert decoded.class_pixel_counts == {
        "impervious": 2,
        "building": 2,
        "low_vegetation": 1,
        "tree": 1,
        "car": 1,
        "clutter": 1,
    }


def test_decode_potsdam_semantics_accepts_band_first_rgb() -> None:
    labels_hwc = np.array(
        [[ISPRS_IMPERVIOUS_RGB, ISPRS_BUILDING_RGB]],
        dtype=np.uint8,
    )
    decoded = decode_potsdam_semantic_labels(np.moveaxis(labels_hwc, -1, 0))
    assert decoded.strict_ground[0, 0]
    assert decoded.building[0, 1]


def test_decode_potsdam_semantics_rejects_unknown_colors() -> None:
    labels = np.array([[ISPRS_BUILDING_RGB, (12, 34, 56)]], dtype=np.uint8)
    with pytest.raises(ValueError, match="outside the official ISPRS palette"):
        decode_potsdam_semantic_labels(labels)


def test_decode_potsdam_semantics_can_reject_unlabeled_pixels() -> None:
    labels = np.array([[ISPRS_BUILDING_RGB, (0, 0, 0)]], dtype=np.uint8)
    with pytest.raises(ValueError, match="black/unlabeled"):
        decode_potsdam_semantic_labels(labels, allow_black_unlabeled=False)
