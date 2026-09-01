from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Official ISPRS 2D semantic-labeling palette (RGB).
ISPRS_IMPERVIOUS_RGB = (255, 255, 255)
ISPRS_BUILDING_RGB = (0, 0, 255)
ISPRS_LOW_VEGETATION_RGB = (0, 255, 255)
ISPRS_TREE_RGB = (0, 255, 0)
ISPRS_CAR_RGB = (255, 255, 0)
ISPRS_CLUTTER_RGB = (255, 0, 0)
ISPRS_UNLABELED_RGB = (0, 0, 0)

ISPRS_CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "impervious": ISPRS_IMPERVIOUS_RGB,
    "building": ISPRS_BUILDING_RGB,
    "low_vegetation": ISPRS_LOW_VEGETATION_RGB,
    "tree": ISPRS_TREE_RGB,
    "car": ISPRS_CAR_RGB,
    "clutter": ISPRS_CLUTTER_RGB,
}


@dataclass(frozen=True)
class PotsdamSemanticMasks:
    building: np.ndarray
    strict_ground: np.ndarray
    expanded_ground: np.ndarray
    known: np.ndarray
    class_pixel_counts: dict[str, int]
    unlabeled_pixels: int


def _as_rgb_hwc(labels: np.ndarray) -> np.ndarray:
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError("Potsdam semantic labels must be a three-band RGB raster")
    if values.shape[-1] == 3:
        rgb = values
    elif values.shape[0] == 3:
        rgb = np.moveaxis(values, 0, -1)
    else:
        raise ValueError("Potsdam semantic labels must contain exactly three RGB bands")
    if rgb.dtype != np.uint8:
        if np.any(~np.isfinite(rgb)) or np.any((rgb < 0) | (rgb > 255)):
            raise ValueError("Potsdam semantic RGB values must lie in [0, 255]")
        rgb = rgb.astype(np.uint8)
    return rgb


def _match(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return np.all(rgb == np.asarray(color, dtype=np.uint8), axis=-1)


def decode_potsdam_semantic_labels(
    labels: np.ndarray,
    *,
    allow_black_unlabeled: bool = True,
) -> PotsdamSemanticMasks:
    """Decode the official six-class ISPRS Potsdam semantic-label palette.

    ``strict_ground`` contains only impervious surfaces. ``expanded_ground`` adds low vegetation;
    it is intentionally emitted separately because changing the local-ground policy is a benchmark
    protocol decision and must never be tuned against a candidate DSM.
    """
    rgb = _as_rgb_hwc(labels)
    class_masks = {name: _match(rgb, color) for name, color in ISPRS_CLASS_COLORS.items()}
    unlabeled = _match(rgb, ISPRS_UNLABELED_RGB)
    known = np.zeros(rgb.shape[:2], dtype=bool)
    for mask in class_masks.values():
        known |= mask

    unknown = ~(known | unlabeled)
    if np.any(unknown):
        colors, counts = np.unique(rgb[unknown].reshape(-1, 3), axis=0, return_counts=True)
        ranked = sorted(
            zip(colors.tolist(), counts.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        preview = ", ".join(f"{tuple(color)}:{count}" for color, count in ranked[:8])
        raise ValueError(
            "semantic label raster contains RGB values outside the official ISPRS palette; "
            f"most frequent unknown colors: {preview}"
        )
    if not allow_black_unlabeled and np.any(unlabeled):
        raise ValueError("semantic label raster contains black/unlabeled pixels")

    building = class_masks["building"] & known
    strict_ground = class_masks["impervious"] & known
    expanded_ground = (class_masks["impervious"] | class_masks["low_vegetation"]) & known
    return PotsdamSemanticMasks(
        building=building,
        strict_ground=strict_ground,
        expanded_ground=expanded_ground,
        known=known,
        class_pixel_counts={name: int(np.count_nonzero(mask)) for name, mask in class_masks.items()},
        unlabeled_pixels=int(np.count_nonzero(unlabeled)),
    )
