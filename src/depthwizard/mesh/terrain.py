from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals


@dataclass(frozen=True)
class MeshExportResult:
    path: Path
    vertices: int
    faces: int
    stride: int
    width_samples: int
    height_samples: int


def _sample_indices(length: int, stride: int) -> np.ndarray:
    values = np.arange(0, length, stride, dtype=np.int64)
    if values[-1] != length - 1:
        values = np.append(values, length - 1)
    return values


def terrain_mesh_from_dsm(
    elevation: np.ndarray,
    rgb: np.ndarray,
    *,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    stride: int = 1,
    valid_mask: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Create a metric, y-up textured terrain mesh from a DSM/rDSM.

    x maps to image columns, z maps to negative image rows so north/up raster orientation remains
    intuitive in Three.js, and vertex y stores elevation. Faces are wound counter-clockwise when
    viewed from above so geometric front normals point upward; analytical materials must never rely
    on double-sided rendering to hide reversed terrain normals. When ``valid_mask`` is supplied,
    faces touching invalid/nodata pixels are omitted rather than rendered as black artificial terrain.
    """
    z = np.asarray(elevation, dtype=np.float32)
    image = np.asarray(rgb)
    if z.ndim != 2:
        raise ValueError("elevation must be a 2D raster")
    if image.ndim != 3 or image.shape[:2] != z.shape or image.shape[2] != 3:
        raise ValueError("rgb must have shape HxWx3 matching elevation")
    if gsd_x <= 0 or gsd_y <= 0:
        raise ValueError("gsd_x and gsd_y must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if not np.all(np.isfinite(z)):
        raise ValueError("mesh export currently requires finite elevation values")

    validity = np.ones(z.shape, dtype=bool)
    if valid_mask is not None:
        validity = np.asarray(valid_mask, dtype=bool)
        if validity.shape != z.shape:
            raise ValueError("valid_mask must match elevation shape")
        if not np.any(validity):
            raise ValueError("valid_mask contains no valid terrain pixels")

    rows = _sample_indices(z.shape[0], stride)
    cols = _sample_indices(z.shape[1], stride)
    sampled = z[np.ix_(rows, cols)]
    sampled_valid = validity[np.ix_(rows, cols)]
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    vertices = np.column_stack(
        [
            cc.reshape(-1) * gsd_x,
            sampled.reshape(-1),
            -rr.reshape(-1) * gsd_y,
        ]
    ).astype(np.float32)

    h, w = sampled.shape
    faces: list[tuple[int, int, int]] = []
    for row in range(h - 1):
        for col in range(w - 1):
            a = row * w + col
            b = a + 1
            c = (row + 1) * w + col
            d = c + 1
            triangle_1_valid = (
                sampled_valid[row, col]
                and sampled_valid[row + 1, col]
                and sampled_valid[row, col + 1]
            )
            triangle_2_valid = (
                sampled_valid[row, col + 1]
                and sampled_valid[row + 1, col]
                and sampled_valid[row + 1, col + 1]
            )
            # z decreases with raster row index, therefore (a, b, c) and (b, d, c) are the
            # counter-clockwise y-up windings when the cell is viewed from above.
            if triangle_1_valid:
                faces.append((a, b, c))
            if triangle_2_valid:
                faces.append((b, d, c))

    if not faces:
        raise ValueError("valid_mask removed every terrain face")

    u = cc.reshape(-1).astype(np.float32) / max(z.shape[1] - 1, 1)
    v = 1.0 - rr.reshape(-1).astype(np.float32) / max(z.shape[0] - 1, 1)
    uv = np.column_stack([u, v])

    if image.dtype != np.uint8:
        finite = np.isfinite(image)
        if not np.all(finite):
            image = np.where(finite, image, 0)
        if image.max() <= 1.0:
            image = np.clip(image * 255.0, 0, 255)
        else:
            low = np.percentile(image, 1, axis=(0, 1), keepdims=True)
            high = np.percentile(image, 99, axis=(0, 1), keepdims=True)
            image = np.clip(
                (image - low) / np.maximum(high - low, 1e-6) * 255.0,
                0,
                255,
            )
        image = image.astype(np.uint8)

    material = PBRMaterial(
        baseColorTexture=Image.fromarray(image, mode="RGB"),
        metallicFactor=0.0,
        roughnessFactor=1.0,
        doubleSided=True,
    )
    visuals = TextureVisuals(uv=uv, material=material)
    return trimesh.Trimesh(
        vertices=vertices,
        faces=np.asarray(faces, dtype=np.int64),
        visual=visuals,
        process=False,
        validate=False,
    )


def export_terrain_glb(
    path: str | Path,
    elevation: np.ndarray,
    rgb: np.ndarray,
    *,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    stride: int = 1,
    valid_mask: np.ndarray | None = None,
) -> MeshExportResult:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = terrain_mesh_from_dsm(
        elevation,
        rgb,
        gsd_x=gsd_x,
        gsd_y=gsd_y,
        stride=stride,
        valid_mask=valid_mask,
    )
    scene = trimesh.Scene(mesh)
    payload = scene.export(file_type="glb")
    if not isinstance(payload, (bytes, bytearray)):
        raise TypeError("trimesh GLB export returned a non-binary payload")
    output.write_bytes(bytes(payload))
    rows = _sample_indices(elevation.shape[0], stride)
    cols = _sample_indices(elevation.shape[1], stride)
    return MeshExportResult(
        path=output,
        vertices=len(mesh.vertices),
        faces=len(mesh.faces),
        stride=stride,
        width_samples=len(cols),
        height_samples=len(rows),
    )


def export_lod_pyramid(
    output_dir: str | Path,
    elevation: np.ndarray,
    rgb: np.ndarray,
    *,
    gsd_x: float = 1.0,
    gsd_y: float = 1.0,
    strides: tuple[int, ...] = (1, 2, 4, 8),
    valid_mask: np.ndarray | None = None,
) -> list[MeshExportResult]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    results = []
    for level, stride in enumerate(strides):
        results.append(
            export_terrain_glb(
                directory / f"terrain-lod{level}.glb",
                elevation,
                rgb,
                gsd_x=gsd_x,
                gsd_y=gsd_y,
                stride=stride,
                valid_mask=valid_mask,
            )
        )
    return results
