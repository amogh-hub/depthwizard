from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileWindow:
    y: int
    x: int
    height: int
    width: int


def _validate_grid(height: int, width: int, tile_size: int, overlap: int) -> int:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    return tile_size - overlap


def _base_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    values = list(range(0, max(length - tile_size + 1, 1), stride))
    final = length - tile_size
    if values[-1] != final:
        values.append(final)
    return values


def _shifted_starts(
    length: int,
    tile_size: int,
    stride: int,
    *,
    offset: int,
    minimum_edge_separation: int,
) -> list[int]:
    """Return a complete grid whose interior phase is shifted from the base lattice.

    Scene edges remain anchored at 0 and ``length - tile_size`` so every pixel is covered. Interior
    starts follow ``offset + n * stride``. A candidate extremely close to the final anchored tile is
    omitted because two almost coincident edge inferences add cost without contributing a genuinely
    different context phase.
    """
    if length <= tile_size:
        return [0]
    if not (0 < offset < stride):
        raise ValueError("shifted-grid offset must satisfy 0 < offset < stride")
    if minimum_edge_separation < 0:
        raise ValueError("minimum_edge_separation must be non-negative")

    final = length - tile_size
    values = [0]
    current = offset
    while current < final:
        if final - current >= minimum_edge_separation:
            values.append(current)
        current += stride
    if values[-1] != final:
        values.append(final)
    return sorted(set(values))


def _windows_from_starts(
    height: int,
    width: int,
    tile_size: int,
    ys: list[int],
    xs: list[int],
) -> list[TileWindow]:
    return [
        TileWindow(
            y=y,
            x=x,
            height=min(tile_size, height - y),
            width=min(tile_size, width - x),
        )
        for y in ys
        for x in xs
    ]


def generate_tiles(
    height: int,
    width: int,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
) -> list[TileWindow]:
    stride = _validate_grid(height, width, tile_size, overlap)
    ys = _base_starts(height, tile_size, stride)
    xs = _base_starts(width, tile_size, stride)
    return _windows_from_starts(height, width, tile_size, ys, xs)


def generate_shifted_tiles(
    height: int,
    width: int,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
    offset_y: int | None = None,
    offset_x: int | None = None,
) -> list[TileWindow]:
    """Generate a second full-coverage lattice shifted by half a production stride.

    A half-stride shift moves both vertical and horizontal tile boundaries away from the primary
    lattice. Averaging complete mosaics from the two phases therefore suppresses artifacts that are
    spatially locked to one inference grid rather than to the underlying scene.
    """
    stride = _validate_grid(height, width, tile_size, overlap)
    if height <= tile_size and width <= tile_size:
        return generate_tiles(height, width, tile_size=tile_size, overlap=overlap)

    default_offset = max(1, stride // 2)
    oy = default_offset if offset_y is None else offset_y
    ox = default_offset if offset_x is None else offset_x
    edge_separation = max(1, overlap)

    ys = (
        [0]
        if height <= tile_size
        else _shifted_starts(
            height,
            tile_size,
            stride,
            offset=oy,
            minimum_edge_separation=edge_separation,
        )
    )
    xs = (
        [0]
        if width <= tile_size
        else _shifted_starts(
            width,
            tile_size,
            stride,
            offset=ox,
            minimum_edge_separation=edge_separation,
        )
    )
    return _windows_from_starts(height, width, tile_size, ys, xs)
