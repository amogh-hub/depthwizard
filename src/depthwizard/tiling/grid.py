from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileWindow:
    y: int
    x: int
    height: int
    width: int



def generate_tiles(
    height: int,
    width: int,
    *,
    tile_size: int = 1024,
    overlap: int = 128,
) -> list[TileWindow]:
    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    stride = tile_size - overlap

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        values = list(range(0, max(length - tile_size + 1, 1), stride))
        final = length - tile_size
        if values[-1] != final:
            values.append(final)
        return values

    result: list[TileWindow] = []
    for y in starts(height):
        for x in starts(width):
            result.append(
                TileWindow(
                    y=y,
                    x=x,
                    height=min(tile_size, height - y),
                    width=min(tile_size, width - x),
                )
            )
    return result
