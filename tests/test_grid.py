from depthwizard.tiling.grid import generate_tiles


def test_generate_tiles_covers_edges_without_out_of_bounds() -> None:
    tiles = generate_tiles(2100, 1900, tile_size=1024, overlap=128)
    assert tiles[0].x == 0 and tiles[0].y == 0
    assert max(tile.x + tile.width for tile in tiles) == 1900
    assert max(tile.y + tile.height for tile in tiles) == 2100
    assert all(tile.width <= 1024 and tile.height <= 1024 for tile in tiles)
