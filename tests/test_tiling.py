import numpy as np

from depthwizard.tiling.blend import WeightedTileAccumulator


def test_tile_accumulator_blends_overlap() -> None:
    acc = WeightedTileAccumulator(4, 6)
    acc.add(np.ones((4, 4)), 0, 0)
    acc.add(np.full((4, 4), 3.0), 0, 2)
    out = acc.finalize()
    assert np.isfinite(out).all()
    assert out[:, 0].mean() == 1.0
    assert out[:, -1].mean() == 3.0
    assert np.all((out[:, 2:4] >= 1.0) & (out[:, 2:4] <= 3.0))
