import numpy as np

from depthwizard.preprocess.radiometry import robust_rgb_normalize


def test_robust_rgb_normalize_is_bounded_and_deterministic() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint16)
    image[..., 0] = np.arange(100).reshape(10, 10)
    image[..., 1] = image[..., 0] * 2
    image[..., 2] = 1000 - image[..., 0]
    a = robust_rgb_normalize(image)
    b = robust_rgb_normalize(image)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    assert float(a.min()) >= 0.0
    assert float(a.max()) <= 1.0
