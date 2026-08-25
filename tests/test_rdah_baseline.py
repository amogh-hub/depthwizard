import numpy as np
import pytest
import torch

from depthwizard.baselines.rdah import preprocess_rdah_inputs


def test_rdah_preprocessing_keeps_raw_depth_prior() -> None:
    rgb = np.full((128, 128, 3), 128, dtype=np.uint8)
    depth = np.linspace(2.0, 80.0, 128 * 128, dtype=np.float32).reshape(128, 128)

    depth_tensor, image_tensor = preprocess_rdah_inputs(rgb, depth, device="cpu")

    assert depth_tensor.shape == (1, 1, 128, 128)
    assert image_tensor.shape == (1, 3, 128, 128)
    assert torch.allclose(depth_tensor[0, 0], torch.from_numpy(depth))
    assert torch.isfinite(image_tensor).all()


def test_rdah_preprocessing_rejects_incompatible_grid() -> None:
    rgb = np.zeros((130, 128, 3), dtype=np.uint8)
    depth = np.zeros((130, 128), dtype=np.float32)

    with pytest.raises(ValueError, match="divisible by 128"):
        preprocess_rdah_inputs(rgb, depth, device="cpu")
