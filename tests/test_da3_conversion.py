import numpy as np

from depthwizard.geometry_prior.da3 import (
    DA3_CHECKPOINT_SHA256,
    DA3_HF_REVISION,
    DA3_MODEL_SOURCE,
    DA3_UPSTREAM_SOURCE_COMMIT,
    depth_to_relative_height,
)


def test_depth_to_relative_height_reverses_depth_order_without_metric_claim() -> None:
    depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    height = depth_to_relative_height(depth, low_percentile=0, high_percentile=100)
    assert height[0, 0] > height[1, 1]
    assert np.isclose(height.min(), 0.0)
    assert np.isclose(height.max(), 1.0)


def test_da3_production_identity_is_revision_and_checkpoint_pinned() -> None:
    assert DA3_MODEL_SOURCE == "depth-anything/DA3MONO-LARGE"
    assert DA3_HF_REVISION == "f465978e618db8cc79c83b8bbf24964857db1875"
    assert DA3_CHECKPOINT_SHA256 == (
        "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
    )
    assert DA3_UPSTREAM_SOURCE_COMMIT == "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
