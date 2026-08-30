from contextlib import nullcontext
from typing import Any

import numpy as np
import pytest

from depthwizard.geometry_prior.da3 import (
    DA3_CHECKPOINT_SHA256,
    DA3_HF_REVISION,
    DA3_MODEL_SOURCE,
    DA3_UPSTREAM_SOURCE_COMMIT,
    DA3MonocularPrior,
    depth_to_affine_height_evidence,
    depth_to_relative_height,
)


def test_depth_to_relative_height_reverses_depth_order_without_metric_claim() -> None:
    depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    height = depth_to_relative_height(depth, low_percentile=0, high_percentile=100)
    assert height[0, 0] > height[1, 1]
    assert np.isclose(height.min(), 0.0)
    assert np.isclose(height.max(), 1.0)


def test_affine_height_evidence_preserves_depth_scale_and_offset() -> None:
    depth = np.array([[2.0, 4.0], [6.0, 8.0]], dtype=np.float32)
    evidence = depth_to_affine_height_evidence(depth)
    assert np.allclose(evidence, -depth)
    assert evidence[0, 0] > evidence[1, 1]
    assert np.isclose(float(np.ptp(evidence)), 6.0)


class _FakePrediction:
    def __init__(self, depth: np.ndarray) -> None:
        self.depth = np.asarray([depth], dtype=np.float32)
        self.conf: None = None


class _FakeModel:
    def __init__(self, depth: np.ndarray) -> None:
        self._depth = depth

    def inference(self, _images: list[Any]) -> _FakePrediction:
        return _FakePrediction(self._depth)


class _FakeTorch:
    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()


def test_da3_adapter_defers_normalization_until_scene_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    depth = np.array([[10.0, 12.0], [30.0, 40.0]], dtype=np.float32)
    prior = DA3MonocularPrior(model_source="controlled-test-source")
    monkeypatch.setattr(prior, "_load", lambda: (_FakeModel(depth), _FakeTorch(), "cpu"))

    output = prior.infer(np.zeros((2, 2, 3), dtype=np.float32))

    # The old defect normalized every inference tile independently to approximately [0, 1].
    # Production must now preserve the raw affine ordering until the full scene is assembled.
    assert np.allclose(output.relative_height, -depth)
    assert float(np.ptp(output.relative_height)) == pytest.approx(30.0)
    assert output.metadata["scene_normalize_relative_height"] is True
    assert output.metadata["output_semantics"] == "affine_relative_surface_height_evidence"


def test_da3_production_identity_is_revision_and_checkpoint_pinned() -> None:
    assert DA3_MODEL_SOURCE == "depth-anything/DA3MONO-LARGE"
    assert DA3_HF_REVISION == "f465978e618db8cc79c83b8bbf24964857db1875"
    assert DA3_CHECKPOINT_SHA256 == (
        "7a799a7f95eb8d4c404c2ca8be3dc3276b350a417ddc4420db72ba850cc0e960"
    )
    assert DA3_UPSTREAM_SOURCE_COMMIT == "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
