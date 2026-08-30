import pytest

from depthwizard.pipeline import policy
from depthwizard.provenance.manifest import canonical_json_hash


def test_geometry_pipeline_revision_is_part_of_persisted_policy_identity() -> None:
    decision = policy.current_production_estimator_decision()
    serialized = decision.as_dict()
    assert serialized["geometry_pipeline_revision"] == "scene-global-affine-mosaic-v2"


def test_geometry_pipeline_revision_change_invalidates_geometry_policy_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = policy.current_production_estimator_decision()
    before = canonical_json_hash(decision.as_dict())

    monkeypatch.setattr(
        policy,
        "PRODUCTION_GEOMETRY_PIPELINE_REVISION",
        "controlled-next-geometry-revision",
    )
    after = canonical_json_hash(decision.as_dict())

    assert after != before
