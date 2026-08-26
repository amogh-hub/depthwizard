from __future__ import annotations

from depthwizard.data.ortholoc import OrthoLoCRemoteScene
from scripts.evaluate_ortholoc_frozen_location_v2 import (
    EXCLUDED_LOCATIONS,
    SAMPLES_PER_LOCATION,
    aggregate_location_holdout,
    select_frozen_location_scenes,
)


def _scene(location: str, index: int, *, same_domain: bool = True) -> OrthoLoCRemoteScene:
    token = "R" if same_domain else "xDOP"
    filename = f"{location}_{token}{index:04d}.tif"
    root = "https://example.test/test_outPlace"
    return OrthoLoCRemoteScene(
        scene_id=filename.removesuffix(".tif"),
        split="test_outPlace",
        location_id=location,
        dop_url=f"{root}/DOPs/{filename}",
        dsm_url=f"{root}/DSMs/{filename}",
    )


def test_v2_selects_first_eligible_untouched_location_without_target_data() -> None:
    assert "L08" in EXCLUDED_LOCATIONS and "L50" in EXCLUDED_LOCATIONS
    discovered = [
        _scene("L08", 0),
        _scene("L50", 0),
        _scene("L51", 2),
        _scene("L51", 0),
        _scene("L51", 1),
        _scene("L52", 0),
        _scene("L52", 1),
        _scene("L49", 0, same_domain=False),
        _scene("L49", 1, same_domain=False),
    ]

    selected = select_frozen_location_scenes(discovered)

    assert len(selected) == SAMPLES_PER_LOCATION
    assert [scene.scene_id for scene in selected] == ["L51_R0000", "L51_R0001"]
    assert {scene.location_id for scene in selected} == {"L51"}


def test_v2_rejects_when_no_untouched_location_has_two_same_domain_scenes() -> None:
    discovered = [
        _scene("L51", 0),
        _scene("L52", 0),
        _scene("L53", 0, same_domain=False),
        _scene("L53", 1, same_domain=False),
    ]

    try:
        select_frozen_location_scenes(discovered)
    except RuntimeError as exc:
        assert "before loading any DSM targets" in str(exc)
    else:
        raise AssertionError("selector must fail before target loading when metadata is insufficient")


def _report(delta: float, weight: float) -> dict[str, object]:
    return {
        "adaptive_rmse_delta_m": delta,
        "selected_refinement_weight": weight,
    }


def test_v2_aggregate_requires_improvement_non_degradation_and_learned_use() -> None:
    import numpy as np

    reference = [np.asarray([10.0, 20.0]), np.asarray([30.0, 40.0])]
    baseline = [np.asarray([11.0, 21.0]), np.asarray([31.0, 41.0])]
    adaptive = [np.asarray([10.5, 20.5]), np.asarray([30.5, 40.5])]
    result = aggregate_location_holdout(
        [_report(-0.5, 1.0), _report(-0.5, 0.5)],
        baseline,
        adaptive,
        reference,
        [],
    )

    assert result["frozen_location_pass"] is True
    assert result["all_scenes_non_degrading"] is True
    assert result["learned_refinement_used_on_any_scene"] is True


def test_v2_aggregate_does_not_pass_exact_da3_fallback_only() -> None:
    import numpy as np

    reference = [np.asarray([10.0, 20.0]), np.asarray([30.0, 40.0])]
    baseline = [np.asarray([11.0, 21.0]), np.asarray([31.0, 41.0])]
    result = aggregate_location_holdout(
        [_report(0.0, 0.0), _report(0.0, 0.0)],
        baseline,
        baseline,
        reference,
        [],
    )

    assert result["frozen_location_pass"] is False
    assert result["all_scenes_non_degrading"] is True
    assert result["learned_refinement_used_on_any_scene"] is False
