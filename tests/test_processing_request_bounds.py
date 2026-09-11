from pathlib import Path

import pytest
from pydantic import ValidationError

from depthwizard.contracts import GroundControlPoint, ProcessingRequest


def test_processing_request_rejects_unbounded_tile_size() -> None:
    with pytest.raises(ValidationError):
        ProcessingRequest(
            source=Path("scene.tif"),
            output_dir=Path("project"),
            tile_size=4097,
        )


def test_processing_request_rejects_unbounded_smoothing_kernel() -> None:
    with pytest.raises(ValidationError):
        ProcessingRequest(
            source=Path("scene.tif"),
            output_dir=Path("project"),
            low_frequency_sigma_px=4097.0,
        )


def test_processing_request_rejects_excessive_gcp_payload() -> None:
    point = GroundControlPoint(x=1.0, y=2.0, elevation_m=3.0)
    with pytest.raises(ValidationError):
        ProcessingRequest(
            source=Path("scene.tif"),
            output_dir=Path("project"),
            gcps=[point] * 10_001,
        )


def test_processing_request_accepts_supported_operational_bounds() -> None:
    request = ProcessingRequest(
        source=Path("scene.tif"),
        output_dir=Path("project"),
        tile_size=4096,
        overlap=1024,
        low_frequency_sigma_px=4096.0,
    )
    assert request.tile_size == 4096
    assert request.overlap == 1024


def test_gcp_default_requires_six_but_explicit_expert_override_allows_four() -> None:
    points = [
        GroundControlPoint(x=float(index), y=float(index % 2), elevation_m=100.0 + index)
        for index in range(4)
    ]

    with pytest.raises(ValidationError, match="at least 6 points"):
        ProcessingRequest(
            source=Path("scene.tif"),
            output_dir=Path("project"),
            gcps=points,
        )

    request = ProcessingRequest(
        source=Path("scene.tif"),
        output_dir=Path("project"),
        gcps=points,
        min_gcp_count=4,
    )
    assert request.min_gcp_count == 4
