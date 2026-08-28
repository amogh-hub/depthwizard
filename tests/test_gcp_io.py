import hashlib
from pathlib import Path

import pytest

from depthwizard.calibration import gcp_io
from depthwizard.calibration.gcp_io import inspect_ground_control_point_file


def test_gcp_csv_import_preserves_metric_evidence(tmp_path: Path) -> None:
    path = tmp_path / "gcps.csv"
    path.write_text(
        "x,y,elevation_m,weight\n"
        "500000,1400000,101.25,2\n"
        "500010,1400010,108.75,1\n",
        encoding="utf-8",
    )
    expected_bytes = path.read_bytes()

    report = inspect_ground_control_point_file(path)

    assert report.point_count == 2
    assert report.minimum_elevation_m == 101.25
    assert report.maximum_elevation_m == 108.75
    assert report.points[0].weight == 2.0
    assert report.sha256 == hashlib.sha256(expected_bytes).hexdigest()
    assert "does not transform or infer coordinates" in report.semantics


def test_gcp_csv_import_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "gcps.csv"
    path.write_text(
        "x,y,z\n"
        "500000,1400000,101\n"
        "500010,1400010,nan\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="non-finite GCP"):
        inspect_ground_control_point_file(path)


def test_gcp_csv_import_rejects_excessive_point_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gcp_io, "_MAX_GCP_POINTS", 2)
    path = tmp_path / "gcps.csv"
    path.write_text(
        "x,y,elevation_m\n"
        "1,1,10\n"
        "2,2,20\n"
        "3,3,30\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="point ingestion limit"):
        inspect_ground_control_point_file(path)


def test_gcp_csv_import_rejects_excessive_file_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gcp_io, "_MAX_GCP_FILE_BYTES", 24)
    path = tmp_path / "gcps.csv"
    path.write_text(
        "x,y,elevation_m\n"
        "500000,1400000,101.25\n"
        "500010,1400010,108.75\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ingestion limit"):
        inspect_ground_control_point_file(path)
