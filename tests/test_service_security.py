from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from fastapi.testclient import TestClient
from pytest import MonkeyPatch
from rasterio.transform import from_origin

from depthwizard.service import app


def _write_rgb(path: Path) -> None:
    data = np.zeros((3, 8, 8), dtype=np.uint8)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=3,
        dtype="uint8",
        crs="EPSG:32643",
        transform=from_origin(500000, 1400000, 1.0, 1.0),
    ) as dst:
        dst.write(data)


def test_health_is_minimal_and_available_for_sidecar_readiness(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", "c" * 64)
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert set(response.json()) == {"status", "version"}


def test_scientific_endpoints_require_exact_session_token(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "rgb.tif"
    _write_rgb(source)
    token = "d" * 64
    monkeypatch.setenv("DEPTHWIZARD_SESSION_TOKEN", token)
    client = TestClient(app)

    missing = client.post("/v1/inspect", json={"path": str(source)})
    assert missing.status_code == 401

    wrong = client.post(
        "/v1/inspect",
        json={"path": str(source)},
        headers={"x-depthwizard-token": "not-the-token"},
    )
    assert wrong.status_code == 401

    accepted = client.post(
        "/v1/inspect",
        json={"path": str(source)},
        headers={"x-depthwizard-token": token},
    )
    assert accepted.status_code == 200
    assert accepted.json()["crs"] == "EPSG:32643"
