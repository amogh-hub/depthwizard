from __future__ import annotations

import hashlib

import pytest

from depthwizard.geometry_prior import da3 as da3_module
from depthwizard.geometry_prior.da3 import DA3MonocularPrior


def test_packaged_da3_snapshot_rejects_checkpoint_hash_mismatch(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / da3_module.DA3_CHECKPOINT_FILE).write_bytes(b"untrusted checkpoint")
    monkeypatch.setenv("DEPTHWIZARD_DA3_SNAPSHOT", str(tmp_path))

    with pytest.raises(RuntimeError, match="checkpoint SHA-256 mismatch"):
        DA3MonocularPrior()._resolve_production_snapshot()


def test_packaged_da3_snapshot_records_verified_byte_identity(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / da3_module.DA3_CHECKPOINT_FILE
    checkpoint.write_bytes(b"test checkpoint bytes")
    expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(da3_module, "DA3_CHECKPOINT_SHA256", expected)
    monkeypatch.setenv("DEPTHWIZARD_DA3_SNAPSHOT", str(tmp_path))
    prior = DA3MonocularPrior()

    snapshot = prior._resolve_production_snapshot()

    assert snapshot == tmp_path
    assert prior._resolved_checkpoint_path == checkpoint.resolve()
    assert prior._resolved_checkpoint_sha256 == expected
