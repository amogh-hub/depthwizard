from __future__ import annotations

from pathlib import Path

import pytest

from scripts.da3_frozen_compat import (
    FROZEN_COMPAT_SIGNATURE,
    GEOMETRY_RELATIVE_PATH,
    UPSTREAM_DECORATED_SIGNATURE,
    apply_da3_frozen_compat,
)


def _write_geometry(root: Path, signature: str) -> Path:
    path = root / GEOMETRY_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "import torch\n\n" + signature + "\n    return A\n",
        encoding="utf-8",
    )
    return path


def test_da3_frozen_compat_rewrites_exact_pinned_signature_and_is_idempotent(tmp_path: Path) -> None:
    geometry = _write_geometry(tmp_path, UPSTREAM_DECORATED_SIGNATURE)

    first = apply_da3_frozen_compat(tmp_path)
    assert first["status"] == "PASS_DA3_FROZEN_TORCHSCRIPT_COMPAT"
    assert first["changed_this_run"] is True
    assert first["model_math_changed"] is False
    text = geometry.read_text(encoding="utf-8")
    assert FROZEN_COMPAT_SIGNATURE in text
    assert UPSTREAM_DECORATED_SIGNATURE not in text

    second = apply_da3_frozen_compat(tmp_path)
    assert second["changed_this_run"] is False
    assert second["after_sha256"] == first["after_sha256"]


def test_da3_frozen_compat_rejects_unexpected_vendor_source(tmp_path: Path) -> None:
    _write_geometry(
        tmp_path,
        "@torch.jit.ignore\ndef affine_inverse(A: torch.Tensor):",
    )

    with pytest.raises(RuntimeError, match="no longer matches"):
        apply_da3_frozen_compat(tmp_path)
