from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GEOMETRY_RELATIVE_PATH = Path("depth_anything_3/utils/geometry.py")
UPSTREAM_DECORATED_SIGNATURE = "@torch.jit.script\ndef affine_inverse(A: torch.Tensor):"
FROZEN_COMPAT_SIGNATURE = "@torch.jit.script_if_tracing\ndef affine_inverse(A: torch.Tensor):"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def apply_da3_frozen_compat(source_root: str | Path) -> dict[str, object]:
    """Apply the pinned DA3 PyInstaller/TorchScript compatibility patch, fail-closed.

    The pinned DA3 geometry module decorates ``affine_inverse`` with ``torch.jit.script``.
    PyInstaller frozen modules do not expose Python source through their loader, so import-time
    scripting fails before packaged DA3 inference can begin. ``torch.jit.script_if_tracing`` is the
    public PyTorch equivalent for library helpers that only need scripting while tracing: ordinary
    eager inference executes the same tensor function without import-time source compilation.

    Only the exact pinned signature is modified. An unexpected vendor source shape is rejected
    instead of being guessed or silently rewritten.
    """
    root = Path(source_root)
    geometry_path = root / GEOMETRY_RELATIVE_PATH
    if not geometry_path.is_file():
        raise RuntimeError(f"pinned DA3 geometry source is missing: {geometry_path}")

    before = geometry_path.read_text(encoding="utf-8")
    upstream_count = before.count(UPSTREAM_DECORATED_SIGNATURE)
    patched_count = before.count(FROZEN_COMPAT_SIGNATURE)

    if upstream_count == 1 and patched_count == 0:
        after = before.replace(UPSTREAM_DECORATED_SIGNATURE, FROZEN_COMPAT_SIGNATURE, 1)
        geometry_path.write_text(after, encoding="utf-8")
        changed = True
    elif upstream_count == 0 and patched_count == 1:
        after = before
        changed = False
    else:
        raise RuntimeError(
            "pinned DA3 geometry.py no longer matches the audited affine_inverse compatibility "
            f"contract: upstream_count={upstream_count}, patched_count={patched_count}"
        )

    verified = geometry_path.read_text(encoding="utf-8")
    if verified.count(FROZEN_COMPAT_SIGNATURE) != 1 or UPSTREAM_DECORATED_SIGNATURE in verified:
        raise RuntimeError("DA3 frozen TorchScript compatibility patch did not verify after write")

    return {
        "status": "PASS_DA3_FROZEN_TORCHSCRIPT_COMPAT",
        "geometry_path": str(geometry_path.resolve()),
        "changed_this_run": changed,
        "upstream_decorator": "torch.jit.script",
        "packaged_decorator": "torch.jit.script_if_tracing",
        "before_sha256": _sha256_text(before),
        "after_sha256": _sha256_text(verified),
        "model_math_changed": False,
        "semantics": (
            "Packaging compatibility only: affine_inverse tensor operations are unchanged; eager "
            "inference avoids import-time TorchScript source compilation, while tracing retains "
            "script compilation through torch.jit.script_if_tracing."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch pinned DA3 for frozen-runtime TorchScript")
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    report = apply_da3_frozen_compat(args.source_root)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
