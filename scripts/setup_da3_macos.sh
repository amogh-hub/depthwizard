#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR_ROOT="$ROOT/.vendor"
DA3_DIR="$VENDOR_ROOT/depth-anything-3"
DA3_REPO="https://github.com/ByteDance-Seed/Depth-Anything-3.git"
DA3_COMMIT="3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"

cd "$ROOT"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "ERROR: .venv is missing. Create the DepthWizard Python 3.12 environment first." >&2
  exit 1
fi

PYTHON="$ROOT/.venv/bin/python"

mkdir -p "$VENDOR_ROOT"

if [[ ! -d "$DA3_DIR/.git" ]]; then
  git clone --filter=blob:none "$DA3_REPO" "$DA3_DIR"
fi

git -C "$DA3_DIR" fetch --depth 1 origin "$DA3_COMMIT"
git -C "$DA3_DIR" checkout --detach "$DA3_COMMIT"
git -C "$DA3_DIR" reset --hard "$DA3_COMMIT"

# The official DA3 API eagerly imports all exporters. On macOS this pulls in optional native
# stacks (notably pycolmap / GS export code) even though DepthWizard only needs monocular depth.
# Those native stacks can load a second OpenMP runtime beside PyTorch and abort the process.
# Preserve the pinned official source while making export loading lazy: inference behavior is
# unchanged, and exporters are imported only if an export is actually requested.
DA3_API="$DA3_DIR/src/depth_anything_3/api.py"
"$PYTHON" - "$DA3_API" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
eager = "from depth_anything_3.utils.export import export\n"
lazy = '''def export(*args, **kwargs):\n    from depth_anything_3.utils.export import export as _export\n\n    return _export(*args, **kwargs)\n\n'''
if eager in text:
    text = text.replace(eager, lazy, 1)
elif lazy not in text:
    raise SystemExit("ERROR: pinned DA3 api.py no longer matches the expected compatibility patch")
path.write_text(text, encoding="utf-8")
print("Applied DA3 macOS lazy-export compatibility patch.")
PY

# The pinned geometry helper uses import-time torch.jit.script. That is valid in a normal Python
# environment but a PyInstaller frozen loader does not expose source for TorchScript compilation.
# Apply the exact audited script_if_tracing compatibility patch used by standalone packaging. The
# helper's tensor math is unchanged, eager inference remains eager, and any unexpected source shape
# fails closed rather than being guessed.
"$PYTHON" -m scripts.da3_frozen_compat "$DA3_DIR/src"

# Keep only the dependencies required by the monocular inference path. DA3 and DepthWizard both
# require NumPy < 2; OpenCV is constrained below the release line that would force NumPy 2.x.
"$PYTHON" -m pip install \
  "numpy==1.26.4" \
  "opencv-python>=4.10,<4.12" \
  "addict>=2.4,<3" \
  "einops>=0.8" \
  "huggingface_hub>=0.34" \
  "imageio>=2.37" \
  "omegaconf>=2.3" \
  "requests>=2.32" \
  "safetensors>=0.5" \
  "tqdm>=4.67" \
  "evo>=1.31" \
  "e3nn>=0.5"

# Export-only native packages from earlier setup attempts are not needed for DepthWizard's DA3
# prior and are deliberately removed to prevent accidental OpenMP/native-library collisions.
"$PYTHON" -m pip uninstall -y pycolmap moviepy pillow-heif plyfile >/dev/null 2>&1 || true

# Re-assert DepthWizard's own declared environment after vendor resolution.
"$PYTHON" -m pip install -e ".[ml,dev]"

SITE_PACKAGES="$($PYTHON - <<'PY'
import site
paths = site.getsitepackages()
if not paths:
    raise SystemExit("Could not resolve virtual-environment site-packages")
print(paths[0])
PY
)"

PTH_FILE="$SITE_PACKAGES/depthwizard_da3_vendor.pth"
printf '%s\n' "$DA3_DIR/src" > "$PTH_FILE"

export PYTORCH_ENABLE_MPS_FALLBACK=1

"$PYTHON" - <<'PY'
import platform

# Import OpenCV before PyTorch on macOS. DA3's input processor uses OpenCV; establishing its
# native runtime first avoids a known class of duplicate-OpenMP initialization failures.
import cv2  # noqa: F401
import numpy as np
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.geometry import affine_inverse

print("Depth Anything 3 import: OK")
print("Architecture:", platform.machine())
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("DA3 API:", DepthAnything3.__name__)

probe = torch.eye(4, dtype=torch.float32)
probe[:3, 3] = torch.tensor([3.0, -2.0, 5.0], dtype=torch.float32)
if not torch.allclose(affine_inverse(probe), torch.linalg.inv(probe), atol=1e-6, rtol=1e-6):
    raise SystemExit("ERROR: DA3 affine_inverse compatibility probe failed")
print("DA3 affine_inverse compatibility probe: PASS")

major = int(np.__version__.split(".", 1)[0])
if major >= 2:
    raise SystemExit(f"ERROR: NumPy {np.__version__} is incompatible with DepthWizard/DA3")
if platform.machine() == "arm64" and not torch.backends.mps.is_available():
    raise SystemExit("ERROR: Apple Silicon detected but MPS is unavailable")
PY

"$PYTHON" -m pip check

echo "DA3 source pinned at: $DA3_COMMIT"
echo "DA3 vendor directory: $DA3_DIR"
echo "DepthWizard DA3 runtime setup: PASS"
