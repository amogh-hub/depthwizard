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

# DA3's published package metadata includes CUDA/desktop extras that are not required for
# DepthWizard's monocular prior path and are problematic on Apple Silicon (notably xformers).
# Keep the official source pinned, and install only the import/runtime dependencies needed by
# the official Python API on macOS. xformers is optional in DA3's DINO layers and falls back.
"$PYTHON" -m pip install \
  "einops>=0.8" \
  "huggingface_hub>=0.34" \
  "imageio>=2.37" \
  "moviepy==1.0.3" \
  "opencv-python>=4.10" \
  "omegaconf>=2.3" \
  "requests>=2.32" \
  "safetensors>=0.5" \
  "tqdm>=4.67" \
  "evo>=1.31" \
  "e3nn>=0.5" \
  "plyfile>=1.1" \
  "pillow-heif>=1.0" \
  "pycolmap>=3.12"

SITE_PACKAGES="$($PYTHON - <<'PY'
import site
paths = site.getsitepackages()
if not paths:
    raise SystemExit("Could not resolve virtual-environment site-packages")
print(paths[0])
PY
)"

# Make the pinned official checkout importable without installing DA3's full dependency metadata.
PTH_FILE="$SITE_PACKAGES/depthwizard_da3_vendor.pth"
printf '%s\n' "$DA3_DIR/src" > "$PTH_FILE"

export PYTORCH_ENABLE_MPS_FALLBACK=1

"$PYTHON" - <<'PY'
import platform

import torch
from depth_anything_3.api import DepthAnything3

print("Depth Anything 3 import: OK")
print("Architecture:", platform.machine())
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
print("DA3 API:", DepthAnything3.__name__)

if platform.machine() == "arm64" and not torch.backends.mps.is_available():
    raise SystemExit("ERROR: Apple Silicon detected but MPS is unavailable")
PY

echo "DA3 source pinned at: $DA3_COMMIT"
echo "DA3 vendor directory: $DA3_DIR"
echo "DepthWizard DA3 runtime setup: PASS"
