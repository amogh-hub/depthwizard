#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$ROOT/apps/desktop"
SOURCE_MESH="$ROOT/artifacts/demo/geotiff-mesh/terrain-lod2.glb"
PUBLIC_DIR="$DESKTOP/public/demo"
PUBLIC_MESH="$PUBLIC_DIR/terrain.glb"

if [[ ! -f "$SOURCE_MESH" ]]; then
  echo "ERROR: $SOURCE_MESH is missing. Run 'make demo-mesh' first." >&2
  exit 1
fi

mkdir -p "$PUBLIC_DIR"
cp "$SOURCE_MESH" "$PUBLIC_MESH"

cd "$DESKTOP"
if [[ ! -d node_modules ]]; then
  echo "Installing desktop dependencies..."
  npm install
fi

# Open the real generated mesh in the DepthWizard scientific workspace after Vite is ready.
(
  sleep 2
  open "http://127.0.0.1:5173/?demo=1"
) >/dev/null 2>&1 &

echo "Starting DepthWizard terrain workspace at http://127.0.0.1:5173/?demo=1"
echo "Press Ctrl+C in this terminal when you are finished previewing it."
exec npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
