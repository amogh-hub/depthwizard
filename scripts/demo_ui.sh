#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$ROOT/apps/desktop"
SOURCE_MESH="$ROOT/artifacts/demo/geotiff-mesh/terrain-lod2.glb"
MESH_REPORT="$ROOT/artifacts/demo/geotiff-mesh/mesh_report.json"
RECON_REPORT="$ROOT/artifacts/demo/geotiff-rdsm/reconstruction_report.json"
PUBLIC_DIR="$DESKTOP/public/demo"
PUBLIC_MESH="$PUBLIC_DIR/terrain.glb"

for required in "$SOURCE_MESH" "$MESH_REPORT" "$RECON_REPORT"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: $required is missing. Run 'make demo-rdsm' and 'make demo-mesh' first." >&2
    exit 1
  fi
done

mkdir -p "$PUBLIC_DIR"
cp "$SOURCE_MESH" "$PUBLIC_MESH"
cp "$MESH_REPORT" "$PUBLIC_DIR/mesh_report.json"
cp "$RECON_REPORT" "$PUBLIC_DIR/reconstruction_report.json"

cd "$DESKTOP"
if [[ ! -d node_modules ]]; then
  echo "Installing desktop dependencies..."
  npm install
fi

(
  sleep 2
  open "http://127.0.0.1:5173/?demo=1"
) >/dev/null 2>&1 &

echo "Starting DepthWizard terrain workspace at http://127.0.0.1:5173/?demo=1"
echo "Press Ctrl+C in this terminal when you are finished previewing it."
exec npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
