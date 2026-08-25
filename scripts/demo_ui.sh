#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP="$ROOT/apps/desktop"
SOURCE_MESH="$ROOT/artifacts/demo/india-absolute-mesh/terrain-lod1.glb"
ABSOLUTE_REPORT="$ROOT/artifacts/demo/india-absolute/demo_report.json"
BENCHMARK_REPORT="$ROOT/artifacts/benchmark/ortholoc-demo/benchmark_report.json"
PUBLIC_DIR="$DESKTOP/public/demo"
PUBLIC_MESH="$PUBLIC_DIR/terrain.glb"

for required in "$SOURCE_MESH" "$ABSOLUTE_REPORT" "$BENCHMARK_REPORT"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: $required is missing." >&2
    echo "Run 'make demo-india-absolute' and 'make benchmark-ortholoc-demo' before 'make demo-ui'." >&2
    exit 1
  fi
done

mkdir -p "$PUBLIC_DIR"
cp "$SOURCE_MESH" "$PUBLIC_MESH"
cp "$ABSOLUTE_REPORT" "$PUBLIC_DIR/absolute_demo_report.json"
cp "$BENCHMARK_REPORT" "$PUBLIC_DIR/benchmark_report.json"

cd "$DESKTOP"
if [[ ! -d node_modules ]]; then
  echo "Installing desktop dependencies..."
  npm install
fi

(
  sleep 2
  open "http://127.0.0.1:5173/?demo=1"
) >/dev/null 2>&1 &

echo "Starting DepthWizard India absolute-DSM workspace at http://127.0.0.1:5173/?demo=1"
echo "Press Ctrl+C in this terminal when you are finished previewing it."
exec npm run dev -- --host 127.0.0.1 --port 5173 --strictPort
