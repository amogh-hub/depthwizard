#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_BIN="$ROOT_DIR/node_modules/.bin/tauri"
SOURCE_ICON="$ROOT_DIR/public/depthwizard-mark.png"
BUNDLE_ICON="$ROOT_DIR/src-tauri/icons/icon.png"

if [[ "$(uname -s)" != "Darwin" ]]; then
  exec "$TAURI_BIN" "$@"
fi

if ! command -v sips >/dev/null 2>&1; then
  echo "DepthWizard: macOS 'sips' is required to prepare the Retina app icon." >&2
  exit 1
fi

if [[ ! -f "$SOURCE_ICON" ]]; then
  echo "DepthWizard: missing canonical in-app logo: $SOURCE_ICON" >&2
  exit 1
fi

if [[ ! -x "$TAURI_BIN" ]]; then
  echo "DepthWizard: Tauri CLI not found at $TAURI_BIN. Run npm ci first." >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
ORIGINAL_ICON="$TMP_DIR/original-icon.png"
GENERATED_ICON="$TMP_DIR/generated-icon.png"

cleanup() {
  if [[ -f "$ORIGINAL_ICON" ]]; then
    cp "$ORIGINAL_ICON" "$BUNDLE_ICON"
  fi
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

cp "$BUNDLE_ICON" "$ORIGINAL_ICON"

# Tauri's macOS bundle conversion needs a Retina-sized source. The canonical
# navbar mark is already visually approved, so generate the packaging-only
# source from that exact artwork with Apple's native image pipeline.
sips -s format png -z 1024 1024 "$SOURCE_ICON" --out "$GENERATED_ICON" >/dev/null

ICON_INFO="$(file "$GENERATED_ICON")"
if [[ "$ICON_INFO" != *"PNG image data, 1024 x 1024"* ]] || [[ "$ICON_INFO" != *"RGBA"* ]]; then
  echo "DepthWizard: generated macOS icon failed RGBA/1024 validation:" >&2
  echo "$ICON_INFO" >&2
  exit 1
fi

cp "$GENERATED_ICON" "$BUNDLE_ICON"
echo "DepthWizard: prepared 1024x1024 RGBA macOS bundle icon from the canonical app mark."

set +e
"$TAURI_BIN" "$@"
STATUS=$?
set -e
exit "$STATUS"
