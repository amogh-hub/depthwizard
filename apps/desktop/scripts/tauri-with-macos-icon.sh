#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAURI_BIN="$ROOT_DIR/node_modules/.bin/tauri"
SOURCE_ICON="$ROOT_DIR/public/depthwizard-mark.png"
ICON_DIR="$ROOT_DIR/src-tauri/icons"

if [[ ! -x "$TAURI_BIN" ]]; then
  echo "DepthWizard: Tauri CLI not found at $TAURI_BIN. Run npm ci first." >&2
  exit 1
fi

# Only prepare the macOS bundle icon set for actual macOS builds. Other Tauri
# subcommands and non-macOS platforms pass straight through unchanged.
if [[ "$(uname -s)" != "Darwin" ]] || [[ "${1:-}" != "build" ]]; then
  exec "$TAURI_BIN" "$@"
fi

if [[ ! -f "$SOURCE_ICON" ]]; then
  echo "DepthWizard: missing canonical app mark: $SOURCE_ICON" >&2
  exit 1
fi

TMP_DIR="$(mktemp -d)"
ORIGINAL_ICONS="$TMP_DIR/original-icons"
EXTRACTED_ICONSET="$TMP_DIR/validated.iconset"
mkdir -p "$ORIGINAL_ICONS"
cp -R "$ICON_DIR/." "$ORIGINAL_ICONS/"

cleanup() {
  rm -rf "$ICON_DIR"
  mkdir -p "$ICON_DIR"
  cp -R "$ORIGINAL_ICONS/." "$ICON_DIR/"
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT INT TERM

# Generate the complete platform icon family with Tauri's official generator.
# This produces the PNG sizes plus a valid macOS ICNS instead of asking the
# bundler to infer an ICNS slot from one arbitrary PNG.
rm -rf "$ICON_DIR"
mkdir -p "$ICON_DIR"
"$TAURI_BIN" icon "$SOURCE_ICON" --output "$ICON_DIR"

required_icons=(
  "32x32.png"
  "128x128.png"
  "128x128@2x.png"
  "icon.icns"
  "icon.ico"
)
for icon in "${required_icons[@]}"; do
  if [[ ! -s "$ICON_DIR/$icon" ]]; then
    echo "DepthWizard: Tauri icon generation did not produce $icon" >&2
    exit 1
  fi
done

# Fail closed if Apple's own icon tooling cannot decode the generated ICNS.
iconutil -c iconset "$ICON_DIR/icon.icns" -o "$EXTRACTED_ICONSET"
if [[ ! -d "$EXTRACTED_ICONSET" ]]; then
  echo "DepthWizard: generated macOS ICNS failed iconutil validation." >&2
  exit 1
fi

PNG_INFO="$(file "$ICON_DIR/128x128@2x.png")"
if [[ "$PNG_INFO" != *"PNG image data"* ]] || [[ "$PNG_INFO" != *"RGBA"* ]]; then
  echo "DepthWizard: generated Retina PNG failed RGBA validation:" >&2
  echo "$PNG_INFO" >&2
  exit 1
fi

echo "DepthWizard: generated and validated the complete Tauri macOS icon set from the canonical app mark."

set +e
"$TAURI_BIN" "$@"
STATUS=$?
set -e
exit "$STATUS"
