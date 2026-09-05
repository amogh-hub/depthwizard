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

# Non-build commands use the committed compile-safe Tauri configuration directly.
# Only bundle builds need the generated Retina/ICNS icon family.
if [[ "${1:-}" != "build" ]]; then
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
# The repository itself intentionally keeps only icons/icon.png so raw Cargo
# compilation, Clippy, tests and CI never depend on generated bundle assets.
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

if [[ "$(uname -s)" == "Darwin" ]]; then
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
fi

# Tauri v2 merges raw JSON supplied through --config over tauri.conf.json.
# Keep the committed config compile-safe with only icons/icon.png, then replace
# bundle.icon exclusively for the packaging command. Using raw JSON avoids any
# alternate config-file directory affecting relative asset paths.
BUNDLE_OVERRIDE='{"bundle":{"icon":["icons/32x32.png","icons/128x128.png","icons/128x128@2x.png","icons/icon.icns","icons/icon.ico"]}}'

# A local macOS package still needs a complete bundle resource seal even when the
# workstation has no Developer ID certificate. Tauri honours APPLE_SIGNING_IDENTITY,
# so default to ad-hoc signing while preserving an explicitly configured release
# identity. Run the DMG helper in its deterministic non-Finder CI mode by default;
# developers can opt back into Finder cosmetics by exporting
# TAURI_BUNDLER_DMG_IGNORE_CI=true before invoking this wrapper.
if [[ "$(uname -s)" == "Darwin" ]]; then
  export APPLE_SIGNING_IDENTITY="${APPLE_SIGNING_IDENTITY:--}"
  export CI="${CI:-true}"
  echo "DepthWizard: macOS bundle signing identity is configured and DMG assembly is non-interactive."
fi

echo "DepthWizard: generated and validated the complete bundle icon set from the canonical app mark."
echo "DepthWizard: raw Cargo uses tracked icons/icon.png; Tauri packaging uses the temporary Retina/ICNS override."

shift
set +e
"$TAURI_BIN" build --config "$BUNDLE_OVERRIDE" "$@"
STATUS=$?
set -e
exit "$STATUS"
