#!/bin/sh
set -eu

PROJECT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
APP_PATH="$PROJECT_DIR/target/release/bundle/osx/Edge TTS Studio.app"
DMG_DIR="$PROJECT_DIR/target/release/bundle/dmg"
DMG_PATH="$DMG_DIR/Edge TTS Studio.dmg"
STAGING_DIR=$(mktemp -d "${TMPDIR:-/tmp}/edge-tts-studio-dmg.XXXXXX")

cleanup() {
    rm -rf "$STAGING_DIR"
}
trap cleanup EXIT INT TERM

cd "$PROJECT_DIR"
cargo bundle --release --format osx

# Ad-hoc signing catches damaged/nested-code issues and lets local builds run.
# Distribution outside the developer's Mac still requires Developer ID signing
# and Apple notarization to avoid Gatekeeper warnings.
codesign --force --deep --sign - --timestamp=none "$APP_PATH"
codesign --verify --deep --strict --verbose=2 "$APP_PATH"

mkdir -p "$DMG_DIR"
ditto "$APP_PATH" "$STAGING_DIR/Edge TTS Studio.app"
ln -s /Applications "$STAGING_DIR/Applications"

# The Applications symlink gives Finder the standard drag-to-install workflow.
hdiutil create \
    -volname "Edge TTS Studio" \
    -srcfolder "$STAGING_DIR" \
    -format UDZO \
    -ov \
    "$DMG_PATH"
hdiutil verify "$DMG_PATH"

printf 'Created installable disk image:\n%s\n' "$DMG_PATH"
