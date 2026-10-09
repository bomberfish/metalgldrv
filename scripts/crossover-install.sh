#!/bin/bash
# Point CrossOver's Wine at metalgldrv instead of Apple's OpenGL (docs/crossover.md).
#
# winemac.so (Wine's macOS driver, x86_64, runs under Rosetta) links
# /System/Library/Frameworks/OpenGL.framework for CGL and dlopen()s it for the gl*
# entry points. This script:
#   1. copies the universal libmetalgldrv.dylib to a stable place (DEST),
#   2. backs up winemac.so (winemac.so.metalgldrv-backup, next to it),
#   3. rewrites winemac.so's OpenGL.framework dependency to DEST/libmetalgldrv.dylib
#      (install_name_tool -change) and re-signs it ad hoc (codesign -f -s -).
# At run time libmetalgldrv sees winemac.so load and redirects its dlopen of
# OpenGL.framework to itself (src/macos/hooks.mm).
#
# Usage: scripts/crossover-install.sh [options]
#   --app PATH       CrossOver app bundle to patch (default: first of
#                    ~/Applications/CrossOver-MetalGL.app, /Applications/CrossOver.app,
#                    "/Applications/CrossOver Preview.app"). Recommended: patch a copy,
#                    e.g. ditto "/Applications/CrossOver.app" ~/Applications/CrossOver-MetalGL.app
#   --wine-root DIR  a bare CrossOver directory instead of an app bundle (the one holding
#                    lib/wine/x86_64-unix/winemac.so), e.g. NotProton's runner:
#                    "~/Library/Application Support/notproton/runners/current"
#   --dylib PATH     driver to install (default: build-universal/libmetalgldrv.dylib;
#                    must contain x86_64)
#   --dest DIR       where the driver is installed (default:
#                    "~/Library/Application Support/metalgldrv")
#   --uninstall      restore winemac.so from the backup (the installed dylib is kept)
#   --dry-run        print what would be done, change nothing
#   -h, --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
APP="" WINE_ROOT="" DYLIB="$REPO/build-universal/libmetalgldrv.dylib" DEST="$HOME/Library/Application Support/metalgldrv"
UNINSTALL=0 DRY=0
WINEMAC_REL="Contents/SharedSupport/CrossOver/lib/wine/x86_64-unix/winemac.so"
BACKUP_SUFFIX=".metalgldrv-backup"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }
run() {
    if [[ $DRY == 1 ]]; then
        printf '[dry-run]'; printf ' %q' "$@"; printf '\n'
    else
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --app) APP="$2"; shift 2 ;;
        --wine-root) WINE_ROOT="$2"; shift 2 ;;
        --dylib) DYLIB="$2"; shift 2 ;;
        --dest) DEST="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

if [[ -n "$WINE_ROOT" ]]; then
    WINE_ROOT="${WINE_ROOT%/}"
    WINEMAC="$WINE_ROOT/lib/wine/x86_64-unix/winemac.so"
    APP="$WINE_ROOT"
elif [[ -z "$APP" ]]; then
    for c in "$HOME/Applications/CrossOver-MetalGL.app" "/Applications/CrossOver.app" "/Applications/CrossOver Preview.app"; do
        if [[ -f "$c/$WINEMAC_REL" ]]; then APP="$c"; break; fi
    done
    [[ -n "$APP" ]] || die "no CrossOver found; pass --app /path/to/CrossOver.app"
fi
APP="${APP%/}"
[[ -n "$WINE_ROOT" ]] || WINEMAC="$APP/$WINEMAC_REL"
BACKUP="$WINEMAC$BACKUP_SUFFIX"
TARGET="$DEST/libmetalgldrv.dylib"
[[ -f "$WINEMAC" ]] || die "$WINEMAC not found (is $APP a CrossOver bundle?)"
say "CrossOver: $APP"
say "winemac.so: $WINEMAC"

# The OpenGL.framework path winemac.so links (or our dylib, when already patched).
opengl_dep() { otool -L "$1" | awk 'NR > 1 && /OpenGL\.framework/ { print $1; exit }'; }
our_dep() { otool -L "$1" | awk 'NR > 1 && /libmetalgldrv\.dylib/ { sub(/^[ \t]+/, ""); sub(/ \(compatibility.*$/, ""); print; exit }'; }

# macOS App Management (Ventura+) blocks Terminal from modifying other developers' apps.
check_writable() {
    local dir probe
    dir="$(dirname "$WINEMAC")"
    probe="$dir/.metalgldrv-write-test.$$"
    if ! ( : > "$probe" ) 2>/dev/null; then
        cat >&2 <<EOF
error: cannot write inside $APP.
macOS blocks modifying another developer's app unless the app doing it (your terminal)
has App Management permission. Either:
  * System Settings > Privacy & Security > App Management: enable your terminal app,
    quit and reopen it, then run this script again; or
  * patch a copy instead of the original (recommended):
      ditto "$APP" "\$HOME/Applications/CrossOver-MetalGL.app"
      $0 --app "\$HOME/Applications/CrossOver-MetalGL.app"
    (if writing into the copy is blocked too, grant App Management as above).
EOF
        exit 1
    fi
    rm -f "$probe"
}

if [[ $UNINSTALL == 1 ]]; then
    [[ -f "$BACKUP" ]] || die "no backup at $BACKUP; nothing to restore"
    [[ $DRY == 1 ]] || check_writable
    if [[ -z "$(our_dep "$WINEMAC")" ]] && cmp -s "$WINEMAC" "$BACKUP"; then
        say "winemac.so is already the original"
    else
        say "restoring winemac.so from $BACKUP"
        run cp -p "$BACKUP" "$WINEMAC"
    fi
    run rm -f "$BACKUP"
    say "uninstalled. $TARGET is left in place (delete it if nothing else uses it)."
    exit 0
fi

# --- install ---------------------------------------------------------------
[[ -f "$DYLIB" ]] || die "$DYLIB not found. Build it first:
  cmake -G Ninja -B build-universal -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF && ninja -C build-universal metalgldrv"
archs="$(lipo -archs "$DYLIB")"
[[ " $archs " == *" x86_64 "* ]] || die "$DYLIB has no x86_64 slice ($archs); CrossOver's Wine runs x86_64 under Rosetta. Use the universal build."
[[ "$(file -b "$WINEMAC")" == *x86_64* ]] || die "winemac.so is not x86_64; this layout is not supported"

current="$(our_dep "$WINEMAC")"
if [[ -n "$current" ]]; then
    say "winemac.so already links $current"
    [[ -f "$BACKUP" ]] || die "winemac.so is patched but $BACKUP is missing; reinstall CrossOver (or restore winemac.so) first"
    src="$BACKUP"
else
    dep="$(opengl_dep "$WINEMAC")"
    [[ -n "$dep" ]] || die "winemac.so doesn't link OpenGL.framework; unknown CrossOver layout"
    if [[ -f "$BACKUP" ]] && ! cmp -s "$WINEMAC" "$BACKUP"; then
        die "$BACKUP exists and differs from the current (unpatched) winemac.so.
CrossOver was probably updated since the last install. Check, then remove the stale backup:
  rm \"$BACKUP\""
    fi
    src="$WINEMAC"
fi
dep="$(opengl_dep "$src")"
[[ -n "$dep" ]] || die "$src doesn't link OpenGL.framework"

if [[ $DRY == 1 ]]; then
    say "(dry run: not probing whether $APP is writable; App Management may block the real run)"
else
    check_writable
fi

say "installing driver: $TARGET"
run mkdir -p "$DEST"
run cp -f "$DYLIB" "$TARGET"
run codesign -f -s - "$TARGET"

if [[ ! -f "$BACKUP" ]]; then
    say "backup: $BACKUP"
    run cp -p "$WINEMAC" "$BACKUP"
fi

say "patching winemac.so: $dep -> $TARGET"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
run cp -p "$src" "$tmp/winemac.so"
if ! run install_name_tool -change "$dep" "$TARGET" "$tmp/winemac.so"; then
    die "install_name_tool failed (no room for the longer path?). Try a shorter --dest."
fi
run codesign -f -s - "$tmp/winemac.so"
if [[ $DRY == 0 ]]; then
    [[ "$(our_dep "$tmp/winemac.so")" == "$TARGET" ]] || die "patched winemac.so doesn't link $TARGET; not installing it"
fi
run cp -f "$tmp/winemac.so" "$WINEMAC"

if [[ $DRY == 0 ]]; then
    say "done. winemac.so now links:"
    otool -L "$WINEMAC" | grep -E "libmetalgldrv|OpenGL" || true
fi
cat <<EOF

Next: start a Windows OpenGL app from this CrossOver ($APP). With
METALGLDRV_LOG=2 (the default) Wine's log / terminal shows lines like
  [metalgldrv I] CrossOver/Wine: .../winemac.so: dlopen rebound (1 slot); OpenGL.framework -> $TARGET
  [metalgldrv I] NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv
  [metalgldrv I] CGL: created GL 4.6 core context ...
Undo with: $0 $([[ -n "$WINE_ROOT" ]] && echo "--wine-root \"$WINE_ROOT\"" || echo "--app \"$APP\"") --uninstall
EOF
