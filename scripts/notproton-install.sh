#!/bin/bash
# Install metalgldrv for NotProton (docs/crossover.md): the latest GitHub release by default, or a
# driver built from this checkout (--build).
#
# Two ways, picked automatically:
#   plug-in  NotProton builds that read ~/.config/notproton/features.d add a "metalgldrv" switch to
#            each game's Properties > Compatibility > Developer section, which loads the driver into
#            that game only. Sub-switches under it turn on MetalFX upscaling and the accuracy
#            profiles. The runner isn't modified, so NotProton runner updates don't undo it.
#   patch    Older builds: the runner's winemac.so is patched (scripts/crossover-install.sh
#            --wine-root), so every NotProton game uses metalgldrv. A runner update undoes it; run
#            this again then.
# A game that is running keeps the driver it started with until it restarts.
#
# Usage: scripts/notproton-install.sh [options]
#   --plugin       install as a NotProton plug-in (default when NotProton supports them)
#   --patch        patch the runner instead
#   --runner DIR   the runner's CrossOver directory, for --patch (default: the newest one under
#                  "~/Library/Application Support/notproton/runners")
#   --release TAG  install that release instead of the latest (e.g. v0.1.1)
#   --build        build the driver from this checkout (build-universal/) instead of downloading it
#   --dylib PATH   install this driver file instead
#   --uninstall    plug-in: remove the switches; patch: restore winemac.so (the dylib is kept)
#   --dry-run      print what would be done, change nothing
#   -h, --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT="$HOME/Library/Application Support"
FEATURES="$HOME/.config/notproton/features.d"
MODE="" RUNNER="" UNINSTALL=0 DRY=0
PASS=()
# Marks the descriptors this script writes, so --uninstall removes only those.
MARK="# Written by metalgldrv's scripts/notproton-install.sh"

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }
say() { echo "==> $*"; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --plugin) MODE=plugin; shift ;;
        --patch) MODE=patch; shift ;;
        --runner) RUNNER="$2"; shift 2 ;;
        --uninstall) UNINSTALL=1; PASS+=("$1"); shift ;;
        --dry-run) DRY=1; PASS+=("$1"); shift ;;
        --build) PASS+=("$1"); shift ;;
        --release|--dylib) PASS+=("$1" "$2"); shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

# NotProton installs its compatibility tools into Steam; their run script reads features.d when the
# installed build supports plug-ins.
supports_plugins() {
    local run
    for run in "$SUPPORT/Steam/compatibilitytools.d"/notproton*/run; do
        [[ -f "$run" ]] && grep -q 'features\.d' "$run" && return 0
    done
    return 1
}

ours_installed() {
    local f
    for f in "$FEATURES"/*.conf; do
        [[ -f "$f" ]] && grep -qxF "$MARK" "$f" && return 0
    done
    return 1
}

if [[ -z "$MODE" && $UNINSTALL == 1 ]]; then
    # Undo whichever this script did.
    if ours_installed; then MODE=plugin; else MODE=patch; fi
elif [[ -z "$MODE" ]]; then
    if supports_plugins; then
        MODE=plugin
    else
        MODE=patch
        say "this NotProton has no plug-in support yet; patching the runner (--plugin to install the switches anyway)"
    fi
fi

# --- plug-in -----------------------------------------------------------------------------------

# One descriptor per switch. The main key is the one bomberfish/NotProton's metalgldrv branch uses,
# so a game's setting means the same in both.
descriptor() {  # file key label description [field ...]
    local file="$FEATURES/$1" key="$2" label="$3" desc="$4"
    shift 4
    if [[ $DRY == 1 ]]; then
        say "would write $file ($key)"
        return
    fi
    {
        echo "$MARK"
        echo "key = $key"
        echo "label = $label"
        echo "description = $desc"
        local f
        for f in "$@"; do echo "$f"; done
    } > "$file.tmp"
    mv -f "$file.tmp" "$file"
    say "wrote $file"
}

plugin_uninstall() {
    local f n=0
    for f in "$FEATURES"/*.conf; do
        [[ -f "$f" ]] && grep -qxF "$MARK" "$f" || continue
        if [[ $DRY == 1 ]]; then say "would remove $f"; else rm -f "$f"; say "removed $f"; fi
        n=$((n + 1))
    done
    [[ $n -gt 0 ]] || say "no metalgldrv switches installed in $FEATURES"
    say "the driver in $SUPPORT/metalgldrv is left in place (delete it if nothing else uses it)"
}

plugin_install() {
    # The driver only; NotProton loads it into each game that has the switch on.
    "$HERE/crossover-install.sh" --driver-only ${PASS[@]+"${PASS[@]}"}

    [[ $DRY == 1 ]] || mkdir -p "$FEATURES"
    local main=NOTPROTON_METALGLDRV
    descriptor 50-metalgldrv.conf "$main" "metalgldrv (OpenGL 4.6 on Metal)" \
        "OpenGL 4.6 on Metal for this game, instead of Apple's OpenGL 4.1. Tested with NotProton's Rosetta build." \
        "inject = 1" \
        "requires = Library/Application Support/metalgldrv/libmetalgldrv.dylib" \
        "env = METALGLDRV_HOOK_NSOPENGL=1"
    descriptor 51-metalgldrv-metalfx.conf NOTPROTON_METALGLDRV_METALFX "MetalFX upscaling" \
        "Presents at the display's native resolution, upscaling the game's frames with MetalFX." \
        "depends = $main" "env = METALGLDRV_USE_METALFX=1"
    descriptor 52-metalgldrv-strict.conf NOTPROTON_METALGLDRV_STRICT "Strict accuracy" \
        "Slower: bounds-checked fetches, precise math, the full tessellation emulation. Use one profile at a time." \
        "depends = $main" "env = METALGLDRV_STRICT=1"
    descriptor 53-metalgldrv-turbo.conf NOTPROTON_METALGLDRV_TURBO "Turbo" \
        "Faster, not conformant: fast math, anisotropy capped at 4x. Use one profile at a time." \
        "depends = $main" "env = METALGLDRV_TURBOHAX=1"
    descriptor 54-metalgldrv-speeddemon.conf NOTPROTON_METALGLDRV_SPEEDDEMON "Speed demon" \
        "Turbo, and queries and glFinish never wait for the GPU, no robust-access checks, anisotropy capped at 2x." \
        "depends = $main" "env = METALGLDRV_SPEEDDEMON=1"

    if [[ $DRY == 0 ]]; then
        local v
        v="$(strings "$SUPPORT/metalgldrv/libmetalgldrv.dylib" | grep -m1 'Core Profile metalgldrv' || true)"
        say "installed: ${v:-unknown version}"
        supports_plugins || say "note: this NotProton doesn't read plug-ins yet; the switches appear once it does"
        echo
        echo "Turn it on per game: Steam > the game's Properties > Compatibility > Developer > metalgldrv."
        echo "Undo with: $0 --plugin --uninstall"
    fi
}

# --- patch -------------------------------------------------------------------------------------

# The runner's CrossOver directory that holds the x86_64 Wine (the Rosetta build's).
find_runner() {
    local d best=""
    for d in "$SUPPORT/notproton/runners"/*/CrossOver "$SUPPORT/notproton/runners/current"; do
        [[ -f "$d/lib/wine/x86_64-unix/winemac.so" ]] || continue
        if [[ -z "$best" || "$d" -nt "$best" ]]; then best="$d"; fi
    done
    printf '%s' "$best"
}

patch_install() {
    [[ -n "$RUNNER" ]] || RUNNER="$(find_runner)"
    [[ -n "$RUNNER" && -d "$RUNNER" ]] || die "no NotProton runner found (install NotProton and start it once, or pass --runner)"
    RUNNER="$(cd "$RUNNER" && pwd -P)"
    say "runner: $RUNNER"
    if pgrep -f "$RUNNER/" >/dev/null 2>&1; then
        say "note: Wine from this runner is running; running games keep the old driver until they restart"
    fi

    "$HERE/crossover-install.sh" --wine-root "$RUNNER" ${PASS[@]+"${PASS[@]}"}

    if [[ $UNINSTALL == 0 && $DRY == 0 ]]; then
        local v
        v="$(strings "$SUPPORT/metalgldrv/libmetalgldrv.dylib" | grep -m1 'Core Profile metalgldrv' || true)"
        say "installed: ${v:-unknown version}"
    fi
}

case "$MODE/$UNINSTALL" in
    plugin/1) plugin_uninstall ;;
    plugin/0) plugin_install ;;
    patch/*) patch_install ;;
esac
