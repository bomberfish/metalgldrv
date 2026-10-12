#!/bin/bash
# Build the universal driver and install it into NotProton's CrossOver runner (docs/crossover.md).
#
# A wrapper around scripts/crossover-install.sh --wine-root for NotProton's runner. NotProton starts
# Steam games with the runner's Wine, so patching the runner's winemac.so makes every NotProton game
# use metalgldrv for OpenGL. A game that is running keeps the driver it started with until it restarts.
# When NotProton updates its runner, the new one is unpatched: run this again.
#
# Usage: scripts/notproton-install.sh [options]
#   --runner DIR   the runner (default: "~/Library/Application Support/notproton/runners/current")
#   --no-build     install build-universal/libmetalgldrv.dylib as it is
#   --uninstall    restore the runner's winemac.so (the installed dylib is kept)
#   --dry-run      print what would be done, change nothing
#   -h, --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RUNNER="$HOME/Library/Application Support/notproton/runners/current"
BUILD=1 PASS=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner) RUNNER="$2"; shift 2 ;;
        --no-build) BUILD=0; shift ;;
        --uninstall) BUILD=0; PASS+=("$1"); shift ;;
        --dry-run) BUILD=0; PASS+=("$1"); shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option $1 (see --help)" ;;
    esac
done

[[ -d "$RUNNER" ]] || die "no NotProton runner at $RUNNER (install NotProton and start it once, or pass --runner)"
RUNNER="$(cd "$RUNNER" && pwd -P)"
echo "==> runner: $RUNNER"

if pgrep -f "$RUNNER/" >/dev/null 2>&1; then
    echo "==> note: Wine from this runner is running; running games keep the old driver until they restart"
fi

if [[ $BUILD == 1 ]]; then
    B="$REPO/build-universal"
    [[ -f "$B/build.ninja" ]] || cmake -G Ninja -S "$REPO" -B "$B" -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF
    ninja -C "$B" metalgldrv
fi

"$HERE/crossover-install.sh" --wine-root "$RUNNER" ${PASS[@]+"${PASS[@]}"}

if [[ ${#PASS[@]} == 0 ]]; then
    v="$(strings "$HOME/Library/Application Support/metalgldrv/libmetalgldrv.dylib" | grep -m1 'Core Profile metalgldrv' || true)"
    echo "==> installed: ${v:-unknown version}"
fi
