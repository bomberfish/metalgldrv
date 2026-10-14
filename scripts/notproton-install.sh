#!/bin/bash
# Install metalgldrv into NotProton's CrossOver runner (docs/crossover.md): the latest GitHub release by
# default, or a driver built from this checkout (--build).
#
# A wrapper around scripts/crossover-install.sh --wine-root for NotProton's runner. NotProton starts
# Steam games with the runner's Wine, so patching the runner's winemac.so makes every NotProton game
# use metalgldrv for OpenGL. A game that is running keeps the driver it started with until it restarts.
# When NotProton updates its runner, the new one is unpatched: run this again.
#
# Usage: scripts/notproton-install.sh [options]
#   --runner DIR   the runner (default: "~/Library/Application Support/notproton/runners/current")
#   --release TAG  install that release instead of the latest (e.g. v0.1.0)
#   --build        build the driver from this checkout (build-universal/) instead of downloading it
#   --dylib PATH   install this driver file instead
#   --uninstall    restore the runner's winemac.so (the installed dylib is kept)
#   --dry-run      print what would be done, change nothing
#   -h, --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HOME/Library/Application Support/notproton/runners/current"
PASS=()

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; }
die() { echo "error: $*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runner) RUNNER="$2"; shift 2 ;;
        --build|--uninstall|--dry-run) PASS+=("$1"); shift ;;
        --release|--dylib) PASS+=("$1" "$2"); shift 2 ;;
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

"$HERE/crossover-install.sh" --wine-root "$RUNNER" ${PASS[@]+"${PASS[@]}"}

if [[ " ${PASS[*]-} " != *" --uninstall "* && " ${PASS[*]-} " != *" --dry-run "* ]]; then
    v="$(strings "$HOME/Library/Application Support/metalgldrv/libmetalgldrv.dylib" | grep -m1 'Core Profile metalgldrv' || true)"
    echo "==> installed: ${v:-unknown version}"
fi
