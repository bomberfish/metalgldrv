#!/usr/bin/env bash
# Build Amethyst (AngelAuraMC/Amethyst-iOS) with metalgldrv as the selectable
# "MetalGL (experimental)" renderer, sign it with a development profile and
# optionally install it on a device. See README.md in this directory.
#
# usage: demo/amethyst/build.sh [step...]
#   steps: fetch metalgl amethyst sign install launch   (default: all but launch;
#          install only runs when DEVICE is set)
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

AMETHYST_REPO=${AMETHYST_REPO:-https://github.com/AngelAuraMC/Amethyst-iOS.git}
AMETHYST_COMMIT=${AMETHYST_COMMIT:-f98c4133c72d39e06ceac42b57cc4477c52105fc}
WORK=${AMETHYST_WORK:-$HOME/metalgldrv-deps/amethyst-work}
MGL_BUILD=${METALGL_BUILD_DIR:-$WORK-metalgldrv-build}
MGL_BUILD_TYPE=${METALGL_BUILD_TYPE:-RelWithDebInfo}
BOOTJDK=${BOOTJDK:-/Library/Java/JavaVirtualMachines/zulu-8.jdk/Contents/Home/bin}
JOBS=${JOBS:-$(sysctl -n hw.ncpu)}
PROFILE=${PROFILE:-}             # .mobileprovision to sign with (unset: leave ldid ad-hoc signature)
SIGN_IDENTITY=${SIGN_IDENTITY:-} # default: the profile's certificate found in the keychain
BUNDLE_ID=${BUNDLE_ID:-}         # default: from the profile's application-identifier
DEVICE=${DEVICE:-}               # devicectl device UDID/name for install/launch

APP=$WORK/artifacts/Payload/AngelAuraAmethyst.app
export PATH=/opt/homebrew/bin:$PATH

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

log() { printf '\033[1;35m==> %s\033[0m\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

step_fetch() {
    log "Amethyst @ ${AMETHYST_COMMIT:0:7} -> $WORK"
    if [ ! -d "$WORK/.git" ]; then
        git clone "$AMETHYST_REPO" "$WORK"
    fi
    cd "$WORK"
    git cat-file -e "$AMETHYST_COMMIT^{commit}" 2>/dev/null || git fetch origin
    # The work tree belongs to this script: reset it and re-apply the patches.
    git am --abort >/dev/null 2>&1 || true
    git checkout -q -f -B metalgl "$AMETHYST_COMMIT"
    git -c user.name=metalgldrv -c user.email=metalgldrv@localhost -c commit.gpgsign=false \
        am -q --3way "$HERE"/patches/*.patch
    # Two submodule repos (khanhduytran0/*) were deleted from GitHub; the pinned
    # commits are still reachable through the upstream fork networks.
    git submodule init -q
    git config submodule.Natives/external/DBNumberedSlider.url https://github.com/immago/DBNumberedSlider
    git config submodule.Natives/external/fishhook.url https://github.com/facebook/fishhook
    git submodule update --init --recursive
}

step_metalgl() {
    log "libmetalgldrv.dylib (iOS, $MGL_BUILD_TYPE) -> $MGL_BUILD"
    cmake -G Ninja -S "$REPO" -B "$MGL_BUILD" \
        -DCMAKE_TOOLCHAIN_FILE="$REPO/cmake/ios.cmake" \
        -DCMAKE_BUILD_TYPE="$MGL_BUILD_TYPE" >/dev/null
    ninja -C "$MGL_BUILD" metalgldrv
}

step_amethyst() {
    log "Amethyst (native java jre assets payload)"
    [ -x "$BOOTJDK/javac" ] || die "JDK 8 not found at BOOTJDK=$BOOTJDK"
    cd "$WORK"
    gmake -j"$JOBS" payload BOOTJDK="$BOOTJDK"
    # actool rewrites a tracked file; keep the work tree clean
    git checkout -q -- Natives/resources/Assets.car
    log "embedding libmetalgldrv.dylib"
    cp "$MGL_BUILD/libmetalgldrv.dylib" "$APP/Frameworks/libmetalgldrv.dylib"
}

sign() {
    local f=$1 out; shift
    out=$(codesign -f -s "$SIGN_IDENTITY" --timestamp=none "$@" "$f" 2>&1) || die "codesign $f: $out"
}

is_macho() { case "$(file -b "$1")" in Mach-O*) return 0;; *) return 1;; esac; }

step_sign() {
    [ -d "$APP" ] || die "no app at $APP (run the amethyst step)"
    if [ -z "$PROFILE" ]; then
        log "PROFILE not set: skipping codesign (app keeps the Makefile's ldid signature)"
        return
    fi
    local prof=$TMP/profile.plist ents=$TMP/entitlements.plist
    security cms -D -i "$PROFILE" >"$prof"
    local appid team
    appid=$(plutil -extract Entitlements.application-identifier raw -o - "$prof")
    team=$(plutil -extract TeamIdentifier.0 raw -o - "$prof")
    [ -n "$BUNDLE_ID" ] || BUNDLE_ID=${appid#"$team".}
    case "$BUNDLE_ID" in *'*'*) die "wildcard profile ($appid): set BUNDLE_ID";; esac
    if [ -z "$SIGN_IDENTITY" ]; then
        # Pick the keychain identity whose certificate is listed in the profile.
        local i=0 sha
        while plutil -extract "DeveloperCertificates.$i" raw -o - "$prof" >"$TMP/cert.b64" 2>/dev/null; do
            sha=$(base64 -d -i "$TMP/cert.b64" | shasum -a 1 | cut -d' ' -f1 | tr a-f A-F)
            if security find-identity -v -p codesigning | grep -q "$sha"; then SIGN_IDENTITY=$sha; break; fi
            i=$((i + 1))
        done
        [ -n "$SIGN_IDENTITY" ] || die "no keychain identity matches $PROFILE; set SIGN_IDENTITY"
    fi
    # Entitlements = exactly what the profile grants (Amethyst's sideload
    # entitlements add com.apple.private.* keys a dev cert can't sign).
    plutil -extract Entitlements xml1 -o "$ents" "$prof"
    log "signing $BUNDLE_ID with $SIGN_IDENTITY"
    plutil -replace CFBundleIdentifier -string "$BUNDLE_ID" "$APP/Info.plist"
    cp "$PROFILE" "$APP/embedded.mobileprovision"

    local f n=0
    # Loose Mach-Os (dylibs, JREs) first, then framework bundles, then the app.
    while IFS= read -r -d '' f; do
        [ "$f" = "$APP/AngelAuraAmethyst" ] && continue
        case "$f" in *.framework/*) continue;; esac
        is_macho "$f" || continue
        sign "$f"
        n=$((n + 1))
    done < <(find "$APP" -type f -print0)
    for f in "$APP"/Frameworks/*.framework; do
        sign "$f"
        n=$((n + 1))
    done
    sign "$APP" --entitlements "$ents" --generate-entitlement-der
    codesign --verify --deep --strict "$APP"
    log "signed $n nested binaries + app"
}

step_install() {
    [ -n "$DEVICE" ] || { log "DEVICE not set: skipping install"; return; }
    log "installing on $DEVICE"
    xcrun devicectl device install app --device "$DEVICE" "$APP"
}

step_launch() {
    [ -n "$DEVICE" ] || die "DEVICE not set"
    local id
    id=$(plutil -extract CFBundleIdentifier raw -o - "$APP/Info.plist")
    log "launching $id on $DEVICE (Ctrl-C to detach)"
    xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --console "$id"
}

steps=("$@")
[ ${#steps[@]} -gt 0 ] || steps=(fetch metalgl amethyst sign install)
for s in "${steps[@]}"; do
    case "$s" in
    fetch|metalgl|amethyst|sign|install|launch) "step_$s" ;;
    *) die "unknown step '$s' (fetch metalgl amethyst sign install launch)" ;;
    esac
done
