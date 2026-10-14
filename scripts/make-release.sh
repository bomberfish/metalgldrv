#!/bin/bash
# Build the release assets that scripts/crossover-install.sh and notproton-install.sh download.
#
# Builds the universal (arm64 + x86_64) driver in build-dist/ (macOS 26 and later, the first with
# Metal 4) from a committed tree, and writes to dist/:
#   libmetalgldrv-macos-universal.zip         libmetalgldrv.dylib (debug info stripped, signed
#                                             ad hoc), VERSION (git describe), LICENCE
#   libmetalgldrv-macos-universal.zip.sha256
#   libmetalgldrv.dylib                       the same dylib on its own, for manual installs
#   libmetalgldrv-macos-universal.dSYM.zip    the debug symbols, for symbolicating crash logs
#   metalgldrv-fabric-<version>.jar           the Fabric mod with that dylib (mods/fabric/build.sh)
# The asset names carry no version, so releases/latest/download/<name> always finds them.
# Tag the release commit first: the driver reports git describe (e.g. v0.1.0) as its version.
#
# Usage: scripts/make-release.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="$REPO/build-dist" OUT="$REPO/dist" NAME="libmetalgldrv-macos-universal"
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || { echo "error: uncommitted changes; commit or stash them first" >&2; exit 1; }
VERSION="$(git describe --tags --always --dirty)"
echo "==> metalgldrv $VERSION"

cmake -G Ninja -S "$REPO" -B "$B" -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0
ninja -C "$B" metalgldrv
xcrun lipo -archs "$B/libmetalgldrv.dylib"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
mkdir -p "$stage/lib" "$stage/dsym"
cp "$B/libmetalgldrv.dylib" "$stage/lib/libmetalgldrv.dylib"
xcrun dsymutil "$stage/lib/libmetalgldrv.dylib" -o "$stage/dsym/libmetalgldrv.dylib.dSYM"
xcrun strip -S "$stage/lib/libmetalgldrv.dylib"
codesign -f -s - "$stage/lib/libmetalgldrv.dylib"
echo "$VERSION" > "$stage/lib/VERSION"
cp "$REPO/LICENCE" "$stage/lib/LICENCE"

mkdir -p "$OUT"
rm -f "$OUT/$NAME.zip" "$OUT/$NAME.zip.sha256" "$OUT/$NAME.dSYM.zip" "$OUT/libmetalgldrv.dylib"
cp "$stage/lib/libmetalgldrv.dylib" "$OUT/libmetalgldrv.dylib"
(cd "$stage/lib" && zip -q -X "$OUT/$NAME.zip" libmetalgldrv.dylib VERSION LICENCE)
(cd "$stage/dsym" && zip -q -r -X "$OUT/$NAME.dSYM.zip" libmetalgldrv.dylib.dSYM)
(cd "$OUT" && shasum -a 256 "$NAME.zip" > "$NAME.zip.sha256")
"$REPO/mods/fabric/build.sh" --dylib "$stage/lib/libmetalgldrv.dylib" --version "${VERSION#v}"
ls -l "$OUT/libmetalgldrv.dylib" "$OUT/$NAME.zip" "$OUT/$NAME.zip.sha256" "$OUT/$NAME.dSYM.zip" "$OUT/metalgldrv-fabric-${VERSION#v}.jar"
