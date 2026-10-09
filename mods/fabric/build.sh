#!/bin/bash
# Builds the metalgldrv Fabric mod: the universal (arm64 + x86_64) libmetalgldrv.dylib,
# then the jar with the dylib inside, written to dist/metalgldrv-fabric-<version>.jar.
#
# Usage: mods/fabric/build.sh [options]
#   --dylib PATH      bundle this universal dylib instead of building one
#   --build-dir DIR   CMake build directory for the dylib (default: build-universal)
#   --version V       mod version (default: from git describe)
#   -h, --help
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
DYLIB="" BUILD_DIR="$REPO/build-universal" VERSION=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dylib) DYLIB="$2"; shift 2 ;;
    --build-dir) BUILD_DIR="$2"; shift 2 ;;
    --version) VERSION="$2"; shift 2 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DYLIB" ]]; then
  # Reconfigure every time so the version baked into the dylib (git describe) is current.
  # Metal 4 needs macOS 26, so that's the oldest macOS the dylib is built for.
  cmake -G Ninja -S "$REPO" -B "$BUILD_DIR" -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0 -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
  ninja -C "$BUILD_DIR" metalgldrv
  DYLIB="$BUILD_DIR/libmetalgldrv.dylib"
fi

echo "dylib: $DYLIB"
lipo -info "$DYLIB"

args=(-p "$HERE" --console=plain -Pdylib="$(cd "$(dirname "$DYLIB")" && pwd)/$(basename "$DYLIB")")
[[ -n "$VERSION" ]] && args+=(-PmodVersion="$VERSION")
gradle "${args[@]}" clean dist

JAR="$(ls -t "$REPO"/dist/metalgldrv-fabric-*.jar | head -1)"
# Check the copy in the jar: both slices, each with a valid (ad hoc) signature.
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
unzip -q -o "$JAR" natives/macos/libmetalgldrv.dylib -d "$TMP"
lipo -info "$TMP/natives/macos/libmetalgldrv.dylib"
for arch in arm64 x86_64; do
  codesign --verify --strict --arch "$arch" "$TMP/natives/macos/libmetalgldrv.dylib"
done
echo "signature: ok (arm64, x86_64)"
echo "built $JAR"
