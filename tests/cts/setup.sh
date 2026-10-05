#!/bin/bash
# Fetch, patch, configure and build the Khronos OpenGL CTS (glcts) for macOS
# arm64 against metalgldrv's EGL/GL dylib.
#
# Usage: tests/cts/setup.sh [--clean]
#   CTS_DIR      checkout location            (default ~/metalgldrv-deps/cts)
#   CTS_BUILD    build directory              (default $CTS_DIR/build)
#   DRIVER_BUILD metalgldrv build dir (rpath) (default <repo>/build)
#   CTS_JOBS     ninja -j                     (default: ninja's default)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CTS_DIR="${CTS_DIR:-$HOME/metalgldrv-deps/cts}"
CTS_BUILD="${CTS_BUILD:-$CTS_DIR/build}"
DRIVER_BUILD="${DRIVER_BUILD:-$REPO/build}"

# Pinned: tag opengl-cts-4.6.8.1
CTS_URL=https://github.com/KhronosGroup/VK-GL-CTS
CTS_TAG=opengl-cts-4.6.8.1
CTS_COMMIT=067e8832315e79817ede1c4863804e440f5d1c80
PATCH="$HERE/cts-macos-metalgl.patch"

export PATH="/opt/homebrew/bin:$PATH"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf "$CTS_BUILD"
fi

# 1. Source (shallow, pinned)
if [[ ! -d "$CTS_DIR/.git" ]]; then
    mkdir -p "$(dirname "$CTS_DIR")"
    git clone --depth 1 --branch "$CTS_TAG" "$CTS_URL" "$CTS_DIR"
fi
cd "$CTS_DIR"
if [[ "$(git rev-parse HEAD)" != "$CTS_COMMIT" ]]; then
    git fetch --depth 1 origin "$CTS_COMMIT"
    git checkout -q --force "$CTS_COMMIT"
fi

# 2. Patch (idempotent: reset patched files, then apply)
if git apply --reverse --check "$PATCH" 2>/dev/null; then
    echo "[setup] patch already applied"
else
    git checkout -q -- . && git clean -fdq framework/platform/metalgl targets/metalgl
    git apply "$PATCH"
    echo "[setup] applied $(basename "$PATCH")"
fi

# 3. External deps (GL-relevant only, shallow)
python3 "$HERE/fetch_deps.py" "$CTS_DIR"

# 4. Driver (glcts dlopens it at runtime; build it so the rpath target exists)
if [[ -f "$REPO/CMakeLists.txt" && ! -f "$DRIVER_BUILD/libmetalgldrv.dylib" ]]; then
    cmake -G Ninja -S "$REPO" -B "$DRIVER_BUILD"
    ninja -C "$DRIVER_BUILD" metalgldrv
fi

# 5. Configure + build glcts
cmake -G Ninja -S "$CTS_DIR" -B "$CTS_BUILD" \
    -DDEQP_TARGET=metalgl \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DDEQP_DISABLE_VK_VIDEO_TESTS=ON \
    -DMETALGLDRV_LIB_DIR="$DRIVER_BUILD"
ninja -C "$CTS_BUILD" ${CTS_JOBS:+-j "$CTS_JOBS"} glcts

# (The whole build tree is ~0.5 GB; the CTS checkout + externals ~1.8 GB.)
echo "[setup] glcts: $CTS_BUILD/external/openglcts/modules/glcts"
