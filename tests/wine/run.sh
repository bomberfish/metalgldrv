#!/bin/bash
# Runs tests/wine/wgltest.exe in CrossOver's Wine with the driver injected (DYLD_INSERT_LIBRARIES +
# METALGLDRV_HOOK_NSOPENGL=1, the way NotProton turns it on per game) and checks for a 4.6 core context
# that renders. The runner is cloned (APFS, no copy) with its original winemac.so, so a runner that
# scripts/crossover-install.sh patched is tested unpatched and left as it is. Opens a small window briefly.
#
# Usage: tests/wine/run.sh RUNNER [DYLIB]
#   RUNNER  a CrossOver directory holding lib/wine/x86_64-unix (NotProton's runner, or
#           CrossOver.app/Contents/SharedSupport/CrossOver); its Wine loader must allow DYLD_*
#           variables (NotProton's runners do; CrossOver.app's doesn't)
#   DYLIB   the driver (default: build-universal/libmetalgldrv.dylib)
# Builds the test with Homebrew's LLVM (clang, llvm-dlltool, lld-link); no Windows SDK needed.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$(cd "${1:?usage: tests/wine/run.sh RUNNER [DYLIB]}" && pwd -P)"
DYLIB="${2:-$HERE/../../build-universal/libmetalgldrv.dylib}"
# Absolute: Wine starts its processes from another directory, where a relative DYLD path finds nothing.
DYLIB="$(cd "$(dirname "$DYLIB")" && pwd -P)/$(basename "$DYLIB")"
LLVM="${LLVM_BIN:-/opt/homebrew/opt/llvm/bin}" LLD="${LLD_BIN:-/opt/homebrew/opt/lld/bin}"
W="$(mktemp -d)"
cleanup() { [[ -x "$W/runner/bin/wineserver" ]] && WINEPREFIX="$W/pfx" "$W/runner/bin/wineserver" -k 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT

# Import libraries for the few functions the test calls, and the test itself.
imports=(
    "kernel32:GetModuleHandleA GetStdHandle WriteFile Sleep ExitProcess"
    "user32:RegisterClassA CreateWindowExA GetDC DefWindowProcA PeekMessageA DispatchMessageA wsprintfA"
    "gdi32:ChoosePixelFormat SetPixelFormat SwapBuffers"
    "opengl32:wglCreateContext wglMakeCurrent wglGetProcAddress glGetString glClearColor glClear glViewport glReadPixels glGetError"
)
libs=()
for spec in "${imports[@]}"; do
    name=${spec%%:*}
    { echo "LIBRARY $name.dll"; echo EXPORTS; for f in ${spec#*:}; do echo "  $f"; done; } > "$W/$name.def"
    "$LLVM/llvm-dlltool" -m i386:x86-64 -d "$W/$name.def" -l "$W/$name.lib"
    libs+=("$W/$name.lib")
done
"$LLVM/clang" --target=x86_64-pc-windows-msvc -O1 -ffreestanding -fno-stack-protector -c "$HERE/wgltest.c" -o "$W/wgltest.obj"
"$LLD/lld-link" /entry:start /subsystem:console /nodefaultlib /out:"$W/wgltest.exe" "$W/wgltest.obj" "${libs[@]}"

cp -cR "$RUNNER" "$W/runner"
winemac="$W/runner/lib/wine/x86_64-unix/winemac.so"
[[ -f "$winemac.metalgldrv-backup" ]] && mv -f "$winemac.metalgldrv-backup" "$winemac"
export WINEPREFIX="$W/pfx" WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml=" CX_ROOT="$W/runner" METALGLDRV_CONF=none
export WINESERVER="$W/runner/bin/wineserver" PATH="$W/runner/bin:$PATH"
wine="$W/runner/lib/wine/x86_64-unix/wine"
echo "==> wineboot (new prefix)"
"$wine" wineboot -i >/dev/null 2>&1
echo "==> wgltest.exe, $DYLIB injected"
out="$(DYLD_INSERT_LIBRARIES="$DYLIB" METALGLDRV_HOOK_NSOPENGL=1 "$wine" "$W/wgltest.exe" 2>/dev/null)" || true
echo "$out"
if [[ "$out" == *"core GL_VERSION: 4.6.0 Core Profile metalgldrv"* && "$out" == *"pixel: 255 0 0 255, glGetError 0"* ]]; then
    echo "==> wine injection: pass"
else
    echo "==> wine injection: FAIL"
    exit 1
fi
