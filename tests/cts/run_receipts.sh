#!/bin/bash
# Full conformance receipt run: every mustpass case list, each archived with save_report.py.
# Usage: tests/cts/run_receipts.sh   (logs in tests/cts/results/receipt-*.log)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
wired_gb() { vm_stat | awk '/wired/ {printf "%d", $4*16384/1e9}'; }
run() {  # name caselist surface [run.sh options...]
    local name=$1 list=$2 surface=$3; shift 3
    # A run once left the Mac with ~48 GB wired and the session was killed for low memory: wait for
    # wired memory to settle below 32 GB before each run, and stop the matrix if it doesn't.
    local waited=0
    while [ "$(wired_gb)" -ge 32 ]; do
        [ $waited -ge 600 ] && { echo "wired memory stuck at $(wired_gb) GB; stopping before $name" >&2; exit 1; }
        sleep 30; waited=$((waited + 30))
    done
    rm -rf "tests/cts/results/receipt-$name"
    tests/cts/run.sh --no-build -l "$list" -s "$surface" "$@" -o "tests/cts/results/receipt-$name" \
        > "tests/cts/results/receipt-$name.log" 2>&1
    python3 tests/cts/save_report.py "tests/cts/results/receipt-$name" "$name" --surface "$surface $*" || echo "save failed for $name"
}
ninja -C build metalgldrv > /dev/null
# The GL 4.6 rows of the Khronos runner's first-config table (glcKhronosMustpassGl.hpp):
# 64x64 seed 1, 113x47 seed 2, and FBO rgba8888d24s8 at 64 x default and default x 64, seed 3.
run KHR-GL46 gl46 pbuffer
run KHR-GL46-113x47 gl46 pbuffer --width 113 --height 47 --seed 2
run KHR-GL46-fbo-64xD gl46 fbo -c rgba8888d24s8 --width 64 --height -1 --seed 3
run KHR-GL46-fbo-Dx64 gl46 fbo -c rgba8888d24s8 --width -1 --height 64 --seed 3
# The runner's other-config table: every other EGL config (src/egl/egl.mm kConfigs ids 2-6)
# runs the main list at 64x64 seed 1 and 113x47 seed 2.
for id in 2 3 4 5 6; do
    run KHR-GL46-cfg$id gl46 pbuffer -c $id
    run KHR-GL46-cfg$id-113x47 gl46 pbuffer -c $id --width 113 --height 47 --seed 2
done
run KHR-Single-GL46 gl46-khr-single pbuffer
run KHR-NoContext gl45-khr-main window
cat tests/cts/reports/LATEST.md
