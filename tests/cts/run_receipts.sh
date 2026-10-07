#!/bin/bash
# Full conformance receipt run: every mustpass case list, each archived with save_report.py.
# Usage: tests/cts/run_receipts.sh   (logs in tests/cts/results/receipt-*.log)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/../.."
run() {  # name caselist surface
    rm -rf "tests/cts/results/receipt-$1"
    tests/cts/run.sh --no-build -l "$2" -s "$3" -o "tests/cts/results/receipt-$1" > "tests/cts/results/receipt-$1.log" 2>&1
    python3 tests/cts/save_report.py "tests/cts/results/receipt-$1" "$1" --surface "$3" || echo "save failed for $1"
}
ninja -C build metalgldrv > /dev/null
run KHR-GL46 gl46 pbuffer
run KHR-Single-GL46 gl46-khr-single pbuffer
run KHR-NoContext gl45-khr-main window
cat tests/cts/reports/LATEST.md
