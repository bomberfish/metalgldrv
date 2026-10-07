#!/bin/bash
# Unattended Minecraft run through Amethyst + metalgldrv on a USB-connected device.
#   demo/amethyst/run_game.sh [profile-name] [seconds]
# Launches Amethyst suspended with AME_AUTOLAUNCH_PROFILE / AME_FORCE_RENDERER,
# attaches lldb (enables JIT like StikDebug, see jit_lldb.py), lets the game run
# for N seconds, then pulls Documents/latestlog.txt into $OUT.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${DEVICE:-00008142-001538391E80C01C}"
BUNDLE="${BUNDLE_ID:-ca.bomberfish.amethyst}"
PROFILE_NAME="${1:-Fabulously Optimized}"
SECONDS_TO_RUN="${2:-120}"
OUT="${OUT:-$HERE/../../build/game-run}"
mkdir -p "$OUT"
ENV_JSON=$(python3 -c "import json,os,sys; e=json.loads(os.environ.get('EXTRA_ENV','{}')); print(json.dumps({**e, 'AME_AUTOLAUNCH_PROFILE': sys.argv[1], 'AME_FORCE_RENDERER': os.environ.get('AME_FORCE_RENDERER','libmetalgldrv.dylib'), 'METALGLDRV_LOG': os.environ.get('METALGLDRV_LOG','2'), 'METALGLDRV_LOG_FILE': 'documents'}))" "$PROFILE_NAME")
echo "== launching $BUNDLE suspended ($PROFILE_NAME)"
xcrun devicectl device process launch --device "$DEVICE" --terminate-existing --start-stopped \
    --environment-variables "$ENV_JSON" --json-output "$OUT/launch.json" "$BUNDLE" >/dev/null || exit 1
PID=$(python3 -c "import json; d=json.load(open('$OUT/launch.json')); print(d['result']['process']['processIdentifier'])")
echo "== pid $PID; attaching lldb (JIT handler)"
PYTHONPATH="$(xcrun lldb -P)" /Applications/Xcode.app/Contents/Developer/usr/bin/python3 -u "$HERE/jit_lldb.py" "$DEVICE" "$PID" > "$OUT/lldb.log" 2>&1 &
LLDB=$!
sleep "$SECONDS_TO_RUN"
echo "== collecting logs"
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source Documents/latestlog.txt --destination "$OUT/latestlog.txt" >/dev/null 2>&1
if [ "${KEEP_RUNNING:-0}" != 1 ]; then
    kill $LLDB 2>/dev/null
    xcrun devicectl device process terminate --device "$DEVICE" --pid "$PID" >/dev/null 2>&1
fi
echo "== logs in $OUT (latestlog.txt, lldb.log)"
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
    --source Documents/latestlog.old.txt --destination "$OUT/latestlog.old.txt" >/dev/null 2>&1
xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" --source Documents/metalgldrv.log --destination "$OUT/metalgldrv.log" >/dev/null 2>&1
# PNGs written by METALGLDRV_CAPTURE_PROG
for f in $(xcrun devicectl device info files --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
             --subdirectory Documents 2>/dev/null | grep -o 'capture_p[0-9]*_[A-Za-z0-9_]*\.png' | sort -u); do
    xcrun devicectl device copy from --device "$DEVICE" --domain-type appDataContainer --domain-identifier "$BUNDLE" \
        --source "Documents/$f" --destination "$OUT/$f" >/dev/null 2>&1
done
grep -E "metalgldrv|EGLBridge|RENDERER|Exception|Error|FATAL|crash" "$OUT/latestlog.txt" | head -60
