#!/bin/bash
# Run glcts against the freshly built metalgldrv.
#
# Usage: tests/cts/run.sh [options] [PATTERN...]
#   PATTERN          fnmatch case filter(s), e.g. 'KHR-GL46.info.*' (default: all of the case list)
#   -l, --caselist F case list file (repeatable). Default: GL 4.6 khronos mustpass
#                    (gl46-main.txt). Shortcuts (gl_cts/data/mustpass/gl/...):
#                      gl30..gl46           khronos_mustpass/main/<x>-main.txt   (KHR-GLxx.*)
#                      gl45-khr-main ...    khronos_mustpass_noctx/main          (KHR-NoContext.*)
#                      gl46-khr-single ...  khronos_mustpass_single/main         (KHR-Single-GLxx.*)
#                      aosp-gl45-es31 ...   aosp_mustpass/main/<x>-main.txt      (dEQP-GL45-ES3x.*)
#                    (GTF-GL* lists need the non-public kc-cts sources and are not built.)
#   -x, --exclude P  exclude fnmatch pattern (repeatable)
#   -o, --out DIR    output dir (default tests/cts/results/<timestamp>)
#   -s, --surface T  pbuffer (default) | window | fbo
#   -c, --config N   --deqp-gl-config-name (default rgba8888d24s8ms0), or an EGL config id if numeric
#   -t, --timeout S  per-case inactivity timeout in seconds (default 60)
#   --width W, --height H  surface size (default 64x64; -1 leaves glcts' default, as the
#                    mustpass FBO configs do)
#   --seed N         --deqp-base-seed (default 1)
#   -j, --jobs N     parallel glcts processes (default: number of P-cores)
#   --no-build       don't rebuild the driver first
#   --log-images     keep images in the QPA log
#   --ios            run on the connected iOS device (glcts app from setup.sh --ios,
#                    reinstalled with the current driver by ios-install.sh)
#   --               remaining args are passed to glcts verbatim
#
# Outputs in DIR: results.qpa (merged log), results.csv, summary.txt,
# Driver env vars (METALGLDRV_LOG=3 to log GL errors, METALGLDRV_DUMP_FAILED_GLSL, ...)
# are passed through to glcts. The driver dylib is snapshotted into DIR first.
#
# crashes.txt (crash/timeout cases with faulting-thread backtraces from the
# macOS .ips crash reports).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
CTS_DIR="${CTS_DIR:-$HOME/metalgldrv-deps/cts}"
CTS_BUILD="${CTS_BUILD:-$CTS_DIR/build}"
DRIVER_BUILD="${DRIVER_BUILD:-$REPO/build}"
MUSTPASS="$CTS_DIR/external/openglcts/data/gl_cts/data/mustpass/gl"
GLCTS_DIR="$CTS_BUILD/external/openglcts/modules"
export PATH="/opt/homebrew/bin:$PATH"

caselists=() filters=() excludes=() extra=()
jobs=$(( $(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || echo 4) ))
out="" surface=pbuffer config=rgba8888d24s8ms0 width=64 height=64 seed=1 timeout=60 build=1 images=disable ios=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        -l|--caselist) caselists+=("$2"); shift 2 ;;
        -x|--exclude)  excludes+=("$2"); shift 2 ;;
        -o|--out)      out="$2"; shift 2 ;;
        -s|--surface)  surface="$2"; shift 2 ;;
        -c|--config)   config="$2"; shift 2 ;;
        -t|--timeout)  timeout="$2"; shift 2 ;;
        --width)       width="$2"; shift 2 ;;
        --height)      height="$2"; shift 2 ;;
        --seed)        seed="$2"; shift 2 ;;
        -j|--jobs)     jobs="$2"; shift 2 ;;
        --no-build)    build=0; shift ;;
        --log-images)  images=enable; shift ;;
        --ios)         ios=1; shift ;;
        -h|--help)     sed -n '2,/^set -e/p' "$0" | sed 's/^# \{0,1\}//;$d'; exit 0 ;;
        --)            shift; extra+=("$@"); break ;;
        *)             filters+=("$1"); shift ;;
    esac
done

[[ ${#caselists[@]} -eq 0 ]] && caselists=(gl46)
size=()
# a numeric -c selects an EGL config by id (as the Khronos runner does for its other-config runs)
if [[ $config =~ ^[0-9]+$ ]]; then cfgarg=(--deqp-gl-config-id="$config"); else cfgarg=(--deqp-gl-config-name="$config"); fi
[[ $width != -1 ]] && size+=(--deqp-surface-width="$width")
[[ $height != -1 ]] && size+=(--deqp-surface-height="$height")
resolved=()
for c in "${caselists[@]}"; do
    if [[ -f "$c" ]]; then resolved+=("$(cd "$(dirname "$c")" && pwd)/$(basename "$c")")
    elif [[ -f "$MUSTPASS/khronos_mustpass/main/$c-main.txt" ]]; then resolved+=("$MUSTPASS/khronos_mustpass/main/$c-main.txt")
    elif [[ -f "$MUSTPASS/khronos_mustpass_noctx/main/$c.txt" ]]; then resolved+=("$MUSTPASS/khronos_mustpass_noctx/main/$c.txt")
    elif [[ -f "$MUSTPASS/khronos_mustpass_single/main/$c.txt" ]]; then resolved+=("$MUSTPASS/khronos_mustpass_single/main/$c.txt")
    elif [[ -f "$MUSTPASS/aosp_mustpass/main/${c#aosp-}-main.txt" ]]; then resolved+=("$MUSTPASS/aosp_mustpass/main/${c#aosp-}-main.txt")
    else echo "case list not found: $c" >&2; exit 2; fi
done

if [[ $ios == 1 ]]; then
    [[ -z "$out" ]] && out="$HERE/results/ios-$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$out"; out="$(cd "$out" && pwd)"
    "$HERE/ios-install.sh"
    (cd "$REPO" && git rev-parse --short HEAD 2>/dev/null; git -C "$REPO" status --short -- src 2>/dev/null | head -50) > "$out/driver-version.txt" || true
    args=(--ios-device "${MGL_DEVICE:-00008142-001538391E80C01C}" --outdir "$out" --timeout "$timeout" --jobs 1)
    for c in "${resolved[@]}"; do args+=(--caselist "$c"); done
    for f in ${filters[@]+"${filters[@]}"}; do args+=(--filter "$f"); done
    for x in ${excludes[@]+"${excludes[@]}"}; do args+=(--exclude "$x"); done
    exec python3 "$HERE/run_cts.py" "${args[@]}" -- \
        --deqp-gl-context-type=egl --deqp-surface-type="$surface" "${cfgarg[@]}" \
        ${size[@]+"${size[@]}"} --deqp-base-seed="$seed" \
        --deqp-screen-rotation=unspecified --deqp-watchdog=disable --deqp-log-images="$images" \
        --deqp-log-shader-sources=disable --deqp-log-flush=enable ${extra[@]+"${extra[@]}"}
fi

if [[ ! -x "$GLCTS_DIR/glcts" ]]; then
    echo "glcts not built; run tests/cts/setup.sh" >&2; exit 2
fi
if [[ $build == 1 ]]; then
    ninja -C "$DRIVER_BUILD" metalgldrv >/dev/null
fi
[[ -f "$DRIVER_BUILD/libmetalgldrv.dylib" ]] || { echo "missing $DRIVER_BUILD/libmetalgldrv.dylib" >&2; exit 2; }

[[ -z "$out" ]] && out="$HERE/results/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$out"; out="$(cd "$out" && pwd)"

# Snapshot the driver so concurrent rebuilds of build/ don't affect a running suite.
cp "$DRIVER_BUILD/libmetalgldrv.dylib" "$out/libmetalgldrv.dylib"
export METALGLDRV_LIBRARY="$out/libmetalgldrv.dylib"
# The per-user metalgldrv.conf (app settings, debugging switches) must not change conformance runs.
export METALGLDRV_CONF="${METALGLDRV_CONF:-none}"
(cd "$REPO" && git rev-parse --short HEAD 2>/dev/null; git -C "$REPO" status --short -- src 2>/dev/null | head -50) > "$out/driver-version.txt" || true

args=(--glcts "$GLCTS_DIR/glcts" --workdir "$GLCTS_DIR" --outdir "$out" --timeout "$timeout" --jobs "$jobs")
for c in "${resolved[@]}"; do args+=(--caselist "$c"); done
for f in ${filters[@]+"${filters[@]}"}; do args+=(--filter "$f"); done
for x in ${excludes[@]+"${excludes[@]}"}; do args+=(--exclude "$x"); done

exec python3 "$HERE/run_cts.py" "${args[@]}" -- \
    --deqp-gl-context-type=egl \
    --deqp-surface-type="$surface" \
    "${cfgarg[@]}" \
    ${size[@]+"${size[@]}"} \
    --deqp-base-seed="$seed" \
    --deqp-screen-rotation=unspecified \
    --deqp-watchdog=disable \
    --deqp-log-images="$images" \
    --deqp-log-flush=enable \
    ${extra[@]+"${extra[@]}"}
