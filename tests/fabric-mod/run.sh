#!/bin/bash
# Standalone checks of the Fabric mod (mods/fabric) and the driver's JNI_OnLoad, with
# no Minecraft: Harness.java runs the mod's preLaunch entrypoint under a stand-in
# FabricLoader (api/), one JVM per scenario, with a temporary config dir and HOME
# (so ~/Library/Caches/metalgldrv is a temporary one too). Nothing creates a window
# or a GL context; the driver's hooks are installed, which is what is being tested.
#
# Usage: tests/fabric-mod/run.sh [--jar JAR] [--java JAVA_HOME] [--agent DYLIB]
#   --jar    the mod jar (default: newest dist/metalgldrv-fabric-*.jar)
#   --java   JDK to run on (default: the Graal 24 JDK Minecraft runs on here)
#   --agent  a libmetalgldrv.dylib for the -agentpath / library= scenarios
#            (default: build-universal/libmetalgldrv.dylib, which also serves an
#            x86_64 JDK under Rosetta)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
JAR="" JDK="/Library/Java/JavaVirtualMachines/graal-24.jdk/Contents/Home" AGENT="$REPO/build-universal/libmetalgldrv.dylib"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --jar) JAR="$2"; shift 2 ;;
    --java) JDK="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
[[ -n "$JAR" ]] || JAR="$(ls -t "$REPO"/dist/metalgldrv-fabric-*.jar | head -1)"

WORK="$(mktemp -d "${TMPDIR:-/tmp}/mgl-fabric-test.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
echo "jar:  $JAR"
echo "java: $("$JDK/bin/java" -version 2>&1 | head -1) ($(file -b "$JDK/bin/java" | grep -oE 'arm64|x86_64' | head -1))"

# Harness + stand-in Fabric API, Java 17 bytecode like the mod.
mkdir -p "$WORK/classes"
"$JDK/bin/javac" --release 17 -d "$WORK/classes" -cp "$JAR" $(find "$HERE/api" -name '*.java') "$HERE/Harness.java"
clang -arch arm64 -arch x86_64 -shared -o "$WORK/libenvprobe.dylib" -I"$JDK/include" -I"$JDK/include/darwin" "$HERE/envprobe.c"
CP="$WORK/classes:$JAR"
# Stand-ins for LWJGL 2 (org.lwjgl.Sys) and LWJGL 3 (org.lwjgl.system.Library in a jar
# whose manifest carries the version), for the scenarios that put them on the class path.
mkdir -p "$WORK/lwjgl2" "$WORK/lwjgl3"
"$JDK/bin/javac" --release 17 -d "$WORK/lwjgl2" "$HERE/lwjgl2/org/lwjgl/Sys.java"
"$JDK/bin/javac" --release 17 -d "$WORK/lwjgl3" "$HERE/lwjgl3/org/lwjgl/system/Library.java"
printf 'Specification-Version: 3.4.3\nImplementation-Version: build 4\n' > "$WORK/lwjgl3.mf"
"$JDK/bin/jar" --create --file "$WORK/lwjgl-3.4.3.jar" --manifest "$WORK/lwjgl3.mf" -C "$WORK/lwjgl3" .

FAILS=0 N=0
# run NAME [env VAR=VALUE...] -- [java args...]: runs the harness with a fresh config
# dir (unless KEEP_CONFIG=1) and the shared HOME; output in $OUT.
HOME_DIR="$WORK/home"
mkdir -p "$HOME_DIR"
HOME_DIR="$(cd "$HOME_DIR" && pwd -P)"  # /var/folders -> /private/var/folders: the path the driver reports
run() {
  local name="$1"; shift
  local envs=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do envs+=("$1"); shift; done
  shift
  N=$((N + 1))
  CONFIG="$WORK/config-$N"
  mkdir -p "$CONFIG"
  [[ -n "${CONFIG_TEXT:-}" ]] && printf '%s\n' "$CONFIG_TEXT" > "$CONFIG/metalgldrv.properties"
  echo "--- $name"
  OUT="$(env HOME="$HOME_DIR" ${envs[@]+"${envs[@]}"} "$JDK/bin/java" -Duser.home="$HOME_DIR" -Dtest.configDir="$CONFIG" \
    -Dtest.envProbe="$WORK/libenvprobe.dylib" "$@" -cp "$CP${EXTRA_CP:+:$EXTRA_CP}" Harness 2>&1)" && CASE_OK=1 || { CASE_OK=0; echo "  FAIL: exit status"; }
}
expect() {  # expect PATTERN (grep -E against the output)
  if ! grep -qE -- "$1" <<<"$OUT"; then echo "  FAIL: expected /$1/"; CASE_OK=0; fi
}
reject() {
  if grep -qE -- "$1" <<<"$OUT"; then echo "  FAIL: unexpected /$1/"; CASE_OK=0; fi
}
done_case() {
  if [[ "$CASE_OK" == 1 ]]; then echo "  ok"; else FAILS=$((FAILS + 1)); echo "$OUT" | sed 's/^/    | /'; fi
}

CACHED="$HOME_DIR/Library/Caches/metalgldrv/[0-9a-f]{16}/libmetalgldrv.dylib"

# 1. First run: default config written, dylib extracted and loaded, JNI_OnLoad activates,
#    libname points at the extracted copy, settings reach the C environment.
CONFIG_TEXT=$'enabled=true\nMETALGLDRV_LOG_UNSUPPORTED=1\nmetalgldrv.stats=0\nMETALGLDRV_PRESET=from-config'
run "fresh install: extract, load, activate" METALGLDRV_PRESET=from-env -- -Dmetalgldrv.foo.bar=baz \
  -Dtest.env=METALGLDRV_LOG_UNSUPPORTED,METALGLDRV_STATS,METALGLDRV_FOO_BAR,METALGLDRV_PRESET,METALGLDRV_ENABLED
expect "extracted the driver to $CACHED"
expect "JNI_OnLoad: METALGLDRV_LOG_UNSUPPORTED=1 \(from Java property metalgldrv.LOG_UNSUPPORTED\)"
expect "system GL replacement on \(JNI_OnLoad\): metalgldrv "
expect "NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv"
expect "JNI_OnLoad: org.lwjgl.opengl.libname=$CACHED"
expect "prop metalgldrv.activation=jni$"
expect "prop org.lwjgl.opengl.libname=$CACHED$"
expect "prop metalgldrv.library=$CACHED$"
expect "prop metalgldrv.version=[^n]"
expect "metalgldrv [^ ]+ active: $CACHED"
expect "env METALGLDRV_LOG_UNSUPPORTED=1$"
expect "env METALGLDRV_STATS=0$"
expect "env METALGLDRV_FOO_BAR=baz$"
expect "env METALGLDRV_PRESET=from-env$"
expect "env METALGLDRV_ENABLED=null$"
done_case
DYLIB_COPY="$(ls "$HOME_DIR"/Library/Caches/metalgldrv/*/libmetalgldrv.dylib)"
for arch in arm64 x86_64; do
  codesign --verify --strict --arch "$arch" "$DYLIB_COPY" || { echo "  FAIL: extracted dylib's $arch signature"; FAILS=$((FAILS + 1)); }
done

# 2. Default config file is created when missing.
CONFIG_TEXT=""
run "no config: default written" --
expect "wrote default config .*/metalgldrv.properties"
expect "using cached $CACHED"
expect "prop metalgldrv.activation=jni$"
done_case
grep -q '^enabled=true' "$CONFIG/metalgldrv.properties" || { echo "  FAIL: default config"; FAILS=$((FAILS + 1)); }

# 3. A damaged cached copy is replaced.
printf 'x' >> "$DYLIB_COPY"
run "damaged cache: re-extracted" --
expect "extracted the driver to $CACHED"
expect "prop metalgldrv.activation=jni$"
done_case

# 4/5. Disabled by config, or by -Dmetalgldrv.enabled=false over the config.
CONFIG_TEXT="enabled=false"
run "enabled=false in config" --
expect "disabled by enabled=false"
expect "prop metalgldrv.activation=null$"
expect "prop org.lwjgl.opengl.libname=null$"
reject "\[metalgldrv I\]"
done_case
CONFIG_TEXT="enabled=true"
run "-Dmetalgldrv.enabled=false" -- -Dmetalgldrv.enabled=false
expect "disabled by -Dmetalgldrv.enabled"
expect "prop metalgldrv.activation=null$"
done_case

# 6-9. Conflicts skip activation; force=true overrides. The libraries must exist: libname
#      arguments pointing at missing files are stale and ignored (case 9b).
OLD_GL="$WORK/old-mesa-libGL.dylib"; OLD_GLFW="$WORK/custom-libglfw.dylib"
touch "$OLD_GL" "$OLD_GLFW"
CONFIG_TEXT=""
run "conflict: -Dorg.lwjgl.opengl.libname" -- -Dorg.lwjgl.opengl.libname="$OLD_GL"
expect "not activating: found -Dorg.lwjgl.opengl.libname=$OLD_GL"
expect "prop metalgldrv.activation=null$"
expect "prop org.lwjgl.opengl.libname=$OLD_GL$"
done_case
run "conflict: -Dorg.lwjgl.glfw.libname" -- -Dorg.lwjgl.glfw.libname="$OLD_GLFW"
expect "not activating: found -Dorg.lwjgl.glfw.libname"
expect "prop metalgldrv.activation=null$"
done_case
run "conflict: VulkanMod" -- -Dtest.mods=fabric-api,vulkanmod
expect "not activating: found mod 'vulkanmod'"
expect "prop metalgldrv.activation=null$"
done_case
CONFIG_TEXT="force=true"
run "force=true over -Dorg.lwjgl.opengl.libname" -- -Dorg.lwjgl.opengl.libname="$OLD_GL"
expect "force=true: replacing org.lwjgl.opengl.libname=$OLD_GL"
expect "prop metalgldrv.activation=jni$"
expect "prop org.lwjgl.opengl.libname=$CACHED$"
done_case

# 9b. Stale libname arguments (files that no longer exist, e.g. an old home folder) are dropped
#     with a warning and don't block activation.
CONFIG_TEXT=""
run "stale libname arguments ignored" -- -Dorg.lwjgl.opengl.libname=/Users/nobody-here/mesa/libGL.dylib \
    -Dorg.lwjgl.egl.libname=/Users/nobody-here/mesa/libEGL.dylib -Dorg.lwjgl.glfw.libname=/Users/nobody-here/glfw/libglfw.3.dylib
expect "ignoring -Dorg.lwjgl.opengl.libname=/Users/nobody-here/mesa/libGL.dylib: the file doesn't exist"
expect "ignoring -Dorg.lwjgl.glfw.libname=/Users/nobody-here/glfw/libglfw.3.dylib: the file doesn't exist"
reject "not activating"
expect "prop metalgldrv.activation=jni$"
expect "prop org.lwjgl.opengl.libname=$CACHED$"
done_case

# 9c. The shader-pack OS define: windows by default once active, configurable, off keeps
#     MC_OS_MAC, and nothing is spoofed when the mod doesn't activate.
CONFIG_TEXT=""
run "shader OS spoof: default" --
expect "prop metalgldrv.activation=jni$"
expect "shaderos=MC_OS_WINDOWS$"
done_case
CONFIG_TEXT="spoof_shader_os=linux"
run "shader OS spoof: linux" --
expect "shaderos=MC_OS_LINUX$"
done_case
CONFIG_TEXT="spoof_shader_os=windows"
run "shader OS spoof: off by -D" -- -Dmetalgldrv.spoof_shader_os=off
expect "prop metalgldrv.activation=jni$"
expect "shaderos=null$"
done_case
CONFIG_TEXT=""
run "shader OS spoof: none when not activated" -- -Dmetalgldrv.enabled=false
expect "shaderos=null$"
done_case

# 9d. Driver defaults: IMPLICIT_PATCHES reaches the driver's environment unless configured.
CONFIG_TEXT=""
run "driver default: IMPLICIT_PATCHES" -- -Dtest.env=METALGLDRV_IMPLICIT_PATCHES
expect "env METALGLDRV_IMPLICIT_PATCHES=1$"
done_case
CONFIG_TEXT="METALGLDRV_IMPLICIT_PATCHES=0"
run "driver default: IMPLICIT_PATCHES off in config" -- -Dtest.env=METALGLDRV_IMPLICIT_PATCHES
expect "env METALGLDRV_IMPLICIT_PATCHES=0$"
done_case

# 10. GLFW already loaded: warned about, still activated.
CONFIG_TEXT=""
run "GLFW class already loaded" -- -Dtest.preloadGlfw=true --add-opens java.base/java.lang=ALL-UNNAMED
expect "org.lwjgl.glfw.GLFW \(.*\) is already loaded"
expect "prop metalgldrv.activation=jni$"
done_case

# 11. -agentpath already active: the mod doesn't load its copy.
run "-agentpath active" -- -agentpath:"$AGENT"
expect "system GL replacement on \(JVMTI agent\)"
expect "metalgldrv is already active \(agent, $AGENT\)"
expect "prop metalgldrv.activation=agent$"
expect "prop org.lwjgl.opengl.libname=$AGENT$"
reject "JNI_OnLoad"
done_case

# 12. A second copy of the driver (another path) in the same process stands down.
CONFIG_TEXT="force=true"  # past the agent's org.lwjgl.opengl.libname
run "second copy (agent active, its report hidden)" -- -agentpath:"$AGENT" -Dtest.hideAgent=true
expect "JNI_OnLoad: metalgldrv at $AGENT is already active in this process; not activating"
expect "prop metalgldrv.activation=skipped: another copy is active: $AGENT$"
expect "prop org.lwjgl.opengl.libname=$AGENT$"
done_case

# 13. No Metal 4 GPU (simulated): the driver declines, nothing is pointed at it.
CONFIG_TEXT=""
run "no Metal 4" -- -Dmetalgldrv.DEBUG_NO_METAL4=1
expect "JNI_OnLoad: METALGLDRV_DEBUG_NO_METAL4 is set; not activating"
expect "prop metalgldrv.activation=skipped: METALGLDRV_DEBUG_NO_METAL4 is set$"
expect "prop org.lwjgl.opengl.libname=null$"
expect "not activated \(skipped"
reject "system GL replacement on"
done_case

# 14. library= loads another driver build instead of the bundled one.
CONFIG_TEXT="library=$AGENT"
run "library= in config" --
expect "using library=$AGENT"
expect "prop metalgldrv.activation=jni$"
expect "prop org.lwjgl.opengl.libname=$AGENT$"
reject "extracted|using cached"
done_case

# 15/16. LWJGL 3's version is logged; LWJGL 2 isn't supported.
EXTRA_CP="$WORK/lwjgl-3.4.3.jar"
run "LWJGL 3 version" --
expect "activating \(LWJGL 3.4.3 \(build 4\)\)"
expect "prop metalgldrv.activation=jni$"
done_case
EXTRA_CP="$WORK/lwjgl2"
run "LWJGL 2" --
expect "not activating: found LWJGL 2"
expect "prop metalgldrv.activation=null$"
done_case
EXTRA_CP=""

# 17. Not macOS: one line, nothing else.
CONFIG_TEXT=""
run "not macOS" -- -Dos.name=Linux
expect "not macOS \(Linux\), nothing to do"
expect "prop metalgldrv.activation=null$"
done_case

echo
if [[ "$FAILS" == 0 ]]; then echo "PASS ($N scenarios)"; else echo "FAIL ($FAILS of $N scenarios)"; exit 1; fi
