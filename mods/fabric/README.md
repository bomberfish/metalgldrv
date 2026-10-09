# metalgldrv for Fabric

A Fabric mod that makes Minecraft (Java Edition, LWJGL 3) on macOS run its OpenGL on
metalgldrv. Drop the jar into `mods/`; no JVM arguments, custom GLFW or launcher
settings needed.

The jar bundles the universal (arm64 + x86_64) `libmetalgldrv.dylib`. A Fabric
`preLaunch` entrypoint runs before Minecraft's `main()`, so before LWJGL opens the
window. It extracts the dylib to `~/Library/Caches/metalgldrv/<sha256 prefix>/` and
loads it with `System.load`. The driver's `JNI_OnLoad` then:

* copies `metalgldrv.<NAME>` Java properties into `METALGLDRV_<NAME>` environment
  variables, skipping variables that are already set;
* replaces `NSOpenGLContext` / `NSOpenGLPixelFormat` with metalgldrv versions and
  points GLFW's / SDL's OpenGL.framework lookups at the driver. See
  `docs/minecraft-macos.md`, route B; the JVMTI agent does the same;
* sets `org.lwjgl.opengl.libname` to the dylib's path.

The mod uses no Minecraft classes, only Fabric Loader's API (fabricloader >= 0.14). The
jar is Java 17 bytecode.

## Requirements and supported versions

* An Apple silicon Mac (Metal 4 GPU) on macOS 26 or later. On anything else the
  driver declines (`not activating (the app keeps Apple's OpenGL)`) and the game runs
  as usual.
* Minecraft with LWJGL 3 on Fabric Loader 0.14 or later:

  | Minecraft | LWJGL | Windowing |
  |---|---|---|
  | 1.20.2 | 3.3.2 | GLFW (NSGL) |
  | 1.21.5, 1.21.11 | 3.3.3 | GLFW (NSGL) |
  | 26.2 | 3.4.1 | GLFW 3.4 (NSGL) |
  | 26.3 | 3.4.3 | SDL 3.4 (Cocoa GL). GLFW 3.5.1 is still bundled |

  These LWJGL versions were checked against their sources, LWJGL 3.3.2 through 3.4.3.
  The checks:
  * `org.lwjgl.opengl.libname` is honoured on macOS: an absolute path is opened with
    `dlopen`.
  * GL functions are looked up in that library. LWJGL 3.4 tries its
    `eglGetProcAddress` first, which metalgldrv exports and which returns the same
    pointers.
  * GLFW's `nsgl_context.m` (3.3 to 3.5.1) and SDL's `SDL_cocoaopengl.m` (3.4) use only
    the NSOpenGL / CFBundle / `dlopen` paths the driver redirects. SDL also uses
    `CVDisplayLinkSetCurrentCGDisplayFromOpenGLContext` for vsync, which the driver
    redirects as well (`tests/nsgl/sdl_test.mm`).
  * Nothing loads GL or creates a window before Fabric's preLaunch.

  Earlier LWJGL 3 releases probably work but weren't checked. Minecraft 26.x needs
  Java 25; 1.20.2 runs on Java 17.

### Not supported

* **LWJGL 2** (Minecraft 1.12 and older, such as a 1.8.9 Legacy Fabric instance). The
  mod detects it and doesn't activate. `force=true` overrides that, untested.
* **NeoForge / Forge.** They open an early loading window before mods load, so a mod
  is too late. That needs other work.
* **Intel Macs and other GPUs without Metal 4.** The driver checks for a Metal 4
  device and steps aside.
* Shader packs and mods that need GL features metalgldrv lacks: look for
  `unimplemented: glXxx` in the log.

## Install

1. Put `metalgldrv-fabric-<version>.jar` into the instance's `mods/` folder (in Prism:
   *Edit > Mods > Add file*, or *Open folder*).
2. Remove the old setups from the instance or the global settings (Prism:
   *Settings*, or *Edit > Settings* with the override boxes ticked):
   * **Java > JVM arguments**: remove `-agentpath:...libmetalgldrv.dylib` and every
     `-Dorg.lwjgl.opengl.libname=`, `-Dorg.lwjgl.glfw.libname=`,
     `-Dorg.lwjgl.egl.libname=` flag. They are left over from the Mesa/KosmicKrisp
     or EGL setups.
   * **Environment variables**: remove `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`,
     `MESA_*`, `LIBGL_DRIVERS_PATH`, `EGL_PLATFORM`, `VK_DRIVER_FILES`. Keep
     `METALGLDRV_*` variables if you want them; they take precedence over the config
     file.
   * **Workarounds**: turn *Use system installation of GLFW* off.
3. Launch. The first run writes `config/metalgldrv.properties`.

The mod refuses to activate while any of the old settings are there (see
*Conflicts*), so a half-cleaned instance keeps working the old way.

## Configuration: `config/metalgldrv.properties`

```properties
enabled=true        # false: mod off, Minecraft uses Apple's OpenGL
force=false         # true: activate despite conflicts (see below)
#library=/path/to/libmetalgldrv.dylib   # load this build instead of the bundled one

# Driver settings: any METALGLDRV_<NAME> variable of the driver
#METALGLDRV_LOG=3
#METALGLDRV_LOG_FILE=documents
#METALGLDRV_LOG_UNSUPPORTED=1
```

* **Turning it off:** set `enabled=false`, or add the JVM argument
  `-Dmetalgldrv.enabled=false`, which overrides the file (`=true` overrides it the
  other way). Removing the jar works too.
* **Driver settings:** `METALGLDRV_<NAME>=value` (or `metalgldrv.<name>=value`)
  becomes the environment variable `METALGLDRV_<NAME>` before the driver starts. The
  JVM argument `-Dmetalgldrv.<name>=value` does the same. An environment variable that
  is already set wins over both.
* **`library=`** loads a local driver build, e.g. `build/libmetalgldrv.dylib`, which
  is handy while working on the driver.

## Log lines to look for

Prism's log window, or `logs/latest.log`. The mod logs as `metalgldrv`; the driver
writes `[metalgldrv I] ...` lines to stderr:

```
[main/INFO] (metalgldrv) metalgldrv mod 0.1.0+abc1234 (driver abc1234): activating (LWJGL 3.3.3 (build 12))
[main/INFO] (metalgldrv) extracted the driver to /Users/you/Library/Caches/metalgldrv/1a2b3c4d5e6f7a8b/libmetalgldrv.dylib
[metalgldrv I] system GL replacement on (JNI_OnLoad): metalgldrv abc1234, /Users/you/Library/Caches/metalgldrv/.../libmetalgldrv.dylib
[metalgldrv I] NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv
[metalgldrv I] JNI_OnLoad: org.lwjgl.opengl.libname=/Users/you/Library/Caches/metalgldrv/.../libmetalgldrv.dylib
[main/INFO] (metalgldrv) metalgldrv abc1234 active: ...
[metalgldrv I] rebound N GL/loader imports in /var/folders/.../libglfw.dylib     (libSDL3.dylib on 26.3)
[metalgldrv I] Metal 4 device: Apple M4 Pro (...)
[metalgldrv I] CGL: created GL 4.6 core context 0x... (pixel format GL4_Core, share 0x0)
[metalgldrv I] NSOpenGL: view 0x... (GLFWContentView) surface 1708x960, scale 2.0
```

Later runs say `using cached ...` instead of `extracted ...`. On Java 24+ the JVM also
prints `WARNING: A restricted method in java.lang.System has been called`. It is
harmless; LWJGL triggers the same warning.

When the mod stays out of the way it says why:

* `disabled by ...`: `enabled=false` or `-Dmetalgldrv.enabled=false`.
* `not activating: found ...`: a conflict, see below.
* `metalgldrv ... not activated (skipped: <reason>)`: the driver declined. Reasons:
  `... does not support Metal 4`, `no Metal device`, or `another copy is active`
  (an `-agentpath` copy at another path).
* `metalgldrv is already active (agent, ...)`: `-agentpath` is still in the JVM
  arguments. The agent does the job and the mod does nothing.
* `activation failed; Minecraft keeps Apple's OpenGL` with a stack trace: an
  unexpected error in the mod. The game continues on Apple's OpenGL.

## Conflicts

The mod doesn't activate, and logs a warning naming the conflict, when it finds:

* `-Dorg.lwjgl.opengl.libname`, `-Dorg.lwjgl.glfw.libname`, `-Dorg.lwjgl.egl.libname`
  or `-Dorg.lwjgl.sdl.libname` already set. These come from the old Mesa setup,
  route A (EGL with a custom GLFW), or Prism's *Use system installation of GLFW*.
* **VulkanMod** (`vulkanmod`), which replaces the OpenGL renderer. Use a separate
  instance without it, e.g. not the "Vulkan Optimized" one.
* **LWJGL 2.**

`force=true` activates anyway. It points `org.lwjgl.opengl.libname` at metalgldrv,
replacing an old value, but leaves custom GLFW/EGL/SDL library names alone. If the
driver then declines, the old value is put back.

The mod also warns if LWJGL's GLFW or SDL class is already loaded. The window might
already exist on Apple's OpenGL; this shouldn't happen with Fabric.

## Building

```sh
mods/fabric/build.sh                      # cmake/ninja build-universal, then the jar
mods/fabric/build.sh --dylib path/to/universal/libmetalgldrv.dylib
mods/fabric/build.sh --version 0.2.0      # default: 0.1.0+<git describe>
```

The jar is written to `dist/metalgldrv-fabric-<version>.jar`. The script:

1. configures `build-universal` with `-DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF
   -DCMAKE_OSX_DEPLOYMENT_TARGET=26.0`;
2. builds the dylib;
3. runs `gradle dist` in this directory (plain Java, `compileOnly` fabric-loader from
   maven.fabricmc.net, no Loom);
4. checks the jar's copy.

Gradle checks that the dylib has both slices. It then signs the bundled copy ad hoc
(`codesign -f -s -`): the linker only signs arm64, and the x86_64 slice should be
signed too. It also records the copy's SHA-256 and the versions in
`metalgldrv-build.properties`. The driver's own version (`git describe` at configure
time) is compiled into the dylib and reported at run time.

Tests without Minecraft are in `tests/fabric-mod/run.sh`. It runs the entrypoint under
a stand-in Fabric Loader in 17 scenarios: extraction and cache reuse, activation,
`libname`, settings reaching the C environment, `enabled=false`, conflicts, `force`,
`-agentpath`, a second copy, no Metal 4, LWJGL 2, and non-macOS. `--java` picks the
JDK; an x86_64 JDK exercises the x86_64 slice under Rosetta.
