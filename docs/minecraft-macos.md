# Minecraft on macOS with metalgldrv

Two ways to run Java Edition (LWJGL 3) on metalgldrv in Prism Launcher. Both use
`libmetalgldrv.dylib` as LWJGL's OpenGL library; they differ in how GLFW gets a
context:

| | Route A: EGL | Route B: JVMTI agent |
|---|---|---|
| GLFW | the EGL fork in `~/glfw-kosmickrisp` (`lucamignatti/glfw`) | the stock GLFW that ships with LWJGL |
| How | GLFW's native context API is routed to EGL; it passes the window's `CALayer` to `eglCreateWindowSurface` | `-agentpath` loads metalgldrv first; it replaces `NSOpenGLContext` / `NSOpenGLPixelFormat` and points GLFW's GL lookups and `org.lwjgl.opengl.libname` at itself |
| Hooks | none (plain EGL) | NSOpenGL methods + import rebinding of non-system libraries (src/macos/hooks.mm) |

Java here is GraalVM 24 (`/Library/Java/JavaVirtualMachines/graal-24.jdk`, arm64).
It has the hardened runtime with `disable-library-validation` but not
`allow-dyld-environment-variables`: `DYLD_INSERT_LIBRARIES` / `DYLD_LIBRARY_PATH`
are stripped, so nothing below uses them. Loading our (ad hoc signed) dylib with
`dlopen` or `-agentpath` is fine.

Below, `DRV` is the driver path. The arm64 build works for an arm64 JVM:

    DRV=/Users/fish/metalgldrv/build/libmetalgldrv.dylib

(the universal `build-universal/libmetalgldrv.dylib` works too). Rebuild with
`ninja -C build`; Minecraft picks the new dylib up at its next launch.

## Clean up the old Mesa settings first

`~/Library/Application Support/PrismLauncher/prismlauncher.cfg` still carries the
Mesa/KosmicKrisp setup (paths under `/Users/hariz`, which no longer exists). In
Prism (global *Settings*, or the instance's *Edit > Settings* with the override
boxes ticked):

* **Java > JVM arguments**: remove the three `-Dorg.lwjgl.*.libname=/Users/hariz/...`
  flags (keep your GC flags).
* **Environment variables**: remove `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`,
  `EGL_PLATFORM`, `LIBGL_DRIVERS_PATH`, `MESA_GLSL_VERSION_OVERRIDE`,
  `MESA_GL_VERSION_OVERRIDE`, `MESA_LOADER_DRIVER_OVERRIDE`, `VK_DRIVER_FILES`
  (and `MTL_HUD_ENABLED` unless you want Apple's Metal HUD, which also works here).
* **Workarounds**: leave *Use system installation of GLFW* off (its saved path,
  `/Users/hariz/glfw/...`, is stale); the routes below choose GLFW with JVM flags.

## Route A: EGL with the GLFW fork

1. JVM arguments (append to your existing ones):

       -Dorg.lwjgl.opengl.libname=/Users/fish/metalgldrv/build/libmetalgldrv.dylib
       -Dorg.lwjgl.egl.libname=/Users/fish/metalgldrv/build/libmetalgldrv.dylib
       -Dorg.lwjgl.glfw.libname=/Users/fish/glfw-kosmickrisp/build/src/libglfw.3.dylib

2. Environment variable (the fork looks for `libEGL.dylib` by name otherwise, which
   the hardened JVM won't find):

       MESA_EGL_LIBRARY=/Users/fish/metalgldrv/build/libmetalgldrv.dylib

The fork is arm64 (`lipo -info` says so); rebuild it from `~/glfw-kosmickrisp` if
needed (`cmake -B build -DBUILD_SHARED_LIBS=ON && cmake --build build`).

What the fork does and what metalgldrv's EGL provides (audited against
`src/egl_context.c` / `src/cocoa_window.m` of the fork):

* `eglGetDisplay(EGL_DEFAULT_DISPLAY)` (no platform extension is picked on Cocoa),
  `eglInitialize`, `eglBindAPI(EGL_OPENGL_API)`, config choice through
  `eglGetConfigs` + `eglGetConfigAttrib` (RGB buffers, `EGL_WINDOW_BIT`,
  `EGL_OPENGL_BIT`; Minecraft gets RGBA8 / D24S8).
* `eglCreateContext` with `EGL_CONTEXT_MAJOR/MINOR_VERSION` 3.2,
  `EGL_CONTEXT_OPENGL_PROFILE_MASK` = core and `EGL_CONTEXT_FLAGS_KHR` =
  forward-compatible (+ debug when requested): a GL 4.6 core context.
  `EGL_CONTEXT_RELEASE_BEHAVIOR_KHR` (EGL_KHR_context_flush_control, now
  advertised) is accepted.
* `eglCreateWindowSurface(config, [view layer])` with optional
  `EGL_GL_COLORSPACE_KHR` (sRGB) / `EGL_RENDER_BUFFER`: metalgldrv adds a
  `CAMetalLayer` sublayer to the view's layer. It now autoresizes with the window,
  follows the view's backing scale (Retina, moving between displays) at every
  `eglSwapBuffers`, has no implicit resize animation, and is opaque (alpha the game
  writes doesn't show the window behind).
* `eglSwapInterval` (vsync on/off via `displaySyncEnabled`), `eglSwapBuffers`,
  `eglMakeCurrent`, `eglGetProcAddress` (all GL functions;
  `EGL_KHR_get_all_proc_addresses` is advertised so GLFW loads no separate GL
  library). `eglGetPlatformDisplayEXT` and `eglCreatePlatformWindowSurfaceEXT`
  exist for `EGL_EXT_platform_base`.

Expected log lines (Prism's log window shows the game's stderr):

    [metalgldrv I] Metal 4 device: Apple M4 Pro (...)
    [metalgldrv I] created GL 4.6 core context (requested 3.2)
    [metalgldrv I] window surface 1708x960

## Route B: stock GLFW + JVMTI agent

1. JVM arguments (no `-Dorg.lwjgl.*.libname` flags at all):

       -agentpath:/Users/fish/metalgldrv/build/libmetalgldrv.dylib

2. No environment variables. Keep *Use system installation of GLFW* off so LWJGL
   uses its bundled GLFW.

What happens: the JVM loads the dylib before any Java code runs and calls
`Agent_OnLoad`, which

* replaces the `NSOpenGLPixelFormat` / `NSOpenGLContext` methods GLFW's
  `nsgl_context.m` uses (`initWithAttributes:`, `initWithFormat:shareContext:`,
  `setView:`, `makeCurrentContext`, `flushBuffer`, `update`,
  `setValues:forParameter:` swap interval / opacity, ...) with versions on
  metalgldrv's CGL. `setView:` adds a `CAMetalLayer` over GLFW's content view,
  at the window's backing scale when the view wants a best-resolution surface
  (GLFW's default `GLFW_COCOA_RETINA_FRAMEBUFFER`), resized by `-update`
  (GLFW calls it on resize) and at each flush;
* rebinds, in every non-system library loaded now or later (the JVM's, LWJGL's
  `liblwjgl.dylib` and `libglfw.dylib`), imports from OpenGL.framework, `dlopen`
  of OpenGL.framework, and `CFBundleGetFunctionPointerForName` on
  `com.apple.opengl` (GLFW's `glfwGetProcAddress`, which GLFW itself uses to read
  `GL_VERSION` while creating the window) to metalgldrv;
* sets `org.lwjgl.opengl.libname` to the dylib's own path when the VM is
  initialized (`System.setProperty`, before Minecraft's main runs), unless it is
  already set. `-agentpath:...dylib=nolibname` skips that.

Expected log lines:

    [metalgldrv I] system GL replacement on (JVMTI agent): /Users/fish/metalgldrv/build/libmetalgldrv.dylib
    [metalgldrv I] NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv
    [metalgldrv I] JVMTI agent: org.lwjgl.opengl.libname=/Users/fish/metalgldrv/build/libmetalgldrv.dylib
    [metalgldrv I] rebound N GL/loader imports in /var/folders/.../lwjgl.../libglfw.dylib
    [metalgldrv I] CGL: created GL 4.6 core context 0x... (pixel format GL4_Core, share 0x0)
    [metalgldrv I] NSOpenGL: view 0x... (GLFWContentView) surface 1708x960, scale 2.0

The agent route should also cover LWJGL 2 instances (1.8.9 uses
NSOpenGL + CGL through `liblwjgl.dylib`), but that is untested.

## Troubleshooting

* `METALGLDRV_LOG=3` (environment variable) logs GL errors and more;
  `METALGLDRV_LOG_FILE=documents` writes the log to `~/Documents/metalgldrv.log`.
* `unimplemented: glXxx` warnings name GL entry points a mod uses that the driver
  doesn't implement yet.
* Route B prints `NSOpenGLContext: CGLCreateContext failed` or a black window: check
  that no `-Dorg.lwjgl.opengl.libname` points elsewhere (the agent leaves an
  existing value alone) and that *Use system installation of GLFW* is off.
* Java2D/AWT uses Metal, not OpenGL, on current JDKs; if something forces AWT's
  OpenGL pipeline (`-Dsun.java2d.opengl=true`), route B redirects that too.
