# Windows OpenGL games in CrossOver on metalgldrv

CrossOver's Wine renders Windows OpenGL (`opengl32.dll` → `win32u` → `winemac.so`)
through Apple's OpenGL.framework, which stops at GL 4.1 core. Pointing
`winemac.so` at metalgldrv gives Windows programs GL 4.6 (core and
compatibility). The long-term target is DOOM (2016)'s OpenGL renderer (GL 4.5 core).

## How it works

* `winemac.so` (`CrossOver.app/Contents/SharedSupport/CrossOver/lib/wine/x86_64-unix/`)
  is x86_64 and runs under Rosetta, so the driver must be the universal build
  (its x86_64 slice is what loads).
* It links `/System/Library/Frameworks/OpenGL.framework/Versions/A/OpenGL` and
  imports CGL (`CGLChoosePixelFormat`, `CGLCreateContext`, `CGLSetParameter`,
  `CGLDescribeRenderer`, pbuffers, ...) and `gluCheckExtension`.
  `scripts/crossover-install.sh` rewrites that dependency (`install_name_tool
  -change`) to the installed `libmetalgldrv.dylib`, which exports all of them
  (src/macos/cgl.mm), and re-signs `winemac.so` ad hoc. Wine's loader has
  `disable-library-validation`, so ad hoc signatures load; it strips `DYLD_*`
  variables, which is why the library is linked rather than injected.
* `winemac.so` gets the `gl*` functions with
  `dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL", RTLD_NOLOAD)` +
  `dlsym`. When `winemac.so` loads, libmetalgldrv (its dependency now) rebinds that
  image's `dlopen` import (src/macos/hooks.mm, fishhook-style, chained fixups and
  read-only `__DATA_CONST,__got` handled) so the OpenGL.framework path returns
  libmetalgldrv's own handle; `dlsym` then finds every GL entry point it exports.
* Wine wraps each CGL context in `WineOpenGLContext`, a subclass of
  `NSOpenGLContext` (`-initWithCGLContextObj:`, `-setView:`, `-update`,
  `-flushBuffer`, `-makeCurrentContext`, `-clearDrawable`). libmetalgldrv replaces
  those `NSOpenGLContext` methods (src/macos/nsgl.mm) so AppKit never passes our
  CGL objects to Apple's: attaching a view adds a `CAMetalLayer` on top of the
  Wine window's content view layer, sized to the view and to Wine's
  `kCGLCPSurfaceBackingSize` (the Windows client size).
* Wine's renderer setup (`macdrv_init_pixel_formats`, `init_gl_info`) sees one
  accelerated renderer (RGB888 / ARGB8888, depth 16/24/32, stencil 8, 2x/4x MSAA,
  single + double buffered) and creates legacy, GL3-core and GL4-core probe
  contexts whose `GL_VERSION` reads `4.6.0 ...`. Wine therefore allows
  `wglCreateContextAttribsARB` up to 4.6 and advertises
  `WGL_ARB_create_context_profile`.

## Install

1. Build the universal driver:

       cmake -G Ninja -B build-universal -DMGL_UNIVERSAL=ON -DMGL_BUILD_TESTS=OFF
       ninja -C build-universal metalgldrv
       lipo -info build-universal/libmetalgldrv.dylib   # x86_64 arm64

2. Patch a copy of CrossOver rather than the original (keeps the real one
   untouched and updatable):

       ditto "/Applications/CrossOver Preview.app" ~/Applications/CrossOver-MetalGL.app
       scripts/crossover-install.sh --app ~/Applications/CrossOver-MetalGL.app

   Without `--app` the script picks `~/Applications/CrossOver-MetalGL.app`,
   then `/Applications/CrossOver.app`, then `/Applications/CrossOver Preview.app`.
   `--dry-run` prints the steps without changing anything.

   The script copies the dylib to
   `~/Library/Application Support/metalgldrv/libmetalgldrv.dylib` (`--dest` to
   change), keeps `winemac.so.metalgldrv-backup` next to `winemac.so`, patches and
   re-signs `winemac.so`, and checks the result with `otool -L`. Run it again after
   rebuilding the driver (it re-copies the dylib; the patch is already in place).
   It refuses to run if a backup exists that differs from an unpatched
   `winemac.so` (CrossOver was updated: delete the stale backup and re-run).

   If macOS refuses the write ("cannot write inside ..."): App Management
   (Ventura and later) blocks modifying another developer's app. Give your
   terminal *System Settings > Privacy & Security > App Management*, restart the
   terminal, run again.

3. Bottles are shared between the copy and the original. Launch games from the
   patched copy.

## Uninstall

    scripts/crossover-install.sh --app ~/Applications/CrossOver-MetalGL.app --uninstall

restores `winemac.so` from the backup (the installed dylib stays; delete
`~/Library/Application Support/metalgldrv` if nothing else uses it). Or just
delete the patched copy. A CrossOver update replaces `winemac.so`, which also
undoes the patch; re-run the installer afterwards.

## Verify

Run something from Terminal so the driver's stderr is visible:

    METALGLDRV_LOG=2 ~/Applications/CrossOver-MetalGL.app/Contents/SharedSupport/CrossOver/bin/wine \
        --bottle default 'C:\path\to\game.exe'

(or set `"METALGLDRV_LOG_FILE" = "documents"` under `[EnvironmentVariables]` in
the bottle's `cxbottle.conf` to get `~/Documents/metalgldrv.log`). Expect:

    [metalgldrv I] CrossOver/Wine: .../winemac.so: dlopen rebound (1 slot); OpenGL.framework -> /Users/fish/Library/Application Support/metalgldrv/libmetalgldrv.dylib
    [metalgldrv I] NSOpenGLContext / NSOpenGLPixelFormat now run on metalgldrv
    [metalgldrv I] Metal 4 device: Apple M4 Pro (...)
    [metalgldrv I] CGL: created GL 4.6 compatibility context 0x... (pixel format Legacy, share 0x0)
    [metalgldrv I] CGL: created GL 4.6 core context 0x... (pixel format GL4_Core, share 0x0)
    [metalgldrv I] dlopen("/System/Library/Frameworks/OpenGL.framework/OpenGL") -> .../libmetalgldrv.dylib
    [metalgldrv I] NSOpenGL: view 0x... (WineContentView) surface 1280x720, scale 1.0
    [metalgldrv I] CGL: context 0x... drawable window 1280x720

A GL info tool inside the bottle should report `GL_RENDERER` =
`metalgldrv (Apple M4 Pro, Metal 4)` and `GL_VERSION` = `4.6.0 ...`.
If the first line is missing, `winemac.so` isn't patched (check `otool -L` on it)
or the app is running from the unpatched CrossOver. "OpenGL support is
disabled" / "glXxx not found in libGL" from Wine means the `dlopen` redirect
didn't happen.

## DOOM (2016)

* id Tech 6's GL renderer asks for a **4.3 compatibility** context through
  `wglCreateContextAttribsARB` (see docs/doom2016.md). Wine only allows versions
  up to what its probe contexts report: `init_gl_info` parses `GL_VERSION`,
  which is 4.6 here.
* CrossOver Preview 20260821's winemac rejects compatibility requests, and core
  requests without the forward-compatible bit, only for versions 3.2–4.1. This
  comes from reading the disassembly of `macdrv_context_create`; it hasn't been
  run yet.
  * DOOM's 4.3 request skips that check. winemac passes no profile attribute,
    so our CGL creates a 4.6 compatibility context.
  * For core 3.2–4.1 titles, set `CX_FWD_COMPAT_GL_CTX=1` in the bottle.
  * Compatibility 3.2–4.1 requests remain unsupported under CrossOver.
* Select OpenGL, not Vulkan, in DOOM's advanced graphics settings (Vulkan would
  go to MoltenVK). It is commonly reported that the `+r_renderAPI 0` launch
  option forces OpenGL.
* Everything metalgldrv does on the CPU (state tracking, the GLSL to MSL compiler)
  runs translated by Rosetta in this setup; expect slower shader compilation
  than native. GPU work is native.
* Watch the log for `unimplemented: glXxx` (an entry point DOOM calls that the
  driver lacks) and for shader compile errors (`METALGLDRV_DUMP_FAILED_GLSL=1`).

## Known risks / limitations

* Untested end to end: none of this has been run inside Wine yet. The x86_64
  slice passes the host suites under Rosetta (`arch -x86_64
  build-universal/tests/mgltest`, `nsgltest`, `winemodetest`); `winemodetest`
  loads a fake `winemac.so` linked against the driver and checks the dlopen
  redirect, the `NSOpenGLContext` subclass path and rendering through it.
* Only the CGL calls that `winemac.so` imports are exercised by Wine; older and
  newer Wine versions differ (pbuffers via `CGLCreatePBuffer`/`CGLSetPBuffer`/
  `CGLTexImagePBuffer` in older ones; newer ones emulate pbuffers themselves).
  `CGLTexImagePBuffer` copies (level 0, mipmaps regenerated) instead of binding;
  rendering into pbuffer mip levels > 0 is not supported.
* Surfaces sit in a `CAMetalLayer` above the Wine view's own content, like
  Wine's default `GL_SURFACE_IN_FRONT_OPAQUE` mode. The "behind" mode
  (`kCGLCPSurfaceOrder` -1) is ignored; `kCGLCPSurfaceOpacity` 0 is honored.
* Single-buffered pixel formats behave like double-buffered ones: the image
  appears at `-flushBuffer` / `CGLFlushDrawable`, not at `glFlush`.
* No accumulation, aux, stereo or floating-point pixel formats (Wine doesn't list
  them because the renderer doesn't report them).
* The bundle's code signature seal is broken by the edit (only `winemac.so` is
  re-signed). CrossOver still launches because only executables are checked when
  they load, but `codesign --verify --deep` on the bundle fails; patching a
  copy keeps the original pristine.
* `install_name_tool` needs header room for the longer path; there are ~100 spare
  bytes in the current `winemac.so`. A very long `--dest` can fail (the script
  then aborts without touching `winemac.so`).
