# Using metalgldrv in an application

metalgldrv can replace the system OpenGL in an app you build yourself, for example when porting an open
source program that uses OpenGL. On macOS that's usually a link-line change; on iOS, where there is no
system OpenGL, the app uses EGL.

What you get compared with Apple's OpenGL:

* OpenGL 4.6 instead of 4.1: compute shaders, SSBOs, image load/store, tessellation and geometry
  shaders, multi-draw indirect, DSA, SPIR-V shaders and the rest of core 4.6 (conformance results in
  [tests/cts/reports/LATEST.md](../tests/cts/reports/LATEST.md)).
* A 4.6 **compatibility** profile when the app asks for a legacy context. Apple's legacy profile stops
  at OpenGL 2.1.
* The same window-system APIs: CGL, NSOpenGL (and through them GLFW, SDL and NSOpenGLView), plus EGL 1.5.

Requirements: a Metal 4 GPU (Apple silicon) on macOS / iOS 26 or later. `metalgldrv_is_supported()`
tells you at run time; on unsupported machines keep using the system's OpenGL.

## Getting the library

```sh
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Release -DMGL_UNIVERSAL=ON   # universal: arm64 + x86_64
ninja -C build
cmake --install build --prefix /path/to/sdk
```

The install has:

| Path | Contents |
|---|---|
| `lib/libmetalgldrv.dylib` | The driver (install name `@rpath/libmetalgldrv.dylib`). |
| `include/GL/glcorearb.h`, `GL/glext.h`, `KHR/khrplatform.h` | Khronos OpenGL headers for the full 4.6 API. |
| `include/EGL/*.h` | Khronos EGL headers. |
| `include/metalgldrv/metalgldrv.h` | The driver's own small API (below). |
| `lib/cmake/metalgldrv` | `find_package(metalgldrv)` provides the `metalgldrv::metalgldrv` target. |
| `lib/pkgconfig/metalgldrv.pc` | The same for pkg-config (relocatable). |

`MGL_UNIVERSAL=ON` matters if the app may run as x86_64 under Rosetta. For iOS use
`-DCMAKE_TOOLCHAIN_FILE=cmake/ios.cmake`.

## Route 1, macOS: link it instead of OpenGL.framework

For apps that use CGL, NSOpenGL, NSOpenGLView, GLFW or SDL. You don't change any code.

* CMake: `find_package(metalgldrv REQUIRED)`, then link `metalgldrv::metalgldrv` where the app linked
  `OpenGL::GL` or `-framework OpenGL`.
* pkg-config / Makefiles: `pkg-config --cflags --libs metalgldrv` in place of `-framework OpenGL`.
* Xcode: remove OpenGL.framework from "Link Binary With Libraries", add `libmetalgldrv.dylib`, and embed
  it (below).

How it works: the library exports every GL entry point and CGL under Apple's names, so the app's GL and
CGL calls bind to it. When it sees that the app's main executable links it, it also switches its
NSOpenGL replacement on at load time. `NSOpenGLContext` / `NSOpenGLPixelFormat`, which GLFW, SDL and
NSOpenGLView use, then create metalgldrv contexts. Libraries bundled with the app that still link or
`dlopen` OpenGL.framework (a bundled GLFW, a GL loader) are pointed at the driver too.
`METALGLDRV_HOOK_NSOPENGL=0` turns the automatic part off.

Headers: Apple's `<OpenGL/gl3.h>` only declares up to 4.1. For newer functions use
`<GL/glcorearb.h>` from the install (with `GL_GLEXT_PROTOTYPES` to call them directly), or keep using a
loader such as glad or GLEW: function lookups through OpenGL.framework resolve to the driver.

Contexts:

| The app asks for | It gets |
|---|---|
| `NSOpenGLProfileVersion3_2Core` / `4_1Core`, `kCGLOGLPVersion_3_2_Core` / `GL4_Core` | OpenGL 4.6 core |
| `NSOpenGLProfileVersionLegacy` / no profile | OpenGL 4.6 compatibility |
| GLFW / SDL: a 3.2+ core, forward-compatible context (what they need on macOS) | OpenGL 4.6 core |

Retina behaves as with Apple's OpenGL: views with `wantsBestResolutionOpenGLSurface` render at the
backing scale, other views at 1x.

## Route 2, macOS: load it at run time

For apps that keep linking OpenGL.framework as before and switch to metalgldrv at run time when it's
available, falling back to Apple's OpenGL otherwise. Load and activate it before the app creates its
first window or GL context:

```c
#include <dlfcn.h>
#include "metalgldrv/metalgldrv.h"

void* lib = dlopen("@executable_path/../Frameworks/libmetalgldrv.dylib", RTLD_NOW | RTLD_GLOBAL);
int (*activate)(void) = lib ? (int (*)(void))dlsym(lib, "metalgldrv_activate") : NULL;
if (activate && activate()) {
    /* NSOpenGL, GLFW and SDL now create metalgldrv contexts. */
}
```

Activation redirects the GL and CGL functions the app (and the libraries it has loaded) imported from
OpenGL.framework to the driver, so existing calls like `glDrawArrays` run on metalgldrv without changes;
libraries loaded later are redirected as they load. `metalgldrv_activate()` returns 0, and changes
nothing, when the machine has no Metal 4 GPU, and the app keeps Apple's OpenGL.

Without rebuilding the app at all (for testing): `DYLD_INSERT_LIBRARIES=/path/to/libmetalgldrv.dylib
METALGLDRV_HOOK_NSOPENGL=1 ./app`. This only works for apps without the hardened runtime, which strips
`DYLD_*` variables. Java apps use `-agentpath:` instead (docs/minecraft-macos.md).

## Route 3, macOS and iOS: EGL

EGL is the portable route and the only one on iOS. Standard EGL 1.5:

```c
EGLDisplay dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
eglInitialize(dpy, NULL, NULL);
eglBindAPI(EGL_OPENGL_API);
const EGLint cfgAttrs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT, EGL_SURFACE_TYPE, EGL_WINDOW_BIT,
                           EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE};
EGLConfig cfg; EGLint n;
eglChooseConfig(dpy, cfgAttrs, &cfg, 1, &n);
const EGLint ctxAttrs[] = {EGL_CONTEXT_MAJOR_VERSION, 4, EGL_CONTEXT_MINOR_VERSION, 6,
                           EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT, EGL_NONE};
EGLContext ctx = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, ctxAttrs);
EGLSurface surf = eglCreateWindowSurface(dpy, cfg, (EGLNativeWindowType)nativeWindow, NULL);
eglMakeCurrent(dpy, surf, surf, ctx);
```

`nativeWindow` can be a `CAMetalLayer`, an `NSView` / `UIView`, or a view's backing `CALayer`; the
driver adds a `CAMetalLayer` that follows the view's size and scale. `eglGetProcAddress` returns every
GL function. Pbuffer surfaces work for offscreen rendering. `EGL_GL_COLORSPACE_SRGB` gives an sRGB
default framebuffer.

GLFW can take this route too, with a GLFW built to use EGL for its native context API (see
docs/minecraft-macos.md, route A). On iOS, the Amethyst patches in `demo/amethyst` are a complete
example of an app embedding the driver and using it through EGL.

## Shipping it in an app

* Copy `libmetalgldrv.dylib` into the app bundle (`Contents/Frameworks` on macOS, `Frameworks` on iOS).
  Make sure the app's runpath search paths include that folder (`@executable_path/../Frameworks` on
  macOS, `@executable_path/Frameworks` on iOS; Xcode's "Runpath Search Paths").
* Sign it with the app (Xcode does this for embedded libraries; otherwise `codesign --force --sign …`).
* Ship the universal build if the app has an x86_64 slice.

## Settings and debugging

The driver reads `METALGLDRV_*` environment variables (docs/ARCHITECTURE.md lists all of them). The
useful ones for porting:

* `METALGLDRV_LOG_UNSUPPORTED=1`: logs GL errors, unsupported calls and failing shaders once each,
  with the reason. `METALGLDRV_LOG_FILE=<path>` writes the log to a file.
* `METALGLDRV_STRICT=1` / `METALGLDRV_SPEEDDEMON=1` and the other profiles (README).
* `METALGLDRV_STATS=1`, `METALGLDRV_HITCH_MS=<n>`: performance diagnostics.

On macOS, `~/Library/Application Support/metalgldrv/metalgldrv.conf` holds `METALGLDRV_NAME=value`
lines for apps whose environment is hard to set; `METALGLDRV_CONF=none` ignores it.

Applications can also select behaviour per executable name through application profiles
(`src/gl/app_profiles.cpp`), when a program needs a workaround.

## Limitations

* No IOSurface interop yet (`CGLTexImageIOSurface2D`). The legacy `CGLSetFullScreen*` modes do nothing;
  full-screen windows work normally.
* Apple's GL-only extras such as `GL_APPLE_client_storage` aren't provided.
* One GPU: the default Metal device. Renderer selection through CGL renderer IDs isn't supported.
