// Activation of the macOS system-GL replacement (hooks.mm) and the NSOpenGL
// method replacements (nsgl.mm). Nothing here runs unless activated:
//  - Wine mode: a loaded winemac.so (patched by scripts/crossover-install.sh to link
//    libmetalgldrv instead of OpenGL.framework) gets its dlopen of OpenGL.framework
//    redirected to us, and NSOpenGLContext/NSOpenGLPixelFormat are hooked.
//  - Replacement mode (-agentpath JVMTI agent, or METALGLDRV_HOOK_NSOPENGL=1): the
//    NSOpenGL hooks, plus every non-system image's OpenGL.framework imports, dlopen
//    of OpenGL.framework and CFBundle lookups in the com.apple.opengl bundle resolve
//    to libmetalgldrv.
#pragma once

namespace mgl::macos {

// Replaces NSOpenGLContext / NSOpenGLPixelFormat methods (idempotent). Objects
// that wrap Apple's CGL handles keep Apple's implementation.
void installNSOpenGLHooks();

// Turns on replacement mode (idempotent). `why` is logged.
void activateReplacement(const char* why);

// Path and dlopen handle of libmetalgldrv itself.
const char* ownPath();
void* ownHandle();

}  // namespace mgl::macos
