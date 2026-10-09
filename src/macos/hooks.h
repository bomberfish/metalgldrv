// Activation of the macOS system-GL replacement (hooks.mm) and the NSOpenGL
// method replacements (nsgl.mm). Nothing here runs unless activated:
//  - Wine mode: a loaded winemac.so (patched by scripts/crossover-install.sh to link
//    libmetalgldrv instead of OpenGL.framework) gets its dlopen of OpenGL.framework
//    redirected to us, and NSOpenGLContext/NSOpenGLPixelFormat are hooked.
//  - Replacement mode (-agentpath JVMTI agent, JNI_OnLoad on System.load, or
//    METALGLDRV_HOOK_NSOPENGL=1): the NSOpenGL hooks, plus every non-system image's
//    OpenGL.framework imports, dlopen of OpenGL.framework and CFBundle lookups in the
//    com.apple.opengl bundle resolve to libmetalgldrv.
#pragma once
#define GL_SILENCE_DEPRECATION 1
#include <CoreGraphics/CGDirectDisplay.h>
#include <OpenGL/OpenGL.h>

namespace mgl::macos {

// Replaces NSOpenGLContext / NSOpenGLPixelFormat methods (idempotent). Objects
// that wrap Apple's CGL handles keep Apple's implementation.
void installNSOpenGLHooks();

// Display of the window whose view one of our NSOpenGLContexts (with this CGL
// context) is attached to; 0 when unknown or off the main thread.
CGDirectDisplayID displayOfContext(CGLContextObj cgl);

// Turns on replacement mode (idempotent). `why` is logged.
void activateReplacement(const char* why);

// Path and dlopen handle of libmetalgldrv itself.
const char* ownPath();
void* ownHandle();

}  // namespace mgl::macos
