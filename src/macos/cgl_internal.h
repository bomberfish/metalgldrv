// Internal interface of the CGL implementation (cgl.mm) for the NSOpenGL layer
// (nsgl.mm) and the activation code (hooks.mm). macOS only.
#pragma once
#define GL_SILENCE_DEPRECATION 1
#import <QuartzCore/CAMetalLayer.h>
#include <OpenGL/OpenGL.h>
#include "gl/context.h"

namespace mgl::cgl {

// Handles created by our CGL (as opposed to Apple's).
bool isOurContext(CGLContextObj ctx);
bool isOurPixelFormat(CGLPixelFormatObj pix);

// A window surface over `layer` laid out per the context's pixel format.
Ref<Surface> newWindowSurface(CGLContextObj ctx, CAMetalLayer* layer);
// Whether a window surface made for another context can serve `ctx` too (contexts
// attached to one view share its surface, like they share an NSOpenGL view surface).
bool surfaceCompatible(CGLContextObj ctx, Surface* s);

// Attaches a window surface (or detaches, with nullptr). Any thread: the context
// switches to it at its next make-current, flush or update on its own thread.
void setWindowDrawable(CGLContextObj ctx, Surface* s);

// kCGLCPSurfaceOpacity of the context (1 = opaque, the default).
int surfaceOpacity(CGLContextObj ctx);

// Our CGL / GLU export with this name (as OpenGL.framework names it), or nullptr.
// Pure table lookup: safe inside dyld callbacks.
void* exportedFunction(const char* name);

}  // namespace mgl::cgl
