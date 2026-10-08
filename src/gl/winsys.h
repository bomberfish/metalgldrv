// Window-system glue shared by the EGL frontend (src/egl) and the macOS CGL /
// NSOpenGL frontend (src/macos): default-framebuffer surfaces, CAMetalLayer
// setup and presentation.
#pragma once
#import <QuartzCore/CAMetalLayer.h>
#include "gl/context.h"

namespace mgl {

// Buffer layout of a default framebuffer (an EGLConfig or a CGL pixel format).
struct SurfaceFormat {
    int r = 8, g = 8, b = 8, a = 8;
    int depth = 24, stencil = 8, samples = 0;
};

// A default-framebuffer surface without storage or layer: callers attach a layer
// (newWindowSurface) or allocate it with resize() (pbuffers).
Surface* newDefaultSurface(const SurfaceFormat& f);

// A window surface presenting into `layer`, sized from the layer. `srgb` makes the
// default framebuffer sRGB-capable (GL_FRAMEBUFFER_SRGB encodes into it).
Surface* newWindowSurface(const SurfaceFormat& f, CAMetalLayer* layer, bool srgb);

// Resolves an EGLNativeWindowType-style handle (CAMetalLayer, UIView/NSView, CALayer)
// to the CAMetalLayer to present into, creating one if needed. Main thread only.
CAMetalLayer* layerFromNative(void* win);

// A CAMetalLayer added on top of `parent`, sized to it, without implicit animations.
// On macOS it autoresizes with the parent and is tracked by followSuperlayer.
// followScale: the layer also takes the parent's contentsScale (EGL); otherwise the
// owner sets contentsScale itself (NSOpenGL, wantsBestResolutionOpenGLSurface).
CAMetalLayer* newSublayer(CALayer* parent, bool followScale);

// Re-syncs a sublayer made by newSublayer with its superlayer (frame and, if it
// follows it, contentsScale). No-op for layers owned by the app. Any thread.
void followSuperlayer(CAMetalLayer* layer);

// Presents the surface's color buffer into the next drawable of its layer, flushes
// ctx, then picks up layer size / scale changes for the next frame.
void presentWindowSurface(Context* ctx, Surface* s);

// Applies a swap interval (0 = don't wait for vsync) to a surface.
void setSurfaceSwapInterval(Surface* s, int interval);

}  // namespace mgl
