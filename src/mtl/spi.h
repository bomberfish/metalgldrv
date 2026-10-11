// Private Metal API ("SPI"): selectors that Apple's own OpenGL driver calls but the public SDK doesn't declare
// (docs/tessellation-performance.md, "Private / SPI Metal APIs"). They are compiled in with the CMake option
// MGL_PRIVATE_API (default ON; turn it off for builds that go through App Store review, which rejects private
// API) and used unless METALGLDRV_MORALLY_WRONG=0 is set. Every call first checks that the object responds to
// the selector, so a Metal without one keeps the public behavior.
#pragma once
#import <Metal/Metal.h>

namespace mgl::spi {

// Compiled in and not disabled by METALGLDRV_MORALLY_WRONG=0 (read once).
bool enabled();

// -[MTLCommandQueueDescriptor setIsOpenGLQueue:YES], as Apple's GL driver sets on its queue.
void markOpenGLQueue(MTLCommandQueueDescriptor* d);

// Wide lines: -[MTL4RenderCommandEncoder setLineWidth:] (and the classic encoder's); public Metal only rasterizes
// 1-pixel lines. lineWidthSupported(): probed once on a Metal 4 render encoder (Device::init).
bool lineWidthSupported();
void probeLineWidth(id encoder);
void setLineWidth(id encoder, float width);
// Widest line GL advertises with setLineWidth: (GL_ALIASED_LINE_WIDTH_RANGE), 1 without it.
float maxLineWidth();


}  // namespace mgl::spi
