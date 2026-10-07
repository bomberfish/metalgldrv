// Driver-internal Metal shaders: presentation, scissored/masked clears, blits.
#pragma once
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>
#include "gl/context.h"

namespace mgl {

// Copies (with vertical flip) the surface's color image into the drawable and presents it.
void presentSurface(Context* ctx, Surface* s, id<CAMetalDrawable> drawable);

// Clears the given attachments of the current pass with a quad, honoring
// scissor + color/depth/stencil write masks. `drawBufferMask` selects color
// draw buffers. Values for int formats come from colorInt.
struct QuadClear {
    uint8_t drawBufferMask = 0;
    float color[kMaxDrawBuffers][4];
    uint32_t colorInt[kMaxDrawBuffers][4];
    uint8_t colorWriteMask[kMaxDrawBuffers];
    bool depth = false, stencil = false;
    float depthValue = 1;
    uint32_t stencilValue = 0, stencilWriteMask = 0xff;
    MTLScissorRect scissor;
    bool scissored = false;
};
void clearWithQuad(Context* ctx, const QuadClear& q);

// Generic textured blit (glBlitFramebuffer / glCopyTexImage paths). Draws src
// rect of `src` (level/slice) into the current pass's dst rect. Coordinates
// are GL-convention (row 0 == Metal row 0).
struct BlitParams {
    id<MTLTexture> src;
    int srcLevel = 0, srcSlice = 0;
    float srcRect[4];  // x0,y0,x1,y1 (may be mirrored)
    float dstRect[4];
    bool linear = false;
    enum What { Color, Depth, Stencil } what = Color;
    MTLScissorRect scissor;
    bool scissored = false;
};
// Blits into an explicit destination texture via its own render pass.
void blitToTexture(Context* ctx, const BlitParams& p, id<MTLTexture> dst, int dstLevel, int dstSlice);

id<MTLLibrary> internalLibrary();

// Expands `texels` tightly packed 12-byte texels at srcAddr into 16-byte texels at dstAddr
// (4th word = alphaBits), on the GPU in command order.
// Copies sample 0 of an integer multisample texture into a single-sample one.
void resolveSample0(Context* ctx, id<MTLTexture> src, id<MTLTexture> dst, bool isSigned);
void expandRgb32(Context* ctx, uint64_t srcAddr, uint64_t dstAddr, uint32_t texels, uint32_t alphaBits);

// Multisample images (METALGLDRV_STRICT, draw.mm): copy every sample of the multisample texture
// `ms` (all layers) into the 2D array `shadow` (slice = layer * samples + sample), and back.
// kind: 0 float, 1 int, 2 uint.
void msImageToShadow(Context* ctx, id<MTLTexture> ms, id<MTLTexture> shadow, int kind);
void msImageFromShadow(Context* ctx, id<MTLTexture> shadow, id<MTLTexture> ms, int kind);

}  // namespace mgl
