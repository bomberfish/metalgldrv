// Draw calls, compute dispatch, memory barriers.
#import "gl/texture.h"
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include "gl/gltypes.h"
#include "gl/internal_shaders.h"
#include "gl/pipeline.h"
#include "mtl/spi.h"
#include "util/log.h"
#include "util/stats.h"

using namespace mgl;
std::string pipelineValidationError(ProgramPipelineObj* pp);  // api_shader.mm
#define API extern "C" __attribute__((visibility("default")))

GLenum framebufferStatus(Context* ctx, Framebuffer* fb);
const char* framebufferIncompleteReason(Context* ctx, Framebuffer* fb);
bool mglCheckDrawConditional(Context* ctx);
ProgramObj* mglCompatPrepareDraw(Context* ctx, ProgramObj* prog);  // compat.mm (fixed-function emulation + gl_* uniforms)
bool mglXfbPrepareDraw(Context* ctx, GLenum mode, uint64_t vertices);  // api_xfb.mm
void mglQueryNotifyDraw(Context* ctx, GLenum mode, uint64_t vertices, uint64_t instances);  // api_query.mm
void mglQueryAddPrimitives(Context* ctx, int stream, uint64_t generated, uint64_t written, bool xfbActive, bool overflow);  // api_query.mm
bool mglQueryPrimitivesActive(Context* ctx);  // api_query.mm
bool mglQueryVisibilityActive(Context* ctx);  // api_query.mm
uint64_t mglPrimitivesFor(GLenum mode, uint64_t vertices);  // api_query.mm

namespace {

struct DrawParams {
    GLenum mode = GL_TRIANGLES;
    GLsizei count = 0;
    GLsizei instances = 1;
    GLint first = 0;
    bool indexed = false;
    GLenum indexType = GL_UNSIGNED_INT;
    const void* indices = nullptr;
    GLint baseVertex = 0;
    GLuint baseInstance = 0;
    int drawID = 0;
    Buffer* indirect = nullptr;
    GLintptr indirectOffset = 0;
};

bool validMode(GLenum m) {
    switch (m) {
    case GL_POINTS: case GL_LINES: case GL_LINE_LOOP: case GL_LINE_STRIP: case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN:
    case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: case GL_TRIANGLES_ADJACENCY: case GL_TRIANGLE_STRIP_ADJACENCY: case GL_PATCHES:
        return true;
    case GL_QUADS: case GL_QUAD_STRIP: case GL_POLYGON:
        return true;  // compat only (checked by caller)
    default: return false;
    }
}

int indexSize(GLenum t) { return t == GL_UNSIGNED_BYTE ? 1 : t == GL_UNSIGNED_SHORT ? 2 : 4; }

MTLPrimitiveTopologyClass topologyClass(GLenum mode) {
    switch (mode) {
    case GL_POINTS: return MTLPrimitiveTopologyClassPoint;
    case GL_LINES: case GL_LINE_LOOP: case GL_LINE_STRIP: case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY:
        return MTLPrimitiveTopologyClassLine;
    default: return MTLPrimitiveTopologyClassTriangle;
    }
}

bool isTriangleMode(GLenum m) {
    return m == GL_TRIANGLES || m == GL_TRIANGLE_STRIP || m == GL_TRIANGLE_FAN || m == GL_QUADS || m == GL_QUAD_STRIP || m == GL_POLYGON ||
           m == GL_TRIANGLES_ADJACENCY || m == GL_TRIANGLE_STRIP_ADJACENCY;
}

// Reads index i from a CPU index array.
inline uint32_t readIndex(const uint8_t* p, GLenum type, size_t i) {
    switch (type) {
    case GL_UNSIGNED_BYTE: return p[i];
    case GL_UNSIGNED_SHORT: { uint16_t v; memcpy(&v, p + 2 * i, 2); return v; }
    default: { uint32_t v; memcpy(&v, p + 4 * i, 4); return v; }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// state application
// ---------------------------------------------------------------------------
// noWrites: depth and stencil writes off (ARB_post_depth_coverage pre-passes). E: a Metal 4 or (tessellation
// segments) classic render encoder; the state setters have the same selectors.
template <class E>
static void applyDynamicState(Context* ctx, E e, GLenum mode, bool viewportArray = false, bool noWrites = false) {
    auto& es = ctx->enc;
    const PassDesc& p = ctx->curPass;
    // Viewport 0, or all of them when the last pre-rasterization stage writes gl_ViewportIndex
    // (y is not flipped: GL row 0 == Metal row 0, see glsl.h)
    int n = viewportArray ? kMaxViewports : 1;
    MTLViewport vps[kMaxViewports];
    MTLScissorRect scs[kMaxViewports];
    for (int i = 0; i < n; i++) {
        MTLViewport& vp = vps[i];
        vp.originX = ctx->viewport[i][0];
        vp.originY = ctx->viewport[i][1];
        vp.width = std::max(0.0f, ctx->viewport[i][2]);
        vp.height = std::max(0.0f, ctx->viewport[i][3]);
        vp.znear = ctx->depthRange[i][0];
        vp.zfar = ctx->depthRange[i][1];
        MTLScissorRect sc = {0, 0, (NSUInteger)p.width, (NSUInteger)p.height};
        if (ctx->scissorEnabled & (1u << i)) {
            int x0 = std::clamp(ctx->scissor[i][0], 0, p.width), y0 = std::clamp(ctx->scissor[i][1], 0, p.height);
            int x1 = std::clamp(ctx->scissor[i][0] + ctx->scissor[i][2], 0, p.width);
            int y1 = std::clamp(ctx->scissor[i][1] + ctx->scissor[i][3], 0, p.height);
            sc = {(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)std::max(0, x1 - x0), (NSUInteger)std::max(0, y1 - y0)};
        }
        scs[i] = sc;
    }
    if (es.numViewports != n || memcmp(es.vp, vps, sizeof(MTLViewport) * n) != 0) {
        if (n == 1) [e setViewport:vps[0]];
        else [e setViewports:vps count:n];
        memcpy(es.vp, vps, sizeof(MTLViewport) * n);
        es.numViewports = n;
    }
    if (es.numScissors != n || memcmp(es.sc, scs, sizeof(MTLScissorRect) * n) != 0) {
        if (n == 1) [e setScissorRect:scs[0]];
        else [e setScissorRects:scs count:n];
        memcpy(es.sc, scs, sizeof(MTLScissorRect) * n);
        es.numScissors = n;
    }
    int cull = MTLCullModeNone;
    static const bool noCull = envFlag("METALGLDRV_DEBUG_NOCULL");  // bisect switch
    if (ctx->cullFace && !noCull) cull = ctx->cullMode == GL_FRONT ? MTLCullModeFront : ctx->cullMode == GL_BACK ? MTLCullModeBack : MTLCullModeNone;
    if (es.cull != cull) {
        [e setCullMode:(MTLCullMode)cull];
        es.cull = cull;
    }
    // y is negated in clip space, which inverts orientation.
    bool flip = ctx->clipOrigin == GL_LOWER_LEFT;
    int winding = (ctx->frontFace == GL_CCW) == flip ? MTLWindingClockwise : MTLWindingCounterClockwise;
    if (es.winding != winding) {
        [e setFrontFacingWinding:(MTLWinding)winding];
        es.winding = winding;
    }
    int fill = ctx->polygonMode == GL_FILL ? MTLTriangleFillModeFill : MTLTriangleFillModeLines;
    if (es.fill != fill) {
        [e setTriangleFillMode:(MTLTriangleFillMode)fill];
        es.fill = fill;
    }
    int clip = ctx->depthClamp ? MTLDepthClipModeClamp : MTLDepthClipModeClip;
    if (es.depthClip != clip) {
        [e setDepthClipMode:(MTLDepthClipMode)clip];
        es.depthClip = clip;
    }
    bool offsetOn = (ctx->polygonMode == GL_FILL && ctx->polygonOffsetFill) || (ctx->polygonMode == GL_LINE && ctx->polygonOffsetLine) ||
                    (ctx->polygonMode == GL_POINT && ctx->polygonOffsetPoint);
    if (!isTriangleMode(mode)) offsetOn = false;
    float bias = offsetOn ? ctx->polygonOffsetUnits : 0, slope = offsetOn ? ctx->polygonOffsetFactor : 0,
          clampv = offsetOn ? ctx->polygonOffsetClamp : 0;
    if (es.depthBias[0] != bias || es.depthBias[1] != slope || es.depthBias[2] != clampv) {
        // Both GL and Metal define the constant bias in units of the format's minimum resolvable difference.
        [e setDepthBias:bias slopeScale:slope clamp:clampv];
        es.depthBias[0] = bias;
        es.depthBias[1] = slope;
        es.depthBias[2] = clampv;
    }
    if (memcmp(es.blendColor, ctx->blendColor, 16) != 0) {
        [e setBlendColorRed:ctx->blendColor[0] green:ctx->blendColor[1] blue:ctx->blendColor[2] alpha:ctx->blendColor[3]];
        memcpy(es.blendColor, ctx->blendColor, 16);
    }
    // glLineWidth: Metal rasterizes 1-pixel lines; the private setLineWidth: widens them (spi.h)
    if (spi::lineWidthSupported()) {
        float w = std::clamp(ctx->lineWidth, 1.0f, spi::maxLineWidth());
        if (es.lineWidth != w) {
            spi::setLineWidth(e, w);
            es.lineWidth = w;
        }
    }
    uint32_t refF = ctx->stencil[0].ref & 0xff, refB = ctx->stencil[1].ref & 0xff;
    if (es.stencilRef[0] != refF || es.stencilRef[1] != refB) {
        [e setStencilFrontReferenceValue:refF backReferenceValue:refB];
        es.stencilRef[0] = refF;
        es.stencilRef[1] = refB;
    }
    // depth/stencil state
    DepthStencilKey dk;
    bool hasDepth = p.depth.tex != nil, hasStencil = p.stencil.tex != nil;
    dk.depthFunc = (ctx->depthTest && hasDepth) ? compareFunc(ctx->depthFunc) : MTLCompareFunctionAlways;
    dk.depthWrite = ctx->depthTest && hasDepth && ctx->depthWrite && !noWrites;
    dk.stencilEnabled = ctx->stencilTest && hasStencil;
    if (dk.stencilEnabled) {
        for (int f = 0; f < 2; f++) {
            const StencilFace& s = ctx->stencil[f];
            dk.face[f].func = compareFunc(s.func);
            dk.face[f].sfail = stencilOp(s.sfail);
            dk.face[f].dpfail = stencilOp(s.dpfail);
            dk.face[f].dppass = stencilOp(s.dppass);
            dk.face[f].readMask = s.valueMask & 0xff;
            dk.face[f].writeMask = noWrites ? 0 : s.writeMask & 0xff;
        }
        // Metal's front/back follow the (flipped) winding we configured, so GL front == Metal front.
    }
    id<MTLDepthStencilState> ds = ctx->pipelines_->depthStencil(dk);
    if (es.depthStencil != ds) {
        [e setDepthStencilState:ds];
        es.depthStencil = ds;
    }
}

// Fills the program's argument slots. Returns GPU address of the slot buffer.
// GL 4.6 §8.26: an image unit is invalid when its level lies outside [base, max], when
// it is the base level of a texture that is not base-complete, or another level of a
// texture that is not mipmap-complete (Mesa's rule, which the CTS follows).
static bool imageUnitValid(const ImageUnit& iu) {
    const Texture* t = iu.tex.get();
    if (t->target == GL_TEXTURE_BUFFER) return t->buffer && t->buffer->mtl;
    if (t->viewParent) return true;
    int maxLevel = t->immutable ? std::min(t->maxLevel, t->immutableLevels - 1) : t->maxLevel;
    if (iu.level < t->baseLevel || iu.level > maxLevel) return false;
    SamplerState ss = t->sampler;
    ss.minFilter = iu.level == t->baseLevel ? GL_NEAREST : GL_NEAREST_MIPMAP_NEAREST;
    int first = 0, last = 0;
    if (!textureComplete(t, ss, &first, &last)) return false;
    return iu.level <= last || t->samples > 0;
}

// One depth slice of a 3D texture as a standalone 2D texture (non-layered image binding).
static id<MTLTexture> imageSliceCopy(Context* ctx, id<MTLTexture> src, MTLPixelFormat pf, NSUInteger level, int layer, bool writable) {
    NSUInteger w = std::max<NSUInteger>(1, src.width >> level), h = std::max<NSUInteger>(1, src.height >> level);
    NSUInteger depth = std::max<NSUInteger>(1, src.depth >> level);
    if (layer < 0 || (NSUInteger)layer >= depth) return nil;
    MTLTextureDescriptor* td = texture2DDescriptor(src.pixelFormat, w, h);
    td.storageMode = MTLStorageModePrivate;
    td.usage = src.usage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    id<MTLTexture> tmp = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (!tmp) return nil;
    ctx->dev->makeResident(tmp);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromTexture:src sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(0, 0, layer) sourceSize:MTLSizeMake(w, h, 1)
             toTexture:tmp destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    uint64_t serial = ctx->serial();
    if (writable) ctx->imageWritebacks.push_back({tmp, src, level, (NSUInteger)layer});
    ctx->dev->releaseAfter(tmp, serial);
    return pf == tmp.pixelFormat ? tmp : [tmp newTextureViewWithPixelFormat:pf];
}

// Multisample images (METALGLDRV_STRICT; otherwise GL_MAX_IMAGE_SAMPLES is 0 and these image
// units are invalid). Metal can't write multisample textures from shaders, so the shader sees a
// per-sample 2D-array shadow (slice = layer * samples + sample, see imageFn() in msl.cpp). Each
// draw/dispatch copies the texture into the shadow first, and writable bindings copy the shadow
// back into every sample afterwards (flushImageWritebacks), so all other uses of the texture
// (sampling, attachments, blits, later image bindings) see the image writes.
static id<MTLTexture> msImageShadow(Context* ctx, Texture* t, const ImageUnit& iu, MTLPixelFormat pf, const FormatInfo* f) {
    id<MTLTexture> ms = t->mtl;
    if (pf != ms.pixelFormat) {
        ms = [ms newTextureViewWithPixelFormat:pf];
        if (!ms) return nil;
        ctx->dev->retainUntil(ms, ctx->serial());
    }
    NSUInteger S = ms.sampleCount, layers = ms.textureType == MTLTextureType2DMultisampleArray ? ms.arrayLength : 1;
    id<MTLTexture> sh = t->msShadow;
    if (!sh || sh.pixelFormat != pf || sh.width != ms.width || sh.height != ms.height || sh.arrayLength != layers * S) {
        MTLTextureDescriptor* d = [MTLTextureDescriptor new];
        d.textureType = MTLTextureType2DArray;
        d.pixelFormat = pf;
        d.width = ms.width;
        d.height = ms.height;
        d.arrayLength = layers * S;
        d.storageMode = MTLStorageModePrivate;
        d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
        if (pf == MTLPixelFormatR32Uint || pf == MTLPixelFormatR32Sint) d.usage |= MTLTextureUsageShaderAtomic;
        if (t->msShadow) ctx->dev->releaseAfter(t->msShadow, ctx->serial());
        t->msShadow = sh = [ctx->dev->mtl newTextureWithDescriptor:d];
        if (!sh) return nil;
        ctx->dev->makeResident(sh);
    }
    FmtKind k = f ? f->kind : t->storageFmt->kind;
    int kind = k == FmtKind::UInt ? 2 : k == FmtKind::SInt ? 1 : 0;
    MGL_STAT("ms-image-shadow");
    msImageToShadow(ctx, ms, sh, kind);
    if (iu.access != GL_READ_ONLY) {
        bool queued = false;
        for (auto& wb : ctx->msImageWritebacks) queued |= wb.shadow == sh;
        if (!queued) ctx->msImageWritebacks.push_back({sh, ms, kind});
    }
    if (layers > 1 && !iu.layered) {
        if ((NSUInteger)iu.layer >= layers) return nil;
        id<MTLTexture> v = [sh newTextureViewWithPixelFormat:pf textureType:MTLTextureType2DArray levels:NSMakeRange(0, 1)
                                                      slices:NSMakeRange(iu.layer * S, S)];
        if (v) ctx->dev->retainUntil(v, ctx->serial());
        return v;
    }
    return sh;
}

// Copies image slices written through imageSliceCopy() back into their 3D textures, and
// multisample image shadows back into their textures.
static void flushImageWritebacks(Context* ctx) {
    for (auto& wb : ctx->msImageWritebacks) msImageFromShadow(ctx, wb.shadow, wb.ms, wb.kind);
    ctx->msImageWritebacks.clear();
    if (ctx->imageWritebacks.empty()) return;
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
    for (auto& wb : ctx->imageWritebacks)
        [e copyFromTexture:wb.tmp sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(wb.tmp.width, wb.tmp.height, 1) toTexture:wb.dst destinationSlice:0 destinationLevel:wb.level
         destinationOrigin:MTLOriginMake(0, 0, wb.z)];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    ctx->imageWritebacks.clear();
}

// Feedback loops (GL 4.6 §9.3.1, ARB_texture_barrier): a texture sampled while it is a color
// attachment of the draw framebuffer. On tile-based GPUs the shader's memory reads race with the
// pass's tile loads/stores (texels near tile edges read stale or partial data), so sampling uses a
// copy taken before the render pass. GL only defines reads of texels not written since the last
// glTextureBarrier, which ends the pass; every pass gets a fresh copy.
static id<MTLTexture> feedbackSnapshot(Context* ctx, Texture* t, id<MTLTexture> view) {
    Framebuffer* fb = ctx->drawFb.get();
    if (!fb || fb->isDefault || !t->mtl || t->target == GL_TEXTURE_BUFFER) return view;
    id<MTLTexture> base = t->mtl.parentTexture ? t->mtl.parentTexture : t->mtl;
    // only attachments the pass writes (enabled draw buffers) at a level the sampler can read
    NSUInteger lo = view.parentTexture ? view.parentRelativeLevel : 0, n = view.mipmapLevelCount;
    bool attached = false;
    for (int j = 0; j < kMaxDrawBuffers && !attached; j++) {
        GLenum db = fb->drawBuffers[j];
        if (db < GL_COLOR_ATTACHMENT0 || db >= GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) continue;
        const Attachment& at = fb->att[db - GL_COLOR_ATTACHMENT0];
        Texture* a = at.tex.get();
        if (at.type != GL_TEXTURE || !a || !a->mtl) continue;
        id<MTLTexture> ab = a->mtl.parentTexture ? a->mtl.parentTexture : a->mtl;
        NSUInteger lvl = (NSUInteger)at.level + (a->mtl.parentTexture ? a->mtl.parentRelativeLevel : 0);
        attached = ab == base && lvl >= lo && lvl < lo + n;
    }
    if (!attached || t->mtl.parentTexture) return view;
    uint64_t pass = ctx->passOpen ? ctx->passCounter : ctx->passCounter + 1;
    if (!t->fbSnapshot || t->fbSnapshotSrc != t->mtl || t->fbSnapshotPass != pass) {
        if (!t->fbSnapshot || t->fbSnapshotSrc != t->mtl) {
            MTLTextureDescriptor* d = [MTLTextureDescriptor new];
            d.textureType = t->mtl.textureType;
            d.pixelFormat = t->mtl.pixelFormat;
            d.width = t->mtl.width;
            d.height = t->mtl.height;
            d.depth = t->mtl.depth;
            d.arrayLength = t->mtl.arrayLength;
            d.mipmapLevelCount = t->mtl.mipmapLevelCount;
            d.sampleCount = t->mtl.sampleCount;
            d.usage = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
            d.storageMode = MTLStorageModePrivate;
            d.hazardTrackingMode = MTLHazardTrackingModeUntracked;
            if (t->fbSnapshot) ctx->dev->releaseAfter(t->fbSnapshot, ctx->serial());
            t->fbSnapshot = [ctx->dev->mtl newTextureWithDescriptor:d];
            t->fbSnapshotSrc = t->mtl;
            for (auto& [k, v] : t->fbSnapshotViews) ctx->dev->retainUntil(v, ctx->serial());
            t->fbSnapshotViews.clear();
            if (!t->fbSnapshot) return view;
            ctx->dev->makeResident(t->fbSnapshot);
        }
        MGL_STAT("feedback-snapshot");
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();  // ends the render pass
        [e copyFromTexture:t->mtl toTexture:t->fbSnapshot];
        ctx->endEncoder();
        t->fbSnapshotPass = ctx->passCounter + 1;  // the pass this draw will open
    }
    if (view == t->mtl) return t->fbSnapshot;
    NSUInteger slices = view.arrayLength * (view.textureType == MTLTextureTypeCube || view.textureType == MTLTextureTypeCubeArray ? 6 : 1);
    Texture::ViewKey key{(uint64_t)view.pixelFormat | (uint64_t)view.textureType << 16 | (uint64_t)view.parentRelativeLevel << 24 |
                             (uint64_t)view.mipmapLevelCount << 32,
                         (uint64_t)view.swizzle.red | (uint64_t)view.swizzle.green << 8 | (uint64_t)view.swizzle.blue << 16 |
                             (uint64_t)view.swizzle.alpha << 24 | (uint64_t)view.parentRelativeSlice << 32};
    auto it = t->fbSnapshotViews.find(key);
    if (it != t->fbSnapshotViews.end()) return it->second;
    id<MTLTexture> v = [t->fbSnapshot newTextureViewWithPixelFormat:view.pixelFormat textureType:view.textureType
                                                              levels:NSMakeRange(view.parentRelativeLevel, view.mipmapLevelCount)
                                                              slices:NSMakeRange(view.parentRelativeSlice, slices) swizzle:view.swizzle];
    if (!v) return view;
    t->fbSnapshotViews[key] = v;
    return v;
}

// VariantKey::msArraySlots / msArray4: multisample sampler slots bound to emulated (2D array)
// multisample storage. Multisample images always use that layout (ImageSamples slot).
static void msArraySlots(Context* ctx, ProgramObj* prog, glsl::VariantKey& vk) {
    if (!ctx->dev->sparseTextures.load(std::memory_order_relaxed) || prog->composite) return;
    const auto& slots = prog->refl.slots;
    for (size_t i = 0; i < slots.size() && i < 64; i++) {
        const glsl::Slot& s = slots[i];
        if (s.textureTarget != GL_TEXTURE_2D_MULTISAMPLE && s.textureTarget != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) continue;
        if (s.kind != glsl::SlotKind::Texture) continue;
        int ti = texTargetIndex(s.textureTarget);
        Texture* t = ti >= 0 ? ctx->textures[std::clamp(prog->samplerUnits[i], 0, kMaxTextureUnits - 1)][ti].get() : nullptr;
        if (t && t->viewParent) t = t->viewParent.get();
        if (!t || !t->msArray) continue;
        uint64_t bits = 1ull << i;
        if (s.element <= i) bits |= 1ull << (i - s.element);  // arrays are declared by their first slot
        vk.msArraySlots |= bits;
        if (t->samples == 4) vk.msArray4 |= bits;
    }
}

// VariantKey::sparseImageSlots: image slots whose unit holds a sparse texture.
// Whether a draw's fragments go through the sample mask / sample coverage operations: GL_SAMPLE_MASK with a mask
// that drops samples, or GL_SAMPLE_COVERAGE, on a multisample draw framebuffer with GL_MULTISAMPLE enabled
// (GL 4.6 14.9.3: they apply only when SAMPLE_BUFFERS is one). Single-sample targets need no mask variant.
static bool sampleMaskActive(Context* ctx) {
    bool wanted = (ctx->sampleMaskEnabled && (ctx->sampleMaskValue & 0xffu) != 0xffu) || ctx->sampleCoverage;
    if (!wanted || !ctx->multisample) return false;
    PassDesc p;
    if (ctx->passOpen && !ctx->passDirty) p = ctx->curPass;
    else if (!ctx->resolvePass(p)) return wanted;
    return p.samples > 1;
}

static uint64_t sparseImageSlots(Context* ctx, ProgramObj* prog) {
    if (!ctx->dev->sparseTextures.load(std::memory_order_relaxed) || prog->composite) return 0;
    uint64_t m = 0;
    const auto& slots = prog->refl.slots;
    for (size_t i = 0; i < slots.size() && i < 64; i++) {
        if (slots[i].kind != glsl::SlotKind::Image) continue;
        Texture* t = ctx->imageUnits[std::clamp(prog->samplerUnits[i], 0, kMaxImageUnits - 1)].tex.get();
        if (t && t->viewParent) t = t->viewParent.get();
        if (!t || !t->sparse) continue;
        m |= 1ull << i;
        if (slots[i].element <= i) m |= 1ull << (i - slots[i].element);  // arrays are checked by their first slot
    }
    return m;
}

// ARB_post_depth_coverage pre-passes (`inert` arguments): a private copy of a buffer range that the
// pre-rasterization stages may write, so that re-running them leaves no side effects.
static uint64_t inertBufferCopy(Context* ctx, Buffer* b, size_t offset, size_t size) {
    size = std::min(size, offset < (size_t)b->size ? (size_t)b->size - offset : 0);
    id<MTLBuffer> c = ctx->dev->newBuffer(std::max<size_t>(size, 64), MTLResourceStorageModePrivate);
    if (!c) return 0;
    ctx->dev->releaseAfter(c, ctx->serial());
    if (size) {
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
        [e copyFromBuffer:b->mtl sourceOffset:offset toBuffer:c destinationOffset:0 size:size];
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    }
    b->lastUse = ctx->serial();
    return c.gpuAddress;
}

// The same for an image view (one level).
static id<MTLTexture> inertTextureCopy(Context* ctx, id<MTLTexture> v) {
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = v.textureType;
    td.pixelFormat = v.pixelFormat;
    td.width = v.width;
    td.height = v.height;
    td.depth = v.depth;
    td.arrayLength = v.arrayLength;
    td.sampleCount = v.sampleCount;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (v.pixelFormat == MTLPixelFormatR32Uint || v.pixelFormat == MTLPixelFormatR32Sint) td.usage |= MTLTextureUsageShaderAtomic;
    id<MTLTexture> c = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (!c) return nil;
    ctx->dev->makeResident(c);
    ctx->dev->releaseAfter(c, ctx->serial());
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromTexture:v toTexture:c];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    return c;
}

// `inert` (ARB_post_depth_coverage pre-passes): storage buffers, atomic counter buffers and writable images that
// the pre-rasterization stages use are bound as copies, so the pre-pass vertex processing reads the same data as the
// real draw but its writes are discarded.
static uint64_t buildArguments(Context* ctx, ProgramObj* prog, bool forCompute, bool inert = false) {
    const glsl::Reflection& r = prog->refl;
    const uint32_t preRaster = r.stageMask & ~(1u << (int)glsl::Stage::Fragment);
    size_t nslots = r.slots.size();
    if (nslots == 0) return 0;
    UploadAlloc args = ctx->upload(nslots * 8, 16);
    uint64_t* slot = (uint64_t*)args.cpu;
    uint64_t serial = ctx->serial();
    uint64_t uniformsAddr = 0;
    for (size_t i = 0; i < nslots; i++) {
        const glsl::Slot& s = r.slots[i];
        uint64_t v = 0;
        switch (s.kind) {
        case glsl::SlotKind::DefaultUniforms: {
            if (!uniformsAddr) {
                size_t sz = std::max<size_t>(prog->uniformData.size(), 16);
                UploadAlloc u = ctx->upload(sz, 16);
                memcpy(u.cpu, prog->uniformData.data(), prog->uniformData.size());
                uniformsAddr = u.gpu;
            }
            v = uniformsAddr;
            break;
        }
        case glsl::SlotKind::Ubo: {
            int binding = s.index < prog->blockBindings.size() ? prog->blockBindings[s.index] : 0;
            const IndexedBufferBinding& b = ctx->uniformBuffers[binding];
            if (b.buffer && b.buffer->mtl) {
                v = b.buffer->gpuAddress() + b.offset;
                b.buffer->lastUse = serial;
            } else {
                // Unbound UBO: point at zeroed memory to avoid faults.
                UploadAlloc z = ctx->upload(r.uniformBlocks[s.index].dataSize ? r.uniformBlocks[s.index].dataSize : 16, 16);
                memset(z.cpu, 0, r.uniformBlocks[s.index].dataSize ? r.uniformBlocks[s.index].dataSize : 16);
                v = z.gpu;
            }
            break;
        }
        case glsl::SlotKind::Ssbo:
        case glsl::SlotKind::BufferSize: {
            int binding = s.index < prog->storageBindings.size() ? prog->storageBindings[s.index] : 0;
            const IndexedBufferBinding& b = ctx->storageBuffers[binding];
            if (s.kind == glsl::SlotKind::BufferSize) {
                v = b.buffer ? (b.size ? b.size : b.buffer->size - b.offset) : 0;
            } else if (inert && b.buffer && b.buffer->mtl && s.index < r.storageBlocks.size() && (r.storageBlocks[s.index].stageMask & preRaster)) {
                v = inertBufferCopy(ctx, b.buffer.get(), b.offset, b.size ? b.size : b.buffer->size - b.offset);
            } else if (b.buffer && b.buffer->mtl) {
                v = b.buffer->gpuAddress() + b.offset;
                b.buffer->lastUse = serial;
                b.buffer->lastGpuWrite = serial;
                MGL_DBG("prog %u ssbo slot %d -> binding %d buffer %u (+%lld, %lld bytes) first word %08x", prog->name, s.index, binding,
                        b.buffer->name, (long long)b.offset, (long long)b.buffer->size,
                        b.buffer->cpu ? *(const uint32_t*)(b.buffer->cpu + b.offset) : 0xdeadbeefu);
            } else {
                MGL_DBG("prog %u ssbo slot %d -> binding %d: nothing bound", prog->name, s.index, binding);
                UploadAlloc z = ctx->upload(64, 16);
                memset(z.cpu, 0, 64);
                v = z.gpu;
            }
            break;
        }
        case glsl::SlotKind::AtomicCounter: {
            int binding = s.index < r.atomicCounterBuffers.size() ? r.atomicCounterBuffers[s.index].binding : 0;
            const IndexedBufferBinding& b = ctx->atomicCounterBuffers[std::min(binding, kMaxAtomicCounterBindings - 1)];
            if (inert && b.buffer && b.buffer->mtl && s.index < r.atomicCounterBuffers.size() && (r.atomicCounterBuffers[s.index].stageMask & preRaster)) {
                v = inertBufferCopy(ctx, b.buffer.get(), b.offset, b.buffer->size - b.offset);
            } else if (b.buffer && b.buffer->mtl) {
                v = b.buffer->gpuAddress() + b.offset;
                b.buffer->lastUse = b.buffer->lastGpuWrite = serial;
            } else {
                UploadAlloc z = ctx->upload(64, 16);
                memset(z.cpu, 0, 64);
                v = z.gpu;
            }
            break;
        }
        case glsl::SlotKind::Texture: {
            int unit = prog->samplerUnits[i];
            int ti = texTargetIndex(s.textureTarget);
            const glsl::Uniform& u = r.uniforms[s.index];
            BaseType bt = BaseType::Float;
            GLenum ut = u.type;
            bool isInt = (ut >= GL_INT_SAMPLER_1D && ut <= GL_INT_SAMPLER_BUFFER) || ut == GL_INT_SAMPLER_2D_RECT ||
                         ut == GL_INT_SAMPLER_CUBE_MAP_ARRAY || ut == GL_INT_SAMPLER_2D_MULTISAMPLE || ut == GL_INT_SAMPLER_2D_MULTISAMPLE_ARRAY;
            bool isUint = (ut >= GL_UNSIGNED_INT_SAMPLER_1D && ut <= GL_UNSIGNED_INT_SAMPLER_BUFFER) || ut == GL_UNSIGNED_INT_SAMPLER_2D_RECT ||
                          ut == GL_UNSIGNED_INT_SAMPLER_CUBE_MAP_ARRAY || ut == GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE ||
                          ut == GL_UNSIGNED_INT_SAMPLER_2D_MULTISAMPLE_ARRAY;
            bool isShadow = ut == GL_SAMPLER_1D_SHADOW || ut == GL_SAMPLER_2D_SHADOW || ut == GL_SAMPLER_1D_ARRAY_SHADOW ||
                            ut == GL_SAMPLER_2D_ARRAY_SHADOW || ut == GL_SAMPLER_CUBE_SHADOW || ut == GL_SAMPLER_2D_RECT_SHADOW ||
                            ut == GL_SAMPLER_CUBE_MAP_ARRAY_SHADOW;
            (void)bt;
            id<MTLTexture> view = nil;
            Texture* t = (unit >= 0 && unit < kMaxTextureUnits && ti >= 0) ? ctx->textures[unit][ti].get() : nullptr;
            if (t) {
                bool complete = false;
                Sampler* so = ctx->samplers[std::clamp(unit, 0, kMaxTextureUnits - 1)].get();
                SamplerState saved = t->sampler;
                if (so) t->sampler = so->state;  // completeness uses the effective sampler
                view = textureSamplingView(ctx, t, s.textureTarget, false, &complete);
                if (so) t->sampler = saved;
                if (view && !forCompute) view = feedbackSnapshot(ctx, t, view);
                if (view) {
                    t->lastUse = serial;
                    // Kind mismatch (e.g. float sampler on int texture) is undefined; keep the view.
                }
            }
            if (!view && t) {
                // A bound but unusable texture (incomplete / wrong target) samples as a dummy.
                thread_local std::set<std::pair<GLuint, GLuint>> warned;
                if (warned.insert({prog->name, t->name}).second)
                    MGL_WARN("program %u sampler '%s' (unit %d): texture %u target %x incomplete or mismatched (immutable levels %d, min filter %x, base %d max %d, %dx%d fmt %x); sampling a dummy",
                             prog->name, u.name.c_str(), unit, t->name, t->target, t->immutableLevels, t->sampler.minFilter, t->baseLevel,
                             t->maxLevel, t->levels[0][std::clamp(t->baseLevel, 0, 15)].width, t->levels[0][std::clamp(t->baseLevel, 0, 15)].height, t->levels[0][std::clamp(t->baseLevel, 0, 15)].internalFormat);
            }
            if (!view) {
                MTLTextureType tt = mtlTextureType(s.textureTarget, 0);
                FmtKind k = isUint ? FmtKind::UInt : isInt ? FmtKind::SInt : isShadow ? FmtKind::Depth : FmtKind::Unorm;
                view = dummyTexture(ctx, tt, k);
            }
            MTLResourceID rid = view.gpuResourceID;
            memcpy(&v, &rid, 8);
            break;
        }
        case glsl::SlotKind::SamplerBorder: {
            // {int4 border color, int4(clamp-to-border axis mask)} for integer textures.
            int unit = std::clamp(prog->samplerUnits[i], 0, kMaxTextureUnits - 1);
            int ti = texTargetIndex(s.textureTarget);
            Texture* t = ti >= 0 ? ctx->textures[unit][ti].get() : nullptr;
            SamplerState st = t ? t->sampler : SamplerState();
            if (Sampler* so = ctx->samplers[unit].get()) st = so->state;
            bool isInt = t && t->storageFmt && (t->storageFmt->kind == FmtKind::UInt || t->storageFmt->kind == FmtKind::SInt);
            int mask = 0;
            if (isInt) {
                mask = (st.wrapS == GL_CLAMP_TO_BORDER ? 1 : 0) | (st.wrapT == GL_CLAMP_TO_BORDER ? 2 : 0) | (st.wrapR == GL_CLAMP_TO_BORDER ? 4 : 0);
                if (t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY) mask &= 1;
            }
            UploadAlloc u = ctx->upload(32, 16);
            int32_t* d = (int32_t*)u.cpu;
            for (int k = 0; k < 4; k++) d[k] = (int32_t)(int64_t)st.borderColor[k];
            d[4] = mask;
            d[5] = d[6] = d[7] = 0;
            v = u.gpu;
            break;
        }
        case glsl::SlotKind::Sampler: {
            int unit = std::clamp(prog->samplerUnits[i], 0, kMaxTextureUnits - 1);
            int ti = texTargetIndex(s.textureTarget);
            Texture* t = ti >= 0 ? ctx->textures[unit][ti].get() : nullptr;
            const glsl::Uniform& u = r.uniforms[s.index];
            TypeShape ts = typeShape(u.type);
            (void)ts;
            SamplerState st = t ? t->sampler : SamplerState();
            if (Sampler* so = ctx->samplers[unit].get()) st = so->state;
            bool isInt = t && t->storageFmt && (t->storageFmt->kind == FmtKind::UInt || t->storageFmt->kind == FmtKind::SInt);
            if (t && t->storageFmt && t->storageFmt->kind == FmtKind::Stencil) isInt = true;
            if (!ctx->textureCubeSeamless) st.seamless = false;
            id<MTLSamplerState> ss = samplerStateFor(st, isInt);
            MTLResourceID rid = ss.gpuResourceID;
            memcpy(&v, &rid, 8);
            break;
        }
        case glsl::SlotKind::LodBias: {
            // GL_TEXTURE_LOD_BIAS of the effective sampler state, added by the shader
            int unit = std::clamp(prog->samplerUnits[i], 0, kMaxTextureUnits - 1);
            int ti = texTargetIndex(s.textureTarget);
            Texture* t = ti >= 0 ? ctx->textures[unit][ti].get() : nullptr;
            float bias = t ? t->sampler.lodBias : 0.0f;
            if (Sampler* so = ctx->samplers[unit].get()) bias = so->state.lodBias;
            bias = std::clamp(bias, -16.0f, 16.0f);  // MAX_TEXTURE_LOD_BIAS
            uint32_t bits;
            memcpy(&bits, &bias, 4);
            v = bits;
            // .y: 1 when a cube map is filtered without TEXTURE_CUBE_MAP_SEAMLESS. Metal always
            // filters across cube faces; the shader keeps such lookups inside the selected face.
            if (s.textureTarget == GL_TEXTURE_CUBE_MAP || s.textureTarget == GL_TEXTURE_CUBE_MAP_ARRAY) {
                SamplerState st = t ? t->sampler : SamplerState();
                if (Sampler* so = ctx->samplers[unit].get()) st = so->state;
                bool seamless = ctx->textureCubeSeamless && st.seamless;
                bool linear = st.magFilter == GL_LINEAR || st.minFilter == GL_LINEAR || st.minFilter == GL_LINEAR_MIPMAP_NEAREST ||
                              st.minFilter == GL_LINEAR_MIPMAP_LINEAR;
                if (!seamless && linear) {
                    float one = 1.0f;
                    uint32_t ob;
                    memcpy(&ob, &one, 4);
                    v |= (uint64_t)ob << 32;
                }
            }
            break;
        }
        case glsl::SlotKind::Image: {
            int unit = std::clamp(prog->samplerUnits[i], 0, kMaxImageUnits - 1);
            ImageUnit& iu = ctx->imageUnits[unit];
            id<MTLTexture> view = nil;
            // inert: a writable image of a pre-rasterization stage binds a copy (none for buffer and multisample images)
            bool inertImage = inert && iu.access != GL_READ_ONLY && s.index < r.uniforms.size() && (r.uniforms[s.index].stageMask & preRaster);
            if (inertImage && iu.tex && (iu.tex->target == GL_TEXTURE_BUFFER || (iu.tex->mtl && iu.tex->mtl.sampleCount > 1))) {
                static bool warned = false;
                if (!warned) MGL_WARN("post_depth_coverage pre-pass: buffer/multisample images of pre-rasterization stages read as zero");
                warned = true;
                v = 0;
                break;
            }
            // Metal has no color views of depth/stencil storage (GL leaves such image access undefined).
            if (iu.tex && imageUnitValid(iu) && ensureTextureStorage(ctx, iu.tex.get()) &&
                !(isDepthOrStencil(iu.tex->storageFmt) && formatInfo(iu.format) && !isDepthOrStencil(formatInfo(iu.format)))) {
                Texture* t = iu.tex.get();
                const FormatInfo* f = formatInfo(iu.format);
                MTLPixelFormat pf = f ? f->mtl : t->mtl.pixelFormat;
                if (t->target == GL_TEXTURE_BUFFER) {
                    view = t->mtl;
                    // image stores go to the buffer's memory: CPU access must wait for them
                    if (t->buffer) t->buffer->lastUse = t->buffer->lastGpuWrite = serial;
                } else if (t->mtl.sampleCount > 1) {
                    if (pf != t->mtl.pixelFormat && !(t->mtl.usage & MTLTextureUsagePixelFormatView)) pf = t->mtl.pixelFormat;
                    if (ctx->strict()) view = msImageShadow(ctx, t, iu, pf, f);
                } else {
                    MTLTextureType vt = t->msArray ? MTLTextureType2DArray : mtlTextureType(s.textureTarget, t->samples);
                    // imageCube / imageCubeArray are declared as texture2d_array in MSL.
                    if (vt == MTLTextureTypeCube || vt == MTLTextureTypeCubeArray) vt = MTLTextureType2DArray;
                    NSUInteger slices = t->mtl.arrayLength * (t->mtl.textureType == MTLTextureTypeCube || t->mtl.textureType == MTLTextureTypeCubeArray ? 6 : 1);
                    NSRange sr = NSMakeRange(0, slices);
                    if (t->msArray) {
                        // one layer = `samples` slices; the shader indexes them (VariantKey::msArraySlots)
                        if (!iu.layered && t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY)
                            sr = NSMakeRange(std::min<NSUInteger>((NSUInteger)iu.layer * t->samples, slices - t->samples), t->samples);
                    } else if (!iu.layered && (t->mtl.textureType == MTLTextureType2DArray || t->mtl.textureType == MTLTextureTypeCube ||
                                               t->mtl.textureType == MTLTextureTypeCubeArray)) {
                        sr = NSMakeRange(std::min<NSUInteger>(iu.layer, slices - 1), 1);
                        vt = MTLTextureType2D;
                    }
                    if (pf != t->mtl.pixelFormat && !(t->mtl.usage & MTLTextureUsagePixelFormatView)) pf = t->mtl.pixelFormat;
                    NSUInteger lvl = std::min(iu.level, t->storageLevels - 1);
                    if (t->mtl.textureType == MTLTextureType3D && !iu.layered && vt == MTLTextureType2D) {
                        view = imageSliceCopy(ctx, t->mtl, pf, lvl, iu.layer, iu.access != GL_READ_ONLY && !inertImage);
                    } else {
                        // read-only images of sparse textures read through the residency-correct twin
                        id<MTLTexture> base = iu.access == GL_READ_ONLY && t->sparseRead ? t->sparseRead : t->mtl;
                        view = [base newTextureViewWithPixelFormat:pf textureType:vt levels:NSMakeRange(lvl, 1) slices:sr];
                        if (view && inertImage) view = inertTextureCopy(ctx, view);
                    }
                    ctx->dev->retainUntil(view, serial);
                }
                t->lastUse = serial;
                if (!inertImage) t->lastGpuWrite = serial;
            }
            // Invalid image units (GL 4.6 §8.26) get a null texture: loads return zero and
            // stores are dropped.
            if (view) {
                MTLResourceID rid = view.gpuResourceID;
                memcpy(&v, &rid, 8);
            } else {
                v = 0;
            }
            break;
        }
        case glsl::SlotKind::ImageSamples: {
            // uint2 {samples, 0}: the multisample image's shadow has samples slices per layer
            const ImageUnit& iu = ctx->imageUnits[std::clamp(prog->samplerUnits[i], 0, kMaxImageUnits - 1)];
            // (sparse multisample textures are stored in that layout and bound directly)
            if (iu.tex && iu.tex->msArray) v = iu.tex->samples;
            else v = ctx->strict() && iu.tex && iu.tex->mtl ? iu.tex->mtl.sampleCount : 0;
            break;
        }
        case glsl::SlotKind::SubroutineIndices: {
            const glsl::StageSubroutines& ss = r.subroutines[std::min<uint32_t>(s.index, (uint32_t)glsl::Stage::Count - 1)];
            int n = std::max(1, ss.numLocations);
            UploadAlloc u = ctx->upload(n * 4, 16);
            const auto& sel = ctx->subroutineSelection[std::min<uint32_t>(s.index, (uint32_t)glsl::Stage::Count - 1)];
            for (int k = 0; k < n; k++) ((uint32_t*)u.cpu)[k] = k < (int)sel.size() ? sel[k] : 0xffffffffu;
            v = u.gpu;
            break;
        }
        case glsl::SlotKind::SystemValues:
        case glsl::SlotKind::XfbBuffer:
            break;
        }
        slot[i] = v;
    }
    return args.gpu;
}

static uint64_t buildSys(Context* ctx, ProgramObj* prog, int drawID, GLint baseVertex, GLuint baseInstance, GLint first,
                         bool indexed = true) {
    UploadAlloc sa = ctx->upload(sizeof(glsl::MGLSys), 16);
    glsl::MGLSys* s = (glsl::MGLSys*)sa.cpu;
    s->depthRangeNear = ctx->depthRange[0][0];
    s->depthRangeFar = ctx->depthRange[0][1];
    s->depthMode = ctx->clipDepthMode == GL_ZERO_TO_ONE ? 1 : 0;
    s->clipDistanceMask = ctx->clipDistanceMask;
    for (int k = 0; k < 4; k++) s->viewport[k] = ctx->viewport[0][k];
    s->pointSize = ctx->pointSize;
    s->flags = (ctx->programPointSize || ctx->compatVertexPointSize ? 1u : 0u) | (ctx->pointSpriteOrigin == GL_LOWER_LEFT ? 2u : 0u) | (ctx->clipOrigin == GL_UPPER_LEFT ? 4u : 0u) |
               (ctx->sampleCoverage && ctx->sampleCoverageInvert ? 8u : 0u) |
               (indexed ? 0u : 16u);  // gl_BaseVertex is 0 for commands without a baseVertex parameter
    s->sampleCoverage = ctx->sampleCoverage ? ctx->sampleCoverageValue : 1.0f;
    s->baseVertex = baseVertex;
    s->baseInstance = baseInstance;
    s->drawID = drawID;
    s->firstVertex = first;
    s->sampleMask = ctx->sampleMaskEnabled ? ctx->sampleMaskValue : ~0u;
    s->alphaRef = ctx->compatAlphaRef;
    s->alphaFunc = ctx->compatAlphaFunc;
    s->minSampleShading = ctx->sampleShading ? ctx->minSampleShading : 0;
    s->renderTargetHeight = (float)ctx->curPass.height;
    memcpy(s->tessDefaultOuter, ctx->patchDefaultOuter, sizeof s->tessDefaultOuter);
    memcpy(s->tessDefaultInner, ctx->patchDefaultInner, sizeof s->tessDefaultInner);
    memcpy(s->currentAttrib, ctx->currentAttrib, sizeof(float) * 4 * kMaxVertexAttribs);
    memcpy(s->currentAttribD, ctx->currentAttribD, sizeof(double) * 4 * kMaxVertexAttribs);
    return sa.gpu;
}

namespace mgl { void debugCaptureBeforeDraw(Context* ctx, ProgramObj* prog); }  // debug_capture.mm

// Buffer-backed vertex layouts Metal's fetch can't express (attribute offsets or strides
// not 4-byte aligned) are pulled by the vertex shader itself (VariantKey::manualFetch).
// Client arrays still go through the CPU repack path.
static bool wantManualFetch(Context* ctx) {
    VertexArray* vao = ctx->vao.get();
    if (!vao) return false;
    bool unaligned = false;
    for (int l = 0; l < kMaxVertexAttribs; l++) {
        const VertexAttrib& va = vao->attribs[l];
        if (!va.enabled) continue;
        if (va.clientPointer) return false;
        const VertexBinding& vb = vao->bindings[va.binding];
        if ((vb.stride & 3) || (vb.offset & 3) || (va.relativeOffset & 3)) unaligned = true;
    }
    return unaligned;
}

// stage_in carries the upper half of a dvec3/dvec4 in Metal attribute 16 + L, which
// doesn't exist for L = 15 (Metal has 31 attributes): such programs fetch in the shader.
static bool wantManualFetch(Context* ctx, ProgramObj* prog) {
    if (wantManualFetch(ctx)) return true;
    VertexArray* vao = ctx->vao.get();
    if (!vao) return false;
    bool wideAt15 = false;
    for (const auto& a : prog->refl.attributes) {
        TypeShape ts = typeShape(a.type);
        if (ts.base == BaseType::Double && ts.rows > 2 && a.location >= 0 && a.location + std::max(1, a.locationsUsed) > 15) wideAt15 = true;
    }
    if (!wideAt15) return false;
    for (int l = 0; l < kMaxVertexAttribs; l++)
        if (vao->attribs[l].enabled && vao->attribs[l].clientPointer) return false;
    return true;
}

// Builds the render pipeline key & binds vertex buffers. Returns false if the draw must be skipped.
// Vertex buffers of the bound VAO for shaders that fetch attributes themselves (MGLDrawInfo::vertexBuffer:
// the binding's address, its size for the shader's range check, stride and divisor).
static void fillVertexBindings(Context* ctx, glsl::MGLDrawInfo* info) {
    VertexArray* vao = ctx->vao.get();
    uint64_t serial = ctx->serial();
    for (int b = 0; b < kMaxVertexBindings; b++) {
        const VertexBinding& vb = vao->bindings[b];
        if (vb.buffer && vb.buffer->mtl && vb.offset >= 0 && vb.offset < vb.buffer->size) {
            info->vertexBuffer[b] = vb.buffer->gpuAddress() + vb.offset;
            info->vertexBufferSize[b] = (uint32_t)std::min<GLsizeiptr>(vb.buffer->size - vb.offset, UINT32_MAX);
            vb.buffer->lastUse = serial;
        }
        info->vertexStride[b] = vb.stride;
        info->vertexDivisor[b] = vb.divisor;
    }
}

// Pointers a draw has nothing for point at the device's zero buffer instead of 0, so that no shader can
// follow a null address (MGLDrawInfo::stats stays null: shaders test it). One sub-draw if there is no table.
static void failSafeDrawInfo(Context* ctx, glsl::MGLDrawInfo* info) {
    uint64_t zero = ctx->dev->zeroBufferAddress();
    for (uint64_t* p : {&info->indexBuffer, &info->tcsVertexOut, &info->tcsPatchOut, &info->counters, &info->itemCounts, &info->itemOffsets,
                        &info->tesVertexOut, &info->tesPrimOut, &info->gsRecord, &info->spillBuffer, &info->spillCounter, &info->subDraws})
        if (!*p) *p = zero;
    if (!info->subDrawCount) info->subDrawCount = 1;  // the zero buffer: one sub-draw from patch 0, list from indexBuffer
}

static bool setupVertexInput(Context* ctx, ProgramObj* prog, RenderPipelineKey& key, id<MTL4ArgumentTable> table, const DrawParams& d,
                             uint32_t minIndex, uint32_t maxIndex) {
    VertexArray* vao = ctx->vao.get();
    uint64_t serial = ctx->serial();
    if (key.variant && key.variant->manualFetch) {
        // Attributes are fetched in the shader: bind the vertex buffers through MGLDrawInfo.
        UploadAlloc di = ctx->upload(sizeof(glsl::MGLDrawInfo), 16);
        glsl::MGLDrawInfo* info = (glsl::MGLDrawInfo*)di.cpu;
        memset(info, 0, sizeof *info);
        fillVertexBindings(ctx, info);
        info->baseInstance = d.baseInstance;
        failSafeDrawInfo(ctx, info);
        [table setAddress:di.gpu atIndex:glsl::kBufStageIO];
        return true;
    }
    bool needCurrent = false;
    uint32_t bindingUsed = 0;
    uint32_t wideMask = 0;  // locations holding a dvec3/dvec4
    for (const auto& a : prog->refl.attributes) {
        TypeShape ts = typeShape(a.type);
        if (a.location < 0 || ts.base != BaseType::Double || ts.rows <= 2) continue;
        for (int k = 0; k < std::max(1, a.locationsUsed) && a.location + k < kMaxVertexAttribs; k++) wideMask |= 1u << (a.location + k);
    }
    for (const auto& a : prog->refl.attributes) {
        if (a.location < 0) continue;
        TypeShape ts = typeShape(a.type);
        int nloc = std::max(1, a.locationsUsed) ;
        for (int k = 0; k < nloc; k++) {
            int loc = a.location + k;
            if (loc >= kMaxVertexAttribs) break;
            VertexAttribKey& ak = key.attribs[loc];
            // dvec3/dvec4 (one location each) arrive in two Metal attributes: L and secondHalfAttribute(L)
            bool wideDouble = ts.base == BaseType::Double && ts.rows > 2;
            VertexAttribKey& ak2 = key.attribs[glsl::secondHalfAttribute(loc, wideMask)];
            const VertexAttrib* va = vao ? &vao->attribs[loc] : nullptr;
            bool enabled = va && va->enabled;
            if (enabled) {
                const VertexBinding& vb = vao->bindings[va->binding];
                bool hasBuffer = vb.buffer && vb.buffer->mtl;
                if (!hasBuffer && !va->clientPointer) enabled = false;
            }
            if (enabled) {
                MTLVertexFormat f = vertexFormatFor(va->type, va->size, va->normalized, va->integer || ts.base == BaseType::Int || ts.base == BaseType::UInt);
                if (ts.base == BaseType::Float && !va->normalized && !va->integer && va->type != GL_FLOAT && va->type != GL_HALF_FLOAT &&
                    va->type != GL_UNSIGNED_INT_10F_11F_11F_REV) {
                    // Unnormalized integer data feeding a float input: Metal converts integer formats to float.
                    f = vertexFormatFor(va->type, va->size, false, true);
                }
                if (va->type == GL_DOUBLE && va->isLong) {
                    // dvecN arrive as raw bits: double/dvec2 -> uint2/uint4 at L; dvec3/4 -> uint4 at L
                    // plus uint2/uint4 at the second-half attribute (same binding, +16 bytes)
                    int dwords = va->size * 2;
                    f = std::min(dwords, 4) == 2 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4;
                    if (wideDouble) {
                        int rest = std::max(2, dwords - 4);
                        ak2.format = (uint8_t)(rest == 2 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4);
                        ak2.bufferIndex = glsl::kBufVertex0 + va->binding;
                        ak2.offset = va->relativeOffset + 16;
                    }
                }
                if (va->type == GL_FIXED || (va->type == GL_DOUBLE && !va->isLong)) {
                    static bool warned = false;
                    if (!warned) MGL_WARN("GL_FIXED / non-L GL_DOUBLE vertex attributes not supported yet");
                    warned = true;
                    enabled = false;
                } else if (f == MTLVertexFormatInvalid) {
                    enabled = false;
                } else {
                    ak.format = (uint8_t)f;
                    ak.bufferIndex = glsl::kBufVertex0 + va->binding;
                    ak.offset = va->relativeOffset;
                    bindingUsed |= 1u << va->binding;
                }
            }
            if (!enabled) {
                needCurrent = true;
                MTLVertexFormat f = ts.base == BaseType::Int ? MTLVertexFormatInt4 : ts.base == BaseType::UInt ? MTLVertexFormatUInt4 : MTLVertexFormatFloat4;
                ak.format = (uint8_t)f;
                ak.bufferIndex = kBufCurrentAttribs;
                ak.offset = 16 * loc;
                if (ts.base == BaseType::Double) {
                    // double current values: 32 bytes per location after the float block
                    ak.format = (uint8_t)(ts.rows == 1 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4);
                    ak.offset = 16 * kMaxVertexAttribs + 32 * loc;
                    if (wideDouble) {
                        ak2.format = (uint8_t)(ts.rows == 3 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4);
                        ak2.bufferIndex = kBufCurrentAttribs;
                        ak2.offset = ak.offset + 16;
                    }
                }
            }
        }
    }
    for (int b = 0; b < kMaxVertexBindings; b++) {
        if (!(bindingUsed & (1u << b))) continue;
        const VertexBinding& vb = vao->bindings[b];
        VertexBufferKey& bk = key.buffers[glsl::kBufVertex0 + b];
        bk.used = 1;
        bk.stepFunction = vb.divisor ? MTLVertexStepFunctionPerInstance : MTLVertexStepFunctionPerVertex;
        bk.stepRate = vb.divisor ? vb.divisor : 1;
        GLsizei stride = vb.stride;
        uint64_t addr = 0;
        // client-side arrays (compatibility profile)
        const VertexAttrib* clientAttr = nullptr;
        for (int l = 0; l < kMaxVertexAttribs; l++)
            if (vao->attribs[l].enabled && vao->attribs[l].binding == (GLuint)b && vao->attribs[l].clientPointer) clientAttr = &vao->attribs[l];
        if (clientAttr && !(vb.buffer && vb.buffer->mtl)) {
            uint32_t lo = vb.divisor ? 0 : minIndex;
            uint32_t hi = vb.divisor ? (uint32_t)((d.instances + vb.divisor - 1) / vb.divisor + d.baseInstance) : maxIndex + 1;
            size_t bytes = (size_t)(hi - lo) * stride + 64;
            UploadAlloc u = ctx->upload(bytes, 16);
            memcpy(u.cpu, (const uint8_t*)clientAttr->clientPointer + (size_t)lo * stride, (size_t)(hi - lo) * stride);
            addr = u.gpu - (uint64_t)lo * stride;  // vertex index lo maps to the start of the upload
            // relative offsets were computed against pointer==binding offset, which is 0 for client arrays
        } else if (vb.buffer && vb.buffer->mtl) {
            addr = vb.buffer->gpuAddress() + vb.offset;
            vb.buffer->lastUse = serial;
        }
        bool unaligned = (stride & 3) || (addr & 3);
        for (int l = 0; l < kMaxVertexAttribs; l++)
            if (key.attribs[l].format && key.attribs[l].bufferIndex == glsl::kBufVertex0 + b && (key.attribs[l].offset & 3)) unaligned = true;
        if (unaligned) {
            // Metal needs 4-byte aligned strides/offsets: repack the used vertex range into
            // an aligned interleaved layout (each attribute at a 4-byte aligned offset).
            const uint8_t* src = nullptr;
            if (clientAttr && !(vb.buffer && vb.buffer->mtl)) src = (const uint8_t*)clientAttr->clientPointer;
            else if (vb.buffer && vb.buffer->cpu) {
                if (vb.buffer->lastGpuWrite && !ctx->dev->isComplete(vb.buffer->lastGpuWrite)) ctx->waitForResource(vb.buffer->lastGpuWrite);
                src = vb.buffer->cpu + vb.offset;
            }
            uint32_t lo = vb.divisor ? 0 : minIndex;
            uint32_t hi = vb.divisor ? (uint32_t)((d.instances + vb.divisor - 1) / vb.divisor + d.baseInstance) : maxIndex + 1;
            struct Rep { int loc; uint32_t srcOff, size, dstOff; };
            std::vector<Rep> reps;
            uint32_t newStride = 0;
            for (int l = 0; l < kMaxVertexAttribs; l++) {
                VertexAttribKey& ak = key.attribs[l];
                if (!ak.format || ak.bufferIndex != glsl::kBufVertex0 + b) continue;
                const VertexAttrib& va = vao->attribs[l];
                int comps = va.size == GL_BGRA ? 4 : va.size;
                int cs = (va.type == GL_BYTE || va.type == GL_UNSIGNED_BYTE) ? 1 : (va.type == GL_SHORT || va.type == GL_UNSIGNED_SHORT || va.type == GL_HALF_FLOAT) ? 2 : va.type == GL_DOUBLE ? 8 : 4;
                uint32_t size = (va.type == GL_INT_2_10_10_10_REV || va.type == GL_UNSIGNED_INT_2_10_10_10_REV || va.type == GL_UNSIGNED_INT_10F_11F_11F_REV) ? 4 : (uint32_t)(cs * comps);
                reps.push_back({l, ak.offset, size, newStride});
                ak.offset = newStride;
                newStride += (size + 3) & ~3u;
            }
            if (src && newStride && hi > lo) {
                size_t n = hi - lo;
                UploadAlloc u = ctx->upload(n * newStride + 16, 16);
                for (size_t v = 0; v < n; v++) {
                    const uint8_t* sv = src + (size_t)(lo + v) * stride;
                    uint8_t* dv = u.cpu + v * newStride;
                    for (auto& r : reps) memcpy(dv + r.dstOff, sv + r.srcOff, r.size);
                }
                addr = u.gpu - (uint64_t)lo * newStride;
                stride = (GLsizei)newStride;
            }
        }
        [table setAddress:addr attributeStride:stride atIndex:glsl::kBufVertex0 + b];
    }
    if (needCurrent) {
        // floats (16 bytes per location) followed by doubles (32 bytes per location)
        UploadAlloc ca = ctx->upload(48 * kMaxVertexAttribs, 16);
        memcpy(ca.cpu, ctx->currentAttrib, 16 * kMaxVertexAttribs);
        memcpy(ca.cpu + 16 * kMaxVertexAttribs, ctx->currentAttribD, 32 * kMaxVertexAttribs);
        VertexBufferKey& bk = key.buffers[kBufCurrentAttribs];
        bk.used = 1;
        bk.stepFunction = MTLVertexStepFunctionConstant;
        bk.stepRate = 0;
        [table setAddress:ca.gpu attributeStride:0 atIndex:kBufCurrentAttribs];
    }
    return true;
}

// KHR_blend_equation_advanced: index into glsl::kAdvancedBlendEquations, or -1.
static int advancedBlendIndex(GLenum eq) {
    if (eq < glsl::kAdvancedBlendEquations[0]) return -1;  // the classic equations are all below
    for (int i = 0; i < 15; i++)
        if (eq == glsl::kAdvancedBlendEquations[i]) return i;
    return -1;
}

// VariantKey::advancedBlend for the current state (0 = none): draw buffer 0 blends with an advanced
// equation, which the fragment shader computes from the destination it reads with framebuffer fetch.
static uint8_t advancedBlendKey(Context* ctx) {
    const BlendState& b = ctx->blend[0];
    if (!b.enabled || !ctx->advancedBlendUsed) return 0;
    int mode = advancedBlendIndex(b.eqRGB);
    if (mode < 0) return 0;
    Framebuffer* fb = ctx->drawFb.get();
    const FormatInfo* f = nullptr;
    bool noAlpha = false;
    if (fb->isDefault) {
        if (fb->drawBuffers[0] != GL_NONE && fb->surface && fb->surface->color) {
            f = formatInfoForMTL(fb->surface->color.pixelFormat);
            noAlpha = !fb->surface->hasAlpha;
        }
    } else if (fb->drawBuffers[0] >= GL_COLOR_ATTACHMENT0 && fb->drawBuffers[0] < GL_COLOR_ATTACHMENT0 + kMaxDrawBuffers) {
        const Attachment& a = fb->att[fb->drawBuffers[0] - GL_COLOR_ATTACHMENT0];
        if (a.type == GL_RENDERBUFFER && a.rb) f = a.rb->fmt;
        else if (a.type == GL_TEXTURE && a.tex) f = a.tex->storageFmt;
    }
    // No color buffer 0, or an integer one (never blended): plain output.
    if (!f || f->kind == FmtKind::UInt || f->kind == FmtKind::SInt) return 0;
    uint8_t k = (uint8_t)(mode + 1);
    if (noAlpha || f->aBits == 0) k |= glsl::kAdvancedBlendNoDstAlpha;
    if (f->kind == FmtKind::Unorm || f->kind == FmtKind::Snorm) k |= glsl::kAdvancedBlendClamp;
    return k;
}

// EXT_fragment_shading_rate (METALGLDRV_STRICT): a rate as log2 width << 2 | log2 height (gl_ShadingRateEXT).
static uint32_t packedShadingRate(GLenum r) {
    switch (r) {
    case GL_SHADING_RATE_1X2_PIXELS_EXT: return 1;
    case GL_SHADING_RATE_1X4_PIXELS_EXT: return 2;
    case GL_SHADING_RATE_2X1_PIXELS_EXT: return 4;
    case GL_SHADING_RATE_2X2_PIXELS_EXT: return 5;
    case GL_SHADING_RATE_2X4_PIXELS_EXT: return 6;
    case GL_SHADING_RATE_4X1_PIXELS_EXT: return 8;
    case GL_SHADING_RATE_4X2_PIXELS_EXT: return 9;
    case GL_SHADING_RATE_4X4_PIXELS_EXT: return 10;
    default: return 0;
    }
}

static Texture* shadingRateAttachment(Context* ctx) {
    Framebuffer* fb = ctx->drawFb.get();
    return fb && !fb->isDefault && fb->shadingRateLayers > 0 ? fb->shadingRateTex.get() : nullptr;
}

// Whether a draw may shade fragments larger than a pixel, i.e. needs the coarse-shading variant
// (VariantKey::fsr): an upper bound of the effective rate, with the largest primitive rate (if the last
// pre-rasterization stage may write one) and attachment rate (the combiner ops are monotonic). The
// effective rate is 1x1 with sample shading, a SAMPLE_MASK_VALUE without all bits set, or a fragment
// shader that forces it (Reflection::fsrForce1x1).
static bool fsrWanted(Context* ctx, bool primRate, const glsl::Reflection* fs) {
    if (!ctx->strict() || !fs || !(fs->stageMask & (1u << (int)glsl::Stage::Fragment)) || fs->fsrForce1x1) return false;
    if (ctx->sampleShading || ctx->sampleMaskValue != ~0u) return false;
    auto comb = [](uint32_t a, uint32_t b, GLenum op) {
        uint32_t ax = a >> 2, ay = a & 3, bx = b >> 2, by = b & 3, x = ax, y = ay;
        switch (op) {
        case GL_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_EXT: x = bx, y = by; break;
        case GL_FRAGMENT_SHADING_RATE_COMBINER_OP_MIN_EXT: x = std::min(ax, bx), y = std::min(ay, by); break;
        case GL_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_EXT: x = std::max(ax, bx), y = std::max(ay, by); break;
        case GL_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_EXT: x = std::min(ax + bx, 2u), y = std::min(ay + by, 2u); break;
        default: break;
        }
        return x << 2 | y;
    };
    uint32_t r = comb(packedShadingRate(ctx->shadingRate), primRate ? 10 : 0, ctx->shadingRateCombiner[0]);
    return comb(r, shadingRateAttachment(ctx) ? 10 : 0, ctx->shadingRateCombiner[1]) != 0;
}

// The glsl::MGLFsr of a VariantKey::fsr draw, uploaded. Resolves the attachment's storage, which can end the
// render pass: call before acquiring the encoder.
static uint64_t fsrUpload(Context* ctx) {
    glsl::MGLFsr f = {};
    f.state = packedShadingRate(ctx->shadingRate) | (ctx->shadingRateCombiner[0] - GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT) << 4 |
              (ctx->shadingRateCombiner[1] - GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT) << 7;
    if (Texture* t = shadingRateAttachment(ctx); t && ensureTextureStorage(ctx, t) && t->mtl) {
        id<MTLTexture> v = t->mtl;
        if (v.textureType != MTLTextureType2D || v.mipmapLevelCount > 1) {
            // layer 0, level 0 (the layer limit is 1)
            v = [v newTextureViewWithPixelFormat:v.pixelFormat textureType:MTLTextureType2D levels:NSMakeRange(0, 1) slices:NSMakeRange(0, 1)];
            ctx->dev->retainUntil(v, ctx->serial());
        }
        t->lastUse = ctx->serial();
        Framebuffer* fb = ctx->drawFb.get();
        f.state |= 1u << 10 | (uint32_t)fb->shadingRateTexelLog2[0] << 11 | (uint32_t)fb->shadingRateTexelLog2[1] << 16;
        MTLResourceID rid = v.gpuResourceID;
        memcpy(&f.attachment, &rid, 8);
    }
    UploadAlloc u = ctx->upload(sizeof f, 16);
    memcpy(u.cpu, &f, sizeof f);
    return u.gpu;
}

// KHR_blend_equation_advanced draw-time errors (INVALID_OPERATION), or null.
static const char* advancedBlendError(Context* ctx, ProgramObj* prog) {
    Framebuffer* fb = ctx->drawFb.get();
    int mode = -1;
    bool otherBuffers = false;
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (fb->drawBuffers[i] == GL_NONE) continue;
        if (i > 0) otherBuffers = true;
        if (ctx->blend[i].enabled && advancedBlendIndex(ctx->blend[i].eqRGB) >= 0) mode = advancedBlendIndex(ctx->blend[i].eqRGB);
    }
    if (mode < 0) return nullptr;
    if (otherBuffers) return "advanced blending with more than one draw buffer";
    if (fb->drawBuffers[0] == GL_FRONT_AND_BACK) return "advanced blending into FRONT_AND_BACK";
    ProgramObj* fs = ctx->pipeline && !ctx->program ? ctx->pipeline->stages[(int)glsl::Stage::Fragment].get() : prog;
    if (!fs || !(fs->refl.blendSupport & (1u << mode)))
        return "the fragment shader does not declare the blend_support qualifier of the advanced blend equation";
    return nullptr;
}

static bool usesSource1(GLenum f) {
    return f == GL_SRC1_COLOR || f == GL_ONE_MINUS_SRC1_COLOR || f == GL_SRC1_ALPHA || f == GL_ONE_MINUS_SRC1_ALPHA;
}

// SRC1 blend factors on draw buffer 0 with a fragment shader that has no index-1 output but one at location 1:
// that output is the second source (VariantKey::source1FromLocation1). GL leaves this undefined; Direct3D's
// dual-source blending reads output 1, other GL drivers do the same, and Unigine relies on it.
static bool source1FromLocation1(Context* ctx, const glsl::Reflection& refl) {
    const BlendState& b = ctx->blend[0];
    if (!b.enabled || !(usesSource1(b.srcRGB) || usesSource1(b.dstRGB) || usesSource1(b.srcA) || usesSource1(b.dstA))) return false;
    bool loc1 = false;
    for (auto& o : refl.fragOutputs) {
        if (o.index == 1) return false;
        loc1 |= o.location <= 1 && o.location + o.arraySize > 1;
    }
    return loc1;
}

// Dual-source factors with no second source at all: undefined in GL, a pipeline error in Metal. Blend as
// if the second source were zero.
static MTLBlendFactor withoutSource1(MTLBlendFactor f) {
    switch (f) {
    case MTLBlendFactorSource1Color: case MTLBlendFactorSource1Alpha: return MTLBlendFactorZero;
    case MTLBlendFactorOneMinusSource1Color: case MTLBlendFactorOneMinusSource1Alpha: return MTLBlendFactorOne;
    default: return f;
    }
}

static void fillColorKey(Context* ctx, RenderPipelineKey& key, const glsl::Reflection& refl) {
    const PassDesc& p = ctx->curPass;
    bool source1 = source1FromLocation1(ctx, refl);
    for (auto& o : refl.fragOutputs) source1 |= o.index == 1;
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (!p.color[i].tex) continue;
        MTLPixelFormat pf = p.color[i].tex.pixelFormat;
        // GL_FRAMEBUFFER_SRGB off: write linear values into sRGB-capable surfaces.
        if (!ctx->framebufferSRGB) {
            if (pf == MTLPixelFormatRGBA8Unorm_sRGB && (p.color[i].tex.usage & MTLTextureUsagePixelFormatView)) {
                // encoder writes through the base format; handled by attachment view creation (TODO).
            }
        }
        key.colorFormats[i] = (uint32_t)pf;
        const BlendState& b = ctx->blend[i];
        BlendKey& bk = key.blend[i];
        const FormatInfo* f = formatInfoForMTL(pf);
        bool intFmt = f && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt);
        // Advanced equations (draw buffer 0 only) blend in the fragment shader (advancedBlendKey).
        bk.enabled = b.enabled && !intFmt && !(i == 0 && advancedBlendIndex(b.eqRGB) >= 0);
        if (bk.enabled) {
            bk.srcRGB = blendFactor(b.srcRGB);
            bk.dstRGB = blendFactor(b.dstRGB);
            bk.srcA = blendFactor(b.srcA);
            bk.dstA = blendFactor(b.dstA);
            bk.eqRGB = blendOp(b.eqRGB);
            bk.eqA = blendOp(b.eqA);
            if (!source1)
                for (uint8_t* f : {&bk.srcRGB, &bk.dstRGB, &bk.srcA, &bk.dstA}) *f = (uint8_t)withoutSource1((MTLBlendFactor)*f);
        }
        bk.writeMask = p.color[i].noAlpha ? b.colorMask & ~8u : b.colorMask;
        // A fragment output must exist for the attachment; if the shader doesn't
        // write it, Metal leaves it unchanged (writes are masked by output presence).
    }
    key.sampleCount = std::max(1, p.samples);
    key.alphaToCoverage = ctx->sampleAlphaToCoverage && key.sampleCount > 1;
    key.alphaToOne = ctx->sampleAlphaToOne && key.sampleCount > 1;
}

// ---------------------------------------------------------------------------
// index processing
// ---------------------------------------------------------------------------
struct IndexInfo {
    uint64_t addr = 0;
    size_t length = 0;
    MTLIndexType type = MTLIndexTypeUInt32;
    uint32_t count = 0;
    MTLPrimitiveType prim = MTLPrimitiveTypeTriangle;
    bool generated = false;  // indices produced on CPU (fans/loops/quads/u8)
};

namespace {
void decomposeToList(Context* ctx, const DrawParams& d, const uint8_t* src, std::vector<uint32_t>& out);
GLenum listModeFor(GLenum mode);
}  // namespace

// Without a geometry shader the adjacency vertices of a primitive are processed but not
// rasterized: [a0 v0 v1 a1] -> [v0 v1], [v0 a0 v1 a1 v2 a2] -> [v0 v1 v2] (decomposeToList order).
static GLenum dropAdjacency(GLenum listMode, std::vector<uint32_t>& ids) {
    std::vector<uint32_t> out;
    if (listMode == GL_LINES_ADJACENCY) {
        for (size_t i = 0; i + 3 < ids.size(); i += 4) out.insert(out.end(), {ids[i + 1], ids[i + 2]});
        ids.swap(out);
        return GL_LINES;
    }
    if (listMode == GL_TRIANGLES_ADJACENCY) {
        for (size_t i = 0; i + 5 < ids.size(); i += 6) out.insert(out.end(), {ids[i], ids[i + 2], ids[i + 4]});
        ids.swap(out);
        return GL_TRIANGLES;
    }
    return listMode;
}

// Expands primitives Metal lacks and converts u8 / custom restart indices.
static bool prepareIndices(Context* ctx, const DrawParams& d, IndexInfo& out, uint32_t* minIdx, uint32_t* maxIdx) {
    GLenum mode = d.mode;
    const uint8_t* src = nullptr;
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    size_t isz = d.indexed ? indexSize(d.indexType) : 0;
    if (d.indexed) {
        if (ib) {
            if (!ib->mtl) return false;
            size_t off = (size_t)d.indices;
            if (off + (size_t)d.count * isz > (size_t)ib->size) {
                ctx->setError(GL_INVALID_OPERATION, "index range out of bounds");
                return false;
            }
            src = ib->cpu ? ib->cpu + off : nullptr;  // sparse: fetched below if the CPU needs the indices
        } else {
            src = (const uint8_t*)d.indices;
            if (!src) return false;
        }
    }
    bool needCpuMinMax = false;
    // client arrays in compat need the referenced vertex range
    if (ctx->vao)
        for (int l = 0; l < kMaxVertexAttribs; l++) {
            const VertexAttrib& va = ctx->vao->attribs[l];
            if (!va.enabled) continue;
            const VertexBinding& vb = ctx->vao->bindings[va.binding];
            // client arrays and Metal-incompatible (unaligned) layouts are repacked from a vertex range
            if (va.clientPointer || (vb.stride & 3) || (vb.offset & 3) || (va.relativeOffset & 3)) needCpuMinMax = true;
        }
    if (needCpuMinMax && wantManualFetch(ctx)) needCpuMinMax = false;  // no CPU-side vertex range needed
    bool restart = ctx->primitiveRestart || ctx->primitiveRestartFixed;
    uint32_t restartIdx = ctx->primitiveRestartFixed ? (d.indexType == GL_UNSIGNED_BYTE ? 0xff : d.indexType == GL_UNSIGNED_SHORT ? 0xffff : 0xffffffffu)
                                                      : ctx->primitiveRestartIndex;
    uint32_t allOnes = d.indexType == GL_UNSIGNED_BYTE ? 0xff : d.indexType == GL_UNSIGNED_SHORT ? 0xffff : 0xffffffffu;
    bool expand = mode == GL_TRIANGLE_FAN || mode == GL_LINE_LOOP || mode == GL_QUADS || mode == GL_QUAD_STRIP || mode == GL_POLYGON;
    bool convert = d.indexed && (d.indexType == GL_UNSIGNED_BYTE || (restart && restartIdx != allOnes));
    if (d.indexed && ib && (expand || convert || needCpuMinMax) && ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite))
        ctx->waitForResource(ib->lastGpuWrite);
    // Adjacency topologies without a geometry shader draw their non-adjacent vertices, and list
    // topologies with primitive restart drop the incomplete primitive before each restart
    // (Metal only restarts strips): decompose to a plain list on the CPU.
    bool adjacency = mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY || mode == GL_TRIANGLES_ADJACENCY ||
                     mode == GL_TRIANGLE_STRIP_ADJACENCY;
    bool listRestart = restart && d.indexed && (mode == GL_POINTS || mode == GL_LINES || mode == GL_TRIANGLES);
    if (d.indexed && ib && !src && (expand || convert || needCpuMinMax || adjacency || listRestart))
        src = bufferBytes(ctx, ib, (size_t)d.indices, (size_t)d.count * isz);
    *minIdx = 0;
    *maxIdx = d.indexed ? 0 : (uint32_t)(d.first + d.count - 1);
    if (!d.indexed) *minIdx = d.first;
    if (d.indexed && needCpuMinMax) {
        uint32_t mn = UINT32_MAX, mx = 0;
        for (GLsizei i = 0; i < d.count; i++) {
            uint32_t v = readIndex(src, d.indexType, i);
            if (restart && v == restartIdx) continue;
            mn = std::min(mn, v);
            mx = std::max(mx, v);
        }
        *minIdx = mn == UINT32_MAX ? 0 : mn + d.baseVertex;
        *maxIdx = mx + d.baseVertex;
    }
    if (adjacency || listRestart) {
        if (d.indexed && ib && ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
        std::vector<uint32_t> list, idx;
        decomposeToList(ctx, d, d.indexed ? src : nullptr, list);
        if (mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY) {
            for (size_t i = 0; i + 3 < list.size(); i += 4) idx.insert(idx.end(), {list[i + 1], list[i + 2]});
            out.prim = MTLPrimitiveTypeLine;
        } else if (adjacency) {
            for (size_t i = 0; i + 5 < list.size(); i += 6) idx.insert(idx.end(), {list[i], list[i + 2], list[i + 4]});
            out.prim = MTLPrimitiveTypeTriangle;
        } else {
            idx = std::move(list);
            out.prim = mode == GL_POINTS ? MTLPrimitiveTypePoint : mode == GL_LINES ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
        }
        out.generated = true;
        out.count = (uint32_t)idx.size();
        if (idx.empty()) return true;
        UploadAlloc u = ctx->upload(idx.size() * 4, 16);
        memcpy(u.cpu, idx.data(), idx.size() * 4);
        out.addr = u.gpu;
        out.length = idx.size() * 4;
        out.type = MTLIndexTypeUInt32;
        return true;
    }
    switch (mode) {
    case GL_POINTS: out.prim = MTLPrimitiveTypePoint; break;
    case GL_LINES: out.prim = MTLPrimitiveTypeLine; break;
    case GL_LINE_STRIP: out.prim = MTLPrimitiveTypeLineStrip; break;
    case GL_TRIANGLES: out.prim = MTLPrimitiveTypeTriangle; break;
    case GL_TRIANGLE_STRIP: out.prim = MTLPrimitiveTypeTriangleStrip; break;
    case GL_LINE_LOOP: out.prim = MTLPrimitiveTypeLine; break;
    default: out.prim = MTLPrimitiveTypeTriangle; break;
    }
    if (!expand && !convert) {
        if (d.indexed) {
            if (ib) {
                out.addr = ib->gpuAddress() + (size_t)d.indices;
                out.length = (size_t)d.count * isz;
                ib->lastUse = ctx->serial();
            } else {
                UploadAlloc u = ctx->upload((size_t)d.count * isz + 4, 16);
                memcpy(u.cpu, src, (size_t)d.count * isz);
                out.addr = u.gpu;
                out.length = (size_t)d.count * isz;
            }
            out.type = d.indexType == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
            out.count = d.count;
        }
        return true;
    }
    // Generate a 32-bit index list on the CPU.
    std::vector<uint32_t> idx;
    auto get = [&](size_t i) -> uint32_t { return d.indexed ? readIndex(src, d.indexType, i) : (uint32_t)(d.first + i); };
    auto isRestart = [&](uint32_t v) { return d.indexed && restart && v == restartIdx; };
    // Split into restart-delimited runs.
    std::vector<std::pair<size_t, size_t>> runs;
    size_t start = 0;
    for (size_t i = 0; i < (size_t)d.count; i++) {
        if (isRestart(get(i))) {
            runs.push_back({start, i});
            start = i + 1;
        }
    }
    runs.push_back({start, (size_t)d.count});
    bool keepRestart = !expand;  // strips keep restart semantics with 0xFFFFFFFF
    for (size_t r = 0; r < runs.size(); r++) {
        auto [b, e] = runs[r];
        size_t n = e - b;
        if (!expand) {
            for (size_t i = b; i < e; i++) idx.push_back(get(i));
            if (keepRestart && r + 1 < runs.size()) idx.push_back(0xffffffffu);
            continue;
        }
        switch (mode) {
        case GL_TRIANGLE_FAN:
        case GL_POLYGON:
            for (size_t i = 2; i < n; i++) {
                // GL provoking vertex for fans is the last vertex (i); keep i last.
                idx.push_back(get(b));
                idx.push_back(get(b + i - 1));
                idx.push_back(get(b + i));
            }
            break;
        case GL_LINE_LOOP:
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; i++) {
                idx.push_back(get(b + i));
                idx.push_back(get(b + i + 1));
            }
            idx.push_back(get(b + n - 1));
            idx.push_back(get(b));
            break;
        case GL_QUADS:
            for (size_t q = 0; q + 3 < n; q += 4) {
                uint32_t v0 = get(b + q), v1 = get(b + q + 1), v2 = get(b + q + 2), v3 = get(b + q + 3);
                idx.insert(idx.end(), {v0, v1, v3, v1, v2, v3});
            }
            break;
        case GL_QUAD_STRIP:
            for (size_t q = 0; q + 3 < n; q += 2) {
                uint32_t v0 = get(b + q), v1 = get(b + q + 1), v2 = get(b + q + 2), v3 = get(b + q + 3);
                idx.insert(idx.end(), {v0, v1, v3, v0, v3, v2});
            }
            break;
        }
    }
    if (idx.empty()) {
        out.count = 0;
        return true;
    }
    UploadAlloc u = ctx->upload(idx.size() * 4, 16);
    memcpy(u.cpu, idx.data(), idx.size() * 4);
    out.addr = u.gpu;
    out.length = idx.size() * 4;
    out.type = MTLIndexTypeUInt32;
    out.count = (uint32_t)idx.size();
    out.generated = true;
    return true;
}

// ---------------------------------------------------------------------------
// ARB_post_depth_coverage (super strict)
// ---------------------------------------------------------------------------
// Metal's [[sample_mask]] input doesn't reflect the depth/stencil tests, even with early fragment tests. A draw
// whose fragment shader declares post_depth_coverage and reads gl_SampleMaskIn (Reflection::postDepthCoverage),
// with a depth or stencil test in effect, first renders the same geometry into a transient coverage attachment
// (R8, cleared to 0) with the draw's depth/stencil tests, depth/stencil writes off and a fragment function that
// writes 1 (mgl_fs_pdc): exactly the samples that pass are marked. The draw itself (VariantKey::pdc) then ANDs
// gl_SampleMaskIn with the marked samples of its pixel. Both passes use the same vertex function (invariant
// position) and the same rasterizer and depth/stencil state, so they rasterize and test the same samples.
// When the draw writes depth or stencil, its primitives affect each other in order: it is split into a pre-pass
// and a draw per primitive and instance, in GL order. Pre-passes don't count for queries or transform feedback,
// and bind copies of the storage the pre-rasterization stages may write (buildArguments `inert`).
namespace {

struct PdcInfo {
    bool tests = false;   // a depth or stencil test is in effect: pre-pass + coverage-reading draw (VariantKey::pdc)
    bool writes = false;  // ... that may write depth or stencil: split per primitive and instance
};

PdcInfo pdcInfo(Context* ctx) {
    PdcInfo r;
    PassDesc p;
    if (!ctx->resolvePass(p)) return r;
    bool depth = ctx->depthTest && p.depth.tex, stencil = ctx->stencilTest && p.stencil.tex;
    r.tests = depth || stencil;
    r.writes = depth && ctx->depthWrite;
    if (stencil)
        for (int f = 0; f < 2; f++) {
            const StencilFace& s = ctx->stencil[f];
            if ((s.writeMask & 0xff) && (s.sfail != GL_KEEP || s.dpfail != GL_KEEP || s.dppass != GL_KEEP)) r.writes = true;
        }
    return r;
}

// Per-instance attributes with a divisor above 1 can't be fetched by Metal for one instance of a split draw (the
// instance is passed as the base instance): such draws fetch in the shader (VariantKey::manualFetch).
bool pdcNeedsManualFetch(Context* ctx) {
    VertexArray* vao = ctx->vao.get();
    if (!vao) return false;
    bool divisor = false, client = false;
    for (int l = 0; l < kMaxVertexAttribs; l++) {
        const VertexAttrib& va = vao->attribs[l];
        if (!va.enabled) continue;
        if (va.clientPointer) client = true;
        if (vao->bindings[va.binding].divisor > 1) divisor = true;
    }
    if (divisor && client) {
        static bool warned = false;
        if (!warned) MGL_WARN("post_depth_coverage: instanced client arrays with a divisor above 1 fetch the wrong instance");
        warned = true;
    }
    return divisor && !client;
}

std::unordered_map<Context*, id<MTLTexture>> g_pdcCoverage;  // per context, released by releaseDrawCaches

// The coverage attachment for a pass (2D multisample array / 2D array with the pass's layers).
id<MTLTexture> pdcCoverage(Context* ctx, const PassDesc& p) {
    __strong id<MTLTexture>& c = g_pdcCoverage[ctx];
    NSUInteger layers = std::max(1, p.layers), samples = std::max(1, p.samples);
    if (c && c.width == (NSUInteger)p.width && c.height == (NSUInteger)p.height && c.arrayLength == layers && c.sampleCount == samples)
        return c;
    if (c) ctx->dev->releaseAfter(c, ctx->serial());
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = samples > 1 ? MTLTextureType2DMultisampleArray : MTLTextureType2DArray;
    td.pixelFormat = MTLPixelFormatR8Unorm;
    td.width = std::max(1, p.width);
    td.height = std::max(1, p.height);
    td.arrayLength = layers;
    td.sampleCount = samples;
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    c = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (c) ctx->dev->makeResident(c);
    return c;
}

// Ends the open pass and begins a pre-pass of `main`: the coverage attachment (cleared) and main's depth/stencil.
id<MTL4RenderCommandEncoder> pdcBeginPrepass(Context* ctx, const PassDesc& main, id<MTLTexture> cov) {
    ctx->endPass();
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    MTLRenderPassColorAttachmentDescriptor* c = rp.colorAttachments[0];
    c.texture = cov;
    c.loadAction = MTLLoadActionClear;
    c.clearColor = MTLClearColorMake(0, 0, 0, 0);
    c.storeAction = MTLStoreActionStore;
    if (main.depth.tex) {
        MTLRenderPassDepthAttachmentDescriptor* d = rp.depthAttachment;
        d.texture = main.depth.tex;
        d.level = main.depth.level;
        d.slice = main.depth.slice;
        d.loadAction = MTLLoadActionLoad;
        d.storeAction = MTLStoreActionStore;
    }
    if (main.stencil.tex) {
        MTLRenderPassStencilAttachmentDescriptor* s = rp.stencilAttachment;
        s.texture = main.stencil.tex;
        s.level = main.stencil.level;
        s.slice = main.stencil.slice;
        s.loadAction = MTLLoadActionLoad;
        s.storeAction = MTLStoreActionStore;
    }
    rp.renderTargetWidth = main.width;
    rp.renderTargetHeight = main.height;
    if (main.layers > 1) rp.renderTargetArrayLength = main.layers;
    rp.defaultRasterSampleCount = main.samples;
    if (ctx->computeEnc) {
        [ctx->computeEnc endEncoding];
        ctx->computeEnc = nil;
    }
    id<MTL4RenderCommandEncoder> e = ctx->newRenderEncoder(rp);
    if (!e) return nil;
    MGL_STAT("pdc-prepass");
    [e barrierAfterQueueStages:MTLStageAll
                  beforeStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh
             visibilityOptions:MTL4VisibilityOptionDevice];
    [e setArgumentTable:ctx->renderArgTable atStages:MTLRenderStageVertex | MTLRenderStageFragment | MTLRenderStageObject | MTLRenderStageMesh];
    PassDesc p;
    p.color[0].tex = cov;
    p.depth = main.depth;
    p.stencil = main.stencil;
    p.width = main.width;
    p.height = main.height;
    p.layers = main.layers;
    p.samples = main.samples;
    ctx->renderEnc = e;
    ctx->curPass = p;
    ctx->passOpen = true;
    ctx->passDirty = true;  // not the draw framebuffer's pass
    ctx->passVisibilityBuffer = nil;
    ctx->encVisibilityOffset = -1;  // no occlusion counting
    ctx->enc.reset();
    return e;
}

// The pre-pass pipeline key of a draw's key: one R8 attachment, no blending / alpha to coverage, mgl_fs_pdc.
RenderPipelineKey pdcPrepassKey(const RenderPipelineKey& key) {
    RenderPipelineKey pk = key;
    for (int i = 0; i < 8; i++) {
        pk.colorFormats[i] = MTLPixelFormatInvalid;
        pk.blend[i] = BlendKey();
    }
    pk.colorFormats[0] = MTLPixelFormatR8Unorm;
    pk.blend[0].writeMask = 0xf;
    pk.alphaToCoverage = pk.alphaToOne = 0;
    pk.logicOp = 0;
    pk.pdcPrepass = 1;
    return pk;
}

// glsl::MGLPdc for the draw after a pre-pass.
uint64_t pdcUpload(Context* ctx, id<MTLTexture> cov, uint32_t primBase, bool layered) {
    UploadAlloc u = ctx->upload(sizeof(glsl::MGLPdc), 16);
    glsl::MGLPdc* p = (glsl::MGLPdc*)u.cpu;
    memset(p, 0, sizeof *p);
    MTLResourceID rid = cov.gpuResourceID;
    memcpy(cov.sampleCount > 1 ? &p->coverageMS : &p->coverage, &rid, 8);
    p->primBase = primBase;
    p->layered = layered && cov.arrayLength > 1;
    return u.gpu;
}

// The primitives of a split draw as vertex tuples in GL order (strip parity, fans, loops, adjacency without a
// geometry shader, restart), each rotated (which keeps its winding) so that Metal's provoking vertex, the first,
// is GL's. ids: draw-relative vertex ids (index values for indexed draws, base vertex not applied); prim: the
// gl_PrimitiveID of each tuple; vpp: vertices per tuple (1, 2, 3).
bool pdcPrimitives(Context* ctx, const DrawParams& d, std::vector<uint32_t>& ids, std::vector<uint32_t>& prim, int& vpp) {
    const uint8_t* src = nullptr;
    if (d.indexed) {
        Buffer* ib = ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
        if (ib) {
            if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
            src = ib->cpu ? ib->cpu + (size_t)d.indices : bufferBytes(ctx, ib, (size_t)d.indices, (size_t)d.count * indexSize(d.indexType));
        } else {
            src = (const uint8_t*)d.indices;
        }
        if (!src) return false;
    }
    std::vector<uint32_t> list;
    decomposeToList(ctx, d, src, list);
    GLenum lm = listModeFor(d.mode);
    if (lm == GL_LINES_ADJACENCY || lm == GL_TRIANGLES_ADJACENCY) lm = dropAdjacency(lm, list);
    vpp = lm == GL_POINTS ? 1 : lm == GL_LINES ? 2 : 3;
    // Strip triangles alternate their first-convention provoking vertex (decomposeToList swaps the first two
    // vertices of odd triangles): parity within each restart-delimited run.
    std::vector<uint8_t> odd;
    if (d.mode == GL_TRIANGLE_STRIP || d.mode == GL_TRIANGLE_STRIP_ADJACENCY) {
        bool restart = src && (ctx->primitiveRestart || ctx->primitiveRestartFixed);
        uint32_t restartIdx = ctx->primitiveRestartFixed ? (d.indexType == GL_UNSIGNED_BYTE ? 0xff : d.indexType == GL_UNSIGNED_SHORT ? 0xffff : 0xffffffffu)
                                                          : ctx->primitiveRestartIndex;
        auto run = [&](size_t n) {
            size_t tris = d.mode == GL_TRIANGLE_STRIP ? (n > 2 ? n - 2 : 0) : (n >= 6 ? (n - 4) / 2 : 0);
            for (size_t i = 0; i < tris; i++) odd.push_back(i & 1);
        };
        size_t n = 0;
        for (size_t i = 0; i < (size_t)d.count; i++) {
            if (restart && readIndex(src, d.indexType, i) == restartIdx) {
                run(n);
                n = 0;
            } else {
                n++;
            }
        }
        run(n);
    }
    bool last = ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION;
    size_t n = list.size() / vpp;
    ids.resize(n * vpp);
    prim.resize(n);
    for (size_t k = 0; k < n; k++) {
        int pv;  // position of GL's provoking vertex in the tuple
        uint32_t id = (uint32_t)k;
        switch (d.mode) {
        case GL_QUADS: pv = 2; id = (uint32_t)(k / 2); break;  // compatibility: the quad's last vertex
        case GL_QUAD_STRIP: pv = (k & 1) ? 1 : 2; id = (uint32_t)(k / 2); break;
        case GL_POLYGON: pv = 0; id = 0; break;
        case GL_TRIANGLE_FAN: pv = last ? 2 : 1; break;
        case GL_TRIANGLE_STRIP:
        case GL_TRIANGLE_STRIP_ADJACENCY: pv = last ? 2 : (k < odd.size() && odd[k]) ? 1 : 0; break;
        default: pv = last ? vpp - 1 : 0; break;
        }
        for (int j = 0; j < vpp; j++) ids[k * vpp + j] = list[k * vpp + (pv + j) % vpp];
        prim[k] = id;
    }
    return true;
}

}  // namespace

// The primitives of a draw split per primitive (pdcPrimitives). They are read before the draw's uploads: reading
// the indices may wait for the GPU, which retires upload memory.
struct PdcSplit {
    std::vector<uint32_t> ids, prims;
    int vpp = 0;
};

// Executes a VariantKey::pdc draw on the classic pipeline: pre-pass(es) and the draw (see above), split per
// primitive and instance if `split`. argAddr / fsArgAddr: the draw's arguments, built before any encoder (shared by
// the passes of a split draw).
static void executePdc(Context* ctx, ProgramObj* prog, ProgramVariant* var, ProgramObj* fsProg, ProgramVariant* fsVar, bool mixedFs,
                       const glsl::Reflection& frefl, const DrawParams& d, const IndexInfo& ii, uint32_t minIdx, uint32_t maxIdx,
                       uint64_t argAddr, uint64_t fsArgAddr, const PdcSplit* split) {
    const glsl::Reflection& refl = prog->refl;
    int vpp = split ? split->vpp : 0;
    uint64_t idsAddr = 0;
    if (split) {
        const std::vector<uint32_t>& ids = split->ids;
        if (split->prims.empty()) return;
        UploadAlloc u = ctx->upload(ids.size() * 4, 16);
        memcpy(u.cpu, ids.data(), ids.size() * 4);
        idsAddr = u.gpu;
        if (d.indexed) {  // the vertex range of CPU-side vertex copies
            uint32_t mn = UINT32_MAX, mx = 0;
            for (uint32_t v : ids) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
            minIdx = mn + d.baseVertex;
            maxIdx = mx + d.baseVertex;
        }
    }
    uint64_t preArgAddr = buildArguments(ctx, prog, false, true);
    // The draw framebuffer's pass (applies pending clears) defines the attachments.
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) return;
    PassDesc mainPass = ctx->curPass;
    id<MTLTexture> cov = pdcCoverage(ctx, mainPass);
    if (!cov) return;
    id<MTL4ArgumentTable> table = ctx->renderArgTable;
    RenderPipelineKey key;
    key.variant = var;
    if (mixedFs) {
        key.fsVariant = fsVar;
        key.fsVariantUid = fsVar ? fsVar->uid : ~0ull;
    }
    fillColorKey(ctx, key, frefl);
    key.rasterEnabled = true;
    key.topologyClass = (uint8_t)(refl.writesLayer ? topologyClass(d.mode) : MTLPrimitiveTopologyClassUnspecified);
    if (!setupVertexInput(ctx, prog, key, table, d, minIdx, maxIdx)) return;
    id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
    id<MTLRenderPipelineState> prePso = ctx->pipelines_->renderPipeline(pdcPrepassKey(key));
    if (!pso || !prePso) return;
    uint64_t sysAddr = buildSys(ctx, prog, d.drawID, d.baseVertex, d.baseInstance, d.first, d.indexed);
    uint64_t fsSysAddr = mixedFs && fsProg ? buildSys(ctx, fsProg, d.drawID, d.baseVertex, d.baseInstance, d.first, d.indexed) : sysAddr;
    MTLPrimitiveType listPrim = vpp == 1 ? MTLPrimitiveTypePoint : vpp == 2 ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
    // k < 0: the whole draw, else primitive k of instance `inst`
    auto encodeDraw = [&](id<MTL4RenderCommandEncoder> enc, long k, int inst) {
        if (k >= 0) {
            [enc drawIndexedPrimitives:listPrim indexCount:vpp indexType:MTLIndexTypeUInt32 indexBuffer:idsAddr + (uint64_t)k * vpp * 4
                     indexBufferLength:(NSUInteger)vpp * 4 instanceCount:1 baseVertex:d.indexed ? d.baseVertex : 0
                          baseInstance:d.baseInstance + inst];
        } else if (d.indexed || ii.generated) {
            [enc drawIndexedPrimitives:ii.prim indexCount:ii.count indexType:ii.type indexBuffer:ii.addr indexBufferLength:ii.length
                         instanceCount:d.instances baseVertex:d.indexed ? d.baseVertex : 0 baseInstance:d.baseInstance];
        } else {
            [enc drawPrimitives:ii.prim vertexStart:d.first vertexCount:d.count instanceCount:d.instances baseInstance:d.baseInstance];
        }
    };
    auto prepass = [&](long k, int inst) {
        id<MTL4RenderCommandEncoder> pe = pdcBeginPrepass(ctx, mainPass, cov);
        if (!pe) return false;
        [pe setRenderPipelineState:prePso];
        ctx->enc.pipeline = prePso;
        applyDynamicState(ctx, pe, d.mode, refl.writesViewportIndex, true);
        [table setAddress:preArgAddr ? preArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [table setAddress:sysAddr atIndex:glsl::kBufSys];
        encodeDraw(pe, k, inst);
        ctx->endPass();
        return true;
    };
    auto draw = [&](long k, int inst) {
        id<MTL4RenderCommandEncoder> me = ctx->renderEncoder();
        if (!me || !(ctx->curPass == mainPass)) return false;
        [me setRenderPipelineState:pso];
        ctx->enc.pipeline = pso;
        applyDynamicState(ctx, me, d.mode, refl.writesViewportIndex);
        [table setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [table setAddress:sysAddr atIndex:glsl::kBufSys];
        id<MTL4ArgumentTable> fsTable = table;
        if (mixedFs) {
            [me setArgumentTable:ctx->renderArgTableFS atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = true;
            [ctx->renderArgTableFS setAddress:fsArgAddr ? fsArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
            [ctx->renderArgTableFS setAddress:fsSysAddr atIndex:glsl::kBufSys];
            fsTable = ctx->renderArgTableFS;
        }
        [fsTable setAddress:pdcUpload(ctx, cov, k >= 0 ? split->prims[k] : 0, refl.writesLayer) atIndex:glsl::kBufPdc];
        mglQueryNotifyDraw(ctx, d.mode, d.count, d.instances);
        encodeDraw(me, k, inst);
        return true;
    };
    if (!split) {
        if (prepass(-1, 0)) draw(-1, 0);
        return;
    }
    // GL order: instance by instance, primitive by primitive
    for (int inst = 0; inst < d.instances; inst++)
        for (size_t k = 0; k < split->prims.size(); k++)
            if (!prepass((long)k, inst) || !draw((long)k, inst)) return;
}

// ---------------------------------------------------------------------------
// multi-pass pipelines (GS-on-mesh, tessellation, transform feedback capture)
// ---------------------------------------------------------------------------
namespace {

// Vertices per primitive of a list topology.
int listVerts(GLenum mode, int patchVerts) {
    switch (mode) {
    case GL_POINTS: return 1;
    case GL_LINES: return 2;
    case GL_TRIANGLES: return 3;
    case GL_LINES_ADJACENCY: return 4;
    case GL_TRIANGLES_ADJACENCY: return 6;
    case GL_PATCHES: return patchVerts;
    default: return 3;
    }
}

GLenum listModeFor(GLenum mode) {
    switch (mode) {
    case GL_POINTS: return GL_POINTS;
    case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP: return GL_LINES;
    case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: return GL_LINES_ADJACENCY;
    case GL_TRIANGLES_ADJACENCY: case GL_TRIANGLE_STRIP_ADJACENCY: return GL_TRIANGLES_ADJACENCY;
    case GL_PATCHES: return GL_PATCHES;
    default: return GL_TRIANGLES;
    }
}

// Decomposes any topology into a list of vertex ids following GL's primitive
// decomposition rules (strip parity, fans, loops, adjacency, restart).
// `ids` are draw-relative vertex ids (index values for indexed draws).
void decomposeToList(Context* ctx, const DrawParams& d, const uint8_t* src, std::vector<uint32_t>& out) {
    auto get = [&](size_t i) -> uint32_t { return src ? readIndex(src, d.indexType, i) : (uint32_t)(d.first + i); };
    bool restart = src && (ctx->primitiveRestart || ctx->primitiveRestartFixed);
    uint32_t restartIdx = ctx->primitiveRestartFixed ? (d.indexType == GL_UNSIGNED_BYTE ? 0xff : d.indexType == GL_UNSIGNED_SHORT ? 0xffff : 0xffffffffu)
                                                      : ctx->primitiveRestartIndex;
    std::vector<std::pair<size_t, size_t>> runs;
    size_t start = 0;
    for (size_t i = 0; i < (size_t)d.count; i++)
        if (restart && get(i) == restartIdx) {
            runs.push_back({start, i});
            start = i + 1;
        }
    runs.push_back({start, (size_t)d.count});
    for (auto [b, e] : runs) {
        size_t n = e - b;
        auto v = [&](size_t i) { return get(b + i); };
        switch (d.mode) {
        case GL_POINTS: for (size_t i = 0; i < n; i++) out.push_back(v(i)); break;
        case GL_LINES: for (size_t i = 0; i + 1 < n; i += 2) out.insert(out.end(), {v(i), v(i + 1)}); break;
        case GL_LINE_STRIP: for (size_t i = 0; i + 1 < n; i++) out.insert(out.end(), {v(i), v(i + 1)}); break;
        case GL_LINE_LOOP:
            if (n < 2) break;
            for (size_t i = 0; i + 1 < n; i++) out.insert(out.end(), {v(i), v(i + 1)});
            out.insert(out.end(), {v(n - 1), v(0)});
            break;
        case GL_TRIANGLES: for (size_t i = 0; i + 2 < n; i += 3) out.insert(out.end(), {v(i), v(i + 1), v(i + 2)}); break;
        case GL_TRIANGLE_STRIP:
            for (size_t i = 0; i + 2 < n; i++) {
                if (i & 1) out.insert(out.end(), {v(i + 1), v(i), v(i + 2)});
                else out.insert(out.end(), {v(i), v(i + 1), v(i + 2)});
            }
            break;
        case GL_TRIANGLE_FAN: case GL_POLYGON:
            for (size_t i = 1; i + 1 < n; i++) out.insert(out.end(), {v(0), v(i), v(i + 1)});
            break;
        case GL_QUADS:
            for (size_t q = 0; q + 3 < n; q += 4) out.insert(out.end(), {v(q), v(q + 1), v(q + 3), v(q + 1), v(q + 2), v(q + 3)});
            break;
        case GL_QUAD_STRIP:
            for (size_t q = 0; q + 3 < n; q += 2) out.insert(out.end(), {v(q), v(q + 1), v(q + 3), v(q), v(q + 3), v(q + 2)});
            break;
        case GL_LINES_ADJACENCY: for (size_t i = 0; i + 3 < n; i += 4) out.insert(out.end(), {v(i), v(i + 1), v(i + 2), v(i + 3)}); break;
        case GL_LINE_STRIP_ADJACENCY: for (size_t i = 0; i + 3 < n; i++) out.insert(out.end(), {v(i), v(i + 1), v(i + 2), v(i + 3)}); break;
        case GL_TRIANGLES_ADJACENCY: for (size_t i = 0; i + 5 < n; i += 6) for (int k = 0; k < 6; k++) out.push_back(v(i + k)); break;
        case GL_TRIANGLE_STRIP_ADJACENCY: {
            // GL 4.6 table 10.1: triangle i of a strip with adjacency.
            if (n < 6) break;
            size_t tris = (n - 4) / 2;
            for (size_t i = 0; i < tris; i++) {
                uint32_t p[6];
                if (tris == 1) {
                    p[0] = v(0); p[1] = v(1); p[2] = v(2); p[3] = v(5); p[4] = v(4); p[5] = v(3);
                } else if (i == 0) {
                    p[0] = v(0); p[1] = v(1); p[2] = v(2); p[3] = v(6); p[4] = v(4); p[5] = v(3);
                } else if (i == tris - 1) {
                    if (i & 1) { p[0] = v(2 * i + 2); p[1] = v(2 * i - 2); p[2] = v(2 * i); p[3] = v(2 * i + 3); p[4] = v(2 * i + 4); p[5] = v(2 * i + 5); }
                    else { p[0] = v(2 * i); p[1] = v(2 * i - 2); p[2] = v(2 * i + 2); p[3] = v(2 * i + 5); p[4] = v(2 * i + 4); p[5] = v(2 * i + 3); }
                } else {
                    if (i & 1) { p[0] = v(2 * i + 2); p[1] = v(2 * i - 2); p[2] = v(2 * i); p[3] = v(2 * i + 3); p[4] = v(2 * i + 4); p[5] = v(2 * i + 6); }
                    else { p[0] = v(2 * i); p[1] = v(2 * i - 2); p[2] = v(2 * i + 2); p[3] = v(2 * i + 6); p[4] = v(2 * i + 4); p[5] = v(2 * i + 3); }
                }
                out.insert(out.end(), p, p + 6);
            }
            break;
        }
        case GL_PATCHES: {
            int pv = ctx->patchVertices;
            for (size_t i = 0; i + pv <= n; i += pv)
                for (int k = 0; k < pv; k++) out.push_back(v(i + k));
            break;
        }
        }
    }
}

// Locations where integer vertex data of one signedness feeds an input of the other.
static uint32_t attribSignFlip(Context* ctx, ProgramObj* prog) {
    VertexArray* vao = ctx->vao.get();
    if (!vao) return 0;
    uint32_t mask = 0;
    for (const auto& a : prog->refl.attributes) {
        TypeShape ts = typeShape(a.type);
        if (a.location < 0 || (ts.base != BaseType::Int && ts.base != BaseType::UInt)) continue;
        for (int k = 0; k < std::max(1, a.locationsUsed); k++) {
            int loc = a.location + k;
            if (loc >= kMaxVertexAttribs) break;
            const VertexAttrib& va = vao->attribs[loc];
            if (!va.enabled) continue;
            bool srcSigned;
            switch (va.type) {
            case GL_BYTE: case GL_SHORT: case GL_INT: srcSigned = true; break;
            case GL_UNSIGNED_BYTE: case GL_UNSIGNED_SHORT: case GL_UNSIGNED_INT: srcSigned = false; break;
            default: continue;
            }
            if (srcSigned != (ts.base == BaseType::Int)) mask |= 1u << loc;
        }
    }
    return mask;
}

// Fills VariantKey::attribs (manual vertex fetch) from the bound VAO.
void fillVertexFetch(Context* ctx, ProgramObj* prog, glsl::VariantKey& vk) {
    VertexArray* vao = ctx->vao.get();
    for (const auto& a : prog->refl.attributes) {
        int nloc = std::max(1, a.locationsUsed);
        for (int k = 0; k < nloc; k++) {
            int loc = a.location + k;
            if (loc < 0 || loc >= kMaxVertexAttribs) continue;
            glsl::VertexFetch& f = vk.attribs[loc];
            const VertexAttrib& va = vao->attribs[loc];
            const VertexBinding& vb = vao->bindings[va.binding];
            f.enabled = va.enabled && vb.buffer && vb.buffer->mtl;
            if (!f.enabled) continue;
            f.glType = va.type;
            f.size = va.size == GL_BGRA ? 5 : (uint8_t)va.size;
            f.normalized = va.normalized;
            f.integer = va.integer;
            f.isDouble = va.isLong;
            f.binding = (uint8_t)va.binding;
            f.relativeOffset = va.relativeOffset;
        }
    }
}

}  // namespace


// Program pipelines whose geometry/tessellation stages (and the stages around them) come from
// different separable programs: draws go through a composite program generated from all stage
// programs (pipelineComposite), each stage reading its own program's argument buffer. Pipelines
// without geometry/tessellation stages use the classic path (fragment stage mixed separately).
ProgramObj* mglPipelineDrawProgram(Context* ctx) {
    if (ctx->program || !ctx->pipeline) return ctx->activeProgramForStage(glsl::Stage::Vertex);
    ProgramPipelineObj* pp = ctx->pipeline.get();
    ProgramObj* vs = pp->stages[(int)glsl::Stage::Vertex].get();
    if (!vs) return nullptr;
    ProgramObj* st[(int)glsl::Stage::Count] = {};
    bool multipass = false, mixed = false;
    for (int s = 0; s < (int)glsl::Stage::Compute; s++) {
        st[s] = pp->stages[s].get();
        if (st[s] && !st[s]->program) st[s] = nullptr;
        if (!st[s]) continue;
        if (s == (int)glsl::Stage::Geometry || s == (int)glsl::Stage::TessEval || s == (int)glsl::Stage::TessControl) multipass = true;
        if (st[s] != vs) mixed = true;
    }
    if (!multipass || !mixed) return vs;
    std::vector<uint64_t> key;
    for (int s = 0; s < (int)glsl::Stage::Compute; s++) {
        key.push_back((uint64_t)(uintptr_t)st[s]);
        key.push_back(st[s] ? (uint64_t)(uintptr_t)st[s]->program.get() : 0);
    }
    if (pp->composite && pp->compositeKey == key) return pp->composite.get();
    Ref<ProgramObj> c = new ProgramObj();
    c->composite = true;
    c->name = vs->name;
    c->linked = c->separable = true;
    c->program = vs->program;
    glsl::Reflection r = vs->refl;
    const glsl::LinkedProgram* lp[(int)glsl::Stage::Count] = {};
    r.stageMask = 0;
    for (int s = 0; s < (int)glsl::Stage::Compute; s++) {
        c->stagePrograms[s] = st[s];
        if (!st[s]) continue;
        c->stageLinked[s] = st[s]->program;
        lp[s] = st[s]->program.get();
        r.stageMask |= 1u << s;
    }
    if (ProgramObj* g = st[(int)glsl::Stage::Geometry]) {
        r.gsInputPrimitive = g->refl.gsInputPrimitive;
        r.gsOutputPrimitive = g->refl.gsOutputPrimitive;
        r.gsMaxVertices = g->refl.gsMaxVertices;
        r.gsInvocations = g->refl.gsInvocations;
    }
    if (ProgramObj* t = st[(int)glsl::Stage::TessControl]) r.tcsOutputVertices = t->refl.tcsOutputVertices;
    if (ProgramObj* t = st[(int)glsl::Stage::TessEval]) {
        r.tesPrimitiveMode = t->refl.tesPrimitiveMode;
        r.tesSpacing = t->refl.tesSpacing;
        r.tesOrder = t->refl.tesOrder;
        r.tesPointMode = t->refl.tesPointMode;
    }
    ProgramObj* last = st[(int)glsl::Stage::Geometry] ? st[(int)glsl::Stage::Geometry] : st[(int)glsl::Stage::TessEval] ? st[(int)glsl::Stage::TessEval] : vs;
    r.xfbVaryings = last->refl.xfbVaryings;
    r.writesLayer = last->refl.writesLayer;
    r.writesViewportIndex = last->refl.writesViewportIndex;
    ProgramObj* fs = st[(int)glsl::Stage::Fragment];
    r.fragOutputs = fs ? fs->refl.fragOutputs : std::vector<glsl::FragOutput>();
    r.usesFragDepth = fs && fs->refl.usesFragDepth;
    r.earlyFragmentTests = fs && fs->refl.earlyFragmentTests;
    r.blendSupport = fs ? fs->refl.blendSupport : 0;
    r.usesDiscard = fs && fs->refl.usesDiscard;
    r.usesFlatShading = fs && fs->refl.usesFlatShading;
    r.multipass = glsl::multipassInfo(lp);
    c->refl = std::move(r);
    pp->composite = c;
    pp->compositeKey = key;
    return c.get();
}

// Argument buffer address for a draw: a composite program gets a table of per-stage buffers.
static uint64_t drawArguments(Context* ctx, ProgramObj* prog, bool inert = false) {
    if (!prog->composite) return buildArguments(ctx, prog, false, inert);
    uint64_t addr[(int)glsl::Stage::Count] = {};
    for (int s = 0; s < (int)glsl::Stage::Count; s++) {
        ProgramObj* p = prog->stagePrograms[s].get();
        if (!p) continue;
        for (int o = 0; o < s; o++)
            if (prog->stagePrograms[o].get() == p) addr[s] = addr[o];
        if (!addr[s]) addr[s] = buildArguments(ctx, p, false, inert);
        if (!addr[s]) addr[s] = ctx->upload(16).gpu;
    }
    UploadAlloc t = ctx->upload(sizeof addr, 16);
    memcpy(t.cpu, addr, sizeof addr);
    return t.gpu;
}

// ---------------------------------------------------------------------------
// Multi-pass pipelines. Work whose output size only the GPU knows (geometry shader
// amplification, tessellation) runs as count kernel -> CPU readback + prefix sums ->
// write kernels, so the runtime knows exact primitive counts (transform feedback offsets,
// primitive queries, DrawTransformFeedback). Only transform feedback / primitive queries and
// tessellation feeding a geometry shader take the counted path; plain GS / tessellation
// rendering stays on the single-pass mesh pipelines.
// ---------------------------------------------------------------------------
namespace {

// Largest mesh grid width drawn at once (executeMultiPass drawMesh), and the most object payload memory a
// draw's grid may need (threadgroups x EntryPoint::payloadBytes).
constexpr uint32_t kMeshGridChunk = 65536;
constexpr size_t kMeshGridPayloadBytes = 64u << 20;

struct MPState {
    Context* ctx;
    ProgramObj* prog;
    const DrawParams* d;
    ProgramVariant* var;
    std::vector<uint32_t> ids;
    GLenum listMode;
    int vpp;
    uint32_t prims;
    // persistent scratch (survives the CPU waits between passes)
    id<MTLBuffer> tcsV = nil, tcsP = nil, counts = nil, tesV = nil, tesP = nil, gsRec = nil;
    uint64_t tcsVAddr = 0, tcsPAddr = 0;  // pass 1 records in upload memory (mpTessControl `transient`)
    uint64_t tessFactors = 0;             // Metal's tessellation factors (VariantKey::tessMetal)
    uint32_t itemCount = 0, tesPrimCount = 0;
    std::vector<uint32_t> offsets;  // uint4 per item, uploaded with the draw info
    uint32_t xfbPrimLimit[4] = {0, 0, 0, 0};
    uint32_t xfbStreamBase[4] = {0, 0, 0, 0};
    bool xfb = false;
    bool inert = false;  // pre-rasterization storage bound as copies (buildArguments `inert`)
    bool noStats = false;  // no pipeline statistics (ARB_post_depth_coverage pre-pass)
    // tessellation: the sub-draws (MGLDrawInfo::subDraws); empty: the draw `d` alone, its list in `ids`
    std::vector<glsl::MGLSubDraw> subDraws;
    uint32_t subDrawIndexType = 0;
    // varying spill of the mesh pass (EntryPoint::spillRecordBytes)
    uint64_t spillBuffer = 0, spillCounter = 0;
    id<MTLBuffer> spillCounterBuffer = nil;
    uint32_t spillCapacity = 0;
    // per phase (upload memory is retired by every CPU wait)
    uint64_t infoAddr = 0, argAddr = 0, sysAddr = 0;
    const glsl::MGLDrawInfo* info = nullptr;  // CPU view of infoAddr (copies for mesh grid chunks)
};

// Uploads the per-phase draw description (vertex ids, MGLDrawInfo, arguments, system values).
void mpUpload(MPState& s) {
    Context* ctx = s.ctx;
    const DrawParams& d = *s.d;
    const glsl::Reflection& refl = s.prog->refl;
    UploadAlloc idxUp = ctx->upload(std::max<size_t>(4, s.ids.size() * 4), 16);
    memcpy(idxUp.cpu, s.ids.data(), s.ids.size() * 4);
    UploadAlloc di = ctx->upload(sizeof(glsl::MGLDrawInfo), 16);
    glsl::MGLDrawInfo* info = (glsl::MGLDrawInfo*)di.cpu;
    memset(info, 0, sizeof *info);
    uint64_t serial = ctx->serial();
    fillVertexBindings(ctx, info);
    info->indexBuffer = idxUp.gpu;  // always a 32-bit list of vertex ids (base vertex not yet applied)
    info->indexType = 4;
    info->count = (uint32_t)s.ids.size();
    info->baseVertex = d.indexed ? d.baseVertex : 0;
    info->baseInstance = d.baseInstance;
    info->firstVertex = 0;
    info->instanceCount = d.instances;
    info->primitiveCount = s.prims;
    info->primitiveMode = s.listMode;
    info->verticesPerPrimitive = s.vpp;
    info->patchVertices = ctx->patchVertices;
    if (s.xfb) {
        TransformFeedbackObj* x = ctx->xfb.get();
        for (int i = 0; i < kMaxXfbBuffers; i++) {
            auto& bb = x->buffers[i];
            info->xfbBufferStream[i] = refl.multipass.xfbBufferStream[i];
            if (!bb.buffer || !bb.buffer->mtl) continue;
            info->xfbStride[i] = refl.multipass.xfbStride[i];
            info->xfbBuffer[i] = bb.buffer->gpuAddress() + bb.offset;
            GLsizeiptr range = bb.size ? bb.size : bb.buffer->size - bb.offset;
            info->xfbCapacity[i] = (uint32_t)std::max<GLsizeiptr>(0, range);
            bb.buffer->lastUse = bb.buffer->lastGpuWrite = serial;
        }
        info->xfbVertexBase = s.xfbStreamBase[0];
    }
    for (int k = 0; k < 4; k++) {
        info->xfbPrimLimit[k] = s.xfbPrimLimit[k];
        info->xfbStreamBase[k] = s.xfbStreamBase[k];
    }
    UploadAlloc counters = ctx->upload(64, 16);
    memset(counters.cpu, 0, 64);
    info->counters = counters.gpu;
    if (ctx->statsQueriesActive && !s.noStats)
        info->stats = mglStatsGpuSlot(ctx, refl.stageMask & (1u << (int)glsl::Stage::Geometry), refl.stageMask & (1u << (int)glsl::Stage::TessEval));
    info->tcsVertexOut = s.tcsV ? s.tcsV.gpuAddress : s.tcsVAddr;
    info->tcsPatchOut = s.tcsP ? s.tcsP.gpuAddress : s.tcsPAddr;
    info->itemCounts = s.counts ? s.counts.gpuAddress : 0;
    info->tesVertexOut = s.tesV ? s.tesV.gpuAddress : 0;
    info->tesPrimOut = s.tesP ? s.tesP.gpuAddress : 0;
    info->itemCount = s.itemCount;
    info->gsRecord = s.gsRec ? s.gsRec.gpuAddress : 0;
    info->tesPrimCount = s.tesPrimCount;
    info->spillBuffer = s.spillBuffer;
    info->spillCounter = s.spillCounter;
    info->spillCapacity = s.spillCapacity;
    info->tessFactors = s.tessFactors;
    if (!s.offsets.empty()) {
        UploadAlloc off = ctx->upload(s.offsets.size() * 4, 16);
        memcpy(off.cpu, s.offsets.data(), s.offsets.size() * 4);
        info->itemOffsets = off.gpu;
    }
    {  // tessellation's sub-draw table (one sub-draw unless batched; other kinds don't read it)
        glsl::MGLSubDraw one = {};
        one.drawID = (uint32_t)d.drawID;
        one.baseVertex = d.indexed ? d.baseVertex : 0;
        const glsl::MGLSubDraw* sd = s.subDraws.empty() ? &one : s.subDraws.data();
        size_t n = s.subDraws.empty() ? 1 : s.subDraws.size();
        UploadAlloc t = ctx->upload(n * sizeof(glsl::MGLSubDraw), 16);
        memcpy(t.cpu, sd, n * sizeof(glsl::MGLSubDraw));
        info->subDraws = t.gpu;
        info->subDrawCount = (uint32_t)n;
        info->subDrawIndexType = s.subDraws.empty() ? 0 : s.subDrawIndexType;
        // the GL index buffer is read on the GPU (by this command buffer: uploads follow budget splits)
        if (info->subDrawIndexType >= 2 && ctx->vao->elementBuffer) ctx->vao->elementBuffer->lastUse = serial;
    }
    failSafeDrawInfo(ctx, info);
    s.infoAddr = di.gpu;
    s.info = info;
    s.argAddr = drawArguments(ctx, s.prog, s.inert);
    if (!s.argAddr) s.argAddr = ctx->upload(16).gpu;
    s.sysAddr = buildSys(ctx, s.prog, d.drawID, d.baseVertex, d.baseInstance, d.first);
}

// Encodes one compute dispatch of kernel `name` in its own encoder (ordered after all prior work).
bool mpDispatch(MPState& s, const char* name, MTLSize grid, bool threadgroups) {
    Context* ctx = s.ctx;
    const glsl::EntryPoint* k = s.var->entry(name);
    id<MTLComputePipelineState> cps = k ? ctx->pipelines_->computePipeline(s.var, name) : nil;
    if (!cps) return false;
    if (grid.width == 0 || grid.height == 0 || grid.depth == 0) return true;
    id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
    [ce setComputePipelineState:cps];
    [ctx->computeArgTable setAddress:s.argAddr atIndex:glsl::kBufArgs];
    [ctx->computeArgTable setAddress:s.sysAddr atIndex:glsl::kBufSys];
    [ctx->computeArgTable setAddress:s.infoAddr atIndex:glsl::kBufStageIO];
    MTLSize tg = MTLSizeMake(std::max(1, k->threadsPerThreadgroup[0]), std::max(1, k->threadsPerThreadgroup[1]), std::max(1, k->threadsPerThreadgroup[2]));
    if (threadgroups) [ce dispatchThreadgroups:grid threadsPerThreadgroup:tg];
    else [ce dispatchThreads:grid threadsPerThreadgroup:tg];
    ctx->endEncoder();
    return true;
}

// Submits everything recorded so far and waits for it (count readback).
void mpWait(MPState& s) {
    Context* ctx = s.ctx;
    uint64_t serial = ctx->serial();
    ctx->waitForResource(serial);
}

id<MTLBuffer> mpScratch(MPState& s, size_t bytes) {
    id<MTLBuffer> b = s.ctx->dev->newBuffer(std::max<size_t>(16, bytes));
    return b;
}

// Pass 1 of tessellation: VS + TCS for every patch. `transient`: no CPU wait follows before the
// records are consumed, so they live in upload memory (retired with the command buffer).
bool mpTessControl(MPState& s, bool transient = false) {
    const glsl::Reflection& refl = s.prog->refl;
    const DrawParams& d = *s.d;
    int cpts = refl.multipass.tcsOutputVertices ? (int)refl.multipass.tcsOutputVertices : s.ctx->patchVertices;
    size_t patches = (size_t)s.prims * d.instances;
    size_t vBytes = patches * std::max(1, cpts) * std::max<uint32_t>(16, refl.multipass.tcsVertexOutBytes) + 16;
    size_t pBytes = patches * std::max<uint32_t>(32, refl.multipass.tcsPatchOutBytes) + 16;
    if (transient) {
        s.tcsVAddr = s.ctx->upload(vBytes, 16).gpu;
        s.tcsPAddr = s.ctx->upload(pBytes, 16).gpu;
    } else {
        s.tcsV = mpScratch(s, vBytes);
        s.tcsP = mpScratch(s, pBytes);
    }
    mpUpload(s);
    return mpDispatch(s, "mgl_tcs", MTLSizeMake(s.prims, d.instances, 1), true);
}

// The patches of a tessellation draw, or of all sub-draws of a multi-draw (one emulation pass), as
// MGLDrawInfo::subDraws. Indexed draws read the GL index buffer on the GPU when they can (no primitive
// restart, 16/32-bit indices in range); otherwise the CPU decomposes the lists into `ids`.
void tessSubDraws(MPState& s, const DrawParams* ds, int n) {
    Context* ctx = s.ctx;
    int pv = std::max(1, ctx->patchVertices);
    s.listMode = GL_PATCHES;
    s.vpp = pv;
    Buffer* ib = ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    const DrawParams& d0 = ds[0];
    uint32_t type = 1;  // non-indexed
    if (d0.indexed) {
        type = 0;
        bool restart = ctx->primitiveRestart || ctx->primitiveRestartFixed;
        if (ib && ib->mtl && !restart && (d0.indexType == GL_UNSIGNED_SHORT || d0.indexType == GL_UNSIGNED_INT)) {
            size_t isz = indexSize(d0.indexType);
            type = (uint32_t)isz;
            for (int i = 0; i < n && type; i++) {
                size_t off = (size_t)ds[i].indices;
                if (off % isz || off + (size_t)std::max(0, ds[i].count) * isz > (size_t)ib->size) type = 0;
            }
        }
    }
    uint64_t ibAddr = type >= 2 ? ib->gpuAddress() : 0;
    uint32_t total = 0;
    for (int i = 0; i < n; i++) {
        const DrawParams& d = ds[i];
        if (d.count <= 0) continue;
        glsl::MGLSubDraw sd = {};
        sd.drawID = (uint32_t)d.drawID;
        sd.baseVertex = d.indexed ? d.baseVertex : 0;
        uint32_t patches = (uint32_t)(d.count / pv);
        if (type == 0) {
            const uint8_t* src = nullptr;
            if (ib) {
                if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
                src = ib->cpu ? ib->cpu + (size_t)d.indices : bufferBytes(ctx, ib, (size_t)d.indices, (size_t)d.count * indexSize(d.indexType));
            } else {
                src = (const uint8_t*)d.indices;
            }
            size_t at = s.ids.size();
            decomposeToList(ctx, d, src, s.ids);
            patches = (uint32_t)((s.ids.size() - at) / pv);
            sd.first = (uint32_t)at;
        } else if (type == 1) {
            sd.first = (uint32_t)d.first;
        } else {
            sd.indices = ibAddr + (uint64_t)(size_t)d.indices;
        }
        if (!patches) continue;
        sd.firstPatch = total;
        total += patches;
        s.subDraws.push_back(sd);
    }
    if (type >= 2) ib->lastUse = ctx->serial();
    s.subDrawIndexType = type;
    s.prims = total;
}

// TES output records: count (+wait) and generate. Leaves tesPrimCount set.
bool mpTessRecords(MPState& s) {
    const glsl::Reflection& refl = s.prog->refl;
    const DrawParams& d = *s.d;
    s.itemCount = s.prims * d.instances;
    s.counts = mpScratch(s, (size_t)s.itemCount * 16);
    mpUpload(s);
    if (!mpDispatch(s, "mgl_tes_count", MTLSizeMake(s.itemCount, 1, 1), false)) return false;
    mpWait(s);
    const uint32_t* c = (const uint32_t*)bufferContents(s.counts);
    s.offsets.assign((size_t)s.itemCount * 4, 0);
    uint64_t prims = 0, verts = 0;
    uint32_t maxUnits = 0;
    for (uint32_t i = 0; i < s.itemCount; i++) {
        s.offsets[i * 4 + 0] = (uint32_t)prims;
        s.offsets[i * 4 + 1] = (uint32_t)verts;
        prims += c[i * 4 + 0];
        verts += c[i * 4 + 1];
        maxUnits = std::max(maxUnits, c[i * 4 + 2]);
    }
    int ipp = refl.tesPointMode ? 1 : (refl.tesPrimitiveMode == GL_ISOLINES ? 2 : 3);
    s.tesPrimCount = (uint32_t)prims;
    s.tesV = mpScratch(s, verts * std::max<uint32_t>(16, refl.multipass.tesVertexOutBytes));
    s.tesP = mpScratch(s, prims * ipp * 4);
    mpUpload(s);
    return mpDispatch(s, "mgl_tes_gen", MTLSizeMake(s.itemCount, maxUnits, 1), true);
}

// Primitives of each stream that still fit in the bound transform feedback buffers.
void mpXfbLimits(MPState& s, int outVpp) {
    Context* ctx = s.ctx;
    TransformFeedbackObj* x = ctx->xfb.get();
    const glsl::MultiPassInfo& mp = s.prog->refl.multipass;
    for (int st = 0; st < 4; st++) {
        s.xfbStreamBase[st] = x->written[st];
        uint64_t limit = s.xfb ? UINT32_MAX : 0;
        bool any = false;
        for (int b = 0; b < kMaxXfbBuffers; b++) {
            if (!(mp.xfbBufferMask & (1u << b)) || mp.xfbBufferStream[b] != (uint32_t)st || !mp.xfbStride[b]) continue;
            any = true;
            auto& bb = x->buffers[b];
            uint64_t cap = 0;
            if (bb.buffer && bb.buffer->mtl) {
                GLsizeiptr range = bb.size ? bb.size : bb.buffer->size - bb.offset;
                cap = (uint64_t)std::max<GLsizeiptr>(0, range) / mp.xfbStride[b];
            }
            uint64_t fit = cap > x->written[st] ? (cap - x->written[st]) / std::max(1, outVpp) : 0;
            limit = std::min(limit, fit);
        }
        if (!any) limit = s.xfb ? UINT32_MAX : 0;  // nothing captured from this stream: never overflows
        s.xfbPrimLimit[st] = (uint32_t)std::min<uint64_t>(limit, UINT32_MAX);
    }
}

// Records the outcome of a counted pass: transform feedback progress and primitive queries.
void mpFinishCounts(MPState& s, const uint64_t generated[4], int outVpp, bool countGenerated = true) {
    Context* ctx = s.ctx;
    TransformFeedbackObj* x = ctx->xfb.get();
    const glsl::MultiPassInfo& mp = s.prog->refl.multipass;
    for (int st = 0; st < 4; st++) {
        bool captured = false;
        for (int b = 0; b < kMaxXfbBuffers; b++)
            if ((mp.xfbBufferMask & (1u << b)) && mp.xfbBufferStream[b] == (uint32_t)st) captured = true;
        uint64_t written = s.xfb ? std::min<uint64_t>(generated[st], s.xfbPrimLimit[st]) : 0;
        if (s.xfb && !captured) written = generated[st];  // GL counts primitives of streams without buffers as written
        if (s.xfb && captured) x->written[st] += (uint32_t)(written * outVpp);
        mglQueryAddPrimitives(ctx, st, countGenerated ? generated[st] : 0, written, s.xfb, written < generated[st]);
    }
}

}  // namespace

static double tessItemsPerPatch(Context* ctx, ProgramObj* prog);

// Tessellation emulation passes and fused fast-path draws so far (mglTestTessCounters).
static uint64_t g_tessPasses = 0, g_tessFusedDraws = 0;

// Test hook: the longest GPU time of a command buffer completed since the last call (seconds).
extern "C" __attribute__((visibility("default"))) double mglTestMaxGpuSeconds(void) {
    return mgl::Device::get()->takeMaxGpuSeconds();
}

// ---------------------------------------------------------------------------
// Emulated geometry per command buffer (watchdog safety). The tessellation and geometry shader emulation can
// put unbounded work into one command buffer (an ocean of tessellated water, a shadow pass, ...), and the GPU
// watchdog kills command buffers that run for a few seconds ("CDM Kill timeout"). Draws are weighed in items
// (expected output triangles); once a command buffer holds kMeshWorkBudgetNs worth of them it is committed and
// the next draw (or the next chunk of a large draw) goes into a new one. The ns per item comes from the GPU
// times of completed command buffers that consisted mostly of such work; until then a conservative guess.
// ---------------------------------------------------------------------------
static constexpr double kMeshWorkBudgetNs = 250e6;
// Items per patch of the tessellation fast path (a classic vertex pipeline, but each vertex runs the VS for the
// whole patch and the TCS: cheaper than an emulated output triangle, not free).
static constexpr double kFusedPatchItems = 0.25;

static double meshWorkBudgetItems(Context* ctx) { return kMeshWorkBudgetNs / ctx->meshWorkNsPerItem; }

// Moves the count of a submitted command buffer to the pending list and learns from completed ones.
static void meshWorkPoll(Context* ctx) {
    uint64_t cur = ctx->serial();
    if (ctx->meshWorkSerial != cur) {
        if (ctx->meshWorkItems > 0 && ctx->meshWorkPending.size() < 64) ctx->meshWorkPending.push_back({ctx->meshWorkSerial, ctx->meshWorkItems});
        ctx->meshWorkSerial = cur;
        ctx->meshWorkItems = 0;
    }
    auto& pending = ctx->meshWorkPending;
    for (size_t i = 0; i < pending.size();) {
        double start, end;
        auto [serial, items] = pending[i];
        if (ctx->dev->gpuTimes(serial, &start, &end)) {
            // only command buffers mostly made of emulated geometry say something about its cost
            if (items >= 0.25 * meshWorkBudgetItems(ctx)) {
                double ns = (end - start) * 1e9 / items;
                double& est = ctx->meshWorkNsPerItem;
                est = ns > est ? ns : 0.9 * est + 0.1 * ns;  // up at once, down slowly
                est = std::clamp(est, 1.0, 1e6);
            }
        } else if (!ctx->dev->isComplete(serial)) {
            i++;
            continue;
        }
        pending.erase(pending.begin() + (long)i);
    }
}

// Before encoding a draw of `items`: commits the command buffer if the draw would take it over the budget.
static void meshWorkReserve(Context* ctx, double items) {
    meshWorkPoll(ctx);
    double budget = meshWorkBudgetItems(ctx);
    if (ctx->meshWorkItems > 0 && ctx->meshWorkItems + std::min(items, budget) > budget) {
        MGL_STAT("split:mesh-work-budget");
        ctx->flush();
        meshWorkPoll(ctx);
    }
}

// Test hook: the number of rasterizing tessellation passes (one per draw, or per batched multi-draw)
// and how many of them took the fused fast path.
extern "C" __attribute__((visibility("default"))) void mglTestTessCounters(uint64_t* passes, uint64_t* fused) {
    *passes = g_tessPasses;
    *fused = g_tessFusedDraws;
}

// Runs a draw through a multi-pass pipeline. Returns false if unsupported.
//   Classic + capture   : VS transform feedback kernel (rasterization is done by the caller)
//   MeshGeometry        : VS + GS on a mesh pipeline
//   Tessellation        : VS+TCS kernel, then TES on object+mesh (or TES records + GS on mesh)
//   XfbCapture          : counted capture/count passes for transform feedback and primitive queries
//   inertArgs: the stages' storage writes go to copies (ARB_post_depth_coverage: the draw's own vertex
//   processing has the side effects)
//   batch: the sub-draws of a multi-draw (tessellation without a geometry shader, nothing counted):
//   all of them in one emulation pass (`d` is the first)
static bool executeMultiPass(Context* ctx, ProgramObj* prog, const DrawParams& d, glsl::VariantKey vk, bool xfbCapture,
                             bool inertArgs = false, const DrawParams* batch = nullptr, int batchCount = 0) {
    const glsl::Reflection& refl = prog->refl;
    if (d.indirect) {
        static bool warned = false;
        if (!warned) MGL_WARN("indirect draws with geometry/tessellation/transform feedback not supported yet");
        warned = true;
        return false;
    }
    MPState s;
    s.ctx = ctx;
    s.prog = prog;
    s.d = &d;
    s.xfb = xfbCapture;
    s.inert = inertArgs;
    bool hasGS = refl.stageMask & (1u << (int)glsl::Stage::Geometry);
    bool hasTess = refl.stageMask & (1u << (int)glsl::Stage::TessEval);
    // --- primitive list ---
    if (vk.kind == glsl::PipelineKind::Tessellation && !hasGS) {
        tessSubDraws(s, batch ? batch : &d, batch ? batchCount : 1);
    } else {
        Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
        const uint8_t* src = nullptr;
        if (d.indexed) {
            if (ib) {
                if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
                src = ib->cpu ? ib->cpu + (size_t)d.indices : bufferBytes(ctx, ib, (size_t)d.indices, (size_t)d.count * indexSize(d.indexType));
            } else {
                src = (const uint8_t*)d.indices;
            }
        }
        s.listMode = listModeFor(d.mode);
        s.vpp = listVerts(s.listMode, ctx->patchVertices);
        decomposeToList(ctx, d, src, s.ids);
        s.prims = (uint32_t)(s.ids.size() / std::max(1, s.vpp));
    }
    if (!s.prims) return true;

    // Emulated geometry (tessellation without the fast path, geometry shaders, VS on mesh): weigh the draw
    // (expected output triangles) and keep its command buffer within the watchdog budget.
    double itemsPerColumn = 0;  // per grid column: patch / primitive x instances (x invocations)
    if ((vk.kind == glsl::PipelineKind::Tessellation && !vk.tessFused) || vk.kind == glsl::PipelineKind::MeshGeometry) {
        if (vk.kind == glsl::PipelineKind::Tessellation) itemsPerColumn = tessItemsPerPatch(ctx, prog) * d.instances;
        else itemsPerColumn = (double)d.instances * std::max(1, refl.gsInvocations) * std::max(1.0, refl.gsMaxVertices / 3.0);
        meshWorkReserve(ctx, itemsPerColumn * s.prims);
    } else if (vk.tessFused) {
        meshWorkReserve(ctx, kFusedPatchItems * d.instances * s.prims);
    }

    fillVertexFetch(ctx, prog, vk);
    vk.primitive = (uint8_t)s.listMode;
    s.var = ctx->pipelines_->variant(prog, vk);
    if (!s.var || !s.var->ok) return !vk.tessFused;
    ProgramVariant* var = s.var;
    uint64_t fsrAddr = vk.fsr ? fsrUpload(ctx) : 0;  // before any encoder
    uint64_t serialStart = ctx->serial();
    auto retire = [&]() {
        uint64_t sr = ctx->serial();
        for (id<MTLBuffer> b : {s.tcsV, s.tcsP, s.counts, s.tesV, s.tesP, s.gsRec})
            if (b) ctx->dev->releaseAfter(b, sr);
        (void)serialStart;
    };

    // --- tessellation fast path: VS + TCS + TES fused in a classic vertex function, 3 (quads: 6)
    // vertices per patch (VariantKey::tessFused) ---
    if (vk.tessFused) {
        mpUpload(s);
        id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
        if (!e) return true;
        RenderPipelineKey key;
        key.variant = var;
        fillColorKey(ctx, key, refl);
        key.rasterEnabled = true;
        id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
        if (!pso) return true;
        if (ctx->enc.pipeline != pso) {
            [e setRenderPipelineState:pso];
            ctx->enc.pipeline = pso;
        }
        applyDynamicState(ctx, e, GL_TRIANGLES, refl.writesViewportIndex);
        id<MTL4ArgumentTable> table = ctx->renderArgTable;
        [table setAddress:s.argAddr atIndex:glsl::kBufArgs];
        [table setAddress:s.sysAddr atIndex:glsl::kBufSys];
        [table setAddress:s.infoAddr atIndex:glsl::kBufStageIO];
        if (ctx->enc.fsTableSplit) {
            [e setArgumentTable:table atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = false;
        }
        mglQueryNotifyDraw(ctx, GL_NONE, 0, 0);  // visibility only (no primitive queries on this path)
        g_tessPasses++;
        g_tessFusedDraws++;
        NSUInteger vpp = refl.tesPrimitiveMode == GL_QUADS ? 6 : 3;
        [e drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:(NSUInteger)s.prims * vpp instanceCount:d.instances baseInstance:0];
        meshWorkPoll(ctx);
        ctx->meshWorkItems += kFusedPatchItems * d.instances * s.prims;
        return true;
    }

    // --- counted capture (transform feedback / primitive queries) ---
    if (vk.kind == glsl::PipelineKind::XfbCapture) {
        uint64_t generated[4] = {0, 0, 0, 0};
        if (!hasGS && !hasTess) {
            // VS only: one thread per vertex; counts are known on the CPU
            uint64_t prims = (uint64_t)s.prims * d.instances;
            mpXfbLimits(s, s.vpp);
            mpUpload(s);
            mpDispatch(s, "mgl_xfb", MTLSizeMake(s.ids.size() * d.instances, 1, 1), false);
            generated[0] = prims;
            mpFinishCounts(s, generated, s.vpp, false);
            return true;
        }
        int outVpp;
        if (hasGS) {
            uint32_t op = refl.gsOutputPrimitive;
            outVpp = op == GL_POINTS ? 1 : op == GL_LINE_STRIP ? 2 : 3;
        } else {
            outVpp = refl.tesPointMode ? 1 : (refl.tesPrimitiveMode == GL_ISOLINES ? 2 : 3);
        }
        mpXfbLimits(s, outVpp);
        if (hasTess) {
            if (!mpTessControl(s) || !mpTessRecords(s)) { retire(); return false; }
            if (!hasGS) {
                generated[0] = s.tesPrimCount;
                mpUpload(s);
                mpDispatch(s, "mgl_tes_xfb", MTLSizeMake(s.tesPrimCount, 1, 1), false);
                mpFinishCounts(s, generated, outVpp);
                retire();
                return true;
            }
        }
        // geometry shader: count, prefix sums, write
        uint32_t inv = (uint32_t)std::max(1, refl.gsInvocations);
        s.itemCount = (hasTess ? s.tesPrimCount : s.prims * d.instances) * inv;
        s.offsets.clear();
        s.counts = mpScratch(s, (size_t)s.itemCount * 16);
        // Record mode: the count pass also stores every item's captured primitives and a copy
        // pass places them, so the shaders run exactly once (side effects!). Re-executing the
        // GS in a write pass is the fallback when the records would be too large.
        const glsl::EntryPoint* copyEp = var->entry("mgl_gs_copy");
        size_t recBytes = copyEp ? (size_t)s.itemCount * copyEp->scratchWordsPerItem * 4 : 0;
        bool record = s.xfb && copyEp && recBytes <= ((size_t)64 << 20);
        if (record) s.gsRec = mpScratch(s, recBytes);
        mpUpload(s);
        if (!mpDispatch(s, "mgl_gs_count", MTLSizeMake(s.itemCount, 1, 1), false)) { retire(); return false; }
        mpWait(s);
        const uint32_t* c = (const uint32_t*)bufferContents(s.counts);
        s.offsets.assign((size_t)s.itemCount * 4, 0);
        for (uint32_t i = 0; i < s.itemCount; i++)
            for (int st = 0; st < 4; st++) {
                s.offsets[i * 4 + st] = (uint32_t)generated[st];
                generated[st] += c[i * 4 + st];
            }
        if (s.xfb) {
            mpUpload(s);
            mpDispatch(s, record ? "mgl_gs_copy" : "mgl_gs_write", MTLSizeMake(s.itemCount, 1, 1), false);
        }
        mpFinishCounts(s, generated, outVpp);
        retire();
        return true;
    }

    // --- transform feedback captured by the rasterizing vertex shader (VariantKey::xfbInVertex):
    // a non-indexed draw over the decomposed list, so side effects of the VS happen once ---
    if (vk.kind == glsl::PipelineKind::Classic && vk.xfbInVertex) {
        uint64_t generated[4] = {(uint64_t)s.prims * d.instances, 0, 0, 0};
        mpXfbLimits(s, s.vpp);
        mpUpload(s);
        id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
        if (!e) return true;
        RenderPipelineKey key;
        key.variant = var;
        fillColorKey(ctx, key, refl);
        key.rasterEnabled = true;
        id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
        if (!pso) return true;
        if (ctx->enc.pipeline != pso) {
            [e setRenderPipelineState:pso];
            ctx->enc.pipeline = pso;
        }
        applyDynamicState(ctx, e, s.listMode);
        id<MTL4ArgumentTable> table = ctx->renderArgTable;
        [table setAddress:s.argAddr atIndex:glsl::kBufArgs];
        [table setAddress:s.sysAddr atIndex:glsl::kBufSys];
        [table setAddress:s.infoAddr atIndex:glsl::kBufStageIO];
        if (ctx->enc.fsTableSplit) {
            [e setArgumentTable:table atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = false;
        }
        if (fsrAddr) [table setAddress:fsrAddr atIndex:glsl::kBufFsr];
        mglQueryNotifyDraw(ctx, GL_NONE, 0, 0);  // visibility only; primitive counts are added by the caller
        MTLPrimitiveType prim = s.listMode == GL_POINTS ? MTLPrimitiveTypePoint : s.listMode == GL_LINES ? MTLPrimitiveTypeLine
                                                                                                           : MTLPrimitiveTypeTriangle;
        [e drawPrimitives:prim vertexStart:0 vertexCount:s.ids.size() instanceCount:d.instances baseInstance:0];
        mpFinishCounts(s, generated, s.vpp, false);
        return true;
    }
    // --- tessellation pass 1: VS+TCS compute ---
    // Fragment inputs past Metal's mesh output budget travel through a spill buffer (one record per
    // emitted vertex). Sized for the worst case of the grid, capped; overflowing vertices read zeros.
    if (const glsl::EntryPoint* sm = var->entry(glsl::EntryPoint::Mesh); sm && sm->spillRecordBytes) {
        const uint64_t kCap = 64ull << 20;
        uint64_t worst = (vk.kind == glsl::PipelineKind::Tessellation)
                             ? kCap
                             : (uint64_t)s.prims * d.instances * std::max(1, refl.gsInvocations) * std::max<uint32_t>(1, sm->maxMeshVertices) *
                                   sm->spillRecordBytes;
        uint64_t bytes = std::max<uint64_t>(sm->spillRecordBytes, std::min(worst, kCap));
        id<MTLBuffer> sb = ctx->dev->newBuffer((size_t)bytes);
        ctx->dev->releaseAfter(sb, ctx->serial());
        id<MTLBuffer> sc = ctx->dev->newBuffer(16);
        memset(bufferContents(sc), 0, 16);
        ctx->dev->releaseAfter(sc, ctx->serial());
        s.spillBuffer = sb.gpuAddress;
        s.spillCounter = sc.gpuAddress;
        s.spillCounterBuffer = sc;
        s.spillCapacity = (uint32_t)(bytes / sm->spillRecordBytes);
    }
    if (vk.kind == glsl::PipelineKind::Tessellation && var->entry("mgl_tcs")) {
        // pass 1 as a compute kernel (otherwise VS + TCS run in the object stage of the TES pipeline)
        if (!mpTessControl(s, !hasGS)) { retire(); return false; }
        if (hasGS && !mpTessRecords(s)) { retire(); return false; }
    } else {
        mpUpload(s);
    }
    // --- mesh pass (GS or TES[+GS]) ---
    const glsl::EntryPoint* ms = var->entry(glsl::EntryPoint::Mesh);
    const glsl::EntryPoint* os = var->entry(glsl::EntryPoint::Object);
    if (!ms) { retire(); return false; }
    if (hasTess && hasGS && s.tesPrimCount == 0) { retire(); return true; }
    // ARB_post_depth_coverage (VariantKey::pdc, executePdc): one pre-pass for the whole draw. The rasterized
    // primitives come from the GPU here, so a draw that writes depth/stencil isn't split per primitive.
    bool pdc = vk.pdc && !ctx->rasterizerDiscard;
    uint64_t preInfoAddr = 0, preArgAddr = 0;
    const glsl::MGLDrawInfo* preInfo = nullptr;
    if (pdc) {  // the pre-pass's draw description: storage copies, no statistics (before any encoder)
        MPState pre = s;
        pre.inert = pre.noStats = true;
        mpUpload(pre);
        preInfoAddr = pre.infoAddr;
        preArgAddr = pre.argAddr;
        preInfo = pre.info;
    }
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) { retire(); return true; }
    RenderPipelineKey key;
    key.variant = var;
    fillColorKey(ctx, key, refl);
    key.rasterEnabled = !ctx->rasterizerDiscard;
    id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
    if (!pso) { retire(); return true; }
    GLenum rasterMode = hasGS ? refl.gsOutputPrimitive : hasTess ? GL_TRIANGLES : s.listMode;
    id<MTL4ArgumentTable> table = ctx->renderArgTable;
    MTLSize meshTG = MTLSizeMake(ms->threadsPerThreadgroup[0], ms->threadsPerThreadgroup[1], ms->threadsPerThreadgroup[2]);
    // Mesh grids are drawn in chunks along x (the patch / primitive index); each chunk binds a copy of the
    // draw info whose gridBase the shaders add to their x position. Apple GPUs don't cope with object grids
    // whose payloads add up to a few hundred MB: a tessellation draw of 2^18 patches with 736-byte payloads
    // (193 MB) takes 40 ms, one of 3 x 2^17 (289 MB) never finishes (the watchdog kills it or the GPU
    // faults). Batched multi-draws (a Sodium region) easily reach such sizes; chunks stay below
    // kMeshGridPayloadBytes and kMeshGridChunk threadgroups.
    // A draw larger than the command buffer budget continues in a new command buffer between chunks
    // (`restart` re-uploads the draw's description and binds everything again). Not with pass 1 records in
    // upload memory, spill buffers or a post-depth-coverage pre-pass, which belong to this command buffer.
    std::function<id<MTL4RenderCommandEncoder>()> restart;
    auto drawMesh = [&](id<MTL4RenderCommandEncoder> enc, const glsl::MGLDrawInfo* info, bool splittable) {
        MTLSize tg = MTLSizeMake(1, 1, 1);
        uint32_t n, y, z, chunk = kMeshGridChunk, per = 1;  // per: patches per object threadgroup (grid column)
        if (os) {
            tg = MTLSizeMake(os->threadsPerThreadgroup[0], os->threadsPerThreadgroup[1], os->threadsPerThreadgroup[2]);
            per = std::max<uint32_t>(1, os->patchesPerObjectGroup);
            n = (s.prims + per - 1) / per, y = (uint32_t)d.instances, z = 1;
            size_t perColumn = (size_t)std::max<uint32_t>(1, os->payloadBytes) * y;  // y object threadgroups per grid column
            chunk = (uint32_t)std::clamp<size_t>(kMeshGridPayloadBytes / perColumn, 1, kMeshGridChunk);
        } else {
            z = (uint32_t)std::max(1, refl.gsInvocations);
            n = hasTess ? s.tesPrimCount : s.prims;
            y = hasTess ? 1 : (uint32_t)d.instances;
        }
        // items per grid column of this grid (the TES records of tessellation + GS: one GS input primitive each)
        double perColumn = hasTess && !os ? (double)z * std::max(1.0, refl.gsMaxVertices / 3.0) : itemsPerColumn * per;
        if (perColumn > 0) chunk = (uint32_t)std::clamp(meshWorkBudgetItems(ctx) / perColumn, 1.0, (double)chunk);
        for (uint32_t base = 0; base < n; base += chunk) {
            uint32_t count = std::min(chunk, n - base);
            if (splittable && restart && base && ctx->meshWorkItems > 0 && ctx->meshWorkItems + perColumn * count > meshWorkBudgetItems(ctx)) {
                MGL_STAT("split:mesh-work-budget");
                enc = restart();
                if (!enc) return;
                info = s.info;
            }
            if (base) {
                UploadAlloc c = ctx->upload(sizeof(glsl::MGLDrawInfo), 16);
                memcpy(c.cpu, info, sizeof(glsl::MGLDrawInfo));
                ((glsl::MGLDrawInfo*)c.cpu)->gridBase = base * per;
                [table setAddress:c.gpu atIndex:glsl::kBufStageIO];
            }
            [enc drawMeshThreadgroups:MTLSizeMake(count, y, z) threadsPerObjectThreadgroup:tg threadsPerMeshThreadgroup:meshTG];
            meshWorkPoll(ctx);
            ctx->meshWorkItems += perColumn * count;
        }
    };
    id<MTLTexture> cov = nil;
    if (pdc) {
        PassDesc mainPass = ctx->curPass;
        cov = pdcCoverage(ctx, mainPass);
        id<MTLRenderPipelineState> prePso = ctx->pipelines_->renderPipeline(pdcPrepassKey(key));
        if (!cov || !prePso) { retire(); return true; }
        id<MTL4RenderCommandEncoder> pe = pdcBeginPrepass(ctx, mainPass, cov);
        if (!pe) { retire(); return true; }
        [pe setRenderPipelineState:prePso];
        ctx->enc.pipeline = prePso;
        applyDynamicState(ctx, pe, rasterMode, refl.writesViewportIndex, true);
        [table setAddress:preArgAddr ? preArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [table setAddress:s.sysAddr atIndex:glsl::kBufSys];
        [table setAddress:preInfoAddr atIndex:glsl::kBufStageIO];
        drawMesh(pe, preInfo, false);
        ctx->endPass();
        if (s.spillCounterBuffer) {  // the draw's vertices spill from the start again
            id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
            [ce fillBuffer:s.spillCounterBuffer range:NSMakeRange(0, 16) value:0];
            ctx->endEncoder();
        }
        e = ctx->renderEncoder();
        if (!e || !(ctx->curPass == mainPass)) { retire(); return true; }
    }
    auto bind = [&](id<MTL4RenderCommandEncoder> enc) {
        if (ctx->enc.pipeline != pso) {
            [enc setRenderPipelineState:pso];
            ctx->enc.pipeline = pso;
        }
        applyDynamicState(ctx, enc, rasterMode, refl.writesViewportIndex);
        [table setAddress:s.argAddr atIndex:glsl::kBufArgs];
        [table setAddress:s.sysAddr atIndex:glsl::kBufSys];
        [table setAddress:s.infoAddr atIndex:glsl::kBufStageIO];
        if (ctx->enc.fsTableSplit) {
            [enc setArgumentTable:table atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = false;
        }
        if (fsrAddr) [table setAddress:fsrAddr atIndex:glsl::kBufFsr];
        mglQueryNotifyDraw(ctx, GL_NONE, 0, 0);  // visibility only; primitive counts come from counted passes
    };
    bind(e);
    if (pdc) [table setAddress:pdcUpload(ctx, cov, 0, refl.writesLayer) atIndex:glsl::kBufPdc];
    if (!pdc && !s.tcsVAddr && !s.spillCounterBuffer)
        restart = [&]() -> id<MTL4RenderCommandEncoder> {
            ctx->flush();
            meshWorkPoll(ctx);
            mpUpload(s);
            if (vk.fsr) fsrAddr = fsrUpload(ctx);
            id<MTL4RenderCommandEncoder> enc = ctx->renderEncoder();
            if (enc) bind(enc);
            return enc;
        };
    if (hasTess) g_tessPasses++;
    drawMesh(e, s.info, true);
    retire();
    return true;
}

// ---------------------------------------------------------------------------
// Metal's tessellator (VariantKey::tessMetal). Metal 4 has no tessellation, but classic Metal's drawPatches does, and
// Apple's GL driver uses it: VS + TCS run in the mgl_tcs kernel (records and Metal's factors in upload memory) on
// the pre-pass command buffer of a tessellation segment (Context::openTessSegment), and the segment's render pass
// draws the patches with the TES as the post-tessellation vertex function. Consecutive tessellated draws into
// the same pass share a segment.
// ---------------------------------------------------------------------------
static bool tessMetalWanted(Context* ctx, ProgramObj* prog, const glsl::VariantKey& vk, const DrawParams& d) {
    const glsl::Reflection& r = prog->refl;
    if (!r.tessMetal || prog->composite || d.indirect || ctx->patchVertices < 1 || ctx->patchVertices > 32) return false;
    // nothing that needs the emulation's per-primitive work or a Metal 4 render pass
    if (vk.pdc || vk.fsr || ctx->rasterizerDiscard || ctx->statsQueriesActive || mglQueryVisibilityActive(ctx)) return false;
    if (ctx->polygonMode == GL_POINT) return false;
    // METALGLDRV_TESS_METAL=0: always the emulation (read per draw: tests switch it)
    return envFlag("METALGLDRV_TESS_METAL", true) && ctx->dev->classicQueue();
}

static uint64_t g_tessMetalDraws = 0;
extern "C" __attribute__((visibility("default"))) uint64_t mglTestTessMetalDraws(void) { return g_tessMetalDraws; }

// False: nothing was encoded and the emulation draws it.
static bool executeTessMetal(Context* ctx, ProgramObj* prog, const DrawParams& d, glsl::VariantKey vk, const DrawParams* batch, int batchCount) {
    const glsl::Reflection& refl = prog->refl;
    fillVertexFetch(ctx, prog, vk);
    vk.primitive = GL_PATCHES;
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return false;
    const glsl::EntryPoint* k = var->entry("mgl_tcs");
    id<MTLComputePipelineState> tcs = k ? ctx->pipelines_->classicComputePipeline(var, "mgl_tcs") : nil;
    if (!tcs) return false;
    bool quads = refl.tesPrimitiveMode == GL_QUADS;
    const glsl::MultiPassInfo& mp = refl.multipass;
    int cps = mp.tcsOutputVertices ? (int)mp.tcsOutputVertices : ctx->patchVertices;
    RenderPipelineKey key;
    key.variant = var;
    key.rasterEnabled = 1;
    key.tessMetal = 1;
    key.tessPartition = (uint8_t)(refl.tesSpacing == GL_FRACTIONAL_EVEN ? MTLTessellationPartitionModeFractionalEven
                                  : refl.tesSpacing == GL_FRACTIONAL_ODD ? MTLTessellationPartitionModeFractionalOdd
                                                                         : MTLTessellationPartitionModeInteger);
    key.tessWinding = (uint8_t)(refl.tesOrder == GL_CW ? MTLWindingClockwise : MTLWindingCounterClockwise);
    // Setting up the draw can record Metal 4 work (texture storage, feedback-loop copies), which ends the segment:
    // then once more in a new one.
    for (int attempt = 0; attempt < 2; attempt++) {
        PassDesc p;
        if (!ctx->resolvePass(p)) return true;
        Context::TessSegment& seg = ctx->tessSeg;
        if (!seg.open || !(seg.pass == p) || ctx->pendingClear.any())
            if (!ctx->openTessSegment(p)) return false;
        uint64_t segCount = seg.count;
        MPState s;
        s.ctx = ctx;
        s.prog = prog;
        s.d = &d;
        s.var = var;
        tessSubDraws(s, batch ? batch : &d, batch ? batchCount : 1);
        if (!s.prims) return true;
        size_t patches = (size_t)s.prims * d.instances;
        s.tcsVAddr = ctx->upload(patches * std::max(1, cps) * std::max<uint32_t>(16, mp.tcsVertexOutBytes) + 16, 16).gpu;
        s.tcsPAddr = ctx->upload(patches * std::max<uint32_t>(32, mp.tcsPatchOutBytes) + 16, 16).gpu;
        s.tessFactors = ctx->upload(patches * (quads ? 12 : 8), 16).gpu;
        mpUpload(s);
        if (!seg.open || seg.count != segCount) continue;
        NSUInteger argOff = 0, sysOff = 0, infoOff = 0, tfOff = 0;
        id<MTLBuffer> argBuf = ctx->uploadBuffer(s.argAddr, &argOff), sysBuf = ctx->uploadBuffer(s.sysAddr, &sysOff);
        id<MTLBuffer> infoBuf = ctx->uploadBuffer(s.infoAddr, &infoOff), tfBuf = ctx->uploadBuffer(s.tessFactors, &tfOff);
        if (!argBuf || !sysBuf || !infoBuf || !tfBuf) return false;
        fillColorKey(ctx, key, refl);
        key.depthFormat = p.depth.tex ? (uint32_t)p.depth.tex.pixelFormat : 0;
        key.stencilFormat = p.stencil.tex ? (uint32_t)p.stencil.tex.pixelFormat : 0;
        id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
        if (!pso) return false;
        MGL_STAT("draw:tess-metal");
        g_tessMetalDraws++;
        id<MTLComputeCommandEncoder> ce = seg.compute;
        [ce setComputePipelineState:tcs];
        [ce setBuffer:argBuf offset:argOff atIndex:glsl::kBufArgs];
        [ce setBuffer:sysBuf offset:sysOff atIndex:glsl::kBufSys];
        [ce setBuffer:infoBuf offset:infoOff atIndex:glsl::kBufStageIO];
        uint32_t per = std::max<uint32_t>(1, k->patchesPerObjectGroup);
        [ce dispatchThreadgroups:MTLSizeMake((s.prims + per - 1) / per, (NSUInteger)d.instances, 1)
           threadsPerThreadgroup:MTLSizeMake((NSUInteger)k->threadsPerThreadgroup[0], 1, 1)];
        id<MTLRenderCommandEncoder> re = seg.render;
        if (ctx->enc.pipeline != pso) {
            [re setRenderPipelineState:pso];
            ctx->enc.pipeline = pso;
        }
        applyDynamicState(ctx, re, GL_TRIANGLES);
        [re setVertexBuffer:argBuf offset:argOff atIndex:glsl::kBufArgs];
        [re setVertexBuffer:sysBuf offset:sysOff atIndex:glsl::kBufSys];
        [re setVertexBuffer:infoBuf offset:infoOff atIndex:glsl::kBufStageIO];
        [re setFragmentBuffer:argBuf offset:argOff atIndex:glsl::kBufArgs];
        [re setFragmentBuffer:sysBuf offset:sysOff atIndex:glsl::kBufSys];
        [re setFragmentBuffer:infoBuf offset:infoOff atIndex:glsl::kBufStageIO];
        [re setTessellationFactorBuffer:tfBuf offset:tfOff instanceStride:(NSUInteger)s.prims * (quads ? 12 : 8)];
        [re drawPatches:(NSUInteger)cps patchStart:0 patchCount:s.prims patchIndexBuffer:nil patchIndexBufferOffset:0
              instanceCount:(NSUInteger)d.instances baseInstance:0];
        return true;
    }
    return false;
}

// Fast path for indexed triangle/line lists from a buffer: rotate the whole index buffer
// once (per content version) and draw from the same offset in the rotated copy. This
// serves every draw into a shared index buffer (Sodium) with one allocation. Buffers without a
// CPU copy (sparse) would need a readback and a copy as large as the whole buffer (1 GB of
// mostly uncommitted virtual range in the sparse buffer CTS): they rotate the drawn range only.
static bool rotatedWholeBuffer(Context* ctx, const DrawParams& d, IndexInfo& out) {
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    if (!ib || !ib->cpu || (d.mode != GL_TRIANGLES && d.mode != GL_LINES)) return false;
    if (d.indexType != GL_UNSIGNED_SHORT && d.indexType != GL_UNSIGNED_INT) return false;
    if (ctx->primitiveRestart || ctx->primitiveRestartFixed) return false;
    if (ib->mapped && (ib->mapAccess & GL_MAP_PERSISTENT_BIT) && (ib->mapAccess & GL_MAP_WRITE_BIT)) return false;
    size_t isz = indexSize(d.indexType);
    size_t off = (size_t)d.indices;
    if (off % isz || off + (size_t)d.count * isz > (size_t)ib->size) return false;
    uint32_t vpp = d.mode == GL_TRIANGLES ? 3 : 2;
    uint32_t phase = (uint32_t)((off / isz) % vpp);
    uint32_t layout = (uint32_t)isz | vpp << 8 | phase << 16;
    if (!ib->rotated || ib->rotatedVersion != ib->contentVersion || ib->rotatedGpuWrite != ib->lastGpuWrite ||
        ib->rotatedLayout != layout || ib->rotated.length < (NSUInteger)ib->size) {
        if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
        if (ib->rotated) ctx->dev->releaseAfter(ib->rotated, ib->rotatedLastUse);
        ib->rotated = ctx->dev->newBuffer((size_t)ib->size);
        if (!ib->rotated) return false;
        uint8_t* dst = (uint8_t*)bufferContents(ib->rotated);
        memcpy(dst, ib->cpu, (size_t)ib->size);
        size_t n = (size_t)ib->size / isz;
        auto rot = [&](auto* v) {
            for (size_t i = phase; i + vpp <= n; i += vpp) {
                if (vpp == 3) {
                    auto a = v[i], b = v[i + 1], c = v[i + 2];
                    v[i] = c;  // rotation keeps the winding
                    v[i + 1] = a;
                    v[i + 2] = b;
                } else {
                    std::swap(v[i], v[i + 1]);
                }
            }
        };
        if (isz == 2) rot((uint16_t*)dst);
        else rot((uint32_t*)dst);
        ib->rotatedVersion = ib->contentVersion;
        ib->rotatedGpuWrite = ib->lastGpuWrite;
        ib->rotatedLayout = layout;
    }
    ib->rotatedLastUse = ctx->serial();
    out = IndexInfo();
    out.prim = vpp == 2 ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
    out.type = isz == 2 ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32;
    out.count = (uint32_t)(d.count / vpp * vpp);
    out.generated = true;
    out.addr = ib->rotated.gpuAddress + off;
    out.length = out.count * isz;
    return true;
}

// Rotated index lists for flat shading, cached per context; ~Context drops a context's
// (releaseDrawCaches) so its buffers don't outlive it.
struct RotEntry {
    IndexInfo info;
    uint32_t minIdx, maxIdx;
    id<MTLBuffer> buffer;
};
struct RotCache {
    std::unordered_map<std::string, RotEntry> entries;
    size_t bytes = 0;
};
static std::unordered_map<Context*, RotCache> g_rotCaches;

void mgl::releaseDrawCaches(Context* ctx) {
    if (auto c = g_pdcCoverage.find(ctx); c != g_pdcCoverage.end()) {
        if (c->second) ctx->dev->releaseAfter(c->second, ctx->dev->lastReservedSerial());
        g_pdcCoverage.erase(c);
    }
    auto it = g_rotCaches.find(ctx);
    if (it == g_rotCaches.end()) return;
    uint64_t s = ctx->dev->lastReservedSerial();
    for (auto& [k, e] : it->second.entries) ctx->dev->releaseAfter(e.buffer, s);
    g_rotCaches.erase(it);
}

// Builds (or reuses from cache) a rotated 32-bit index list for flat shading.
static bool prepareRotatedIndices(Context* ctx, const DrawParams& d, IndexInfo& out, uint32_t* minIdx, uint32_t* maxIdx) {
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    const uint8_t* src = nullptr;
    if (d.indexed) {
        if (ib) {
            if (!ib->mtl || (size_t)d.indices + (size_t)d.count * indexSize(d.indexType) > (size_t)ib->size) return false;
            if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
            src = ib->cpu ? ib->cpu + (size_t)d.indices : bufferBytes(ctx, ib, (size_t)d.indices, (size_t)d.count * indexSize(d.indexType));
        } else {
            src = (const uint8_t*)d.indices;
        }
    }
    // Cache key: only buffer-backed indices whose content can't change behind our back.
    bool cacheable = ib && !(ib->mapped && (ib->mapAccess & GL_MAP_PERSISTENT_BIT) && (ib->mapAccess & GL_MAP_WRITE_BIT));
    // GPU writes (shaders, copies into GPU-only buffers) change lastGpuWrite, not contentVersion.
    char key[160];
    snprintf(key, sizeof key, "%p:%u:%llu:%llu:%zu:%d:%x:%x:%d:%u", (void*)ib, ib ? ib->generation : 0,
             (unsigned long long)(ib ? ib->contentVersion : 0), (unsigned long long)(ib ? ib->lastGpuWrite : 0), (size_t)d.indices,
             d.count, d.indexType, d.mode, d.first, ctx->primitiveRestart || ctx->primitiveRestartFixed ? ctx->primitiveRestartIndex : 0xdead);
    RotCache& rc = g_rotCaches[ctx];
    auto& cache = rc.entries;
    if (cacheable) {
        auto it = cache.find(key);
        if (it != cache.end()) {
            out = it->second.info;
            *minIdx = it->second.minIdx;
            *maxIdx = it->second.maxIdx;
            return true;
        }
    }
    std::vector<uint32_t> ids;
    decomposeToList(ctx, d, src, ids);
    GLenum lm = dropAdjacency(listModeFor(d.mode), ids);
    int vpp = listVerts(lm, ctx->patchVertices);
    size_t prims = ids.size() / vpp;
    ids.resize(prims * vpp);
    uint32_t mn = UINT32_MAX, mx = 0;
    for (size_t p = 0; p < prims; p++) {
        uint32_t* v = &ids[p * vpp];
        if (vpp == 3) {
            uint32_t a = v[0], b = v[1], c = v[2];
            v[0] = c;  // rotation keeps the winding
            v[1] = a;
            v[2] = b;
        } else if (vpp == 2) {
            std::swap(v[0], v[1]);
        }
        for (int k = 0; k < vpp; k++) {
            mn = std::min(mn, v[k]);
            mx = std::max(mx, v[k]);
        }
    }
    out = IndexInfo();
    out.prim = vpp == 2 ? MTLPrimitiveTypeLine : MTLPrimitiveTypeTriangle;
    out.type = MTLIndexTypeUInt32;
    out.count = (uint32_t)ids.size();
    out.generated = true;
    out.length = ids.size() * 4;
    *minIdx = mn == UINT32_MAX ? 0 : mn + (d.indexed ? d.baseVertex : 0);
    *maxIdx = mx + (d.indexed ? d.baseVertex : 0);
    if (ids.empty()) return true;
    if (cacheable) {
        id<MTLBuffer> buf = ctx->dev->newBuffer(ids.size() * 4);
        memcpy(bufferContents(buf), ids.data(), ids.size() * 4);
        out.addr = buf.gpuAddress;
        if (cache.size() > 512 || rc.bytes > (32u << 20)) {
            for (auto& [k, e] : cache) ctx->dev->releaseAfter(e.buffer, ctx->serial());
            cache.clear();
            rc.bytes = 0;
        }
        rc.bytes += ids.size() * 4;
        cache[key] = {out, *minIdx, *maxIdx, buf};
    } else {
        UploadAlloc u = ctx->upload(ids.size() * 4, 16);
        memcpy(u.cpu, ids.data(), ids.size() * 4);
        out.addr = u.gpu;
    }
    return true;
}

// ---------------------------------------------------------------------------
// pipeline statistics & CPU-resolved indirect draws
// ---------------------------------------------------------------------------
// Primitives assembled from n vertices of one restart-delimited run.
static uint64_t primitivesOf(GLenum mode, uint64_t n, int patchVertices) {
    switch (mode) {
    case GL_POINTS: return n;
    case GL_LINES: return n / 2;
    case GL_LINE_STRIP: return n > 1 ? n - 1 : 0;
    case GL_LINE_LOOP: return n > 2 ? n : n == 2 ? 1 : 0;
    case GL_TRIANGLES: return n / 3;
    case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: return n > 2 ? n - 2 : 0;
    case GL_QUADS: return n / 4;
    case GL_QUAD_STRIP: return n >= 4 ? (n - 2) / 2 : 0;
    case GL_POLYGON: return n >= 3 ? 1 : 0;
    case GL_LINES_ADJACENCY: return n / 4;
    case GL_LINE_STRIP_ADJACENCY: return n > 3 ? n - 3 : 0;
    case GL_TRIANGLES_ADJACENCY: return n / 6;
    case GL_TRIANGLE_STRIP_ADJACENCY: return n >= 6 ? (n - 4) / 2 : 0;
    case GL_PATCHES: return patchVertices > 0 ? n / (uint64_t)patchVertices : 0;
    default: return 0;
    }
}

// Vertices (restart indices excluded) and primitives of one instance of a direct draw.
static void countDrawPrimitives(Context* ctx, const DrawParams& d, uint64_t* verts, uint64_t* prims) {
    bool restart = d.indexed && (ctx->primitiveRestart || ctx->primitiveRestartFixed);
    const uint8_t* src = nullptr;
    if (restart) {
        Buffer* ib = ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
        if (ib) {
            if ((size_t)d.indices + (size_t)d.count * indexSize(d.indexType) <= (size_t)ib->size && ib->cpu) {
                if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
                src = ib->cpu + (size_t)d.indices;
            }
        } else {
            src = (const uint8_t*)d.indices;
        }
    }
    if (!src) {
        *verts = (uint64_t)d.count;
        *prims = primitivesOf(d.mode, (uint64_t)d.count, ctx->patchVertices);
        return;
    }
    uint32_t restartIdx = ctx->primitiveRestartFixed ? (d.indexType == GL_UNSIGNED_BYTE ? 0xff : d.indexType == GL_UNSIGNED_SHORT ? 0xffff : 0xffffffffu)
                                                      : ctx->primitiveRestartIndex;
    uint64_t v = 0, p = 0, run = 0;
    for (GLsizei i = 0; i < d.count; i++) {
        if (readIndex(src, d.indexType, i) == restartIdx) {
            p += primitivesOf(d.mode, run, ctx->patchVertices);
            run = 0;
            continue;
        }
        v++;
        run++;
    }
    p += primitivesOf(d.mode, run, ctx->patchVertices);
    *verts = v;
    *prims = p;
}

static void countDrawStatistics(Context* ctx, const DrawParams& d, const glsl::Reflection& refl) {
    uint64_t v, p;
    countDrawPrimitives(ctx, d, &v, &p);
    uint64_t inst = (uint64_t)std::max(0, d.instances);
    bool hasGS = refl.stageMask & (1u << (int)glsl::Stage::Geometry);
    bool hasTess = refl.stageMask & (1u << (int)glsl::Stage::TessEval);
    DrawStats st;
    st.vertices = v * inst;
    st.primitives = p * inst;
    st.vsInvocations = v * inst;
    if (hasTess) st.patches = p * inst;  // TES invocations / tessellated primitives are counted on the GPU
    if (hasGS) st.gsInvocations = p * inst * (uint64_t)std::max(1, refl.gsInvocations);  // emitted primitives: GPU
    if (!hasGS && !hasTess) st.clipPrimitives = p * inst;
    st.fsEstimate = ctx->rasterizerDiscard ? 0 : p * inst;
    mglStatsCountDraw(ctx, st);
}

// Reads an indirect draw command on the CPU (waiting for GPU writes to the buffer) and
// turns it into the equivalent direct draw.
static bool resolveIndirectDraw(Context* ctx, const DrawParams& d, DrawParams& out) {
    Buffer* b = d.indirect;
    size_t size = d.indexed ? 20 : 16;
    if (!b || !b->mtl || (size_t)d.indirectOffset + size > (size_t)b->size) return false;
    uint32_t c[5] = {};
    readBuffer(ctx, b, (GLintptr)d.indirectOffset, (GLsizeiptr)size, c);
    out = d;
    out.indirect = nullptr;
    out.indirectOffset = 0;
    out.count = (GLsizei)c[0];
    out.instances = (GLsizei)c[1];
    if (d.indexed) {
        out.indices = (const void*)((uintptr_t)c[2] * indexSize(d.indexType));
        out.baseVertex = (GLint)c[3];
        out.baseInstance = c[4];
    } else {
        out.first = (GLint)c[2];
        out.baseInstance = c[3];
    }
    return true;
}

// Indirect draws Metal's indirect commands can't express (or that need CPU-side work) are
// executed as direct draws after reading the command on the CPU.
static bool indirectNeedsCpu(Context* ctx, const DrawParams& d, ProgramObj* prog) {
    switch (d.mode) {
    case GL_POINTS: case GL_LINES: case GL_LINE_STRIP: case GL_TRIANGLES: case GL_TRIANGLE_STRIP: break;
    default: return true;  // fans, loops, quads, adjacency, patches
    }
    if (ctx->statsQueriesActive) return true;
    const glsl::Reflection& r = prog->refl;
    if (r.stageMask & ((1u << (int)glsl::Stage::Geometry) | (1u << (int)glsl::Stage::TessEval))) return true;
    if (ctx->xfb && ctx->xfb->active && !ctx->xfb->paused) return true;
    if (d.indexed) {
        if (d.indexType == GL_UNSIGNED_BYTE) return true;
        uint32_t allOnes = d.indexType == GL_UNSIGNED_SHORT ? 0xffffu : 0xffffffffu;
        if (ctx->primitiveRestart && !ctx->primitiveRestartFixed && ctx->primitiveRestartIndex != allOnes) return true;
    }
    // flat shading with GL's last-vertex provoking convention needs rotated primitives
    ProgramObj* f = ctx->activeProgramForStage(glsl::Stage::Fragment);
    const glsl::Reflection& fr = f ? f->refl : r;
    if (fr.usesFlatShading && ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION && d.mode != GL_POINTS) return true;
    // ARB_post_depth_coverage pre-passes (executePdc) and draws split per primitive
    if (ctx->superStrict() && fr.postDepthCoverage) return true;
    return false;
}

// ---------------------------------------------------------------------------
// main draw
// ---------------------------------------------------------------------------
static bool validateDraw(Context* ctx, GLenum& mode) {
    if (!validMode(mode)) {
        ctx->setError(GL_INVALID_ENUM, "invalid draw mode 0x%x", mode);
        return false;
    }
    if ((mode == GL_QUADS || mode == GL_QUAD_STRIP || mode == GL_POLYGON) && ctx->coreProfile) {
        ctx->setError(GL_INVALID_ENUM);
        return false;
    }
    if (!ctx->vao || (ctx->coreProfile && ctx->vao == ctx->defaultVao)) {
        ctx->setError(GL_INVALID_OPERATION, "no VAO bound");
        return false;
    }
    Framebuffer* fb = ctx->drawFb.get();
    if (!fb || framebufferStatus(ctx, fb) != GL_FRAMEBUFFER_COMPLETE) {
        ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION, "%s", framebufferIncompleteReason(ctx, fb));
        return false;
    }
    if (ctx->depthTest || ctx->stencilTest) ctx->useDefaultDepthStencil(fb);
    if (ctx->pipeline && !ctx->program) {
        ProgramPipelineObj* pp = ctx->pipeline.get();
        std::string err = pp->stages[(int)glsl::Stage::Vertex] || pp->stages[(int)glsl::Stage::Fragment] ? pipelineValidationError(pp) : "";
        if (!err.empty() && err != "pipeline has no vertex or compute program") {
            ctx->setError(GL_INVALID_OPERATION, "%s", err.c_str());
            return false;
        }
        if (!pp->stages[(int)glsl::Stage::Vertex]) return false;  // missing stages: undefined rendering, no error
    }
    ProgramObj* prog = mglPipelineDrawProgram(ctx);
    if (ctx->advancedBlendUsed)
        if (const char* err = advancedBlendError(ctx, prog)) {
            ctx->setError(GL_INVALID_OPERATION, "%s", err);
            return false;
        }
    if (!prog && ctx->coreProfile) return false;  // no program/pipeline: undefined rendering, not an error
    if (prog && (prog->refl.stageMask & (1u << (int)glsl::Stage::Mesh))) {
        ctx->setError(GL_INVALID_OPERATION, "programs with a mesh shader draw with glDrawMeshTasks*EXT");
        return false;
    }
    bool hasTess = prog && (prog->refl.stageMask & (1u << (int)glsl::Stage::TessEval));
    if (mode == GL_PATCHES && !hasTess) {
        ctx->setError(GL_INVALID_OPERATION, "GL_PATCHES draw without a tessellation evaluation shader (program %u, stages 0x%x)",
                      prog ? prog->name : 0, prog ? prog->refl.stageMask : 0);
        return false;
    }
    // METALGLDRV_IMPLICIT_PATCHES=1 (the Fabric mod sets it): a draw whose primitives have
    // GL_PATCH_VERTICES vertices (triangles with 3, lines with 2, points with 1) while
    // tessellation is active runs as patches, as Apple's and NVIDIA's drivers do. GL says
    // INVALID_OPERATION; Iris + Sodium draw terrain this way with tessellation shader packs.
    // (Read only on this would-be-error path, so it costs nothing in conformant use.)
    if (hasTess && mode != GL_PATCHES && envFlag("METALGLDRV_IMPLICIT_PATCHES", false)) {
        int n = mode == GL_TRIANGLES ? 3 : mode == GL_LINES ? 2 : mode == GL_POINTS ? 1 : 0;
        if (n && n == ctx->patchVertices) {
            static bool logged = false;
            if (!logged) MGL_INFO("tessellation active with draw mode 0x%x: drawing %d-vertex patches (METALGLDRV_IMPLICIT_PATCHES)", mode, n);
            logged = true;
            mode = GL_PATCHES;
        }
    }
    if (hasTess && mode != GL_PATCHES) {
        ctx->setError(GL_INVALID_OPERATION, "program %u has tessellation shaders (stages 0x%x) but the draw mode is 0x%x, not GL_PATCHES",
                      prog->name, prog->refl.stageMask, mode);
        return false;
    }
    if (prog) {
        // GL 4.6 11.3.1: the primitives reaching the geometry shader must match its input type.
        const glsl::Reflection& r = prog->refl;
        bool hasGS = r.stageMask & (1u << (int)glsl::Stage::Geometry);
        auto cls = [](GLenum m) -> int {  // 0 points, 1 lines, 2 lines adj, 3 triangles, 4 triangles adj, -1 other
            switch (m) {
            case GL_POINTS: return 0;
            case GL_LINES: case GL_LINE_STRIP: case GL_LINE_LOOP: return 1;
            case GL_LINES_ADJACENCY: case GL_LINE_STRIP_ADJACENCY: return 2;
            case GL_TRIANGLES: case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: return 3;
            case GL_TRIANGLES_ADJACENCY: case GL_TRIANGLE_STRIP_ADJACENCY: return 4;
            default: return -1;
            }
        };
        int tesOut = -1;
        if (hasTess) tesOut = r.tesPointMode ? 0 : r.tesPrimitiveMode == GL_ISOLINES ? 1 : 3;
        if (hasGS) {
            int in = cls(r.gsInputPrimitive);
            int got = hasTess ? tesOut : cls(mode);
            if (in >= 0 && got != in) {
                ctx->setError(GL_INVALID_OPERATION, "draw primitives do not match the geometry shader input");
                return false;
            }
        }
        // GL 4.6 13.3.1: with transform feedback active, the captured primitive type must match.
        TransformFeedbackObj* x = ctx->xfb.get();
        if (x && x->active && !x->paused) {
            int out;
            if (hasGS) out = r.gsOutputPrimitive == GL_POINTS ? 0 : r.gsOutputPrimitive == GL_LINE_STRIP ? 1 : 3;
            else if (hasTess) out = tesOut;
            else out = cls(mode);
            if (out == 2) out = 1;
            if (out == 4) out = 3;
            int want = x->primitiveMode == GL_POINTS ? 0 : x->primitiveMode == GL_LINES ? 1 : 3;
            if (out != want) {
                ctx->setError(GL_INVALID_OPERATION, "primitive type does not match the transform feedback mode");
                return false;
            }
        }
    }
    // mapped non-persistent buffers in use
    for (int b = 0; b < kMaxVertexBindings; b++) {
        Buffer* buf = ctx->vao->bindings[b].buffer.get();
        if (buf && buf->mapped && !(buf->mapAccess & GL_MAP_PERSISTENT_BIT)) {
            ctx->setError(GL_INVALID_OPERATION, "vertex buffer is mapped");
            return false;
        }
    }
    if (ctx->vao->elementBuffer && ctx->vao->elementBuffer->mapped && !(ctx->vao->elementBuffer->mapAccess & GL_MAP_PERSISTENT_BIT)) {
        ctx->setError(GL_INVALID_OPERATION, "index buffer is mapped");
        return false;
    }
    return true;
}

// Whether the vertex shader may write memory (SSBOs, images, atomic counters).
static bool vertexSideEffects(const glsl::Reflection& r) {
    const uint32_t vs = 1u << (int)glsl::Stage::Vertex;
    for (const auto& b : r.storageBlocks)
        if (b.stageMask & vs) return true;
    for (const auto& a : r.atomicCounterBuffers)
        if (a.stageMask & vs) return true;
    for (const auto& s : r.slots)
        if (s.kind == glsl::SlotKind::Image && s.index < r.uniforms.size() && (r.uniforms[s.index].stageMask & vs)) return true;
    return false;
}

static bool programHasSideEffects(const glsl::Reflection& r) {
    if (!r.storageBlocks.empty() || !r.atomicCounterBuffers.empty()) return true;
    for (auto& sl : r.slots)
        if (sl.kind == glsl::SlotKind::Image) return true;
    return false;
}

// batch: all sub-draws of a multi-draw that tessBatchable() accepted (`d` is the first), drawn
// in one emulation pass.
static void executeDrawImpl(Context* ctx, const DrawParams& d, const DrawParams* batch, int batchCount);
static void executeDraw(Context* ctx, const DrawParams& d, const DrawParams* batch = nullptr, int batchCount = 0) {
    executeDrawImpl(ctx, d, batch, batchCount);
    flushImageWritebacks(ctx);
}

// Outer / inner levels the tessellation domain uses.
static void tessDomainLevels(const glsl::Reflection& r, int& nOuter, int& nInner) {
    nOuter = r.tesPrimitiveMode == GL_TRIANGLES ? 3 : r.tesPrimitiveMode == GL_QUADS ? 4 : 2;
    nInner = r.tesPrimitiveMode == GL_TRIANGLES ? 1 : r.tesPrimitiveMode == GL_QUADS ? 2 : 0;
}

// The draw's tessellation levels the domain uses (outer at 0..3, inner at 4..5): Reflection::tessLevels
// evaluated with the current uniform values (in float, like the TCS), or GL_PATCH_DEFAULT_*_LEVEL without a TCS.
// False when the TCS computes them in a way the CPU can't follow (per patch).
static bool evalTessLevels(Context* ctx, ProgramObj* prog, float lv[6]) {
    const glsl::Reflection& r = prog->refl;
    int nOuter, nInner;
    tessDomainLevels(r, nOuter, nInner);
    if (!(r.stageMask & (1u << (int)glsl::Stage::TessControl))) {
        memcpy(lv, ctx->patchDefaultOuter, 4 * sizeof(float));
        memcpy(lv + 4, ctx->patchDefaultInner, 2 * sizeof(float));
        return true;
    }
    const std::vector<uint8_t>& u = prog->uniformData;
    auto load = [&](uint32_t off, void* dst) {
        if ((size_t)off + 4 > u.size()) return false;
        memcpy(dst, u.data() + off, 4);
        return true;
    };
    for (int i = 0; i < 6; i++) {
        lv[i] = 1.0f;
        if ((i < 4 && i >= nOuter) || (i >= 4 && i - 4 >= nInner)) continue;
        if (r.tessLevels[i].empty()) return false;
        float st[16];
        int n = 0;
        for (const glsl::TessLevelOp& op : r.tessLevels[i]) {
            using K = glsl::TessLevelOp;
            int need = op.kind == K::Clamp ? 3 : op.kind >= K::Add ? 2 : op.kind == K::Neg ? 1 : 0;
            if (n < need || (need == 0 && n >= 16)) return false;
            switch (op.kind) {
            case K::Const: st[n++] = op.value; break;
            case K::Float: if (!load(op.offset, &st[n++])) return false; break;
            case K::Int: { int32_t v; if (!load(op.offset, &v)) return false; st[n++] = (float)v; break; }
            case K::UInt: { uint32_t v; if (!load(op.offset, &v)) return false; st[n++] = (float)v; break; }
            case K::Neg: st[n - 1] = -st[n - 1]; break;
            case K::Add: st[n - 2] = st[n - 2] + st[n - 1]; n--; break;
            case K::Sub: st[n - 2] = st[n - 2] - st[n - 1]; n--; break;
            case K::Mul: st[n - 2] = st[n - 2] * st[n - 1]; n--; break;
            case K::Div: st[n - 2] = st[n - 2] / st[n - 1]; n--; break;
            case K::Min: st[n - 2] = fminf(st[n - 2], st[n - 1]); n--; break;
            case K::Max: st[n - 2] = fmaxf(st[n - 2], st[n - 1]); n--; break;
            case K::Clamp: st[n - 3] = fminf(fmaxf(st[n - 3], st[n - 2]), st[n - 1]); n -= 2; break;
            }
        }
        if (n != 1) return false;
        lv[i] = st[0];
    }
    return true;
}

// The levels make every patch exactly one primitive at the domain corners, as mgl_tess_setup's `special` case:
// the outer levels the domain uses in (0, 1] (not culled, one segment with equal and fractional_odd spacing) and
// its inner levels <= 1; fractional_even clamps levels to 2, so it never qualifies.
static bool tessLevelsTrivial(Context* ctx, ProgramObj* prog) {
    const glsl::Reflection& r = prog->refl;
    if (r.tesSpacing == GL_FRACTIONAL_EVEN || r.tesPrimitiveMode == GL_ISOLINES) return false;
    int nOuter, nInner;
    tessDomainLevels(r, nOuter, nInner);
    float lv[6];
    if (!evalTessLevels(ctx, prog, lv)) return false;
    for (int i = 0; i < nOuter; i++)
        if (!(lv[i] > 0.0f && lv[i] <= 1.0f)) return false;
    for (int i = 0; i < nInner; i++)
        if (!(lv[4 + i] <= 1.0f)) return false;
    return true;
}

// Expected output primitives per patch (the emulation's GPU work, for the command buffer budget): from the
// levels when the CPU knows them (segments per spacing as in mgl_tess_level), else a guess for levels of ~6.
static double tessItemsPerPatch(Context* ctx, ProgramObj* prog) {
    const glsl::Reflection& r = prog->refl;
    float lv[6];
    if (!evalTessLevels(ctx, prog, lv)) return 64;
    int nOuter, nInner;
    tessDomainLevels(r, nOuter, nInner);
    auto segments = [&](float x) {
        if (!(x > 0.0f)) return 1.0f;
        if (r.tesSpacing == GL_FRACTIONAL_EVEN) return 2.0f * std::ceil(std::clamp(x, 2.0f, 64.0f) / 2.0f);
        if (r.tesSpacing == GL_FRACTIONAL_ODD) return 2.0f * std::floor(std::ceil(std::clamp(x, 1.0f, 63.0f)) / 2.0f) + 1.0f;
        return std::ceil(std::clamp(x, 1.0f, 64.0f));
    };
    if (r.tesPrimitiveMode == GL_ISOLINES) return std::max(1.0f, segments(lv[0]) * segments(lv[1]));
    float m = 1;
    for (int i = 0; i < nOuter; i++) m = std::max(m, segments(lv[i]));
    for (int i = 0; i < nInner; i++) m = std::max(m, segments(lv[4 + i]));
    return r.tesPrimitiveMode == GL_TRIANGLES ? 1.5 * m * m : 2.0 * m * m;
}

// Whether a tessellation draw takes the fused fast path (VariantKey::tessFused): the program qualifies
// (Reflection::tessFusable; not a program pipeline composite), patches of at most 4 vertices, nothing that
// needs the emulation's per-primitive work (pipeline statistics, rasterizer discard, coarse shading,
// post-depth-coverage pre-passes), and either
//   - levels that make every patch one primitive at the domain corners (the same output as the emulation), or
//   - a TES that is affine in gl_TessCoord (Reflection::tesAffine), at any levels: the tessellated patch covers
//     the same triangle(s) with the same interpolated values as its corners alone (up to rounding); the
//     fused vertex function drops the patches GL culls (an outer level <= 0 or NaN). Not in the strict profiles,
//     which want the tessellated primitives themselves, not an equivalent up to rounding.
// METALGLDRV_TESS_FAST=0 disables it.
static bool tessFusedWanted(Context* ctx, ProgramObj* prog, const glsl::VariantKey& vk) {
    const glsl::Reflection& r = prog->refl;
    if (!r.tessFusable || prog->composite || ctx->patchVertices < 1 || ctx->patchVertices > 4) return false;
    if (vk.pdc || vk.fsr || ctx->rasterizerDiscard || ctx->statsQueriesActive) return false;
    if (!envFlag("METALGLDRV_TESS_FAST", true)) return false;
    if (tessLevelsTrivial(ctx, prog)) return true;
    // affine: the same filled triangles at any levels (polygon modes would draw the sub-triangles' edges or
    // vertices); the fused vertex function culls patches by their outer levels itself
    return r.tesAffine && ctx->polygonMode == GL_FILL && !ctx->strict();
}

// Whether the sub-draws of a multi-draw can share one emulation pass: tessellation without a geometry
// shader, with nothing counted per draw (transform feedback, primitive queries) and no
// ARB_post_depth_coverage pre-passes (super strict). GL runs the sub-draws with the same program and state, and their
// patches are independent; gl_DrawID, gl_BaseVertex and gl_PrimitiveID come from the sub-draw table.
// METALGLDRV_TESS_BATCH=0 draws them one by one.
static bool tessBatchable(Context* ctx) {
    ProgramObj* prog = mglPipelineDrawProgram(ctx);
    if (!prog || !prog->program) return false;
    uint32_t m = prog->refl.stageMask;
    if (!(m & (1u << (int)glsl::Stage::TessEval)) || (m & (1u << (int)glsl::Stage::Geometry))) return false;
    if (ctx->xfb && ctx->xfb->active && !ctx->xfb->paused) return false;
    if (mglQueryPrimitivesActive(ctx)) return false;
    if (ctx->superStrict()) {
        ProgramObj* f = ctx->pipeline && !ctx->program ? ctx->pipeline->stages[(int)glsl::Stage::Fragment].get() : prog;
        if (f && f->refl.postDepthCoverage) return false;
    }
    return envFlag("METALGLDRV_TESS_BATCH", true);
}

static void executeDrawImpl(Context* ctx, const DrawParams& d, const DrawParams* batch, int batchCount) {
    MGL_STAT("draw");
    ProgramObj* prog = mglPipelineDrawProgram(ctx);
    // Compatibility profile: fixed-function emulation (prog == null) and gl_* state uniforms.
    if (!ctx->coreProfile && !(prog && prog->composite)) prog = mglCompatPrepareDraw(ctx, prog);
    if (!prog) return;
    if (!prog->program) return;  // last successfully linked executable
    if (!mglCheckDrawConditional(ctx)) return;
    // METALGLDRV_PROFILE_PASSES: counted when the draw is done, i.e. in the pass it was encoded in.
    struct PassDrawCount {
        Context* c;
        GLuint program;
        ~PassDrawCount() {
            if (!c) return;
            c->passDraws_++;
            if (!c->passFirstProgram_) c->passFirstProgram_ = program;
        }
    } passDrawCount{g_profilePasses ? ctx : nullptr, ctx->program ? ctx->program->name : prog->name};
    mgl::debugCaptureBeforeDraw(ctx, prog);
    if (d.indirect && indirectNeedsCpu(ctx, d, prog)) {
        DrawParams direct;
        if (resolveIndirectDraw(ctx, d, direct)) executeDraw(ctx, direct);
        return;
    }
    if (d.count <= 0 && !d.indirect && !batch) return;
    if (d.instances <= 0 && !d.indirect) return;
    const glsl::Reflection& refl = prog->refl;
    if (ctx->statsQueriesActive && !d.indirect) {
        if (batch) for (int i = 0; i < batchCount; i++) countDrawStatistics(ctx, batch[i], refl);
        else countDrawStatistics(ctx, d, refl);
    }
    bool hasGS = refl.stageMask & (1u << (int)glsl::Stage::Geometry);
    bool hasTess = refl.stageMask & (1u << (int)glsl::Stage::TessEval);
    glsl::VariantKey vk;
    vk.kind = hasTess ? glsl::PipelineKind::Tessellation : hasGS ? glsl::PipelineKind::MeshGeometry : glsl::PipelineKind::Classic;
    // more fragment inputs than a classic vertex function can pass: the vertex shader runs on a mesh stage
    if (vk.kind == glsl::PipelineKind::Classic && refl.vsMeshPassthrough && !d.indirect) vk.kind = glsl::PipelineKind::MeshGeometry;
    vk.primitive = (uint8_t)d.mode;
    // Classic vertex functions only depend on whether the mode is points (point size with gl_Layer; see
    // emitVertex): one variant for the other modes, the triangles one glLinkProgram precompiles.
    if (vk.kind == glsl::PipelineKind::Classic && d.mode != GL_POINTS) vk.primitive = GL_TRIANGLES;
    vk.flatshadeLastVertex = ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION;
    vk.rasterDiscard = ctx->rasterizerDiscard;
    vk.sampleShading = ctx->sampleShading && ctx->minSampleShading > 0;
    vk.sampleMask = sampleMaskActive(ctx);
    vk.sparseImageSlots = sparseImageSlots(ctx, prog);
    msArraySlots(ctx, prog, vk);
    vk.advancedBlend = advancedBlendKey(ctx);
    {
        ProgramObj* f = ctx->pipeline && !ctx->program ? ctx->pipeline->stages[(int)glsl::Stage::Fragment].get() : prog;
        vk.source1FromLocation1 = f && source1FromLocation1(ctx, f->refl);
    }
    if (ctx->strict()) {
        ProgramObj* f = ctx->pipeline && !ctx->program ? ctx->pipeline->stages[(int)glsl::Stage::Fragment].get() : prog;
        vk.fsr = fsrWanted(ctx, refl.writesPrimitiveShadingRate, f ? &f->refl : nullptr);
    }
    // ARB_post_depth_coverage (super strict): a depth/stencil pre-pass for the coverage (executePdc)
    PdcInfo pdc;
    if (ctx->superStrict() && !ctx->rasterizerDiscard) {
        ProgramObj* f = ctx->pipeline && !ctx->program ? ctx->pipeline->stages[(int)glsl::Stage::Fragment].get() : prog;
        if (f && f->refl.postDepthCoverage) pdc = pdcInfo(ctx);
    }
    vk.pdc = pdc.tests;
    if (hasTess) {  // METALGLDRV_TESS_PASS1=compute: VS + TCS in a compute pass even when the object stage could run them
        const char* p1 = getenv("METALGLDRV_TESS_PASS1");
        vk.tessComputePass = p1 && !strcmp(p1, "compute");
        vk.patchVertices = (uint8_t)std::clamp(ctx->patchVertices, 0, 32);
    }
    bool xfbActive = ctx->xfb && ctx->xfb->active && !ctx->xfb->paused;
    bool primQueries = mglQueryPrimitivesActive(ctx);
    if (vk.kind != glsl::PipelineKind::Classic) {
        // GS / tessellation: counted capture pass for transform feedback and primitive queries,
        // then the rasterizing mesh pipeline.
        bool captured = false;
        if ((xfbActive || primQueries) && !d.indirect) {
            glsl::VariantKey xk = vk;
            xk.kind = glsl::PipelineKind::XfbCapture;
            xk.fsr = false;
            xk.pdc = false;
            captured = executeMultiPass(ctx, prog, d, xk, xfbActive);
        }
        // Tessellation that makes every patch one primitive at the domain corners: the fused vertex
        // pipeline (falls back to the emulation if its variant doesn't build).
        if (vk.kind == glsl::PipelineKind::Tessellation && !xfbActive && !primQueries && tessFusedWanted(ctx, prog, vk)) {
            glsl::VariantKey fk = vk;
            fk.tessFused = (uint8_t)ctx->patchVertices;
            if (executeMultiPass(ctx, prog, d, fk, false, false, batch, batchCount)) return;
        }
        if (vk.kind == glsl::PipelineKind::Tessellation && !xfbActive && !primQueries && tessMetalWanted(ctx, prog, vk, d)) {
            glsl::VariantKey hk = vk;
            hk.tessMetal = true;
            if (executeTessMetal(ctx, prog, d, hk, batch, batchCount)) return;
        }
        // With rasterizer discard the mesh pass only matters for shader side effects, which the
        // capture pass already produced; otherwise it runs with rasterization disabled.
        if (!(ctx->rasterizerDiscard && captured && xfbActive)) executeMultiPass(ctx, prog, d, vk, false, false, batch, batchCount);
        return;
    }
    if (primQueries && !d.indirect) {
        uint64_t n = d.count;
        mglQueryAddPrimitives(ctx, 0, mglPrimitivesFor(d.mode, n) * (uint64_t)std::max(0, d.instances), 0, false, false);
    }
    if (xfbActive && !ctx->rasterizerDiscard && !d.indirect && vertexSideEffects(refl) && !vk.pdc) {
        // Capture from the rasterizing vertex shader so its side effects happen once.
        glsl::VariantKey xk = vk;
        xk.xfbInVertex = true;
        executeMultiPass(ctx, prog, d, xk, true);
        return;
    }
    if (xfbActive) {
        glsl::VariantKey xk = vk;
        xk.kind = glsl::PipelineKind::XfbCapture;
        xk.fsr = false;
        xk.pdc = false;
        // with a pre-pass, the draw's own vertex shader has the side effects
        executeMultiPass(ctx, prog, d, xk, true, vk.pdc && vertexSideEffects(refl));
        // the capture kernel already ran the vertex shader once per vertex (side effects included)
        if (ctx->rasterizerDiscard) return;
    }
    // With rasterization discarded the draw only matters for vertex-stage side effects
    // (storage buffer / image / atomic counter writes); run it without a fragment stage then.
    if (ctx->rasterizerDiscard && !programHasSideEffects(refl)) return;
    if (envFlag("METALGLDRV_DUMP_LAYOUT")) {
        // One line per (program, layout) for diagnosing vertex input issues in the field.
        static std::set<std::string> seen;
        std::string desc;
        for (const auto& a : prog->refl.attributes) {
            char buf[256];
            if (a.location < 0 || a.location >= kMaxVertexAttribs) continue;
            const VertexAttrib& va = ctx->vao->attribs[a.location];
            const VertexBinding& vb = ctx->vao->bindings[va.binding];
            snprintf(buf, sizeof buf, " %s@%d(t%x)=%s[%x x%d n%d i%d off%u b%u stride%d boff%lld div%u]", a.name.c_str(), a.location, a.type,
                     va.enabled ? "on" : "off", va.type, va.size, va.normalized, va.integer, va.relativeOffset, va.binding, vb.stride,
                     (long long)vb.offset, vb.divisor);
            desc += buf;
        }
        std::string k = std::to_string(prog->name) + desc;
        if (seen.insert(k).second)
            MGL_INFO("layout prog %u mode %x indexed %d type %x flat %d:%s", prog->name, d.mode, d.indexed, d.indexType,
                     refl.usesFlatShading, desc.c_str());
    }
    vk.manualFetch = wantManualFetch(ctx, prog);
    if (vk.pdc && pdc.writes && d.instances > 1 && !vk.manualFetch && pdcNeedsManualFetch(ctx)) vk.manualFetch = true;
    if (vk.manualFetch) fillVertexFetch(ctx, prog, vk);
    else vk.attribSignFlip = attribSignFlip(ctx, prog);
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return;
    // Program pipeline whose fragment stage comes from a different separable program (or none).
    ProgramObj* fsProg = prog;
    ProgramVariant* fsVar = nullptr;
    bool mixedFs = false;
    if (ctx->pipeline && !ctx->program) {
        ProgramObj* f = ctx->pipeline->stages[(int)glsl::Stage::Fragment].get();
        if (f != prog) {
            mixedFs = true;
            fsProg = f;
            if (f) {
                glsl::VariantKey fk;
                fk.kind = glsl::PipelineKind::Classic;
                fk.primitive = vk.primitive;
                fk.flatshadeLastVertex = vk.flatshadeLastVertex;
                fk.advancedBlend = vk.advancedBlend;
                fk.source1FromLocation1 = vk.source1FromLocation1;
                fk.fsr = vk.fsr;
                fk.pdc = vk.pdc;
                fsVar = ctx->pipelines_->variant(f, fk);
                if (!fsVar || !fsVar->ok) return;
            }
        }
    }
    const glsl::Reflection& frefl = fsProg ? fsProg->refl : refl;
    // Cull FRONT_AND_BACK discards all polygons.
    if (ctx->cullFace && ctx->cullMode == GL_FRONT_AND_BACK && isTriangleMode(d.mode)) return;

    IndexInfo ii;
    uint32_t minIdx = 0, maxIdx = 0;
    // GL's default provoking vertex is the LAST vertex; Metal uses the first. When the
    // program has flat varyings, rotate every primitive so Metal's first is GL's last.
    bool rotate = (frefl.usesFlatShading || (vk.fsr && refl.writesPrimitiveShadingRate)) && ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION && !d.indirect &&
                  d.mode != GL_POINTS && d.mode != GL_PATCHES;
    if (rotate) {
        // The vertex range is only needed for CPU-side vertex copies (client arrays / repacks).
        bool needRange = false;
        if (!vk.manualFetch)
            for (int l = 0; l < kMaxVertexAttribs; l++) {
                const VertexAttrib& va = ctx->vao->attribs[l];
                if (!va.enabled) continue;
                const VertexBinding& vb = ctx->vao->bindings[va.binding];
                if (va.clientPointer || (vb.stride & 3) || (vb.offset & 3) || (va.relativeOffset & 3)) needRange = true;
            }
        if (needRange || !rotatedWholeBuffer(ctx, d, ii))
            if (!prepareRotatedIndices(ctx, d, ii, &minIdx, &maxIdx)) return;
    } else if (!d.indirect && !prepareIndices(ctx, d, ii, &minIdx, &maxIdx)) {
        return;
    }
    if (d.indirect) {
        switch (d.mode) {
        case GL_POINTS: ii.prim = MTLPrimitiveTypePoint; break;
        case GL_LINES: ii.prim = MTLPrimitiveTypeLine; break;
        case GL_LINE_STRIP: ii.prim = MTLPrimitiveTypeLineStrip; break;
        case GL_TRIANGLE_STRIP: ii.prim = MTLPrimitiveTypeTriangleStrip; break;
        default: ii.prim = MTLPrimitiveTypeTriangle; break;
        }
        if (d.mode == GL_TRIANGLE_FAN || d.mode == GL_LINE_LOOP) {
            static bool warned = false;
            if (!warned) MGL_WARN("indirect fan/loop draws not supported yet");
            warned = true;
            return;
        }
    }
    if (!d.indirect && (d.indexed || ii.generated) && ii.count == 0) return;

    // ARB_post_depth_coverage draws that write depth/stencil are split per primitive (executePdc)
    PdcSplit pdcSplit;
    if (vk.pdc && pdc.writes && !pdcPrimitives(ctx, d, pdcSplit.ids, pdcSplit.prims, pdcSplit.vpp)) return;
    // Vertex data read on the CPU (client arrays, Metal-incompatible layouts that get
    // repacked) must be synchronized *before* encoding starts: a GPU wait flushes the
    // command buffer, which would end the render encoder and retire upload memory.
    if (ctx->vao && !vk.manualFetch) {
        for (int l = 0; l < kMaxVertexAttribs; l++) {
            const VertexAttrib& va = ctx->vao->attribs[l];
            if (!va.enabled) continue;
            const VertexBinding& vb = ctx->vao->bindings[va.binding];
            bool cpuRead = (vb.stride & 3) || (vb.offset & 3) || (va.relativeOffset & 3);
            Buffer* b = vb.buffer.get();
            if (cpuRead && b && b->lastGpuWrite && !ctx->dev->isComplete(b->lastGpuWrite)) ctx->waitForResource(b->lastGpuWrite);
        }
    }
    // Resolving bindings can allocate/migrate texture storage (a copy that ends the render
    // pass), so build the argument buffer before acquiring the render encoder.
    uint64_t argAddr = buildArguments(ctx, prog, false);
    uint64_t fsArgAddr = mixedFs && fsProg ? buildArguments(ctx, fsProg, false) : 0;
    uint64_t fsrAddr = vk.fsr ? fsrUpload(ctx) : 0;
    if (vk.pdc) {
        executePdc(ctx, prog, var, fsProg, fsVar, mixedFs, frefl, d, ii, minIdx, maxIdx, argAddr, fsArgAddr, pdc.writes ? &pdcSplit : nullptr);
        return;
    }
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) return;
    id<MTL4ArgumentTable> table = ctx->renderArgTable;
    RenderPipelineKey key;
    key.variant = var;
    if (mixedFs) {
        key.fsVariant = fsVar;
        key.fsVariantUid = fsVar ? fsVar->uid : ~0ull;
    }
    fillColorKey(ctx, key, frefl);
    key.rasterEnabled = !ctx->rasterizerDiscard;
    // Point size is always written; only layered rendering (gl_Layer from the vertex shader) needs a class.
    key.topologyClass = (uint8_t)(refl.writesLayer ? topologyClass(d.mode) : MTLPrimitiveTopologyClassUnspecified);
    if (!setupVertexInput(ctx, prog, key, table, d, minIdx, maxIdx)) return;
    if (ctx->renderEnc != e) {
        // Should not happen (see above); never encode into an ended encoder.
        MGL_ERR("render encoder changed during draw setup; dropping draw");
        return;
    }
    static const int dumpStateProg = envInt("METALGLDRV_DUMP_STATE", 0);
    if (dumpStateProg && (int)prog->name == dumpStateProg) {
        // Field debugging: full draw state of one program, logged once per distinct state.
        static std::set<std::string> seen;
        char b[512];
        std::string st;
        const PassDesc& p = ctx->curPass;
        snprintf(b, sizeof b, "pass %dx%d samples %d;", p.width, p.height, p.samples);
        st += b;
        for (int i = 0; i < kMaxDrawBuffers; i++) {
            if (!p.color[i].tex && !key.colorFormats[i]) continue;
            snprintf(b, sizeof b, " c%d[fmt %d keyfmt %u blend %d (%d,%d,%d,%d eq %d,%d) mask %x]", i, p.color[i].tex ? (int)p.color[i].tex.pixelFormat : -1,
                     key.colorFormats[i], key.blend[i].enabled, key.blend[i].srcRGB, key.blend[i].dstRGB, key.blend[i].srcA, key.blend[i].dstA,
                     key.blend[i].eqRGB, key.blend[i].eqA, key.blend[i].writeMask);
            st += b;
        }
        snprintf(b, sizeof b, " depth[fmt %d test %d func %x write %d] stencil[%d] cull %d/%x front %x clamp %d offset %d(%g,%g) a2c %d drawbuf0 %x vp %g,%g,%g,%g scissor %d [%d %d %d %d]",
                 p.depth.tex ? (int)p.depth.tex.pixelFormat : -1, ctx->depthTest, ctx->depthFunc, ctx->depthWrite, ctx->stencilTest, ctx->cullFace,
                 ctx->cullMode, ctx->frontFace, ctx->depthClamp, ctx->polygonOffsetFill, ctx->polygonOffsetFactor, ctx->polygonOffsetUnits,
                 key.alphaToCoverage, ctx->drawFb ? ctx->drawFb->drawBuffers[0] : 0, ctx->viewport[0][0], ctx->viewport[0][1], ctx->viewport[0][2],
                 ctx->viewport[0][3], ctx->scissorEnabled & 1, ctx->scissor[0][0], ctx->scissor[0][1], ctx->scissor[0][2], ctx->scissor[0][3]);
        st += b;
        if (seen.insert(st).second) MGL_INFO("draw state prog %u mode %x count %d: %s", prog->name, d.mode, d.count, st.c_str());
    }
    id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
    if (!pso) return;
    if (ctx->enc.pipeline != pso) {
        [e setRenderPipelineState:pso];
        ctx->enc.pipeline = pso;
    }
    applyDynamicState(ctx, e, d.mode, refl.writesViewportIndex);
    [table setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
    uint64_t sysAddr = buildSys(ctx, prog, d.drawID, d.baseVertex, d.baseInstance, d.first, d.indexed);
    [table setAddress:sysAddr atIndex:glsl::kBufSys];
    if (mixedFs) {
        // The fragment stage reads its own program's arguments.
        if (!ctx->enc.fsTableSplit) {
            [e setArgumentTable:ctx->renderArgTableFS atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = true;
        }
        [ctx->renderArgTableFS setAddress:fsArgAddr ? fsArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [ctx->renderArgTableFS setAddress:fsProg ? buildSys(ctx, fsProg, d.drawID, d.baseVertex, d.baseInstance, d.first, d.indexed) : sysAddr
                                  atIndex:glsl::kBufSys];
    } else if (ctx->enc.fsTableSplit) {
        [e setArgumentTable:table atStages:MTLRenderStageFragment];
        ctx->enc.fsTableSplit = false;
    }
    if (fsrAddr) [(mixedFs ? ctx->renderArgTableFS : table) setAddress:fsrAddr atIndex:glsl::kBufFsr];
    mglQueryNotifyDraw(ctx, d.mode, d.indirect ? 0 : (d.indexed || ii.generated ? (d.indexed ? d.count : d.count) : d.count), d.instances);

    if (d.indirect) {
        uint64_t ind = d.indirect->gpuAddress() + d.indirectOffset;
        d.indirect->lastUse = ctx->serial();
        if (d.indexed) {
            Buffer* ib = ctx->vao->elementBuffer.get();
            if (!ib || !ib->mtl) return;
            if (d.indexType == GL_UNSIGNED_BYTE) {
                static bool warned = false;
                if (!warned) MGL_WARN("indirect draws with GL_UNSIGNED_BYTE indices not supported");
                warned = true;
                return;
            }
            ib->lastUse = ctx->serial();
            [e drawIndexedPrimitives:ii.prim indexType:d.indexType == GL_UNSIGNED_SHORT ? MTLIndexTypeUInt16 : MTLIndexTypeUInt32
                         indexBuffer:ib->gpuAddress() indexBufferLength:ib->size indirectBuffer:ind];
        } else {
            [e drawPrimitives:ii.prim indirectBuffer:ind];
        }
        return;
    }
    if (d.indexed || ii.generated) {
        // For CPU-generated lists from non-indexed draws the indices already include `first`.
        NSInteger baseVertex = d.indexed ? d.baseVertex : 0;
        [e drawIndexedPrimitives:ii.prim indexCount:ii.count indexType:ii.type indexBuffer:ii.addr indexBufferLength:ii.length
                   instanceCount:d.instances baseVertex:baseVertex baseInstance:d.baseInstance];
    } else {
        [e drawPrimitives:ii.prim vertexStart:d.first vertexCount:d.count instanceCount:d.instances baseInstance:d.baseInstance];
    }
}

static void draw(Context* ctx, const DrawParams& d) {
    DrawParams v = d;  // validateDraw may turn the mode into GL_PATCHES
    if (!validateDraw(ctx, v.mode)) return;
    executeDraw(ctx, v);
}

// ---------------------------------------------------------------------------
// entry points
// ---------------------------------------------------------------------------
static bool validIndexType(GLenum t) { return t == GL_UNSIGNED_BYTE || t == GL_UNSIGNED_SHORT || t == GL_UNSIGNED_INT; }

API void APIENTRY glDrawArraysInstancedBaseInstance(GLenum mode, GLint first, GLsizei count, GLsizei instancecount, GLuint baseinstance) {
    MGL_CTX();
    if (first < 0 || count < 0 || instancecount < 0) return ctx->setError(GL_INVALID_VALUE);
    DrawParams d;
    d.mode = mode;
    d.first = first;
    d.count = count;
    d.instances = instancecount;
    d.baseInstance = baseinstance;
    draw(ctx, d);
}

// glDrawTransformFeedback*: vertex count = vertices captured into `stream` by the last completed
// transform feedback session of `id` (known on the CPU, see the counted capture passes).
static void drawTransformFeedback(Context* ctx, GLenum mode, GLuint id, GLuint stream, GLsizei instances) {
    if (stream >= 4) return ctx->setError(GL_INVALID_VALUE);
    if (instances < 0) return ctx->setError(GL_INVALID_VALUE);
    TransformFeedbackObj* x = id ? ctx->xfbs.get(id) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_VALUE, "not a transform feedback object");
    if (!x->ended) return ctx->setError(GL_INVALID_OPERATION, "EndTransformFeedback was never called");
    DrawParams d;
    d.mode = mode;
    d.first = 0;
    d.count = (GLsizei)x->drawCount[stream];
    d.instances = instances;
    draw(ctx, d);
}
API void APIENTRY glDrawTransformFeedback(GLenum mode, GLuint id) { MGL_CTX(); drawTransformFeedback(ctx, mode, id, 0, 1); }
API void APIENTRY glDrawTransformFeedbackInstanced(GLenum mode, GLuint id, GLsizei instancecount) {
    MGL_CTX();
    drawTransformFeedback(ctx, mode, id, 0, instancecount);
}
API void APIENTRY glDrawTransformFeedbackStream(GLenum mode, GLuint id, GLuint stream) { MGL_CTX(); drawTransformFeedback(ctx, mode, id, stream, 1); }
API void APIENTRY glDrawTransformFeedbackStreamInstanced(GLenum mode, GLuint id, GLuint stream, GLsizei instancecount) {
    MGL_CTX();
    drawTransformFeedback(ctx, mode, id, stream, instancecount);
}

void mglDrawArraysInternal(Context* ctx, GLenum mode, GLint first, GLsizei count) {
    DrawParams d;
    d.mode = mode;
    d.first = first;
    d.count = count;
    draw(ctx, d);
}

API void APIENTRY glDrawArrays(GLenum mode, GLint first, GLsizei count) { glDrawArraysInstancedBaseInstance(mode, first, count, 1, 0); }
API void APIENTRY glDrawArraysInstanced(GLenum mode, GLint first, GLsizei count, GLsizei instancecount) {
    glDrawArraysInstancedBaseInstance(mode, first, count, instancecount, 0);
}

API void APIENTRY glDrawElementsInstancedBaseVertexBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices,
                                                                GLsizei instancecount, GLint basevertex, GLuint baseinstance) {
    MGL_CTX();
    if (count < 0 || instancecount < 0) return ctx->setError(GL_INVALID_VALUE);
    if (!validIndexType(type)) return ctx->setError(GL_INVALID_ENUM);
    DrawParams d;
    d.mode = mode;
    d.count = count;
    d.indexed = true;
    d.indexType = type;
    d.indices = indices;
    d.instances = instancecount;
    d.baseVertex = basevertex;
    d.baseInstance = baseinstance;
    draw(ctx, d);
}

API void APIENTRY glDrawElements(GLenum mode, GLsizei count, GLenum type, const void* indices) {
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, 1, 0, 0);
}
API void APIENTRY glDrawElementsInstanced(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount) {
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, 0, 0);
}
API void APIENTRY glDrawElementsBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLint basevertex) {
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, 1, basevertex, 0);
}
API void APIENTRY glDrawElementsInstancedBaseVertex(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount,
                                                    GLint basevertex) {
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, basevertex, 0);
}
API void APIENTRY glDrawElementsInstancedBaseInstance(GLenum mode, GLsizei count, GLenum type, const void* indices, GLsizei instancecount,
                                                      GLuint baseinstance) {
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, instancecount, 0, baseinstance);
}
API void APIENTRY glDrawRangeElements(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices) {
    MGL_CTX();
    if (end < start) return ctx->setError(GL_INVALID_VALUE);
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, 1, 0, 0);
}
API void APIENTRY glDrawRangeElementsBaseVertex(GLenum mode, GLuint start, GLuint end, GLsizei count, GLenum type, const void* indices,
                                                GLint basevertex) {
    MGL_CTX();
    if (end < start) return ctx->setError(GL_INVALID_VALUE);
    glDrawElementsInstancedBaseVertexBaseInstance(mode, count, type, indices, 1, basevertex, 0);
}

API void APIENTRY glMultiDrawArrays(GLenum mode, const GLint* first, const GLsizei* count, GLsizei drawcount) {
    MGL_CTX();
    if (drawcount < 0) return ctx->setError(GL_INVALID_VALUE);
    if (!validateDraw(ctx, mode)) return;
    if (drawcount > 1 && tessBatchable(ctx)) {
        std::vector<DrawParams> ds(drawcount);
        for (int i = 0; i < drawcount; i++) {
            if (count[i] < 0) return ctx->setError(GL_INVALID_VALUE);
            ds[i].mode = mode;
            ds[i].first = first[i];
            ds[i].count = count[i];
            ds[i].drawID = i;
        }
        executeDraw(ctx, ds[0], ds.data(), drawcount);
        return;
    }
    for (int i = 0; i < drawcount; i++) {
        if (count[i] < 0) return ctx->setError(GL_INVALID_VALUE);
        DrawParams d;
        d.mode = mode;
        d.first = first[i];
        d.count = count[i];
        d.drawID = i;
        executeDraw(ctx, d);
    }
}

API void APIENTRY glMultiDrawElementsBaseVertex(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount,
                                                const GLint* basevertex) {
    MGL_CTX();
    if (drawcount < 0) return ctx->setError(GL_INVALID_VALUE);
    if (!validIndexType(type)) return ctx->setError(GL_INVALID_ENUM);
    if (!validateDraw(ctx, mode)) return;
    if (drawcount > 1 && tessBatchable(ctx)) {
        std::vector<DrawParams> ds(drawcount);
        for (int i = 0; i < drawcount; i++) {
            if (count[i] < 0) return ctx->setError(GL_INVALID_VALUE);
            ds[i].mode = mode;
            ds[i].count = count[i];
            ds[i].indexed = true;
            ds[i].indexType = type;
            ds[i].indices = indices[i];
            ds[i].baseVertex = basevertex ? basevertex[i] : 0;
            ds[i].drawID = i;
        }
        executeDraw(ctx, ds[0], ds.data(), drawcount);
        return;
    }
    for (int i = 0; i < drawcount; i++) {
        if (count[i] < 0) return ctx->setError(GL_INVALID_VALUE);
        DrawParams d;
        d.mode = mode;
        d.count = count[i];
        d.indexed = true;
        d.indexType = type;
        d.indices = indices[i];
        d.baseVertex = basevertex ? basevertex[i] : 0;
        d.drawID = i;
        executeDraw(ctx, d);
    }
}

API void APIENTRY glMultiDrawElements(GLenum mode, const GLsizei* count, GLenum type, const void* const* indices, GLsizei drawcount) {
    glMultiDrawElementsBaseVertex(mode, count, type, indices, drawcount, nullptr);
}

static Buffer* indirectBuffer(Context* ctx, const void* offset, size_t size) {
    Buffer* b = ctx->drawIndirectBuffer.get();
    if (!b) {
        ctx->setError(GL_INVALID_OPERATION, "no indirect buffer");
        return nullptr;
    }
    if ((size_t)offset % 4) {
        ctx->setError(GL_INVALID_VALUE);
        return nullptr;
    }
    if ((size_t)offset + size > (size_t)b->size) {
        ctx->setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    return b;
}

API void APIENTRY glMultiDrawArraysIndirect(GLenum mode, const void* indirect, GLsizei drawcount, GLsizei stride) {
    MGL_CTX();
    if (drawcount < 0 || (stride % 4)) return ctx->setError(GL_INVALID_VALUE);
    if (!validateDraw(ctx, mode)) return;
    GLsizei s = stride ? stride : 16;
    Buffer* b = indirectBuffer(ctx, indirect, drawcount ? (size_t)s * (drawcount - 1) + 16 : 0);
    if (!b) return;
    for (int i = 0; i < drawcount; i++) {
        DrawParams d;
        d.mode = mode;
        d.indirect = b;
        d.indirectOffset = (GLintptr)indirect + (GLintptr)i * s;
        d.drawID = i;
        d.count = 1;
        executeDraw(ctx, d);
    }
}

API void APIENTRY glDrawArraysIndirect(GLenum mode, const void* indirect) { glMultiDrawArraysIndirect(mode, indirect, 1, 0); }

API void APIENTRY glMultiDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect, GLsizei drawcount, GLsizei stride) {
    MGL_CTX();
    if (drawcount < 0 || (stride % 4)) return ctx->setError(GL_INVALID_VALUE);
    if (!validIndexType(type)) return ctx->setError(GL_INVALID_ENUM);
    if (!validateDraw(ctx, mode)) return;
    if (!ctx->vao->elementBuffer) return ctx->setError(GL_INVALID_OPERATION);
    GLsizei s = stride ? stride : 20;
    Buffer* b = indirectBuffer(ctx, indirect, drawcount ? (size_t)s * (drawcount - 1) + 20 : 0);
    if (!b) return;
    for (int i = 0; i < drawcount; i++) {
        DrawParams d;
        d.mode = mode;
        d.indexed = true;
        d.indexType = type;
        d.indirect = b;
        d.indirectOffset = (GLintptr)indirect + (GLintptr)i * s;
        d.drawID = i;
        d.count = 1;
        executeDraw(ctx, d);
    }
}

API void APIENTRY glDrawElementsIndirect(GLenum mode, GLenum type, const void* indirect) { glMultiDrawElementsIndirect(mode, type, indirect, 1, 0); }

static GLsizei readDrawCount(Context* ctx, GLintptr drawcount, GLsizei maxdrawcount) {
    Buffer* pb = ctx->parameterBuffer.get();
    if (!pb) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    if (drawcount % 4 || (size_t)drawcount + 4 > (size_t)pb->size) {
        ctx->setError(GL_INVALID_VALUE);
        return -1;
    }
    uint32_t n = 0;
    readBuffer(ctx, pb, drawcount, 4, &n);
    return std::min<GLsizei>((GLsizei)n, maxdrawcount);
}

API void APIENTRY glMultiDrawArraysIndirectCount(GLenum mode, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) {
    MGL_CTX();
    GLsizei n = readDrawCount(ctx, drawcount, maxdrawcount);
    if (n >= 0) glMultiDrawArraysIndirect(mode, indirect, n, stride);
}

API void APIENTRY glMultiDrawElementsIndirectCount(GLenum mode, GLenum type, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount,
                                                   GLsizei stride) {
    MGL_CTX();
    GLsizei n = readDrawCount(ctx, drawcount, maxdrawcount);
    if (n >= 0) glMultiDrawElementsIndirect(mode, type, indirect, n, stride);
}

API void APIENTRY glMultiDrawArraysIndirectCountARB(GLenum mode, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) {
    glMultiDrawArraysIndirectCount(mode, indirect, drawcount, maxdrawcount, stride);
}
API void APIENTRY glMultiDrawElementsIndirectCountARB(GLenum mode, GLenum type, const void* indirect, GLintptr drawcount, GLsizei maxdrawcount,
                                                      GLsizei stride) {
    glMultiDrawElementsIndirectCount(mode, type, indirect, drawcount, maxdrawcount, stride);
}

// ---------------------------------------------------------------------------
// EXT_mesh_shader draws: an object (task) + mesh render pipeline, no vertex input.
// ---------------------------------------------------------------------------
// Program of the current program / pipeline providing the mesh (and task) stages (null if none);
// *fsProg receives the program of the fragment stage (may differ in a program pipeline).
static ProgramObj* meshDrawProgram(Context* ctx, ProgramObj** fsProg) {
    const uint32_t meshBit = 1u << (int)glsl::Stage::Mesh;
    if (ctx->program) {
        ProgramObj* p = ctx->program.get();
        *fsProg = p;
        return p->program && (p->refl.stageMask & meshBit) ? p : nullptr;
    }
    if (!ctx->pipeline) return nullptr;
    ProgramPipelineObj* pp = ctx->pipeline.get();
    ProgramObj* m = pp->stages[(int)glsl::Stage::Mesh].get();
    if (!m || !m->program) return nullptr;
    // the task stage must come from the mesh stage's program (both run from one variant)
    ProgramObj* t = pp->stages[(int)glsl::Stage::Task].get();
    if (t && t != m) return nullptr;
    if (!t && (m->refl.stageMask & (1u << (int)glsl::Stage::Task))) return nullptr;
    *fsProg = pp->stages[(int)glsl::Stage::Fragment].get();
    return m;
}

// Validation shared by the mesh draw commands; returns the mesh program or null (error set).
static ProgramObj* validateMeshDraw(Context* ctx, ProgramObj** fsProg) {
    if (ctx->pipeline && !ctx->program) {
        std::string err = pipelineValidationError(ctx->pipeline.get());
        if (!err.empty()) {
            ctx->setError(GL_INVALID_OPERATION, "%s", err.c_str());
            return nullptr;
        }
    }
    ProgramObj* prog = meshDrawProgram(ctx, fsProg);
    if (!prog) {
        ctx->setError(GL_INVALID_OPERATION, "no active program with a mesh shader");
        return nullptr;
    }
    Framebuffer* fb = ctx->drawFb.get();
    if (!fb || framebufferStatus(ctx, fb) != GL_FRAMEBUFFER_COMPLETE) {
        ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION, "%s", framebufferIncompleteReason(ctx, fb));
        return nullptr;
    }
    if (ctx->depthTest || ctx->stencilTest) ctx->useDefaultDepthStencil(fb);
    if (ctx->xfb && ctx->xfb->active && !ctx->xfb->paused) {
        ctx->setError(GL_INVALID_OPERATION, "mesh draws cannot be used with active transform feedback");
        return nullptr;
    }
    return prog;
}

// One mesh draw: direct (x, y, z) or with the group counts read by the GPU at `indirect`.
static void executeMeshDraw(Context* ctx, ProgramObj* prog, ProgramObj* fsProg, GLuint x, GLuint y, GLuint z, Buffer* indirect,
                            GLintptr indirectOffset, int drawID) {
    MGL_STAT("draw");
    if (!indirect && (!x || !y || !z)) return;
    if (!mglCheckDrawConditional(ctx)) return;
    const glsl::Reflection& refl = prog->refl;
    glsl::VariantKey vk;
    vk.kind = glsl::PipelineKind::Mesh;
    vk.flatshadeLastVertex = ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION;
    vk.rasterDiscard = ctx->rasterizerDiscard;
    vk.sampleShading = ctx->sampleShading && ctx->minSampleShading > 0;
    vk.sampleMask = sampleMaskActive(ctx);
    vk.fsr = fsrWanted(ctx, true, fsProg ? &fsProg->refl : nullptr);  // the mesh shader may write the primitive's rate
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return;
    const glsl::EntryPoint* ms = var->entry(glsl::EntryPoint::Mesh);
    const glsl::EntryPoint* os = var->entry(glsl::EntryPoint::Object);
    if (!ms) return;
    // Program pipeline whose fragment stage comes from another separable program (or none).
    bool mixedFs = fsProg != prog;
    ProgramVariant* fsVar = nullptr;
    if (mixedFs && fsProg) {
        glsl::VariantKey fk;
        fk.kind = glsl::PipelineKind::Classic;
        fk.flatshadeLastVertex = vk.flatshadeLastVertex;
        fk.sampleShading = vk.sampleShading;
        fk.sampleMask = vk.sampleMask;
        fk.fsr = vk.fsr;
        fsVar = ctx->pipelines_->variant(fsProg, fk);
        if (!fsVar || !fsVar->ok) return;
    }
    const glsl::Reflection& frefl = fsProg ? fsProg->refl : refl;
    // Resolving bindings can end the render pass: build the arguments before acquiring the encoder.
    uint64_t argAddr = buildArguments(ctx, prog, false);
    uint64_t fsArgAddr = mixedFs && fsProg ? buildArguments(ctx, fsProg, false) : 0;
    uint64_t fsrAddr = vk.fsr ? fsrUpload(ctx) : 0;
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) return;
    RenderPipelineKey key;
    key.variant = var;
    if (mixedFs) {
        key.fsVariant = fsVar;
        key.fsVariantUid = fsVar ? fsVar->uid : ~0ull;
    }
    fillColorKey(ctx, key, frefl);
    key.rasterEnabled = !ctx->rasterizerDiscard;
    id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
    if (!pso) return;
    if (ctx->enc.pipeline != pso) {
        [e setRenderPipelineState:pso];
        ctx->enc.pipeline = pso;
    }
    applyDynamicState(ctx, e, refl.meshOutputPrimitive, refl.writesViewportIndex);
    id<MTL4ArgumentTable> table = ctx->renderArgTable;
    uint64_t sysAddr = buildSys(ctx, prog, drawID, 0, 0, 0, false);
    [table setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
    [table setAddress:sysAddr atIndex:glsl::kBufSys];
    if (mixedFs) {
        if (!ctx->enc.fsTableSplit) {
            [e setArgumentTable:ctx->renderArgTableFS atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = true;
        }
        [ctx->renderArgTableFS setAddress:fsArgAddr ? fsArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [ctx->renderArgTableFS setAddress:fsProg ? buildSys(ctx, fsProg, drawID, 0, 0, 0, false) : sysAddr atIndex:glsl::kBufSys];
    } else if (ctx->enc.fsTableSplit) {
        [e setArgumentTable:table atStages:MTLRenderStageFragment];
        ctx->enc.fsTableSplit = false;
    }
    if (fsrAddr) [(mixedFs ? ctx->renderArgTableFS : table) setAddress:fsrAddr atIndex:glsl::kBufFsr];
    mglQueryNotifyDraw(ctx, GL_NONE, 0, 0);  // visibility queries
    MTLSize meshTG = MTLSizeMake(ms->threadsPerThreadgroup[0], ms->threadsPerThreadgroup[1], ms->threadsPerThreadgroup[2]);
    MTLSize objTG = os ? MTLSizeMake(os->threadsPerThreadgroup[0], os->threadsPerThreadgroup[1], os->threadsPerThreadgroup[2])
                       : MTLSizeMake(1, 1, 1);
    if (indirect) {
        // DrawMeshTasksIndirectCommandEXT {x, y, z} matches MTLDispatchThreadgroupsIndirectArguments
        [e drawMeshThreadgroupsWithIndirectBuffer:indirect->gpuAddress() + indirectOffset threadsPerObjectThreadgroup:objTG
                        threadsPerMeshThreadgroup:meshTG];
        indirect->lastUse = ctx->serial();
    } else {
        [e drawMeshThreadgroups:MTLSizeMake(x, y, z) threadsPerObjectThreadgroup:objTG threadsPerMeshThreadgroup:meshTG];
    }
    flushImageWritebacks(ctx);
}

API void APIENTRY glDrawMeshTasksEXT(GLuint x, GLuint y, GLuint z) {
    MGL_CTX();
    ProgramObj* fsProg = nullptr;
    ProgramObj* prog = validateMeshDraw(ctx, &fsProg);
    if (!prog) return;
    // GL_MAX_TASK/MESH_WORK_GROUP_COUNT_EXT and _TOTAL_COUNT_EXT
    if (x > 65535 || y > 65535 || z > 65535 || (uint64_t)x * y * z > (1u << 22)) return ctx->setError(GL_INVALID_VALUE);
    executeMeshDraw(ctx, prog, fsProg, x, y, z, nullptr, 0, 0);
}

API void APIENTRY glMultiDrawMeshTasksIndirectEXT(GLintptr indirect, GLsizei drawcount, GLsizei stride) {
    MGL_CTX();
    if (drawcount <= 0 || (stride % 4)) return ctx->setError(GL_INVALID_VALUE);
    ProgramObj* fsProg = nullptr;
    ProgramObj* prog = validateMeshDraw(ctx, &fsProg);
    if (!prog) return;
    if (indirect < 0) return ctx->setError(GL_INVALID_VALUE);
    GLsizei s = stride ? stride : 12;
    Buffer* b = indirectBuffer(ctx, (const void*)indirect, (size_t)s * (drawcount - 1) + 12);
    if (!b) return;
    for (int i = 0; i < drawcount; i++) executeMeshDraw(ctx, prog, fsProg, 0, 0, 0, b, indirect + (GLintptr)i * s, i);
}

API void APIENTRY glDrawMeshTasksIndirectEXT(GLintptr indirect) {
    MGL_CTX();
    ProgramObj* fsProg = nullptr;
    ProgramObj* prog = validateMeshDraw(ctx, &fsProg);
    if (!prog) return;
    if (indirect < 0) return ctx->setError(GL_INVALID_VALUE);
    Buffer* b = indirectBuffer(ctx, (const void*)indirect, 12);
    if (!b) return;
    executeMeshDraw(ctx, prog, fsProg, 0, 0, 0, b, indirect, 0);
}

API void APIENTRY glMultiDrawMeshTasksIndirectCountEXT(GLintptr indirect, GLintptr drawcount, GLsizei maxdrawcount, GLsizei stride) {
    MGL_CTX();
    if (maxdrawcount < 0 || (stride % 4)) return ctx->setError(GL_INVALID_VALUE);
    GLsizei n = readDrawCount(ctx, drawcount, maxdrawcount);
    if (n < 0) return;
    ProgramObj* fsProg = nullptr;
    ProgramObj* prog = validateMeshDraw(ctx, &fsProg);
    if (!prog || n == 0) return;
    if (indirect < 0) return ctx->setError(GL_INVALID_VALUE);
    GLsizei s = stride ? stride : 12;
    Buffer* b = indirectBuffer(ctx, (const void*)indirect, (size_t)s * (n - 1) + 12);
    if (!b) return;
    for (int i = 0; i < n; i++) executeMeshDraw(ctx, prog, fsProg, 0, 0, 0, b, indirect + (GLintptr)i * s, i);
}

// ---------------------------------------------------------------------------
// compute
// ---------------------------------------------------------------------------
static void dispatch(Context* ctx, GLuint gx, GLuint gy, GLuint gz, Buffer* indirect, GLintptr indirectOffset) {
    ProgramObj* prog = ctx->activeProgramForStage(glsl::Stage::Compute);
    if (!prog || !prog->program || !(prog->refl.stageMask & (1u << (int)glsl::Stage::Compute)))
        return ctx->setError(GL_INVALID_OPERATION, "no compute program");
    if (gx > 65535 || gy > 65535 || gz > 65535) return ctx->setError(GL_INVALID_VALUE);
    if (!indirect && (!gx || !gy || !gz)) return;
    if (!mglCheckDrawConditional(ctx)) return;  // conditional rendering covers compute dispatches too
    glsl::VariantKey vk;
    vk.kind = glsl::PipelineKind::Compute;
    vk.sparseImageSlots = sparseImageSlots(ctx, prog);
    msArraySlots(ctx, prog, vk);
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return;
    id<MTLComputePipelineState> pso = ctx->pipelines_->computePipeline(var);
    if (!pso) return;
    if (ctx->statsQueriesActive) {
        uint32_t g[3] = {gx, gy, gz};
        if (indirect && indirect->cpu) {
            if (indirect->lastGpuWrite && !ctx->dev->isComplete(indirect->lastGpuWrite)) ctx->waitForResource(indirect->lastGpuWrite);
            memcpy(g, indirect->cpu + indirectOffset, 12);
        }
        DrawStats st;
        st.csInvocations = (uint64_t)g[0] * g[1] * g[2] * prog->refl.localSize[0] * prog->refl.localSize[1] * prog->refl.localSize[2];
        mglStatsCountDraw(ctx, st);
    }
    // Arguments first: binding them may encode copies (image shadows) into the compute encoder.
    uint64_t argAddr = buildArguments(ctx, prog, true);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e setComputePipelineState:pso];
    [ctx->computeArgTable setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
    [ctx->computeArgTable setAddress:buildSys(ctx, prog, 0, 0, 0, 0) atIndex:glsl::kBufSys];
    MTLSize tg = MTLSizeMake(prog->refl.localSize[0], prog->refl.localSize[1], prog->refl.localSize[2]);
    if (indirect) {
        [e dispatchThreadgroupsWithIndirectBuffer:indirect->gpuAddress() + indirectOffset threadsPerThreadgroup:tg];
        indirect->lastUse = ctx->serial();
    } else {
        [e dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:tg];
    }
    // GL requires explicit glMemoryBarrier for incoherent accesses, but keep dispatches ordered
    // against each other (cheap within one encoder).
    // Sparse images may be sampled through another texture aliasing their tiles (Texture::sparseRead).
    MTL4VisibilityOptions vis = vk.sparseImageSlots ? MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias : MTL4VisibilityOptionDevice;
    [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:vis];
    flushImageWritebacks(ctx);
    if (g_profilePasses) {  // one dispatch per command buffer
        char label[96];
        snprintf(label, sizeof label, "compute program %u, %ux%ux%u groups of %ux%ux%u%s", prog->name, gx, gy, gz, prog->refl.localSize[0],
                 prog->refl.localSize[1], prog->refl.localSize[2], indirect ? " (indirect)" : "");
        profileTag(ctx->serial(), label);
        ctx->flush();
    }
}

API void APIENTRY glDispatchCompute(GLuint x, GLuint y, GLuint z) {
    MGL_CTX();
    dispatch(ctx, x, y, z, nullptr, 0);
}

API void APIENTRY glDispatchComputeIndirect(GLintptr indirect) {
    MGL_CTX();
    Buffer* b = ctx->dispatchIndirectBuffer.get();
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    if (indirect < 0 || (indirect % 4)) return ctx->setError(GL_INVALID_VALUE);
    if ((size_t)indirect + 12 > (size_t)b->size) return ctx->setError(GL_INVALID_OPERATION);
    dispatch(ctx, 0, 0, 0, b, indirect);
}

API void APIENTRY glDispatchComputeGroupSizeARB(GLuint x, GLuint y, GLuint z, GLuint, GLuint, GLuint) { glDispatchCompute(x, y, z); }

// ---------------------------------------------------------------------------
// barriers
// ---------------------------------------------------------------------------
API void APIENTRY glMemoryBarrier(GLbitfield barriers) {
    MGL_CTX();
    // A tessellation segment runs all its VS + TCS kernels before its draws
    ctx->endTessSegment();
    const GLbitfield shaderWrites = GL_SHADER_STORAGE_BARRIER_BIT | GL_ATOMIC_COUNTER_BARRIER_BIT | GL_SHADER_IMAGE_ACCESS_BARRIER_BIT |
                                    GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT | GL_ELEMENT_ARRAY_BARRIER_BIT | GL_UNIFORM_BARRIER_BIT |
                                    GL_COMMAND_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT;
    if (ctx->renderEnc && (barriers & shaderWrites)) {
        // Intra-pass vertex->vertex barriers don't order shader writes between draws on Apple
        // GPUs; end the pass (the next encoder starts with a full barrier).
        ctx->endEncoder();
    } else if (ctx->renderEnc) {
        [ctx->renderEnc barrierAfterEncoderStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh
                              beforeEncoderStages:MTLStageVertex | MTLStageFragment | MTLStageObject | MTLStageMesh
                                visibilityOptions:MTL4VisibilityOptionDevice];
    } else if (ctx->computeEnc) {
        [ctx->computeEnc barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch | MTLStageBlit
                                 visibilityOptions:MTL4VisibilityOptionDevice];
    }
}
API void APIENTRY glMemoryBarrierByRegion(GLbitfield barriers) { glMemoryBarrier(barriers); }
API void APIENTRY glTextureBarrier(void) {
    MGL_CTX();
    // Feedback loops: finish the pass so later draws see earlier writes.
    ctx->endTessSegment();
    ctx->endEncoder();
}
API void APIENTRY glTextureBarrierNV(void) { glTextureBarrier(); }

