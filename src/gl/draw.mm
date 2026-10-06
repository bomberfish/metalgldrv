// Draw calls, compute dispatch, memory barriers.
#import "gl/texture.h"
#include <cmath>
#include <set>
#include <string>
#include <unordered_map>
#include "gl/gltypes.h"
#include "gl/pipeline.h"
#include "util/log.h"
#include "util/stats.h"

using namespace mgl;
std::string pipelineValidationError(ProgramPipelineObj* pp);  // api_shader.mm
#define API extern "C" __attribute__((visibility("default")))

GLenum framebufferStatus(Context* ctx, Framebuffer* fb);
bool mglCheckDrawConditional(Context* ctx);
ProgramObj* mglCompatPrepareDraw(Context* ctx, ProgramObj* prog);  // compat.mm (fixed-function emulation + gl_* uniforms)
bool mglXfbPrepareDraw(Context* ctx, GLenum mode, uint64_t vertices);  // api_xfb.mm
void mglQueryNotifyDraw(Context* ctx, GLenum mode, uint64_t vertices, uint64_t instances);  // api_query.mm

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
static void applyDynamicState(Context* ctx, id<MTL4RenderCommandEncoder> e, GLenum mode) {
    auto& es = ctx->enc;
    const PassDesc& p = ctx->curPass;
    // Viewport 0 (y is not flipped: GL row 0 == Metal row 0, see glsl.h)
    MTLViewport vp;
    vp.originX = ctx->viewport[0][0];
    vp.originY = ctx->viewport[0][1];
    vp.width = std::max(0.0f, ctx->viewport[0][2]);
    vp.height = std::max(0.0f, ctx->viewport[0][3]);
    vp.znear = ctx->depthRange[0][0];
    vp.zfar = ctx->depthRange[0][1];
    if (es.numViewports != 1 || memcmp(&es.vp[0], &vp, sizeof vp) != 0) {
        [e setViewport:vp];
        es.vp[0] = vp;
        es.numViewports = 1;
    }
    MTLScissorRect sc = {0, 0, (NSUInteger)p.width, (NSUInteger)p.height};
    if (ctx->scissorEnabled & 1) {
        int x0 = std::clamp(ctx->scissor[0][0], 0, p.width), y0 = std::clamp(ctx->scissor[0][1], 0, p.height);
        int x1 = std::clamp(ctx->scissor[0][0] + ctx->scissor[0][2], 0, p.width);
        int y1 = std::clamp(ctx->scissor[0][1] + ctx->scissor[0][3], 0, p.height);
        sc = {(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)std::max(0, x1 - x0), (NSUInteger)std::max(0, y1 - y0)};
    }
    if (memcmp(&es.sc[0], &sc, sizeof sc) != 0) {
        [e setScissorRect:sc];
        es.sc[0] = sc;
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
    dk.depthWrite = ctx->depthTest && hasDepth && ctx->depthWrite;
    dk.stencilEnabled = ctx->stencilTest && hasStencil;
    if (dk.stencilEnabled) {
        for (int f = 0; f < 2; f++) {
            const StencilFace& s = ctx->stencil[f];
            dk.face[f].func = compareFunc(s.func);
            dk.face[f].sfail = stencilOp(s.sfail);
            dk.face[f].dpfail = stencilOp(s.dpfail);
            dk.face[f].dppass = stencilOp(s.dppass);
            dk.face[f].readMask = s.valueMask & 0xff;
            dk.face[f].writeMask = s.writeMask & 0xff;
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
static uint64_t buildArguments(Context* ctx, ProgramObj* prog, bool forCompute) {
    const glsl::Reflection& r = prog->refl;
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
            if (b.buffer && b.buffer->mtl) {
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
        case glsl::SlotKind::Image: {
            int unit = std::clamp(prog->samplerUnits[i], 0, kMaxImageUnits - 1);
            ImageUnit& iu = ctx->imageUnits[unit];
            id<MTLTexture> view = nil;
            if (iu.tex && ensureTextureStorage(ctx, iu.tex.get())) {
                Texture* t = iu.tex.get();
                const FormatInfo* f = formatInfo(iu.format);
                MTLPixelFormat pf = f ? f->mtl : t->mtl.pixelFormat;
                if (t->target == GL_TEXTURE_BUFFER) {
                    view = t->mtl;
                } else {
                    MTLTextureType vt = mtlTextureType(s.textureTarget, t->samples);
                    // imageCube / imageCubeArray are declared as texture2d_array in MSL.
                    if (vt == MTLTextureTypeCube || vt == MTLTextureTypeCubeArray) vt = MTLTextureType2DArray;
                    NSUInteger slices = t->mtl.arrayLength * (t->mtl.textureType == MTLTextureTypeCube || t->mtl.textureType == MTLTextureTypeCubeArray ? 6 : 1);
                    NSRange sr = NSMakeRange(0, slices);
                    if (!iu.layered && (t->mtl.textureType == MTLTextureType2DArray || t->mtl.textureType == MTLTextureTypeCube ||
                                        t->mtl.textureType == MTLTextureTypeCubeArray)) {
                        sr = NSMakeRange(std::min<NSUInteger>(iu.layer, slices - 1), 1);
                        vt = MTLTextureType2D;
                    }
                    if (pf != t->mtl.pixelFormat && !(t->mtl.usage & MTLTextureUsagePixelFormatView)) pf = t->mtl.pixelFormat;
                    view = [t->mtl newTextureViewWithPixelFormat:pf textureType:vt levels:NSMakeRange(std::min(iu.level, t->storageLevels - 1), 1) slices:sr];
                    ctx->dev->retainUntil(view, serial);
                }
                t->lastUse = t->lastGpuWrite = serial;
            }
            if (!view) view = dummyTexture(ctx, mtlTextureType(s.textureTarget, 0), FmtKind::Unorm);
            MTLResourceID rid = view.gpuResourceID;
            memcpy(&v, &rid, 8);
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

static uint64_t buildSys(Context* ctx, ProgramObj* prog, int drawID, GLint baseVertex, GLuint baseInstance, GLint first) {
    UploadAlloc sa = ctx->upload(sizeof(glsl::MGLSys), 16);
    glsl::MGLSys* s = (glsl::MGLSys*)sa.cpu;
    s->depthRangeNear = ctx->depthRange[0][0];
    s->depthRangeFar = ctx->depthRange[0][1];
    s->depthMode = ctx->clipDepthMode == GL_ZERO_TO_ONE ? 1 : 0;
    s->clipDistanceMask = ctx->clipDistanceMask;
    for (int k = 0; k < 4; k++) s->viewport[k] = ctx->viewport[0][k];
    s->pointSize = ctx->pointSize;
    s->flags = (ctx->programPointSize ? 1u : 0u) | (ctx->pointSpriteOrigin == GL_LOWER_LEFT ? 2u : 0u) | (ctx->clipOrigin == GL_UPPER_LEFT ? 4u : 0u);
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

// dvec3/dvec4 (and matrices of them) occupy one GL location but don't fit one Metal
// vertex attribute: such programs fetch their inputs in the shader.
static bool wantManualFetch(Context* ctx, ProgramObj* prog) {
    if (wantManualFetch(ctx)) return true;
    VertexArray* vao = ctx->vao.get();
    if (!vao) return false;
    bool wide = false;
    for (const auto& a : prog->refl.attributes) {
        switch (a.type) {
        case GL_DOUBLE_VEC3: case GL_DOUBLE_VEC4: case GL_DOUBLE_MAT2x3: case GL_DOUBLE_MAT2x4: case GL_DOUBLE_MAT3:
        case GL_DOUBLE_MAT3x4: case GL_DOUBLE_MAT4: case GL_DOUBLE_MAT4x3: wide = true; break;
        default: break;
        }
    }
    if (!wide) return false;
    for (int l = 0; l < kMaxVertexAttribs; l++)
        if (vao->attribs[l].enabled && vao->attribs[l].clientPointer) return false;
    return true;
}

// Builds the render pipeline key & binds vertex buffers. Returns false if the draw must be skipped.
static bool setupVertexInput(Context* ctx, ProgramObj* prog, RenderPipelineKey& key, id<MTL4ArgumentTable> table, const DrawParams& d,
                             uint32_t minIndex, uint32_t maxIndex) {
    VertexArray* vao = ctx->vao.get();
    uint64_t serial = ctx->serial();
    if (key.variant && key.variant->manualFetch) {
        // Attributes are fetched in the shader: bind the vertex buffers through MGLDrawInfo.
        UploadAlloc di = ctx->upload(sizeof(glsl::MGLDrawInfo), 16);
        glsl::MGLDrawInfo* info = (glsl::MGLDrawInfo*)di.cpu;
        memset(info, 0, sizeof *info);
        for (int b = 0; b < kMaxVertexBindings; b++) {
            const VertexBinding& vb = vao->bindings[b];
            if (vb.buffer && vb.buffer->mtl) {
                info->vertexBuffer[b] = vb.buffer->gpuAddress() + vb.offset;
                vb.buffer->lastUse = serial;
            }
            info->vertexStride[b] = vb.stride;
            info->vertexDivisor[b] = vb.divisor;
        }
        info->baseInstance = d.baseInstance;
        [table setAddress:di.gpu atIndex:glsl::kBufStageIO];
        return true;
    }
    bool needCurrent = false;
    uint32_t bindingUsed = 0;
    for (const auto& a : prog->refl.attributes) {
        if (a.location < 0) continue;
        TypeShape ts = typeShape(a.type);
        int nloc = std::max(1, a.locationsUsed) ;
        for (int k = 0; k < nloc; k++) {
            int loc = a.location + k;
            if (loc >= kMaxVertexAttribs) break;
            VertexAttribKey& ak = key.attribs[loc];
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
                    // dvecN arrive as raw bits (uint2/uint4); dvec3/dvec4 only get their first two
                    // components here (programs with them normally use manual fetch).
                    f = va->size == 1 ? MTLVertexFormatUInt2 : MTLVertexFormatUInt4;
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
        UploadAlloc ca = ctx->upload(16 * kMaxVertexAttribs, 16);
        memcpy(ca.cpu, ctx->currentAttrib, 16 * kMaxVertexAttribs);
        VertexBufferKey& bk = key.buffers[kBufCurrentAttribs];
        bk.used = 1;
        bk.stepFunction = MTLVertexStepFunctionConstant;
        bk.stepRate = 0;
        [table setAddress:ca.gpu attributeStride:0 atIndex:kBufCurrentAttribs];
    }
    return true;
}

static void fillColorKey(Context* ctx, RenderPipelineKey& key, const glsl::Reflection& refl) {
    const PassDesc& p = ctx->curPass;
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
        bk.enabled = b.enabled && !intFmt;
        if (bk.enabled) {
            bk.srcRGB = blendFactor(b.srcRGB);
            bk.dstRGB = blendFactor(b.dstRGB);
            bk.srcA = blendFactor(b.srcA);
            bk.dstA = blendFactor(b.dstA);
            bk.eqRGB = blendOp(b.eqRGB);
            bk.eqA = blendOp(b.eqA);
        }
        bk.writeMask = b.colorMask;
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
}  // namespace

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
            src = ib->cpu + off;
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
    // Adjacency topologies without a geometry shader draw their non-adjacent vertices, and list
    // topologies with primitive restart drop the incomplete primitive before each restart
    // (Metal only restarts strips): decompose to a plain list on the CPU.
    bool adjacency = mode == GL_LINES_ADJACENCY || mode == GL_LINE_STRIP_ADJACENCY || mode == GL_TRIANGLES_ADJACENCY ||
                     mode == GL_TRIANGLE_STRIP_ADJACENCY;
    bool listRestart = restart && d.indexed && (mode == GL_POINTS || mode == GL_LINES || mode == GL_TRIANGLES);
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
            for (size_t i = 1; i + 1 < n; i++) out.insert(out.end(), {v(i), v(i + 1), v(0)});
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

// Runs a draw through a multi-pass pipeline. Returns false if unsupported.
static bool executeMultiPass(Context* ctx, ProgramObj* prog, const DrawParams& d, glsl::VariantKey vk, bool xfbCapture) {
    const glsl::Reflection& refl = prog->refl;
    if (d.indirect) {
        static bool warned = false;
        if (!warned) MGL_WARN("indirect draws with geometry/tessellation/transform feedback not supported yet");
        warned = true;
        return false;
    }
    // --- primitive list ---
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    const uint8_t* src = nullptr;
    if (d.indexed) {
        if (ib) {
            if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
            src = ib->cpu + (size_t)d.indices;
        } else {
            src = (const uint8_t*)d.indices;
        }
    }
    GLenum listMode = listModeFor(d.mode);
    int vpp = listVerts(listMode, ctx->patchVertices);
    std::vector<uint32_t> ids;
    decomposeToList(ctx, d, src, ids);
    uint32_t prims = (uint32_t)(ids.size() / vpp);
    if (!prims) return true;
    UploadAlloc idxUp = ctx->upload(ids.size() * 4, 16);
    memcpy(idxUp.cpu, ids.data(), ids.size() * 4);

    fillVertexFetch(ctx, prog, vk);
    vk.primitive = (uint8_t)listMode;
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return true;

    // --- draw info ---
    UploadAlloc di = ctx->upload(sizeof(glsl::MGLDrawInfo), 16);
    glsl::MGLDrawInfo* info = (glsl::MGLDrawInfo*)di.cpu;
    memset(info, 0, sizeof *info);
    uint64_t serial = ctx->serial();
    VertexArray* vao = ctx->vao.get();
    for (int b = 0; b < kMaxVertexBindings; b++) {
        const VertexBinding& vb = vao->bindings[b];
        if (vb.buffer && vb.buffer->mtl) {
            info->vertexBuffer[b] = vb.buffer->gpuAddress() + vb.offset;
            vb.buffer->lastUse = serial;
        }
        info->vertexStride[b] = vb.stride;
        info->vertexDivisor[b] = vb.divisor;
    }
    info->indexBuffer = idxUp.gpu;  // always a 32-bit list of vertex ids (base vertex not yet applied)
    info->indexType = 4;
    info->count = (uint32_t)ids.size();
    info->baseVertex = d.indexed ? d.baseVertex : 0;
    info->baseInstance = d.baseInstance;
    info->firstVertex = 0;
    info->instanceCount = d.instances;
    info->primitiveCount = prims;
    info->primitiveMode = listMode;
    info->verticesPerPrimitive = vpp;
    info->patchVertices = ctx->patchVertices;
    TransformFeedbackObj* x = ctx->xfb.get();
    if (xfbCapture) {
        for (int i = 0; i < kMaxXfbBuffers; i++) {
            auto& bb = x->buffers[i];
            if (!bb.buffer || !bb.buffer->mtl) continue;
            info->xfbStride[i] = refl.multipass.xfbStride[i];
            info->xfbBuffer[i] = bb.buffer->gpuAddress() + bb.offset;
            GLsizeiptr range = bb.size ? bb.size : bb.buffer->size - bb.offset;
            info->xfbCapacity[i] = (uint32_t)std::max<GLsizeiptr>(0, range);  // bytes; shader divides by its stride
            bb.buffer->lastUse = bb.buffer->lastGpuWrite = serial;
        }
        info->xfbVertexBase = (uint32_t)x->verticesWritten;
    }
    UploadAlloc counters = ctx->upload(64, 16);
    memset(counters.cpu, 0, 64);
    info->counters = counters.gpu;

    uint64_t argAddr = buildArguments(ctx, prog, false);
    uint64_t sysAddr = buildSys(ctx, prog, d.drawID, d.baseVertex, d.baseInstance, d.first);

    // --- tessellation pass 1: VS+TCS compute ---
    if (vk.kind == glsl::PipelineKind::Tessellation) {
        const glsl::EntryPoint* k = var->entry(glsl::EntryPoint::Kernel);
        if (!k) return false;
        int cpts = refl.multipass.tcsOutputVertices ? (int)refl.multipass.tcsOutputVertices : ctx->patchVertices;
        size_t vbytes = (size_t)prims * d.instances * std::max(1, cpts) * std::max<uint32_t>(16, refl.multipass.tcsVertexOutBytes) + 16;
        size_t pbytes = (size_t)prims * d.instances * std::max<uint32_t>(32, refl.multipass.tcsPatchOutBytes) + 16;
        id<MTLBuffer> vout = ctx->dev->newBuffer(vbytes);
        id<MTLBuffer> pout = ctx->dev->newBuffer(pbytes);
        ctx->dev->releaseAfter(vout, serial);
        ctx->dev->releaseAfter(pout, serial);
        info->tcsVertexOut = vout.gpuAddress;
        info->tcsPatchOut = pout.gpuAddress;
        id<MTLComputePipelineState> cps = ctx->pipelines_->computePipeline(var);
        if (!cps) return true;
        id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
        [ce setComputePipelineState:cps];
        [ctx->computeArgTable setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [ctx->computeArgTable setAddress:sysAddr atIndex:glsl::kBufSys];
        [ctx->computeArgTable setAddress:di.gpu atIndex:glsl::kBufStageIO];
        [ce dispatchThreadgroups:MTLSizeMake(prims, d.instances, 1)
            threadsPerThreadgroup:MTLSizeMake(k->threadsPerThreadgroup[0], k->threadsPerThreadgroup[1], k->threadsPerThreadgroup[2])];
        ctx->endEncoder();
    }
    // --- transform feedback only (no GS/tess): VS as compute writing captured vertices ---
    if (vk.kind == glsl::PipelineKind::XfbCapture) {
        const glsl::EntryPoint* k = var->entry(glsl::EntryPoint::Kernel);
        id<MTLComputePipelineState> cps = k ? ctx->pipelines_->computePipeline(var) : nil;
        if (!cps) return true;
        id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
        [ce setComputePipelineState:cps];
        [ctx->computeArgTable setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [ctx->computeArgTable setAddress:sysAddr atIndex:glsl::kBufSys];
        [ctx->computeArgTable setAddress:di.gpu atIndex:glsl::kBufStageIO];
        uint32_t total = (uint32_t)ids.size() * d.instances;
        uint32_t tg = std::max(1, k->threadsPerThreadgroup[0]);
        [ce dispatchThreads:MTLSizeMake(total, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        // Later dispatches in this encoder may read what was captured (xfb ping-pong).
        [ce barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit
                    visibilityOptions:MTL4VisibilityOptionDevice];
        x->verticesWritten += total;
        return true;  // rasterization (if enabled) is handled by the caller's classic draw
    }
    // --- GS + transform feedback: a single-thread kernel walks primitives in GL order ---
    if (xfbCapture && vk.kind == glsl::PipelineKind::MeshGeometry) {
        if (var->entry(glsl::EntryPoint::Kernel)) {
            id<MTLComputePipelineState> cps = ctx->pipelines_->computePipeline(var);
            if (cps) {
                id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
                [ce setComputePipelineState:cps];
                [ctx->computeArgTable setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
                [ctx->computeArgTable setAddress:sysAddr atIndex:glsl::kBufSys];
                [ctx->computeArgTable setAddress:di.gpu atIndex:glsl::kBufStageIO];
                [ce dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
                ctx->endEncoder();
            }
        }
    }
    // --- mesh pass (GS or TES[+GS]) ---
    const glsl::EntryPoint* ms = var->entry(glsl::EntryPoint::Mesh);
    const glsl::EntryPoint* os = var->entry(glsl::EntryPoint::Object);
    if (!ms) return false;
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) return true;
    RenderPipelineKey key;
    key.variant = var;
    fillColorKey(ctx, key, refl);
    key.rasterEnabled = !ctx->rasterizerDiscard;
    id<MTLRenderPipelineState> pso = ctx->pipelines_->renderPipeline(key);
    if (!pso) return true;
    if (ctx->enc.pipeline != pso) {
        [e setRenderPipelineState:pso];
        ctx->enc.pipeline = pso;
    }
    applyDynamicState(ctx, e, refl.gsOutputPrimitive ? refl.gsOutputPrimitive : GL_TRIANGLES);
    id<MTL4ArgumentTable> table = ctx->renderArgTable;
    [table setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
    [table setAddress:sysAddr atIndex:glsl::kBufSys];
    [table setAddress:di.gpu atIndex:glsl::kBufStageIO];
    mglQueryNotifyDraw(ctx, d.mode, ids.size(), d.instances);
    MTLSize meshTG = MTLSizeMake(ms->threadsPerThreadgroup[0], ms->threadsPerThreadgroup[1], ms->threadsPerThreadgroup[2]);
    if (os) {
        MTLSize objTG = MTLSizeMake(os->threadsPerThreadgroup[0], os->threadsPerThreadgroup[1], os->threadsPerThreadgroup[2]);
        [e drawMeshThreadgroups:MTLSizeMake(prims, d.instances, 1) threadsPerObjectThreadgroup:objTG threadsPerMeshThreadgroup:meshTG];
    } else {
        int invocations = std::max(1, refl.gsInvocations);
        [e drawMeshThreadgroups:MTLSizeMake(prims, d.instances, invocations) threadsPerObjectThreadgroup:MTLSizeMake(1, 1, 1)
            threadsPerMeshThreadgroup:meshTG];
    }
    // GS output counts are only known on the GPU (MGLDrawInfo::counters); DrawTransformFeedback
    // with GS-generated vertices is not tracked yet.
    return true;
}

// Fast path for indexed triangle/line lists from a buffer: rotate the whole index buffer
// once (per content version) and draw from the same offset in the rotated copy. This
// serves every draw into a shared index buffer (Sodium) with one allocation.
static bool rotatedWholeBuffer(Context* ctx, const DrawParams& d, IndexInfo& out) {
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    if (!ib || !ib->mtl || (d.mode != GL_TRIANGLES && d.mode != GL_LINES)) return false;
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
        uint8_t* dst = (uint8_t*)ib->rotated.contents;
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

// Builds (or reuses from cache) a rotated 32-bit index list for flat shading.
static bool prepareRotatedIndices(Context* ctx, const DrawParams& d, IndexInfo& out, uint32_t* minIdx, uint32_t* maxIdx) {
    Buffer* ib = d.indexed && ctx->vao ? ctx->vao->elementBuffer.get() : nullptr;
    const uint8_t* src = nullptr;
    if (d.indexed) {
        if (ib) {
            if (!ib->mtl || (size_t)d.indices + (size_t)d.count * indexSize(d.indexType) > (size_t)ib->size) return false;
            if (ib->lastGpuWrite && !ctx->dev->isComplete(ib->lastGpuWrite)) ctx->waitForResource(ib->lastGpuWrite);
            src = ib->cpu + (size_t)d.indices;
        } else {
            src = (const uint8_t*)d.indices;
        }
    }
    // Cache key: only buffer-backed indices whose content can't change behind our back.
    bool cacheable = ib && !(ib->mapped && (ib->mapAccess & GL_MAP_PERSISTENT_BIT) && (ib->mapAccess & GL_MAP_WRITE_BIT));
    char key[96];
    snprintf(key, sizeof key, "%p:%u:%llu:%zu:%d:%x:%x:%d:%u", (void*)ib, ib ? ib->generation : 0,
             (unsigned long long)(ib ? ib->contentVersion : 0), (size_t)d.indices, d.count, d.indexType, d.mode, d.first,
             ctx->primitiveRestart || ctx->primitiveRestartFixed ? ctx->primitiveRestartIndex : 0xdead);
    struct RotEntry {
        IndexInfo info;
        uint32_t minIdx, maxIdx;
        id<MTLBuffer> buffer;
    };
    static std::unordered_map<Context*, std::unordered_map<std::string, RotEntry>> caches;
    auto& cache = caches[ctx];
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
    GLenum lm = listModeFor(d.mode);
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
        memcpy(buf.contents, ids.data(), ids.size() * 4);
        out.addr = buf.gpuAddress;
        static std::unordered_map<Context*, size_t> cacheBytes;
        if (cache.size() > 512 || cacheBytes[ctx] > (32u << 20)) {
            for (auto& [k, e] : cache) ctx->dev->releaseAfter(e.buffer, ctx->serial());
            cache.clear();
            cacheBytes[ctx] = 0;
        }
        cacheBytes[ctx] += ids.size() * 4;
        cache[key] = {out, *minIdx, *maxIdx, buf};
    } else {
        UploadAlloc u = ctx->upload(ids.size() * 4, 16);
        memcpy(u.cpu, ids.data(), ids.size() * 4);
        out.addr = u.gpu;
    }
    return true;
}

// ---------------------------------------------------------------------------
// main draw
// ---------------------------------------------------------------------------
static bool validateDraw(Context* ctx, GLenum mode) {
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
        ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
        return false;
    }
    if (ctx->pipeline && !ctx->program) {
        ProgramPipelineObj* pp = ctx->pipeline.get();
        std::string err = pp->stages[(int)glsl::Stage::Vertex] || pp->stages[(int)glsl::Stage::Fragment] ? pipelineValidationError(pp) : "";
        if (!err.empty() && err != "pipeline has no vertex or compute program") {
            ctx->setError(GL_INVALID_OPERATION, "%s", err.c_str());
            return false;
        }
        if (!pp->stages[(int)glsl::Stage::Vertex]) return false;  // missing stages: undefined rendering, no error
    }
    ProgramObj* prog = ctx->activeProgramForStage(glsl::Stage::Vertex);
    if (!prog && ctx->coreProfile) return false;  // no program/pipeline: undefined rendering, not an error
    bool hasTess = prog && (prog->refl.stageMask & (1u << (int)glsl::Stage::TessEval));
    if (mode == GL_PATCHES && !hasTess) {
        ctx->setError(GL_INVALID_OPERATION);
        return false;
    }
    if (hasTess && mode != GL_PATCHES) {
        ctx->setError(GL_INVALID_OPERATION);
        return false;
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

static bool programHasSideEffects(const glsl::Reflection& r) {
    if (!r.storageBlocks.empty() || !r.atomicCounterBuffers.empty()) return true;
    for (auto& sl : r.slots)
        if (sl.kind == glsl::SlotKind::Image) return true;
    return false;
}

static void executeDraw(Context* ctx, const DrawParams& d) {
    MGL_STAT("draw");
    ProgramObj* prog = ctx->activeProgramForStage(glsl::Stage::Vertex);
    // Compatibility profile: fixed-function emulation (prog == null) and gl_* state uniforms.
    if (!ctx->coreProfile) prog = mglCompatPrepareDraw(ctx, prog);
    if (!prog) return;
    if (!prog->program) return;  // last successfully linked executable
    if (!mglCheckDrawConditional(ctx)) return;
    mgl::debugCaptureBeforeDraw(ctx, prog);
    if (d.count <= 0 && !d.indirect) return;
    if (d.instances <= 0 && !d.indirect) return;
    const glsl::Reflection& refl = prog->refl;
    bool hasGS = refl.stageMask & (1u << (int)glsl::Stage::Geometry);
    bool hasTess = refl.stageMask & (1u << (int)glsl::Stage::TessEval);
    glsl::VariantKey vk;
    vk.kind = hasTess ? glsl::PipelineKind::Tessellation : hasGS ? glsl::PipelineKind::MeshGeometry : glsl::PipelineKind::Classic;
    vk.primitive = (uint8_t)d.mode;
    vk.flatshadeLastVertex = ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION;
    vk.rasterDiscard = ctx->rasterizerDiscard;
    bool xfbActive = ctx->xfb && ctx->xfb->active && !ctx->xfb->paused;
    if (vk.kind != glsl::PipelineKind::Classic) {
        executeMultiPass(ctx, prog, d, vk, xfbActive);
        return;
    }
    if (xfbActive) {
        glsl::VariantKey xk = vk;
        xk.kind = glsl::PipelineKind::XfbCapture;
        executeMultiPass(ctx, prog, d, xk, true);
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
    bool rotate = frefl.usesFlatShading && ctx->provokingVertex == GL_LAST_VERTEX_CONVENTION && !d.indirect &&
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
    key.topologyClass = (uint8_t)MTLPrimitiveTopologyClassUnspecified;  // point size is always written; only layered rendering needs a class
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
    applyDynamicState(ctx, e, d.mode);
    [table setAddress:argAddr ? argAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
    uint64_t sysAddr = buildSys(ctx, prog, d.drawID, d.baseVertex, d.baseInstance, d.first);
    [table setAddress:sysAddr atIndex:glsl::kBufSys];
    if (mixedFs) {
        // The fragment stage reads its own program's arguments.
        if (!ctx->enc.fsTableSplit) {
            [e setArgumentTable:ctx->renderArgTableFS atStages:MTLRenderStageFragment];
            ctx->enc.fsTableSplit = true;
        }
        [ctx->renderArgTableFS setAddress:fsArgAddr ? fsArgAddr : ctx->upload(16).gpu atIndex:glsl::kBufArgs];
        [ctx->renderArgTableFS setAddress:fsProg ? buildSys(ctx, fsProg, d.drawID, d.baseVertex, d.baseInstance, d.first) : sysAddr
                                  atIndex:glsl::kBufSys];
    } else if (ctx->enc.fsTableSplit) {
        [e setArgumentTable:table atStages:MTLRenderStageFragment];
        ctx->enc.fsTableSplit = false;
    }
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
    if (!validateDraw(ctx, d.mode)) return;
    executeDraw(ctx, d);
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
    if (ctx->coreProfile && ctx->vao && !ctx->vao->elementBuffer) return ctx->setError(GL_INVALID_OPERATION, "no element buffer");
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
    if (pb->lastGpuWrite && !ctx->dev->isComplete(pb->lastGpuWrite)) ctx->waitForResource(pb->lastGpuWrite);
    uint32_t n;
    memcpy(&n, pb->cpu + drawcount, 4);
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
// compute
// ---------------------------------------------------------------------------
static void dispatch(Context* ctx, GLuint gx, GLuint gy, GLuint gz, Buffer* indirect, GLintptr indirectOffset) {
    ProgramObj* prog = ctx->activeProgramForStage(glsl::Stage::Compute);
    if (!prog || !prog->program || !(prog->refl.stageMask & (1u << (int)glsl::Stage::Compute)))
        return ctx->setError(GL_INVALID_OPERATION, "no compute program");
    if (gx > 65535 || gy > 65535 || gz > 65535) return ctx->setError(GL_INVALID_VALUE);
    if (!indirect && (!gx || !gy || !gz)) return;
    glsl::VariantKey vk;
    vk.kind = glsl::PipelineKind::Compute;
    ProgramVariant* var = ctx->pipelines_->variant(prog, vk);
    if (!var || !var->ok) return;
    id<MTLComputePipelineState> pso = ctx->pipelines_->computePipeline(var);
    if (!pso) return;
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e setComputePipelineState:pso];
    uint64_t argAddr = buildArguments(ctx, prog, true);
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
    [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
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
    ctx->endEncoder();
}
API void APIENTRY glTextureBarrierNV(void) { glTextureBarrier(); }
