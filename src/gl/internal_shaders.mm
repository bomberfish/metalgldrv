#import "gl/internal_shaders.h"
#import <MetalFX/MetalFX.h>
#include <string>
#include <unordered_map>
#include "util/log.h"

namespace mgl {

static const char* kInternalMSL = R"MSL(
#include <metal_stdlib>
using namespace metal;

struct QuadArgs {
    float4 rect;        // dst rect in NDC: x0,y0,x1,y1
    float4 srcRect;     // src texcoords (normalized or texel for depth/stencil): x0,y0,x1,y1
    float depth;
    float lod;
    uint slice;
    uint pad;
};

struct QuadOut {
    float4 pos [[position]];
    float2 uv;
};

vertex QuadOut quad_vs(uint vid [[vertex_id]], constant QuadArgs& a [[buffer(0)]]) {
    // Two triangles as a strip of 4 vertices.
    float2 t = float2(float(vid & 1), float((vid >> 1) & 1));
    QuadOut o;
    float2 p = mix(a.rect.xy, a.rect.zw, t);
    o.pos = float4(p.x, -p.y, a.depth, 1.0);   // GL row 0 == Metal row 0
    o.uv = mix(a.srcRect.xy, a.srcRect.zw, t);
    return o;
}

// Layered framebuffers: one instance per layer.
struct QuadOutLayered {
    float4 pos [[position]];
    float2 uv;
    uint layer [[render_target_array_index]];
};
vertex QuadOutLayered quad_vs_layered(uint vid [[vertex_id]], uint iid [[instance_id]], constant QuadArgs& a [[buffer(0)]]) {
    float2 t = float2(float(vid & 1), float((vid >> 1) & 1));
    QuadOutLayered o;
    float2 p = mix(a.rect.xy, a.rect.zw, t);
    o.pos = float4(p.x, -p.y, a.depth, 1.0);
    o.uv = mix(a.srcRect.xy, a.srcRect.zw, t);
    o.layer = iid;
    return o;
}

struct PresentArgs {
    texture2d<float> src;
};

fragment float4 present_fs(QuadOut in [[stage_in]], constant PresentArgs& a [[buffer(1)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return a.src.sample(s, in.uv);
}

struct BlitColorArgs {
    texture2d_array<float> src;
};
fragment float4 blit_fs_nearest(QuadOut in [[stage_in]], constant BlitColorArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    return a.src.sample(s, in.uv, q.slice, level(q.lod));
}
fragment float4 blit_fs_linear(QuadOut in [[stage_in]], constant BlitColorArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return a.src.sample(s, in.uv, q.slice, level(q.lod));
}
struct BlitIntArgs { texture2d_array<int> src; };
struct BlitUintArgs { texture2d_array<uint> src; };
fragment int4 blit_fs_int(QuadOut in [[stage_in]], constant BlitIntArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    uint2 c = uint2(in.uv);
    return a.src.read(c, q.slice, uint(q.lod));
}
fragment uint4 blit_fs_uint(QuadOut in [[stage_in]], constant BlitUintArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    uint2 c = uint2(in.uv);
    return a.src.read(c, q.slice, uint(q.lod));
}
struct BlitDepthArgs { depth2d_array<float> src; };
struct DepthOut { float d [[depth(any)]]; };
fragment DepthOut blit_fs_depth(QuadOut in [[stage_in]], constant BlitDepthArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    DepthOut o;
    o.d = a.src.read(uint2(in.uv), q.slice, uint(q.lod));
    return o;
}
struct BlitStencilArgs { texture2d_array<uint> src; };
struct StencilOut { uint s [[stencil]]; };
fragment StencilOut blit_fs_stencil(QuadOut in [[stage_in]], constant BlitStencilArgs& a [[buffer(1)]], constant QuadArgs& q [[buffer(0)]]) {
    StencilOut o;
    o.s = a.src.read(uint2(in.uv), q.slice, uint(q.lod)).r;
    return o;
}
// Multisample -> single-sample copy of sample 0 (integer formats have no Metal resolve).
struct MSIntArgs { texture2d_ms<int> src; };
struct MSUintArgs { texture2d_ms<uint> src; };
fragment int4 resolve_fs_int(QuadOut in [[stage_in]], constant MSIntArgs& a [[buffer(1)]]) { return a.src.read(uint2(in.uv), 0); }
fragment uint4 resolve_fs_uint(QuadOut in [[stage_in]], constant MSUintArgs& a [[buffer(1)]]) { return a.src.read(uint2(in.uv), 0); }
// Multisample images (METALGLDRV_STRICT, draw.mm): per-sample 2D-array shadows of multisample
// textures, slice = layer * samples + sample.
#define MS_SHADOW(T, sfx)                                                                                              \
struct MSShadowArgs_##sfx { texture2d_ms<T> ms; texture2d_array<T, access::write> shadow; };                          \
kernel void ms_to_shadow_##sfx(constant MSShadowArgs_##sfx& a [[buffer(0)]], uint3 g [[thread_position_in_grid]]) {   \
    if (g.x >= a.shadow.get_width() || g.y >= a.shadow.get_height() || g.z >= a.shadow.get_array_size()) return;    \
    a.shadow.write(a.ms.read(g.xy, g.z), g.xy, g.z);                                                                  \
}                                                                                                                     \
struct MSArrayShadowArgs_##sfx { texture2d_ms_array<T> ms; texture2d_array<T, access::write> shadow; };               \
kernel void msa_to_shadow_##sfx(constant MSArrayShadowArgs_##sfx& a [[buffer(0)]], uint3 g [[thread_position_in_grid]]) { \
    if (g.x >= a.shadow.get_width() || g.y >= a.shadow.get_height() || g.z >= a.shadow.get_array_size()) return;    \
    uint n = a.ms.get_num_samples();                                                                                  \
    a.shadow.write(a.ms.read(g.xy, g.z / n, g.z % n), g.xy, g.z);                                                     \
}                                                                                                                     \
struct ShadowArgs_##sfx { texture2d_array<T> shadow; };                                                              \
fragment vec<T, 4> shadow_to_ms_##sfx(QuadOut in [[stage_in]], uint s [[sample_id]], constant ShadowArgs_##sfx& a [[buffer(1)]], \
                                      constant QuadArgs& q [[buffer(0)]]) {                                           \
    return a.shadow.read(uint2(in.pos.xy), q.slice + s);                                                             \
}
MS_SHADOW(float, f)
MS_SHADOW(int, i)
MS_SHADOW(uint, u)
// GL_RGB32{F,I,UI} buffer textures: tightly packed 12-byte texels -> RGBA32 (alpha = 1).
kernel void expand_rgb32(device const uint* src [[buffer(0)]], device uint4* dst [[buffer(1)]], constant uint2& p [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.x) return;
    dst[i] = uint4(src[3 * i], src[3 * i + 1], src[3 * i + 2], p.y);
}
// glReadPixels into a pack buffer: 8-bit RGBA <-> BGRA (and alpha = 1 for RGB buffers) in place.
struct PackRgba8Args { uint width, height, rowWords, swapRB, alphaOr; };
kernel void pack_rgba8(device uint* px [[buffer(0)]], constant PackRgba8Args& a [[buffer(1)]], uint2 g [[thread_position_in_grid]]) {
    if (g.x >= a.width || g.y >= a.height) return;
    uint i = g.y * a.rowWords + g.x, c = px[i];
    if (a.swapRB) c = (c & 0xff00ff00u) | ((c >> 16) & 0xffu) | ((c & 0xffu) << 16);
    px[i] = c | a.alphaOr;
}
)MSL";

id<MTLLibrary> internalLibrary() {
    static id<MTLLibrary> lib = [] {
        std::string err;
        id<MTLLibrary> l = Device::get()->compileLibrary(kInternalMSL, &err);
        if (!l) MGL_ERR("internal shader library failed: %s", err.c_str());
        return l;
    }();
    return lib;
}

static id<MTLFunction> internalFn(NSString* name) { return [internalLibrary() newFunctionWithName:name]; }

static MTL4LibraryFunctionDescriptor* fnDesc(id<MTLLibrary> lib, NSString* name) {
    MTL4LibraryFunctionDescriptor* f = [MTL4LibraryFunctionDescriptor new];
    f.library = lib;
    f.name = name;
    return f;
}

struct InternalPipeKey {
    std::string key;
};

static std::mutex g_pipeMutex;
static std::unordered_map<std::string, id<MTLRenderPipelineState>> g_pipes;
static std::unordered_map<std::string, id<MTLDepthStencilState>> g_dss;

static id<MTLRenderPipelineState> internalPipeline(const std::string& key, id<MTLLibrary> lib, NSString* vs, NSString* fs,
                                                   const MTLPixelFormat* colors, const uint8_t* writeMasks, int samples) {
    std::lock_guard<std::mutex> g(g_pipeMutex);
    auto it = g_pipes.find(key);
    if (it != g_pipes.end()) return it->second;
    MTL4RenderPipelineDescriptor* d = [MTL4RenderPipelineDescriptor new];
    d.vertexFunctionDescriptor = fnDesc(lib, vs);
    if (fs) d.fragmentFunctionDescriptor = fnDesc(lib, fs);
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        d.colorAttachments[i].pixelFormat = colors[i];
        if (colors[i] != MTLPixelFormatInvalid) {
            MTLColorWriteMask m = MTLColorWriteMaskNone;
            uint8_t w = writeMasks ? writeMasks[i] : 0xF;
            if (w & 1) m |= MTLColorWriteMaskRed;
            if (w & 2) m |= MTLColorWriteMaskGreen;
            if (w & 4) m |= MTLColorWriteMaskBlue;
            if (w & 8) m |= MTLColorWriteMaskAlpha;
            d.colorAttachments[i].writeMask = m;
        }
    }
    d.rasterSampleCount = std::max(1, samples);
    d.inputPrimitiveTopology = MTLPrimitiveTopologyClassTriangle;
    NSError* err = nil;
    id<MTLRenderPipelineState> p = [Device::get()->compiler newRenderPipelineStateWithDescriptor:d compilerTaskOptions:nil error:&err];
    if (!p) MGL_ERR("internal pipeline %s failed: %s", key.c_str(), err.localizedDescription.UTF8String);
    g_pipes[key] = p;
    return p;
}

static id<MTLDepthStencilState> internalDSS(bool depthWrite, bool stencilWrite, uint32_t stencilMask, bool depthAlways = true) {
    std::string key = std::to_string(depthWrite) + std::to_string(stencilWrite) + "_" + std::to_string(stencilMask);
    std::lock_guard<std::mutex> g(g_pipeMutex);
    auto it = g_dss.find(key);
    if (it != g_dss.end()) return it->second;
    MTLDepthStencilDescriptor* d = [MTLDepthStencilDescriptor new];
    d.depthCompareFunction = MTLCompareFunctionAlways;
    d.depthWriteEnabled = depthWrite;
    if (stencilWrite) {
        MTLStencilDescriptor* s = [MTLStencilDescriptor new];
        s.stencilCompareFunction = MTLCompareFunctionAlways;
        s.depthStencilPassOperation = MTLStencilOperationReplace;
        s.stencilFailureOperation = MTLStencilOperationReplace;
        s.depthFailureOperation = MTLStencilOperationReplace;
        s.writeMask = stencilMask;
        s.readMask = 0xff;
        d.frontFaceStencil = s;
        d.backFaceStencil = s;
    }
    id<MTLDepthStencilState> st = [Device::get()->mtl newDepthStencilStateWithDescriptor:d];
    g_dss[key] = st;
    return st;
}

struct QuadArgsCPU {
    float rect[4];
    float srcRect[4];
    float depth;
    float lod;
    uint32_t slice;
    uint32_t pad;
};

// Upscales the surface color image to the drawable size with MetalFX (spatial).
static id<MTLTexture> metalfxUpscale(Context* ctx, Surface* s, int outW, int outH) {
    if (![MTLFXSpatialScalerDescriptor supportsMetal4FX:ctx->dev->mtl]) return nil;
    id<MTL4FXSpatialScaler> scaler = s->upscaler;
    if (!scaler || scaler.inputWidth != (NSUInteger)s->width || scaler.inputHeight != (NSUInteger)s->height ||
        scaler.outputWidth != (NSUInteger)outW || scaler.outputHeight != (NSUInteger)outH ||
        scaler.colorTextureFormat != s->color.pixelFormat) {  // Surface::useSRGBStorage changes it
        MTLFXSpatialScalerDescriptor* d = [MTLFXSpatialScalerDescriptor new];
        d.colorTextureFormat = s->color.pixelFormat;
        d.outputTextureFormat = s->color.pixelFormat;
        d.inputWidth = s->width;
        d.inputHeight = s->height;
        d.outputWidth = outW;
        d.outputHeight = outH;
        d.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;
        scaler = [d newSpatialScalerWithDevice:ctx->dev->mtl compiler:ctx->dev->compiler];
        if (!scaler) {
            static bool warned = false;
            if (!warned) MGL_WARN("MetalFX spatial scaler unavailable; using plain scaling");
            warned = true;
            return nil;
        }
        if (s->upscaled) ctx->dev->releaseAfter(s->upscaled, s->lastUse);
        MTLTextureDescriptor* td = texture2DDescriptor(s->color.pixelFormat, outW, outH);
        td.storageMode = MTLStorageModePrivate;
        td.usage = scaler.outputTextureUsage | MTLTextureUsageShaderRead;
        s->upscaled = [ctx->dev->mtl newTextureWithDescriptor:td];
        ctx->dev->makeResident(s->upscaled);
        if (s->upscaler) ctx->dev->retainUntil(s->upscaler, s->lastUse);
        s->upscaler = scaler;
        if (!s->upscaleFence) s->upscaleFence = [ctx->dev->mtl newFence];
        MGL_INFO("MetalFX spatial upscaling %dx%d -> %dx%d", s->width, s->height, outW, outH);
    }
    // Order the scaler after all rendering: an empty encoder waits on prior work and signals the fence.
    ctx->endEncoder();
    id<MTL4ComputeCommandEncoder> ce = ctx->newComputeEncoder();
    [ce barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    [ce updateFence:s->upscaleFence afterEncoderStages:MTLStageDispatch];
    [ce endEncoding];
    scaler.colorTexture = s->color;
    scaler.outputTexture = s->upscaled;
    scaler.inputContentWidth = s->width;
    scaler.inputContentHeight = s->height;
    scaler.fence = s->upscaleFence;
    [scaler encodeToCommandBuffer:ctx->commandBuffer()];
    return s->upscaled;
}

void presentSurface(Context* ctx, Surface* s, id<CAMetalDrawable> drawable) {
    id<MTLTexture> dst = drawable.texture;
    // Resolve MSAA default framebuffer first.
    ctx->endEncoder();
    if (s->msColor) {
        MTL4RenderPassDescriptor* rrp = [MTL4RenderPassDescriptor new];
        rrp.colorAttachments[0].texture = s->msColor;
        rrp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rrp.colorAttachments[0].storeAction = MTLStoreActionStoreAndMultisampleResolve;
        rrp.colorAttachments[0].resolveTexture = s->color;
        id<MTL4RenderCommandEncoder> re = ctx->newRenderEncoder(rrp);
        [re barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment visibilityOptions:MTL4VisibilityOptionDevice];
        [re endEncoding];
    }
    id<MTLTexture> srcTex = s->color;
    if (s->outWidth > s->width && (int)dst.width == s->outWidth && (int)dst.height == s->outHeight) {
        if (id<MTLTexture> up = metalfxUpscale(ctx, s, s->outWidth, s->outHeight)) srcTex = up;
    }
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    id<MTL4RenderCommandEncoder> e = ctx->newRenderEncoder(rp);
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment | MTLStageVertex visibilityOptions:MTL4VisibilityOptionDevice];
    [e setArgumentTable:ctx->renderArgTable atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    MTLPixelFormat colors[kMaxDrawBuffers] = {dst.pixelFormat};
    for (int i = 1; i < kMaxDrawBuffers; i++) colors[i] = MTLPixelFormatInvalid;
    std::string key = "present_" + std::to_string(dst.pixelFormat);
    id<MTLRenderPipelineState> pso = internalPipeline(key, internalLibrary(), @"quad_vs", @"present_fs", colors, nullptr, 1);
    [e setRenderPipelineState:pso];
    // Flip: GL row 0 (bottom) is stored at Metal row 0; display needs it at the bottom.
    UploadAlloc qa = ctx->upload(sizeof(QuadArgsCPU));
    QuadArgsCPU q = {{-1, 1, 1, -1}, {0, 1, 1, 0}, 0, 0, 0, 0};
    // rect y is negated in the VS: y=1 -> Metal top (NDC +1 after negation is -1?) handled by passing flipped rect.
    q.rect[0] = -1; q.rect[1] = -1; q.rect[2] = 1; q.rect[3] = 1;
    // GL-convention rect covers the drawable; vs negates y so NDC y=-1 maps to Metal top. We want displayed top
    // (Metal row 0 of drawable) to show GL top (highest GL row = last Metal row of src) -> sample v = 1 at Metal top.
    q.srcRect[0] = 0; q.srcRect[1] = 1; q.srcRect[2] = 1; q.srcRect[3] = 0;
    memcpy(qa.cpu, &q, sizeof q);
    UploadAlloc pa = ctx->upload(8);
    MTLResourceID rid = srcTex.gpuResourceID;
    memcpy(pa.cpu, &rid, 8);
    [ctx->renderArgTable setAddress:qa.gpu atIndex:0];
    [ctx->renderArgTable setAddress:pa.gpu atIndex:1];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    [e endEncoding];
    s->lastUse = ctx->serial();
}

void clearWithQuad(Context* ctx, const QuadClear& q) {
    id<MTL4RenderCommandEncoder> e = ctx->renderEncoder();
    if (!e) return;
    const PassDesc& p = ctx->curPass;
    // Build a fragment shader for this attachment signature.
    MTLPixelFormat colors[kMaxDrawBuffers];
    std::string sig;
    std::string body;
    std::string outStruct = "struct ClearOut {\n";
    bool anyColor = false;
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        colors[i] = p.color[i].tex ? p.color[i].tex.pixelFormat : MTLPixelFormatInvalid;
        if (!p.color[i].tex || !(q.drawBufferMask & (1u << i))) continue;
        const FormatInfo* f = formatInfoForMTL(colors[i]);
        const char* ty = "float4";
        if (f && f->kind == FmtKind::UInt) ty = "uint4";
        if (f && f->kind == FmtKind::SInt) ty = "int4";
        outStruct += std::string("  ") + ty + " c" + std::to_string(i) + " [[color(" + std::to_string(i) + ")]];\n";
        body += "  o.c" + std::to_string(i) + " = as_type<" + ty + ">(v[" + std::to_string(i) + "]);\n";
        sig += std::to_string(i) + ty[0];
        anyColor = true;
    }
    outStruct += anyColor ? "};\n" : "  float dummy [[depth(any)]];\n};\n";
    if (!anyColor) body = "  o.dummy = in.pos.z;\n";
    std::string libKey = "clearlib_" + sig;
    static std::mutex m;
    static std::unordered_map<std::string, id<MTLLibrary>> libs;
    id<MTLLibrary> lib;
    {
        std::lock_guard<std::mutex> g(m);
        lib = libs[libKey];
        if (!lib) {
            std::string src = std::string(kInternalMSL) + outStruct +
                              "fragment ClearOut clear_fs(QuadOut in [[stage_in]], constant uint4* v [[buffer(1)]]) {\n  ClearOut o;\n" +
                              body + "  return o;\n}\n";
            std::string err;
            lib = Device::get()->compileLibrary(src, &err);
            if (!lib) {
                MGL_ERR("clear shader failed: %s", err.c_str());
                return;
            }
            libs[libKey] = lib;
        }
    }
    uint8_t masks[kMaxDrawBuffers];
    // a layered framebuffer is cleared in every layer
    bool layered = p.layers > 1;
    std::string key = "clear_" + sig + "_s" + std::to_string(p.samples) + (layered ? "_L" : "");
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        masks[i] = (q.drawBufferMask & (1u << i)) ? q.colorWriteMask[i] : 0;
        if (p.color[i].noAlpha) masks[i] &= ~8u;  // alpha-less default framebuffer keeps alpha at 1
        key += "_" + std::to_string(colors[i]) + ":" + std::to_string(masks[i]);
    }
    id<MTLRenderPipelineState> pso = internalPipeline(key, lib, layered ? @"quad_vs_layered" : @"quad_vs", @"clear_fs", colors, masks, p.samples);
    if (!pso) return;
    [e setRenderPipelineState:pso];
    ctx->enc.pipeline = pso;
    bool hasDepth = p.depth.tex != nil, hasStencil = p.stencil.tex != nil;
    [e setDepthStencilState:internalDSS(q.depth && hasDepth, q.stencil && hasStencil, q.stencilWriteMask)];
    ctx->enc.depthStencil = nil;
    if (q.stencil) {
        [e setStencilReferenceValue:q.stencilValue];
        ctx->enc.stencilRef[0] = ctx->enc.stencilRef[1] = ~0u;
    }
    MTLViewport vp = {0, 0, (double)p.width, (double)p.height, 0, 1};
    [e setViewport:vp];
    ctx->enc.numViewports = 0;
    ctx->enc.numScissors = 0;
    MTLScissorRect sc = q.scissored ? q.scissor : MTLScissorRect{0, 0, (NSUInteger)p.width, (NSUInteger)p.height};
    [e setScissorRect:sc];
    [e setCullMode:MTLCullModeNone];
    [e setTriangleFillMode:MTLTriangleFillModeFill];
    [e setDepthBias:0 slopeScale:0 clamp:0];
    [e setDepthClipMode:MTLDepthClipModeClamp];
    ctx->enc.cull = ctx->enc.fill = ctx->enc.depthClip = -1;
    ctx->enc.depthBias[0] = NAN;
    UploadAlloc qa = ctx->upload(sizeof(QuadArgsCPU));
    QuadArgsCPU qq = {{-1, -1, 1, 1}, {0, 0, 1, 1}, std::clamp(q.depthValue, 0.0f, 1.0f), 0, 0, 0};
    memcpy(qa.cpu, &qq, sizeof qq);
    UploadAlloc va = ctx->upload(16 * kMaxDrawBuffers);
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        const FormatInfo* f = colors[i] != MTLPixelFormatInvalid ? formatInfoForMTL(colors[i]) : nullptr;
        if (f && (f->kind == FmtKind::UInt || f->kind == FmtKind::SInt)) memcpy(va.cpu + 16 * i, q.colorInt[i], 16);
        else memcpy(va.cpu + 16 * i, q.color[i], 16);
    }
    [ctx->renderArgTable setAddress:qa.gpu atIndex:0];
    [ctx->renderArgTable setAddress:va.gpu atIndex:1];
    [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4 instanceCount:layered ? p.layers : 1];
}

void resolveSample0(Context* ctx, id<MTLTexture> src, id<MTLTexture> dst, bool isSigned) {
    ctx->endEncoder();
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    rp.colorAttachments[0].texture = dst;
    rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
    rp.colorAttachments[0].storeAction = MTLStoreActionStore;
    rp.renderTargetWidth = dst.width;
    rp.renderTargetHeight = dst.height;
    MTLPixelFormat colors[kMaxDrawBuffers];
    for (int i = 0; i < kMaxDrawBuffers; i++) colors[i] = MTLPixelFormatInvalid;
    colors[0] = dst.pixelFormat;
    NSString* fs = isSigned ? @"resolve_fs_int" : @"resolve_fs_uint";
    std::string key = std::string("resolve_") + fs.UTF8String + "_" + std::to_string(colors[0]);
    id<MTLRenderPipelineState> pso = internalPipeline(key, internalLibrary(), @"quad_vs", fs, colors, nullptr, 1);
    id<MTL4RenderCommandEncoder> e = ctx->newRenderEncoder(rp);
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment | MTLStageVertex visibilityOptions:MTL4VisibilityOptionDevice];
    if (pso) {
        [e setArgumentTable:ctx->renderArgTable atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        [e setRenderPipelineState:pso];
        MTLViewport vp = {0, 0, (double)dst.width, (double)dst.height, 0, 1};
        [e setViewport:vp];
        UploadAlloc qa = ctx->upload(sizeof(QuadArgsCPU));
        QuadArgsCPU q;
        q.rect[0] = -1; q.rect[1] = -1; q.rect[2] = 1; q.rect[3] = 1;
        q.srcRect[0] = 0; q.srcRect[1] = 0; q.srcRect[2] = (float)src.width; q.srcRect[3] = (float)src.height;
        q.depth = 0;
        q.lod = 0;
        q.slice = 0;
        q.pad = 0;
        memcpy(qa.cpu, &q, sizeof q);
        UploadAlloc pa = ctx->upload(8);
        MTLResourceID rid = src.gpuResourceID;
        memcpy(pa.cpu, &rid, 8);
        [ctx->renderArgTable setAddress:qa.gpu atIndex:0];
        [ctx->renderArgTable setAddress:pa.gpu atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    [e endEncoding];
}

static const char* const kShadowSuffix[3] = {"f", "i", "u"};

void msImageToShadow(Context* ctx, id<MTLTexture> ms, id<MTLTexture> shadow, int kind) {
    static std::mutex m;
    static id<MTLComputePipelineState> psos[2][3];
    bool arrayed = ms.textureType == MTLTextureType2DMultisampleArray;
    id<MTLComputePipelineState> pso;
    {
        std::lock_guard<std::mutex> g(m);
        __strong id<MTLComputePipelineState>& p = psos[arrayed][kind];
        if (!p) {
            MTL4ComputePipelineDescriptor* d = [MTL4ComputePipelineDescriptor new];
            d.computeFunctionDescriptor =
                fnDesc(internalLibrary(), [NSString stringWithFormat:@"%s_to_shadow_%s", arrayed ? "msa" : "ms", kShadowSuffix[kind]]);
            NSError* err = nil;
            p = [Device::get()->compiler newComputePipelineStateWithDescriptor:d compilerTaskOptions:nil error:&err];
            if (!p) MGL_ERR("ms_to_shadow pipeline failed: %s", err.localizedDescription.UTF8String);
        }
        pso = p;
    }
    if (!pso) return;
    UploadAlloc aa = ctx->upload(16, 8);
    MTLResourceID rids[2] = {ms.gpuResourceID, shadow.gpuResourceID};
    memcpy(aa.cpu, rids, sizeof rids);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e barrierAfterEncoderStages:MTLStageDispatch | MTLStageBlit beforeEncoderStages:MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    [e setComputePipelineState:pso];
    [ctx->computeArgTable setAddress:aa.gpu atIndex:0];
    [e dispatchThreads:MTLSizeMake(shadow.width, shadow.height, shadow.arrayLength) threadsPerThreadgroup:MTLSizeMake(8, 8, 1)];
    [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
}

void msImageFromShadow(Context* ctx, id<MTLTexture> shadow, id<MTLTexture> ms, int kind) {
    ctx->endEncoder();
    MTLPixelFormat colors[kMaxDrawBuffers];
    for (int i = 0; i < kMaxDrawBuffers; i++) colors[i] = MTLPixelFormatInvalid;
    colors[0] = ms.pixelFormat;
    NSString* fs = [NSString stringWithFormat:@"shadow_to_ms_%s", kShadowSuffix[kind]];
    std::string key = std::string("shadow_to_ms_") + kShadowSuffix[kind] + "_" + std::to_string(colors[0]) + "_" + std::to_string(ms.sampleCount);
    id<MTLRenderPipelineState> pso = internalPipeline(key, internalLibrary(), @"quad_vs", fs, colors, nullptr, (int)ms.sampleCount);
    if (!pso) return;
    NSUInteger layers = ms.textureType == MTLTextureType2DMultisampleArray ? ms.arrayLength : 1;
    UploadAlloc pa = ctx->upload(8);
    MTLResourceID rid = shadow.gpuResourceID;
    memcpy(pa.cpu, &rid, 8);
    for (NSUInteger l = 0; l < layers; l++) {
        // every sample of every pixel is written: per-sample shading ([[sample_id]]) of a full quad
        MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
        rp.colorAttachments[0].texture = ms;
        rp.colorAttachments[0].slice = l;
        rp.colorAttachments[0].loadAction = MTLLoadActionDontCare;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        rp.renderTargetWidth = ms.width;
        rp.renderTargetHeight = ms.height;
        id<MTL4RenderCommandEncoder> e = ctx->newRenderEncoder(rp);
        [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment | MTLStageVertex visibilityOptions:MTL4VisibilityOptionDevice];
        [e setArgumentTable:ctx->renderArgTable atStages:MTLRenderStageVertex | MTLRenderStageFragment];
        [e setRenderPipelineState:pso];
        MTLViewport vp = {0, 0, (double)ms.width, (double)ms.height, 0, 1};
        [e setViewport:vp];
        UploadAlloc qa = ctx->upload(sizeof(QuadArgsCPU));
        QuadArgsCPU q;
        q.rect[0] = -1; q.rect[1] = -1; q.rect[2] = 1; q.rect[3] = 1;
        q.srcRect[0] = 0; q.srcRect[1] = 0; q.srcRect[2] = 0; q.srcRect[3] = 0;
        q.depth = 0;
        q.lod = 0;
        q.slice = (uint32_t)(l * ms.sampleCount);
        q.pad = 0;
        memcpy(qa.cpu, &q, sizeof q);
        [ctx->renderArgTable setAddress:qa.gpu atIndex:0];
        [ctx->renderArgTable setAddress:pa.gpu atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
        [e endEncoding];
    }
}

void expandRgb32(Context* ctx, uint64_t srcAddr, uint64_t dstAddr, uint32_t texels, uint32_t alphaBits) {
    static id<MTLComputePipelineState> pso = [] {
        MTL4ComputePipelineDescriptor* d = [MTL4ComputePipelineDescriptor new];
        d.computeFunctionDescriptor = fnDesc(internalLibrary(), @"expand_rgb32");
        NSError* err = nil;
        id<MTLComputePipelineState> p = [Device::get()->compiler newComputePipelineStateWithDescriptor:d compilerTaskOptions:nil error:&err];
        if (!p) MGL_ERR("expand_rgb32 pipeline failed: %s", err.localizedDescription.UTF8String);
        return p;
    }();
    if (!pso || !texels) return;
    UploadAlloc pa = ctx->upload(8);
    uint32_t p[2] = {texels, alphaBits};
    memcpy(pa.cpu, p, 8);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e setComputePipelineState:pso];
    [ctx->computeArgTable setAddress:srcAddr atIndex:0];
    [ctx->computeArgTable setAddress:dstAddr atIndex:1];
    [ctx->computeArgTable setAddress:pa.gpu atIndex:2];
    [e dispatchThreads:MTLSizeMake(texels, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
}

void packRgba8(Context* ctx, uint64_t addr, uint32_t width, uint32_t height, uint32_t rowBytes, bool swapRB, bool opaque) {
    static id<MTLComputePipelineState> pso = [] {
        MTL4ComputePipelineDescriptor* d = [MTL4ComputePipelineDescriptor new];
        d.computeFunctionDescriptor = fnDesc(internalLibrary(), @"pack_rgba8");
        NSError* err = nil;
        id<MTLComputePipelineState> p = [Device::get()->compiler newComputePipelineStateWithDescriptor:d compilerTaskOptions:nil error:&err];
        if (!p) MGL_ERR("pack_rgba8 pipeline failed: %s", err.localizedDescription.UTF8String);
        return p;
    }();
    if (!pso || !width || !height) return;
    UploadAlloc pa = ctx->upload(20);
    uint32_t p[5] = {width, height, rowBytes / 4, swapRB, opaque ? 0xff000000u : 0u};
    memcpy(pa.cpu, p, sizeof p);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e setComputePipelineState:pso];
    [ctx->computeArgTable setAddress:addr atIndex:0];
    [ctx->computeArgTable setAddress:pa.gpu atIndex:1];
    [e dispatchThreads:MTLSizeMake(width, height, 1) threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [e barrierAfterEncoderStages:MTLStageDispatch beforeEncoderStages:MTLStageDispatch | MTLStageBlit visibilityOptions:MTL4VisibilityOptionDevice];
}

void blitToTexture(Context* ctx, const BlitParams& bpIn, id<MTLTexture> dst, int dstLevel, int dstSlice) {
    ctx->endEncoder();
    BlitParams bp = bpIn;
    if (bp.src.textureType == MTLTextureType3D) {
        // A 3D texture can't be viewed as a 2D array; copy the source depth slice
        // into a temporary single-level 2D texture and sample that instead.
        id<MTLTexture> s = bp.src;
        int lw = std::max(1, (int)(s.width >> bp.srcLevel)), lh = std::max(1, (int)(s.height >> bp.srcLevel));
        MTLTextureDescriptor* td = texture2DDescriptor(s.pixelFormat, lw, lh);
        td.usage = MTLTextureUsageShaderRead;
        td.storageMode = MTLStorageModePrivate;
        id<MTLTexture> tmp = [ctx->dev->mtl newTextureWithDescriptor:td];
        if (!tmp) return;
        Device::get()->retainUntil(tmp, ctx->serial());
        id<MTL4ComputeCommandEncoder> ce = ctx->computeEncoder();
        [ce copyFromTexture:s sourceSlice:0 sourceLevel:bp.srcLevel sourceOrigin:MTLOriginMake(0, 0, bp.srcSlice) sourceSize:MTLSizeMake(lw, lh, 1)
                  toTexture:tmp destinationSlice:0 destinationLevel:0 destinationOrigin:MTLOriginMake(0, 0, 0)];
        ctx->endEncoder();
        bp.src = tmp;
        bp.srcLevel = 0;
        bp.srcSlice = 0;
    }
    MTL4RenderPassDescriptor* rp = [MTL4RenderPassDescriptor new];
    MTLPixelFormat colors[kMaxDrawBuffers];
    for (int i = 0; i < kMaxDrawBuffers; i++) colors[i] = MTLPixelFormatInvalid;
    NSString* fs;
    const FormatInfo* sf = formatInfoForMTL(bp.src.pixelFormat);
    if (bp.what == BlitParams::Color) {
        rp.colorAttachments[0].texture = dst;
        rp.colorAttachments[0].level = dstLevel;
        if (dst.textureType == MTLTextureType3D) rp.colorAttachments[0].depthPlane = dstSlice;
        else rp.colorAttachments[0].slice = dstSlice;
        rp.colorAttachments[0].loadAction = MTLLoadActionLoad;
        rp.colorAttachments[0].storeAction = MTLStoreActionStore;
        colors[0] = dst.pixelFormat;
        if (sf && sf->kind == FmtKind::UInt) fs = @"blit_fs_uint";
        else if (sf && sf->kind == FmtKind::SInt) fs = @"blit_fs_int";
        else fs = bp.linear ? @"blit_fs_linear" : @"blit_fs_nearest";
    } else if (bp.what == BlitParams::Depth) {
        rp.depthAttachment.texture = dst;
        rp.depthAttachment.level = dstLevel;
        rp.depthAttachment.slice = dstSlice;
        rp.depthAttachment.loadAction = MTLLoadActionLoad;
        rp.depthAttachment.storeAction = MTLStoreActionStore;
        fs = @"blit_fs_depth";
    } else {
        rp.stencilAttachment.texture = dst;
        rp.stencilAttachment.level = dstLevel;
        rp.stencilAttachment.slice = dstSlice;
        rp.stencilAttachment.loadAction = MTLLoadActionLoad;
        rp.stencilAttachment.storeAction = MTLStoreActionStore;
        fs = @"blit_fs_stencil";
    }
    int dw = std::max(1, (int)(dst.width >> dstLevel)), dh = std::max(1, (int)(dst.height >> dstLevel));
    rp.renderTargetWidth = dw;
    rp.renderTargetHeight = dh;
    id<MTL4RenderCommandEncoder> e = ctx->newRenderEncoder(rp);
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageFragment | MTLStageVertex visibilityOptions:MTL4VisibilityOptionDevice];
    [e setArgumentTable:ctx->renderArgTable atStages:MTLRenderStageVertex | MTLRenderStageFragment];
    std::string key = std::string("blit_") + fs.UTF8String + "_" + std::to_string(colors[0]) + "_" + std::to_string(dst.sampleCount);
    id<MTLRenderPipelineState> pso = internalPipeline(key, internalLibrary(), @"quad_vs", fs, colors, nullptr, (int)dst.sampleCount);
    if (pso) {
        [e setRenderPipelineState:pso];
        if (bp.what == BlitParams::Depth) [e setDepthStencilState:internalDSS(true, false, 0)];
        if (bp.what == BlitParams::Stencil) {
            [e setDepthStencilState:internalDSS(false, true, 0xff)];
        }
        MTLViewport vp = {0, 0, (double)dw, (double)dh, 0, 1};
        [e setViewport:vp];
        if (bp.scissored) [e setScissorRect:bp.scissor];
        UploadAlloc qa = ctx->upload(sizeof(QuadArgsCPU));
        QuadArgsCPU q;
        // dst rect pixels -> NDC (GL convention; VS negates y)
        q.rect[0] = bp.dstRect[0] / dw * 2 - 1;
        q.rect[1] = bp.dstRect[1] / dh * 2 - 1;
        q.rect[2] = bp.dstRect[2] / dw * 2 - 1;
        q.rect[3] = bp.dstRect[3] / dh * 2 - 1;
        int sw = std::max(1, (int)(bp.src.width >> bp.srcLevel)), sh = std::max(1, (int)(bp.src.height >> bp.srcLevel));
        bool normalized = bp.what == BlitParams::Color && !(sf && (sf->kind == FmtKind::UInt || sf->kind == FmtKind::SInt));
        for (int k = 0; k < 4; k++) q.srcRect[k] = normalized ? bp.srcRect[k] / ((k & 1) ? sh : sw) : bp.srcRect[k];
        q.depth = 0;
        q.lod = bp.srcLevel;
        q.slice = bp.srcSlice;
        q.pad = 0;
        memcpy(qa.cpu, &q, sizeof q);
        // The blit shaders use array textures; create a 2D-array view if needed.
        id<MTLTexture> src = bp.src;
        if (src.textureType != MTLTextureType2DArray) {
            MTLPixelFormat vf = src.pixelFormat;
            if (bp.what == BlitParams::Stencil) {
                vf = vf == MTLPixelFormatDepth32Float_Stencil8 ? MTLPixelFormatX32_Stencil8 : vf;
            }
            NSUInteger slices = src.textureType == MTLTextureTypeCube ? 6 : src.textureType == MTLTextureTypeCubeArray ? src.arrayLength * 6 : src.arrayLength;
            if (src.textureType == MTLTextureType2DMultisample) {
                MGL_WARN("blit from multisample texture without resolve");
            }
            src = [src newTextureViewWithPixelFormat:vf textureType:MTLTextureType2DArray levels:NSMakeRange(0, src.mipmapLevelCount)
                                              slices:NSMakeRange(0, slices)];
            Device::get()->retainUntil(src, ctx->serial());
        } else if (bp.what == BlitParams::Stencil && src.pixelFormat == MTLPixelFormatDepth32Float_Stencil8) {
            src = [src newTextureViewWithPixelFormat:MTLPixelFormatX32_Stencil8];
            Device::get()->retainUntil(src, ctx->serial());
        }
        UploadAlloc pa = ctx->upload(8);
        MTLResourceID rid = src.gpuResourceID;
        memcpy(pa.cpu, &rid, 8);
        [ctx->renderArgTable setAddress:qa.gpu atIndex:0];
        [ctx->renderArgTable setAddress:pa.gpu atIndex:1];
        [e drawPrimitives:MTLPrimitiveTypeTriangleStrip vertexStart:0 vertexCount:4];
    }
    [e endEncoding];
}

}  // namespace mgl
