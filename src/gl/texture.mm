// Texture storage, uploads, views and sampler states.
#import "gl/texture.h"
#include "gl/internal_shaders.h"
#include <cmath>
#include <mutex>
#include "mtl/compress.h"
#include "util/log.h"
#include "util/stats.h"

namespace mgl {

Texture::~Texture() {
    dropViews();
    if (mtl) Device::get()->releaseAfter(mtl, std::max(lastUse, lastGpuWrite));
}

void Texture::dropViews() {
    Device* d = Device::get();
    uint64_t s = std::max(lastUse, lastGpuWrite);
    for (auto& [k, v] : views) d->retainUntil(v, s);
    views.clear();
}

Renderbuffer::~Renderbuffer() {
    if (mtl) Device::get()->releaseAfter(mtl, lastUse);
}

bool SamplerState::operator==(const SamplerState& o) const { return memcmp(this, &o, sizeof *this) == 0; }

static int mipCount(int w, int h, int d) {
    int m = std::max(w, std::max(h, d));
    int n = 1;
    while (m > 1) {
        m >>= 1;
        n++;
    }
    return n;
}

MTLTextureType mtlTextureType(GLenum target, int samples) {
    switch (target) {
    case GL_TEXTURE_1D: return MTLTextureType2D;
    case GL_TEXTURE_1D_ARRAY: return MTLTextureType2DArray;
    case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: return MTLTextureType2D;
    case GL_TEXTURE_3D: return MTLTextureType3D;
    case GL_TEXTURE_2D_ARRAY: return MTLTextureType2DArray;
    case GL_TEXTURE_CUBE_MAP: return MTLTextureTypeCube;
    case GL_TEXTURE_CUBE_MAP_ARRAY: return MTLTextureTypeCubeArray;
    case GL_TEXTURE_2D_MULTISAMPLE: return MTLTextureType2DMultisample;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return MTLTextureType2DMultisampleArray;
    case GL_TEXTURE_BUFFER: return MTLTextureTypeTextureBuffer;
    default: return MTLTextureType2D;
    }
}

// Layers (Metal "slices" excluding cube faces) for a level-0 spec.
static int specLayers(const Texture* t, const TexImageLevel& l) {
    switch (t->target) {
    case GL_TEXTURE_1D_ARRAY: return l.height;
    case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return l.depth;
    case GL_TEXTURE_CUBE_MAP_ARRAY: return l.depth / 6;
    default: return 1;
    }
}

static void specDims(const Texture* t, const TexImageLevel& l, int& w, int& h, int& d) {
    w = l.width;
    h = l.height;
    d = 1;
    switch (t->target) {
    case GL_TEXTURE_1D: h = 1; break;
    case GL_TEXTURE_1D_ARRAY: h = 1; break;
    case GL_TEXTURE_3D: d = l.depth; break;
    default: break;
    }
}

// First defined image of a level (cube maps: any face; faces may be specified in any order).
static const TexImageLevel* definedImage(const Texture* t, int level) {
    int faces = t->target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
    for (int f = 0; f < faces; f++)
        if (t->levels[f][level].defined()) return &t->levels[f][level];
    return nullptr;
}

int textureStorageLevelFor(const Texture* t) {
    // Level whose spec defines the storage: level 0 if defined, else the base level.
    if (t->immutable || definedImage(t, 0)) return 0;
    int b = std::clamp(t->baseLevel, 0, Texture::kMaxLevels - 1);
    if (definedImage(t, b)) return b;
    for (int l = 0; l < Texture::kMaxLevels; l++)
        if (definedImage(t, l)) return l;
    return -1;
}

bool ensureTextureStorage(Context* ctx, Texture* t) {
    if (t->target == GL_TEXTURE_BUFFER) return t->mtl != nil || ensureBufferTexture(ctx, t);
    if (t->viewParent) return t->mtl != nil;
    if (!t->storageDirty && t->mtl) return true;
    int ref = textureStorageLevelFor(t);
    if (ref < 0) return false;
    const TexImageLevel& L = t->immutable ? t->levels[0][ref] : *definedImage(t, ref);
    int w, h, d;
    specDims(t, L, w, h, d);
    w = std::max(1, w << ref);
    if (t->target != GL_TEXTURE_1D && t->target != GL_TEXTURE_1D_ARRAY) h = std::max(1, h << ref);
    if (t->target == GL_TEXTURE_3D) d = std::max(1, d << ref);
    int layers = specLayers(t, L);
    const FormatInfo* fmt = L.fmt;
    int levels;
    if (t->immutable) levels = t->immutableLevels;
    else if (t->samples > 0 || t->target == GL_TEXTURE_RECTANGLE) levels = 1;
    else {
        levels = mipCount(w, (t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY) ? 1 : h, d);
    }
    bool same = t->mtl && t->storageFmt == fmt && t->storageW == w && t->storageH == h && t->storageD == d &&
                t->storageLayers == layers && t->storageLevels == levels;
    if (same) {
        t->storageDirty = false;
        return true;
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = mtlTextureType(t->target, t->samples);
    td.pixelFormat = fmt->mtl;
    td.width = w;
    td.height = h;
    td.depth = d;
    td.arrayLength = std::max(1, layers);
    td.mipmapLevelCount = levels;
    if (t->samples > 0) td.sampleCount = t->samples;
    td.storageMode = MTLStorageModePrivate;
    td.hazardTrackingMode = MTLHazardTrackingModeUntracked;
    MTLTextureUsage usage = MTLTextureUsageShaderRead;
    if (fmt->renderable) usage |= MTLTextureUsageRenderTarget;
    if (t->needsShaderWrite) {
        usage |= MTLTextureUsageShaderWrite;
        // Metal allows texture atomics only on 32-bit integer single-channel formats.
        if (fmt->mtl == MTLPixelFormatR32Uint || fmt->mtl == MTLPixelFormatR32Sint) usage |= MTLTextureUsageShaderAtomic;
    }
    if (fmt->srgb || t->needsFormatView || fmt->kind == FmtKind::DepthStencil) usage |= MTLTextureUsagePixelFormatView;
    td.usage = usage;
    if (!losslessCompressionSafe(fmt->mtl)) td.allowGPUOptimizedContents = NO;
    id<MTLTexture> nt = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (nt && g_statsEnabled) statAddCaller("newTexture KB", __builtin_return_address(0), nt.allocatedSize >> 10);
    if (!nt) {
        MGL_ERR("texture allocation failed (%dx%dx%d, %d levels, fmt %d)", w, h, d, levels, (int)fmt->mtl);
        return false;
    }
    if (!t->label.empty()) nt.label = @(t->label.c_str());
    ctx->dev->makeResident(nt);
    id<MTLTexture> old = t->mtl;
    // Migrate compatible contents from the previous storage.
    if (old && t->storageFmt == fmt && old.textureType == nt.textureType && old.sampleCount == nt.sampleCount) {
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
        int faces = t->target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
        for (int l = 0; l < std::min((int)old.mipmapLevelCount, levels); l++) {
            int ow = std::max(1, (int)old.width >> l), oh = std::max(1, (int)old.height >> l);
            int nw = std::max(1, w >> l), nh = std::max(1, h >> l);
            if (ow != nw || oh != nh) continue;
            int slices = std::min((int)old.arrayLength, std::max(1, layers)) * faces;
            if (t->target == GL_TEXTURE_CUBE_MAP_ARRAY) slices = std::min((int)old.arrayLength, layers) * 6;
            [e copyFromTexture:old sourceSlice:0 sourceLevel:l toTexture:nt destinationSlice:0 destinationLevel:l sliceCount:slices levelCount:1];
        }
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        ctx->dev->releaseAfter(old, ctx->serial());
    } else if (old) {
        ctx->dev->releaseAfter(old, std::max(t->lastUse, t->lastGpuWrite));
    }
    t->mtl = nt;
    t->storageFmt = fmt;
    t->storageW = w;
    t->storageH = h;
    t->storageD = d;
    t->storageLayers = layers;
    t->storageLevels = levels;
    t->storageDirty = false;
    t->dropViews();
    t->lastUse = ctx->serial();
    return true;
}

bool ensureBufferTexture(Context* ctx, Texture* t) {
    if (!t->buffer || !t->buffer->mtl) return false;
    const FormatInfo* f = formatInfo(t->bufferFormat);
    if (!f) return false;
    Buffer* b = t->buffer.get();
    bool rgb32 = bufferTexelBytes(f) == 12;
    if (t->mtl && t->bufferGeneration == b->generation &&
        (!rgb32 || (t->expandedVersion == b->contentVersion && t->expandedGpuWrite == b->lastGpuWrite)))
        return true;
    GLsizeiptr size = t->bufferSize < 0 ? b->size - t->bufferOffset : t->bufferSize;
    size = std::min<GLsizeiptr>(size, b->size - t->bufferOffset);
    size_t texels = size > 0 ? size / bufferTexelBytes(f) : 0;
    texels = std::min<size_t>(texels, (size_t)1 << 27);
    if (texels == 0) texels = 1;
    if (rgb32) {
        size_t need = texels * 16;
        if (!t->bufferExpanded || t->bufferExpanded.length < need || t->bufferGeneration != b->generation || !t->mtl) {
            if (t->bufferExpanded) ctx->dev->releaseAfter(t->bufferExpanded, t->lastUse);
            t->bufferExpanded = ctx->dev->newBuffer(need);
            if (!t->bufferExpanded) return false;
            MTLTextureDescriptor* td = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:f->mtl width:texels
                                                                                   resourceOptions:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                                                                             usage:MTLTextureUsageShaderRead];
            id<MTLTexture> nt = [t->bufferExpanded newTextureWithDescriptor:td offset:0 bytesPerRow:need];
            if (!nt) return false;
            if (t->mtl) ctx->dev->retainUntil(t->mtl, t->lastUse);
            t->mtl = nt;
        }
        uint32_t one = f->kind == FmtKind::Float ? 0x3f800000u : 1u;
        size_t avail = std::max<GLsizeiptr>(0, size) / 12;
        expandRgb32(ctx, b->gpuAddress() + t->bufferOffset, t->bufferExpanded.gpuAddress, (uint32_t)std::min(avail, texels), one);
        b->lastUse = ctx->serial();
        t->expandedVersion = b->contentVersion;
        t->expandedGpuWrite = b->lastGpuWrite;
        t->bufferGeneration = b->generation;
        t->storageFmt = f;
        return true;
    }
    MTLTextureDescriptor* td = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:f->mtl width:texels
                                                                           resourceOptions:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked
                                                                                     usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
    size_t bpr = texels * f->bytesPerPixel;
    bpr = (bpr + 15) & ~(size_t)15;
    id<MTLTexture> nt = [b->mtl newTextureWithDescriptor:td offset:t->bufferOffset bytesPerRow:bpr];
    if (!nt) {
        MGL_ERR("buffer texture view failed (offset %ld, texels %zu)", (long)t->bufferOffset, texels);
        return false;
    }
    if (t->mtl) ctx->dev->retainUntil(t->mtl, t->lastUse);
    t->mtl = nt;
    t->bufferGeneration = b->generation;
    t->storageFmt = f;
    return true;
}

// ---------------------------------------------------------------------------
// sampling views
// ---------------------------------------------------------------------------
static MTLTextureSwizzle glSwizzleToMTL(GLenum s, MTLTextureSwizzleChannels base) {
    switch (s) {
    case GL_RED: return base.red;
    case GL_GREEN: return base.green;
    case GL_BLUE: return base.blue;
    case GL_ALPHA: return base.alpha;
    case GL_ZERO: return MTLTextureSwizzleZero;
    case GL_ONE: return MTLTextureSwizzleOne;
    default: return MTLTextureSwizzleRed;
    }
}

static bool isMipmapFilter(GLenum f) { return f != GL_NEAREST && f != GL_LINEAR; }

static bool mipmapComplete(const Texture* t, int base, int maxLevel, int* lastLevel) {
    const TexImageLevel& b = t->levels[0][base];
    int w = b.width, h = b.height, d = b.depth;
    int l = base;
    *lastLevel = base;
    int faces = t->faces();
    while (true) {
        for (int f = 0; f < faces; f++) {
            const TexImageLevel& L = t->levels[f][l];
            if (!L.defined() || L.fmt != b.fmt) return false;
            int ew = w, eh = h, ed = d;
            if (L.width != ew || L.height != eh || L.depth != ed) return false;
        }
        *lastLevel = l;
        if (l >= maxLevel) break;
        bool is1D = t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY;
        bool isArr = t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY;
        if (w == 1 && (is1D || t->target == GL_TEXTURE_1D_ARRAY || h == 1) && (t->target != GL_TEXTURE_3D || d == 1)) break;
        w = std::max(1, w / 2);
        if (!(t->target == GL_TEXTURE_1D_ARRAY)) h = is1D ? h : std::max(1, h / 2);
        if (t->target == GL_TEXTURE_3D) d = std::max(1, d / 2);
        (void)isArr;
        l++;
        if (l >= Texture::kMaxLevels) break;
    }
    return true;
}

// Texture completeness per GL 4.6 §8.17.
bool textureComplete(const Texture* t, const SamplerState& s, int* firstLevel, int* lastLevel) {
    if (t->target == GL_TEXTURE_BUFFER) {
        *firstLevel = *lastLevel = 0;
        return t->buffer && t->buffer->mtl;
    }
    if (t->viewParent) {
        *firstLevel = std::min(t->baseLevel, t->viewNumLevels - 1);
        *lastLevel = std::min(t->maxLevel, t->viewNumLevels - 1);
        return true;
    }
    int base = t->baseLevel;
    if (t->immutable) {
        base = std::min(base, t->immutableLevels - 1);
        int maxl = std::min(t->maxLevel, t->immutableLevels - 1);
        *firstLevel = base;
        *lastLevel = isMipmapFilter(s.minFilter) ? std::max(base, maxl) : base;
        return base <= maxl || !isMipmapFilter(s.minFilter);
    }
    if (base >= Texture::kMaxLevels || base > t->maxLevel) return false;
    const TexImageLevel& b = t->levels[0][base];
    if (!b.defined() || b.width == 0 || b.height == 0 || b.depth == 0) return false;
    if (t->target == GL_TEXTURE_CUBE_MAP) {
        for (int f = 1; f < 6; f++) {
            const TexImageLevel& L = t->levels[f][base];
            if (!L.defined() || L.fmt != b.fmt || L.width != b.width || L.height != b.height) return false;
        }
        if (b.width != b.height) return false;
    }
    *firstLevel = base;
    *lastLevel = base;
    if (!isMipmapFilter(s.minFilter) || t->samples > 0 || t->target == GL_TEXTURE_RECTANGLE) return true;
    int last;
    if (!mipmapComplete(t, base, t->maxLevel, &last)) return false;
    *lastLevel = last;
    return true;
}

static MTLTextureType viewTypeFor(GLenum samplerTarget, MTLTextureType storage) {
    switch (samplerTarget) {
    case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: return MTLTextureType2D;
    case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY: return MTLTextureType2DArray;
    case GL_TEXTURE_3D: return MTLTextureType3D;
    case GL_TEXTURE_CUBE_MAP: return MTLTextureTypeCube;
    case GL_TEXTURE_CUBE_MAP_ARRAY: return MTLTextureTypeCubeArray;
    case GL_TEXTURE_2D_MULTISAMPLE: return MTLTextureType2DMultisample;
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: return MTLTextureType2DMultisampleArray;
    case GL_TEXTURE_BUFFER: return MTLTextureTypeTextureBuffer;
    default: return storage;
    }
}

id<MTLTexture> textureSamplingView(Context* ctx, Texture* t, GLenum samplerTarget, bool stencilSampling, bool* complete) {
    *complete = false;
    const SamplerState& ss = t->sampler;
    int first, last;
    if (!textureComplete(t, ss, &first, &last)) return nil;
    if (!ensureTextureStorage(ctx, t)) return nil;
    if (t->target == GL_TEXTURE_BUFFER) {
        if (!ensureBufferTexture(ctx, t)) return nil;  // follows buffer reallocation / RGB32 refresh
        *complete = true;
        return t->mtl;
    }
    const FormatInfo* f = t->storageFmt ? t->storageFmt : formatInfo(GL_RGBA8);
    MTLTextureSwizzleChannels base = f->swizzle;
    MTLPixelFormat pf = t->mtl.pixelFormat;
    bool depthStencilFmt = f->kind == FmtKind::DepthStencil;
    bool sampleStencil = stencilSampling || (depthStencilFmt && t->depthStencilMode == GL_STENCIL_INDEX) || f->kind == FmtKind::Stencil;
    if (sampleStencil) {
        if (pf == MTLPixelFormatDepth32Float_Stencil8) pf = MTLPixelFormatX32_Stencil8;
        base = MTLTextureSwizzleChannelsMake(MTLTextureSwizzleRed, MTLTextureSwizzleZero, MTLTextureSwizzleZero, MTLTextureSwizzleOne);
    } else if (depthStencilFmt) {
        pf = MTLPixelFormatDepth32Float_Stencil8;  // depth aspect is sampled
    }
    if (f->srgb && t->srgbDecode == GL_SKIP_DECODE_EXT) {
        switch (pf) {
        case MTLPixelFormatRGBA8Unorm_sRGB: pf = MTLPixelFormatRGBA8Unorm; break;
        case MTLPixelFormatBGRA8Unorm_sRGB: pf = MTLPixelFormatBGRA8Unorm; break;
        case MTLPixelFormatBC1_RGBA_sRGB: pf = MTLPixelFormatBC1_RGBA; break;
        case MTLPixelFormatBC2_RGBA_sRGB: pf = MTLPixelFormatBC2_RGBA; break;
        case MTLPixelFormatBC3_RGBA_sRGB: pf = MTLPixelFormatBC3_RGBA; break;
        case MTLPixelFormatBC7_RGBAUnorm_sRGB: pf = MTLPixelFormatBC7_RGBAUnorm; break;
        default: break;
        }
    }
    MTLTextureSwizzleChannels sw;
    sw.red = glSwizzleToMTL(t->swizzle[0], base);
    sw.green = glSwizzleToMTL(t->swizzle[1], base);
    sw.blue = glSwizzleToMTL(t->swizzle[2], base);
    sw.alpha = glSwizzleToMTL(t->swizzle[3], base);
    MTLTextureType vt = viewTypeFor(samplerTarget, t->mtl.textureType);
    int levelStart = first, levelCount = last - first + 1;
    levelStart = std::min(levelStart, (int)t->mtl.mipmapLevelCount - 1);
    levelCount = std::max(1, std::min(levelCount, (int)t->mtl.mipmapLevelCount - levelStart));
    NSUInteger slices = t->mtl.arrayLength;
    if (t->mtl.textureType == MTLTextureTypeCube) slices = 6;
    if (t->mtl.textureType == MTLTextureTypeCubeArray) slices = t->mtl.arrayLength * 6;
    bool identity = pf == t->mtl.pixelFormat && vt == t->mtl.textureType && levelStart == 0 &&
                    levelCount == (int)t->mtl.mipmapLevelCount && sw.red == MTLTextureSwizzleRed &&
                    sw.green == MTLTextureSwizzleGreen && sw.blue == MTLTextureSwizzleBlue && sw.alpha == MTLTextureSwizzleAlpha;
    *complete = true;
    if (identity) return t->mtl;
    Texture::ViewKey key{(uint64_t)pf | (uint64_t)vt << 16 | (uint64_t)levelStart << 24 | (uint64_t)levelCount << 32,
                         (uint64_t)sw.red | (uint64_t)sw.green << 8 | (uint64_t)sw.blue << 16 | (uint64_t)sw.alpha << 24};
    auto it = t->views.find(key);
    if (it != t->views.end()) return it->second;
    id<MTLTexture> v = [t->mtl newTextureViewWithPixelFormat:pf textureType:vt levels:NSMakeRange(levelStart, levelCount)
                                                     slices:NSMakeRange(0, slices) swizzle:sw];
    if (!v) {
        MGL_ERR("texture view creation failed");
        *complete = false;
        return nil;
    }
    t->views[key] = v;
    return v;
}

// ---------------------------------------------------------------------------
// sampler states
// ---------------------------------------------------------------------------
static MTLSamplerAddressMode addressMode(GLenum wrap) {
    switch (wrap) {
    case GL_REPEAT: return MTLSamplerAddressModeRepeat;
    case GL_MIRRORED_REPEAT: return MTLSamplerAddressModeMirrorRepeat;
    case GL_CLAMP_TO_EDGE: return MTLSamplerAddressModeClampToEdge;
    case GL_CLAMP_TO_BORDER: return MTLSamplerAddressModeClampToBorderColor;
    case GL_MIRROR_CLAMP_TO_EDGE: return MTLSamplerAddressModeMirrorClampToEdge;
    case GL_CLAMP: return MTLSamplerAddressModeClampToEdge;  // compat GL_CLAMP approximated
    default: return MTLSamplerAddressModeRepeat;
    }
}

id<MTLSamplerState> samplerStateFor(const SamplerState& s, bool isInteger) {
    static std::mutex m;
    static std::unordered_map<std::string, id<MTLSamplerState>> cache;
    std::string key((const char*)&s, sizeof s);
    key.push_back(isInteger ? 1 : 0);
    std::lock_guard<std::mutex> g(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLSamplerDescriptor* d = [MTLSamplerDescriptor new];
    bool linearMin = s.minFilter == GL_LINEAR || s.minFilter == GL_LINEAR_MIPMAP_NEAREST || s.minFilter == GL_LINEAR_MIPMAP_LINEAR;
    d.minFilter = linearMin && !isInteger ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    d.magFilter = s.magFilter == GL_LINEAR && !isInteger ? MTLSamplerMinMagFilterLinear : MTLSamplerMinMagFilterNearest;
    switch (s.minFilter) {
    case GL_NEAREST_MIPMAP_NEAREST: case GL_LINEAR_MIPMAP_NEAREST: d.mipFilter = MTLSamplerMipFilterNearest; break;
    case GL_NEAREST_MIPMAP_LINEAR: case GL_LINEAR_MIPMAP_LINEAR: d.mipFilter = isInteger ? MTLSamplerMipFilterNearest : MTLSamplerMipFilterLinear; break;
    default: d.mipFilter = MTLSamplerMipFilterNotMipmapped; break;
    }
    d.sAddressMode = addressMode(s.wrapS);
    d.tAddressMode = addressMode(s.wrapT);
    d.rAddressMode = addressMode(s.wrapR);
    // Metal only has three border colors; pick the closest.
    const float* b = s.borderColor;
    if (b[3] < 0.5f) d.borderColor = MTLSamplerBorderColorTransparentBlack;
    else if (b[0] + b[1] + b[2] > 1.5f) d.borderColor = MTLSamplerBorderColorOpaqueWhite;
    else d.borderColor = MTLSamplerBorderColorOpaqueBlack;
    d.lodMinClamp = std::max(0.0f, s.minLod);
    d.lodMaxClamp = std::max(d.lodMinClamp, std::min(s.maxLod, 1000.0f));
    d.lodBias = std::clamp(s.lodBias, -16.0f, 16.0f);
    d.maxAnisotropy = std::clamp((NSUInteger)lroundf(s.maxAnisotropy), (NSUInteger)1, (NSUInteger)16);
    if (d.maxAnisotropy > 1 && d.mipFilter == MTLSamplerMipFilterNotMipmapped) d.maxAnisotropy = 1;
    if (s.compareMode == GL_COMPARE_REF_TO_TEXTURE) d.compareFunction = compareFunc(s.compareFunc);
    d.supportArgumentBuffers = YES;
    d.normalizedCoordinates = YES;
    id<MTLSamplerState> st = [Device::get()->mtl newSamplerStateWithDescriptor:d];
    cache[key] = st;
    return st;
}

// ---------------------------------------------------------------------------
// dummy textures for incomplete bindings (sample as 0,0,0,1)
// ---------------------------------------------------------------------------
id<MTLTexture> dummyTexture(Context* ctx, MTLTextureType type, FmtKind kind) {
    static std::mutex m;
    static std::unordered_map<uint32_t, id<MTLTexture>> cache;
    uint32_t key = (uint32_t)type << 8 | (uint32_t)kind;
    std::lock_guard<std::mutex> g(m);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    MTLPixelFormat pf = kind == FmtKind::UInt ? MTLPixelFormatRGBA8Uint : kind == FmtKind::SInt ? MTLPixelFormatRGBA8Sint
                        : kind == FmtKind::Depth ? MTLPixelFormatDepth32Float : MTLPixelFormatRGBA8Unorm;
    Device* d = ctx->dev;
    MTLTextureDescriptor* td = [MTLTextureDescriptor new];
    td.textureType = type == MTLTextureTypeTextureBuffer ? MTLTextureType2D : type;
    td.pixelFormat = pf;
    td.width = td.height = 1;
    td.depth = 1;
    td.arrayLength = 1;
    if (type == MTLTextureType2DMultisample || type == MTLTextureType2DMultisampleArray) td.sampleCount = 4;
    td.storageMode = MTLStorageModeShared;
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    if (kind == FmtKind::Depth) td.storageMode = MTLStorageModePrivate;
    id<MTLTexture> tex;
    if (type == MTLTextureTypeTextureBuffer) {
        id<MTLBuffer> buf = d->newBuffer(64);
        uint8_t* p = (uint8_t*)buf.contents;
        memset(p, 0, 64);
        p[3] = kind == FmtKind::Unorm ? 255 : 1;
        MTLTextureDescriptor* bd = [MTLTextureDescriptor textureBufferDescriptorWithPixelFormat:pf width:1
                                                                               resourceOptions:MTLResourceStorageModeShared
                                                                                         usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];
        tex = [buf newTextureWithDescriptor:bd offset:0 bytesPerRow:64];
    } else {
        tex = [d->mtl newTextureWithDescriptor:td];
        if (td.storageMode == MTLStorageModeShared && td.sampleCount <= 1) {
            uint8_t px[4] = {0, 0, 0, (uint8_t)(kind == FmtKind::Unorm ? 255 : 1)};
            int slices = type == MTLTextureTypeCube ? 6 : type == MTLTextureTypeCubeArray ? 6 : 1;
            for (int s = 0; s < slices; s++)
                [tex replaceRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:s withBytes:px bytesPerRow:4 bytesPerImage:4];
        }
    }
    d->makeResident(tex);
    cache[key] = tex;
    return tex;
}

// ---------------------------------------------------------------------------
// uploads
// ---------------------------------------------------------------------------
const void* resolveUnpackSource(Context* ctx, const void* pixels, size_t needed, bool* err) {
    *err = false;
    if (Buffer* pbo = ctx->pixelUnpackBuffer.get()) {
        size_t off = (size_t)pixels;
        if (pbo->mapped && !(pbo->mapAccess & GL_MAP_PERSISTENT_BIT)) {
            ctx->setError(GL_INVALID_OPERATION, "unpack buffer is mapped");
            *err = true;
            return nullptr;
        }
        if (off + needed > (size_t)pbo->size) {
            ctx->setError(GL_INVALID_OPERATION, "unpack buffer too small");
            *err = true;
            return nullptr;
        }
        if (pbo->lastGpuWrite && !ctx->dev->isComplete(pbo->lastGpuWrite)) ctx->waitForResource(pbo->lastGpuWrite);
        return pbo->cpu + off;
    }
    return pixels;
}

// Copies a converted image (already in storage layout in `up`) into the texture.
static void copyStagingToTexture(Context* ctx, Texture* t, const UploadAlloc& up, size_t rowBytes, size_t imageBytes, int level,
                                 int slice, int x, int y, int z, int w, int h, int d, MTLBlitOption opt = MTLBlitOptionNone) {
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    bool is3D = t->mtl.textureType == MTLTextureType3D;
    if (is3D) {
        [e copyFromBuffer:up.buffer sourceOffset:up.offset sourceBytesPerRow:rowBytes sourceBytesPerImage:imageBytes
                sourceSize:MTLSizeMake(w, h, d) toTexture:t->mtl destinationSlice:0 destinationLevel:level
         destinationOrigin:MTLOriginMake(x, y, z) options:opt];
    } else {
        for (int i = 0; i < d; i++)
            [e copyFromBuffer:up.buffer sourceOffset:up.offset + i * imageBytes sourceBytesPerRow:rowBytes sourceBytesPerImage:imageBytes
                    sourceSize:MTLSizeMake(w, h, 1) toTexture:t->mtl destinationSlice:slice + i destinationLevel:level
             destinationOrigin:MTLOriginMake(x, y, 0) options:opt];
    }
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    t->lastUse = ctx->serial();
    t->lastGpuWrite = t->lastUse;
}

// Metal slice for (face/layer). For cube maps face is the slice; for arrays z selects slices.
void uploadTexSubImage(Context* ctx, Texture* t, int level, int face, int x, int y, int z, int w, int h, int d, GLenum format,
                       GLenum type, const void* pixels) {
    if (w <= 0 || h <= 0 || d <= 0) return;
    if (!ensureTextureStorage(ctx, t)) return;
    const FormatInfo* f = t->storageFmt;
    // 1D arrays: GL "height" is the layer index.
    int sliceBase = face, mx = x, my = y, mz = z, mw = w, mh = h, md = d;
    if (t->target == GL_TEXTURE_1D_ARRAY) {
        sliceBase = y;
        my = 0;
        mh = 1;
        md = h;
    } else if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        sliceBase = z;
        mz = 0;
    }
    if (level >= t->storageLevels) return;
    ClientLayout cl = clientLayout(format, type, w, h, ctx->unpack);
    size_t need = clientImageSize(format, type, w, h, d, ctx->unpack);
    bool err;
    const uint8_t* src = (const uint8_t*)resolveUnpackSource(ctx, pixels, need, &err);
    if (err || !src) return;
    src += cl.skipBytes;
    size_t rowBytes = storageRowBytes(f, w);
    size_t imageBytes = rowBytes * h;
    if (f->comp == Comp::D32F_S8) {
        // Depth and stencil go through separate staging planes.
        UploadAlloc dp = ctx->upload((size_t)w * 4 * h * d, 16);
        UploadAlloc sp = ctx->upload((size_t)w * h * d, 16);
        std::vector<uint8_t> tmp((size_t)w * 8 * h);
        for (int i = 0; i < d; i++) {
            convertToStorage(f, format, type, src + i * cl.imageStride, cl.rowStride, cl.imageStride, tmp.data(), (size_t)w * 8, 0, w, h, 1,
                             ctx->unpack.swapBytes);
            for (int p = 0; p < w * h; p++) {
                memcpy(dp.cpu + (size_t)i * w * h * 4 + p * 4, tmp.data() + p * 8, 4);
                sp.cpu[(size_t)i * w * h + p] = tmp[p * 8 + 4];
            }
        }
        bool hasDepth = format != GL_STENCIL_INDEX, hasStencil = format == GL_DEPTH_STENCIL || format == GL_STENCIL_INDEX;
        if (hasDepth)
            copyStagingToTexture(ctx, t, dp, (size_t)w * 4, (size_t)w * 4 * h, level, sliceBase, mx, my, mz, mw, mh, md, MTLBlitOptionDepthFromDepthStencil);
        if (hasStencil)
            copyStagingToTexture(ctx, t, sp, (size_t)w, (size_t)w * h, level, sliceBase, mx, my, mz, mw, mh, md, MTLBlitOptionStencilFromDepthStencil);
        return;
    }
    if (f->comp == Comp::Compressed) {
        // Compressed image from uncompressed data: decode to RGBA float, then block-encode.
        const FormatInfo* ff = formatInfo(GL_RGBA32F);
        size_t cRow = storageRowBytes(f, w), cImg = storageImageBytes(f, w, h);
        std::vector<float> tmp((size_t)w * h * 4);
        UploadAlloc up = ctx->upload(cImg * d, 16);
        for (int i = 0; i < d; i++) {
            convertToStorage(ff, format, type, src + i * cl.imageStride, cl.rowStride, cl.imageStride, tmp.data(), (size_t)w * 16,
                             (size_t)w * h * 16, w, h, 1, ctx->unpack.swapBytes);
            compressImage(f, tmp.data(), w, h, up.cpu + i * cImg);
        }
        // Metal copy sizes: block-aligned, clamped to the level size.
        int lw = std::max(1, t->storageW >> level), lh = std::max(1, t->storageH >> level);
        copyStagingToTexture(ctx, t, up, cRow, cImg, level, sliceBase, mx, my, mz, std::min(w, lw - x), std::min(h, lh - y), md);
        return;
    }
    UploadAlloc up = ctx->upload(imageBytes * d, 16);
    convertToStorage(f, format, type, src, cl.rowStride, cl.imageStride, up.cpu, rowBytes, imageBytes, w, h, d, ctx->unpack.swapBytes);
    if (t->target == GL_TEXTURE_1D_ARRAY)
        copyStagingToTexture(ctx, t, up, rowBytes, rowBytes, level, sliceBase, mx, 0, 0, mw, 1, md);
    else
        copyStagingToTexture(ctx, t, up, rowBytes, imageBytes, level, sliceBase, mx, my, mz, mw, mh, md);
}

void uploadCompressedSubImage(Context* ctx, Texture* t, int level, int face, int x, int y, int z, int w, int h, int d, GLsizei imageSize,
                              const void* data) {
    if (!ensureTextureStorage(ctx, t)) return;
    const FormatInfo* f = t->storageFmt;
    size_t rowBytes = storageRowBytes(f, w);
    size_t imgBytes = storageImageBytes(f, w, h);
    if ((size_t)imageSize < imgBytes * d) {
        ctx->setError(GL_INVALID_VALUE, "compressed image size mismatch");
        return;
    }
    int dims = t->target == GL_TEXTURE_1D ? 1
             : (t->target == GL_TEXTURE_3D || t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) ? 3
                                                                                                                           : 2;
    CompressedLayout cl = compressedLayout(f, w, h, dims, ctx->unpack);
    bool err;
    const uint8_t* src = (const uint8_t*)resolveUnpackSource(ctx, data, std::max((size_t)imageSize, cl.extent(d)), &err);
    if (err || !src) return;
    UploadAlloc up = ctx->upload(imgBytes * d, 16);
    if (cl.skipBytes == 0 && cl.rowStride == rowBytes && cl.imageStride == imgBytes) {
        memcpy(up.cpu, src, imgBytes * d);
    } else {
        for (int k = 0; k < d; k++)
            for (int r = 0; r < cl.rows; r++)
                memcpy(up.cpu + k * imgBytes + r * rowBytes, src + cl.skipBytes + k * cl.imageStride + r * cl.rowStride, rowBytes);
    }
    int slice = face, mz = z;
    if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) {
        slice = z;
        mz = 0;
    }
    // Metal requires block-aligned copy sizes clamped to the level size.
    int lw = std::max(1, t->storageW >> level), lh = std::max(1, t->storageH >> level);
    int cw = std::min(w, lw - x), ch = std::min(h, lh - y);
    copyStagingToTexture(ctx, t, up, rowBytes, imgBytes, level, slice, x, y, mz, cw, ch, d);
}

// Reads back a texture region (storage layout) into CPU memory. Synchronous.
bool readTextureRegion(Context* ctx, id<MTLTexture> tex, int level, int slice, int x, int y, int w, int h, std::vector<uint8_t>& out,
                       size_t& rowBytes, const FormatInfo* f, MTLBlitOption opt) {
    size_t bpp = f->bytesPerPixel;
    if (opt == MTLBlitOptionDepthFromDepthStencil) bpp = 4;
    if (opt == MTLBlitOptionStencilFromDepthStencil) bpp = 1;
    rowBytes = (size_t)w * bpp;
    if (f->comp == Comp::Compressed) rowBytes = storageRowBytes(f, w);
    size_t imgBytes = f->comp == Comp::Compressed ? storageImageBytes(f, w, h) : rowBytes * h;
    id<MTLBuffer> buf = ctx->dev->newBuffer(std::max<size_t>(imgBytes, 16));
    id<MTLTexture> src = tex;
    if (tex.sampleCount > 1) {
        MGL_WARN("readback of multisampled texture not resolved");
    }
    // 3D textures: the GL layer is a depth plane, not a slice
    int z = 0;
    if (src.textureType == MTLTextureType3D) {
        z = slice;
        slice = 0;
    }
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromTexture:src sourceSlice:slice sourceLevel:level sourceOrigin:MTLOriginMake(x, y, z) sourceSize:MTLSizeMake(w, h, 1)
              toBuffer:buf destinationOffset:0 destinationBytesPerRow:rowBytes destinationBytesPerImage:imgBytes options:opt];
    uint64_t s = ctx->serial();
    ctx->finish();
    (void)s;
    out.resize(imgBytes);
    memcpy(out.data(), buf.contents, imgBytes);
    ctx->dev->releaseAfter(buf, 0);
    return true;
}

void markTextureWrite(Context* ctx, Texture* t) {
    t->lastUse = ctx->serial();
    t->lastGpuWrite = t->lastUse;
}

id<MTLTexture> attachmentWriteView(Context* ctx, const Attachment& a, id<MTLTexture> t) {
    if (ctx->framebufferSRGB || !t || !(t.usage & MTLTextureUsagePixelFormatView)) return t;
    MTLPixelFormat lin;
    switch (t.pixelFormat) {
    case MTLPixelFormatRGBA8Unorm_sRGB: lin = MTLPixelFormatRGBA8Unorm; break;
    case MTLPixelFormatBGRA8Unorm_sRGB: lin = MTLPixelFormatBGRA8Unorm; break;
    default: return t;
    }
    if (a.type == GL_RENDERBUFFER && a.rb) {
        if (!a.rb->linearView) a.rb->linearView = [t newTextureViewWithPixelFormat:lin];
        return a.rb->linearView;
    }
    if (a.type == GL_TEXTURE && a.tex) {
        Texture* tx = a.tex.get();
        const Texture::ViewKey key{~0ull, (uint64_t)lin};  // not produced by sampling views
        auto it = tx->views.find(key);
        if (it != tx->views.end()) return it->second;
        id<MTLTexture> v = [t newTextureViewWithPixelFormat:lin];
        tx->views[key] = v;
        return v;
    }
    return t;
}

id<MTLTexture> attachmentTexture(Context* ctx, const Attachment& a, const FormatInfo** fmt, int* w, int* h) {
    if (a.type == GL_RENDERBUFFER && a.rb) {
        Renderbuffer* rb = a.rb.get();
        *fmt = rb->fmt;
        *w = rb->width;
        *h = rb->height;
        rb->lastUse = ctx->serial();
        return rb->mtl;
    }
    if (a.type == GL_TEXTURE && a.tex) {
        Texture* t = a.tex.get();
        if (!ensureTextureStorage(ctx, t)) return nil;
        *fmt = t->storageFmt;
        *w = std::max(1, (t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY ? t->storageW : t->storageW) >> a.level);
        *h = std::max(1, t->storageH >> a.level);
        if (t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY) *h = 1;
        markTextureWrite(ctx, t);
        return t->mtl;
    }
    return nil;
}

}  // namespace mgl
