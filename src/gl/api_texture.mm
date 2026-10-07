// Texture and sampler entry points.
#import "gl/texture.h"
#include "mtl/compress.h"
#include <cmath>
#include "gl/internal_shaders.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

static bool isCubeFace(GLenum t) { return t >= GL_TEXTURE_CUBE_MAP_POSITIVE_X && t <= GL_TEXTURE_CUBE_MAP_NEGATIVE_Z; }

static GLenum bindTargetFor(GLenum target) { return isCubeFace(target) ? GL_TEXTURE_CUBE_MAP : target; }

static int faceIndex(GLenum target) { return isCubeFace(target) ? (int)(target - GL_TEXTURE_CUBE_MAP_POSITIVE_X) : 0; }

static bool isProxy(GLenum t) {
    return t == GL_PROXY_TEXTURE_1D || t == GL_PROXY_TEXTURE_2D || t == GL_PROXY_TEXTURE_3D || t == GL_PROXY_TEXTURE_1D_ARRAY ||
           t == GL_PROXY_TEXTURE_2D_ARRAY || t == GL_PROXY_TEXTURE_RECTANGLE || t == GL_PROXY_TEXTURE_CUBE_MAP ||
           t == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY || t == GL_PROXY_TEXTURE_2D_MULTISAMPLE || t == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY;
}

static Texture* texForTarget(Context* ctx, GLenum target) {
    Texture* t = ctx->boundTexture(bindTargetFor(target));
    if (!t) ctx->setError(GL_INVALID_ENUM, "invalid texture target 0x%x", target);
    return t;
}

// glTexParameter* / glGetTexParameter* accept texture binding targets only
// (no cube map faces, proxies or buffer textures).
static Texture* paramTexForTarget(Context* ctx, GLenum target) {
    switch (target) {
    case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY:
    case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE:
    case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        return texForTarget(ctx, target);
    default:
        ctx->setError(GL_INVALID_ENUM, "invalid texture parameter target 0x%x", target);
        return nullptr;
    }
}

static Texture* namedTex(Context* ctx, GLuint name) {
    Texture* t = ctx->share->textures.get(name);
    if (!t) ctx->setError(GL_INVALID_OPERATION, "texture %u does not exist", name);
    return t;
}

// --- names & binding ---
API void APIENTRY glGenTextures(GLsizei n, GLuint* textures) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) textures[i] = ctx->share->textures.genName();
}

static Texture* newTexture(GLenum target) {
    Texture* t = new Texture();
    t->target = target;
    if (target == GL_TEXTURE_RECTANGLE) {
        t->sampler.minFilter = GL_LINEAR;
        t->sampler.wrapS = t->sampler.wrapT = t->sampler.wrapR = GL_CLAMP_TO_EDGE;
    }
    return t;
}

API void APIENTRY glCreateTextures(GLenum target, GLsizei n, GLuint* textures) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    if (texTargetIndex(target) < 0) return ctx->setError(GL_INVALID_ENUM);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) {
        textures[i] = ctx->share->textures.genName();
        ctx->share->textures.create(textures[i], newTexture(target));
    }
}

API void APIENTRY glDeleteTextures(GLsizei n, const GLuint* textures) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    @autoreleasepool {  // releasing Metal textures autoreleases (mtl/device.h)
        for (int i = 0; i < n; i++) {
            if (!textures[i]) continue;
            if (Texture* t = ctx->share->textures.get(textures[i])) ctx->unbindDeletedTexture(t);
            ctx->share->textures.remove(textures[i]);
        }
    }
}

API GLboolean APIENTRY glIsTexture(GLuint texture) {
    MGL_CTX_RET(GL_FALSE);
    return texture && ctx->share->textures.get(texture) != nullptr;
}

static void bindTextureUnit(Context* ctx, int unit, GLenum target, GLuint name) {
    int idx = texTargetIndex(target);
    if (idx < 0) return ctx->setError(GL_INVALID_ENUM);
    if (!name) {
        ctx->textures[unit][idx] = ctx->defaultTextures[idx];
        return;
    }
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    Texture* t = ctx->share->textures.get(name);
    if (!t) {
        if (!ctx->share->textures.isName(name) && ctx->coreProfile) return ctx->setError(GL_INVALID_OPERATION);
        t = ctx->share->textures.create(name, newTexture(target));
    }
    if (t->target != target) return ctx->setError(GL_INVALID_OPERATION, "texture %u target mismatch", name);
    ctx->textures[unit][idx] = t;
}

API void APIENTRY glBindTexture(GLenum target, GLuint texture) {
    MGL_CTX();
    bindTextureUnit(ctx, ctx->activeTexture, target, texture);
}

API void APIENTRY glActiveTexture(GLenum texture) {
    MGL_CTX();
    int u = (int)texture - GL_TEXTURE0;
    if (u < 0 || u >= kMaxTextureUnits) return ctx->setError(GL_INVALID_ENUM);
    ctx->activeTexture = u;
}

API void APIENTRY glBindTextureUnit(GLuint unit, GLuint texture) {
    MGL_CTX();
    if (unit >= kMaxTextureUnits) return ctx->setError(GL_INVALID_VALUE);
    if (!texture) {
        for (int i = 0; i < kNumTexTargets; i++) ctx->textures[unit][i] = ctx->defaultTextures[i];
        return;
    }
    Texture* t = ctx->share->textures.get(texture);
    if (!t) return ctx->setError(GL_INVALID_OPERATION);
    ctx->textures[unit][texTargetIndex(t->target)] = t;
}

API void APIENTRY glBindTextures(GLuint first, GLsizei count, const GLuint* textures) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if ((uint64_t)first + (uint64_t)count > (uint64_t)kMaxTextureUnits) return ctx->setError(GL_INVALID_OPERATION);
    for (int i = 0; i < count; i++) {
        GLuint name = textures ? textures[i] : 0;
        if (!name) {
            for (int k = 0; k < kNumTexTargets; k++) ctx->textures[first + i][k] = ctx->defaultTextures[k];
            continue;
        }
        Texture* t = ctx->share->textures.get(name);
        if (!t || !t->target) {
            ctx->setError(GL_INVALID_OPERATION);
            continue;
        }
        ctx->textures[first + i][texTargetIndex(t->target)] = t;
    }
}

// --- image specification ---
static bool validTexTarget(GLenum target, int dims) {
    switch (dims) {
    case 1: return target == GL_TEXTURE_1D || target == GL_PROXY_TEXTURE_1D;
    case 2:
        return target == GL_TEXTURE_2D || target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_RECTANGLE || isCubeFace(target) ||
               target == GL_PROXY_TEXTURE_2D || target == GL_PROXY_TEXTURE_1D_ARRAY || target == GL_PROXY_TEXTURE_RECTANGLE ||
               target == GL_PROXY_TEXTURE_CUBE_MAP;
    case 3:
        return target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY ||
               target == GL_PROXY_TEXTURE_3D || target == GL_PROXY_TEXTURE_2D_ARRAY || target == GL_PROXY_TEXTURE_CUBE_MAP_ARRAY;
    }
    return false;
}

static void defineLevel(Texture* t, int face, int level, int w, int h, int d, GLenum internalFormat, const FormatInfo* f) {
    TexImageLevel& L = t->levels[face][level];
    bool changed = L.fmt != f || L.width != w || L.height != h || L.depth != d;
    L.width = w;
    L.height = h;
    L.depth = d;
    L.internalFormat = internalFormat;
    L.fmt = f;
    if (changed) {
        // Storage may need reallocation if this level no longer fits.
        if (!t->mtl) t->storageDirty = true;
        else {
            int ref = textureStorageLevelFor(t);
            int sw = std::max(1, t->storageW >> level), sh = std::max(1, t->storageH >> level);
            int sd = t->target == GL_TEXTURE_3D ? std::max(1, t->storageD >> level) : d;
            bool fits = f == t->storageFmt && level < t->storageLevels && w == sw &&
                        (t->target == GL_TEXTURE_1D || t->target == GL_TEXTURE_1D_ARRAY || h == sh) && (t->target != GL_TEXTURE_3D || d == sd);
            if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY)
                fits = fits && d == (t->target == GL_TEXTURE_CUBE_MAP_ARRAY ? t->storageLayers * 6 : t->storageLayers);
            if (t->target == GL_TEXTURE_1D_ARRAY) fits = fits && h == t->storageLayers;
            if (!fits || level == ref) t->storageDirty = true;
        }
    }
}

static void texImage(Context* ctx, GLenum target, GLint level, GLint internalformat, GLsizei w, GLsizei h, GLsizei d, GLint border,
                     GLenum format, GLenum type, const void* pixels, int dims) {
    if (!validTexTarget(target, dims)) return ctx->setError(GL_INVALID_ENUM);
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    if (w < 0 || h < 0 || d < 0 || border != 0) return ctx->setError(GL_INVALID_VALUE);
    if (w > 16384 || h > 16384 || d > 2048) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = pixelFormatTypeError(format, type)) return ctx->setError(e, "bad format/type 0x%x/0x%x", format, type);
    const FormatInfo* f = resolveInternalFormat(internalformat, format, type);
    if (!f) return ctx->setError(GL_INVALID_VALUE, "unsupported internal format 0x%x", internalformat);
    if (GLenum e = texImageFormatError(f, format, type, target))
        return ctx->setError(e, "format 0x%x/type 0x%x incompatible with internal format 0x%x", format, type, internalformat);
    // Compressed formats have no 1D images (4x4 blocks), and only some support 3D.
    bool noCompressed = (dims == 1 && f->comp == Comp::Compressed) ||
                        ((target == GL_TEXTURE_3D || target == GL_PROXY_TEXTURE_3D) && !compressedSupports3D(f));
    if (noCompressed) {
        // A generic compressed format may resolve to an uncompressed one; do so where the
        // chosen compressed format can't be used.
        GLenum generic = 0;
        switch (internalformat) {
        case GL_COMPRESSED_RED: generic = GL_RED; break;
        case GL_COMPRESSED_RG: generic = GL_RG; break;
        case GL_COMPRESSED_RGB: generic = GL_RGB; break;
        case GL_COMPRESSED_RGBA: generic = GL_RGBA; break;
        case GL_COMPRESSED_SRGB: generic = GL_SRGB; break;
        case GL_COMPRESSED_SRGB_ALPHA: generic = GL_SRGB_ALPHA; break;
        default: break;
        }
        if (generic) f = resolveInternalFormat(generic, format, type);
    }
    if ((dims == 1 && f->comp == Comp::Compressed) || ((target == GL_TEXTURE_3D || target == GL_PROXY_TEXTURE_3D) && !compressedSupports3D(f)))
        return ctx->setError(GL_INVALID_OPERATION, "compressed format 0x%x can't be used for this target", f->internalFormat);
    if (f->comp == Comp::Compressed && (pixels || ctx->pixelUnpackBuffer) && !cpuCompressible(f)) {
        // No CPU encoder for this format (ASTC): store the image uncompressed.
        f = formatInfo(f->srgb ? GL_SRGB8_ALPHA8 : GL_RGBA8);
    }
    if (isProxy(target)) {
        // Proxy: record the level if it would succeed, otherwise clear it (GL 4.6 §8.22).
        Ref<Texture>& pt = ctx->proxyTextures[target];
        if (!pt) {
            pt = newTexture(target);
            pt->target = target;
        }
        int maxDim = (target == GL_PROXY_TEXTURE_3D || target == GL_PROXY_TEXTURE_2D_ARRAY) ? 2048 : 16384;
        bool fits = w <= 16384 && (dims < 2 || h <= 16384) && (dims < 3 || d <= 2048) && (w >> level) <= maxDim;
        int face = 0;
        if (fits) {
            pt->levels[face][level] = TexImageLevel{w, dims >= 2 ? h : 1, dims >= 3 ? d : 1, (GLenum)internalformat, f};
        } else {
            for (int l = 0; l < Texture::kMaxLevels; l++) pt->levels[face][l] = TexImageLevel();
        }
        return;
    }
    if (isCubeFace(target) && w != h) return ctx->setError(GL_INVALID_VALUE);
    Texture* t = texForTarget(ctx, target);
    if (!t) return;
    if (t->immutable) return ctx->setError(GL_INVALID_OPERATION, "texture is immutable");
    if (target == GL_TEXTURE_RECTANGLE && level != 0) return ctx->setError(GL_INVALID_VALUE);
    int face = faceIndex(target);
    if (dims == 1) {
        h = 1;
        d = 1;
    } else if (dims == 2) {
        d = 1;
    }
    if (target == GL_TEXTURE_CUBE_MAP_ARRAY && (w != h || d % 6)) return ctx->setError(GL_INVALID_VALUE);
    // A generic compressed internal format reports the specific format chosen.
    GLenum reported = f->comp == Comp::Compressed ? f->internalFormat : (GLenum)internalformat;
    defineLevel(t, face, level, w, h, d, reported, f);
    if (pixels || ctx->pixelUnpackBuffer) {
        if (w && h && d) uploadTexSubImage(ctx, t, level, face, 0, 0, 0, w, h, d, format, type, pixels);
    }
    if (t->generateMipmapHint && level == t->baseLevel) {
        extern void generateMipmaps(Context*, Texture*);
        generateMipmaps(ctx, t);
    }
}

API void APIENTRY glTexImage1D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLint border, GLenum format, GLenum type,
                               const void* pixels) {
    MGL_CTX();
    texImage(ctx, target, level, internalformat, width, 1, 1, border, format, type, pixels, 1);
}

API void APIENTRY glTexImage2D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLint border, GLenum format,
                               GLenum type, const void* pixels) {
    MGL_CTX();
    texImage(ctx, target, level, internalformat, width, height, 1, border, format, type, pixels, 2);
}

API void APIENTRY glTexImage3D(GLenum target, GLint level, GLint internalformat, GLsizei width, GLsizei height, GLsizei depth, GLint border,
                               GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    texImage(ctx, target, level, internalformat, width, height, depth, border, format, type, pixels, 3);
}

// Bytes of one datum of `type` (a whole pixel for packed types): PBO offsets must be multiples.
static int typeDatumBytes(GLenum type) {
    switch (type) {
    case GL_UNSIGNED_BYTE: case GL_BYTE: case GL_UNSIGNED_BYTE_3_3_2: case GL_UNSIGNED_BYTE_2_3_3_REV: return 1;
    case GL_UNSIGNED_SHORT: case GL_SHORT: case GL_HALF_FLOAT: case GL_UNSIGNED_SHORT_5_6_5: case GL_UNSIGNED_SHORT_5_6_5_REV:
    case GL_UNSIGNED_SHORT_4_4_4_4: case GL_UNSIGNED_SHORT_4_4_4_4_REV: case GL_UNSIGNED_SHORT_5_5_5_1: case GL_UNSIGNED_SHORT_1_5_5_5_REV:
        return 2;
    case GL_FLOAT_32_UNSIGNED_INT_24_8_REV: return 8;
    default: return 4;
    }
}

static bool pboOffsetMisaligned(Buffer* pbo, const void* pixels, GLenum type) {
    return pbo && ((uintptr_t)pixels % typeDatumBytes(type)) != 0;
}

static void texSubImage(Context* ctx, Texture* t, GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                        GLenum format, GLenum type, const void* pixels) {
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    if (w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = pixelFormatTypeError(format, type)) return ctx->setError(e);
    if (pboOffsetMisaligned(ctx->pixelUnpackBuffer.get(), pixels, type)) return ctx->setError(GL_INVALID_OPERATION, "unaligned PBO offset");
    int face = faceIndex(target);
    if (t->target == GL_TEXTURE_CUBE_MAP && !isCubeFace(target)) {
        // DSA path on cube maps: z selects faces
        face = z;
        if (z < 0 || z + d > 6) return ctx->setError(GL_INVALID_VALUE);
        for (int i = 0; i < d; i++) {
            const TexImageLevel& L = t->levels[z + i][level];
            if (!L.defined()) return ctx->setError(GL_INVALID_OPERATION);
        }
        size_t per = clientImageSize(format, type, w, h, 1, ctx->unpack);
        ClientLayout cl = clientLayout(format, type, w, h, ctx->unpack);
        (void)per;
        for (int i = 0; i < d; i++)
            uploadTexSubImage(ctx, t, level, z + i, x, y, 0, w, h, 1, format, type, (const uint8_t*)pixels + i * cl.imageStride);
        return;
    }
    const TexImageLevel& L = t->levels[face][level];
    if (!L.defined()) return ctx->setError(GL_INVALID_OPERATION, "level %d not defined", level);
    if (x < 0 || y < 0 || z < 0 || x + w > L.width || y + h > L.height || z + d > L.depth) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = texImageFormatError(L.fmt, format, type, t->target)) return ctx->setError(e);
    if (L.fmt->comp == Comp::Compressed) {
        // Uncompressed data into a compressed image: block-aligned regions only (§8.7), encoded on the CPU.
        const FormatInfo* f = L.fmt;
        if (x % f->blockW || y % f->blockH || (w % f->blockW && x + w != L.width) || (h % f->blockH && y + h != L.height))
            return ctx->setError(GL_INVALID_OPERATION, "unaligned region in compressed texture");
        if (!cpuCompressible(f)) return ctx->setError(GL_INVALID_OPERATION, "no encoder for compressed format 0x%x", f->internalFormat);
    }
    if (!w || !h || !d) return;
    if (!pixels && !ctx->pixelUnpackBuffer) return;
    uploadTexSubImage(ctx, t, level, face, x, y, z, w, h, d, format, type, pixels);
    if (t->generateMipmapHint && level == t->baseLevel) {
        extern void generateMipmaps(Context*, Texture*);
        generateMipmaps(ctx, t);
    }
}

API void APIENTRY glTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    if (target != GL_TEXTURE_1D) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texSubImage(ctx, t, target, level, xoffset, 0, 0, width, 1, 1, format, type, pixels);
}

API void APIENTRY glTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height, GLenum format,
                                  GLenum type, const void* pixels) {
    MGL_CTX();
    if (!validTexTarget(target, 2) || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texSubImage(ctx, t, target, level, xoffset, yoffset, 0, width, height, 1, format, type, pixels);
}

API void APIENTRY glTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height,
                                  GLsizei depth, GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    if (!validTexTarget(target, 3) || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texSubImage(ctx, t, target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

API void APIENTRY glTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texSubImage(ctx, t, t->target, level, xoffset, 0, 0, width, 1, 1, format, type, pixels);
}

API void APIENTRY glTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                                      GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texSubImage(ctx, t, t->target, level, xoffset, yoffset, 0, width, height, 1, format, type, pixels);
}

API void APIENTRY glTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                      GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texSubImage(ctx, t, t->target, level, xoffset, yoffset, zoffset, width, height, depth, format, type, pixels);
}

// --- compressed ---
static void compressedTexImage(Context* ctx, GLenum target, GLint level, GLenum internalformat, GLsizei w, GLsizei h, GLsizei d,
                               GLint border, GLsizei imageSize, const void* data, int dims) {
    if (!validTexTarget(target, dims)) return ctx->setError(GL_INVALID_ENUM);
    if (level < 0 || level >= Texture::kMaxLevels || border != 0 || w < 0 || h < 0 || d < 0 || imageSize < 0)
        return ctx->setError(GL_INVALID_VALUE);
    const FormatInfo* f = formatInfo(internalformat);
    if (!f || f->comp != Comp::Compressed) return ctx->setError(GL_INVALID_ENUM);
    if ((target == GL_TEXTURE_3D && !compressedSupports3D(f)) || target == GL_TEXTURE_RECTANGLE || dims == 1)
        return ctx->setError(GL_INVALID_OPERATION);
    if ((size_t)imageSize != storageImageBytes(f, w, h) * (dims == 3 ? d : 1)) return ctx->setError(GL_INVALID_VALUE);
    if (isProxy(target)) return;
    Texture* t = texForTarget(ctx, target);
    if (!t) return;
    if (t->immutable) return ctx->setError(GL_INVALID_OPERATION);
    int face = faceIndex(target);
    if (dims == 2) d = 1;
    defineLevel(t, face, level, w, h, d, internalformat, f);
    if ((data || ctx->pixelUnpackBuffer) && w && h && d) uploadCompressedSubImage(ctx, t, level, face, 0, 0, 0, w, h, d, imageSize, data);
}

API void APIENTRY glCompressedTexImage1D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLint border, GLsizei imageSize,
                                         const void* data) {
    MGL_CTX();
    compressedTexImage(ctx, target, level, internalformat, width, 1, 1, border, imageSize, data, 1);
}

API void APIENTRY glCompressedTexImage2D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLint border,
                                         GLsizei imageSize, const void* data) {
    MGL_CTX();
    compressedTexImage(ctx, target, level, internalformat, width, height, 1, border, imageSize, data, 2);
}

API void APIENTRY glCompressedTexImage3D(GLenum target, GLint level, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth,
                                         GLint border, GLsizei imageSize, const void* data) {
    MGL_CTX();
    compressedTexImage(ctx, target, level, internalformat, width, height, depth, border, imageSize, data, 3);
}

static void compressedSubImage(Context* ctx, Texture* t, GLenum target, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h,
                               GLsizei d, GLenum format, GLsizei imageSize, const void* data, int dims = 0) {
    switch (format) {
    case GL_COMPRESSED_RED: case GL_COMPRESSED_RG: case GL_COMPRESSED_RGB: case GL_COMPRESSED_RGBA: case GL_COMPRESSED_SRGB:
    case GL_COMPRESSED_SRGB_ALPHA:
        return ctx->setError(GL_INVALID_ENUM, "generic compressed format");
    default: break;
    }
    if (dims) {  // DSA: the texture's target must suit the entry point
        GLenum tt = t->target;
        bool ok = dims == 1   ? tt == GL_TEXTURE_1D
                  : dims == 2 ? (tt == GL_TEXTURE_2D || tt == GL_TEXTURE_1D_ARRAY)
                              : (tt == GL_TEXTURE_2D_ARRAY || tt == GL_TEXTURE_CUBE_MAP_ARRAY || tt == GL_TEXTURE_3D || tt == GL_TEXTURE_CUBE_MAP);
        if (!ok) return ctx->setError(GL_INVALID_OPERATION, "compressed sub-image: wrong texture target 0x%x", tt);
    }
    if (Buffer* pbo = ctx->pixelUnpackBuffer.get())
        if (pbo->mapped && !(pbo->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION, "unpack buffer is mapped");
    int face = faceIndex(target);
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    const TexImageLevel& L = t->levels[face][level];
    if (!L.defined()) return ctx->setError(GL_INVALID_OPERATION);
    if (L.fmt->internalFormat != format && L.internalFormat != format) return ctx->setError(GL_INVALID_OPERATION);
    const FormatInfo* f = L.fmt;
    if (f->comp != Comp::Compressed) return ctx->setError(GL_INVALID_OPERATION);
    if (x % f->blockW || y % f->blockH) return ctx->setError(GL_INVALID_OPERATION);
    if ((w % f->blockW && x + w != L.width) || (h % f->blockH && y + h != L.height)) return ctx->setError(GL_INVALID_OPERATION);
    if (x < 0 || y < 0 || x + w > L.width || y + h > L.height || z < 0 || z + d > L.depth) return ctx->setError(GL_INVALID_VALUE);
    if ((size_t)imageSize != storageImageBytes(f, w, h) * d) return ctx->setError(GL_INVALID_VALUE);
    uploadCompressedSubImage(ctx, t, level, face, x, y, z, w, h, d, imageSize, data);
}

API void APIENTRY glCompressedTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                                            GLenum format, GLsizei imageSize, const void* data) {
    MGL_CTX();
    Texture* t = texForTarget(ctx, target);
    if (t) compressedSubImage(ctx, t, target, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data);
}

API void APIENTRY glCompressedTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                            GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data) {
    MGL_CTX();
    Texture* t = texForTarget(ctx, target);
    if (t) compressedSubImage(ctx, t, target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data);
}

API void APIENTRY glCompressedTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize,
                                            const void* data) {
    MGL_CTX();
    ctx->setError(GL_INVALID_OPERATION);
}

API void APIENTRY glCompressedTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLsizei width, GLenum format, GLsizei imageSize,
                                                const void* data) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) compressedSubImage(ctx, t, t->target, level, xoffset, 0, 0, width, 1, 1, format, imageSize, data, 1);
}

API void APIENTRY glCompressedTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLsizei width, GLsizei height,
                                                GLenum format, GLsizei imageSize, const void* data) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) compressedSubImage(ctx, t, t->target, level, xoffset, yoffset, 0, width, height, 1, format, imageSize, data, 2);
}

API void APIENTRY glCompressedTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                                GLsizei height, GLsizei depth, GLenum format, GLsizei imageSize, const void* data) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) compressedSubImage(ctx, t, t->target, level, xoffset, yoffset, zoffset, width, height, depth, format, imageSize, data, 3);
}

// --- immutable storage ---
static void texStorage(Context* ctx, Texture* t, GLsizei levels, GLenum internalformat, GLsizei w, GLsizei h, GLsizei d, int dims) {
    GLenum tt = t->target;
    bool targetOk = dims == 1   ? tt == GL_TEXTURE_1D
                    : dims == 2 ? (tt == GL_TEXTURE_2D || tt == GL_TEXTURE_1D_ARRAY || tt == GL_TEXTURE_RECTANGLE || tt == GL_TEXTURE_CUBE_MAP)
                                : (tt == GL_TEXTURE_3D || tt == GL_TEXTURE_2D_ARRAY || tt == GL_TEXTURE_CUBE_MAP_ARRAY);
    if (!targetOk) return ctx->setError(GL_INVALID_OPERATION, "texture storage: wrong texture target 0x%x", tt);  // DSA only
    if (levels < 1 || w < 1 || h < 1 || d < 1) return ctx->setError(GL_INVALID_VALUE);
    const FormatInfo* f = formatInfo(internalformat);
    if (!f) return ctx->setError(GL_INVALID_ENUM, "texStorage: unsized/unsupported format 0x%x", internalformat);
    if (t->immutable) return ctx->setError(GL_INVALID_OPERATION);
    if (!t->name) return ctx->setError(GL_INVALID_OPERATION);
    int maxDim = std::max(w, dims >= 2 && t->target != GL_TEXTURE_1D_ARRAY ? h : 1);
    if (t->target == GL_TEXTURE_3D) maxDim = std::max(maxDim, d);
    int maxLevels = 1;
    while (maxDim >> maxLevels) maxLevels++;
    if (levels > maxLevels) return ctx->setError(GL_INVALID_OPERATION);
    if (t->target == GL_TEXTURE_RECTANGLE && levels != 1) return ctx->setError(GL_INVALID_OPERATION);
    if ((t->target == GL_TEXTURE_CUBE_MAP || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) && w != h) return ctx->setError(GL_INVALID_VALUE);
    if (t->target == GL_TEXTURE_CUBE_MAP_ARRAY && d % 6) return ctx->setError(GL_INVALID_VALUE);
    if (t->target == GL_TEXTURE_3D && !compressedSupports3D(f))
        return ctx->setError(GL_INVALID_OPERATION, "compressed format 0x%x can't be used for 3D textures", internalformat);
    if (t->sparse && !sparseStorageCheck(ctx, t, f, w, h, d)) return;
    for (int face = 0; face < 6; face++)
        for (int l = 0; l < Texture::kMaxLevels; l++) t->levels[face][l] = TexImageLevel();
    int faces = t->target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
    for (int l = 0; l < levels; l++) {
        int lw = std::max(1, w >> l);
        int lh = t->target == GL_TEXTURE_1D_ARRAY ? h : std::max(1, h >> l);
        int ld = t->target == GL_TEXTURE_3D ? std::max(1, d >> l) : d;
        if (dims == 1) lh = 1;
        for (int face = 0; face < faces; face++) {
            TexImageLevel& L = t->levels[face][l];
            L.width = lw;
            L.height = lh;
            L.depth = ld;
            L.internalFormat = internalformat;
            L.fmt = f;
        }
    }
    t->immutable = true;
    t->immutableLevels = levels;
    t->storageDirty = true;
    ensureTextureStorage(ctx, t);
}

API void APIENTRY glTexStorage1D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width) {
    MGL_CTX();
    if (target != GL_TEXTURE_1D) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorage(ctx, t, levels, internalformat, width, 1, 1, 1);
}

API void APIENTRY glTexStorage2D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
    MGL_CTX();
    if (target != GL_TEXTURE_2D && target != GL_TEXTURE_1D_ARRAY && target != GL_TEXTURE_RECTANGLE && target != GL_TEXTURE_CUBE_MAP)
        return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorage(ctx, t, levels, internalformat, width, height, 1, 2);
}

API void APIENTRY glTexStorage3D(GLenum target, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) {
    MGL_CTX();
    if (target != GL_TEXTURE_3D && target != GL_TEXTURE_2D_ARRAY && target != GL_TEXTURE_CUBE_MAP_ARRAY) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorage(ctx, t, levels, internalformat, width, height, depth, 3);
}

API void APIENTRY glTextureStorage1D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texStorage(ctx, t, levels, internalformat, width, 1, 1, 1);
}

API void APIENTRY glTextureStorage2D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texStorage(ctx, t, levels, internalformat, width, height, 1, 2);
}

API void APIENTRY glTextureStorage3D(GLuint texture, GLsizei levels, GLenum internalformat, GLsizei width, GLsizei height, GLsizei depth) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texStorage(ctx, t, levels, internalformat, width, height, depth, 3);
}

static void texStorageMS(Context* ctx, Texture* t, GLsizei samples, GLenum internalformat, GLsizei w, GLsizei h, GLsizei d,
                         GLboolean fixed, bool immutable, bool is3D) {
    if (t->target != (is3D ? GL_TEXTURE_2D_MULTISAMPLE_ARRAY : GL_TEXTURE_2D_MULTISAMPLE))
        return ctx->setError(GL_INVALID_OPERATION, "multisample storage: wrong texture target");  // DSA only
    const FormatInfo* f = resolveInternalFormat(internalformat, GL_RGBA, GL_UNSIGNED_BYTE);
    // TexStorage*Multisample needs a sized format.
    if (!f || !f->renderable || (immutable && f->internalFormat != internalformat)) return ctx->setError(GL_INVALID_ENUM);
    // TexStorage*Multisample needs sizes >= 1; TexImage*Multisample only rejects negative ones.
    int minSize = immutable ? 1 : 0;
    if (samples < 1 || w < minSize || h < minSize || d < minSize) return ctx->setError(GL_INVALID_VALUE);
    if (w > 16384 || h > 16384 || d > 2048) return ctx->setError(GL_INVALID_VALUE);
    if (samples > 4) return ctx->setError(GL_INVALID_OPERATION);
    if (t->immutable) return ctx->setError(GL_INVALID_OPERATION);
    if (immutable && t->sparse && !sparseStorageCheck(ctx, t, f, w, h, d)) return;
    // Metal multisample textures need >= 2 samples; GL allows rounding the count up.
    int s = samples <= 2 && [ctx->dev->mtl supportsTextureSampleCount:2] ? 2 : 4;
    t->samples = s;
    t->fixedSampleLocations = fixed;
    TexImageLevel& L = t->levels[0][0];
    L.width = w;
    L.height = h;
    L.depth = d;
    L.internalFormat = internalformat;
    L.fmt = f;
    t->immutable = immutable;
    t->immutableLevels = 1;
    t->storageDirty = true;
    if (w && h && d) ensureTextureStorage(ctx, t);
}

API void APIENTRY glTexImage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                          GLboolean fixed) {
    MGL_CTX();
    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE) return;
    if (target != GL_TEXTURE_2D_MULTISAMPLE) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, 1, fixed, false, false);
}

API void APIENTRY glTexImage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                          GLsizei depth, GLboolean fixed) {
    MGL_CTX();
    if (target == GL_PROXY_TEXTURE_2D_MULTISAMPLE_ARRAY) return;
    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, depth, fixed, false, true);
}

API void APIENTRY glTexStorage2DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                            GLboolean fixed) {
    MGL_CTX();
    if (target != GL_TEXTURE_2D_MULTISAMPLE) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, 1, fixed, true, false);
}

API void APIENTRY glTexStorage3DMultisample(GLenum target, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                            GLsizei depth, GLboolean fixed) {
    MGL_CTX();
    if (target != GL_TEXTURE_2D_MULTISAMPLE_ARRAY) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, depth, fixed, true, true);
}

API void APIENTRY glTextureStorage2DMultisample(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                                GLboolean fixed) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, 1, fixed, true, false);
}

API void APIENTRY glTextureStorage3DMultisample(GLuint texture, GLsizei samples, GLenum internalformat, GLsizei width, GLsizei height,
                                                GLsizei depth, GLboolean fixed) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texStorageMS(ctx, t, samples, internalformat, width, height, depth, fixed, true, true);
}

// --- parameters ---
static bool validWrap(GLenum w, bool core) {
    return w == GL_REPEAT || w == GL_MIRRORED_REPEAT || w == GL_CLAMP_TO_EDGE || w == GL_CLAMP_TO_BORDER || w == GL_MIRROR_CLAMP_TO_EDGE ||
           (!core && w == GL_CLAMP);
}

// Applies a sampler parameter; returns false if pname isn't a sampler parameter.
static bool samplerParam(Context* ctx, SamplerState& s, GLenum pname, const double* v, bool isInt, bool rect) {
    GLenum e = (GLenum)(int64_t)v[0];
    switch (pname) {
    case GL_TEXTURE_MIN_FILTER:
        if (e != GL_NEAREST && e != GL_LINEAR && e != GL_NEAREST_MIPMAP_NEAREST && e != GL_LINEAR_MIPMAP_NEAREST &&
            e != GL_NEAREST_MIPMAP_LINEAR && e != GL_LINEAR_MIPMAP_LINEAR)
            return ctx->setError(GL_INVALID_ENUM), true;
        if (rect && e != GL_NEAREST && e != GL_LINEAR) return ctx->setError(GL_INVALID_ENUM), true;
        s.minFilter = e;
        return true;
    case GL_TEXTURE_MAG_FILTER:
        if (e != GL_NEAREST && e != GL_LINEAR) return ctx->setError(GL_INVALID_ENUM), true;
        s.magFilter = e;
        return true;
    case GL_TEXTURE_WRAP_S: case GL_TEXTURE_WRAP_T: case GL_TEXTURE_WRAP_R:
        if (!validWrap(e, ctx->coreProfile)) return ctx->setError(GL_INVALID_ENUM), true;
        if (rect && (e == GL_REPEAT || e == GL_MIRRORED_REPEAT || e == GL_MIRROR_CLAMP_TO_EDGE)) return ctx->setError(GL_INVALID_ENUM), true;
        (pname == GL_TEXTURE_WRAP_S ? s.wrapS : pname == GL_TEXTURE_WRAP_T ? s.wrapT : s.wrapR) = e;
        return true;
    case GL_TEXTURE_MIN_LOD: s.minLod = (float)v[0]; return true;
    case GL_TEXTURE_MAX_LOD: s.maxLod = (float)v[0]; return true;
    case GL_TEXTURE_LOD_BIAS: s.lodBias = (float)v[0]; return true;
    case GL_TEXTURE_COMPARE_MODE:
        if (e != GL_NONE && e != GL_COMPARE_REF_TO_TEXTURE) return ctx->setError(GL_INVALID_ENUM), true;
        s.compareMode = e;
        return true;
    case GL_TEXTURE_COMPARE_FUNC:
        if (e < GL_NEVER || e > GL_ALWAYS) return ctx->setError(GL_INVALID_ENUM), true;
        s.compareFunc = e;
        return true;
    case GL_TEXTURE_MAX_ANISOTROPY:
        if (v[0] < 1) return ctx->setError(GL_INVALID_VALUE), true;
        s.maxAnisotropy = (float)std::min(v[0], 16.0);
        return true;
    case GL_TEXTURE_CUBE_MAP_SEAMLESS: s.seamless = v[0] != 0; return true;
    case GL_TEXTURE_REDUCTION_MODE_ARB:
        if (!ctx->dev->caps.samplerReduction) return false;
        if (e != GL_WEIGHTED_AVERAGE_ARB && e != GL_MIN && e != GL_MAX) return ctx->setError(GL_INVALID_ENUM), true;
        s.reductionMode = e;
        return true;
    default: return false;
    }
}

static void texParam(Context* ctx, Texture* t, GLenum pname, const double* v, int count, bool isIntVec, bool isIntegerBorder = false, bool dsa = false) {
    bool rect = t->target == GL_TEXTURE_RECTANGLE;
    bool ms = t->target == GL_TEXTURE_2D_MULTISAMPLE || t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    if (ms) {
        // Multisample textures have no sampler state (GL 4.6 8.10).
        switch (pname) {
        case GL_TEXTURE_MIN_FILTER: case GL_TEXTURE_MAG_FILTER: case GL_TEXTURE_WRAP_S: case GL_TEXTURE_WRAP_T: case GL_TEXTURE_WRAP_R:
        case GL_TEXTURE_MIN_LOD: case GL_TEXTURE_MAX_LOD: case GL_TEXTURE_LOD_BIAS: case GL_TEXTURE_COMPARE_MODE:
        case GL_TEXTURE_COMPARE_FUNC: case GL_TEXTURE_MAX_ANISOTROPY: case GL_TEXTURE_BORDER_COLOR: case GL_TEXTURE_REDUCTION_MODE_ARB:
            return ctx->setError(dsa ? GL_INVALID_OPERATION : GL_INVALID_ENUM, "sampler state on a multisample texture");
        default: break;
        }
    }
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int k = 0; k < 4; k++) t->sampler.borderColor[k] = (float)v[k];
        t->sampler.borderIsInt = isIntegerBorder;
        return;
    }
    if (samplerParam(ctx, t->sampler, pname, v, isIntVec, rect)) return;
    GLenum e = (GLenum)(int64_t)v[0];
    switch (pname) {
    case GL_TEXTURE_BASE_LEVEL:
        if (v[0] < 0) return ctx->setError(GL_INVALID_VALUE);
        if ((rect || ms) && v[0] != 0) return ctx->setError(GL_INVALID_OPERATION);
        t->baseLevel = (int)v[0];
        t->dropViews();
        if (!t->immutable) t->storageDirty = t->storageDirty || !t->levels[0][0].defined();
        break;
    case GL_TEXTURE_MAX_LEVEL:
        if (v[0] < 0) return ctx->setError(GL_INVALID_VALUE);
        t->maxLevel = (int)v[0];
        t->dropViews();
        break;
    case GL_TEXTURE_SWIZZLE_R: case GL_TEXTURE_SWIZZLE_G: case GL_TEXTURE_SWIZZLE_B: case GL_TEXTURE_SWIZZLE_A:
        if (e != GL_RED && e != GL_GREEN && e != GL_BLUE && e != GL_ALPHA && e != GL_ZERO && e != GL_ONE) return ctx->setError(GL_INVALID_ENUM);
        t->swizzle[pname - GL_TEXTURE_SWIZZLE_R] = e;
        t->dropViews();
        break;
    case GL_TEXTURE_SWIZZLE_RGBA:
        for (int k = 0; k < 4; k++) {
            GLenum s = (GLenum)(int64_t)v[k];
            if (s != GL_RED && s != GL_GREEN && s != GL_BLUE && s != GL_ALPHA && s != GL_ZERO && s != GL_ONE) return ctx->setError(GL_INVALID_ENUM);
        }
        for (int k = 0; k < 4; k++) t->swizzle[k] = (GLenum)(int64_t)v[k];
        t->dropViews();
        break;
    case GL_DEPTH_STENCIL_TEXTURE_MODE:
        if (e != GL_DEPTH_COMPONENT && e != GL_STENCIL_INDEX) return ctx->setError(GL_INVALID_ENUM);
        t->depthStencilMode = e;
        t->dropViews();
        break;
    case GL_TEXTURE_SRGB_DECODE_EXT:
        if (e != GL_DECODE_EXT && e != GL_SKIP_DECODE_EXT) return ctx->setError(GL_INVALID_ENUM);
        t->srgbDecode = e;
        t->dropViews();
        break;
    case GL_GENERATE_MIPMAP:
        if (ctx->coreProfile) return ctx->setError(GL_INVALID_ENUM);
        t->generateMipmapHint = v[0] != 0;
        break;
    case GL_TEXTURE_PRIORITY: case GL_DEPTH_TEXTURE_MODE: case GL_TEXTURE_COMPARE_FAIL_VALUE_ARB:
        if (ctx->coreProfile) return ctx->setError(GL_INVALID_ENUM);
        break;
    case GL_TEXTURE_SPARSE_ARB: case GL_VIRTUAL_PAGE_SIZE_INDEX_ARB:
        if (!ctx->dev->caps.sparse) return ctx->setError(GL_INVALID_ENUM, "glTexParameter(0x%x)", pname);
        if (t->immutable) return ctx->setError(GL_INVALID_OPERATION, "sparse parameters of immutable storage");
        if (pname == GL_VIRTUAL_PAGE_SIZE_INDEX_ARB) {
            t->pageSizeIndex = (int)v[0];
        } else {
            if (v[0] != 0 && !sparseTargetSupported(t->target)) return ctx->setError(GL_INVALID_VALUE, "target 0x%x can't be sparse", t->target);
            t->sparse = v[0] != 0;
        }
        break;
    default: ctx->setError(GL_INVALID_ENUM, "glTexParameter(0x%x)", pname);
    }
}

#define TEXPARAM_IMPL(Name, Type, IsInt, count)                                           \
    API void APIENTRY glTexParameter##Name(GLenum target, GLenum pname, Type param) {   \
        MGL_CTX();                                                                        \
        Texture* t = paramTexForTarget(ctx, target);                                           \
        if (!t) return;                                                                   \
        if (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA)         \
            return ctx->setError(GL_INVALID_ENUM);                                        \
        double v[4] = {(double)param, 0, 0, 0};                                           \
        texParam(ctx, t, pname, v, 1, IsInt);                                             \
    }                                                                                     \
    API void APIENTRY glTextureParameter##Name(GLuint texture, GLenum pname, Type param) { \
        MGL_CTX();                                                                        \
        Texture* t = namedTex(ctx, texture);                                              \
        if (!t) return;                                                                   \
        if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);   \
        if (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA)         \
            return ctx->setError(GL_INVALID_ENUM);                                        \
        double v[4] = {(double)param, 0, 0, 0};                                           \
        texParam(ctx, t, pname, v, 1, IsInt, false, true);                                            \
    }
TEXPARAM_IMPL(f, GLfloat, false, 1)
TEXPARAM_IMPL(i, GLint, true, 1)

static void texParamVec(Context* ctx, Texture* t, GLenum pname, const double* v, bool isInt, bool integerBorder, bool dsa = false) {
    int n = (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA) ? 4 : 1;
    texParam(ctx, t, pname, v, n, isInt, integerBorder, dsa);
}

#define TEXPARAMV_IMPL(Name, Type, IsInt, Norm, IntBorder)                                          \
    API void APIENTRY glTexParameter##Name(GLenum target, GLenum pname, const Type* params) {     \
        MGL_CTX();                                                                                  \
        Texture* t = paramTexForTarget(ctx, target);                                                     \
        if (!t) return;                                                                             \
        double v[4];                                                                                \
        int n = (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA) ? 4 : 1;    \
        for (int k = 0; k < n; k++) v[k] = (Norm && pname == GL_TEXTURE_BORDER_COLOR) ? params[k] / 2147483647.0 : (double)params[k]; \
        texParamVec(ctx, t, pname, v, IsInt, IntBorder);                                            \
    }                                                                                               \
    API void APIENTRY glTextureParameter##Name(GLuint texture, GLenum pname, const Type* params) { \
        MGL_CTX();                                                                                  \
        Texture* t = namedTex(ctx, texture);                                                        \
        if (!t) return;                                                                             \
        if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);             \
        double v[4];                                                                                \
        int n = (pname == GL_TEXTURE_BORDER_COLOR || pname == GL_TEXTURE_SWIZZLE_RGBA) ? 4 : 1;    \
        for (int k = 0; k < n; k++) v[k] = (Norm && pname == GL_TEXTURE_BORDER_COLOR) ? params[k] / 2147483647.0 : (double)params[k]; \
        texParamVec(ctx, t, pname, v, IsInt, IntBorder, true);                                           \
    }
TEXPARAMV_IMPL(fv, GLfloat, false, false, false)
TEXPARAMV_IMPL(iv, GLint, true, true, false)
TEXPARAMV_IMPL(Iiv, GLint, true, false, true)
TEXPARAMV_IMPL(Iuiv, GLuint, true, false, true)

static int viewLayersOf(const Texture* t);

static bool getTexParam(Context* ctx, Texture* t, GLenum pname, double* out, int* n) {
    *n = 1;
    const SamplerState& s = t->sampler;
    switch (pname) {
    case GL_TEXTURE_MIN_FILTER: out[0] = s.minFilter; return true;
    case GL_TEXTURE_MAG_FILTER: out[0] = s.magFilter; return true;
    case GL_TEXTURE_WRAP_S: out[0] = s.wrapS; return true;
    case GL_TEXTURE_WRAP_T: out[0] = s.wrapT; return true;
    case GL_TEXTURE_WRAP_R: out[0] = s.wrapR; return true;
    case GL_TEXTURE_MIN_LOD: out[0] = s.minLod; return true;
    case GL_TEXTURE_MAX_LOD: out[0] = s.maxLod; return true;
    case GL_TEXTURE_LOD_BIAS: out[0] = s.lodBias; return true;
    case GL_TEXTURE_COMPARE_MODE: out[0] = s.compareMode; return true;
    case GL_TEXTURE_COMPARE_FUNC: out[0] = s.compareFunc; return true;
    case GL_TEXTURE_MAX_ANISOTROPY: out[0] = s.maxAnisotropy; return true;
    case GL_TEXTURE_REDUCTION_MODE_ARB: out[0] = s.reductionMode; return ctx->dev->caps.samplerReduction;
    case GL_TEXTURE_BORDER_COLOR: *n = 4; for (int k = 0; k < 4; k++) out[k] = s.borderColor[k]; return true;
    case GL_TEXTURE_BASE_LEVEL: out[0] = t->baseLevel; return true;
    case GL_TEXTURE_MAX_LEVEL: out[0] = t->maxLevel; return true;
    case GL_TEXTURE_SWIZZLE_R: case GL_TEXTURE_SWIZZLE_G: case GL_TEXTURE_SWIZZLE_B: case GL_TEXTURE_SWIZZLE_A:
        out[0] = t->swizzle[pname - GL_TEXTURE_SWIZZLE_R]; return true;
    case GL_TEXTURE_SWIZZLE_RGBA: *n = 4; for (int k = 0; k < 4; k++) out[k] = t->swizzle[k]; return true;
    case GL_DEPTH_STENCIL_TEXTURE_MODE: out[0] = t->depthStencilMode; return true;
    case GL_TEXTURE_SRGB_DECODE_EXT: out[0] = t->srgbDecode; return true;
    case GL_TEXTURE_IMMUTABLE_FORMAT: out[0] = t->immutable; return true;
    case GL_TEXTURE_IMMUTABLE_LEVELS: out[0] = t->immutable ? (t->viewParent ? t->viewImmutableLevels : t->immutableLevels) : 0; return true;
    case GL_TEXTURE_VIEW_MIN_LEVEL: out[0] = t->viewMinLevel; return true;
    case GL_TEXTURE_VIEW_NUM_LEVELS: out[0] = t->viewParent ? t->viewNumLevels : t->immutable ? t->immutableLevels : 0; return true;
    case GL_TEXTURE_VIEW_MIN_LAYER: out[0] = t->viewMinLayer; return true;
    case GL_TEXTURE_VIEW_NUM_LAYERS: out[0] = t->viewParent ? t->viewNumLayers : t->immutable ? viewLayersOf(t) : 0; return true;
    case GL_TEXTURE_TARGET: out[0] = t->target; return true;
    case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE: out[0] = GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE; return true;
    case GL_GENERATE_MIPMAP: out[0] = t->generateMipmapHint; return !ctx->coreProfile;
    case GL_TEXTURE_RESIDENT: out[0] = 1; return !ctx->coreProfile;
    case GL_TEXTURE_PRIORITY: out[0] = 1; return !ctx->coreProfile;
    case GL_DEPTH_TEXTURE_MODE: out[0] = GL_LUMINANCE; return !ctx->coreProfile;
    case GL_TEXTURE_SPARSE_ARB: out[0] = t->sparse; return ctx->dev->caps.sparse;
    case GL_VIRTUAL_PAGE_SIZE_INDEX_ARB: out[0] = t->pageSizeIndex; return ctx->dev->caps.sparse;
    case GL_NUM_SPARSE_LEVELS_ARB: out[0] = t->sparse && t->immutable ? t->sparseLevels : 0; return ctx->dev->caps.sparse;
    default: return false;
    }
}

#define GETTEXPARAM_IMPL(Name, Type)                                                         \
    API void APIENTRY glGetTexParameter##Name(GLenum target, GLenum pname, Type* params) {  \
        MGL_CTX();                                                                           \
        Texture* t = paramTexForTarget(ctx, target);                                              \
        if (!t) return;                                                                      \
        double v[4];                                                                         \
        int n;                                                                               \
        if (!getTexParam(ctx, t, pname, v, &n)) return ctx->setError(GL_INVALID_ENUM);       \
        for (int k = 0; k < n; k++) params[k] = (Type)v[k];                                  \
    }                                                                                        \
    API void APIENTRY glGetTextureParameter##Name(GLuint texture, GLenum pname, Type* params) { \
        MGL_CTX();                                                                           \
        Texture* t = namedTex(ctx, texture);                                                 \
        if (!t) return;                                                                      \
        if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);      \
        double v[4];                                                                         \
        int n;                                                                               \
        if (!getTexParam(ctx, t, pname, v, &n)) return ctx->setError(GL_INVALID_ENUM);       \
        for (int k = 0; k < n; k++) params[k] = (Type)v[k];                                  \
    }
GETTEXPARAM_IMPL(fv, GLfloat)
GETTEXPARAM_IMPL(Iiv, GLint)
GETTEXPARAM_IMPL(Iuiv, GLuint)

API void APIENTRY glGetTexParameteriv(GLenum target, GLenum pname, GLint* params) {
    MGL_CTX();
    Texture* t = paramTexForTarget(ctx, target);
    if (!t) return;
    double v[4];
    int n;
    if (!getTexParam(ctx, t, pname, v, &n)) return ctx->setError(GL_INVALID_ENUM);
    for (int k = 0; k < n; k++) params[k] = pname == GL_TEXTURE_BORDER_COLOR ? (GLint)llround(std::clamp(v[k], -1.0, 1.0) * 2147483647.0) : (GLint)lround(v[k]);
}

API void APIENTRY glGetTextureParameteriv(GLuint texture, GLenum pname, GLint* params) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (!t) return;
    if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);
    double v[4];
    int n;
    if (!getTexParam(ctx, t, pname, v, &n)) return ctx->setError(GL_INVALID_ENUM);
    for (int k = 0; k < n; k++) params[k] = pname == GL_TEXTURE_BORDER_COLOR ? (GLint)llround(std::clamp(v[k], -1.0, 1.0) * 2147483647.0) : (GLint)lround(v[k]);
}

static bool getLevelParam(Context* ctx, Texture* t, int face, GLint level, GLenum pname, double* out) {
    // buffer textures (and other single-level targets) only have level 0
    bool single = t->target == GL_TEXTURE_BUFFER || t->target == GL_TEXTURE_RECTANGLE || t->target == GL_TEXTURE_2D_MULTISAMPLE ||
                  t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    if (level < 0 || level >= Texture::kMaxLevels || (single && level != 0)) {
        ctx->setError(GL_INVALID_VALUE);
        return false;
    }
    const TexImageLevel& L = t->target == GL_TEXTURE_BUFFER ? t->levels[0][0] : t->levels[face][level];
    const FormatInfo* f = L.fmt;
    if (t->target == GL_TEXTURE_BUFFER) f = formatInfo(t->bufferFormat);
    switch (pname) {
    case GL_TEXTURE_WIDTH:
        if (t->target == GL_TEXTURE_BUFFER) {
            GLsizeiptr sz = t->buffer ? (t->bufferSize < 0 ? t->buffer->size - t->bufferOffset : t->bufferSize) : 0;
            // Clamped to MAX_TEXTURE_BUFFER_SIZE (GL 4.6 8.9).
            *out = f ? std::min<GLsizeiptr>(sz / bufferTexelBytes(f), (GLsizeiptr)1 << 27) : 0;
        } else *out = L.width;
        return true;
    case GL_TEXTURE_HEIGHT: *out = t->target == GL_TEXTURE_BUFFER ? 1 : L.height; return true;
    case GL_TEXTURE_DEPTH: *out = t->target == GL_TEXTURE_BUFFER ? 1 : L.depth; return true;
    case GL_TEXTURE_INTERNAL_FORMAT:
        *out = t->target == GL_TEXTURE_BUFFER ? t->bufferFormat : L.defined() ? L.internalFormat : (ctx->coreProfile ? GL_RGBA : 1);
        return true;
    case GL_TEXTURE_SAMPLES: *out = t->samples; return true;
    case GL_TEXTURE_FIXED_SAMPLE_LOCATIONS: *out = t->fixedSampleLocations; return true;
    case GL_TEXTURE_COMPRESSED: *out = f && f->comp == Comp::Compressed; return true;
    case GL_TEXTURE_COMPRESSED_IMAGE_SIZE:
        if (!f || f->comp != Comp::Compressed) return ctx->setError(GL_INVALID_OPERATION), false;
        *out = (double)storageImageBytes(f, L.width, L.height) * std::max(1, L.depth);
        return true;
    case GL_TEXTURE_BUFFER_DATA_STORE_BINDING: *out = t->buffer ? t->buffer->name : 0; return true;
    case GL_TEXTURE_BUFFER_OFFSET: *out = t->bufferOffset; return true;
    case GL_TEXTURE_BUFFER_SIZE: *out = t->buffer ? (t->bufferSize < 0 ? t->buffer->size : t->bufferSize) : 0; return true;
    case GL_TEXTURE_RED_SIZE: *out = f ? f->rBits : 0; return true;
    case GL_TEXTURE_GREEN_SIZE: *out = f ? f->gBits : 0; return true;
    case GL_TEXTURE_BLUE_SIZE: *out = f ? f->bBits : 0; return true;
    case GL_TEXTURE_ALPHA_SIZE: *out = f ? f->aBits : 0; return true;
    case GL_TEXTURE_DEPTH_SIZE: *out = f ? f->dBits : 0; return true;
    case GL_TEXTURE_STENCIL_SIZE: *out = f ? f->sBits : 0; return true;
    case GL_TEXTURE_SHARED_SIZE: *out = f && f->comp == Comp::P999E5 ? 5 : 0; return true;
    case GL_TEXTURE_LUMINANCE_SIZE: *out = f && (f->baseFormat == GL_LUMINANCE || f->baseFormat == GL_LUMINANCE_ALPHA) ? f->rBits : 0; return true;
    case GL_TEXTURE_INTENSITY_SIZE: *out = f && f->baseFormat == GL_INTENSITY ? f->rBits : 0; return true;
    case GL_TEXTURE_RED_TYPE: case GL_TEXTURE_GREEN_TYPE: case GL_TEXTURE_BLUE_TYPE: case GL_TEXTURE_ALPHA_TYPE: {
        if (!f) { *out = GL_NONE; return true; }
        int bits = pname == GL_TEXTURE_RED_TYPE ? f->rBits : pname == GL_TEXTURE_GREEN_TYPE ? f->gBits : pname == GL_TEXTURE_BLUE_TYPE ? f->bBits : f->aBits;
        *out = bits ? f->glType : GL_NONE;
        return true;
    }
    case GL_TEXTURE_DEPTH_TYPE: *out = f && f->dBits ? (f->glType == GL_FLOAT ? GL_FLOAT : GL_UNSIGNED_NORMALIZED) : GL_NONE; return true;
    case GL_TEXTURE_LUMINANCE_TYPE: case GL_TEXTURE_INTENSITY_TYPE: *out = GL_NONE; return true;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
}

static GLenum levelQueryTarget(Context* ctx, GLenum target, Texture** t, int* face) {
    *face = faceIndex(target);
    if (isProxy(target)) {
        auto it = ctx->proxyTextures.find(target);
        *t = it == ctx->proxyTextures.end() ? nullptr : it->second.get();
        *face = 0;
        return target;
    }
    if (target == GL_TEXTURE_CUBE_MAP) {
        ctx->setError(GL_INVALID_ENUM);
        *t = nullptr;
        return 0;
    }
    *t = texForTarget(ctx, target);
    return target;
}

API void APIENTRY glGetTexLevelParameteriv(GLenum target, GLint level, GLenum pname, GLint* params) {
    MGL_CTX();
    Texture* t;
    int face;
    if (!levelQueryTarget(ctx, target, &t, &face)) return;
    if (!t) {
        if (isProxy(target)) *params = 0;
        return;
    }
    double v;
    if (getLevelParam(ctx, t, face, level, pname, &v)) *params = (GLint)v;
}

API void APIENTRY glGetTexLevelParameterfv(GLenum target, GLint level, GLenum pname, GLfloat* params) {
    MGL_CTX();
    Texture* t;
    int face;
    if (!levelQueryTarget(ctx, target, &t, &face)) return;
    if (!t) {
        if (isProxy(target)) *params = 0;
        return;
    }
    double v;
    if (getLevelParam(ctx, t, face, level, pname, &v)) *params = (GLfloat)v;
}

API void APIENTRY glGetTextureLevelParameteriv(GLuint texture, GLint level, GLenum pname, GLint* params) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (!t) return;
    double v;
    if (getLevelParam(ctx, t, 0, level, pname, &v)) *params = (GLint)v;
}

API void APIENTRY glGetTextureLevelParameterfv(GLuint texture, GLint level, GLenum pname, GLfloat* params) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (!t) return;
    double v;
    if (getLevelParam(ctx, t, 0, level, pname, &v)) *params = (GLfloat)v;
}

// --- mipmap generation ---
void generateMipmaps(Context* ctx, Texture* t) {
    int base = std::clamp(t->baseLevel, 0, Texture::kMaxLevels - 1);
    const TexImageLevel& B = t->levels[0][base];
    if (!B.defined()) return;
    if (!t->immutable) {
        int w = B.width, h = B.height, d = B.depth;
        int l = base;
        int faces = t->faces();
        while ((w > 1 || h > 1 || (t->target == GL_TEXTURE_3D && d > 1)) && l < std::min(t->maxLevel, Texture::kMaxLevels - 1)) {
            w = std::max(1, w / 2);
            if (t->target != GL_TEXTURE_1D_ARRAY && t->target != GL_TEXTURE_1D) h = std::max(1, h / 2);
            if (t->target == GL_TEXTURE_3D) d = std::max(1, d / 2);
            l++;
            for (int f = 0; f < faces; f++) defineLevel(t, f, l, w, h, d, B.internalFormat, B.fmt);
        }
    }
    if (!ensureTextureStorage(ctx, t)) return;
    if (t->storageLevels <= 1) return;
    id<MTLTexture> tex = t->mtl;
    if (base > 0 || t->maxLevel < t->storageLevels - 1) {
        int last = std::min(t->maxLevel, t->storageLevels - 1);
        if (last <= base) return;
        NSUInteger slices = tex.textureType == MTLTextureTypeCube ? 6 : tex.textureType == MTLTextureTypeCubeArray ? tex.arrayLength * 6 : tex.arrayLength;
        tex = [t->mtl newTextureViewWithPixelFormat:t->mtl.pixelFormat textureType:t->mtl.textureType
                                             levels:NSMakeRange(base, last - base + 1) slices:NSMakeRange(0, slices)];
        ctx->dev->retainUntil(tex, ctx->serial());
    }
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e generateMipmapsForTexture:tex];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    markTextureWrite(ctx, t);
}

static void genMip(Context* ctx, Texture* t) {
    if (t->target == GL_TEXTURE_BUFFER || t->target == GL_TEXTURE_RECTANGLE || t->samples > 0) return ctx->setError(GL_INVALID_ENUM);
    if (t->target == GL_TEXTURE_CUBE_MAP) {
        int b = t->baseLevel;
        for (int f = 1; f < 6; f++)
            if (t->levels[f][b].fmt != t->levels[0][b].fmt || t->levels[f][b].width != t->levels[0][b].width)
                return ctx->setError(GL_INVALID_OPERATION);
    }
    const TexImageLevel& B = t->levels[0][std::clamp(t->baseLevel, 0, Texture::kMaxLevels - 1)];
    if (B.defined() && (B.fmt->comp == Comp::Compressed || isDepthOrStencil(B.fmt) || !B.fmt->filterable)) {
        if (B.fmt->comp == Comp::Compressed || B.fmt->kind == FmtKind::UInt || B.fmt->kind == FmtKind::SInt) return ctx->setError(GL_INVALID_OPERATION);
    }
    generateMipmaps(ctx, t);
}

API void APIENTRY glGenerateMipmap(GLenum target) {
    MGL_CTX();
    if (target != GL_TEXTURE_1D && target != GL_TEXTURE_2D && target != GL_TEXTURE_3D && target != GL_TEXTURE_1D_ARRAY &&
        target != GL_TEXTURE_2D_ARRAY && target != GL_TEXTURE_CUBE_MAP && target != GL_TEXTURE_CUBE_MAP_ARRAY)
        return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) genMip(ctx, t);
}

API void APIENTRY glGenerateTextureMipmap(GLuint texture) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) genMip(ctx, t);
}

// --- buffer textures ---
static void texBuffer(Context* ctx, Texture* t, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size, bool range) {
    if (t->target != GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);
    const FormatInfo* f = formatInfo(internalformat);
    if (!f || f->comp == Comp::Compressed || isDepthOrStencil(f) || f->baseFormat == GL_LUMINANCE) return ctx->setError(GL_INVALID_ENUM);
    Buffer* b = nullptr;
    if (buffer) {
        b = ctx->share->buffers.get(buffer);
        if (!b) return ctx->setError(GL_INVALID_OPERATION);
    }
    if (range && b) {
        if (offset < 0 || size <= 0 || offset + size > b->size) return ctx->setError(GL_INVALID_VALUE);
        if (offset % 64) return ctx->setError(GL_INVALID_VALUE);
    }
    t->buffer = b;
    t->bufferFormat = internalformat;
    t->bufferOffset = range ? offset : 0;
    t->bufferSize = range ? size : -1;
    t->bufferGeneration = ~0u;
    if (t->mtl) ctx->dev->retainUntil(t->mtl, t->lastUse);
    t->mtl = nil;
    t->levels[0][0].fmt = f;
    t->levels[0][0].internalFormat = internalformat;
}

API void APIENTRY glTexBuffer(GLenum target, GLenum internalformat, GLuint buffer) {
    MGL_CTX();
    if (target != GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_ENUM);
    texBuffer(ctx, texForTarget(ctx, target), internalformat, buffer, 0, 0, false);
}

API void APIENTRY glTexBufferRange(GLenum target, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    MGL_CTX();
    if (target != GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_ENUM);
    texBuffer(ctx, texForTarget(ctx, target), internalformat, buffer, offset, size, true);
}

API void APIENTRY glTextureBuffer(GLuint texture, GLenum internalformat, GLuint buffer) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texBuffer(ctx, t, internalformat, buffer, 0, 0, false);
}

API void APIENTRY glTextureBufferRange(GLuint texture, GLenum internalformat, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) texBuffer(ctx, t, internalformat, buffer, offset, size, true);
}

// --- readback ---

// Linear (non-sRGB) twin of a compressed sRGB format, for reading raw encoded values.
static GLenum linearCompressedTwin(GLenum f) {
    switch (f) {
    case GL_COMPRESSED_SRGB_S3TC_DXT1_EXT: return GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT: return GL_COMPRESSED_RGBA_S3TC_DXT1_EXT;
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT: return GL_COMPRESSED_RGBA_S3TC_DXT3_EXT;
    case GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT: return GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
    case GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM: return GL_COMPRESSED_RGBA_BPTC_UNORM;
    case GL_COMPRESSED_SRGB8_ETC2: return GL_COMPRESSED_RGB8_ETC2;
    case GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2: return GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
    case GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC: return GL_COMPRESSED_RGBA8_ETC2_EAC;
    default:
        if (f >= GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR && f <= GL_COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR)
            return f - GL_COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR + GL_COMPRESSED_RGBA_ASTC_4x4_KHR;
        return f;
    }
}

// Decodes a region of one slice of a compressed (non-3D) texture level to RGBA32F by
// sampling it on the GPU. Synchronous.
static bool decodeCompressedRegion(Context* ctx, Texture* t, int level, int slice, int x, int y, int w, int h, std::vector<uint8_t>& out,
                                   size_t& rowBytes) {
    const FormatInfo* f = t->storageFmt;
    id<MTLTexture> src = t->mtl;
    if (f->srgb) {
        if (!(src.usage & MTLTextureUsagePixelFormatView)) return false;
        const FormatInfo* lf = formatInfo(linearCompressedTwin(f->internalFormat));
        if (!lf) return false;
        src = [src newTextureViewWithPixelFormat:lf->mtl];
        ctx->dev->retainUntil(src, ctx->serial());
    }
    int lw = std::max(1, (int)(t->mtl.width >> level)), lh = std::max(1, (int)(t->mtl.height >> level));
    MTLTextureDescriptor* td = texture2DDescriptor(MTLPixelFormatRGBA32Float, lw, lh);
    td.storageMode = MTLStorageModePrivate;
    td.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
    id<MTLTexture> tmp = [ctx->dev->mtl newTextureWithDescriptor:td];
    if (!tmp) return false;
    ctx->dev->makeResident(tmp);
    BlitParams bp;
    bp.src = src;
    bp.srcLevel = level;
    bp.srcSlice = slice;
    bp.srcRect[0] = 0, bp.srcRect[1] = 0, bp.srcRect[2] = lw, bp.srcRect[3] = lh;
    bp.dstRect[0] = 0, bp.dstRect[1] = 0, bp.dstRect[2] = lw, bp.dstRect[3] = lh;
    blitToTexture(ctx, bp, tmp, 0, 0);
    bool ok = readTextureRegion(ctx, tmp, 0, 0, x, y, w, h, out, rowBytes, formatInfo(GL_RGBA32F), MTLBlitOptionNone);
    ctx->dev->releaseAfter(tmp, ctx->serial());
    return ok;
}

static void getTexImage(Context* ctx, Texture* t, GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels,
                        int x = 0, int y = 0, int z = 0, int w = -1, int h = -1, int d = -1, bool rawCompressed = false) {
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    if (GLenum e = pixelFormatTypeError(format, type)) return ctx->setError(e);
    int face = faceIndex(target);
    const TexImageLevel& L = t->levels[face][level];
    if (!L.defined()) return;
    {
        // §8.11.4: integer-ness must match; DEPTH_STENCIL needs a depth/stencil texture,
        // DEPTH_COMPONENT a depth one, STENCIL_INDEX a stencil one; color formats a color texture.
        const FormatInfo* f = L.fmt;
        bool intTex = f->kind == FmtKind::UInt || f->kind == FmtKind::SInt;
        bool intFmt = format == GL_RED_INTEGER || format == GL_GREEN_INTEGER || format == GL_BLUE_INTEGER || format == GL_RG_INTEGER ||
                      format == GL_RGB_INTEGER || format == GL_RGBA_INTEGER || format == GL_BGR_INTEGER || format == GL_BGRA_INTEGER ||
                      format == GL_ALPHA_INTEGER;
        bool depth = f->baseFormat == GL_DEPTH_COMPONENT, ds = f->baseFormat == GL_DEPTH_STENCIL, st = f->baseFormat == GL_STENCIL_INDEX;
        bool bad = false;
        if (format == GL_DEPTH_STENCIL) bad = !ds;
        else if (format == GL_DEPTH_COMPONENT) bad = !depth && !ds;
        else if (format == GL_STENCIL_INDEX) bad = !st && !ds;
        else bad = depth || ds || st || intTex != intFmt;
        if (bad) return ctx->setError(GL_INVALID_OPERATION);
    }
    if (w >= 0) {
        // §8.11.4 GetTextureSubImage: the region must lie in the image
        int W = L.width, H = L.height, D = t->target == GL_TEXTURE_CUBE_MAP ? 6 : L.depth;
        if (t->target == GL_TEXTURE_1D) H = D = 1;
        else if (t->target == GL_TEXTURE_1D_ARRAY || t->target == GL_TEXTURE_2D || t->target == GL_TEXTURE_RECTANGLE) D = 1;
        if ((int64_t)x + w > W || (int64_t)y + h > H || (int64_t)z + d > D) return ctx->setError(GL_INVALID_VALUE);
    }
    if (w < 0) {
        w = L.width;
        h = L.height;
        d = t->target == GL_TEXTURE_CUBE_MAP && !isCubeFace(target) ? 6 : L.depth;
    }
    if (!ensureTextureStorage(ctx, t)) return;
    const FormatInfo* f = t->storageFmt;
    size_t need = clientImageSize(format, type, w, h, d, ctx->pack);
    uint8_t* dst = (uint8_t*)pixels;
    PackStaging staging;
    if (Buffer* pbo = ctx->pixelPackBuffer.get()) {
        if (pboOffsetMisaligned(pbo, pixels, type)) return ctx->setError(GL_INVALID_OPERATION, "unaligned PBO offset");
        if (pbo->mapped && !(pbo->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION);
        if ((size_t)pixels + need > (size_t)pbo->size) return ctx->setError(GL_INVALID_OPERATION);
        if (pbo->lastUse && !ctx->dev->isComplete(pbo->lastUse)) ctx->waitForResource(pbo->lastUse);
        dst = staging.begin(ctx, pbo, (size_t)pixels, need);
    } else if (bufSize >= 0 && need > (size_t)bufSize) {
        return ctx->setError(GL_INVALID_OPERATION);
    }
    ClientLayout cl = clientLayout(format, type, w, h, ctx->pack);
    dst += cl.skipBytes;
    for (int i = 0; i < d; i++) {
        int slice, mz = z + i;
        if (t->target == GL_TEXTURE_CUBE_MAP) slice = isCubeFace(target) ? face : mz;
        else if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) slice = mz;
        else slice = 0;
        if (f->comp == Comp::Compressed && !rawCompressed && t->target != GL_TEXTURE_3D) {
            // Uncompressed readback of a compressed image: decode, then convert.
            std::vector<uint8_t> data;
            size_t rowBytes;
            if (decodeCompressedRegion(ctx, t, level, slice, x, y, w, h, data, rowBytes))
                convertFromStorage(formatInfo(GL_RGBA32F), format, type, data.data(), rowBytes, dst + i * cl.imageStride, cl.rowStride, w, h,
                                   ctx->pack.swapBytes);
            continue;
        }
        if (t->target == GL_TEXTURE_3D) {
            // 3D: read one depth plane at a time via a 2D view is not possible; use copy with origin z.
            size_t rowBytes = (size_t)w * f->bytesPerPixel;
            id<MTLBuffer> buf = ctx->dev->newBuffer(rowBytes * h);
            id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
            if (t->sparse)  // see readTextureRegion
                [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
                         visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias];
            [e copyFromTexture:t->mtl sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(x, y, mz) sourceSize:MTLSizeMake(w, h, 1)
                      toBuffer:buf destinationOffset:0 destinationBytesPerRow:rowBytes destinationBytesPerImage:rowBytes * h];
            ctx->finish();
            convertFromStorage(f, format, type, bufferContents(buf), rowBytes, dst + i * cl.imageStride, cl.rowStride, w, h, ctx->pack.swapBytes);
            ctx->dev->releaseAfter(buf, 0);
            continue;
        }
        int ry = y, rh = h, rslice = slice;
        if (t->target == GL_TEXTURE_1D_ARRAY) {
            // rows are layers
            for (int row = 0; row < h; row++) {
                std::vector<uint8_t> data;
                size_t rowBytes;
                readTextureRegion(ctx, t->mtl, level, y + row, x, 0, w, 1, data, rowBytes, f, MTLBlitOptionNone);
                convertFromStorage(f, format, type, data.data(), rowBytes, dst + row * cl.rowStride, cl.rowStride, w, 1, ctx->pack.swapBytes);
            }
            return;
        }
        std::vector<uint8_t> data;
        size_t rowBytes;
        if (f->comp == Comp::D32F_S8) {
            std::vector<uint8_t> dd, sd;
            size_t rb1, rb2;
            readTextureRegion(ctx, t->mtl, level, rslice, x, ry, w, rh, dd, rb1, f, MTLBlitOptionDepthFromDepthStencil);
            readTextureRegion(ctx, t->mtl, level, rslice, x, ry, w, rh, sd, rb2, f, MTLBlitOptionStencilFromDepthStencil);
            data.resize((size_t)w * rh * 8);
            for (int p = 0; p < w * rh; p++) {
                memcpy(&data[p * 8], &dd[p * 4], 4);
                uint32_t s = sd[p];
                memcpy(&data[p * 8 + 4], &s, 4);
            }
            rowBytes = (size_t)w * 8;
        } else {
            readTextureRegion(ctx, t->mtl, level, rslice, x, ry, w, rh, data, rowBytes, f, MTLBlitOptionNone);
        }
        if (f->comp == Comp::Compressed) {
            memcpy(dst + i * data.size(), data.data(), data.size());  // glGetCompressedTexImage: images tightly packed
        } else {
            convertFromStorage(f, format, type, data.data(), rowBytes, dst + i * cl.imageStride, cl.rowStride, w, rh, ctx->pack.swapBytes);
        }
    }
}

API void APIENTRY glGetTexImage(GLenum target, GLint level, GLenum format, GLenum type, void* pixels) {
    MGL_CTX();
    if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_BUFFER || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) getTexImage(ctx, t, target, level, format, type, -1, pixels);
}

API void APIENTRY glGetnTexImage(GLenum target, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels) {
    MGL_CTX();
    Texture* t = texForTarget(ctx, target);
    if (t) getTexImage(ctx, t, target, level, format, type, bufSize, pixels);
}

// Whole-texture DSA image queries: target, rectangle level and cube completeness checks.
static bool dsaImageQueryOk(Context* ctx, Texture* t, GLint level) {
    switch (t->target) {
    case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY:
    case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_CUBE_MAP: break;
    default: return ctx->setError(GL_INVALID_OPERATION, "image query on texture target 0x%x", t->target), false;
    }
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE), false;
    if (t->target == GL_TEXTURE_RECTANGLE && level != 0) return ctx->setError(GL_INVALID_VALUE), false;
    if (t->target == GL_TEXTURE_CUBE_MAP) {
        const TexImageLevel& L0 = t->levels[0][level];
        for (int f = 0; f < 6; f++) {
            const TexImageLevel& L = t->levels[f][level];
            if (!L.defined() || L.width != L0.width || L.height != L0.height || L.fmt != L0.fmt)
                return ctx->setError(GL_INVALID_OPERATION, "cube map not cube complete"), false;
        }
    }
    return true;
}

API void APIENTRY glGetTextureImage(GLuint texture, GLint level, GLenum format, GLenum type, GLsizei bufSize, void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t && dsaImageQueryOk(ctx, t, level)) getTexImage(ctx, t, t->target, level, format, type, bufSize, pixels);
}

API void APIENTRY glGetTextureSubImage(GLuint texture, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                                       GLenum format, GLenum type, GLsizei bufSize, void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (!t || !dsaImageQueryOk(ctx, t, level)) return;
    if (x < 0 || y < 0 || z < 0 || w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
    getTexImage(ctx, t, t->target, level, format, type, bufSize, pixels, x, y, z, w, h, d);
}

// Compressed image readback: raw blocks of one face/layer range.
static void getCompressedImage(Context* ctx, Texture* t, GLenum target, GLint level, GLsizei bufSize, void* pixels, int x = 0, int y = 0,
                               int z = 0, int w = -1, int h = -1, int d = -1) {
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    bool wholeCube = t->target == GL_TEXTURE_CUBE_MAP && !isCubeFace(target);
    int face = faceIndex(target);
    const TexImageLevel& L = t->levels[face][level];
    if (!L.defined() || L.fmt->comp != Comp::Compressed) return ctx->setError(GL_INVALID_OPERATION, "not a compressed image");
    if (w >= 0) {
        // §8.11.4 GetCompressedTextureSubImage: the region must lie in the image and start/end on block edges
        int W = L.width, H = L.height, D = t->target == GL_TEXTURE_CUBE_MAP ? 6 : L.depth;
        if (t->target == GL_TEXTURE_1D) H = D = 1;
        else if (t->target == GL_TEXTURE_1D_ARRAY || t->target == GL_TEXTURE_2D || t->target == GL_TEXTURE_RECTANGLE) D = 1;
        if ((int64_t)x + w > W || (int64_t)y + h > H || (int64_t)z + d > D) return ctx->setError(GL_INVALID_VALUE);
        int bw = L.fmt->blockW, bh = L.fmt->blockH;
        if (x % bw || y % bh || (w % bw && x + w != W) || (h % bh && y + h != H)) return ctx->setError(GL_INVALID_OPERATION);
    }
    if (w < 0) {
        w = L.width;
        h = L.height;
        d = wholeCube ? 6 : L.depth;
    }
    const FormatInfo* f = L.fmt;
    size_t sliceBytes = storageImageBytes(f, w, h);
    size_t need = sliceBytes * d;
    uint8_t* dst = (uint8_t*)pixels;
    PackStaging staging;
    if (Buffer* pbo = ctx->pixelPackBuffer.get()) {
        if (pbo->mapped && !(pbo->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION, "pack buffer is mapped");
        if ((size_t)pixels + need > (size_t)pbo->size) return ctx->setError(GL_INVALID_OPERATION, "pack buffer too small");
        if (pbo->lastUse && !ctx->dev->isComplete(pbo->lastUse)) ctx->waitForResource(pbo->lastUse);
        dst = staging.begin(ctx, pbo, (size_t)pixels, need);
        pbo->contentVersion++;
    } else {
        if (bufSize >= 0 && need > (size_t)bufSize) return ctx->setError(GL_INVALID_OPERATION, "bufSize too small");
        if (!pixels) return ctx->setError(GL_INVALID_OPERATION);
    }
    if (!ensureTextureStorage(ctx, t) || !t->storageFmt || t->storageFmt->comp != Comp::Compressed) {
        memset(dst, 0, need);  // stored decompressed: no block data to return
        return;
    }
    for (int i = 0; i < d; i++) {
        int slice = wholeCube ? z + i : t->target == GL_TEXTURE_CUBE_MAP ? face : z + i;
        if (t->target == GL_TEXTURE_2D || t->target == GL_TEXTURE_RECTANGLE) slice = 0;
        std::vector<uint8_t> data;
        size_t rowBytes;
        if (t->target == GL_TEXTURE_3D) {
            // 3D compressed textures are read a depth plane at a time.
            size_t bpr = (size_t)((w + f->blockW - 1) / f->blockW) * f->bytesPerPixel;
            id<MTLBuffer> buf = ctx->dev->newBuffer(sliceBytes);
            id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
            [e copyFromTexture:t->mtl sourceSlice:0 sourceLevel:level sourceOrigin:MTLOriginMake(x, y, z + i) sourceSize:MTLSizeMake(w, h, 1)
                      toBuffer:buf destinationOffset:0 destinationBytesPerRow:bpr destinationBytesPerImage:sliceBytes];
            ctx->finish();
            memcpy(dst + i * sliceBytes, bufferContents(buf), sliceBytes);
            ctx->dev->releaseAfter(buf, 0);
            continue;
        }
        readTextureRegion(ctx, t->mtl, level, slice, x, y, w, h, data, rowBytes, f, MTLBlitOptionNone);
        memcpy(dst + i * sliceBytes, data.data(), std::min(sliceBytes, data.size()));
    }
}

API void APIENTRY glGetCompressedTexImage(GLenum target, GLint level, void* img) {
    MGL_CTX();
    if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_BUFFER || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) getCompressedImage(ctx, t, target, level, -1, img);
}

API void APIENTRY glGetnCompressedTexImage(GLenum target, GLint level, GLsizei bufSize, void* img) {
    MGL_CTX();
    if (target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_BUFFER || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) getCompressedImage(ctx, t, target, level, bufSize, img);
}

API void APIENTRY glGetCompressedTextureImage(GLuint texture, GLint level, GLsizei bufSize, void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t && dsaImageQueryOk(ctx, t, level)) getCompressedImage(ctx, t, t->target, level, bufSize, pixels);
}

API void APIENTRY glGetCompressedTextureSubImage(GLuint texture, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                                                 GLsizei bufSize, void* pixels) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (!t || !dsaImageQueryOk(ctx, t, level)) return;
    if (x < 0 || y < 0 || z < 0 || w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
    getCompressedImage(ctx, t, t->target, level, bufSize, pixels, x, y, z, w, h, d);
}

// --- copies from the framebuffer ---
id<MTLTexture> readAttachmentTexture(Context* ctx, Framebuffer* fb, const FormatInfo** f, int* level, int* slice);  // api_fbo.mm
id<MTLTexture> readDepthStencilTexture(Context* ctx, Framebuffer* fb, bool stencil, const FormatInfo** f, int* level, int* slice);

static void copyTexSubImage(Context* ctx, Texture* t, GLenum target, GLint level, int dx, int dy, int dz, int x, int y, int w, int h) {
    if (w <= 0 || h <= 0) return;
    Framebuffer* fb = ctx->readFb.get();
    if (!ensureTextureStorage(ctx, t)) return;
    // Depth/stencil destinations copy from the read framebuffer's depth/stencil buffer.
    const FormatInfo* tf = formatInfoForMTL(t->mtl.pixelFormat);
    bool depthDst = tf && isDepthOrStencil(tf);
    const FormatInfo* sf = nullptr;
    int slevel, sslice;
    id<MTLTexture> src = depthDst ? readDepthStencilTexture(ctx, fb, tf->kind == FmtKind::Stencil, &sf, &slevel, &sslice)
                                  : readAttachmentTexture(ctx, fb, &sf, &slevel, &sslice);
    if (!src) return ctx->setError(GL_INVALID_OPERATION, "no read buffer");
    int face = faceIndex(target);
    int dstSlice = face;
    if (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_CUBE_MAP_ARRAY) dstSlice = dz;
    if (t->target == GL_TEXTURE_1D_ARRAY) {
        dstSlice = dy;
        dy = 0;
    }
    // Clip source rect to the read buffer.
    int sw = std::max(1, (int)src.width >> slevel), sh = std::max(1, (int)src.height >> slevel);
    int x0 = std::max(0, x), y0 = std::max(0, y), x1 = std::min(sw, x + w), y1 = std::min(sh, y + h);
    if (x1 <= x0 || y1 <= y0) return;
    dx += x0 - x;
    dy += y0 - y;
    w = x1 - x0;
    h = y1 - y0;
    ctx->applyClearNow();
    if (src.pixelFormat == t->mtl.pixelFormat && src.sampleCount == 1 && t->target != GL_TEXTURE_3D) {
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
        [e copyFromTexture:src sourceSlice:sslice sourceLevel:slevel sourceOrigin:MTLOriginMake(x0, y0, 0) sourceSize:MTLSizeMake(w, h, 1)
                 toTexture:t->mtl destinationSlice:dstSlice destinationLevel:level destinationOrigin:MTLOriginMake(dx, dy, 0)];
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        markTextureWrite(ctx, t);
        return;
    }
    BlitParams bp;
    bp.src = src;
    bp.srcLevel = slevel;
    bp.srcSlice = sslice;
    bp.srcRect[0] = x0;
    bp.srcRect[1] = y0;
    bp.srcRect[2] = x0 + w;
    bp.srcRect[3] = y0 + h;
    bp.dstRect[0] = dx;
    bp.dstRect[1] = dy;
    bp.dstRect[2] = dx + w;
    bp.dstRect[3] = dy + h;
    bp.what = isDepthOrStencil(sf) ? BlitParams::Depth : BlitParams::Color;
    blitToTexture(ctx, bp, t->mtl, level, t->target == GL_TEXTURE_3D ? dz : dstSlice);
    markTextureWrite(ctx, t);
}

bool mglFramebufferComplete(Context* ctx, Framebuffer* fb, GLenum* status);  // api_fbo.mm

static bool readFbMultisampled(Framebuffer* fb) {
    if (fb->isDefault) return fb->surface && fb->surface->samples > 1;
    for (auto& a : fb->att)
        if ((a.type == GL_RENDERBUFFER && a.rb && a.rb->samples > 1) || (a.type == GL_TEXTURE && a.tex && a.tex->samples > 1)) return true;
    return false;
}

// Shared validation for Copy{Tex,Texture}SubImage{1,2,3}D. `t` is the texture
// (already resolved); `target` is the GL target (cube face for TexSubImage2D);
// `dsa` selects DSA error semantics (wrong texture target -> INVALID_OPERATION).
static void copySubImageChecked(Context* ctx, Texture* t, GLenum target, int dims, bool dsa, GLint level, int xo, int yo, int zo, int x,
                                int y, int w, int h) {
    GLenum tt = t->target;
    bool ok;
    switch (dims) {
    case 1: ok = tt == GL_TEXTURE_1D; break;
    case 2: ok = tt == GL_TEXTURE_2D || tt == GL_TEXTURE_1D_ARRAY || tt == GL_TEXTURE_RECTANGLE || (!dsa && tt == GL_TEXTURE_CUBE_MAP); break;
    default: ok = tt == GL_TEXTURE_3D || tt == GL_TEXTURE_2D_ARRAY || tt == GL_TEXTURE_CUBE_MAP_ARRAY || (dsa && tt == GL_TEXTURE_CUBE_MAP); break;
    }
    if (!ok) return ctx->setError(dsa ? GL_INVALID_OPERATION : GL_INVALID_ENUM, "copy: bad texture target");
    if (level < 0 || level >= Texture::kMaxLevels || (tt == GL_TEXTURE_RECTANGLE && level != 0)) return ctx->setError(GL_INVALID_VALUE);
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    Framebuffer* fb = ctx->readFb.get();
    GLenum status;
    if (!fb || !mglFramebufferComplete(ctx, fb, &status)) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    if (readFbMultisampled(fb)) return ctx->setError(GL_INVALID_OPERATION, "copy: multisampled read framebuffer");
    if (dsa && tt == GL_TEXTURE_CUBE_MAP) {
        if (zo < 0 || zo > 5) return ctx->setError(GL_INVALID_VALUE);
        target = GL_TEXTURE_CUBE_MAP_POSITIVE_X + zo;
        zo = 0;
    }
    if (!dsa && tt == GL_TEXTURE_CUBE_MAP && !isCubeFace(target)) return ctx->setError(GL_INVALID_ENUM);
    const TexImageLevel& L = t->levels[faceIndex(target)][level];
    if (!L.defined()) return ctx->setError(GL_INVALID_OPERATION, "copy: level not defined");
    int lw = L.width, lh = L.height, ld = L.depth;
    if (xo < 0 || xo + w > lw) return ctx->setError(GL_INVALID_VALUE);
    if (dims >= 2 && (yo < 0 || yo + h > lh)) return ctx->setError(GL_INVALID_VALUE);
    if (dims == 3 && (zo < 0 || zo >= ld)) return ctx->setError(GL_INVALID_VALUE);
    if (fb->readBuffer == GL_NONE) return ctx->setError(GL_INVALID_OPERATION, "copy: read buffer is NONE");
    copyTexSubImage(ctx, t, target, level, xo, yo, zo, x, y, w, h);
}

static void copyTexImage(Context* ctx, GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei w, GLsizei h,
                         GLint border, int dims) {
    if (!validTexTarget(target, dims) || isProxy(target)) return ctx->setError(GL_INVALID_ENUM);
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    if (border != 0 || w < 0 || h < 0 || w > 16384 || h > 16384) return ctx->setError(GL_INVALID_VALUE);
    if (target == GL_TEXTURE_RECTANGLE && level != 0) return ctx->setError(GL_INVALID_VALUE);
    if (isCubeFace(target) && w != h) return ctx->setError(GL_INVALID_VALUE);
    Framebuffer* fb = ctx->readFb.get();
    GLenum status;
    if (!fb || !mglFramebufferComplete(ctx, fb, &status)) return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION);
    if (readFbMultisampled(fb)) return ctx->setError(GL_INVALID_OPERATION, "copy: multisampled read framebuffer");
    bool depthReq = internalformat == GL_DEPTH_COMPONENT || internalformat == GL_DEPTH_COMPONENT16 || internalformat == GL_DEPTH_COMPONENT24 ||
                    internalformat == GL_DEPTH_COMPONENT32 || internalformat == GL_DEPTH_COMPONENT32F;
    bool dsReq = internalformat == GL_DEPTH_STENCIL || internalformat == GL_DEPTH24_STENCIL8 || internalformat == GL_DEPTH32F_STENCIL8;
    bool stencilReq = internalformat == GL_STENCIL_INDEX || internalformat == GL_STENCIL_INDEX8;
    const FormatInfo* rf = nullptr;
    int rl, rs;
    const FormatInfo* f = nullptr;
    if (dsReq && !fb->isDefault && (fb->att[kAttDepth].type == GL_NONE || fb->att[kAttStencil].type == GL_NONE))
        return ctx->setError(GL_INVALID_OPERATION, "depth/stencil copy without depth and stencil buffers");
    if (depthReq || dsReq || stencilReq) {
        if (!readDepthStencilTexture(ctx, fb, stencilReq, &rf, &rl, &rs)) return ctx->setError(GL_INVALID_OPERATION, "copy: no depth/stencil buffer");
        if (internalformat == GL_DEPTH_COMPONENT) f = formatInfo(rf->kind == FmtKind::DepthStencil ? GL_DEPTH_COMPONENT24 : rf->internalFormat);
        else if (internalformat == GL_DEPTH_STENCIL) f = formatInfo(GL_DEPTH24_STENCIL8);
        else if (internalformat == GL_STENCIL_INDEX) f = formatInfo(GL_STENCIL_INDEX8);
        else f = formatInfo(internalformat);
        if (f && !isDepthOrStencil(f)) f = nullptr;
    } else {
        if (fb->readBuffer == GL_NONE || !readAttachmentTexture(ctx, fb, &rf, &rl, &rs))
            return ctx->setError(GL_INVALID_OPERATION, "copy: no read buffer");
        bool srcInt = rf->kind == FmtKind::UInt || rf->kind == FmtKind::SInt;
        GLenum fmt = srcInt ? GL_RGBA_INTEGER : GL_RGBA;
        GLenum type = rf->kind == FmtKind::Float ? GL_FLOAT : rf->kind == FmtKind::SInt ? GL_INT : rf->kind == FmtKind::UInt ? GL_UNSIGNED_INT
                                                                                                                         : GL_UNSIGNED_BYTE;
        f = resolveInternalFormat(internalformat, fmt, type);
        if (f && (isDepthOrStencil(f) || f->comp == Comp::Compressed)) return ctx->setError(GL_INVALID_OPERATION);
        if (f) {
            bool dstInt = f->kind == FmtKind::UInt || f->kind == FmtKind::SInt;
            if (dstInt != srcInt) return ctx->setError(GL_INVALID_OPERATION, "copy: integer/non-integer mismatch");
            if (dstInt && (f->kind == FmtKind::SInt) != (rf->kind == FmtKind::SInt)) return ctx->setError(GL_INVALID_OPERATION);
        }
    }
    if (!f) return ctx->setError(GL_INVALID_VALUE, "copy: unsupported internal format 0x%x", internalformat);
    Texture* t = texForTarget(ctx, target);
    if (!t) return;
    if (t->immutable) return ctx->setError(GL_INVALID_OPERATION, "texture is immutable");
    defineLevel(t, faceIndex(target), level, w, dims == 2 ? h : 1, 1, internalformat, f);
    if (w > 0 && h > 0) copyTexSubImage(ctx, t, target, level, 0, 0, 0, x, y, w, dims == 2 ? h : 1);
}

API void APIENTRY glCopyTexImage2D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLsizei height,
                                   GLint border) {
    MGL_CTX();
    copyTexImage(ctx, target, level, internalformat, x, y, width, height, border, 2);
}

API void APIENTRY glCopyTexImage1D(GLenum target, GLint level, GLenum internalformat, GLint x, GLint y, GLsizei width, GLint border) {
    MGL_CTX();
    copyTexImage(ctx, target, level, internalformat, x, y, width, 1, border, 1);
}

API void APIENTRY glCopyTexSubImage1D(GLenum target, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width) {
    MGL_CTX();
    if (target != GL_TEXTURE_1D) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) copySubImageChecked(ctx, t, target, 1, false, level, xoffset, 0, 0, x, y, width, 1);
}

API void APIENTRY glCopyTexSubImage2D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width,
                                      GLsizei height) {
    MGL_CTX();
    if (!(target == GL_TEXTURE_2D || target == GL_TEXTURE_1D_ARRAY || target == GL_TEXTURE_RECTANGLE || isCubeFace(target)))
        return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) copySubImageChecked(ctx, t, target, 2, false, level, xoffset, yoffset, 0, x, y, width, height);
}

API void APIENTRY glCopyTexSubImage3D(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y,
                                      GLsizei width, GLsizei height) {
    MGL_CTX();
    if (!(target == GL_TEXTURE_3D || target == GL_TEXTURE_2D_ARRAY || target == GL_TEXTURE_CUBE_MAP_ARRAY)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = texForTarget(ctx, target);
    if (t) copySubImageChecked(ctx, t, target, 3, false, level, xoffset, yoffset, zoffset, x, y, width, height);
}

API void APIENTRY glCopyTextureSubImage1D(GLuint texture, GLint level, GLint xoffset, GLint x, GLint y, GLsizei width) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) copySubImageChecked(ctx, t, t->target, 1, true, level, xoffset, 0, 0, x, y, width, 1);
}

API void APIENTRY glCopyTextureSubImage2D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint x, GLint y, GLsizei width,
                                          GLsizei height) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) copySubImageChecked(ctx, t, t->target, 2, true, level, xoffset, yoffset, 0, x, y, width, height);
}

API void APIENTRY glCopyTextureSubImage3D(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLint x, GLint y,
                                          GLsizei width, GLsizei height) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) copySubImageChecked(ctx, t, t->target, 3, true, level, xoffset, yoffset, zoffset, x, y, width, height);
}

// --- copy image ---
namespace {
struct CopyImageRef {
    const FormatInfo* f = nullptr;
    Texture* tex = nullptr;
    Renderbuffer* rb = nullptr;
    int w = 0, h = 0, d = 0;  // level dimensions in GL terms (d = depth / layers / faces)
    int samples = 0;
};
}  // namespace

static bool copyImageTargetValid(GLenum t) {
    switch (t) {
    case GL_RENDERBUFFER: case GL_TEXTURE_1D: case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D: case GL_TEXTURE_2D_ARRAY:
    case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_3D: case GL_TEXTURE_CUBE_MAP:
    case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_RECTANGLE: return true;
    default: return false;
    }
}

// Resolves one side of glCopyImageSubData (§18.3.3), returning the GL error to raise.
static GLenum copyImageResolve(Context* ctx, GLuint name, GLenum target, GLint level, CopyImageRef& r) {
    if (target == GL_RENDERBUFFER) {
        Renderbuffer* rb = name ? ctx->share->renderbuffers.get(name) : nullptr;
        if (!rb || !rb->fmt) return GL_INVALID_VALUE;
        if (level != 0) return GL_INVALID_VALUE;
        r.rb = rb;
        r.f = rb->fmt;
        r.w = rb->width;
        r.h = rb->height;
        r.d = 1;
        r.samples = rb->samples;
        return GL_NO_ERROR;
    }
    Texture* t = name ? ctx->share->textures.get(name) : nullptr;
    if (!t || !t->target) return GL_INVALID_VALUE;
    if (t->target != target) return GL_INVALID_ENUM;
    if (level < 0 || level >= Texture::kMaxLevels) return GL_INVALID_VALUE;
    if (t->immutable && level >= t->immutableLevels) return GL_INVALID_VALUE;
    const TexImageLevel& L = t->levels[0][level];
    if (!L.defined()) return GL_INVALID_VALUE;
    int first, last;
    if (!textureComplete(t, t->sampler, &first, &last)) return GL_INVALID_OPERATION;
    r.tex = t;
    r.f = L.fmt;
    r.w = L.width;
    r.h = target == GL_TEXTURE_1D || target == GL_TEXTURE_1D_ARRAY ? 1 : L.height;
    // §18.3.3: slices of array textures (1D arrays included) and cube faces are addressed by z.
    switch (target) {
    case GL_TEXTURE_1D_ARRAY: r.d = L.height; break;
    case GL_TEXTURE_3D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
        r.d = L.depth;
        break;
    case GL_TEXTURE_CUBE_MAP: r.d = 6; break;
    default: r.d = 1; break;
    }
    r.samples = t->samples;
    return GL_NO_ERROR;
}

// Region checks for one side; w/h/d are in this image's texels.
static bool copyImageRegionValid(const CopyImageRef& r, int x, int y, int z, int w, int h, int d) {
    if (x < 0 || y < 0 || z < 0) return false;
    if ((int64_t)x + w > r.w || (int64_t)y + h > r.h || (int64_t)z + d > r.d) return false;
    if (r.f->comp == Comp::Compressed) {
        int bw = r.f->blockW, bh = r.f->blockH;
        if (x % bw || y % bh) return false;
        if (w % bw && x + w != r.w) return false;
        if (h % bh && y + h != r.h) return false;
    }
    return true;
}

// GL (x, y, z) of an image -> Metal slice + origin.
static void copyImageLoc(GLenum target, int x, int y, int z, NSUInteger& slice, MTLOrigin& o) {
    switch (target) {
    case GL_TEXTURE_3D: slice = 0; o = MTLOriginMake(x, y, z); break;
    case GL_TEXTURE_1D_ARRAY: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_CUBE_MAP:
    case GL_TEXTURE_CUBE_MAP_ARRAY:
        slice = z; o = MTLOriginMake(x, y, 0); break;
    default: slice = 0; o = MTLOriginMake(x, y, 0); break;
    }
}

// Reallocates a texture's storage so it can be reinterpreted with format views,
// keeping its contents.
static bool ensureFormatViewable(Context* ctx, Texture* t) {
    if (t->viewParent) t = t->viewParent.get();
    if (t->mtl && (t->mtl.usage & MTLTextureUsagePixelFormatView)) return true;
    t->needsFormatView = true;
    id<MTLTexture> old = t->mtl;
    t->mtl = nil;
    t->storageFmt = nullptr;
    t->storageDirty = true;
    if (!ensureTextureStorage(ctx, t)) return false;
    if (old) {
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
        [e copyFromTexture:old toTexture:t->mtl];
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
        ctx->dev->releaseAfter(old, ctx->serial());
    }
    return true;
}

API void APIENTRY glCopyImageSubData(GLuint srcName, GLenum srcTarget, GLint srcLevel, GLint srcX, GLint srcY, GLint srcZ, GLuint dstName,
                                     GLenum dstTarget, GLint dstLevel, GLint dstX, GLint dstY, GLint dstZ, GLsizei w, GLsizei h,
                                     GLsizei d) {
    MGL_CTX();
    if (!copyImageTargetValid(srcTarget) || !copyImageTargetValid(dstTarget)) return ctx->setError(GL_INVALID_ENUM);
    CopyImageRef s, dr;
    if (GLenum e = copyImageResolve(ctx, srcName, srcTarget, srcLevel, s)) return ctx->setError(e);
    if (GLenum e = copyImageResolve(ctx, dstName, dstTarget, dstLevel, dr)) return ctx->setError(e);
    if (w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
    if (s.samples != dr.samples) return ctx->setError(GL_INVALID_OPERATION);
    if (!copyImageCompatible(s.f, dr.f)) return ctx->setError(GL_INVALID_OPERATION);
    // Destination extent: compressed <-> uncompressed copies map one block to one texel.
    int sbw = s.f->blockW, sbh = s.f->blockH, dbw = dr.f->blockW, dbh = dr.f->blockH;
    int dw = w, dh = h;
    if (sbw != dbw) dw = (w + sbw - 1) / sbw * dbw;
    if (sbh != dbh) dh = (h + sbh - 1) / sbh * dbh;
    if (!copyImageRegionValid(s, srcX, srcY, srcZ, w, h, d) || !copyImageRegionValid(dr, dstX, dstY, dstZ, dw, dh, d))
        return ctx->setError(GL_INVALID_VALUE);
    if (w == 0 || h == 0 || d == 0) return;

    if (s.tex && !ensureTextureStorage(ctx, s.tex)) return;
    if (dr.tex && !ensureTextureStorage(ctx, dr.tex)) return;
    id<MTLTexture> src = s.tex ? s.tex->mtl : s.rb->mtl;
    id<MTLTexture> dst = dr.tex ? dr.tex->mtl : dr.rb->mtl;
    if (!src || !dst) return;
    if (src.pixelFormat != dst.pixelFormat && src.sampleCount > 1) {
        // Multisample images can't go through a buffer: reinterpret the source with a view.
        if (!s.tex || !ensureFormatViewable(ctx, s.tex)) {
            MGL_WARN("glCopyImageSubData: multisample format reinterpretation unsupported");
            return;
        }
        src = [s.tex->mtl newTextureViewWithPixelFormat:dst.pixelFormat];
        ctx->dev->retainUntil(src, ctx->serial());
    }
    bool direct = src.pixelFormat == dst.pixelFormat;
    if (!direct && s.f->bytesPerPixel != dr.f->bytesPerPixel) {
        MGL_WARN("glCopyImageSubData: storage size mismatch (%d vs %d)", s.f->bytesPerPixel, dr.f->bytesPerPixel);
        return;
    }

    // Staging layout for the reinterpreting path: raw bytes, one source block per destination texel.
    size_t bpr = (size_t)((w + sbw - 1) / sbw) * s.f->bytesPerPixel;
    size_t img = bpr * ((h + sbh - 1) / sbh);
    UploadAlloc stage{};
    if (!direct) stage = ctx->upload(img * d, 256);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    for (int pass = 0; pass < (direct ? 1 : 2); pass++) {
        for (int k = 0; k < d; k++) {
            NSUInteger ss, ds;
            MTLOrigin so, dO;
            copyImageLoc(srcTarget, srcX, srcY, srcZ + k, ss, so);
            copyImageLoc(dstTarget, dstX, dstY, dstZ + k, ds, dO);
            size_t off = stage.offset + img * k;
            if (direct) {
                [e copyFromTexture:src sourceSlice:ss sourceLevel:srcLevel sourceOrigin:so sourceSize:MTLSizeMake(w, h, 1)
                         toTexture:dst destinationSlice:ds destinationLevel:dstLevel destinationOrigin:dO];
            } else if (pass == 0) {
                [e copyFromTexture:src sourceSlice:ss sourceLevel:srcLevel sourceOrigin:so sourceSize:MTLSizeMake(w, h, 1)
                          toBuffer:stage.buffer destinationOffset:off destinationBytesPerRow:bpr destinationBytesPerImage:img];
            } else {
                [e copyFromBuffer:stage.buffer sourceOffset:off sourceBytesPerRow:bpr sourceBytesPerImage:img sourceSize:MTLSizeMake(dw, dh, 1)
                        toTexture:dst destinationSlice:ds destinationLevel:dstLevel destinationOrigin:dO];
            }
        }
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    }
    if (s.tex) s.tex->lastUse = ctx->serial();
    if (s.rb) s.rb->lastUse = ctx->serial();
    if (dr.tex) markTextureWrite(ctx, dr.tex);
    if (dr.rb) dr.rb->lastUse = ctx->serial();
}

// --- texture views ---
// Table 8.20: view targets compatible with an original texture's target.
static bool viewTargetCompatible(GLenum orig, GLenum view) {
    switch (orig) {
    case GL_TEXTURE_1D: case GL_TEXTURE_1D_ARRAY: return view == GL_TEXTURE_1D || view == GL_TEXTURE_1D_ARRAY;
    case GL_TEXTURE_2D: return view == GL_TEXTURE_2D || view == GL_TEXTURE_2D_ARRAY;
    case GL_TEXTURE_3D: return view == GL_TEXTURE_3D;
    case GL_TEXTURE_RECTANGLE: return view == GL_TEXTURE_RECTANGLE;
    case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY:
        return view == GL_TEXTURE_CUBE_MAP || view == GL_TEXTURE_2D || view == GL_TEXTURE_2D_ARRAY || view == GL_TEXTURE_CUBE_MAP_ARRAY;
    case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        return view == GL_TEXTURE_2D_MULTISAMPLE || view == GL_TEXTURE_2D_MULTISAMPLE_ARRAY;
    default: return false;
    }
}

// Layers of an immutable texture (or view) in glTextureView terms: array layers,
// cube faces (6 per cube), 1 otherwise.
static int viewLayersOf(const Texture* t) {
    if (t->viewParent) return t->viewNumLayers;
    const TexImageLevel& L = t->levels[0][0];
    switch (t->target) {
    case GL_TEXTURE_1D_ARRAY: return L.height;
    case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY: return L.depth;
    case GL_TEXTURE_CUBE_MAP: return 6;
    default: return 1;
    }
}

API void APIENTRY glTextureView(GLuint texture, GLenum target, GLuint origtexture, GLenum internalformat, GLuint minlevel, GLuint numlevels,
                                GLuint minlayer, GLuint numlayers) {
    MGL_CTX();
    // §8.18 errors, in the order the spec lists them.
    if (texture == 0) return ctx->setError(GL_INVALID_VALUE);
    Texture* t = ctx->share->textures.get(texture);
    if (!t && !ctx->share->textures.isReservedOnly(texture)) return ctx->setError(GL_INVALID_OPERATION, "texture is not a generated name");
    if (t && (t->target || t->immutable)) return ctx->setError(GL_INVALID_OPERATION, "texture already has a target");
    Texture* orig = origtexture ? ctx->share->textures.get(origtexture) : nullptr;
    if (!orig || !orig->target) return ctx->setError(GL_INVALID_VALUE);
    if (!orig->immutable) return ctx->setError(GL_INVALID_OPERATION, "origtexture is mutable");
    if (!viewTargetCompatible(orig->target, target)) return ctx->setError(GL_INVALID_OPERATION, "incompatible view target");
    const FormatInfo* f = formatInfo(internalformat);
    const FormatInfo* of = orig->levels[0][0].fmt;
    if (!f || !of) return ctx->setError(GL_INVALID_OPERATION);
    if (f->internalFormat != of->internalFormat) {
        int vc = viewClass(of->internalFormat);
        if (!vc || vc != viewClass(f->internalFormat)) return ctx->setError(GL_INVALID_OPERATION, "incompatible view format");
    }
    int origLevels = orig->viewParent ? orig->viewNumLevels : orig->immutableLevels;
    int origLayers = viewLayersOf(orig);
    if ((int)minlevel >= origLevels || (int)minlayer >= origLayers) return ctx->setError(GL_INVALID_VALUE);
    switch (target) {
    case GL_TEXTURE_CUBE_MAP: if (numlayers != 6) return ctx->setError(GL_INVALID_VALUE); break;
    case GL_TEXTURE_CUBE_MAP_ARRAY: if (numlayers % 6) return ctx->setError(GL_INVALID_VALUE); break;
    case GL_TEXTURE_1D: case GL_TEXTURE_2D: case GL_TEXTURE_3D: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_2D_MULTISAMPLE:
        if (numlayers != 1) return ctx->setError(GL_INVALID_VALUE);
        break;
    default: break;
    }
    const TexImageLevel& B = orig->levels[0][minlevel];
    if ((target == GL_TEXTURE_CUBE_MAP || target == GL_TEXTURE_CUBE_MAP_ARRAY) && B.width != B.height)
        return ctx->setError(GL_INVALID_OPERATION, "cube view of non-square image");
    int nLevels = std::min<int>(numlevels, origLevels - minlevel);
    int nLayers = std::min<int>(numlayers, origLayers - minlayer);
    if (nLevels < 1 || nLayers < 1) return ctx->setError(GL_INVALID_VALUE);
    if (!ensureTextureStorage(ctx, orig)) return ctx->setError(GL_INVALID_OPERATION);
    // Level/layer specs relative to the root texture.
    Texture* root = orig;
    int rootLevel = minlevel, rootLayer = minlayer;
    if (orig->viewParent) {
        root = orig->viewParent.get();
        rootLevel += orig->viewMinLevel;
        rootLayer += orig->viewMinLayer;
    }
    // Reinterpreting the format or the cube-ness of the storage needs a view-capable texture (Apple GPUs
    // lay out compressed cube and array storage differently).
    MTLTextureType vt = mtlTextureType(target, root->samples);
    bool cubeView = vt == MTLTextureTypeCube || vt == MTLTextureTypeCubeArray;
    bool cubeRoot = root->mtl.textureType == MTLTextureTypeCube || root->mtl.textureType == MTLTextureTypeCubeArray;
    if ((f->mtl != root->storageFmt->mtl || (cubeView && vt != root->mtl.textureType) || (cubeRoot && vt != root->mtl.textureType)) &&
        !ensureFormatViewable(ctx, root))
        return ctx->setError(GL_OUT_OF_MEMORY);
    if (!t) t = ctx->share->textures.create(texture, newTexture(target));
    t->target = target;
    t->viewParent = root;
    t->viewMinLevel = rootLevel;
    t->viewNumLevels = nLevels;
    t->viewMinLayer = rootLayer;
    t->viewNumLayers = nLayers;
    t->immutable = true;
    t->immutableLevels = nLevels;
    t->viewImmutableLevels = orig->viewParent ? orig->viewImmutableLevels : orig->immutableLevels;
    int faces = target == GL_TEXTURE_CUBE_MAP ? 6 : 1;
    for (int l = 0; l < Texture::kMaxLevels; l++)
        for (int face = 0; face < 6; face++) t->levels[face][l] = TexImageLevel();
    for (int l = 0; l < nLevels && l < Texture::kMaxLevels; l++) {
        TexImageLevel L = orig->levels[0][minlevel + l];
        L.fmt = f;
        L.internalFormat = internalformat;
        switch (target) {
        case GL_TEXTURE_1D: L.height = 1; L.depth = 1; break;
        case GL_TEXTURE_1D_ARRAY: L.height = nLayers; L.depth = 1; break;
        case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY: case GL_TEXTURE_CUBE_MAP_ARRAY: L.depth = nLayers; break;
        case GL_TEXTURE_3D: break;
        default: L.depth = 1; break;
        }
        for (int face = 0; face < faces; face++) t->levels[face][l] = L;
    }
    t->mtl = [root->mtl newTextureViewWithPixelFormat:f->mtl textureType:vt levels:NSMakeRange(rootLevel, nLevels)
                                               slices:NSMakeRange(rootLayer, vt == MTLTextureType3D ? 1 : nLayers)];
    t->storageFmt = f;
    t->storageLevels = nLevels;
    t->storageW = std::max(1, root->storageW >> rootLevel);
    t->storageH = std::max(1, root->storageH >> rootLevel);
    t->storageD = target == GL_TEXTURE_3D ? std::max(1, root->storageD >> rootLevel) : 1;
    // Same convention as allocated storage: cube maps count cubes, not faces.
    t->storageLayers = target == GL_TEXTURE_CUBE_MAP ? 1 : target == GL_TEXTURE_CUBE_MAP_ARRAY ? nLayers / 6 : nLayers;
    t->storageDirty = false;
    t->samples = root->samples;
    t->sampler = SamplerState();
    if (target == GL_TEXTURE_RECTANGLE) t->sampler.minFilter = GL_LINEAR, t->sampler.wrapS = t->sampler.wrapT = t->sampler.wrapR = GL_CLAMP_TO_EDGE;
}

// --- clear texture ---
static void clearTexImage(Context* ctx, Texture* t, GLint level, int x, int y, int z, int w, int h, int d, GLenum format, GLenum type,
                          const void* data, bool whole) {
    if (t->target == GL_TEXTURE_BUFFER) return ctx->setError(GL_INVALID_OPERATION);
    if (level < 0 || level >= Texture::kMaxLevels) return ctx->setError(GL_INVALID_VALUE);
    const TexImageLevel& L = t->levels[0][level];
    if (!L.defined()) return ctx->setError(GL_INVALID_OPERATION);
    if (GLenum e = pixelFormatTypeError(format, type)) return ctx->setError(e);
    const FormatInfo* lf = L.fmt;
    if (lf->comp == Comp::Compressed) return ctx->setError(GL_INVALID_OPERATION, "clearing a compressed texture");
    {
        // §8.21: the client format must match the internal format's kind.
        bool bad;
        switch (lf->baseFormat) {
        case GL_DEPTH_COMPONENT: bad = format != GL_DEPTH_COMPONENT; break;
        case GL_DEPTH_STENCIL: bad = format != GL_DEPTH_STENCIL; break;
        case GL_STENCIL_INDEX: bad = format != GL_STENCIL_INDEX; break;
        default:
            bad = format == GL_DEPTH_COMPONENT || format == GL_DEPTH_STENCIL || format == GL_STENCIL_INDEX ||
                  (lf->kind == FmtKind::UInt || lf->kind == FmtKind::SInt) != isIntegerFormat(format);
            break;
        }
        if (bad) return ctx->setError(GL_INVALID_OPERATION, "format 0x%x incompatible with internal format 0x%x", format, lf->internalFormat);
    }
    // Level extent in GL terms: 1D arrays keep layers in y, cube maps have 6 faces in z.
    int W = L.width, H = L.height, D = L.depth;
    switch (t->target) {
    case GL_TEXTURE_1D: H = 1; D = 1; break;
    case GL_TEXTURE_1D_ARRAY: D = 1; break;
    case GL_TEXTURE_CUBE_MAP: D = 6; break;
    case GL_TEXTURE_2D: case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_2D_MULTISAMPLE: D = 1; break;
    default: break;
    }
    if (whole) {
        x = y = z = 0;
        w = W;
        h = H;
        d = D;
    } else {
        if (w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
        if (x < 0 || y < 0 || z < 0 || (int64_t)x + w > W || (int64_t)y + h > H || (int64_t)z + d > D)
            return ctx->setError(GL_INVALID_OPERATION, "clear region outside the image");
    }
    if (w == 0 || h == 0 || d == 0) return;
    if (!ensureTextureStorage(ctx, t)) return;
    const FormatInfo* f = t->storageFmt;
    // One texel of the fill value in storage layout (zeros when data is NULL).
    uint8_t px[16] = {};
    if (data) convertToStorage(f, format, type, data, 0, 0, px, 0, 0, 1, 1, 1, false);
    bool ds = f->comp == Comp::D32F_S8;
    // Depth/stencil storage is filled one aspect at a time.
    int planes = ds ? 2 : 1;
    bool is3D = t->target == GL_TEXTURE_3D, is1DArray = t->target == GL_TEXTURE_1D_ARRAY;
    int rows = is1DArray ? 1 : h, slices = is1DArray ? h : d;
    for (int p = 0; p < planes; p++) {
        int bpp = ds ? (p == 0 ? 4 : 1) : f->bytesPerPixel;
        const uint8_t* val = ds && p == 1 ? px + 4 : px;
        size_t rowBytes = (size_t)w * bpp, imgBytes = rowBytes * rows;
        UploadAlloc up = ctx->upload(imgBytes, 16);
        for (size_t i = 0; i < (size_t)w * rows; i++) memcpy(up.cpu + i * bpp, val, bpp);
        MTLBlitOption opt = !ds ? MTLBlitOptionNone : p == 0 ? MTLBlitOptionDepthFromDepthStencil : MTLBlitOptionStencilFromDepthStencil;
        id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
        for (int i = 0; i < slices; i++) {
            NSUInteger slice = is3D ? 0 : is1DArray ? y + i : z + i;
            MTLOrigin o = MTLOriginMake(x, is1DArray ? 0 : y, is3D ? z + i : 0);
            [e copyFromBuffer:up.buffer sourceOffset:up.offset sourceBytesPerRow:rowBytes sourceBytesPerImage:imgBytes
                   sourceSize:MTLSizeMake(w, rows, 1) toTexture:t->mtl destinationSlice:slice destinationLevel:level
            destinationOrigin:o options:opt];
        }
        [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    }
    markTextureWrite(ctx, t);
}

API void APIENTRY glClearTexImage(GLuint texture, GLint level, GLenum format, GLenum type, const void* data) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) clearTexImage(ctx, t, level, 0, 0, 0, 0, 0, 0, format, type, data, true);
}

API void APIENTRY glClearTexSubImage(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                     GLsizei height, GLsizei depth, GLenum format, GLenum type, const void* data) {
    MGL_CTX();
    Texture* t = namedTex(ctx, texture);
    if (t) clearTexImage(ctx, t, level, xoffset, yoffset, zoffset, width, height, depth, format, type, data, false);
}

API void APIENTRY glInvalidateTexImage(GLuint texture, GLint level) {
    MGL_CTX();
    if (!ctx->share->textures.get(texture)) return ctx->setError(GL_INVALID_VALUE);
}
API void APIENTRY glInvalidateTexSubImage(GLuint texture, GLint level, GLint, GLint, GLint, GLsizei, GLsizei, GLsizei) {
    MGL_CTX();
    if (!ctx->share->textures.get(texture)) return ctx->setError(GL_INVALID_VALUE);
}

// --- image units ---
// GL 4.6 table 8.33: internal formats usable with image units.
static bool isImageUnitFormat(GLenum f) {
    switch (f) {
    case GL_RGBA32F: case GL_RGBA16F: case GL_RG32F: case GL_RG16F: case GL_R11F_G11F_B10F: case GL_R32F: case GL_R16F:
    case GL_RGBA32UI: case GL_RGBA16UI: case GL_RGB10_A2UI: case GL_RGBA8UI: case GL_RG32UI: case GL_RG16UI: case GL_RG8UI:
    case GL_R32UI: case GL_R16UI: case GL_R8UI:
    case GL_RGBA32I: case GL_RGBA16I: case GL_RGBA8I: case GL_RG32I: case GL_RG16I: case GL_RG8I: case GL_R32I: case GL_R16I: case GL_R8I:
    case GL_RGBA16: case GL_RGB10_A2: case GL_RGBA8: case GL_RG16: case GL_RG8: case GL_R16: case GL_R8:
    case GL_RGBA16_SNORM: case GL_RGBA8_SNORM: case GL_RG16_SNORM: case GL_RG8_SNORM: case GL_R16_SNORM: case GL_R8_SNORM:
        return true;
    default: return false;
    }
}

API void APIENTRY glBindImageTexture(GLuint unit, GLuint texture, GLint level, GLboolean layered, GLint layer, GLenum access, GLenum format) {
    MGL_CTX();
    if (unit >= kMaxImageUnits || level < 0 || layer < 0) return ctx->setError(GL_INVALID_VALUE);
    if (access != GL_READ_ONLY && access != GL_WRITE_ONLY && access != GL_READ_WRITE) return ctx->setError(GL_INVALID_ENUM);
    if (!isImageUnitFormat(format) || !formatInfo(format)) return ctx->setError(GL_INVALID_VALUE);
    ImageUnit& iu = ctx->imageUnits[unit];
    if (!texture) {
        iu = ImageUnit();
        return;
    }
    Texture* t = ctx->share->textures.get(texture);
    if (!t) return ctx->setError(GL_INVALID_VALUE);
    iu.tex = t;
    // Usage requirements apply to the storage, i.e. the root of a texture view.
    if (t->viewParent) t = t->viewParent.get();
    // Accessing the texture through a different (size-compatible) format needs a format view.
    const FormatInfo* tf = t->storageFmt ? t->storageFmt : t->levels[0][std::clamp(t->baseLevel, 0, Texture::kMaxLevels - 1)].fmt;
    bool reinterpret = tf && t->target != GL_TEXTURE_BUFFER && formatInfo(format)->mtl != tf->mtl;
    bool needView = reinterpret && !t->needsFormatView;
    if (needView) t->needsFormatView = true;
    // Multisample images are accessed through a shadow (draw.mm, msImageShadow): the texture
    // itself only needs a format view.
    bool ms = t->samples > 0;
    if ((!t->needsShaderWrite && !ms) || needView) {
        t->needsShaderWrite |= !ms;
        bool missing = t->mtl && ((!ms && !(t->mtl.usage & MTLTextureUsageShaderWrite)) ||
                                  (needView && !(t->mtl.usage & MTLTextureUsagePixelFormatView)));
        if (missing && t->target != GL_TEXTURE_BUFFER && !t->sparse) {  // sparse storage has every usage it supports
            // Reallocate with write (and format view) usage, copying contents.
            id<MTLTexture> old = t->mtl;
            const FormatInfo* oldFmt = t->storageFmt;
            t->mtl = nil;
            t->storageFmt = nullptr;
            t->storageDirty = true;
            if (ensureTextureStorage(ctx, t) && oldFmt == t->storageFmt) {
                // A fresh encoder: AGX's compute-based texture copy can crash (null internal
                // pipeline) when issued in an encoder after some dispatches (seen after an image
                // store to a multisample texture).
                ctx->endEncoder();
                id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
                [e copyFromTexture:old toTexture:t->mtl];
                [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
            }
            ctx->dev->releaseAfter(old, ctx->serial());
        }
    }
    iu.level = level;
    iu.layered = layered;
    iu.layer = layer;
    iu.access = access;
    iu.format = format;
}

API void APIENTRY glBindImageTextures(GLuint first, GLsizei count, const GLuint* textures) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if ((uint64_t)first + (uint64_t)count > (uint64_t)kMaxImageUnits) return ctx->setError(GL_INVALID_OPERATION);
    for (int i = 0; i < count; i++) {
        GLuint name = textures ? textures[i] : 0;
        Texture* t = name ? ctx->share->textures.get(name) : nullptr;
        if (name && (!t || !t->target)) {
            ctx->setError(GL_INVALID_OPERATION, "BindImageTextures: %u is not a texture", name);
            continue;
        }
        if (t && !(t->target == GL_TEXTURE_BUFFER ? t->buffer != nullptr : t->levels[0][0].defined())) {
            ctx->setError(GL_INVALID_OPERATION, "BindImageTextures: texture %u has no level 0", name);
            continue;
        }
        GLenum fmt = t && t->levels[0][0].defined() ? t->levels[0][0].fmt->internalFormat : GL_R8;
        glBindImageTexture(first + i, name, 0, t && (t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_3D || t->isCube()), 0,
                           GL_READ_WRITE, fmt);
    }
}

// --- samplers ---
API void APIENTRY glGenSamplers(GLsizei n, GLuint* samplers) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        samplers[i] = ctx->share->samplers.genName();
        ctx->share->samplers.create(samplers[i], new Sampler());
    }
}
API void APIENTRY glCreateSamplers(GLsizei n, GLuint* samplers) { glGenSamplers(n, samplers); }

API void APIENTRY glDeleteSamplers(GLsizei n, const GLuint* samplers) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        Sampler* s = ctx->share->samplers.get(samplers[i]);
        if (!s) continue;
        for (auto& u : ctx->samplers)
            if (u.get() == s) u = nullptr;
        ctx->share->samplers.remove(samplers[i]);
    }
}

API GLboolean APIENTRY glIsSampler(GLuint sampler) {
    MGL_CTX_RET(GL_FALSE);
    return sampler && ctx->share->samplers.get(sampler) != nullptr;
}

API void APIENTRY glBindSampler(GLuint unit, GLuint sampler) {
    MGL_CTX();
    if (unit >= kMaxTextureUnits) return ctx->setError(GL_INVALID_VALUE);
    if (!sampler) {
        ctx->samplers[unit] = nullptr;
        return;
    }
    Sampler* s = ctx->share->samplers.get(sampler);
    if (!s) return ctx->setError(GL_INVALID_OPERATION);
    ctx->samplers[unit] = s;
}

API void APIENTRY glBindSamplers(GLuint first, GLsizei count, const GLuint* samplers) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if ((uint64_t)first + (uint64_t)count > (uint64_t)kMaxTextureUnits) return ctx->setError(GL_INVALID_OPERATION);
    for (int i = 0; i < count; i++) {
        GLuint s = samplers ? samplers[i] : 0;
        Sampler* so = s ? ctx->share->samplers.get(s) : nullptr;
        if (s && !so) {
            ctx->setError(GL_INVALID_OPERATION, "BindSamplers: %u is not a sampler", s);
            continue;
        }
        ctx->samplers[first + i] = so;
    }
}

static Sampler* samplerObj(Context* ctx, GLuint name) {
    Sampler* s = ctx->share->samplers.get(name);
    if (!s) ctx->setError(GL_INVALID_OPERATION);
    return s;
}

static void samplerParamv(Context* ctx, GLuint sampler, GLenum pname, const double* v, bool intBorder) {
    Sampler* s = samplerObj(ctx, sampler);
    if (!s) return;
    if (pname == GL_TEXTURE_BORDER_COLOR) {
        for (int k = 0; k < 4; k++) s->state.borderColor[k] = (float)v[k];
        s->state.borderIsInt = intBorder;
        return;
    }
    if (!samplerParam(ctx, s->state, pname, v, false, false)) ctx->setError(GL_INVALID_ENUM);
}

API void APIENTRY glSamplerParameteri(GLuint sampler, GLenum pname, GLint param) {
    MGL_CTX();
    if (pname == GL_TEXTURE_BORDER_COLOR) return ctx->setError(GL_INVALID_ENUM);
    double v[4] = {(double)param};
    samplerParamv(ctx, sampler, pname, v, false);
}
API void APIENTRY glSamplerParameterf(GLuint sampler, GLenum pname, GLfloat param) {
    MGL_CTX();
    if (pname == GL_TEXTURE_BORDER_COLOR) return ctx->setError(GL_INVALID_ENUM);
    double v[4] = {(double)param};
    samplerParamv(ctx, sampler, pname, v, false);
}
API void APIENTRY glSamplerParameteriv(GLuint sampler, GLenum pname, const GLint* p) {
    MGL_CTX();
    double v[4] = {(double)p[0]};
    if (pname == GL_TEXTURE_BORDER_COLOR)
        for (int k = 0; k < 4; k++) v[k] = p[k] / 2147483647.0;
    samplerParamv(ctx, sampler, pname, v, false);
}
API void APIENTRY glSamplerParameterfv(GLuint sampler, GLenum pname, const GLfloat* p) {
    MGL_CTX();
    double v[4] = {(double)p[0]};
    if (pname == GL_TEXTURE_BORDER_COLOR)
        for (int k = 0; k < 4; k++) v[k] = p[k];
    samplerParamv(ctx, sampler, pname, v, false);
}
API void APIENTRY glSamplerParameterIiv(GLuint sampler, GLenum pname, const GLint* p) {
    MGL_CTX();
    double v[4] = {(double)p[0]};
    if (pname == GL_TEXTURE_BORDER_COLOR)
        for (int k = 0; k < 4; k++) v[k] = p[k];
    samplerParamv(ctx, sampler, pname, v, true);
}
API void APIENTRY glSamplerParameterIuiv(GLuint sampler, GLenum pname, const GLuint* p) {
    MGL_CTX();
    double v[4] = {(double)p[0]};
    if (pname == GL_TEXTURE_BORDER_COLOR)
        for (int k = 0; k < 4; k++) v[k] = p[k];
    samplerParamv(ctx, sampler, pname, v, true);
}

static bool getSamplerParam(Context* ctx, GLuint sampler, GLenum pname, double* out, int* n) {
    Sampler* s = samplerObj(ctx, sampler);
    if (!s) return false;
    Texture tmp;
    tmp.sampler = s->state;
    *n = 1;
    const SamplerState& st = s->state;
    switch (pname) {
    case GL_TEXTURE_MIN_FILTER: out[0] = st.minFilter; return true;
    case GL_TEXTURE_MAG_FILTER: out[0] = st.magFilter; return true;
    case GL_TEXTURE_WRAP_S: out[0] = st.wrapS; return true;
    case GL_TEXTURE_WRAP_T: out[0] = st.wrapT; return true;
    case GL_TEXTURE_WRAP_R: out[0] = st.wrapR; return true;
    case GL_TEXTURE_MIN_LOD: out[0] = st.minLod; return true;
    case GL_TEXTURE_MAX_LOD: out[0] = st.maxLod; return true;
    case GL_TEXTURE_LOD_BIAS: out[0] = st.lodBias; return true;
    case GL_TEXTURE_COMPARE_MODE: out[0] = st.compareMode; return true;
    case GL_TEXTURE_COMPARE_FUNC: out[0] = st.compareFunc; return true;
    case GL_TEXTURE_MAX_ANISOTROPY: out[0] = st.maxAnisotropy; return true;
    case GL_TEXTURE_REDUCTION_MODE_ARB:
        if (!ctx->dev->caps.samplerReduction) return ctx->setError(GL_INVALID_ENUM), false;
        out[0] = st.reductionMode;
        return true;
    case GL_TEXTURE_CUBE_MAP_SEAMLESS: out[0] = st.seamless; return true;
    case GL_TEXTURE_BORDER_COLOR: *n = 4; for (int k = 0; k < 4; k++) out[k] = st.borderColor[k]; return true;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
}

API void APIENTRY glGetSamplerParameteriv(GLuint sampler, GLenum pname, GLint* params) {
    MGL_CTX();
    double v[4];
    int n;
    if (!getSamplerParam(ctx, sampler, pname, v, &n)) return;
    for (int k = 0; k < n; k++) params[k] = pname == GL_TEXTURE_BORDER_COLOR ? (GLint)llround(std::clamp(v[k], -1.0, 1.0) * 2147483647.0) : (GLint)lround(v[k]);
}
API void APIENTRY glGetSamplerParameterfv(GLuint sampler, GLenum pname, GLfloat* params) {
    MGL_CTX();
    double v[4];
    int n;
    if (!getSamplerParam(ctx, sampler, pname, v, &n)) return;
    for (int k = 0; k < n; k++) params[k] = (GLfloat)v[k];
}
API void APIENTRY glGetSamplerParameterIiv(GLuint sampler, GLenum pname, GLint* params) {
    MGL_CTX();
    double v[4];
    int n;
    if (!getSamplerParam(ctx, sampler, pname, v, &n)) return;
    for (int k = 0; k < n; k++) params[k] = (GLint)v[k];
}
API void APIENTRY glGetSamplerParameterIuiv(GLuint sampler, GLenum pname, GLuint* params) {
    MGL_CTX();
    double v[4];
    int n;
    if (!getSamplerParam(ctx, sampler, pname, v, &n)) return;
    for (int k = 0; k < n; k++) params[k] = (GLuint)v[k];
}
