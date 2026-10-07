// Sparse textures and buffers: ARB_sparse_texture(2), ARB_sparse_buffer.
//
// Sparse storage is a Metal placement sparse resource (MTLTextureDescriptor.placementSparsePageSize,
// newBufferWithLength:options:placementSparsePageSize:). Committing a region takes one device tile
// (Device::kSparsePageSize) per Metal sparse tile from the device's placement heaps and maps it with
// a MTL4CommandQueue mapping update; decommitting unmaps and returns the tiles. Unmapped tiles read
// as zero and drop writes (MTLTextureSparseTier1 / MTLBufferSparseTier1), which is what
// ARB_sparse_texture2 requires of uncommitted texels.
//
// Virtual pages are the standard 64 KB page shapes for the format's texel size (the shapes
// ARB_sparse_texture2 implementations report), each an exact multiple of the Metal tile; levels from
// the Metal mip tail on (MTLTexture.firstMipmapInTail) form the GL mip tail, committed per layer.
#import "gl/texture.h"
#include <unistd.h>
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {

static bool sparseTextureTarget(GLenum target) {
    switch (target) {
    case GL_TEXTURE_2D: case GL_TEXTURE_2D_ARRAY: case GL_TEXTURE_CUBE_MAP: case GL_TEXTURE_CUBE_MAP_ARRAY: case GL_TEXTURE_3D:
    case GL_TEXTURE_RECTANGLE: case GL_TEXTURE_2D_MULTISAMPLE: case GL_TEXTURE_2D_MULTISAMPLE_ARRAY:
        return true;
    default: return false;
    }
}

bool sparseTargetSupported(GLenum target) { return sparseTextureTarget(target); }

static bool sparseFormat(const FormatInfo* f) {
    return f && f->comp != Comp::Compressed && f->kind != FmtKind::Stencil && f->kind != FmtKind::DepthStencil;
}

// Standard virtual page shape of a 64 KB page for a texel size in bytes.
static bool standardPageShape(int bytes, int* x, int* y) {
    switch (bytes) {
    case 1: *x = 256; *y = 256; return true;
    case 2: *x = 256; *y = 128; return true;
    case 4: *x = 128; *y = 128; return true;
    case 8: *x = 128; *y = 64; return true;
    case 16: *x = 64; *y = 64; return true;
    default: return false;
    }
}

bool sparsePageSize(GLenum target, const FormatInfo* f, int index, int* x, int* y, int* z) {
    Device* dev = Device::get();
    if (!dev || !dev->caps.sparse || index != 0 || !sparseFormat(f)) return false;
    MTLTextureType type = target == GL_TEXTURE_3D ? MTLTextureType3D : MTLTextureType2D;
    MTLSize tile = [dev->mtl sparseTileSizeWithTextureType:type pixelFormat:f->mtl sampleCount:1 sparsePageSize:Device::kSparsePageSize];
    MTLSize page = [dev->mtl sparseTileSizeWithTextureType:type pixelFormat:f->mtl sampleCount:1 sparsePageSize:MTLSparsePageSize64];
    if (!tile.width || !tile.height || !tile.depth || !page.width || !page.height || !page.depth) return false;
    int px = (int)page.width, py = (int)page.height, pz = (int)page.depth;
    int bits = f->rBits + f->gBits + f->bBits + f->aBits + f->dBits + f->sBits;
    // 16-bit packed formats are stored as RGBA8, but their pages are those of a 16-bit format.
    if (f->internalFormat == GL_RGB565 || f->internalFormat == GL_RGB5_A1 || f->internalFormat == GL_RGBA4) bits = 16;
    int sx, sy;
    if (target != GL_TEXTURE_3D && bits % 8 == 0 && standardPageShape(bits / 8, &sx, &sy) && sx % tile.width == 0 && sy % tile.height == 0) {
        px = sx;
        py = sy;
        pz = 1;
    }
    if (px % tile.width || py % tile.height || pz % tile.depth) return false;
    *x = px;
    *y = py;
    *z = pz;
    return true;
}

int sparseNumPageSizes(GLenum target, const FormatInfo* f) {
    int x, y, z;
    return sparseTextureTarget(target) && sparsePageSize(target, f, 0, &x, &y, &z) ? 1 : 0;
}

bool sparseStorageCheck(Context* ctx, Texture* t, const FormatInfo* f, int w, int h, int d) {
    int px, py, pz;
    if (!sparseTextureTarget(t->target) || !sparsePageSize(t->target, f, t->pageSizeIndex, &px, &py, &pz)) {
        ctx->setError(GL_INVALID_OPERATION, "sparse texture: no virtual page size %d for format 0x%x", t->pageSizeIndex,
                      f ? f->internalFormat : 0);
        return false;
    }
    bool tooBig = t->target == GL_TEXTURE_3D ? std::max(w, std::max(h, d)) > kMaxSparse3DTextureSize
                                             : std::max(w, h) > kMaxSparseTextureSize;
    if ((t->target == GL_TEXTURE_2D_ARRAY || t->target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY) && d > kMaxSparseArrayTextureLayers) tooBig = true;
    if (t->target == GL_TEXTURE_CUBE_MAP_ARRAY && d > kMaxSparseArrayTextureLayers * 6) tooBig = true;
    if (tooBig) {
        ctx->setError(GL_INVALID_VALUE, "sparse texture exceeds the maximum sparse size");
        return false;
    }
    // ARB_sparse_texture2: sizes need not be page multiples, and SPARSE_TEXTURE_FULL_ARRAY_CUBE_MIPMAPS
    // is TRUE (the mip tail is per layer), so no further restrictions.
    return true;
}

MTLTextureUsage sparseTextureUsage(const FormatInfo* f) {
    // Sparse storage is never reallocated (that would drop its mappings): request up front every
    // usage glBindImageTexture / texture views / framebuffers may need.
    MTLTextureUsage u = MTLTextureUsageShaderRead | MTLTextureUsagePixelFormatView;
    if (f->renderable) u |= MTLTextureUsageRenderTarget;
    if (f->kind != FmtKind::Depth && f->mtl != MTLPixelFormatRGB9E5Float) u |= MTLTextureUsageShaderWrite;
    if (f->mtl == MTLPixelFormatR32Uint || f->mtl == MTLPixelFormatR32Sint) u |= MTLTextureUsageShaderAtomic;
    return u;
}

void releaseSparseTiles(Texture* t, uint64_t serial) {
    Device* dev = Device::get();
    for (auto& [k, tiles] : t->sparseTiles) dev->freeSparseTiles(tiles, serial);
    t->sparseTiles.clear();
}

// Orders a mapping update after the work recorded so far. A decommit also waits for that work
// (the freed tiles may be remapped to other resources right away), and a commit soon after a
// decommit of the same resource waits until the GPU dropped the stale translations of the unmap
// (Device::kSparseSettleTime); mapping through them would silently lose the commit.
static void syncForMappingUpdate(Context* ctx, bool commit, double lastUnmap) {
    if (!commit) return ctx->finish();
    double wait = lastUnmap + Device::kSparseSettleTime - CFAbsoluteTimeGetCurrent();
    if (wait <= 0) return ctx->flush();
    ctx->finish();
    usleep((useconds_t)(wait * 1e6));
}

// Key of one committed Metal tile (or of a layer's mip tail: level = first tail level, x = y = z = 0).
static uint64_t sparseTileKey(int level, int slice, uint32_t tx, uint32_t ty, uint32_t tz) {
    return (uint64_t)level << 59 | (uint64_t)slice << 44 | (uint64_t)tz << 32 | (uint64_t)ty << 16 | tx;
}

// Commits or decommits a page-aligned region of a sparse texture level (z: layer / cube face / depth).
static void commitRegion(Context* ctx, Texture* t, int level, int x, int y, int z, int w, int h, int d, bool commit) {
    Device* dev = ctx->dev;
    id<MTLTexture> tex = t->mtl;
    if (!tex) return;
    syncForMappingUpdate(ctx, commit, t->sparseUnmapTime);
    MTLSize ts = [dev->mtl sparseTileSizeWithTextureType:tex.textureType pixelFormat:tex.pixelFormat sampleCount:tex.sampleCount
                                          sparsePageSize:Device::kSparsePageSize];
    const bool is3D = tex.textureType == MTLTextureType3D;
    const int tail = (int)tex.firstMipmapInTail;
    // emulated multisample storage: each layer is `samples` slices
    if (t->msArray) {
        z *= t->samples;
        d *= t->samples;
    }
    std::vector<Device::SparseOp> ops;
    auto apply = [&](uint64_t key, NSUInteger lvl, NSUInteger slice, MTLRegion region, uint32_t count) {
        auto it = t->sparseTiles.find(key);
        Device::SparseOp op;
        op.level = lvl;
        op.slice = slice;
        op.region = region;
        if (commit) {
            if (it != t->sparseTiles.end()) return true;
            if (!dev->allocSparseTiles(count, op.tiles)) return false;
            t->sparseTiles[key] = op.tiles;
        } else {
            if (it == t->sparseTiles.end()) return true;
            op.tiles = it->second;
            op.unmap = true;
            dev->freeSparseTiles(it->second, 0, true);
            t->sparseTiles.erase(it);
        }
        ops.push_back(op);
        return true;
    };
    bool ok = true;
    if (level >= tail) {
        // The mip tail of each layer is committed as a whole (3D textures have a single tail).
        uint32_t count = (uint32_t)((tex.tailSizeInBytes + dev->sparseTileBytes - 1) / dev->sparseTileBytes);
        int s0 = is3D ? 0 : z, s1 = is3D ? 1 : z + d;
        for (int s = s0; s < s1 && ok; s++) ok = apply(sparseTileKey(tail, s, 0, 0, 0), tail, s, MTLRegionMake3D(0, 0, 0, count, 1, 1), count);
    } else {
        int lw = std::max(1, (int)tex.width >> level), lh = std::max(1, (int)tex.height >> level);
        int ld = is3D ? std::max(1, (int)tex.depth >> level) : 1;
        auto range = [](int o, int n, int lim, int tile, uint32_t& a, uint32_t& b) {
            a = (uint32_t)(o / tile);
            b = (uint32_t)((std::min(o + n, lim) + tile - 1) / tile);
        };
        uint32_t x0, x1, y0, y1, z0 = 0, z1 = 1;
        range(x, w, lw, (int)ts.width, x0, x1);
        range(y, h, lh, (int)ts.height, y0, y1);
        if (is3D) range(z, d, ld, (int)ts.depth, z0, z1);
        int s0 = is3D ? 0 : z, s1 = is3D ? 1 : z + d;
        for (int s = s0; s < s1 && ok; s++)
            for (uint32_t tz = z0; tz < z1 && ok; tz++)
                for (uint32_t ty = y0; ty < y1 && ok; ty++)
                    for (uint32_t tx = x0; tx < x1 && ok; tx++)
                        ok = apply(sparseTileKey(level, s, tx, ty, tz), level, s, MTLRegionMake3D(tx, ty, tz, 1, 1, 1), 1);
    }
    if (!ok) ctx->setError(GL_OUT_OF_MEMORY, "sparse texture commitment: out of tile memory");
    if (!ops.empty()) dev->updateSparseMappings(tex, ops);
    if (!ops.empty() && t->sparseRead) dev->updateSparseMappings(t->sparseRead, ops);
    if (!commit && !ops.empty()) t->sparseUnmapTime = CFAbsoluteTimeGetCurrent();
}

static void texPageCommitment(Context* ctx, Texture* t, GLint level, GLint x, GLint y, GLint z, GLsizei w, GLsizei h, GLsizei d,
                              GLboolean commit) {
    if (!t->immutable || !t->sparse || !t->mtl) return ctx->setError(GL_INVALID_OPERATION, "texture is not an immutable sparse texture");
    if (level < 0 || level >= t->immutableLevels) return ctx->setError(GL_INVALID_VALUE);
    if (x < 0 || y < 0 || z < 0 || w < 0 || h < 0 || d < 0) return ctx->setError(GL_INVALID_VALUE);
    const TexImageLevel& L = t->levels[0][level];
    int W = L.width, H = L.height, D = t->target == GL_TEXTURE_CUBE_MAP ? 6 : L.depth;
    if ((int64_t)x + w > W || (int64_t)y + h > H || (int64_t)z + d > D)
        return ctx->setError(GL_INVALID_OPERATION, "commitment region exceeds the level");
    int px, py, pz;
    if (!sparsePageSize(t->target, L.fmt, t->pageSizeIndex, &px, &py, &pz)) return ctx->setError(GL_INVALID_OPERATION);
    if (x % px || y % py || z % pz) return ctx->setError(GL_INVALID_VALUE, "commitment offset is not a multiple of the page size");
    if ((w % px && x + w != W) || (h % py && y + h != H) || (d % pz && z + d != D))
        return ctx->setError(GL_INVALID_OPERATION, "commitment size is not a multiple of the page size");
    if (!w || !h || !d) return;
    commitRegion(ctx, t, level, x, y, z, w, h, d, commit);
}

// --- buffers ------------------------------------------------------------------------------
bool sparseBufferStorage(Context* ctx, Buffer* b, GLsizeiptr size, GLbitfield flags) {
    if (flags & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT)) {
        ctx->setError(GL_INVALID_VALUE, "sparse buffers cannot be mapped");
        return false;
    }
    Device* dev = ctx->dev;
    size_t page = dev->sparseTileBytes;
    size_t pages = ((size_t)size + page - 1) / page;
    id<MTLBuffer> buf = [dev->mtl newBufferWithLength:pages * page
                                              options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked
                              placementSparsePageSize:Device::kSparsePageSize];
    if (!buf) {
        ctx->setError(GL_OUT_OF_MEMORY);
        return false;
    }
    if (!b->label.empty()) buf.label = @(b->label.c_str());
    dev->makeResident(buf);
    if (b->mtl) dev->releaseAfter(b->mtl, std::max(b->lastUse, b->lastGpuWrite));
    b->mtl = buf;
    b->cpu = nullptr;
    b->size = size;
    b->sparse = true;
    b->sparsePages.assign(pages, SparseTiles());
    b->contentVersion++;
    b->lastUse = 0;
    b->lastGpuWrite = 0;
    b->generation++;
    return true;
}

void releaseSparsePages(Buffer* b, uint64_t serial) {
    Device* dev = Device::get();
    for (auto& p : b->sparsePages) dev->freeSparseTiles(p, serial);
    b->sparsePages.clear();
}

static void bufferPageCommitment(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, GLboolean commit) {
    if (!b->sparse) return ctx->setError(GL_INVALID_OPERATION, "buffer is not sparse");
    const GLsizeiptr page = (GLsizeiptr)ctx->dev->sparseTileBytes;
    if (offset < 0 || size < 0 || offset + size > b->size) return ctx->setError(GL_INVALID_VALUE);
    if (offset % page || (size % page && offset + size != b->size))
        return ctx->setError(GL_INVALID_VALUE, "commitment range is not page aligned");
    if (!size) return;
    syncForMappingUpdate(ctx, commit, b->sparseUnmapTime);
    Device* dev = ctx->dev;
    std::vector<Device::SparseOp> ops;
    size_t p0 = (size_t)(offset / page), p1 = (size_t)((offset + size + page - 1) / page);
    for (size_t p = p0; p < p1; p++) {
        SparseTiles& cur = b->sparsePages[p];
        Device::SparseOp op;
        op.region = MTLRegionMake1D(p, 1);
        if (commit) {
            if (cur.count) continue;
            if (!dev->allocSparseTiles(1, op.tiles)) {
                ctx->setError(GL_OUT_OF_MEMORY, "sparse buffer commitment: out of tile memory");
                break;
            }
            cur = op.tiles;
        } else {
            if (!cur.count) continue;
            op.tiles = cur;
            op.unmap = true;
            dev->freeSparseTiles(cur, 0, true);
            cur = SparseTiles();
        }
        ops.push_back(op);
    }
    if (!ops.empty()) dev->updateSparseMappings(b->mtl, ops);
    if (!commit && !ops.empty()) b->sparseUnmapTime = CFAbsoluteTimeGetCurrent();
    b->contentVersion++;
}

}  // namespace mgl

// --- entry points -----------------------------------------------------------------------------
API void APIENTRY glTexPageCommitmentARB(GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                         GLsizei height, GLsizei depth, GLboolean commit) {
    MGL_CTX();
    if (!ctx->dev->caps.sparse || !sparseTextureTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    Texture* t = ctx->boundTexture(target);
    if (!t) return ctx->setError(GL_INVALID_OPERATION);
    texPageCommitment(ctx, t, level, xoffset, yoffset, zoffset, width, height, depth, commit);
}

API void APIENTRY glTexturePageCommitmentEXT(GLuint texture, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width,
                                             GLsizei height, GLsizei depth, GLboolean commit) {
    MGL_CTX();
    if (!ctx->dev->caps.sparse) return ctx->setError(GL_INVALID_OPERATION);
    Texture* t = texture ? ctx->share->textures.get(texture) : nullptr;
    if (!t || !t->target) return ctx->setError(GL_INVALID_OPERATION, "glTexturePageCommitmentEXT: %u is not a texture", texture);
    texPageCommitment(ctx, t, level, xoffset, yoffset, zoffset, width, height, depth, commit);
}

API void APIENTRY glBufferPageCommitmentARB(GLenum target, GLintptr offset, GLsizeiptr size, GLboolean commit) {
    MGL_CTX();
    Ref<Buffer>* r = ctx->dev->caps.sparse ? ctx->bufferBindingRef(target) : nullptr;
    if (!r) return ctx->setError(GL_INVALID_ENUM);
    if (!*r) return ctx->setError(GL_INVALID_OPERATION, "no buffer bound to 0x%x", target);
    bufferPageCommitment(ctx, r->get(), offset, size, commit);
}

API void APIENTRY glNamedBufferPageCommitmentARB(GLuint buffer, GLintptr offset, GLsizeiptr size, GLboolean commit) {
    MGL_CTX();
    Buffer* b = ctx->dev->caps.sparse ? ctx->share->buffers.get(buffer) : nullptr;
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    bufferPageCommitment(ctx, b, offset, size, commit);
}

API void APIENTRY glNamedBufferPageCommitmentEXT(GLuint buffer, GLintptr offset, GLsizeiptr size, GLboolean commit) {
    glNamedBufferPageCommitmentARB(buffer, offset, size, commit);
}
