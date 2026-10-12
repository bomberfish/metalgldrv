// Texture helpers shared by the texture/framebuffer/draw code.
#pragma once
#include <vector>
#include <IOSurface/IOSurfaceRef.h>
#include "gl/context.h"
#include "gl/pipeline.h"

namespace mgl {

MTLTextureType mtlTextureType(GLenum target, int samples);
int textureStorageLevelFor(const Texture* t);
bool ensureBufferTexture(Context* ctx, Texture* t);
// CGLTexImageIOSurface2D: makes level 0 of `t` (any earlier images dropped) a w x h image of format `f` whose
// storage is plane `plane` of `surface`. False if Metal can't make a texture over it (nothing changed then).
bool textureFromIOSurface(Context* ctx, Texture* t, GLenum internalFormat, const FormatInfo* f, int w, int h, IOSurfaceRef surface,
                          uint32_t plane);
// Bytes per texel of a buffer texture's data store (GL_RGB32* are 12, stored as 16).
inline int bufferTexelBytes(const FormatInfo* f) { return f->baseFormat == GL_RGB && f->bytesPerPixel == 16 ? 12 : f->bytesPerPixel; }
bool textureComplete(const Texture* t, const SamplerState& s, int* firstLevel, int* lastLevel);
id<MTLTexture> dummyTexture(Context* ctx, MTLTextureType type, FmtKind kind);
const void* resolveUnpackSource(Context* ctx, const void* pixels, size_t needed, bool* err);
void uploadTexSubImage(Context* ctx, Texture* t, int level, int face, int x, int y, int z, int w, int h, int d, GLenum format,
                       GLenum type, const void* pixels);
void uploadCompressedSubImage(Context* ctx, Texture* t, int level, int face, int x, int y, int z, int w, int h, int d,
                              GLsizei imageSize, const void* data);
bool readTextureRegion(Context* ctx, id<MTLTexture> tex, int level, int slice, int x, int y, int w, int h,
                       std::vector<uint8_t>& out, size_t& rowBytes, const FormatInfo* f, MTLBlitOption opt);

// buffer helpers (api_buffer.mm)
void writeBuffer(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, const void* data);
void syncBufferForCpuRead(Context* ctx, Buffer* b);
// CPU reads of a buffer's contents. Sparse buffers have no CPU mapping (cpu == nullptr): their
// data is copied out on the GPU (synchronous).
void readBuffer(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, void* dst);
// `size` bytes at `offset` for reading on the CPU: the mapping itself, or a per-thread copy for
// sparse buffers (valid until the next call). Callers synchronize with GPU writes as for `cpu`.
const uint8_t* bufferBytes(Context* ctx, Buffer* b, size_t offset, size_t size);
// Client-memory view of a pixel pack range: the mapping, or for a sparse buffer a staging copy
// written back to the buffer when the object goes out of scope.
struct PackStaging {
    Context* ctx = nullptr;
    Buffer* buffer = nullptr;
    size_t offset = 0;
    std::vector<uint8_t> data;
    uint8_t* begin(Context* c, Buffer* b, size_t off, size_t size);
    ~PackStaging();
};

// sparse textures / buffers (sparse.mm)
enum { kMaxSparseTextureSize = 16384, kMaxSparse3DTextureSize = 2048, kMaxSparseArrayTextureLayers = 2048 };
bool sparseTargetSupported(GLenum target);
// VIRTUAL_PAGE_SIZE_{X,Y,Z}_ARB of page size `index`; false if the format can't be sparse.
bool sparsePageSize(GLenum target, const FormatInfo* f, int index, int* x, int* y, int* z);
int sparseNumPageSizes(GLenum target, const FormatInfo* f);
// TexStorage* checks for a TEXTURE_SPARSE_ARB texture (records the error).
bool sparseStorageCheck(Context* ctx, Texture* t, const FormatInfo* f, int w, int h, int d);
MTLTextureUsage sparseTextureUsage(const FormatInfo* f);
void releaseSparseTiles(Texture* t, uint64_t serial);
// glBufferStorage(SPARSE_STORAGE_BIT_ARB) (records the error).
bool sparseBufferStorage(Context* ctx, Buffer* b, GLsizeiptr size, GLbitfield flags);
void releaseSparsePages(Buffer* b, uint64_t serial);

}  // namespace mgl
