// Texture helpers shared by the texture/framebuffer/draw code.
#pragma once
#include <vector>
#include "gl/context.h"
#include "gl/pipeline.h"

namespace mgl {

MTLTextureType mtlTextureType(GLenum target, int samples);
int textureStorageLevelFor(const Texture* t);
bool ensureBufferTexture(Context* ctx, Texture* t);
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

}  // namespace mgl
