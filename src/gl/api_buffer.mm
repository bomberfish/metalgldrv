// Buffer objects.
#import "gl/texture.h"
#include <cstring>
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {

Buffer::~Buffer() {
    if (mtl) Device::get()->releaseAfter(mtl, lastUse);
    if (rotated) Device::get()->releaseAfter(rotated, rotatedLastUse);
    if (sparse) {
        releaseSparsePages(this, std::max(lastUse, lastGpuWrite));
        Device::get()->retainSparse(mtl, std::max(lastUse, lastGpuWrite));
    }
    free(mapShadow);
}

// Replace storage with a fresh allocation. Old storage stays alive until the GPU is done.
static bool allocStorage(Buffer* b, GLsizeiptr size) {
    Device* d = Device::get();
    if (b->mtl) d->releaseAfter(b->mtl, std::max(b->lastUse, b->lastGpuWrite));
    b->mtl = d->newBuffer((size_t)std::max<GLsizeiptr>(size, 4));
    if (!b->mtl) {
        b->cpu = nullptr;
        b->size = 0;
        return false;
    }
    b->cpu = (uint8_t*)b->mtl.contents;
    b->size = size;
    b->contentVersion++;
    b->lastUse = 0;
    b->lastGpuWrite = 0;
    b->generation++;
    return true;
}

static bool busy(Context* ctx, Buffer* b) { return b->lastUse && !ctx->dev->isComplete(b->lastUse); }

// Make GPU writes visible to the CPU.
void syncBufferForCpuRead(Context* ctx, Buffer* b) {
    if (b->lastGpuWrite && !ctx->dev->isComplete(b->lastGpuWrite)) ctx->waitForResource(b->lastGpuWrite);
}

// Write `size` bytes at `offset` preserving GL ordering semantics.
void writeBuffer(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, const void* data) {
    if (size <= 0) return;
    b->contentVersion++;
    if (b->cpu && !busy(ctx, b)) {
        memcpy(b->cpu + offset, data, size);
        return;
    }
    const GLsizeiptr kRenameLimit = 256 * 1024;
    bool gpuWritePending = b->lastGpuWrite && !ctx->dev->isComplete(b->lastGpuWrite);
    if (!b->immutable && b->size <= kRenameLimit && !gpuWritePending) {
        // Rename: new storage with old contents + update. In-flight work keeps the old buffer.
        id<MTLBuffer> old = b->mtl;
        uint8_t* oldCpu = b->cpu;
        uint64_t oldUse = b->lastUse;
        Device* d = ctx->dev;
        b->mtl = d->newBuffer((size_t)std::max<GLsizeiptr>(b->size, 4));
        b->cpu = (uint8_t*)b->mtl.contents;
        memcpy(b->cpu, oldCpu, b->size);
        memcpy(b->cpu + offset, data, size);
        d->releaseAfter(old, oldUse);
        b->lastUse = 0;
        b->generation++;
        return;
    }
    // Staged GPU copy, ordered in the command stream.
    UploadAlloc up = ctx->upload(size, 16);
    memcpy(up.cpu, data, size);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromBuffer:up.buffer sourceOffset:up.offset toBuffer:b->mtl destinationOffset:offset size:size];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    b->lastUse = ctx->serial();
    b->lastGpuWrite = b->lastUse;
}

void readBuffer(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, void* dst) {
    if (size <= 0) return;
    if (b->cpu) {
        syncBufferForCpuRead(ctx, b);
        memcpy(dst, b->cpu + offset, size);
        return;
    }
    id<MTLBuffer> tmp = ctx->dev->newBuffer((size_t)size);
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    // Writes to uncommitted pages linger in the (virtually tagged) GPU caches until flushed with
    // ResourceAlias visibility; without it they would read back instead of zeros.
    [e barrierAfterQueueStages:MTLStageAll beforeStages:MTLStageBlit
             visibilityOptions:MTL4VisibilityOptionDevice | MTL4VisibilityOptionResourceAlias];
    [e copyFromBuffer:b->mtl sourceOffset:offset toBuffer:tmp destinationOffset:0 size:size];
    b->lastUse = ctx->serial();
    ctx->finish();
    memcpy(dst, tmp.contents, size);
    ctx->dev->releaseAfter(tmp, 0);
}

const uint8_t* bufferBytes(Context* ctx, Buffer* b, size_t offset, size_t size) {
    if (b->cpu) return b->cpu + offset;
    static thread_local std::vector<uint8_t> copy;
    copy.resize(std::max<size_t>(size, 1));
    readBuffer(ctx, b, (GLintptr)offset, (GLsizeiptr)size, copy.data());
    return copy.data();
}

uint8_t* PackStaging::begin(Context* c, Buffer* b, size_t off, size_t size) {
    if (b->cpu) return b->cpu + off;
    ctx = c;
    buffer = b;
    offset = off;
    data.resize(size);
    readBuffer(c, b, (GLintptr)off, (GLsizeiptr)size, data.data());  // bytes the pack skips keep their contents
    return data.data();
}

PackStaging::~PackStaging() {
    if (buffer && !data.empty()) writeBuffer(ctx, buffer, (GLintptr)offset, (GLsizeiptr)data.size(), data.data());
}

}  // namespace mgl

static Buffer* lookupBuffer(Context* ctx, GLuint name) { return ctx->share->buffers.get(name); }

static Buffer* boundBuffer(Context* ctx, GLenum target, bool errIfNone = true) {
    Ref<Buffer>* r = ctx->bufferBindingRef(target);
    if (!r) {
        ctx->setError(GL_INVALID_ENUM, "bad buffer target 0x%x", target);
        return nullptr;
    }
    if (!*r) {
        if (errIfNone) ctx->setError(GL_INVALID_OPERATION, "no buffer bound to 0x%x", target);
        return nullptr;
    }
    return r->get();
}

static bool validUsage(GLenum u) {
    switch (u) {
    case GL_STREAM_DRAW: case GL_STREAM_READ: case GL_STREAM_COPY: case GL_STATIC_DRAW: case GL_STATIC_READ: case GL_STATIC_COPY:
    case GL_DYNAMIC_DRAW: case GL_DYNAMIC_READ: case GL_DYNAMIC_COPY:
        return true;
    default: return false;
    }
}

// --- names ---
API void APIENTRY glGenBuffers(GLsizei n, GLuint* buffers) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) buffers[i] = ctx->share->buffers.genName();
}

API void APIENTRY glCreateBuffers(GLsizei n, GLuint* buffers) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) {
        buffers[i] = ctx->share->buffers.genName();
        ctx->share->buffers.create(buffers[i], new Buffer());
    }
}

API void APIENTRY glDeleteBuffers(GLsizei n, const GLuint* buffers) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) {
        if (!buffers[i]) continue;
        Buffer* b = ctx->share->buffers.get(buffers[i]);
        if (b) {
            if (b->mapped) {
                b->mapped = false;
                b->mapPointer = nullptr;
            }
            ctx->unbindDeletedBuffer(b);
        }
        ctx->share->buffers.remove(buffers[i]);
    }
}

API GLboolean APIENTRY glIsBuffer(GLuint buffer) {
    MGL_CTX_RET(GL_FALSE);
    return buffer && ctx->share->buffers.get(buffer) != nullptr;
}

static Buffer* bufferForBind(Context* ctx, GLuint name) {
    if (!name) return nullptr;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    Buffer* b = ctx->share->buffers.get(name);
    if (!b) {
        if (!ctx->share->buffers.isName(name) && ctx->coreProfile) {
            ctx->setError(GL_INVALID_OPERATION, "glBindBuffer: %u is not a name from glGenBuffers", name);
            return (Buffer*)-1;
        }
        b = ctx->share->buffers.create(name, new Buffer());
    }
    return b;
}

API void APIENTRY glBindBuffer(GLenum target, GLuint buffer) {
    MGL_CTX();
    Ref<Buffer>* r = ctx->bufferBindingRef(target);
    if (!r) return ctx->setError(GL_INVALID_ENUM);
    Buffer* b = bufferForBind(ctx, buffer);
    if (b == (Buffer*)-1) return;
    *r = b;
    if (target == GL_ELEMENT_ARRAY_BUFFER && ctx->vao) ctx->vao->version++;
}

static bool indexedTargetInfo(Context* ctx, GLenum target, GLuint index, IndexedBufferBinding** out, Ref<Buffer>** generic) {
    switch (target) {
    case GL_UNIFORM_BUFFER:
        if (index >= kMaxUniformBufferBindings) return ctx->setError(GL_INVALID_VALUE), false;
        *out = &ctx->uniformBuffers[index];
        *generic = &ctx->uniformBufferGeneric;
        return true;
    case GL_SHADER_STORAGE_BUFFER:
        if (index >= kMaxSSBOBindings) return ctx->setError(GL_INVALID_VALUE), false;
        *out = &ctx->storageBuffers[index];
        *generic = &ctx->ssboGeneric;
        return true;
    case GL_ATOMIC_COUNTER_BUFFER:
        if (index >= kMaxAtomicCounterBindings) return ctx->setError(GL_INVALID_VALUE), false;
        *out = &ctx->atomicCounterBuffers[index];
        *generic = &ctx->atomicCounterGeneric;
        return true;
    case GL_TRANSFORM_FEEDBACK_BUFFER:
        if (index >= kMaxXfbBuffers) return ctx->setError(GL_INVALID_VALUE), false;
        if (ctx->xfb->active) return ctx->setError(GL_INVALID_OPERATION), false;
        *out = nullptr;
        *generic = &ctx->xfbGeneric;
        return true;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
}

static void bindRange(Context* ctx, GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size, bool whole) {
    IndexedBufferBinding* ib;
    Ref<Buffer>* generic;
    if (!indexedTargetInfo(ctx, target, index, &ib, &generic)) return;
    Buffer* b = bufferForBind(ctx, buffer);
    if (b == (Buffer*)-1) return;
    bool unbind = !b;
    if (!whole) {
        // BindBufferRange: size is only checked with a buffer, the offset alignment always (as Mesa).
        if (b && (size <= 0 || offset < 0)) return ctx->setError(GL_INVALID_VALUE);
        GLintptr align = target == GL_UNIFORM_BUFFER ? 16 : target == GL_SHADER_STORAGE_BUFFER ? 16 : 4;
        if (offset % align) return ctx->setError(GL_INVALID_VALUE, "unaligned buffer range offset");
        if (b && target == GL_TRANSFORM_FEEDBACK_BUFFER && (size % 4)) return ctx->setError(GL_INVALID_VALUE);
    }
    *generic = b;
    if (unbind) whole = true;  // unbinding resets start/size to 0
    if (target == GL_TRANSFORM_FEEDBACK_BUFFER) {
        auto& bb = ctx->xfb->buffers[index];
        bb.buffer = b;
        bb.offset = whole ? 0 : offset;
        bb.size = whole ? 0 : size;
        return;
    }
    ib->buffer = b;
    ib->offset = whole ? 0 : offset;
    ib->size = whole ? 0 : size;
}

API void APIENTRY glBindBufferBase(GLenum target, GLuint index, GLuint buffer) {
    MGL_CTX();
    bindRange(ctx, target, index, buffer, 0, 0, true);
}

API void APIENTRY glBindBufferRange(GLenum target, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    MGL_CTX();
    bindRange(ctx, target, index, buffer, offset, size, false);
}

// ARB_multi_bind (GL 4.6 §6.1.1): first+count beyond the binding points is INVALID_OPERATION;
// a bad entry raises an error and is skipped while the others still bind.
static bool multiBindCheck(Context* ctx, GLenum target, GLuint first, GLsizei count) {
    int max;
    switch (target) {
    case GL_UNIFORM_BUFFER: max = kMaxUniformBufferBindings; break;
    case GL_SHADER_STORAGE_BUFFER: max = kMaxSSBOBindings; break;
    case GL_ATOMIC_COUNTER_BUFFER: max = kMaxAtomicCounterBindings; break;
    case GL_TRANSFORM_FEEDBACK_BUFFER: max = kMaxXfbBuffers; break;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
    if (count < 0) return ctx->setError(GL_INVALID_VALUE), false;
    if ((uint64_t)first + (uint64_t)count > (uint64_t)max) return ctx->setError(GL_INVALID_OPERATION), false;
    return true;
}

API void APIENTRY glBindBuffersBase(GLenum target, GLuint first, GLsizei count, const GLuint* buffers) {
    MGL_CTX();
    if (!multiBindCheck(ctx, target, first, count)) return;
    for (int i = 0; i < count; i++) {
        GLuint b = buffers ? buffers[i] : 0;
        if (b && !ctx->share->buffers.get(b)) {
            ctx->setError(GL_INVALID_OPERATION, "BindBuffersBase: %u is not a buffer", b);
            continue;
        }
        bindRange(ctx, target, first + i, b, 0, 0, true);
    }
}

API void APIENTRY glBindBuffersRange(GLenum target, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets,
                                     const GLsizeiptr* sizes) {
    MGL_CTX();
    if (!multiBindCheck(ctx, target, first, count)) return;
    for (int i = 0; i < count; i++) {
        GLuint b = buffers ? buffers[i] : 0;
        if (b && !ctx->share->buffers.get(b)) {
            ctx->setError(GL_INVALID_OPERATION, "BindBuffersRange: %u is not a buffer", b);
            continue;
        }
        if (b && (offsets[i] < 0 || sizes[i] <= 0)) {
            ctx->setError(GL_INVALID_VALUE);
            continue;
        }
        bindRange(ctx, target, first + i, b, b ? offsets[i] : 0, b ? sizes[i] : 0, b == 0);
    }
}

// --- storage ---
static void bufferData(Context* ctx, Buffer* b, GLsizeiptr size, const void* data, GLenum usage) {
    if (size < 0) return ctx->setError(GL_INVALID_VALUE);
    if (!validUsage(usage)) return ctx->setError(GL_INVALID_ENUM);
    if (b->immutable) return ctx->setError(GL_INVALID_OPERATION, "buffer storage is immutable");
    if (b->mapped) {
        b->mapped = false;
        b->mapPointer = nullptr;
    }
    if (!allocStorage(b, size)) return ctx->setError(GL_OUT_OF_MEMORY);
    if (data) memcpy(b->cpu, data, size);
    b->usage = usage;
}

static void bufferStorage(Context* ctx, Buffer* b, GLsizeiptr size, const void* data, GLbitfield flags) {
    if (size <= 0) return ctx->setError(GL_INVALID_VALUE);
    GLbitfield valid = GL_DYNAMIC_STORAGE_BIT | GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT |
                       GL_MAP_COHERENT_BIT | GL_CLIENT_STORAGE_BIT;
    if (ctx->dev->caps.sparse) valid |= GL_SPARSE_STORAGE_BIT_ARB;
    if (flags & ~valid) return ctx->setError(GL_INVALID_VALUE);
    if ((flags & GL_MAP_PERSISTENT_BIT) && !(flags & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT))) return ctx->setError(GL_INVALID_VALUE);
    if ((flags & GL_MAP_COHERENT_BIT) && !(flags & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_VALUE);
    if (b->immutable) return ctx->setError(GL_INVALID_OPERATION);
    if (flags & GL_SPARSE_STORAGE_BIT_ARB) {
        // ARB_sparse_buffer: no pages are committed yet, so `data` has nowhere to go.
        if (!sparseBufferStorage(ctx, b, size, flags)) return;
    } else {
        if (!allocStorage(b, size)) return ctx->setError(GL_OUT_OF_MEMORY);
        if (data) memcpy(b->cpu, data, size);
    }
    b->immutable = true;
    b->storageFlags = flags;
    b->usage = GL_DYNAMIC_DRAW;
}

static void bufferSubData(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, const void* data) {
    if (offset < 0 || size < 0 || offset + size > b->size) return ctx->setError(GL_INVALID_VALUE);
    // Updating a non-persistently mapped range is an error; the rest of the buffer stays writable.
    if (b->mapped && !(b->mapAccess & GL_MAP_PERSISTENT_BIT) && offset < b->mapOffset + b->mapLength && b->mapOffset < offset + size)
        return ctx->setError(GL_INVALID_OPERATION);
    if (b->immutable && !(b->storageFlags & GL_DYNAMIC_STORAGE_BIT)) return ctx->setError(GL_INVALID_OPERATION);
    if (!data || !size) return;
    writeBuffer(ctx, b, offset, size, data);
}

API void APIENTRY glBufferData(GLenum target, GLsizeiptr size, const void* data, GLenum usage) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) bufferData(ctx, b, size, data, usage);
}

API void APIENTRY glNamedBufferData(GLuint buffer, GLsizeiptr size, const void* data, GLenum usage) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    bufferData(ctx, b, size, data, usage);
}

API void APIENTRY glBufferStorage(GLenum target, GLsizeiptr size, const void* data, GLbitfield flags) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) bufferStorage(ctx, b, size, data, flags);
}

API void APIENTRY glNamedBufferStorage(GLuint buffer, GLsizeiptr size, const void* data, GLbitfield flags) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    bufferStorage(ctx, b, size, data, flags);
}

API void APIENTRY glBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, const void* data) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) bufferSubData(ctx, b, offset, size, data);
}

API void APIENTRY glNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, const void* data) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    bufferSubData(ctx, b, offset, size, data);
}

// --- mapping ---
static void* mapRange(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    const GLbitfield valid = GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT |
                             GL_MAP_FLUSH_EXPLICIT_BIT | GL_MAP_UNSYNCHRONIZED_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT;
    if (offset < 0 || length < 0 || offset + length > b->size || (access & ~valid)) return ctx->setError(GL_INVALID_VALUE), nullptr;
    if (length == 0) return ctx->setError(GL_INVALID_OPERATION), nullptr;  // GL 4.6 6.3.1
    if (!(access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT))) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    if ((access & GL_MAP_READ_BIT) && (access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT | GL_MAP_UNSYNCHRONIZED_BIT)))
        return ctx->setError(GL_INVALID_OPERATION), nullptr;
    if ((access & GL_MAP_FLUSH_EXPLICIT_BIT) && !(access & GL_MAP_WRITE_BIT)) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    if (b->mapped) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    if (b->immutable) {
        GLbitfield need = access & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT | GL_MAP_COHERENT_BIT);
        if ((need & b->storageFlags) != need) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    }
    b->mapped = true;
    b->mapAccess = access;
    b->mapOffset = offset;
    b->mapLength = length;
    bool persistent = access & GL_MAP_PERSISTENT_BIT;
    if (persistent || (access & GL_MAP_UNSYNCHRONIZED_BIT)) {
        // Caller manages synchronization.
        if (access & GL_MAP_READ_BIT) syncBufferForCpuRead(ctx, b);
        b->mapPointer = b->cpu + offset;
        return b->mapPointer;
    }
    if ((access & GL_MAP_INVALIDATE_BUFFER_BIT) && !b->immutable && busy(ctx, b)) {
        // Orphan: fresh storage, contents undefined.
        GLsizeiptr sz = b->size;
        allocStorage(b, sz);
        b->mapPointer = b->cpu + offset;
        return b->mapPointer;
    }
    if (access & GL_MAP_READ_BIT) {
        // Reading: GPU writes must land; writing while GPU reads requires full sync.
        if (access & GL_MAP_WRITE_BIT) {
            if (busy(ctx, b)) ctx->waitForResource(b->lastUse);
        } else {
            syncBufferForCpuRead(ctx, b);
        }
        b->mapPointer = b->cpu + offset;
        return b->mapPointer;
    }
    // Write-only.
    if (!busy(ctx, b)) {
        b->mapPointer = b->cpu + offset;
        return b->mapPointer;
    }
    // Busy: stage writes in a shadow and apply on unmap/flush.
    free(b->mapShadow);
    b->mapShadow = (uint8_t*)malloc(length);
    if (!(access & (GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_INVALIDATE_BUFFER_BIT))) {
        syncBufferForCpuRead(ctx, b);
        memcpy(b->mapShadow, b->cpu + offset, length);
    }
    b->mapPointer = b->mapShadow;
    return b->mapPointer;
}

static GLboolean unmap(Context* ctx, Buffer* b) {
    if (!b->mapped) {
        ctx->setError(GL_INVALID_OPERATION);
        return GL_FALSE;
    }
    if (b->mapAccess & GL_MAP_WRITE_BIT) b->contentVersion++;
    if (b->mapShadow) {
        if (!(b->mapAccess & GL_MAP_FLUSH_EXPLICIT_BIT)) writeBuffer(ctx, b, b->mapOffset, b->mapLength, b->mapShadow);
        free(b->mapShadow);
        b->mapShadow = nullptr;
    }
    b->mapped = false;
    b->mapPointer = nullptr;
    b->mapAccess = 0;
    b->mapOffset = 0;
    b->mapLength = 0;
    return GL_TRUE;
}

static void flushRange(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr length) {
    if (!b->mapped || !(b->mapAccess & GL_MAP_FLUSH_EXPLICIT_BIT)) return ctx->setError(GL_INVALID_OPERATION);
    if (offset < 0 || length < 0 || offset + length > b->mapLength) return ctx->setError(GL_INVALID_VALUE);
    if (b->mapShadow) writeBuffer(ctx, b, b->mapOffset + offset, length, b->mapShadow + offset);
}

API void* APIENTRY glMapBufferRange(GLenum target, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    MGL_CTX_RET(nullptr);
    Buffer* b = boundBuffer(ctx, target);
    return b ? mapRange(ctx, b, offset, length, access) : nullptr;
}

API void* APIENTRY glMapNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length, GLbitfield access) {
    MGL_CTX_RET(nullptr);
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    return mapRange(ctx, b, offset, length, access);
}

static GLbitfield accessFromEnum(GLenum access) {
    switch (access) {
    case GL_READ_ONLY: return GL_MAP_READ_BIT;
    case GL_WRITE_ONLY: return GL_MAP_WRITE_BIT;
    case GL_READ_WRITE: return GL_MAP_READ_BIT | GL_MAP_WRITE_BIT;
    default: return 0;
    }
}

API void* APIENTRY glMapBuffer(GLenum target, GLenum access) {
    MGL_CTX_RET(nullptr);
    Buffer* b = boundBuffer(ctx, target);
    if (!b) return nullptr;
    GLbitfield a = accessFromEnum(access);
    if (!a) return ctx->setError(GL_INVALID_ENUM), nullptr;
    if (b->size == 0) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    void* p = mapRange(ctx, b, 0, b->size, a);
    return p;
}

API void* APIENTRY glMapNamedBuffer(GLuint buffer, GLenum access) {
    MGL_CTX_RET(nullptr);
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION), nullptr;
    GLbitfield a = accessFromEnum(access);
    if (!a) return ctx->setError(GL_INVALID_ENUM), nullptr;
    return mapRange(ctx, b, 0, b->size, a);
}

API GLboolean APIENTRY glUnmapBuffer(GLenum target) {
    MGL_CTX_RET(GL_FALSE);
    Buffer* b = boundBuffer(ctx, target);
    return b ? unmap(ctx, b) : GL_FALSE;
}

API GLboolean APIENTRY glUnmapNamedBuffer(GLuint buffer) {
    MGL_CTX_RET(GL_FALSE);
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION), GL_FALSE;
    return unmap(ctx, b);
}

API void APIENTRY glFlushMappedBufferRange(GLenum target, GLintptr offset, GLsizeiptr length) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) flushRange(ctx, b, offset, length);
}

API void APIENTRY glFlushMappedNamedBufferRange(GLuint buffer, GLintptr offset, GLsizeiptr length) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    flushRange(ctx, b, offset, length);
}

// --- copies / clears / reads ---
static void copySubData(Context* ctx, Buffer* src, Buffer* dst, GLintptr ro, GLintptr wo, GLsizeiptr size) {
    if (ro < 0 || wo < 0 || size < 0 || ro + size > src->size || wo + size > dst->size) return ctx->setError(GL_INVALID_VALUE);
    if ((src->mapped && !(src->mapAccess & GL_MAP_PERSISTENT_BIT)) || (dst->mapped && !(dst->mapAccess & GL_MAP_PERSISTENT_BIT)))
        return ctx->setError(GL_INVALID_OPERATION);
    if (src == dst && ((ro <= wo && wo < ro + size) || (wo <= ro && ro < wo + size))) return ctx->setError(GL_INVALID_VALUE);
    if (!size) return;
    dst->contentVersion++;
    bool srcGpuWrite = src->lastGpuWrite && !ctx->dev->isComplete(src->lastGpuWrite);
    if (src->cpu && dst->cpu && !srcGpuWrite && !busy(ctx, dst)) {
        memcpy(dst->cpu + wo, src->cpu + ro, size);
        return;
    }
    id<MTL4ComputeCommandEncoder> e = ctx->computeEncoder();
    [e copyFromBuffer:src->mtl sourceOffset:ro toBuffer:dst->mtl destinationOffset:wo size:size];
    [e barrierAfterEncoderStages:MTLStageBlit beforeEncoderStages:MTLStageBlit | MTLStageDispatch visibilityOptions:MTL4VisibilityOptionDevice];
    src->lastUse = dst->lastUse = ctx->serial();
    dst->lastGpuWrite = dst->lastUse;
}

API void APIENTRY glCopyBufferSubData(GLenum rt, GLenum wt, GLintptr ro, GLintptr wo, GLsizeiptr size) {
    MGL_CTX();
    Buffer* s = boundBuffer(ctx, rt);
    Buffer* d = boundBuffer(ctx, wt);
    if (s && d) copySubData(ctx, s, d, ro, wo, size);
}

API void APIENTRY glCopyNamedBufferSubData(GLuint rb, GLuint wb, GLintptr ro, GLintptr wo, GLsizeiptr size) {
    MGL_CTX();
    Buffer* s = lookupBuffer(ctx, rb);
    Buffer* d = lookupBuffer(ctx, wb);
    if (!s || !d) return ctx->setError(GL_INVALID_OPERATION);
    copySubData(ctx, s, d, ro, wo, size);
}

static void getSubData(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, void* data) {
    if (offset < 0 || size < 0 || offset + size > b->size) return ctx->setError(GL_INVALID_VALUE);
    if (b->mapped && !(b->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION);
    readBuffer(ctx, b, offset, size, data);
}

API void APIENTRY glGetBufferSubData(GLenum target, GLintptr offset, GLsizeiptr size, void* data) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) getSubData(ctx, b, offset, size, data);
}

API void APIENTRY glGetNamedBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr size, void* data) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    getSubData(ctx, b, offset, size, data);
}

void clearBufferRange(Context* ctx, Buffer* b, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format, GLenum type,
                      const void* data) {
    const FormatInfo* f = formatInfo(internalformat);
    if (!f || f->comp == Comp::Compressed || isDepthOrStencil(f)) return ctx->setError(GL_INVALID_ENUM);
    // Invalid format/type are INVALID_VALUE here (GL 4.6 6.2.1, as Mesa).
    if (pixelFormatTypeError(format, type)) return ctx->setError(GL_INVALID_VALUE, "clear buffer format/type");
    // Buffer clear uses the GL texel size of the internal format (no RGB padding).
    int texel = clientPixelSize(f->baseFormat == GL_RGB ? GL_RGB : f->baseFormat == GL_RGBA ? GL_RGBA : f->baseFormat == GL_RG ? GL_RG : GL_RED,
                                f->kind == FmtKind::Float ? (f->rBits == 16 ? GL_HALF_FLOAT : GL_FLOAT)
                                : f->kind == FmtKind::UInt || f->kind == FmtKind::Unorm ? (f->rBits == 8 ? GL_UNSIGNED_BYTE : f->rBits == 16 ? GL_UNSIGNED_SHORT : GL_UNSIGNED_INT)
                                : (f->rBits == 8 ? GL_BYTE : f->rBits == 16 ? GL_SHORT : GL_INT));
    if (texel <= 0) texel = f->bytesPerPixel;
    if (offset < 0 || size < 0 || offset + size > b->size) return ctx->setError(GL_INVALID_VALUE);
    if (offset % texel || size % texel) return ctx->setError(GL_INVALID_VALUE);
    if (b->mapped && !(b->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION);
    uint8_t pattern[16] = {};
    if (data) {
        // Convert the single client pixel to the storage format, then trim to the GL texel size.
        uint8_t tmp[16] = {};
        convertToStorage(f, format, type, data, 0, 0, tmp, 0, 0, 1, 1, 1, false);
        memcpy(pattern, tmp, std::min(texel, 16));
    }
    if (b->sparse) {
        // Only committed pages hold data (a clear of a mostly uncommitted 1 GB buffer must not
        // stage 1 GB): clear each committed run of the range.
        const GLsizeiptr page = (GLsizeiptr)ctx->dev->sparseTileBytes;
        GLsizeiptr end = offset + size;
        for (GLsizeiptr p = offset / page * page; p < end;) {
            if (!b->sparsePages[(size_t)(p / page)].count) {
                p += page;
                continue;
            }
            GLsizeiptr q = p;
            while (q < end && b->sparsePages[(size_t)(q / page)].count) q += page;
            GLsizeiptr a = std::max(p, offset), e = std::min(q, end);
            std::vector<uint8_t> fill((size_t)(e - a));
            for (GLsizeiptr i = a; i < e; i++) fill[(size_t)(i - a)] = pattern[(i - offset) % texel];
            writeBuffer(ctx, b, a, e - a, fill.data());
            p = q;
        }
        return;
    }
    std::vector<uint8_t> fill(size);
    for (GLsizeiptr i = 0; i < size; i += texel) memcpy(fill.data() + i, pattern, texel);
    writeBuffer(ctx, b, offset, size, fill.data());
}

API void APIENTRY glClearBufferData(GLenum target, GLenum internalformat, GLenum format, GLenum type, const void* data) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) clearBufferRange(ctx, b, internalformat, 0, b->size, format, type, data);
}

API void APIENTRY glClearBufferSubData(GLenum target, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format,
                                       GLenum type, const void* data) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    if (b) clearBufferRange(ctx, b, internalformat, offset, size, format, type, data);
}

API void APIENTRY glClearNamedBufferData(GLuint buffer, GLenum internalformat, GLenum format, GLenum type, const void* data) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    clearBufferRange(ctx, b, internalformat, 0, b->size, format, type, data);
}

API void APIENTRY glClearNamedBufferSubData(GLuint buffer, GLenum internalformat, GLintptr offset, GLsizeiptr size, GLenum format,
                                            GLenum type, const void* data) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    clearBufferRange(ctx, b, internalformat, offset, size, format, type, data);
}

API void APIENTRY glInvalidateBufferData(GLuint buffer) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_VALUE);
    if (b->mapped && !(b->mapAccess & GL_MAP_PERSISTENT_BIT)) return ctx->setError(GL_INVALID_OPERATION);
    if (!b->immutable && busy(ctx, b)) allocStorage(b, b->size);
}

API void APIENTRY glInvalidateBufferSubData(GLuint buffer, GLintptr offset, GLsizeiptr length) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_VALUE);
    if (offset < 0 || length < 0 || offset + length > b->size) return ctx->setError(GL_INVALID_VALUE);
}

// --- queries ---
static bool bufferParam(Context* ctx, Buffer* b, GLenum pname, int64_t* out) {
    switch (pname) {
    case GL_BUFFER_SIZE: *out = b->size; return true;
    case GL_BUFFER_USAGE: *out = b->usage; return true;
    case GL_BUFFER_ACCESS: {
        GLbitfield a = b->mapAccess & (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT);
        *out = a == (GL_MAP_READ_BIT | GL_MAP_WRITE_BIT) ? GL_READ_WRITE : a == GL_MAP_READ_BIT ? GL_READ_ONLY : a ? GL_WRITE_ONLY : GL_READ_WRITE;
        return true;
    }
    case GL_BUFFER_ACCESS_FLAGS: *out = b->mapAccess; return true;
    case GL_BUFFER_MAPPED: *out = b->mapped; return true;
    case GL_BUFFER_MAP_OFFSET: *out = b->mapOffset; return true;
    case GL_BUFFER_MAP_LENGTH: *out = b->mapLength; return true;
    case GL_BUFFER_IMMUTABLE_STORAGE: *out = b->immutable; return true;
    case GL_BUFFER_STORAGE_FLAGS: *out = b->storageFlags; return true;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
}

API void APIENTRY glGetBufferParameteriv(GLenum target, GLenum pname, GLint* params) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    int64_t v;
    if (b && bufferParam(ctx, b, pname, &v)) *params = (GLint)v;
}

API void APIENTRY glGetBufferParameteri64v(GLenum target, GLenum pname, GLint64* params) {
    MGL_CTX();
    Buffer* b = boundBuffer(ctx, target);
    int64_t v;
    if (b && bufferParam(ctx, b, pname, &v)) *params = v;
}

API void APIENTRY glGetNamedBufferParameteriv(GLuint buffer, GLenum pname, GLint* params) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    int64_t v;
    if (bufferParam(ctx, b, pname, &v)) *params = (GLint)v;
}

API void APIENTRY glGetNamedBufferParameteri64v(GLuint buffer, GLenum pname, GLint64* params) {
    MGL_CTX();
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    int64_t v;
    if (bufferParam(ctx, b, pname, &v)) *params = v;
}

API void APIENTRY glGetBufferPointerv(GLenum target, GLenum pname, void** params) {
    MGL_CTX();
    if (pname != GL_BUFFER_MAP_POINTER) return ctx->setError(GL_INVALID_ENUM);
    Buffer* b = boundBuffer(ctx, target);
    if (b) *params = b->mapPointer;
}

API void APIENTRY glGetNamedBufferPointerv(GLuint buffer, GLenum pname, void** params) {
    MGL_CTX();
    if (pname != GL_BUFFER_MAP_POINTER) return ctx->setError(GL_INVALID_ENUM);
    Buffer* b = lookupBuffer(ctx, buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    *params = b->mapPointer;
}

// EXT_transform_feedback: binds from `offset` to the end of the buffer.
API void APIENTRY glBindBufferOffsetEXT(GLenum target, GLuint index, GLuint buffer, GLintptr offset) {
    MGL_CTX();
    if (target != GL_TRANSFORM_FEEDBACK_BUFFER) return ctx->setError(GL_INVALID_ENUM);
    IndexedBufferBinding* ib;
    Ref<Buffer>* generic;
    if (!indexedTargetInfo(ctx, target, index, &ib, &generic)) return;
    if (offset < 0 || (offset % 4)) return ctx->setError(GL_INVALID_VALUE);
    Buffer* b = bufferForBind(ctx, buffer);
    if (b == (Buffer*)-1) return;
    *generic = b;
    auto& bb = ctx->xfb->buffers[index];
    bb.buffer = b;
    bb.offset = b ? offset : 0;
    bb.size = 0;
}
