// Apple's vendor extensions (GL_APPLE_*), as Apple's OpenGL exposes them, so code written against it runs
// unchanged. Most map onto the core features they were the precursors of.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

// --- GL_APPLE_vertex_array_object -----------------------------------------------------------------------
// Gen/Delete/IsVertexArraysAPPLE are the core functions (gl.xml aliases). Binding differs: any name works,
// and binding one that GenVertexArraysAPPLE didn't return creates it (unlike ARB_vertex_array_object).
API void APIENTRY glBindVertexArrayAPPLE(GLuint array) {
    MGL_CTX();
    if (array && !ctx->vaos.isName(array)) ctx->vaos.reserve(array);
    glBindVertexArray(array);
}

// --- GL_APPLE_fence -------------------------------------------------------------------------------------
// Fences are command buffer serials, like sync objects (glFenceSync). Testing one submits the command
// buffer it waits on, so polling loops finish.
static constexpr GLenum GL_FENCE_APPLE_ = 0x8A0B, GL_DRAW_PIXELS_APPLE_ = 0x8A0A, GL_VERTEX_ARRAY_APPLE_ = 0x85B5;

API void APIENTRY glGenFencesAPPLE(GLsizei n, GLuint* fences) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    GLuint next = 1;
    for (int i = 0; i < n; i++) {
        while (ctx->appleFences.count(next)) next++;
        ctx->appleFences[next] = 0;
        fences[i] = next;
    }
}

API void APIENTRY glDeleteFencesAPPLE(GLsizei n, const GLuint* fences) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) ctx->appleFences.erase(fences[i]);
}

API GLboolean APIENTRY glIsFenceAPPLE(GLuint fence) {
    MGL_CTX_RET(GL_FALSE);
    auto it = ctx->appleFences.find(fence);
    return it != ctx->appleFences.end() && it->second != 0;  // generated but never set isn't a fence yet
}

API void APIENTRY glSetFenceAPPLE(GLuint fence) {
    MGL_CTX();
    if (!fence) return ctx->setError(GL_INVALID_OPERATION);
    ctx->appleFences[fence] = ctx->serial();  // SetFence on an unused name also makes it a fence
}

static bool testSerial(Context* ctx, uint64_t serial) {
    if (!serial || ctx->dev->isComplete(serial)) return true;
    ctx->flushIfPending(serial);
    return ctx->dev->isComplete(serial);
}

// The serial of the last commands that use object `name` of `object`'s kind; 0 when nothing is pending.
static bool objectSerial(Context* ctx, GLenum object, GLuint name, uint64_t* serial) {
    switch (object) {
    case GL_FENCE_APPLE_: {
        auto it = ctx->appleFences.find(name);
        if (it == ctx->appleFences.end()) return false;
        *serial = it->second;
        return true;
    }
    case GL_TEXTURE: {
        std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
        Texture* t = ctx->share->textures.get(name);
        if (!t) return false;
        *serial = std::max(t->lastUse, t->lastGpuWrite);
        return true;
    }
    case GL_BUFFER: {
        std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
        Buffer* b = ctx->share->buffers.get(name);
        if (!b) return false;
        *serial = std::max(b->lastUse, b->lastGpuWrite);
        return true;
    }
    case GL_VERTEX_ARRAY_APPLE_:
        if (name && !ctx->vaos.get(name)) return false;
        *serial = ctx->serial();  // everything recorded so far may read its arrays
        return true;
    case GL_DRAW_PIXELS_APPLE_:
        *serial = ctx->serial();
        return true;
    default:
        return false;
    }
}

API GLboolean APIENTRY glTestFenceAPPLE(GLuint fence) {
    MGL_CTX_RET(GL_TRUE);
    auto it = ctx->appleFences.find(fence);
    if (it == ctx->appleFences.end()) return ctx->setError(GL_INVALID_OPERATION), GL_TRUE;
    return testSerial(ctx, it->second);
}

API void APIENTRY glFinishFenceAPPLE(GLuint fence) {
    MGL_CTX();
    auto it = ctx->appleFences.find(fence);
    if (it == ctx->appleFences.end()) return ctx->setError(GL_INVALID_OPERATION);
    ctx->waitForResource(it->second);
}

API GLboolean APIENTRY glTestObjectAPPLE(GLenum object, GLuint name) {
    MGL_CTX_RET(GL_TRUE);
    uint64_t serial = 0;
    if (!objectSerial(ctx, object, name, &serial)) {
        ctx->setError(object == GL_FENCE_APPLE_ || object == GL_TEXTURE || object == GL_BUFFER || object == GL_VERTEX_ARRAY_APPLE_
                          ? GL_INVALID_OPERATION
                          : GL_INVALID_ENUM);
        return GL_TRUE;
    }
    return testSerial(ctx, serial);
}

API void APIENTRY glFinishObjectAPPLE(GLenum object, GLint name) {
    MGL_CTX();
    uint64_t serial = 0;
    if (!objectSerial(ctx, object, (GLuint)name, &serial))
        return ctx->setError(object == GL_FENCE_APPLE_ || object == GL_TEXTURE || object == GL_BUFFER || object == GL_VERTEX_ARRAY_APPLE_
                                 ? GL_INVALID_OPERATION
                                 : GL_INVALID_ENUM);
    ctx->waitForResource(serial);
}
