// Apple's vendor extensions (GL_APPLE_*), as Apple's OpenGL exposes them, so code written against it runs
// unchanged, and GL_NV_fence, which works like GL_APPLE_fence. Most map onto the core features they were the
// precursors of.
#import "gl/context.h"
#import "gl/winsys.h"
#include <TargetConditionals.h>
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

// --- GL_APPLE_vertex_array_object -----------------------------------------------------------------------
// Gen/Delete/IsVertexArraysAPPLE are the core functions (gl.xml aliases). Binding differs: any name works,
// and binding one that GenVertexArraysAPPLE didn't return creates it (unlike ARB_vertex_array_object).
API void APIENTRY glBindVertexArrayAPPLE(GLuint array) {
    MGL_CTX();
    auto lk = ctx->containerLock();
    if (array && !ctx->vaos.isName(array)) ctx->vaos.reserve(array);
    glBindVertexArray(array);
}

// --- GL_APPLE_fence -------------------------------------------------------------------------------------
// Fences are command buffer serials, like sync objects (glFenceSync). Testing one submits the command
// buffer it waits on, so polling loops finish.
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
    case GL_FENCE_APPLE: {
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
    case GL_VERTEX_ARRAY:  // vertex array objects / ranges (also by their binding enum)
    case GL_VERTEX_ARRAY_BINDING_APPLE:
        if (name) {
            auto lk = ctx->containerLock();
            if (!ctx->vaos.get(name)) return false;
        }
        *serial = ctx->serial();  // everything recorded so far may read its arrays
        return true;
    case GL_DRAW_PIXELS_APPLE:
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
        ctx->setError(object == GL_FENCE_APPLE || object == GL_TEXTURE || object == GL_BUFFER || object == GL_VERTEX_ARRAY
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
        return ctx->setError(object == GL_FENCE_APPLE || object == GL_TEXTURE || object == GL_BUFFER || object == GL_VERTEX_ARRAY
                                 ? GL_INVALID_OPERATION
                                 : GL_INVALID_ENUM);
    ctx->waitForResource(serial);
}

// --- GL_APPLE_flush_render ------------------------------------------------------------------------------
// Not in the Khronos registry: exported here and listed in procs.cpp's kExtraProcs.
// FlushRender submits the work without presenting; SwapAPPLE presents the current drawable.
API void APIENTRY glFlushRenderAPPLE(void) { glFlush(); }

API void APIENTRY glFinishRenderAPPLE(void) { glFinish(); }

#if !TARGET_OS_IPHONE
extern "C" int CGLFlushDrawable(void* ctx);
extern "C" void* CGLGetCurrentContext(void);
#endif

API void APIENTRY glSwapAPPLE(void) {
    MGL_CTX();
#if !TARGET_OS_IPHONE
    if (void* cgl = CGLGetCurrentContext()) {
        CGLFlushDrawable(cgl);
        return;
    }
#endif
    Surface* s = ctx->drawSurface.get();
    if (s && s->layer) presentWindowSurface(ctx, s);
    else ctx->flush();
}

// --- GL_APPLE_object_purgeable --------------------------------------------------------------------------
// Metal's purgeable states: VOLATILE_APPLE lets the system reclaim the memory under pressure, RELEASED_APPLE
// discards it now; making the object unpurgeable again reports whether the contents survived.
static bool purgeableTarget(Context* ctx, GLenum objectType, GLuint name, id<MTLResource>* res, GLenum** flag) {
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    if (objectType == GL_BUFFER_OBJECT_APPLE) {
        Buffer* b = ctx->share->buffers.get(name);
        if (!b) return false;
        *res = b->mtl;
        *flag = &b->applePurgeable;
        return true;
    }
    if (objectType == GL_TEXTURE) {
        Texture* t = ctx->share->textures.get(name);
        if (!t) return false;
        Texture* root = t->viewParent ? t->viewParent.get() : t;
        *res = root->mtl;
        *flag = &root->applePurgeable;
        return true;
    }
    return false;
}

API GLenum APIENTRY glObjectPurgeableAPPLE(GLenum objectType, GLuint name, GLenum option) {
    MGL_CTX_RET(0);
    if (objectType != GL_BUFFER_OBJECT_APPLE && objectType != GL_TEXTURE) return ctx->setError(GL_INVALID_ENUM), 0;
    if (option != GL_VOLATILE_APPLE && option != GL_RELEASED_APPLE) return ctx->setError(GL_INVALID_ENUM), 0;
    id<MTLResource> res = nil;
    GLenum* flag = nullptr;
    if (!name || !purgeableTarget(ctx, objectType, name, &res, &flag)) return ctx->setError(GL_INVALID_VALUE), 0;
    if (*flag) return ctx->setError(GL_INVALID_OPERATION), 0;  // already purgeable
    *flag = option;
    if (!res) return GL_RELEASED_APPLE;  // no storage: nothing to keep
    if (option == GL_RELEASED_APPLE) {
        // Commands still using the memory must finish before it goes.
        ctx->finish();
        [res setPurgeableState:MTLPurgeableStateEmpty];
        return GL_RELEASED_APPLE;
    }
    [res setPurgeableState:MTLPurgeableStateVolatile];
    return GL_VOLATILE_APPLE;
}

API GLenum APIENTRY glObjectUnpurgeableAPPLE(GLenum objectType, GLuint name, GLenum option) {
    MGL_CTX_RET(0);
    if (objectType != GL_BUFFER_OBJECT_APPLE && objectType != GL_TEXTURE) return ctx->setError(GL_INVALID_ENUM), 0;
    if (option != GL_RETAINED_APPLE && option != GL_UNDEFINED_APPLE) return ctx->setError(GL_INVALID_ENUM), 0;
    id<MTLResource> res = nil;
    GLenum* flag = nullptr;
    if (!name || !purgeableTarget(ctx, objectType, name, &res, &flag)) return ctx->setError(GL_INVALID_VALUE), 0;
    if (!*flag) return ctx->setError(GL_INVALID_OPERATION), 0;  // not purgeable
    GLenum was = *flag;
    *flag = 0;
    if (!res) return GL_UNDEFINED_APPLE;
    MTLPurgeableState prev = [res setPurgeableState:MTLPurgeableStateNonVolatile];
    // Released contents count as gone even when Metal kept the pages.
    return was == GL_RELEASED_APPLE || prev == MTLPurgeableStateEmpty ? GL_UNDEFINED_APPLE : GL_RETAINED_APPLE;
}

API void APIENTRY glGetObjectParameterivAPPLE(GLenum objectType, GLuint name, GLenum pname, GLint* params) {
    MGL_CTX();
    if (objectType != GL_BUFFER_OBJECT_APPLE && objectType != GL_TEXTURE) return ctx->setError(GL_INVALID_ENUM);
    if (pname != GL_PURGEABLE_APPLE) return ctx->setError(GL_INVALID_ENUM);
    id<MTLResource> res = nil;
    GLenum* flag = nullptr;
    if (!name || !purgeableTarget(ctx, objectType, name, &res, &flag)) return ctx->setError(GL_INVALID_VALUE);
    *params = *flag ? GL_TRUE : GL_FALSE;
}

// --- GL_NV_fence: GL_APPLE_fence with a completion condition (only ALL_COMPLETED_NV) and status queries ---
API void APIENTRY glGenFencesNV(GLsizei n, GLuint* fences) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    GLuint next = 1;
    for (int i = 0; i < n; i++) {
        while (ctx->nvFences.count(next)) next++;
        ctx->nvFences[next] = 0;
        fences[i] = next;
    }
}

API void APIENTRY glDeleteFencesNV(GLsizei n, const GLuint* fences) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) ctx->nvFences.erase(fences[i]);
}

API GLboolean APIENTRY glIsFenceNV(GLuint fence) {
    MGL_CTX_RET(GL_FALSE);
    auto it = ctx->nvFences.find(fence);
    return it != ctx->nvFences.end() && it->second != 0;
}

API void APIENTRY glSetFenceNV(GLuint fence, GLenum condition) {
    MGL_CTX();
    if (condition != GL_ALL_COMPLETED_NV) return ctx->setError(GL_INVALID_ENUM);
    if (!fence) return ctx->setError(GL_INVALID_OPERATION);
    ctx->nvFences[fence] = ctx->serial();
}

static uint64_t* nvFence(Context* ctx, GLuint fence) {
    auto it = ctx->nvFences.find(fence);
    if (it == ctx->nvFences.end() || !it->second) return ctx->setError(GL_INVALID_OPERATION), nullptr;  // not set
    return &it->second;
}

API GLboolean APIENTRY glTestFenceNV(GLuint fence) {
    MGL_CTX_RET(GL_TRUE);
    uint64_t* s = nvFence(ctx, fence);
    return s ? testSerial(ctx, *s) : GL_TRUE;
}

API void APIENTRY glFinishFenceNV(GLuint fence) {
    MGL_CTX();
    if (uint64_t* s = nvFence(ctx, fence)) ctx->waitForResource(*s);
}

API void APIENTRY glGetFenceivNV(GLuint fence, GLenum pname, GLint* params) {
    MGL_CTX();
    uint64_t* s = nvFence(ctx, fence);
    if (!s) return;
    switch (pname) {
    case GL_FENCE_STATUS_NV: *params = testSerial(ctx, *s); break;
    case GL_FENCE_CONDITION_NV: *params = GL_ALL_COMPLETED_NV; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
