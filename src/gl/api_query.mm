// Queries, sync objects, conditional rendering.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {
void writeBuffer(Context* ctx, Buffer* b, GLintptr offset, GLsizeiptr size, const void* data);
bool SyncObj::signaled() const { return Device::get()->isComplete(serial); }
}

static int queryTargetIndex(GLenum target) {
    switch (target) {
    case GL_SAMPLES_PASSED: return 0;
    case GL_ANY_SAMPLES_PASSED: return 1;
    case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: return 2;
    case GL_TIME_ELAPSED: return 3;
    case GL_PRIMITIVES_GENERATED: return 4;
    case GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN: return 5;
    case GL_TRANSFORM_FEEDBACK_OVERFLOW: return 6;
    case GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW: return 7;
    default: return -1;
    }
}

static bool isOcclusion(GLenum t) { return t == GL_SAMPLES_PASSED || t == GL_ANY_SAMPLES_PASSED || t == GL_ANY_SAMPLES_PASSED_CONSERVATIVE; }

static Query* activeOcclusion(Context* ctx) {
    for (int i = 0; i < 3; i++)
        if (ctx->activeQueries[i][0]) return ctx->activeQueries[i][0].get();
    return nullptr;
}

// Called before each draw: routes visibility counting to the active occlusion query.
void mglQueryNotifyDraw(Context* ctx, GLenum mode, uint64_t vertices, uint64_t instances) {
    Query* oq = activeOcclusion(ctx);
    id<MTL4RenderCommandEncoder> e = ctx->renderEnc;
    if (e) {
        int64_t want = oq ? (int64_t)oq->resultOffset : -1;
        if (oq && ctx->passVisibilityBuffer != oq->resultBuffer) want = -1;  // pass predates the query buffer
        if (ctx->encVisibilityOffset != want) {
            if (want >= 0) [e setVisibilityResultMode:MTLVisibilityResultModeCounting offset:(NSUInteger)want];
            else [e setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
            ctx->encVisibilityOffset = want;
        }
    }
    // Primitive counting on the CPU (no geometry amplification).
    auto prims = [&](GLenum m, uint64_t n) -> uint64_t {
        switch (m) {
        case GL_POINTS: return n;
        case GL_LINES: return n / 2;
        case GL_LINE_STRIP: return n > 1 ? n - 1 : 0;
        case GL_LINE_LOOP: return n > 1 ? n : 0;
        case GL_TRIANGLES: return n / 3;
        case GL_TRIANGLE_STRIP: case GL_TRIANGLE_FAN: case GL_POLYGON: return n > 2 ? n - 2 : 0;
        case GL_QUADS: return n / 4 * 2;
        case GL_LINES_ADJACENCY: return n / 4;
        case GL_LINE_STRIP_ADJACENCY: return n > 3 ? n - 3 : 0;
        case GL_TRIANGLES_ADJACENCY: return n / 6;
        case GL_TRIANGLE_STRIP_ADJACENCY: return n > 5 ? (n - 4) / 2 : 0;
        default: return 0;
        }
    };
    if (Query* pq = ctx->activeQueries[4][0].get()) pq->cpuCount += prims(mode, vertices) * instances;
}

// Ensures render passes started from now on carry the visibility buffer of `q`.
static void attachVisibility(Context* ctx, Query* q) {
    const uint32_t kSlots = 4096;
    if (!ctx->visibilityBuffer || ctx->visibilityNext >= kSlots) {
        ctx->visibilityBuffer = ctx->dev->newBuffer(kSlots * 8);
        memset(ctx->visibilityBuffer.contents, 0, kSlots * 8);
        ctx->visibilityNext = 0;
    }
    q->resultBuffer = ctx->visibilityBuffer;
    q->resultOffset = ctx->visibilityNext++ * 8;
    memset((uint8_t*)q->resultBuffer.contents + q->resultOffset, 0, 8);
    if (ctx->passOpen && ctx->passVisibilityBuffer != ctx->visibilityBuffer) {
        ctx->applyClearNow();
        ctx->endPass();
        ctx->passDirty = true;
    }
}

API void APIENTRY glGenQueries(GLsizei n, GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) ids[i] = ctx->queries.genName();
}

API void APIENTRY glCreateQueries(GLenum target, GLsizei n, GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    if (queryTargetIndex(target) < 0 && target != GL_TIMESTAMP) return ctx->setError(GL_INVALID_ENUM);
    for (int i = 0; i < n; i++) {
        ids[i] = ctx->queries.genName();
        Query* q = new Query();
        q->target = target;
        ctx->queries.create(ids[i], q);
    }
}

API void APIENTRY glDeleteQueries(GLsizei n, const GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        Query* q = ctx->queries.get(ids[i]);
        if (q && q->active) {
            int ti = queryTargetIndex(q->target);
            if (ti >= 0) ctx->activeQueries[ti][q->index] = nullptr;
        }
        ctx->queries.remove(ids[i]);
    }
}

API GLboolean APIENTRY glIsQuery(GLuint id) {
    MGL_CTX_RET(GL_FALSE);
    return id && ctx->queries.get(id) != nullptr;
}

static void beginQuery(Context* ctx, GLenum target, GLuint index, GLuint id) {
    int ti = queryTargetIndex(target);
    if (ti < 0) return ctx->setError(GL_INVALID_ENUM);
    int maxIndex = (target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ||
                    target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW) ? 4 : 1;
    if (index >= (GLuint)maxIndex) return ctx->setError(GL_INVALID_VALUE);
    if (ctx->activeQueries[ti][index]) return ctx->setError(GL_INVALID_OPERATION, "query already active");
    if (isOcclusion(target))
        for (int i = 0; i < 3; i++)
            if (ctx->activeQueries[i][0]) return ctx->setError(GL_INVALID_OPERATION);
    if (!id) return ctx->setError(GL_INVALID_OPERATION);
    Query* q = ctx->queries.get(id);
    if (!q) {
        if (!ctx->queries.isName(id)) return ctx->setError(GL_INVALID_OPERATION);
        q = ctx->queries.create(id, new Query());
        q->target = target;
    }
    if (q->active) return ctx->setError(GL_INVALID_OPERATION);
    if (q->target && q->target != target) return ctx->setError(GL_INVALID_OPERATION, "query target mismatch");
    q->target = target;
    q->index = index;
    q->active = true;
    q->everActive = true;
    q->resultAvailable = false;
    q->result = 0;
    q->cpuCount = 0;
    q->serial = 0;
    ctx->activeQueries[ti][index] = q;
    if (isOcclusion(target)) attachVisibility(ctx, q);
    if (target == GL_TIME_ELAPSED) {
        ctx->flush();
        q->beginSerial = ctx->serial();
    }
}

static void endQuery(Context* ctx, GLenum target, GLuint index) {
    int ti = queryTargetIndex(target);
    if (ti < 0) return ctx->setError(GL_INVALID_ENUM);
    if (index >= 4) return ctx->setError(GL_INVALID_VALUE);
    Ref<Query> q = ctx->activeQueries[ti][index];
    if (!q) return ctx->setError(GL_INVALID_OPERATION, "query not active");
    ctx->activeQueries[ti][index] = nullptr;
    q->active = false;
    q->serial = ctx->serial();
    if (target == GL_TIME_ELAPSED) ctx->flush();
    if (isOcclusion(target) && ctx->renderEnc && ctx->encVisibilityOffset >= 0) {
        [ctx->renderEnc setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        ctx->encVisibilityOffset = -1;
    }
    if (target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN || target == GL_TRANSFORM_FEEDBACK_OVERFLOW ||
        target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW) {
        q->result = q->cpuCount;
        if (target == GL_TRANSFORM_FEEDBACK_OVERFLOW || target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW) q->result = 0;
        q->resultAvailable = true;
    }
}

API void APIENTRY glBeginQuery(GLenum target, GLuint id) {
    MGL_CTX();
    beginQuery(ctx, target, 0, id);
}
API void APIENTRY glBeginQueryIndexed(GLenum target, GLuint index, GLuint id) {
    MGL_CTX();
    beginQuery(ctx, target, index, id);
}
API void APIENTRY glEndQuery(GLenum target) {
    MGL_CTX();
    endQuery(ctx, target, 0);
}
API void APIENTRY glEndQueryIndexed(GLenum target, GLuint index) {
    MGL_CTX();
    endQuery(ctx, target, index);
}

API void APIENTRY glQueryCounter(GLuint id, GLenum target) {
    MGL_CTX();
    if (target != GL_TIMESTAMP) return ctx->setError(GL_INVALID_ENUM);
    Query* q = ctx->queries.get(id);
    if (!q) {
        if (!ctx->queries.isName(id)) return ctx->setError(GL_INVALID_OPERATION);
        q = ctx->queries.create(id, new Query());
    }
    if (q->active || (q->target && q->target != GL_TIMESTAMP)) return ctx->setError(GL_INVALID_OPERATION);
    q->target = GL_TIMESTAMP;
    q->serial = ctx->serial();
    q->resultAvailable = false;
    q->everActive = true;
    ctx->flush();
}

// Resolves query results if the GPU has finished. `wait` blocks until done.
static bool resolveQuery(Context* ctx, Query* q, bool wait) {
    if (q->resultAvailable) return true;
    if (q->serial && !ctx->dev->isComplete(q->serial)) {
        if (!wait) {
            // Guarantee forward progress: make sure the command buffer is submitted.
            ctx->flush();
            return false;
        }
        ctx->waitForResource(q->serial);
    }
    switch (q->target) {
    case GL_SAMPLES_PASSED: case GL_ANY_SAMPLES_PASSED: case GL_ANY_SAMPLES_PASSED_CONSERVATIVE: {
        uint64_t v = 0;
        if (q->resultBuffer) memcpy(&v, (uint8_t*)q->resultBuffer.contents + q->resultOffset, 8);
        q->result = q->target == GL_SAMPLES_PASSED ? v : (v != 0);
        break;
    }
    case GL_TIME_ELAPSED: {
        double s0, e0, s1, e1;
        bool a = ctx->dev->gpuTimes(q->beginSerial, &s0, &e0);
        bool b = ctx->dev->gpuTimes(q->serial, &s1, &e1);
        if (a && b && e1 > s0) q->result = (uint64_t)((e1 - s0) * 1e9);
        else if (b) q->result = (uint64_t)((e1 - s1) * 1e9);
        else q->result = 0;
        break;
    }
    case GL_TIMESTAMP: {
        double s, e;
        q->result = ctx->dev->gpuTimes(q->serial, &s, &e) ? (uint64_t)(e * 1e9) : (uint64_t)(CACurrentMediaTime() * 1e9);
        break;
    }
    default: break;
    }
    q->resultAvailable = true;
    return true;
}

template <typename T>
static void getQueryObject(Context* ctx, GLuint id, GLenum pname, T* params) {
    Query* q = ctx->queries.get(id);
    if (!q || !q->everActive) return ctx->setError(GL_INVALID_OPERATION, "invalid query");
    if (q->active) return ctx->setError(GL_INVALID_OPERATION, "query active");
    // Results written into a bound GL_QUERY_BUFFER (offset in params).
    Buffer* qb = ctx->queryBuffer.get();
    T value = 0;
    switch (pname) {
    case GL_QUERY_RESULT: resolveQuery(ctx, q, true); value = (T)q->result; break;
    case GL_QUERY_RESULT_NO_WAIT:
        if (!resolveQuery(ctx, q, false)) return;
        value = (T)q->result;
        break;
    case GL_QUERY_RESULT_AVAILABLE: value = resolveQuery(ctx, q, false) ? 1 : 0; break;
    case GL_QUERY_TARGET: value = (T)q->target; break;
    default: return ctx->setError(GL_INVALID_ENUM);
    }
    if (qb) {
        size_t off = (size_t)params;
        if (off + sizeof(T) > (size_t)qb->size) return ctx->setError(GL_INVALID_OPERATION);
        mgl::writeBuffer(ctx, qb, off, sizeof(T), &value);
        return;
    }
    *params = value;
}

API void APIENTRY glGetQueryObjectiv(GLuint id, GLenum pname, GLint* params) { MGL_CTX(); getQueryObject(ctx, id, pname, params); }
API void APIENTRY glGetQueryObjectuiv(GLuint id, GLenum pname, GLuint* params) { MGL_CTX(); getQueryObject(ctx, id, pname, params); }
API void APIENTRY glGetQueryObjecti64v(GLuint id, GLenum pname, GLint64* params) { MGL_CTX(); getQueryObject(ctx, id, pname, params); }
API void APIENTRY glGetQueryObjectui64v(GLuint id, GLenum pname, GLuint64* params) { MGL_CTX(); getQueryObject(ctx, id, pname, params); }

template <typename T>
static void getQueryBufferObject(Context* ctx, GLuint id, GLuint buffer, GLenum pname, GLintptr offset) {
    Buffer* b = ctx->share->buffers.get(buffer);
    if (!b) return ctx->setError(GL_INVALID_OPERATION);
    Ref<Buffer> saved = ctx->queryBuffer;
    ctx->queryBuffer = b;
    getQueryObject(ctx, id, pname, (T*)offset);
    ctx->queryBuffer = saved;
}
API void APIENTRY glGetQueryBufferObjectiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) { MGL_CTX(); getQueryBufferObject<GLint>(ctx, id, buffer, pname, offset); }
API void APIENTRY glGetQueryBufferObjectuiv(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) { MGL_CTX(); getQueryBufferObject<GLuint>(ctx, id, buffer, pname, offset); }
API void APIENTRY glGetQueryBufferObjecti64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) { MGL_CTX(); getQueryBufferObject<GLint64>(ctx, id, buffer, pname, offset); }
API void APIENTRY glGetQueryBufferObjectui64v(GLuint id, GLuint buffer, GLenum pname, GLintptr offset) { MGL_CTX(); getQueryBufferObject<GLuint64>(ctx, id, buffer, pname, offset); }

static void getQueryiv(Context* ctx, GLenum target, GLuint index, GLenum pname, GLint* params) {
    if (target == GL_TIMESTAMP) {
        if (pname == GL_QUERY_COUNTER_BITS) *params = 64;
        else if (pname == GL_CURRENT_QUERY) *params = 0;
        else ctx->setError(GL_INVALID_ENUM);
        return;
    }
    int ti = queryTargetIndex(target);
    if (ti < 0) return ctx->setError(GL_INVALID_ENUM);
    if (index >= 4) return ctx->setError(GL_INVALID_VALUE);
    switch (pname) {
    case GL_CURRENT_QUERY: *params = ctx->activeQueries[ti][index] ? ctx->activeQueries[ti][index]->name : 0; break;
    case GL_QUERY_COUNTER_BITS: *params = target == GL_ANY_SAMPLES_PASSED || target == GL_ANY_SAMPLES_PASSED_CONSERVATIVE ? 1 : 64; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetQueryiv(GLenum target, GLenum pname, GLint* params) { MGL_CTX(); getQueryiv(ctx, target, 0, pname, params); }
API void APIENTRY glGetQueryIndexediv(GLenum target, GLuint index, GLenum pname, GLint* params) { MGL_CTX(); getQueryiv(ctx, target, index, pname, params); }

// --- conditional rendering ---
bool mglCheckDrawConditional(Context* ctx) { return ctx->conditionalRenderMode == 0 || ctx->conditionalPass; }

API void APIENTRY glBeginConditionalRender(GLuint id, GLenum mode) {
    MGL_CTX();
    if (ctx->conditionalRenderMode) return ctx->setError(GL_INVALID_OPERATION);
    switch (mode) {
    case GL_QUERY_WAIT: case GL_QUERY_NO_WAIT: case GL_QUERY_BY_REGION_WAIT: case GL_QUERY_BY_REGION_NO_WAIT:
    case GL_QUERY_WAIT_INVERTED: case GL_QUERY_NO_WAIT_INVERTED: case GL_QUERY_BY_REGION_WAIT_INVERTED: case GL_QUERY_BY_REGION_NO_WAIT_INVERTED:
        break;
    default: return ctx->setError(GL_INVALID_ENUM);
    }
    Query* q = ctx->queries.get(id);
    if (!q || q->active || !isOcclusion(q->target)) return ctx->setError(GL_INVALID_OPERATION);
    bool inverted = mode == GL_QUERY_WAIT_INVERTED || mode == GL_QUERY_NO_WAIT_INVERTED || mode == GL_QUERY_BY_REGION_WAIT_INVERTED ||
                    mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
    bool noWait = mode == GL_QUERY_NO_WAIT || mode == GL_QUERY_BY_REGION_NO_WAIT || mode == GL_QUERY_NO_WAIT_INVERTED ||
                  mode == GL_QUERY_BY_REGION_NO_WAIT_INVERTED;
    bool avail = resolveQuery(ctx, q, !noWait);
    bool passed = avail ? q->result != 0 : true;  // NO_WAIT without result: render
    ctx->conditionalPass = inverted && avail ? !passed : passed;
    ctx->conditionalRenderMode = mode;
    ctx->conditionalQuery = q;
}

API void APIENTRY glEndConditionalRender(void) {
    MGL_CTX();
    if (!ctx->conditionalRenderMode) return ctx->setError(GL_INVALID_OPERATION);
    ctx->conditionalRenderMode = 0;
    ctx->conditionalQuery = nullptr;
    ctx->conditionalPass = true;
}

// --- sync objects ---
API GLsync APIENTRY glFenceSync(GLenum condition, GLbitfield flags) {
    MGL_CTX_RET(nullptr);
    if (condition != GL_SYNC_GPU_COMMANDS_COMPLETE) return ctx->setError(GL_INVALID_ENUM), nullptr;
    if (flags) return ctx->setError(GL_INVALID_VALUE), nullptr;
    Ref<SyncObj> s = new SyncObj();
    s->serial = ctx->serial();
    ctx->flush();
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    ctx->share->syncs[s.get()] = s;
    return (GLsync)s.get();
}

static SyncObj* lookupSync(Context* ctx, GLsync sync) {
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    auto it = ctx->share->syncs.find((SyncObj*)sync);
    return it == ctx->share->syncs.end() ? nullptr : it->second.get();
}

API GLboolean APIENTRY glIsSync(GLsync sync) {
    MGL_CTX_RET(GL_FALSE);
    return lookupSync(ctx, sync) != nullptr;
}

API void APIENTRY glDeleteSync(GLsync sync) {
    MGL_CTX();
    if (!sync) return;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    auto it = ctx->share->syncs.find((SyncObj*)sync);
    if (it == ctx->share->syncs.end()) return ctx->setError(GL_INVALID_VALUE);
    ctx->share->syncs.erase(it);
}

API GLenum APIENTRY glClientWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MGL_CTX_RET(GL_WAIT_FAILED);
    SyncObj* s = lookupSync(ctx, sync);
    if (!s) {
        ctx->setError(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    if (flags & ~GL_SYNC_FLUSH_COMMANDS_BIT) {
        ctx->setError(GL_INVALID_VALUE);
        return GL_WAIT_FAILED;
    }
    if (s->signaled()) return GL_ALREADY_SIGNALED;
    if (timeout == 0) return GL_TIMEOUT_EXPIRED;
    ctx->waitForResource(s->serial);
    return s->signaled() ? GL_CONDITION_SATISFIED : GL_TIMEOUT_EXPIRED;
}

API void APIENTRY glWaitSync(GLsync sync, GLbitfield flags, GLuint64 timeout) {
    MGL_CTX();
    if (!lookupSync(ctx, sync)) return ctx->setError(GL_INVALID_VALUE);
    if (flags || timeout != GL_TIMEOUT_IGNORED) return ctx->setError(GL_INVALID_VALUE);
    // Single in-order queue: server waits are implicit.
}

API void APIENTRY glGetSynciv(GLsync sync, GLenum pname, GLsizei count, GLsizei* length, GLint* values) {
    MGL_CTX();
    SyncObj* s = lookupSync(ctx, sync);
    if (!s) return ctx->setError(GL_INVALID_VALUE);
    if (count < 1) {
        if (length) *length = 0;
        return;
    }
    GLint v;
    switch (pname) {
    case GL_OBJECT_TYPE: v = GL_SYNC_FENCE; break;
    case GL_SYNC_STATUS: v = s->signaled() ? GL_SIGNALED : GL_UNSIGNALED; break;
    case GL_SYNC_CONDITION: v = GL_SYNC_GPU_COMMANDS_COMPLETE; break;
    case GL_SYNC_FLAGS: v = 0; break;
    default: return ctx->setError(GL_INVALID_ENUM);
    }
    values[0] = v;
    if (length) *length = 1;
}
