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
    // ARB_pipeline_statistics_query
    case GL_VERTICES_SUBMITTED: return 8;
    case GL_PRIMITIVES_SUBMITTED: return 9;
    case GL_VERTEX_SHADER_INVOCATIONS: return 10;
    case GL_TESS_CONTROL_SHADER_PATCHES: return 11;
    case GL_TESS_EVALUATION_SHADER_INVOCATIONS: return 12;
    case GL_GEOMETRY_SHADER_INVOCATIONS: return 13;
    case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED: return 14;
    case GL_FRAGMENT_SHADER_INVOCATIONS: return 15;
    case GL_COMPUTE_SHADER_INVOCATIONS: return 16;
    case GL_CLIPPING_INPUT_PRIMITIVES: return 17;
    case GL_CLIPPING_OUTPUT_PRIMITIVES: return 18;
    default: return -1;
    }
}

static bool isStatistics(GLenum t) { int i = queryTargetIndex(t); return i >= 8 && i <= 18; }

static bool isOcclusion(GLenum t) { return t == GL_SAMPLES_PASSED || t == GL_ANY_SAMPLES_PASSED || t == GL_ANY_SAMPLES_PASSED_CONSERVATIVE; }

static Query* activeOcclusion(Context* ctx) {
    for (int i = 0; i < 3; i++)
        if (ctx->activeQueries[i][0]) return ctx->activeQueries[i][0].get();
    return nullptr;
}

// The query fed by the visibility counter of the current render pass: an occlusion query,
// else an active FRAGMENT_SHADER_INVOCATIONS query (samples reaching the fragment stage).
static Query* activeVisibilityQuery(Context* ctx) {
    if (Query* q = activeOcclusion(ctx)) return q;
    Query* f = ctx->activeQueries[15][0].get();
    return f && f->resultBuffer ? f : nullptr;
}

// Called before each draw: routes visibility counting to the active occlusion query.
void mglQueryNotifyDraw(Context* ctx, GLenum mode, uint64_t vertices, uint64_t instances) {
    Query* oq = activeVisibilityQuery(ctx);
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
}

// Primitives a draw of `n` vertices in mode `m` assembles (no geometry amplification).
uint64_t mglPrimitivesFor(GLenum m, uint64_t n) {
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
    return prims(m, n);
}

// Primitive query bookkeeping (all counts are known on the CPU, see draw.mm):
// PRIMITIVES_GENERATED / TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN per vertex stream, and the
// overflow queries (any stream / one stream) while transform feedback is active.
void mglQueryAddPrimitives(Context* ctx, int stream, uint64_t generated, uint64_t written, bool xfbActive, bool overflow) {
    if (stream < 0 || stream > 3) return;
    if (Query* q = ctx->activeQueries[4][stream].get()) q->cpuCount += generated;
    if (!xfbActive) return;
    if (Query* q = ctx->activeQueries[5][stream].get()) q->cpuCount += written;
    if (overflow) {
        if (Query* q = ctx->activeQueries[6][0].get()) q->cpuCount = 1;
        if (Query* q = ctx->activeQueries[7][stream].get()) q->cpuCount = 1;
    }
}

bool mglQueryPrimitivesActive(Context* ctx) {
    for (int ti = 4; ti < 8; ti++)
        for (int i = 0; i < 4; i++)
            if (ctx->activeQueries[ti][i]) return true;
    return false;
}

// ARB_pipeline_statistics_query: per-draw counts computed by the draw code.
// `gsPrims`/`tesInv`/`tessPrims` are only known on the GPU; mglStatsGpuSlot hands out the
// counters the multi-pass shaders accumulate them into.
void mgl::mglStatsCountDraw(Context* ctx, const DrawStats& st) {
    auto add = [&](int ti, uint64_t v) {
        if (Query* q = ctx->activeQueries[ti][0].get()) q->cpuCount += v;
    };
    add(8, st.vertices);
    add(9, st.primitives);
    add(10, st.vsInvocations);
    add(11, st.patches);
    add(12, st.tesInvocations);
    add(13, st.gsInvocations);
    add(14, st.gsPrimitives);
    add(16, st.csInvocations);
    add(17, st.clipPrimitives);
    add(18, st.clipPrimitives);
    // fragment invocations without the visibility counter: at least one per rasterized primitive
    if (Query* f = ctx->activeQueries[15][0].get())
        if (activeVisibilityQuery(ctx) != f) f->cpuCount += st.fsEstimate;
}

uint64_t mgl::mglStatsGpuSlot(Context* ctx, bool gs, bool tess) {
    Query* gsq = ctx->activeQueries[14][0].get();
    Query* tesq = ctx->activeQueries[12][0].get();
    Query* c1 = ctx->activeQueries[17][0].get();
    Query* c2 = ctx->activeQueries[18][0].get();
    if (!(gs && (gsq || c1 || c2)) && !(tess && (tesq || ((c1 || c2) && !gs)))) return 0;
    const uint32_t kSlots = 4096, kSlotBytes = 16;
    if (!ctx->statsSlots || ctx->statsNext >= kSlots) {
        ctx->statsSlots = ctx->dev->newBuffer(kSlots * kSlotBytes);
        memset(bufferContents(ctx->statsSlots), 0, kSlots * kSlotBytes);
        ctx->statsNext = 0;
    }
    uint32_t off = ctx->statsNext++ * kSlotBytes;
    id<MTLBuffer> b = ctx->statsSlots;
    memset((uint8_t*)bufferContents(b) + off, 0, kSlotBytes);
    // [0] GS primitives emitted, [1] TES invocations, [2] tessellated primitives
    if (gs && gsq) gsq->gpuCounts.push_back({b, off});
    if (tess && tesq) tesq->gpuCounts.push_back({b, off + 4});
    for (Query* c : {c1, c2})
        if (c) c->gpuCounts.push_back({b, gs ? off : off + 8});
    return b.gpuAddress + off;
}

// Ensures render passes started from now on carry the visibility buffer of `q`.
static void attachVisibility(Context* ctx, Query* q) {
    const uint32_t kSlots = 4096;
    if (!ctx->visibilityBuffer || ctx->visibilityNext >= kSlots) {
        ctx->visibilityBuffer = ctx->dev->newBuffer(kSlots * 8);
        memset(bufferContents(ctx->visibilityBuffer), 0, kSlots * 8);
        ctx->visibilityNext = 0;
    }
    q->resultBuffer = ctx->visibilityBuffer;
    q->resultOffset = ctx->visibilityNext++ * 8;
    memset((uint8_t*)bufferContents(q->resultBuffer) + q->resultOffset, 0, 8);
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
            if (isStatistics(q->target)) ctx->statsQueriesActive--;
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
    q->gpuCounts.clear();
    q->resultBuffer = nil;
    if (isOcclusion(target)) attachVisibility(ctx, q);
    if (isStatistics(target)) {
        ctx->statsQueriesActive++;
        // Fragment invocations are sampled with the visibility counter unless an occlusion
        // query owns it (draws then fall back to a CPU estimate, see mglStatsCountDraw).
        if (target == GL_FRAGMENT_SHADER_INVOCATIONS && !activeOcclusion(ctx)) attachVisibility(ctx, q);
    }
    if (target == GL_TIME_ELAPSED) {
        ctx->flush();
        q->beginSerial = ctx->serial();
    }
}

static void endQuery(Context* ctx, GLenum target, GLuint index) {
    int ti = queryTargetIndex(target);
    if (ti < 0) return ctx->setError(GL_INVALID_ENUM);
    bool indexedT = target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN || target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW;
    if (index >= (indexedT ? 4u : 1u)) return ctx->setError(GL_INVALID_VALUE);
    Ref<Query> q = ctx->activeQueries[ti][index];
    if (!q) return ctx->setError(GL_INVALID_OPERATION, "query not active");
    bool routed = activeVisibilityQuery(ctx) == q.get();
    ctx->activeQueries[ti][index] = nullptr;
    q->active = false;
    q->serial = ctx->serial();
    if (target == GL_TIME_ELAPSED) ctx->flush();
    if (isStatistics(target)) {
        ctx->statsQueriesActive--;
        if (!q->resultBuffer && q->gpuCounts.empty()) {
            q->result = q->cpuCount;
            q->resultAvailable = true;
        }
    }
    if (routed && ctx->renderEnc && ctx->encVisibilityOffset >= 0) {
        [ctx->renderEnc setVisibilityResultMode:MTLVisibilityResultModeDisabled offset:0];
        ctx->encVisibilityOffset = -1;
    }
    if (target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN || target == GL_TRANSFORM_FEEDBACK_OVERFLOW ||
        target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW) {
        q->result = q->cpuCount;
        if (target == GL_TRANSFORM_FEEDBACK_OVERFLOW || target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW) q->result = q->cpuCount != 0;
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
        if (q->resultBuffer) memcpy(&v, (uint8_t*)bufferContents(q->resultBuffer) + q->resultOffset, 8);
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
    case GL_VERTICES_SUBMITTED: case GL_PRIMITIVES_SUBMITTED: case GL_VERTEX_SHADER_INVOCATIONS:
    case GL_TESS_CONTROL_SHADER_PATCHES: case GL_TESS_EVALUATION_SHADER_INVOCATIONS: case GL_GEOMETRY_SHADER_INVOCATIONS:
    case GL_GEOMETRY_SHADER_PRIMITIVES_EMITTED: case GL_FRAGMENT_SHADER_INVOCATIONS: case GL_COMPUTE_SHADER_INVOCATIONS:
    case GL_CLIPPING_INPUT_PRIMITIVES: case GL_CLIPPING_OUTPUT_PRIMITIVES: {
        uint64_t v = q->cpuCount;
        if (q->resultBuffer) {
            uint64_t samples = 0;
            memcpy(&samples, (uint8_t*)bufferContents(q->resultBuffer) + q->resultOffset, 8);
            v += samples;
        }
        for (auto& [buf, off] : q->gpuCounts) {
            uint32_t c = 0;
            memcpy(&c, (uint8_t*)bufferContents(buf) + off, 4);
            v += c;
        }
        q->gpuCounts.clear();
        q->result = v;
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
    if (pname != GL_QUERY_RESULT && pname != GL_QUERY_RESULT_NO_WAIT && pname != GL_QUERY_RESULT_AVAILABLE && pname != GL_QUERY_TARGET)
        return ctx->setError(GL_INVALID_ENUM);
    Query* q = ctx->queries.get(id);
    if (!q || (!q->everActive && !q->target))
        return ctx->setError(GL_INVALID_OPERATION, "query %u %s", id, !q ? "does not exist in this context" : "was never begun");
    if (q->active && pname != GL_QUERY_TARGET) return ctx->setError(GL_INVALID_OPERATION, "query active");
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
    if (offset < 0) return ctx->setError(GL_INVALID_VALUE);
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
    bool indexed = target == GL_PRIMITIVES_GENERATED || target == GL_TRANSFORM_FEEDBACK_PRIMITIVES_WRITTEN ||
                   target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW;
    if (index >= (indexed ? 4u : 1u)) return ctx->setError(GL_INVALID_VALUE);
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
    bool overflowQ = q && (q->target == GL_TRANSFORM_FEEDBACK_OVERFLOW || q->target == GL_TRANSFORM_FEEDBACK_STREAM_OVERFLOW);
    if (!q || q->active || !(isOcclusion(q->target) || overflowQ)) return ctx->setError(GL_INVALID_OPERATION);
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
    // No flush here: the fence completes with the command buffer being recorded,
    // which is submitted by a later flush (glClientWaitSync with
    // SYNC_FLUSH_COMMANDS_BIT / a timeout, glFlush, or the next swap).
    s->serial = ctx->serial();
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
    if ((flags & GL_SYNC_FLUSH_COMMANDS_BIT) || timeout > 0) ctx->flushIfPending(s->serial);
    if (timeout == 0) return s->signaled() ? GL_ALREADY_SIGNALED : GL_TIMEOUT_EXPIRED;
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
