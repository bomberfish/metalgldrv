// KHR_debug, object labels, internal format queries, transform feedback objects.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

// ---------------------------------------------------------------------------
// KHR_debug
// ---------------------------------------------------------------------------
static bool validSource(GLenum s, bool allowDontCare) {
    switch (s) {
    case GL_DEBUG_SOURCE_API: case GL_DEBUG_SOURCE_WINDOW_SYSTEM: case GL_DEBUG_SOURCE_SHADER_COMPILER: case GL_DEBUG_SOURCE_THIRD_PARTY:
    case GL_DEBUG_SOURCE_APPLICATION: case GL_DEBUG_SOURCE_OTHER:
        return true;
    case GL_DONT_CARE: return allowDontCare;
    default: return false;
    }
}
static bool validType(GLenum t, bool allowDontCare) {
    switch (t) {
    case GL_DEBUG_TYPE_ERROR: case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR: case GL_DEBUG_TYPE_PORTABILITY:
    case GL_DEBUG_TYPE_PERFORMANCE: case GL_DEBUG_TYPE_OTHER: case GL_DEBUG_TYPE_MARKER: case GL_DEBUG_TYPE_PUSH_GROUP:
    case GL_DEBUG_TYPE_POP_GROUP:
        return true;
    case GL_DONT_CARE: return allowDontCare;
    default: return false;
    }
}
static bool validSeverity(GLenum s, bool allowDontCare) {
    switch (s) {
    case GL_DEBUG_SEVERITY_HIGH: case GL_DEBUG_SEVERITY_MEDIUM: case GL_DEBUG_SEVERITY_LOW: case GL_DEBUG_SEVERITY_NOTIFICATION: return true;
    case GL_DONT_CARE: return allowDontCare;
    default: return false;
    }
}

API void APIENTRY glDebugMessageCallback(GLDEBUGPROC callback, const void* userParam) {
    MGL_CTX();
    ctx->debugCallback = callback;
    ctx->debugUserParam = userParam;
}
API void APIENTRY glDebugMessageCallbackARB(GLDEBUGPROCARB callback, const void* userParam) { glDebugMessageCallback((GLDEBUGPROC)callback, userParam); }
API void APIENTRY glDebugMessageCallbackKHR(GLDEBUGPROCKHR callback, const void* userParam) { glDebugMessageCallback((GLDEBUGPROC)callback, userParam); }

API void APIENTRY glDebugMessageControl(GLenum source, GLenum type, GLenum severity, GLsizei count, const GLuint* ids, GLboolean enabled) {
    MGL_CTX();
    if (!validSource(source, true) || !validType(type, true) || !validSeverity(severity, true)) return ctx->setError(GL_INVALID_ENUM);
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if (count > 0 && (source == GL_DONT_CARE || type == GL_DONT_CARE || severity != GL_DONT_CARE)) return ctx->setError(GL_INVALID_OPERATION);
    if (count > 0) {
        for (int i = 0; i < count; i++) ctx->debugIdControl[(uint64_t)source << 40 | (uint64_t)type << 32 | ids[i]] = enabled;
        return;
    }
    // Later rules override earlier ones; also clear id rules that this matches.
    ctx->debugControl[(uint64_t)source << 40 | (uint64_t)type << 20 | severity] = enabled;
    for (auto it = ctx->debugIdControl.begin(); it != ctx->debugIdControl.end();) {
        GLenum ks = (GLenum)(it->first >> 40), kt = (GLenum)((it->first >> 32) & 0xff);
        (void)kt;
        if (source == GL_DONT_CARE || ks == source) it = ctx->debugIdControl.erase(it);
        else ++it;
    }
}
API void APIENTRY glDebugMessageControlARB(GLenum s, GLenum t, GLenum v, GLsizei c, const GLuint* i, GLboolean e) { glDebugMessageControl(s, t, v, c, i, e); }
API void APIENTRY glDebugMessageControlKHR(GLenum s, GLenum t, GLenum v, GLsizei c, const GLuint* i, GLboolean e) { glDebugMessageControl(s, t, v, c, i, e); }

API void APIENTRY glDebugMessageInsert(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* buf) {
    MGL_CTX();
    if ((source != GL_DEBUG_SOURCE_APPLICATION && source != GL_DEBUG_SOURCE_THIRD_PARTY) || !validType(type, false) ||
        !validSeverity(severity, false))
        return ctx->setError(GL_INVALID_ENUM);
    std::string msg = length < 0 ? std::string(buf) : std::string(buf, length);
    if (msg.size() >= 1024) return ctx->setError(GL_INVALID_VALUE);
    ctx->debugMessage(source, type, id, severity, msg);
}
API void APIENTRY glDebugMessageInsertARB(GLenum s, GLenum t, GLuint i, GLenum v, GLsizei l, const GLchar* b) { glDebugMessageInsert(s, t, i, v, l, b); }
API void APIENTRY glDebugMessageInsertKHR(GLenum s, GLenum t, GLuint i, GLenum v, GLsizei l, const GLchar* b) { glDebugMessageInsert(s, t, i, v, l, b); }

API GLuint APIENTRY glGetDebugMessageLog(GLuint count, GLsizei bufSize, GLenum* sources, GLenum* types, GLuint* ids, GLenum* severities,
                                         GLsizei* lengths, GLchar* messageLog) {
    MGL_CTX_RET(0);
    if (bufSize < 0 && messageLog) {
        ctx->setError(GL_INVALID_VALUE);
        return 0;
    }
    GLuint n = 0;
    size_t used = 0;
    while (n < count && !ctx->debugLog.empty()) {
        DebugMessage& m = ctx->debugLog.front();
        size_t need = m.text.size() + 1;
        if (messageLog) {
            if (used + need > (size_t)bufSize) break;
            memcpy(messageLog + used, m.text.c_str(), need);
            used += need;
        }
        if (sources) sources[n] = m.source;
        if (types) types[n] = m.type;
        if (ids) ids[n] = m.id;
        if (severities) severities[n] = m.severity;
        if (lengths) lengths[n] = (GLsizei)need;
        ctx->debugLog.erase(ctx->debugLog.begin());
        n++;
    }
    return n;
}
API GLuint APIENTRY glGetDebugMessageLogARB(GLuint c, GLsizei b, GLenum* s, GLenum* t, GLuint* i, GLenum* v, GLsizei* l, GLchar* m) { return glGetDebugMessageLog(c, b, s, t, i, v, l, m); }
API GLuint APIENTRY glGetDebugMessageLogKHR(GLuint c, GLsizei b, GLenum* s, GLenum* t, GLuint* i, GLenum* v, GLsizei* l, GLchar* m) { return glGetDebugMessageLog(c, b, s, t, i, v, l, m); }

API void APIENTRY glPushDebugGroup(GLenum source, GLuint id, GLsizei length, const GLchar* message) {
    MGL_CTX();
    if (source != GL_DEBUG_SOURCE_APPLICATION && source != GL_DEBUG_SOURCE_THIRD_PARTY) return ctx->setError(GL_INVALID_ENUM);
    if (ctx->debugGroups.size() >= 63) return ctx->setError(GL_STACK_OVERFLOW);
    std::string msg = length < 0 ? std::string(message) : std::string(message, length);
    if (msg.size() >= 1024) return ctx->setError(GL_INVALID_VALUE);
    ctx->debugGroups.push_back({source, id, msg});
    ctx->debugMessage(source, GL_DEBUG_TYPE_PUSH_GROUP, id, GL_DEBUG_SEVERITY_NOTIFICATION, msg);
}
API void APIENTRY glPushDebugGroupKHR(GLenum s, GLuint i, GLsizei l, const GLchar* m) { glPushDebugGroup(s, i, l, m); }

API void APIENTRY glPopDebugGroup(void) {
    MGL_CTX();
    if (ctx->debugGroups.empty()) return ctx->setError(GL_STACK_UNDERFLOW);
    auto g = ctx->debugGroups.back();
    ctx->debugGroups.pop_back();
    ctx->debugMessage(g.source, GL_DEBUG_TYPE_POP_GROUP, g.id, GL_DEBUG_SEVERITY_NOTIFICATION, g.message);
}
API void APIENTRY glPopDebugGroupKHR(void) { glPopDebugGroup(); }

static Object* labelTarget(Context* ctx, GLenum identifier, GLuint name) {
    switch (identifier) {
    case GL_BUFFER: return ctx->share->buffers.get(name);
    case GL_SHADER: return dynamic_cast<ShaderObj*>(ctx->share->shaderPrograms.get(name));
    case GL_PROGRAM: return dynamic_cast<ProgramObj*>(ctx->share->shaderPrograms.get(name));
    case GL_VERTEX_ARRAY: return ctx->vaos.get(name);
    case GL_QUERY: return ctx->queries.get(name);
    case GL_PROGRAM_PIPELINE: return ctx->pipelines.get(name);
    case GL_TRANSFORM_FEEDBACK: return ctx->xfbs.get(name);
    case GL_SAMPLER: return ctx->share->samplers.get(name);
    case GL_TEXTURE: return ctx->share->textures.get(name);
    case GL_RENDERBUFFER: return ctx->share->renderbuffers.get(name);
    case GL_FRAMEBUFFER: return ctx->framebuffers.get(name);
    default: return (Object*)-1;
    }
}

API void APIENTRY glObjectLabel(GLenum identifier, GLuint name, GLsizei length, const GLchar* label) {
    MGL_CTX();
    Object* o = labelTarget(ctx, identifier, name);
    if (o == (Object*)-1) return ctx->setError(GL_INVALID_ENUM);
    if (!o) return ctx->setError(GL_INVALID_VALUE);
    if (!label) {
        o->label.clear();
        return;
    }
    std::string l = length < 0 ? std::string(label) : std::string(label, length);
    if (l.size() >= 256) return ctx->setError(GL_INVALID_VALUE);
    o->label = l;
    if (Texture* t = dynamic_cast<Texture*>(o)) {
        if (t->mtl) t->mtl.label = @(l.c_str());
    } else if (Buffer* b = dynamic_cast<Buffer*>(o)) {
        if (b->mtl) b->mtl.label = @(l.c_str());
    }
}
API void APIENTRY glObjectLabelKHR(GLenum i, GLuint n, GLsizei l, const GLchar* s) { glObjectLabel(i, n, l, s); }

API void APIENTRY glGetObjectLabel(GLenum identifier, GLuint name, GLsizei bufSize, GLsizei* length, GLchar* label) {
    MGL_CTX();
    Object* o = labelTarget(ctx, identifier, name);
    if (o == (Object*)-1) return ctx->setError(GL_INVALID_ENUM);
    if (!o) return ctx->setError(GL_INVALID_VALUE);
    if (bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    GLsizei n = label && bufSize > 0 ? std::min<GLsizei>((GLsizei)o->label.size(), bufSize - 1) : (GLsizei)o->label.size();
    if (label && bufSize > 0) {
        memcpy(label, o->label.data(), n);
        label[n] = 0;
    }
    if (length) *length = n;
}
API void APIENTRY glGetObjectLabelKHR(GLenum i, GLuint n, GLsizei b, GLsizei* l, GLchar* s) { glGetObjectLabel(i, n, b, l, s); }

API void APIENTRY glObjectPtrLabel(const void* ptr, GLsizei length, const GLchar* label) {
    MGL_CTX();
    if (!ctx->share->syncs.count((SyncObj*)ptr)) return ctx->setError(GL_INVALID_VALUE);
}
API void APIENTRY glGetObjectPtrLabel(const void* ptr, GLsizei bufSize, GLsizei* length, GLchar* label) {
    MGL_CTX();
    if (!ctx->share->syncs.count((SyncObj*)ptr)) return ctx->setError(GL_INVALID_VALUE);
    if (length) *length = 0;
    if (label && bufSize > 0) label[0] = 0;
}

// ---------------------------------------------------------------------------
// ARB_internalformat_query(2)
// ---------------------------------------------------------------------------
API void APIENTRY glGetInternalformativ(GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint* params) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if (count == 0) return;
    const FormatInfo* f = formatInfo(internalformat);
    bool ms = target == GL_TEXTURE_2D_MULTISAMPLE || target == GL_TEXTURE_2D_MULTISAMPLE_ARRAY || target == GL_RENDERBUFFER;
    switch (pname) {
    case GL_NUM_SAMPLE_COUNTS:
        params[0] = f && f->renderable && ms && f->comp != Comp::Compressed ? 2 : 0;
        return;
    case GL_SAMPLES: {
        if (f && f->renderable && ms) {
            GLint s[2] = {4, 2};
            for (int i = 0; i < std::min(count, 2); i++) params[i] = s[i];
        }
        return;
    }
    case GL_INTERNALFORMAT_SUPPORTED: params[0] = f ? GL_TRUE : GL_FALSE; return;
    case GL_INTERNALFORMAT_PREFERRED: params[0] = f ? internalformat : GL_NONE; return;
    case GL_INTERNALFORMAT_RED_SIZE: params[0] = f ? f->rBits : 0; return;
    case GL_INTERNALFORMAT_GREEN_SIZE: params[0] = f ? f->gBits : 0; return;
    case GL_INTERNALFORMAT_BLUE_SIZE: params[0] = f ? f->bBits : 0; return;
    case GL_INTERNALFORMAT_ALPHA_SIZE: params[0] = f ? f->aBits : 0; return;
    case GL_INTERNALFORMAT_DEPTH_SIZE: params[0] = f ? f->dBits : 0; return;
    case GL_INTERNALFORMAT_STENCIL_SIZE: params[0] = f ? f->sBits : 0; return;
    case GL_MAX_WIDTH: case GL_MAX_HEIGHT: params[0] = f ? 16384 : 0; return;
    case GL_MAX_DEPTH: params[0] = f ? 2048 : 0; return;
    case GL_MAX_LAYERS: params[0] = f ? 2048 : 0; return;
    case GL_COLOR_RENDERABLE: params[0] = f && f->renderable && !isDepthOrStencil(f); return;
    case GL_DEPTH_RENDERABLE: params[0] = f && (f->kind == FmtKind::Depth || f->kind == FmtKind::DepthStencil); return;
    case GL_STENCIL_RENDERABLE: params[0] = f && (f->kind == FmtKind::Stencil || f->kind == FmtKind::DepthStencil); return;
    case GL_FRAMEBUFFER_RENDERABLE: case GL_FRAMEBUFFER_BLEND: params[0] = f && f->renderable ? GL_FULL_SUPPORT : GL_NONE; return;
    case GL_TEXTURE_IMAGE_FORMAT: params[0] = f ? (f->baseFormat == GL_LUMINANCE ? GL_RED : f->baseFormat) : GL_NONE; return;
    case GL_TEXTURE_IMAGE_TYPE: case GL_READ_PIXELS_TYPE:
        params[0] = !f ? GL_NONE : f->kind == FmtKind::Float ? GL_FLOAT : f->kind == FmtKind::SInt ? GL_INT : f->kind == FmtKind::UInt ? GL_UNSIGNED_INT : GL_UNSIGNED_BYTE;
        return;
    case GL_READ_PIXELS_FORMAT: params[0] = f ? GL_RGBA : GL_NONE; return;
    case GL_READ_PIXELS: case GL_FILTER: case GL_MIPMAP: case GL_TEXTURE_SHADOW: case GL_TEXTURE_GATHER: case GL_TEXTURE_GATHER_SHADOW:
    case GL_SHADER_IMAGE_LOAD: case GL_SHADER_IMAGE_STORE: case GL_SHADER_IMAGE_ATOMIC: case GL_VERTEX_TEXTURE: case GL_FRAGMENT_TEXTURE:
    case GL_COMPUTE_TEXTURE: case GL_GEOMETRY_TEXTURE: case GL_TESS_CONTROL_TEXTURE: case GL_TESS_EVALUATION_TEXTURE:
    case GL_SRGB_READ: case GL_SRGB_WRITE: case GL_CLEAR_BUFFER: case GL_TEXTURE_VIEW: case GL_MANUAL_GENERATE_MIPMAP:
    case GL_AUTO_GENERATE_MIPMAP:
        params[0] = f ? GL_FULL_SUPPORT : GL_NONE;
        return;
    case GL_IMAGE_TEXEL_SIZE: params[0] = f ? f->bytesPerPixel * 8 : 0; return;
    case GL_IMAGE_FORMAT_COMPATIBILITY_TYPE: params[0] = GL_IMAGE_FORMAT_COMPATIBILITY_BY_SIZE; return;
    case GL_TEXTURE_COMPRESSED: params[0] = f && f->comp == Comp::Compressed; return;
    case GL_TEXTURE_COMPRESSED_BLOCK_WIDTH: params[0] = f ? f->blockW : 0; return;
    case GL_TEXTURE_COMPRESSED_BLOCK_HEIGHT: params[0] = f ? f->blockH : 0; return;
    case GL_TEXTURE_COMPRESSED_BLOCK_SIZE: params[0] = f && f->comp == Comp::Compressed ? f->bytesPerPixel : 0; return;
    default: params[0] = 0; return;
    }
}

API void APIENTRY glGetInternalformati64v(GLenum target, GLenum internalformat, GLenum pname, GLsizei count, GLint64* params) {
    std::vector<GLint> tmp(std::max(1, count));
    glGetInternalformativ(target, internalformat, pname, count, tmp.data());
    for (int i = 0; i < count; i++) params[i] = tmp[i];
}

// ---------------------------------------------------------------------------
// transform feedback objects (capture itself lives in the draw path)
// ---------------------------------------------------------------------------
API void APIENTRY glGenTransformFeedbacks(GLsizei n, GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) ids[i] = ctx->xfbs.genName();
}

API void APIENTRY glCreateTransformFeedbacks(GLsizei n, GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        ids[i] = ctx->xfbs.genName();
        ctx->xfbs.create(ids[i], new TransformFeedbackObj());
    }
}

API void APIENTRY glDeleteTransformFeedbacks(GLsizei n, const GLuint* ids) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        TransformFeedbackObj* x = ctx->xfbs.get(ids[i]);
        if (x && x->active) return ctx->setError(GL_INVALID_OPERATION);
    }
    for (int i = 0; i < n; i++) {
        TransformFeedbackObj* x = ctx->xfbs.get(ids[i]);
        if (x && ctx->xfb.get() == x) ctx->xfb = ctx->defaultXfb;
        ctx->xfbs.remove(ids[i]);
    }
}

API GLboolean APIENTRY glIsTransformFeedback(GLuint id) {
    MGL_CTX_RET(GL_FALSE);
    return id && ctx->xfbs.get(id) != nullptr;
}

API void APIENTRY glBindTransformFeedback(GLenum target, GLuint id) {
    MGL_CTX();
    if (target != GL_TRANSFORM_FEEDBACK) return ctx->setError(GL_INVALID_ENUM);
    if (ctx->xfb->active && !ctx->xfb->paused) return ctx->setError(GL_INVALID_OPERATION);
    if (!id) {
        ctx->xfb = ctx->defaultXfb;
        return;
    }
    TransformFeedbackObj* x = ctx->xfbs.get(id);
    if (!x) {
        if (!ctx->xfbs.isName(id)) return ctx->setError(GL_INVALID_OPERATION);
        x = ctx->xfbs.create(id, new TransformFeedbackObj());
    }
    ctx->xfb = x;
}

API void APIENTRY glBeginTransformFeedback(GLenum primitiveMode) {
    MGL_CTX();
    if (primitiveMode != GL_POINTS && primitiveMode != GL_LINES && primitiveMode != GL_TRIANGLES) return ctx->setError(GL_INVALID_ENUM);
    TransformFeedbackObj* x = ctx->xfb.get();
    if (x->active) return ctx->setError(GL_INVALID_OPERATION);
    // the program of the last vertex-processing stage provides the captured varyings
    ProgramObj* p = nullptr;
    for (glsl::Stage st : {glsl::Stage::Geometry, glsl::Stage::TessEval, glsl::Stage::Vertex})
        if (!p) p = ctx->activeProgramForStage(st);
    if (!p || p->refl.xfbVaryings.empty()) return ctx->setError(GL_INVALID_OPERATION, "no transform feedback varyings");
    x->active = true;
    x->paused = false;
    x->primitiveMode = primitiveMode;
    x->program = p;
    for (auto& w : x->written) w = 0;
    static bool warned = false;
    warned = true;
}

API void APIENTRY glEndTransformFeedback(void) {
    MGL_CTX();
    TransformFeedbackObj* x = ctx->xfb.get();
    if (!x->active) return ctx->setError(GL_INVALID_OPERATION);
    x->active = false;
    x->paused = false;
    x->program = nullptr;
}

API void APIENTRY glPauseTransformFeedback(void) {
    MGL_CTX();
    TransformFeedbackObj* x = ctx->xfb.get();
    if (!x->active || x->paused) return ctx->setError(GL_INVALID_OPERATION);
    x->paused = true;
}

API void APIENTRY glResumeTransformFeedback(void) {
    MGL_CTX();
    TransformFeedbackObj* x = ctx->xfb.get();
    if (!x->active || !x->paused) return ctx->setError(GL_INVALID_OPERATION);
    x->paused = false;
}

API void APIENTRY glTransformFeedbackBufferBase(GLuint xfb, GLuint index, GLuint buffer) {
    MGL_CTX();
    TransformFeedbackObj* x = xfb ? ctx->xfbs.get(xfb) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_OPERATION);
    if (index >= kMaxXfbBuffers) return ctx->setError(GL_INVALID_VALUE);
    Buffer* b = buffer ? ctx->share->buffers.get(buffer) : nullptr;
    if (buffer && !b) return ctx->setError(GL_INVALID_OPERATION);
    x->buffers[index] = {b, 0, 0};
}

API void APIENTRY glTransformFeedbackBufferRange(GLuint xfb, GLuint index, GLuint buffer, GLintptr offset, GLsizeiptr size) {
    MGL_CTX();
    TransformFeedbackObj* x = xfb ? ctx->xfbs.get(xfb) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_OPERATION);
    if (index >= kMaxXfbBuffers || offset < 0 || size <= 0 || (offset % 4) || (size % 4)) return ctx->setError(GL_INVALID_VALUE);
    Buffer* b = buffer ? ctx->share->buffers.get(buffer) : nullptr;
    if (buffer && !b) return ctx->setError(GL_INVALID_OPERATION);
    x->buffers[index] = {b, offset, size};
}

API void APIENTRY glGetTransformFeedbackiv(GLuint xfb, GLenum pname, GLint* param) {
    MGL_CTX();
    TransformFeedbackObj* x = xfb ? ctx->xfbs.get(xfb) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_OPERATION);
    switch (pname) {
    case GL_TRANSFORM_FEEDBACK_PAUSED: *param = x->paused; break;
    case GL_TRANSFORM_FEEDBACK_ACTIVE: *param = x->active; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetTransformFeedbacki_v(GLuint xfb, GLenum pname, GLuint index, GLint* param) {
    MGL_CTX();
    TransformFeedbackObj* x = xfb ? ctx->xfbs.get(xfb) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_OPERATION);
    if (index >= kMaxXfbBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (pname != GL_TRANSFORM_FEEDBACK_BUFFER_BINDING) return ctx->setError(GL_INVALID_ENUM);
    *param = x->buffers[index].buffer ? x->buffers[index].buffer->name : 0;
}

API void APIENTRY glGetTransformFeedbacki64_v(GLuint xfb, GLenum pname, GLuint index, GLint64* param) {
    MGL_CTX();
    TransformFeedbackObj* x = xfb ? ctx->xfbs.get(xfb) : ctx->defaultXfb.get();
    if (!x) return ctx->setError(GL_INVALID_OPERATION);
    if (index >= kMaxXfbBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (pname == GL_TRANSFORM_FEEDBACK_BUFFER_START) *param = x->buffers[index].offset;
    else if (pname == GL_TRANSFORM_FEEDBACK_BUFFER_SIZE) *param = x->buffers[index].size;
    else ctx->setError(GL_INVALID_ENUM);
}

// Context flush control & misc no-ops
API void APIENTRY glMaxShaderCompilerThreadsKHR(GLuint count) {}
API void APIENTRY glMaxShaderCompilerThreadsARB(GLuint count) {}
API void APIENTRY glMinSampleShadingARB(GLfloat value) {
    MGL_CTX();
    ctx->minSampleShading = value;
}
