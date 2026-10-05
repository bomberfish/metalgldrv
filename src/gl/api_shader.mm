// Shader & program objects, uniforms, program interface queries, pipelines.
#import "gl/context.h"
#include <cmath>
#include "gl/gltypes.h"
#include "gl/pipeline.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {
ProgramObj::~ProgramObj() = default;

static glsl::Stage stageForType(GLenum type, bool* ok) {
    *ok = true;
    switch (type) {
    case GL_VERTEX_SHADER: return glsl::Stage::Vertex;
    case GL_TESS_CONTROL_SHADER: return glsl::Stage::TessControl;
    case GL_TESS_EVALUATION_SHADER: return glsl::Stage::TessEval;
    case GL_GEOMETRY_SHADER: return glsl::Stage::Geometry;
    case GL_FRAGMENT_SHADER: return glsl::Stage::Fragment;
    case GL_COMPUTE_SHADER: return glsl::Stage::Compute;
    default: *ok = false; return glsl::Stage::Vertex;
    }
}

// Rebuilds derived per-program state after a successful link.
static void setupLinkedProgram(ProgramObj* p) {
    const glsl::Reflection& r = p->refl;
    p->uniformData.assign(std::max<uint32_t>(r.defaultUniformSize, 16), 0);
    p->locations.clear();
    int maxLoc = -1;
    for (size_t i = 0; i < r.uniforms.size(); i++) {
        const auto& u = r.uniforms[i];
        if (u.blockIndex >= 0 || u.location < 0) continue;
        maxLoc = std::max(maxLoc, u.location + std::max(1, u.arraySize) - 1);
    }
    p->locations.assign(maxLoc + 1, ProgramObj::Loc{-1, 0});
    for (size_t i = 0; i < r.uniforms.size(); i++) {
        const auto& u = r.uniforms[i];
        if (u.blockIndex >= 0 || u.location < 0) continue;
        for (int e = 0; e < std::max(1, u.arraySize); e++) p->locations[u.location + e] = {(int)i, e};
    }
    p->samplerUnits.assign(r.slots.size(), 0);
    // Initial opaque bindings from layout(binding=)
    for (size_t s = 0; s < r.slots.size(); s++) {
        const auto& sl = r.slots[s];
        if (sl.kind == glsl::SlotKind::Texture || sl.kind == glsl::SlotKind::Sampler || sl.kind == glsl::SlotKind::Image) {
            if (sl.index < r.uniforms.size()) p->samplerUnits[s] = r.uniforms[sl.index].binding + sl.element;
        }
    }
    p->blockBindings.clear();
    for (auto& b : r.uniformBlocks) p->blockBindings.push_back(b.binding);
    p->storageBindings.clear();
    for (auto& b : r.storageBlocks) p->storageBindings.push_back(b.binding);
    p->variants.clear();
    p->uniformVersion++;
}
}  // namespace mgl

static ShaderObj* getShader(Context* ctx, GLuint name, bool reportWrongType = true) {
    Object* o = ctx->share->shaderPrograms.get(name);
    if (!o) {
        ctx->setError(GL_INVALID_VALUE, "no shader %u", name);
        return nullptr;
    }
    ShaderObj* s = dynamic_cast<ShaderObj*>(o);
    if (!s && reportWrongType) ctx->setError(GL_INVALID_OPERATION, "%u is not a shader", name);
    return s;
}

static ProgramObj* getProgram(Context* ctx, GLuint name) {
    Object* o = ctx->share->shaderPrograms.get(name);
    if (!o) {
        ctx->setError(GL_INVALID_VALUE, "no program %u", name);
        return nullptr;
    }
    ProgramObj* p = dynamic_cast<ProgramObj*>(o);
    if (!p) ctx->setError(GL_INVALID_OPERATION, "%u is not a program", name);
    return p;
}

// ---------------------------------------------------------------------------
// shaders
// ---------------------------------------------------------------------------
API GLuint APIENTRY glCreateShader(GLenum type) {
    MGL_CTX_RET(0);
    bool ok;
    stageForType(type, &ok);
    if (!ok) {
        ctx->setError(GL_INVALID_ENUM);
        return 0;
    }
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    GLuint n = ctx->share->shaderPrograms.genName();
    ShaderObj* s = new ShaderObj();
    s->type = type;
    ctx->share->shaderPrograms.create(n, s);
    return n;
}

static void maybeFreeShader(Context* ctx, ShaderObj* s) {
    if (!s->deleted) return;
    // Still attached somewhere? The Ref from ProgramObj::attached keeps it alive anyway.
}

API void APIENTRY glDeleteShader(GLuint shader) {
    MGL_CTX();
    if (!shader) return;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    Ref<Object> keep = ctx->share->shaderPrograms.remove(shader);
    (void)keep;
}

API GLboolean APIENTRY glIsShader(GLuint shader) {
    MGL_CTX_RET(GL_FALSE);
    return dynamic_cast<ShaderObj*>(ctx->share->shaderPrograms.get(shader)) != nullptr;
}

API void APIENTRY glShaderSource(GLuint shader, GLsizei count, const GLchar* const* strings, const GLint* lengths) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    s->sources.clear();
    for (int i = 0; i < count; i++) {
        if (!strings[i]) {
            s->sources.emplace_back();
            continue;
        }
        if (lengths && lengths[i] >= 0) s->sources.emplace_back(strings[i], lengths[i]);
        else s->sources.emplace_back(strings[i]);
    }
}

API void APIENTRY glCompileShader(GLuint shader) {
    MGL_CTX();
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    bool ok;
    glsl::Stage st = stageForType(s->type, &ok);
    glsl::CompileOptions opts;
    opts.compatibilityProfile = !ctx->coreProfile;
    glsl::CompileResult r = glsl::compile(st, s->sources, opts);
    s->shader = r.shader;
    s->compiled = r.shader != nullptr;
    s->infoLog = r.infoLog;
    if (!s->compiled) {
        MGL_DBG("shader %u compile failed:\n%s", shader, r.infoLog.c_str());
        if (envFlag("MGL_DUMP_FAILED_GLSL")) {
            std::string src;
            for (auto& x : s->sources) src += x;
            MGL_ERR("shader %u failed to compile:\n%s\n---- source ----\n%s", shader, r.infoLog.c_str(), src.c_str());
        }
    }
}

API void APIENTRY glGetShaderiv(GLuint shader, GLenum pname, GLint* params) {
    MGL_CTX();
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    switch (pname) {
    case GL_SHADER_TYPE: *params = s->type; break;
    case GL_DELETE_STATUS: *params = s->deleted; break;
    case GL_COMPILE_STATUS: *params = s->compiled; break;
    case GL_INFO_LOG_LENGTH: *params = s->infoLog.empty() ? 0 : (GLint)s->infoLog.size() + 1; break;
    case GL_SHADER_SOURCE_LENGTH: {
        size_t n = 0;
        for (auto& x : s->sources) n += x.size();
        *params = n ? (GLint)n + 1 : 0;
        break;
    }
    case GL_SPIR_V_BINARY: *params = GL_FALSE; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

static void copyLog(const std::string& log, GLsizei bufSize, GLsizei* length, GLchar* out) {
    if (bufSize <= 0 || !out) {
        if (length) *length = 0;
        return;
    }
    GLsizei n = std::min<GLsizei>((GLsizei)log.size(), bufSize - 1);
    memcpy(out, log.data(), n);
    out[n] = 0;
    if (length) *length = n;
}

API void APIENTRY glGetShaderInfoLog(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    MGL_CTX();
    if (bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    ShaderObj* s = getShader(ctx, shader);
    if (s) copyLog(s->infoLog, bufSize, length, infoLog);
}

API void APIENTRY glGetShaderSource(GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* source) {
    MGL_CTX();
    if (bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    std::string all;
    for (auto& x : s->sources) all += x;
    copyLog(all, bufSize, length, source);
}

API void APIENTRY glShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary, GLsizei length) {
    MGL_CTX();
    ctx->setError(GL_INVALID_ENUM, "no shader binary formats supported");
}

API void APIENTRY glSpecializeShader(GLuint shader, const GLchar* pEntryPoint, GLuint n, const GLuint* idx, const GLuint* val) {
    MGL_CTX();
    ctx->setError(GL_INVALID_OPERATION, "SPIR-V shaders not supported");
}

API void APIENTRY glReleaseShaderCompiler(void) {}

API void APIENTRY glGetShaderPrecisionFormat(GLenum shadertype, GLenum precisiontype, GLint* range, GLint* precision) {
    MGL_CTX();
    switch (precisiontype) {
    case GL_LOW_FLOAT: case GL_MEDIUM_FLOAT: case GL_HIGH_FLOAT:
        range[0] = range[1] = 127;
        *precision = 23;
        break;
    case GL_LOW_INT: case GL_MEDIUM_INT: case GL_HIGH_INT:
        range[0] = range[1] = 31;
        *precision = 0;
        break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

// ---------------------------------------------------------------------------
// programs
// ---------------------------------------------------------------------------
API GLuint APIENTRY glCreateProgram(void) {
    MGL_CTX_RET(0);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    GLuint n = ctx->share->shaderPrograms.genName();
    ctx->share->shaderPrograms.create(n, new ProgramObj());
    return n;
}

API void APIENTRY glDeleteProgram(GLuint program) {
    MGL_CTX();
    if (!program) return;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    // A program in use stays alive via ctx->program until unbound.
    ctx->share->shaderPrograms.remove(program);
}

API GLboolean APIENTRY glIsProgram(GLuint program) {
    MGL_CTX_RET(GL_FALSE);
    return dynamic_cast<ProgramObj*>(ctx->share->shaderPrograms.get(program)) != nullptr;
}

API void APIENTRY glAttachShader(GLuint program, GLuint shader) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    ShaderObj* s = p ? getShader(ctx, shader) : nullptr;
    if (!p || !s) return;
    for (auto& a : p->attached) {
        if (a.get() == s) return ctx->setError(GL_INVALID_OPERATION, "already attached");
        if (a->type == s->type && s->type != GL_NONE && false) return ctx->setError(GL_INVALID_OPERATION);
    }
    p->attached.push_back(s);
}

API void APIENTRY glDetachShader(GLuint program, GLuint shader) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    Object* o = ctx->share->shaderPrograms.get(shader);
    for (auto it = p->attached.begin(); it != p->attached.end(); ++it) {
        if (it->get()->name == shader) {
            p->attached.erase(it);
            return;
        }
    }
    if (!o) return ctx->setError(GL_INVALID_VALUE);
    if (!dynamic_cast<ShaderObj*>(o)) return ctx->setError(GL_INVALID_OPERATION);
    ctx->setError(GL_INVALID_OPERATION, "not attached");
}

API void APIENTRY glGetAttachedShaders(GLuint program, GLsizei maxCount, GLsizei* count, GLuint* shaders) {
    MGL_CTX();
    if (maxCount < 0) return ctx->setError(GL_INVALID_VALUE);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    int n = 0;
    for (auto& a : p->attached) {
        if (n >= maxCount) break;
        shaders[n++] = a->name;
    }
    if (count) *count = n;
}

API void APIENTRY glBindAttribLocation(GLuint program, GLuint index, const GLchar* name) {
    MGL_CTX();
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    if (strncmp(name, "gl_", 3) == 0) return ctx->setError(GL_INVALID_OPERATION);
    ProgramObj* p = getProgram(ctx, program);
    if (p) p->attribBindings[name] = index;
}

API void APIENTRY glBindFragDataLocationIndexed(GLuint program, GLuint colorNumber, GLuint index, const GLchar* name) {
    MGL_CTX();
    if (index > 1 || colorNumber >= kMaxDrawBuffers || (index == 1 && colorNumber >= 1)) return ctx->setError(GL_INVALID_VALUE);
    if (strncmp(name, "gl_", 3) == 0) return ctx->setError(GL_INVALID_OPERATION);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    p->fragDataBindings[name] = colorNumber;
    p->fragDataIndices[name] = index;
}

API void APIENTRY glBindFragDataLocation(GLuint program, GLuint color, const GLchar* name) { glBindFragDataLocationIndexed(program, color, 0, name); }

API void APIENTRY glTransformFeedbackVaryings(GLuint program, GLsizei count, const GLchar* const* varyings, GLenum bufferMode) {
    MGL_CTX();
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if (bufferMode != GL_INTERLEAVED_ATTRIBS && bufferMode != GL_SEPARATE_ATTRIBS) return ctx->setError(GL_INVALID_ENUM);
    if (bufferMode == GL_SEPARATE_ATTRIBS && count > kMaxXfbBuffers) return ctx->setError(GL_INVALID_VALUE);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    p->xfbVaryings.clear();
    for (int i = 0; i < count; i++) p->xfbVaryings.push_back(varyings[i]);
    p->xfbBufferMode = bufferMode;
}

API void APIENTRY glProgramParameteri(GLuint program, GLenum pname, GLint value) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    switch (pname) {
    case GL_PROGRAM_SEPARABLE: p->separable = value; break;
    case GL_PROGRAM_BINARY_RETRIEVABLE_HINT: p->retrievableHint = value; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glLinkProgram(GLuint program) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (ctx->xfb->active && ctx->xfb->program.get() == p) return ctx->setError(GL_INVALID_OPERATION);
    glsl::LinkInput in;
    bool allCompiled = true;
    for (auto& s : p->attached) {
        if (!s->compiled || !s->shader) allCompiled = false;
        else in.shaders.push_back(s->shader);
    }
    in.attribBindings = p->attribBindings;
    in.fragDataBindings = p->fragDataBindings;
    in.fragDataIndices = p->fragDataIndices;
    in.xfbVaryings = p->xfbVaryings;
    in.xfbInterleaved = p->xfbBufferMode == GL_INTERLEAVED_ATTRIBS;
    in.separable = p->separable;
    p->linkGeneration++;
    if (!allCompiled || in.shaders.empty()) {
        p->linked = false;
        p->program = nullptr;
        p->infoLog = in.shaders.empty() ? "error: no shaders attached\n" : "error: attached shader not successfully compiled\n";
        return;
    }
    glsl::LinkResult r = glsl::link(in);
    p->infoLog = r.infoLog;
    if (!r.program) {
        p->linked = false;
        p->program = nullptr;
        MGL_DBG("program %u link failed:\n%s", program, r.infoLog.c_str());
        return;
    }
    p->linked = true;
    p->program = r.program;
    p->refl = std::move(r.reflection);
    setupLinkedProgram(p);
    // Compile the common variant now so the MSL compile cost lands at link time, not mid-frame.
    uint32_t sm = p->refl.stageMask;
    bool multiPass = sm & ((1u << (int)glsl::Stage::Geometry) | (1u << (int)glsl::Stage::TessEval));
    if (!multiPass && !envFlag("MGL_LAZY_COMPILE")) {
        glsl::VariantKey vk;
        vk.kind = (sm & (1u << (int)glsl::Stage::Compute)) ? glsl::PipelineKind::Compute : glsl::PipelineKind::Classic;
        ProgramVariant* v = ctx->pipelines_->variant(p, vk);
        if (v && !v->ok) {
            // Report backend failures through the link status so apps see them.
            p->linked = false;
            p->infoLog += "error: internal Metal shader compilation failed: " + v->error + "\n";
        }
    }
}

API void APIENTRY glUseProgram(GLuint program) {
    MGL_CTX();
    if (ctx->xfb->active && !ctx->xfb->paused) return ctx->setError(GL_INVALID_OPERATION);
    if (!program) {
        ctx->program = nullptr;
        return;
    }
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!p->linked) return ctx->setError(GL_INVALID_OPERATION, "program not linked");
    ctx->program = p;
}

API void APIENTRY glValidateProgram(GLuint program) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    p->validated = p->linked;
    if (!p->linked) p->infoLog += "validation failed: program not linked\n";
}

API void APIENTRY glGetProgramiv(GLuint program, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    const glsl::Reflection& r = p->refl;
    auto maxLen = [](auto& v, auto get) {
        GLint m = 0;
        for (auto& x : v) m = std::max<GLint>(m, (GLint)get(x).size() + 1);
        return m;
    };
    switch (pname) {
    case GL_DELETE_STATUS: *params = p->deleted; break;
    case GL_LINK_STATUS: *params = p->linked; break;
    case GL_VALIDATE_STATUS: *params = p->validated; break;
    case GL_INFO_LOG_LENGTH: *params = p->infoLog.empty() ? 0 : (GLint)p->infoLog.size() + 1; break;
    case GL_ATTACHED_SHADERS: *params = (GLint)p->attached.size(); break;
    case GL_ACTIVE_ATTRIBUTES: *params = p->linked ? (GLint)r.attributes.size() : 0; break;
    case GL_ACTIVE_ATTRIBUTE_MAX_LENGTH: *params = p->linked ? maxLen(r.attributes, [](auto& a) -> const std::string& { return a.name; }) : 0; break;
    case GL_ACTIVE_UNIFORMS: *params = p->linked ? (GLint)r.uniforms.size() : 0; break;
    case GL_ACTIVE_UNIFORM_MAX_LENGTH: *params = p->linked ? maxLen(r.uniforms, [](auto& a) -> const std::string& { return a.name; }) : 0; break;
    case GL_ACTIVE_UNIFORM_BLOCKS: *params = (GLint)r.uniformBlocks.size(); break;
    case GL_ACTIVE_UNIFORM_BLOCK_MAX_NAME_LENGTH: *params = maxLen(r.uniformBlocks, [](auto& a) -> const std::string& { return a.name; }); break;
    case GL_ACTIVE_ATOMIC_COUNTER_BUFFERS: *params = (GLint)r.atomicCounterBuffers.size(); break;
    case GL_TRANSFORM_FEEDBACK_BUFFER_MODE: *params = p->xfbBufferMode; break;
    case GL_TRANSFORM_FEEDBACK_VARYINGS: *params = p->linked ? (GLint)r.xfbVaryings.size() : 0; break;
    case GL_TRANSFORM_FEEDBACK_VARYING_MAX_LENGTH: *params = maxLen(r.xfbVaryings, [](auto& a) -> const std::string& { return a.name; }); break;
    case GL_GEOMETRY_VERTICES_OUT: if (!(r.stageMask & (1 << (int)glsl::Stage::Geometry))) return ctx->setError(GL_INVALID_OPERATION); *params = r.gsMaxVertices; break;
    case GL_GEOMETRY_INPUT_TYPE: if (!(r.stageMask & (1 << (int)glsl::Stage::Geometry))) return ctx->setError(GL_INVALID_OPERATION); *params = r.gsInputPrimitive; break;
    case GL_GEOMETRY_OUTPUT_TYPE: if (!(r.stageMask & (1 << (int)glsl::Stage::Geometry))) return ctx->setError(GL_INVALID_OPERATION); *params = r.gsOutputPrimitive; break;
    case GL_GEOMETRY_SHADER_INVOCATIONS: *params = r.gsInvocations; break;
    case GL_TESS_CONTROL_OUTPUT_VERTICES: *params = r.tcsOutputVertices; break;
    case GL_TESS_GEN_MODE: *params = r.tesPrimitiveMode; break;
    case GL_TESS_GEN_SPACING: *params = r.tesSpacing; break;
    case GL_TESS_GEN_VERTEX_ORDER: *params = r.tesOrder; break;
    case GL_TESS_GEN_POINT_MODE: *params = r.tesPointMode; break;
    case GL_COMPUTE_WORK_GROUP_SIZE:
        if (!p->linked || !(r.stageMask & (1 << (int)glsl::Stage::Compute))) return ctx->setError(GL_INVALID_OPERATION);
        params[0] = r.localSize[0];
        params[1] = r.localSize[1];
        params[2] = r.localSize[2];
        break;
    case GL_PROGRAM_SEPARABLE: *params = p->separable; break;
    case GL_PROGRAM_BINARY_RETRIEVABLE_HINT: *params = p->retrievableHint; break;
    case GL_PROGRAM_BINARY_LENGTH: *params = 0; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetProgramInfoLog(GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    MGL_CTX();
    if (bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    ProgramObj* p = getProgram(ctx, program);
    if (p) copyLog(p->infoLog, bufSize, length, infoLog);
}

API void APIENTRY glGetProgramBinary(GLuint program, GLsizei bufSize, GLsizei* length, GLenum* binaryFormat, void* binary) {
    MGL_CTX();
    ctx->setError(GL_INVALID_OPERATION, "program binaries not supported");
}

API void APIENTRY glProgramBinary(GLuint program, GLenum binaryFormat, const void* binary, GLsizei length) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    p->linked = false;
    p->infoLog = "program binary format not supported\n";
    ctx->setError(GL_INVALID_ENUM);
}

// --- attributes / frag data ---
API GLint APIENTRY glGetAttribLocation(GLuint program, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    for (auto& a : p->refl.attributes)
        if (a.name == name) return a.location;
    return -1;
}

API void APIENTRY glGetActiveAttrib(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.attributes.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& a = p->refl.attributes[index];
    copyLog(a.name, bufSize, length, name);
    if (size) *size = a.arraySize;
    if (type) *type = a.type;
}

API GLint APIENTRY glGetFragDataLocation(GLuint program, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    std::string n = name;
    for (auto& o : p->refl.fragOutputs) {
        if (o.name == n) return o.location;
        // array element access "out[2]"
        if (n.size() > o.name.size() && n.compare(0, o.name.size(), o.name) == 0 && n[o.name.size()] == '[')
            return o.location + atoi(n.c_str() + o.name.size() + 1);
    }
    return -1;
}

API GLint APIENTRY glGetFragDataIndex(GLuint program, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    for (auto& o : p->refl.fragOutputs)
        if (o.name == name) return o.index;
    return -1;
}

// --- uniform queries ---
// Parses "name", "name[3]" -> base, element. Returns false on malformed.
static bool parseArrayName(const std::string& n, std::string& base, int& elem, bool& hadIndex) {
    hadIndex = false;
    elem = 0;
    if (n.empty() || n.back() != ']') {
        base = n;
        return true;
    }
    size_t lb = n.rfind('[');
    if (lb == std::string::npos) return false;
    std::string idx = n.substr(lb + 1, n.size() - lb - 2);
    if (idx.empty()) return false;
    for (char c : idx)
        if (c < '0' || c > '9') return false;
    if (idx.size() > 1 && idx[0] == '0') return false;
    elem = atoi(idx.c_str());
    base = n.substr(0, lb);
    hadIndex = true;
    return true;
}

static GLint uniformLocation(ProgramObj* p, const char* name) {
    std::string n = name;
    std::string base;
    int elem;
    bool hadIndex;
    if (!parseArrayName(n, base, elem, hadIndex)) return -1;
    for (auto& u : p->refl.uniforms) {
        if (u.blockIndex >= 0 || u.location < 0) continue;
        bool isArray = u.name.size() > 3 && u.name.compare(u.name.size() - 3, 3, "[0]") == 0;
        std::string ubase = isArray ? u.name.substr(0, u.name.size() - 3) : u.name;
        if (n == u.name) return u.location;
        if (isArray && base == ubase) {
            if (elem >= std::max(1, u.arraySize)) return -1;
            return u.location + elem;
        }
        if (!isArray && !hadIndex && n == ubase) return u.location;
    }
    return -1;
}

API GLint APIENTRY glGetUniformLocation(GLuint program, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    return uniformLocation(p, name);
}

API void APIENTRY glGetActiveUniform(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLint* size, GLenum* type, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.uniforms.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& u = p->refl.uniforms[index];
    copyLog(u.name, bufSize, length, name);
    if (size) *size = u.arraySize;
    if (type) *type = u.type;
}

API void APIENTRY glGetActiveUniformName(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.uniforms.size()) return ctx->setError(GL_INVALID_VALUE);
    copyLog(p->refl.uniforms[index].name, bufSize, length, name);
}

API void APIENTRY glGetUniformIndices(GLuint program, GLsizei count, const GLchar* const* names, GLuint* indices) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    for (int i = 0; i < count; i++) {
        indices[i] = GL_INVALID_INDEX;
        std::string n = names[i];
        for (size_t k = 0; k < p->refl.uniforms.size(); k++) {
            const std::string& un = p->refl.uniforms[k].name;
            if (un == n || (un.size() > 3 && un.compare(un.size() - 3, 3, "[0]") == 0 && un.substr(0, un.size() - 3) == n)) {
                indices[i] = (GLuint)k;
                break;
            }
        }
    }
}

API void APIENTRY glGetActiveUniformsiv(GLuint program, GLsizei count, const GLuint* indices, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++)
        if (indices[i] >= p->refl.uniforms.size()) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++) {
        const auto& u = p->refl.uniforms[indices[i]];
        switch (pname) {
        case GL_UNIFORM_TYPE: params[i] = u.type; break;
        case GL_UNIFORM_SIZE: params[i] = u.arraySize; break;
        case GL_UNIFORM_NAME_LENGTH: params[i] = (GLint)u.name.size() + 1; break;
        case GL_UNIFORM_BLOCK_INDEX: params[i] = u.blockIndex; break;
        case GL_UNIFORM_OFFSET: params[i] = u.blockIndex >= 0 ? u.offset : -1; break;
        case GL_UNIFORM_ARRAY_STRIDE: params[i] = u.blockIndex >= 0 ? u.arrayStride : -1; break;
        case GL_UNIFORM_MATRIX_STRIDE: params[i] = u.blockIndex >= 0 ? u.matrixStride : -1; break;
        case GL_UNIFORM_IS_ROW_MAJOR: params[i] = u.rowMajor; break;
        case GL_UNIFORM_ATOMIC_COUNTER_BUFFER_INDEX: params[i] = u.atomicCounterBufferIndex; break;
        default: return ctx->setError(GL_INVALID_ENUM);
        }
    }
}

API GLuint APIENTRY glGetUniformBlockIndex(GLuint program, const GLchar* name) {
    MGL_CTX_RET(GL_INVALID_INDEX);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return GL_INVALID_INDEX;
    for (size_t i = 0; i < p->refl.uniformBlocks.size(); i++)
        if (p->refl.uniformBlocks[i].name == name) return (GLuint)i;
    return GL_INVALID_INDEX;
}

API void APIENTRY glGetActiveUniformBlockName(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.uniformBlocks.size()) return ctx->setError(GL_INVALID_VALUE);
    copyLog(p->refl.uniformBlocks[index].name, bufSize, length, name);
}

API void APIENTRY glGetActiveUniformBlockiv(GLuint program, GLuint index, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.uniformBlocks.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& b = p->refl.uniformBlocks[index];
    using S = glsl::Stage;
    switch (pname) {
    case GL_UNIFORM_BLOCK_BINDING: *params = p->blockBindings[index]; break;
    case GL_UNIFORM_BLOCK_DATA_SIZE: *params = b.dataSize; break;
    case GL_UNIFORM_BLOCK_NAME_LENGTH: *params = (GLint)b.name.size() + 1; break;
    case GL_UNIFORM_BLOCK_ACTIVE_UNIFORMS: *params = (GLint)b.activeUniforms.size(); break;
    case GL_UNIFORM_BLOCK_ACTIVE_UNIFORM_INDICES: for (size_t i = 0; i < b.activeUniforms.size(); i++) params[i] = b.activeUniforms[i]; break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_VERTEX_SHADER: *params = !!(b.stageMask & (1 << (int)S::Vertex)); break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_CONTROL_SHADER: *params = !!(b.stageMask & (1 << (int)S::TessControl)); break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_TESS_EVALUATION_SHADER: *params = !!(b.stageMask & (1 << (int)S::TessEval)); break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_GEOMETRY_SHADER: *params = !!(b.stageMask & (1 << (int)S::Geometry)); break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_FRAGMENT_SHADER: *params = !!(b.stageMask & (1 << (int)S::Fragment)); break;
    case GL_UNIFORM_BLOCK_REFERENCED_BY_COMPUTE_SHADER: *params = !!(b.stageMask & (1 << (int)S::Compute)); break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glUniformBlockBinding(GLuint program, GLuint index, GLuint binding) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.uniformBlocks.size() || binding >= kMaxUniformBufferBindings) return ctx->setError(GL_INVALID_VALUE);
    p->blockBindings[index] = binding;
}

API void APIENTRY glShaderStorageBlockBinding(GLuint program, GLuint index, GLuint binding) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.storageBlocks.size() || binding >= kMaxSSBOBindings) return ctx->setError(GL_INVALID_VALUE);
    p->storageBindings[index] = binding;
}

API void APIENTRY glGetActiveAtomicCounterBufferiv(GLuint program, GLuint index, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.atomicCounterBuffers.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& a = p->refl.atomicCounterBuffers[index];
    using S = glsl::Stage;
    switch (pname) {
    case GL_ATOMIC_COUNTER_BUFFER_BINDING: *params = a.binding; break;
    case GL_ATOMIC_COUNTER_BUFFER_DATA_SIZE: *params = a.dataSize; break;
    case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTERS: *params = (GLint)a.activeUniforms.size(); break;
    case GL_ATOMIC_COUNTER_BUFFER_ACTIVE_ATOMIC_COUNTER_INDICES: for (size_t i = 0; i < a.activeUniforms.size(); i++) params[i] = a.activeUniforms[i]; break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_VERTEX_SHADER: *params = !!(a.stageMask & (1 << (int)S::Vertex)); break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_CONTROL_SHADER: *params = !!(a.stageMask & (1 << (int)S::TessControl)); break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_TESS_EVALUATION_SHADER: *params = !!(a.stageMask & (1 << (int)S::TessEval)); break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_GEOMETRY_SHADER: *params = !!(a.stageMask & (1 << (int)S::Geometry)); break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_FRAGMENT_SHADER: *params = !!(a.stageMask & (1 << (int)S::Fragment)); break;
    case GL_ATOMIC_COUNTER_BUFFER_REFERENCED_BY_COMPUTE_SHADER: *params = !!(a.stageMask & (1 << (int)S::Compute)); break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetTransformFeedbackVarying(GLuint program, GLuint index, GLsizei bufSize, GLsizei* length, GLsizei* size, GLenum* type,
                                                GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (index >= p->refl.xfbVaryings.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& v = p->refl.xfbVaryings[index];
    copyLog(v.name, bufSize, length, name);
    if (size) *size = v.arraySize;
    if (type) *type = v.type;
}

// ---------------------------------------------------------------------------
// uniform setters
// ---------------------------------------------------------------------------
enum class SrcKind { Float, Int, UInt, Double };

static void setUniform(Context* ctx, ProgramObj* p, GLint location, GLsizei count, SrcKind kind, int comps, int cols, bool transpose,
                       const void* data) {
    if (!p) return ctx->setError(GL_INVALID_OPERATION, "no current program");
    if (!p->linked) return ctx->setError(GL_INVALID_OPERATION);
    if (location == -1) return;
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    if (location < 0 || location >= (GLint)p->locations.size() || p->locations[location].uniform < 0)
        return ctx->setError(GL_INVALID_OPERATION, "invalid uniform location %d", location);
    auto loc = p->locations[location];
    const glsl::Uniform& u = p->refl.uniforms[loc.uniform];
    TypeShape ts = typeShape(u.type);
    bool matrix = cols > 1;
    int rows = matrix ? comps / cols : comps;
    // type compatibility
    if (ts.base == BaseType::Sampler || ts.base == BaseType::Image) {
        if (kind != SrcKind::Int || comps != 1 || matrix) return ctx->setError(GL_INVALID_OPERATION);
    } else if (ts.base == BaseType::Bool) {
        if (kind == SrcKind::Double || matrix || ts.components() != comps) return ctx->setError(GL_INVALID_OPERATION);
    } else {
        bool ok = (ts.base == BaseType::Float && kind == SrcKind::Float) || (ts.base == BaseType::Int && kind == SrcKind::Int) ||
                  (ts.base == BaseType::UInt && kind == SrcKind::UInt) || (ts.base == BaseType::Double && kind == SrcKind::Double);
        if (!ok) return ctx->setError(GL_INVALID_OPERATION, "uniform type mismatch");
        if ((ts.cols > 1) != matrix || ts.cols != (matrix ? cols : 1) || ts.rows != rows) return ctx->setError(GL_INVALID_OPERATION, "uniform size mismatch");
    }
    int arraySize = std::max(1, u.arraySize);
    if (count > 1 && u.arraySize <= 1 && !(u.name.size() > 3 && u.name.compare(u.name.size() - 3, 3, "[0]") == 0))
        return ctx->setError(GL_INVALID_OPERATION, "count > 1 for non-array uniform");
    int n = std::min(count, arraySize - loc.element);
    if (ts.base == BaseType::Sampler || ts.base == BaseType::Image) {
        const GLint* v = (const GLint*)data;
        int maxUnit = ts.base == BaseType::Sampler ? kMaxTextureUnits : kMaxImageUnits;
        for (int i = 0; i < n; i++)
            if (v[i] < 0 || v[i] >= maxUnit) return ctx->setError(GL_INVALID_VALUE, "sampler unit out of range");
        // update every slot of this uniform/element
        for (size_t s = 0; s < p->refl.slots.size(); s++) {
            const auto& sl = p->refl.slots[s];
            if ((int)sl.index != loc.uniform) continue;
            if (sl.kind != glsl::SlotKind::Texture && sl.kind != glsl::SlotKind::Sampler && sl.kind != glsl::SlotKind::Image) continue;
            int e = (int)sl.element - loc.element;
            if (e >= 0 && e < n) p->samplerUnits[s] = v[e];
        }
        return;
    }
    if (u.offset < 0) return;
    uint8_t* base = p->uniformData.data() + u.offset;
    int elemSize = ts.base == BaseType::Double ? 8 : 4;
    int arrayStride = u.arrayStride ? u.arrayStride : (matrix ? cols * u.matrixStride : comps * elemSize);
    int matStride = u.matrixStride ? u.matrixStride : rows * elemSize;
    for (int i = 0; i < n; i++) {
        uint8_t* dst = base + (size_t)(loc.element + i) * arrayStride;
        if (dst + (matrix ? cols * matStride : comps * elemSize) > p->uniformData.data() + p->uniformData.size()) break;
        if (matrix) {
            for (int c = 0; c < cols; c++)
                for (int r = 0; r < rows; r++) {
                    int srcIdx = transpose ? r * cols + c : c * rows + r;
                    if (kind == SrcKind::Double) {
                        double v = ((const double*)data)[i * comps + srcIdx];
                        memcpy(dst + c * matStride + r * 8, &v, 8);
                    } else {
                        float v = ((const float*)data)[i * comps + srcIdx];
                        memcpy(dst + c * matStride + r * 4, &v, 4);
                    }
                }
        } else if (ts.base == BaseType::Bool) {
            for (int k = 0; k < comps; k++) {
                uint32_t b;
                if (kind == SrcKind::Float) b = ((const float*)data)[i * comps + k] != 0.0f;
                else b = ((const uint32_t*)data)[i * comps + k] != 0;
                memcpy(dst + 4 * k, &b, 4);
            }
        } else {
            memcpy(dst, (const uint8_t*)data + (size_t)i * comps * elemSize, comps * elemSize);
        }
    }
    p->uniformVersion++;
}

static ProgramObj* activeUniformProgram(Context* ctx) {
    if (ctx->program) return ctx->program.get();
    if (ctx->pipeline) return ctx->pipeline->activeProgram.get();
    return nullptr;
}

#define UNIFORM_SCALARS(N, T, Kind, suffix)                                                                                  \
    API void APIENTRY glUniform1##suffix(GLint loc, T v0) {                                                                    \
        MGL_CTX();                                                                                                           \
        T v[1] = {v0};                                                                                                       \
        setUniform(ctx, activeUniformProgram(ctx), loc, 1, Kind, 1, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform2##suffix(GLint loc, T v0, T v1) {                                                              \
        MGL_CTX();                                                                                                           \
        T v[2] = {v0, v1};                                                                                                   \
        setUniform(ctx, activeUniformProgram(ctx), loc, 1, Kind, 2, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform3##suffix(GLint loc, T v0, T v1, T v2) {                                                        \
        MGL_CTX();                                                                                                           \
        T v[3] = {v0, v1, v2};                                                                                               \
        setUniform(ctx, activeUniformProgram(ctx), loc, 1, Kind, 3, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform4##suffix(GLint loc, T v0, T v1, T v2, T v3) {                                                  \
        MGL_CTX();                                                                                                           \
        T v[4] = {v0, v1, v2, v3};                                                                                           \
        setUniform(ctx, activeUniformProgram(ctx), loc, 1, Kind, 4, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform1##suffix##v(GLint loc, GLsizei c, const T* v) {                                                \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, activeUniformProgram(ctx), loc, c, Kind, 1, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform2##suffix##v(GLint loc, GLsizei c, const T* v) {                                                \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, activeUniformProgram(ctx), loc, c, Kind, 2, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform3##suffix##v(GLint loc, GLsizei c, const T* v) {                                                \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, activeUniformProgram(ctx), loc, c, Kind, 3, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glUniform4##suffix##v(GLint loc, GLsizei c, const T* v) {                                                \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, activeUniformProgram(ctx), loc, c, Kind, 4, 1, false, v);                                           \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform1##suffix(GLuint prog, GLint loc, T v0) {                                                \
        MGL_CTX();                                                                                                           \
        T v[1] = {v0};                                                                                                       \
        setUniform(ctx, getProgram(ctx, prog), loc, 1, Kind, 1, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform2##suffix(GLuint prog, GLint loc, T v0, T v1) {                                          \
        MGL_CTX();                                                                                                           \
        T v[2] = {v0, v1};                                                                                                   \
        setUniform(ctx, getProgram(ctx, prog), loc, 1, Kind, 2, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform3##suffix(GLuint prog, GLint loc, T v0, T v1, T v2) {                                    \
        MGL_CTX();                                                                                                           \
        T v[3] = {v0, v1, v2};                                                                                               \
        setUniform(ctx, getProgram(ctx, prog), loc, 1, Kind, 3, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform4##suffix(GLuint prog, GLint loc, T v0, T v1, T v2, T v3) {                              \
        MGL_CTX();                                                                                                           \
        T v[4] = {v0, v1, v2, v3};                                                                                           \
        setUniform(ctx, getProgram(ctx, prog), loc, 1, Kind, 4, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform1##suffix##v(GLuint prog, GLint loc, GLsizei c, const T* v) {                            \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, getProgram(ctx, prog), loc, c, Kind, 1, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform2##suffix##v(GLuint prog, GLint loc, GLsizei c, const T* v) {                            \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, getProgram(ctx, prog), loc, c, Kind, 2, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform3##suffix##v(GLuint prog, GLint loc, GLsizei c, const T* v) {                            \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, getProgram(ctx, prog), loc, c, Kind, 3, 1, false, v);                                               \
    }                                                                                                                        \
    API void APIENTRY glProgramUniform4##suffix##v(GLuint prog, GLint loc, GLsizei c, const T* v) {                            \
        MGL_CTX();                                                                                                           \
        setUniform(ctx, getProgram(ctx, prog), loc, c, Kind, 4, 1, false, v);                                               \
    }

UNIFORM_SCALARS(_, GLfloat, SrcKind::Float, f)
UNIFORM_SCALARS(_, GLint, SrcKind::Int, i)
UNIFORM_SCALARS(_, GLuint, SrcKind::UInt, ui)
UNIFORM_SCALARS(_, GLdouble, SrcKind::Double, d)

#define UNIFORM_MATRIX(name, T, Kind, C, R)                                                                              \
    API void APIENTRY glUniformMatrix##name(GLint loc, GLsizei c, GLboolean t, const T* v) {                              \
        MGL_CTX();                                                                                                       \
        setUniform(ctx, activeUniformProgram(ctx), loc, c, Kind, C * R, C, t, v);                                       \
    }                                                                                                                    \
    API void APIENTRY glProgramUniformMatrix##name(GLuint prog, GLint loc, GLsizei c, GLboolean t, const T* v) {          \
        MGL_CTX();                                                                                                       \
        setUniform(ctx, getProgram(ctx, prog), loc, c, Kind, C * R, C, t, v);                                           \
    }

UNIFORM_MATRIX(2fv, GLfloat, SrcKind::Float, 2, 2)
UNIFORM_MATRIX(3fv, GLfloat, SrcKind::Float, 3, 3)
UNIFORM_MATRIX(4fv, GLfloat, SrcKind::Float, 4, 4)
UNIFORM_MATRIX(2x3fv, GLfloat, SrcKind::Float, 2, 3)
UNIFORM_MATRIX(3x2fv, GLfloat, SrcKind::Float, 3, 2)
UNIFORM_MATRIX(2x4fv, GLfloat, SrcKind::Float, 2, 4)
UNIFORM_MATRIX(4x2fv, GLfloat, SrcKind::Float, 4, 2)
UNIFORM_MATRIX(3x4fv, GLfloat, SrcKind::Float, 3, 4)
UNIFORM_MATRIX(4x3fv, GLfloat, SrcKind::Float, 4, 3)
UNIFORM_MATRIX(2dv, GLdouble, SrcKind::Double, 2, 2)
UNIFORM_MATRIX(3dv, GLdouble, SrcKind::Double, 3, 3)
UNIFORM_MATRIX(4dv, GLdouble, SrcKind::Double, 4, 4)
UNIFORM_MATRIX(2x3dv, GLdouble, SrcKind::Double, 2, 3)
UNIFORM_MATRIX(3x2dv, GLdouble, SrcKind::Double, 3, 2)
UNIFORM_MATRIX(2x4dv, GLdouble, SrcKind::Double, 2, 4)
UNIFORM_MATRIX(4x2dv, GLdouble, SrcKind::Double, 4, 2)
UNIFORM_MATRIX(3x4dv, GLdouble, SrcKind::Double, 3, 4)
UNIFORM_MATRIX(4x3dv, GLdouble, SrcKind::Double, 4, 3)

// --- glGetUniform ---
template <typename T>
static void getUniform(Context* ctx, GLuint program, GLint location, GLsizei bufSize, T* out) {
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!p->linked) return ctx->setError(GL_INVALID_OPERATION);
    if (location < 0 || location >= (GLint)p->locations.size() || p->locations[location].uniform < 0) return ctx->setError(GL_INVALID_OPERATION);
    auto loc = p->locations[location];
    const glsl::Uniform& u = p->refl.uniforms[loc.uniform];
    TypeShape ts = typeShape(u.type);
    int comps = ts.components();
    if (bufSize >= 0 && (size_t)bufSize < comps * sizeof(T)) return ctx->setError(GL_INVALID_OPERATION);
    if (ts.base == BaseType::Sampler || ts.base == BaseType::Image) {
        for (size_t s = 0; s < p->refl.slots.size(); s++) {
            const auto& sl = p->refl.slots[s];
            if ((int)sl.index == loc.uniform && (int)sl.element == loc.element) {
                out[0] = (T)p->samplerUnits[s];
                return;
            }
        }
        out[0] = (T)u.binding;
        return;
    }
    int elemSize = ts.base == BaseType::Double ? 8 : 4;
    int matStride = u.matrixStride ? u.matrixStride : ts.rows * elemSize;
    int arrayStride = u.arrayStride ? u.arrayStride : (ts.cols > 1 ? ts.cols * matStride : comps * elemSize);
    const uint8_t* src = p->uniformData.data() + u.offset + (size_t)loc.element * arrayStride;
    for (int c = 0; c < ts.cols; c++)
        for (int r = 0; r < ts.rows; r++) {
            const uint8_t* e = src + c * matStride + r * elemSize;
            T v;
            switch (ts.base) {
            case BaseType::Float: { float f; memcpy(&f, e, 4); v = (T)f; break; }
            case BaseType::Double: { double d; memcpy(&d, e, 8); v = (T)d; break; }
            case BaseType::Int: { int32_t i; memcpy(&i, e, 4); v = (T)i; break; }
            case BaseType::UInt: case BaseType::Bool: { uint32_t i; memcpy(&i, e, 4); v = (T)i; break; }
            default: v = 0;
            }
            out[c * ts.rows + r] = v;
        }
}

API void APIENTRY glGetUniformfv(GLuint program, GLint location, GLfloat* params) { MGL_CTX(); getUniform(ctx, program, location, -1, params); }
API void APIENTRY glGetUniformiv(GLuint program, GLint location, GLint* params) { MGL_CTX(); getUniform(ctx, program, location, -1, params); }
API void APIENTRY glGetUniformuiv(GLuint program, GLint location, GLuint* params) { MGL_CTX(); getUniform(ctx, program, location, -1, params); }
API void APIENTRY glGetUniformdv(GLuint program, GLint location, GLdouble* params) { MGL_CTX(); getUniform(ctx, program, location, -1, params); }
API void APIENTRY glGetnUniformfv(GLuint program, GLint location, GLsizei bufSize, GLfloat* params) { MGL_CTX(); getUniform(ctx, program, location, bufSize, params); }
API void APIENTRY glGetnUniformiv(GLuint program, GLint location, GLsizei bufSize, GLint* params) { MGL_CTX(); getUniform(ctx, program, location, bufSize, params); }
API void APIENTRY glGetnUniformuiv(GLuint program, GLint location, GLsizei bufSize, GLuint* params) { MGL_CTX(); getUniform(ctx, program, location, bufSize, params); }
API void APIENTRY glGetnUniformdv(GLuint program, GLint location, GLsizei bufSize, GLdouble* params) { MGL_CTX(); getUniform(ctx, program, location, bufSize, params); }

// ---------------------------------------------------------------------------
// program interface query (GL 4.3)
// ---------------------------------------------------------------------------
namespace {
struct Resource {
    std::string name;
    GLenum type = 0;
    int arraySize = 1, location = -1, locationIndex = -1, blockIndex = -1, offset = -1, arrayStride = -1, matrixStride = -1;
    int isRowMajor = 0, bufferBinding = -1, bufferDataSize = 0, atomicIndex = -1, topLevelArraySize = 1, topLevelArrayStride = 0;
    uint32_t stageMask = 0;
    std::vector<int> activeVariables;
    bool isBlock = false, isPerPatch = false;
};
}  // namespace

static std::vector<Resource> resourcesFor(ProgramObj* p, GLenum iface, bool* ok) {
    *ok = true;
    std::vector<Resource> out;
    const glsl::Reflection& r = p->refl;
    switch (iface) {
    case GL_UNIFORM:
        for (auto& u : r.uniforms) {
            Resource x;
            x.name = u.name;
            x.type = u.type;
            x.arraySize = u.arraySize;
            x.location = u.blockIndex < 0 && !(typeShape(u.type).base == BaseType::AtomicCounter) ? u.location : -1;
            x.blockIndex = u.blockIndex;
            x.offset = u.offset;
            x.arrayStride = u.blockIndex >= 0 || typeShape(u.type).base == BaseType::AtomicCounter ? u.arrayStride : -1;
            x.matrixStride = u.blockIndex >= 0 ? u.matrixStride : -1;
            if (u.blockIndex < 0) x.offset = typeShape(u.type).base == BaseType::AtomicCounter ? u.offset : -1;
            x.isRowMajor = u.rowMajor;
            x.atomicIndex = u.atomicCounterBufferIndex;
            x.stageMask = u.stageMask;
            out.push_back(x);
        }
        break;
    case GL_UNIFORM_BLOCK:
    case GL_SHADER_STORAGE_BLOCK: {
        auto& blocks = iface == GL_UNIFORM_BLOCK ? r.uniformBlocks : r.storageBlocks;
        auto& bindings = iface == GL_UNIFORM_BLOCK ? p->blockBindings : p->storageBindings;
        for (size_t i = 0; i < blocks.size(); i++) {
            Resource x;
            x.name = blocks[i].name;
            x.bufferBinding = bindings[i];
            x.bufferDataSize = blocks[i].dataSize;
            x.activeVariables = blocks[i].activeUniforms;
            x.stageMask = blocks[i].stageMask;
            x.isBlock = true;
            out.push_back(x);
        }
        break;
    }
    case GL_BUFFER_VARIABLE:
        for (auto& v : r.bufferVariables) {
            Resource x;
            x.name = v.name;
            x.type = v.type;
            x.arraySize = v.arraySize;
            x.blockIndex = v.blockIndex;
            x.offset = v.offset;
            x.arrayStride = v.arrayStride;
            x.matrixStride = v.matrixStride;
            x.isRowMajor = v.rowMajor;
            x.topLevelArraySize = v.topLevelArraySize;
            x.topLevelArrayStride = v.topLevelArrayStride;
            if (v.blockIndex >= 0 && (size_t)v.blockIndex < r.storageBlocks.size()) x.stageMask = r.storageBlocks[v.blockIndex].stageMask;
            out.push_back(x);
        }
        break;
    case GL_PROGRAM_INPUT:
        for (auto& a : r.attributes) {
            Resource x;
            x.name = a.name;
            x.type = a.type;
            x.arraySize = a.arraySize;
            x.location = a.location;
            x.stageMask = 1 << (int)glsl::Stage::Vertex;
            out.push_back(x);
        }
        break;
    case GL_PROGRAM_OUTPUT:
        for (auto& o : r.fragOutputs) {
            Resource x;
            x.name = o.name;
            x.type = o.type;
            x.arraySize = o.arraySize;
            x.location = o.location;
            x.locationIndex = o.index;
            x.stageMask = 1 << (int)glsl::Stage::Fragment;
            out.push_back(x);
        }
        break;
    case GL_TRANSFORM_FEEDBACK_VARYING:
        for (auto& v : r.xfbVaryings) {
            Resource x;
            x.name = v.name;
            x.type = v.type;
            x.arraySize = v.arraySize;
            x.offset = v.offset;
            out.push_back(x);
        }
        break;
    case GL_ATOMIC_COUNTER_BUFFER:
        for (auto& a : r.atomicCounterBuffers) {
            Resource x;
            x.bufferBinding = a.binding;
            x.bufferDataSize = a.dataSize;
            x.activeVariables = a.activeUniforms;
            x.stageMask = a.stageMask;
            out.push_back(x);
        }
        break;
    case GL_TRANSFORM_FEEDBACK_BUFFER:
    case GL_VERTEX_SUBROUTINE: case GL_TESS_CONTROL_SUBROUTINE: case GL_TESS_EVALUATION_SUBROUTINE: case GL_GEOMETRY_SUBROUTINE:
    case GL_FRAGMENT_SUBROUTINE: case GL_COMPUTE_SUBROUTINE: case GL_VERTEX_SUBROUTINE_UNIFORM: case GL_TESS_CONTROL_SUBROUTINE_UNIFORM:
    case GL_TESS_EVALUATION_SUBROUTINE_UNIFORM: case GL_GEOMETRY_SUBROUTINE_UNIFORM: case GL_FRAGMENT_SUBROUTINE_UNIFORM:
    case GL_COMPUTE_SUBROUTINE_UNIFORM:
        break;
    default: *ok = false;
    }
    return out;
}

API void APIENTRY glGetProgramInterfaceiv(GLuint program, GLenum iface, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    bool ok;
    auto res = resourcesFor(p, iface, &ok);
    if (!ok) return ctx->setError(GL_INVALID_ENUM);
    switch (pname) {
    case GL_ACTIVE_RESOURCES: *params = (GLint)res.size(); break;
    case GL_MAX_NAME_LENGTH: {
        if (iface == GL_ATOMIC_COUNTER_BUFFER || iface == GL_TRANSFORM_FEEDBACK_BUFFER) return ctx->setError(GL_INVALID_OPERATION);
        GLint m = 0;
        for (auto& x : res) m = std::max<GLint>(m, (GLint)x.name.size() + 1);
        *params = m;
        break;
    }
    case GL_MAX_NUM_ACTIVE_VARIABLES: {
        if (iface != GL_UNIFORM_BLOCK && iface != GL_SHADER_STORAGE_BLOCK && iface != GL_ATOMIC_COUNTER_BUFFER &&
            iface != GL_TRANSFORM_FEEDBACK_BUFFER)
            return ctx->setError(GL_INVALID_OPERATION);
        GLint m = 0;
        for (auto& x : res) m = std::max<GLint>(m, (GLint)x.activeVariables.size());
        *params = m;
        break;
    }
    case GL_MAX_NUM_COMPATIBLE_SUBROUTINES: *params = 0; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API GLuint APIENTRY glGetProgramResourceIndex(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(GL_INVALID_INDEX);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return GL_INVALID_INDEX;
    bool ok;
    auto res = resourcesFor(p, iface, &ok);
    if (!ok || iface == GL_ATOMIC_COUNTER_BUFFER || iface == GL_TRANSFORM_FEEDBACK_BUFFER) {
        ctx->setError(GL_INVALID_ENUM);
        return GL_INVALID_INDEX;
    }
    std::string n = name;
    for (size_t i = 0; i < res.size(); i++) {
        const std::string& rn = res[i].name;
        if (rn == n) return (GLuint)i;
        if (rn.size() > 3 && rn.compare(rn.size() - 3, 3, "[0]") == 0 && rn.substr(0, rn.size() - 3) == n) return (GLuint)i;
    }
    return GL_INVALID_INDEX;
}

API void APIENTRY glGetProgramResourceName(GLuint program, GLenum iface, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    bool ok;
    auto res = resourcesFor(p, iface, &ok);
    if (!ok) return ctx->setError(GL_INVALID_ENUM);
    if (index >= res.size()) return ctx->setError(GL_INVALID_VALUE);
    copyLog(res[index].name, bufSize, length, name);
}

API void APIENTRY glGetProgramResourceiv(GLuint program, GLenum iface, GLuint index, GLsizei propCount, const GLenum* props, GLsizei bufSize,
                                         GLsizei* length, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    bool ok;
    auto res = resourcesFor(p, iface, &ok);
    if (!ok) return ctx->setError(GL_INVALID_ENUM);
    if (index >= res.size()) return ctx->setError(GL_INVALID_VALUE);
    const Resource& x = res[index];
    std::vector<GLint> vals;
    using S = glsl::Stage;
    for (int i = 0; i < propCount; i++) {
        switch (props[i]) {
        case GL_NAME_LENGTH: vals.push_back((GLint)x.name.size() + 1); break;
        case GL_TYPE: vals.push_back(x.type); break;
        case GL_ARRAY_SIZE: vals.push_back(x.arraySize); break;
        case GL_OFFSET: vals.push_back(x.offset); break;
        case GL_BLOCK_INDEX: vals.push_back(x.blockIndex); break;
        case GL_ARRAY_STRIDE: vals.push_back(x.arrayStride); break;
        case GL_MATRIX_STRIDE: vals.push_back(x.matrixStride); break;
        case GL_IS_ROW_MAJOR: vals.push_back(x.isRowMajor); break;
        case GL_ATOMIC_COUNTER_BUFFER_INDEX: vals.push_back(x.atomicIndex); break;
        case GL_BUFFER_BINDING: vals.push_back(x.bufferBinding); break;
        case GL_BUFFER_DATA_SIZE: vals.push_back(x.bufferDataSize); break;
        case GL_NUM_ACTIVE_VARIABLES: vals.push_back((GLint)x.activeVariables.size()); break;
        case GL_ACTIVE_VARIABLES: for (int v : x.activeVariables) vals.push_back(v); break;
        case GL_REFERENCED_BY_VERTEX_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::Vertex))); break;
        case GL_REFERENCED_BY_TESS_CONTROL_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::TessControl))); break;
        case GL_REFERENCED_BY_TESS_EVALUATION_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::TessEval))); break;
        case GL_REFERENCED_BY_GEOMETRY_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::Geometry))); break;
        case GL_REFERENCED_BY_FRAGMENT_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::Fragment))); break;
        case GL_REFERENCED_BY_COMPUTE_SHADER: vals.push_back(!!(x.stageMask & (1 << (int)S::Compute))); break;
        case GL_TOP_LEVEL_ARRAY_SIZE: vals.push_back(x.topLevelArraySize); break;
        case GL_TOP_LEVEL_ARRAY_STRIDE: vals.push_back(x.topLevelArrayStride); break;
        case GL_LOCATION: vals.push_back(x.location); break;
        case GL_LOCATION_INDEX: vals.push_back(x.locationIndex); break;
        case GL_IS_PER_PATCH: vals.push_back(x.isPerPatch); break;
        case GL_LOCATION_COMPONENT: vals.push_back(0); break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_INDEX: vals.push_back(0); break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE: vals.push_back(0); break;
        default: return ctx->setError(GL_INVALID_ENUM);
        }
    }
    GLsizei n = std::min<GLsizei>(bufSize, (GLsizei)vals.size());
    for (int i = 0; i < n; i++) params[i] = vals[i];
    if (length) *length = n;
}

API GLint APIENTRY glGetProgramResourceLocation(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return -1;
    }
    switch (iface) {
    case GL_UNIFORM: return uniformLocation(p, name);
    case GL_PROGRAM_INPUT: return glGetAttribLocation(program, name);
    case GL_PROGRAM_OUTPUT: return glGetFragDataLocation(program, name);
    default:
        if (iface >= GL_VERTEX_SUBROUTINE_UNIFORM && iface <= GL_COMPUTE_SUBROUTINE_UNIFORM) return -1;
        ctx->setError(GL_INVALID_ENUM);
        return -1;
    }
}

API GLint APIENTRY glGetProgramResourceLocationIndex(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(-1);
    if (iface != GL_PROGRAM_OUTPUT) {
        ctx->setError(GL_INVALID_ENUM);
        return -1;
    }
    return glGetFragDataIndex(program, name);
}

// ---------------------------------------------------------------------------
// separable programs & pipelines
// ---------------------------------------------------------------------------
API GLuint APIENTRY glCreateShaderProgramv(GLenum type, GLsizei count, const GLchar* const* strings) {
    MGL_CTX_RET(0);
    GLuint s = glCreateShader(type);
    if (!s) return 0;
    glShaderSource(s, count, strings, nullptr);
    glCompileShader(s);
    GLuint p = glCreateProgram();
    ProgramObj* po = getProgram(ctx, p);
    po->separable = true;
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (ok) {
        glAttachShader(p, s);
        glLinkProgram(p);
        glDetachShader(p, s);
    }
    ShaderObj* so = getShader(ctx, s);
    po->infoLog = so->infoLog + po->infoLog;
    glDeleteShader(s);
    return p;
}

API void APIENTRY glGenProgramPipelines(GLsizei n, GLuint* pipelines) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) pipelines[i] = ctx->pipelines.genName();
}

API void APIENTRY glCreateProgramPipelines(GLsizei n, GLuint* pipelines) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        pipelines[i] = ctx->pipelines.genName();
        ctx->pipelines.create(pipelines[i], new ProgramPipelineObj());
    }
}

API void APIENTRY glDeleteProgramPipelines(GLsizei n, const GLuint* pipelines) {
    MGL_CTX();
    for (int i = 0; i < n; i++) {
        ProgramPipelineObj* pp = ctx->pipelines.get(pipelines[i]);
        if (pp && ctx->pipeline.get() == pp) ctx->pipeline = nullptr;
        ctx->pipelines.remove(pipelines[i]);
    }
}

API GLboolean APIENTRY glIsProgramPipeline(GLuint pipeline) {
    MGL_CTX_RET(GL_FALSE);
    return pipeline && ctx->pipelines.get(pipeline) != nullptr;
}

static ProgramPipelineObj* pipelineObj(Context* ctx, GLuint name, bool create) {
    ProgramPipelineObj* pp = ctx->pipelines.get(name);
    if (!pp && create && ctx->pipelines.isName(name)) pp = ctx->pipelines.create(name, new ProgramPipelineObj());
    if (!pp) ctx->setError(GL_INVALID_OPERATION);
    return pp;
}

API void APIENTRY glBindProgramPipeline(GLuint pipeline) {
    MGL_CTX();
    if (!pipeline) {
        ctx->pipeline = nullptr;
        return;
    }
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (pp) ctx->pipeline = pp;
}

API void APIENTRY glUseProgramStages(GLuint pipeline, GLbitfield stages, GLuint program) {
    MGL_CTX();
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (!pp) return;
    const GLbitfield all = GL_VERTEX_SHADER_BIT | GL_TESS_CONTROL_SHADER_BIT | GL_TESS_EVALUATION_SHADER_BIT | GL_GEOMETRY_SHADER_BIT |
                           GL_FRAGMENT_SHADER_BIT | GL_COMPUTE_SHADER_BIT;
    if (stages != GL_ALL_SHADER_BITS && (stages & ~all)) return ctx->setError(GL_INVALID_VALUE);
    ProgramObj* p = nullptr;
    if (program) {
        p = getProgram(ctx, program);
        if (!p) return;
        if (!p->separable || !p->linked) return ctx->setError(GL_INVALID_OPERATION);
    }
    const std::pair<GLbitfield, glsl::Stage> map[] = {
        {GL_VERTEX_SHADER_BIT, glsl::Stage::Vertex}, {GL_TESS_CONTROL_SHADER_BIT, glsl::Stage::TessControl},
        {GL_TESS_EVALUATION_SHADER_BIT, glsl::Stage::TessEval}, {GL_GEOMETRY_SHADER_BIT, glsl::Stage::Geometry},
        {GL_FRAGMENT_SHADER_BIT, glsl::Stage::Fragment}, {GL_COMPUTE_SHADER_BIT, glsl::Stage::Compute}};
    for (auto& [bit, st] : map) {
        if (!(stages & bit)) continue;
        pp->stages[(int)st] = (p && (p->refl.stageMask & (1u << (int)st))) ? p : nullptr;
    }
    pp->validated = false;
}

API void APIENTRY glActiveShaderProgram(GLuint pipeline, GLuint program) {
    MGL_CTX();
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (!pp) return;
    ProgramObj* p = program ? getProgram(ctx, program) : nullptr;
    if (program && !p) return;
    if (p && !p->linked) return ctx->setError(GL_INVALID_OPERATION);
    pp->activeProgram = p;
}

API void APIENTRY glValidateProgramPipeline(GLuint pipeline) {
    MGL_CTX();
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (!pp) return;
    pp->validated = pp->stages[(int)glsl::Stage::Vertex] || pp->stages[(int)glsl::Stage::Compute];
    pp->infoLog = pp->validated ? "" : "pipeline has no vertex or compute program\n";
}

API void APIENTRY glGetProgramPipelineiv(GLuint pipeline, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (!pp) return;
    auto nm = [](ProgramObj* p) { return p ? (GLint)p->name : 0; };
    switch (pname) {
    case GL_ACTIVE_PROGRAM: *params = nm(pp->activeProgram.get()); break;
    case GL_VERTEX_SHADER: *params = nm(pp->stages[(int)glsl::Stage::Vertex].get()); break;
    case GL_TESS_CONTROL_SHADER: *params = nm(pp->stages[(int)glsl::Stage::TessControl].get()); break;
    case GL_TESS_EVALUATION_SHADER: *params = nm(pp->stages[(int)glsl::Stage::TessEval].get()); break;
    case GL_GEOMETRY_SHADER: *params = nm(pp->stages[(int)glsl::Stage::Geometry].get()); break;
    case GL_FRAGMENT_SHADER: *params = nm(pp->stages[(int)glsl::Stage::Fragment].get()); break;
    case GL_COMPUTE_SHADER: *params = nm(pp->stages[(int)glsl::Stage::Compute].get()); break;
    case GL_VALIDATE_STATUS: *params = pp->validated; break;
    case GL_INFO_LOG_LENGTH: *params = pp->infoLog.empty() ? 0 : (GLint)pp->infoLog.size() + 1; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetProgramPipelineInfoLog(GLuint pipeline, GLsizei bufSize, GLsizei* length, GLchar* infoLog) {
    MGL_CTX();
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (pp) copyLog(pp->infoLog, bufSize, length, infoLog);
}

// --- subroutines (ARB_shader_subroutine): no subroutines are ever active yet ---
API GLint APIENTRY glGetSubroutineUniformLocation(GLuint program, GLenum shadertype, const GLchar* name) { return -1; }
API GLuint APIENTRY glGetSubroutineIndex(GLuint program, GLenum shadertype, const GLchar* name) { return GL_INVALID_INDEX; }
API void APIENTRY glUniformSubroutinesuiv(GLenum shadertype, GLsizei count, const GLuint* indices) {
    MGL_CTX();
    if (count != 0) ctx->setError(GL_INVALID_VALUE);
}
API void APIENTRY glGetUniformSubroutineuiv(GLenum shadertype, GLint location, GLuint* params) {
    MGL_CTX();
    ctx->setError(GL_INVALID_VALUE);
}
API void APIENTRY glGetProgramStageiv(GLuint program, GLenum shadertype, GLenum pname, GLint* values) { *values = 0; }
