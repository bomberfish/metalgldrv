#include <limits>
#include <type_traits>
#include <cmath>
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
void setupLinkedProgram(ProgramObj* p) {
    const glsl::Reflection& r = p->refl;
    p->uniformData.assign(std::max<uint32_t>(r.defaultUniformSize, 16), 0);
    // Uniform initializers (`uniform float x = 1.0;`)
    if (!r.defaultUniformInit.empty())
        memcpy(p->uniformData.data(), r.defaultUniformInit.data(), std::min(r.defaultUniformInit.size(), p->uniformData.size()));
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
        if (sl.kind == glsl::SlotKind::Texture || sl.kind == glsl::SlotKind::Sampler || sl.kind == glsl::SlotKind::Image ||
            sl.kind == glsl::SlotKind::LodBias ||
            sl.kind == glsl::SlotKind::SamplerBorder) {
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

// A shader flagged for deletion keeps its name until it is no longer attached anywhere.
static void maybeFreeShader(Context* ctx, ShaderObj* s) {
    if (s->deleted && s->attachCount <= 0 && ctx->share->shaderPrograms.get(s->name) == s) ctx->share->shaderPrograms.remove(s->name);
}

API void APIENTRY glDeleteShader(GLuint shader) {
    MGL_CTX();
    if (!shader) return;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    s->deleted = true;
    maybeFreeShader(ctx, s);
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
    s->spirvBinary = false;  // GL 4.6 §7.2: ShaderSource clears SPIR_V_BINARY
    s->specialized = false;
    s->spirvModule.reset();
    for (int i = 0; i < count; i++) {
        if (!strings[i]) {
            s->sources.emplace_back();
            continue;
        }
        if (lengths && lengths[i] >= 0) s->sources.emplace_back(strings[i], lengths[i]);
        else s->sources.emplace_back(strings[i]);
    }
    // Field debugging: METALGLDRV_GLSL_PATCH="find=>replace[;;find=>replace...]" edits sources.
    if (const char* patch = getenv("METALGLDRV_GLSL_PATCH")) {
        std::string all = patch;
        size_t pos = 0;
        while (pos < all.size()) {
            size_t end = all.find(";;", pos);
            std::string rule = all.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
            pos = end == std::string::npos ? all.size() : end + 2;
            size_t arrow = rule.find("=>");
            if (arrow == std::string::npos) continue;
            std::string find = rule.substr(0, arrow), repl = rule.substr(arrow + 2);
            for (auto& src : s->sources)
                for (size_t at = src.find(find); at != std::string::npos; at = src.find(find, at + repl.size())) {
                    src.replace(at, find.size(), repl);
                    MGL_INFO("shader %u: applied GLSL patch '%s'", shader, find.c_str());
                }
        }
    }
}

API void APIENTRY glCompileShader(GLuint shader) {
    MGL_CTX();
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    if (s->spirvBinary) return ctx->setError(GL_INVALID_OPERATION, "CompileShader on a SPIR-V shader");
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
        if (envFlag("METALGLDRV_DUMP_FAILED_GLSL")) {
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
    case GL_COMPLETION_STATUS_ARB: *params = GL_TRUE; break;  // compiles are synchronous
    case GL_INFO_LOG_LENGTH: *params = s->infoLog.empty() ? 0 : (GLint)s->infoLog.size() + 1; break;
    case GL_SHADER_SOURCE_LENGTH: {
        size_t n = 0;
        for (auto& x : s->sources) n += x.size();
        *params = n ? (GLint)n + 1 : 0;
        break;
    }
    case GL_SPIR_V_BINARY: *params = s->spirvBinary; break;
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

// ARB_gl_spirv (GL 4.6 §7.2): the only shader binary format is SPIR-V. The module is stored on every
// shader in `shaders`; each must then be specialized (glSpecializeShader) before it can be linked.
API void APIENTRY glShaderBinary(GLsizei count, const GLuint* shaders, GLenum binaryFormat, const void* binary, GLsizei length) {
    MGL_CTX();
    if (count < 0 || length < 0) return ctx->setError(GL_INVALID_VALUE);
    if (binaryFormat != GL_SHADER_BINARY_FORMAT_SPIR_V) return ctx->setError(GL_INVALID_ENUM, "unsupported shader binary format 0x%x", binaryFormat);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    std::vector<ShaderObj*> objs;
    for (GLsizei i = 0; i < count; i++) {
        ShaderObj* s = getShader(ctx, shaders[i]);
        if (!s) return;
        for (ShaderObj* o : objs)
            if (o == s) return ctx->setError(GL_INVALID_OPERATION, "shader %u listed twice", shaders[i]);
        objs.push_back(s);
    }
    std::string err;
    auto mod = glsl::parseSpirv(binary, (size_t)length, &err);
    if (!mod) return ctx->setError(GL_INVALID_VALUE, "glShaderBinary: %s", err.c_str());
    for (ShaderObj* s : objs) {
        s->spirvModule = mod;
        s->spirvBinary = true;
        s->specialized = false;
        s->compiled = false;
        s->shader.reset();
        s->infoLog.clear();
        s->sources.clear();
    }
}

API void APIENTRY glSpecializeShader(GLuint shader, const GLchar* pEntryPoint, GLuint n, const GLuint* idx, const GLuint* val) {
    MGL_CTX();
    ShaderObj* s = getShader(ctx, shader);
    if (!s) return;
    if (!s->spirvBinary || !s->spirvModule) return ctx->setError(GL_INVALID_OPERATION, "shader %u has no SPIR-V module", shader);
    if (s->specialized) return ctx->setError(GL_INVALID_OPERATION, "shader %u is already specialized", shader);
    if (!pEntryPoint) return ctx->setError(GL_INVALID_VALUE);
    if (n && (!idx || !val)) return ctx->setError(GL_INVALID_VALUE);
    bool ok;
    glsl::Stage st = stageForType(s->type, &ok);
    std::vector<std::pair<uint32_t, uint32_t>> constants;
    for (GLuint i = 0; i < n; i++) constants.push_back({idx[i], val[i]});
    glsl::CompileOptions opts;
    opts.compatibilityProfile = false;
    glsl::SpecializeResult r = glsl::specializeSpirv(*s->spirvModule, st, pEntryPoint, constants, opts);
    switch (r.status) {
    case glsl::SpecializeStatus::BadEntryPoint: return ctx->setError(GL_INVALID_VALUE, "no SPIR-V entry point '%s' for this stage", pEntryPoint);
    case glsl::SpecializeStatus::BadConstantId: return ctx->setError(GL_INVALID_VALUE, "unknown specialization constant id");
    default: break;
    }
    s->specialized = true;
    s->shader = r.compile.shader;
    s->compiled = r.compile.shader != nullptr;
    s->infoLog = r.compile.infoLog;
    if (!s->compiled) {
        MGL_DBG("shader %u SPIR-V specialization failed:\n%s", shader, r.compile.infoLog.c_str());
        if (envFlag("METALGLDRV_DUMP_FAILED_GLSL")) MGL_ERR("shader %u SPIR-V specialization failed:\n%s\n---- generated GLSL ----\n%s", shader,
                                                           r.compile.infoLog.c_str(), r.glsl.c_str());
    }
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
    // A program in use is only flagged (its name stays valid, GL 4.6 7.3); it is freed
    // when it stops being the current program.
    if (ctx->program.get() == p) {
        p->deleted = true;
        return;
    }
    for (auto& a : p->attached) {
        a->attachCount--;
        maybeFreeShader(ctx, a.get());
    }
    p->attached.clear();
    ctx->share->shaderPrograms.remove(program);
}

// Frees the current program's name if it was flagged for deletion while in use.
static void releaseCurrentProgram(Context* ctx) {
    ProgramObj* old = ctx->program.get();
    if (!old || !old->deleted) return;
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    if (ctx->share->shaderPrograms.get(old->name) != old) return;
    for (auto& a : old->attached) {
        a->attachCount--;
        maybeFreeShader(ctx, a.get());
    }
    old->attached.clear();
    ctx->share->shaderPrograms.remove(old->name);
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
    s->attachCount++;
}

API void APIENTRY glDetachShader(GLuint program, GLuint shader) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    Object* o = ctx->share->shaderPrograms.get(shader);
    for (auto it = p->attached.begin(); it != p->attached.end(); ++it) {
        if (it->get()->name == shader) {
            Ref<ShaderObj> s = *it;
            p->attached.erase(it);
            s->attachCount--;
            maybeFreeShader(ctx, s.get());
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
    // gl_NextBuffer / gl_SkipComponents* are only valid with interleaved capture
    if (bufferMode == GL_SEPARATE_ATTRIBS)
        for (int i = 0; i < count; i++)
            if (varyings[i] && (!strcmp(varyings[i], "gl_NextBuffer") || !strncmp(varyings[i], "gl_SkipComponents", 17)))
                return ctx->setError(GL_INVALID_OPERATION);
    int nextBuffers = 0;
    for (int i = 0; i < count; i++)
        if (varyings[i] && !strcmp(varyings[i], "gl_NextBuffer")) nextBuffers++;
    if (nextBuffers >= kMaxXfbBuffers) return ctx->setError(GL_INVALID_OPERATION);
    p->xfbVaryings.clear();
    for (int i = 0; i < count; i++) p->xfbVaryings.push_back(varyings[i]);
    p->xfbBufferMode = bufferMode;
}

API void APIENTRY glProgramParameteri(GLuint program, GLenum pname, GLint value) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (pname != GL_PROGRAM_SEPARABLE && pname != GL_PROGRAM_BINARY_RETRIEVABLE_HINT) return ctx->setError(GL_INVALID_ENUM);
    if (value != GL_TRUE && value != GL_FALSE) return ctx->setError(GL_INVALID_VALUE);
    if (pname == GL_PROGRAM_SEPARABLE) p->separable = value;
    else p->retrievableHint = value;
}

API void APIENTRY glLinkProgram(GLuint program) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (ctx->xfb->active && ctx->xfb->program.get() == p) return ctx->setError(GL_INVALID_OPERATION);
    glsl::LinkInput in;
    bool allCompiled = true;
    int spirvShaders = 0;
    for (auto& s : p->attached) {
        if (s->spirvBinary) spirvShaders++;
        if (!s->compiled || !s->shader) allCompiled = false;
        else in.shaders.push_back(s->shader);
    }
    if (spirvShaders && spirvShaders != (int)p->attached.size()) {
        // GL 4.6 §7.3: all attached shaders must have the same SPIR_V_BINARY state
        p->linkGeneration++;
        p->linked = false;
        p->infoLog = "error: SPIR-V and GLSL shaders cannot be linked into one program\n";
        return;
    }
    in.attribBindings = p->attribBindings;
    in.fragDataBindings = p->fragDataBindings;
    in.fragDataIndices = p->fragDataIndices;
    in.xfbVaryings = p->xfbVaryings;
    in.xfbInterleaved = p->xfbBufferMode == GL_INTERLEAVED_ATTRIBS;
    in.separable = p->separable;
    p->linkGeneration++;
    // A failed (re)link keeps the previous executable (p->program, refl, variants): GL keeps
    // using it while the program stays current (§7.3); only LINK_STATUS turns false.
    if (!allCompiled || in.shaders.empty()) {
        p->linked = false;
        p->infoLog = in.shaders.empty() ? "error: no shaders attached\n" : "error: attached shader not successfully compiled\n";
        MGL_DBG("program %u link failed: %s", program, p->infoLog.c_str());
        return;
    }
    glsl::LinkResult r = glsl::link(in);
    p->infoLog = r.infoLog;
    if (!r.program) {
        p->linked = false;
        MGL_DBG("program %u link failed:\n%s", program, r.infoLog.c_str());
        return;
    }
    p->linked = true;
    p->linkedSeparable = p->separable;
    p->linkedSpirv = spirvShaders > 0;
    if (p->linkedSpirv) {
        // SPIR-V programs: names are optional debug information (ARB_gl_spirv); resources are
        // identified by location/binding and report empty names.
        auto& rf = r.reflection;
        for (auto& x : rf.attributes) x.name.clear();
        for (auto& x : rf.fragOutputs) x.name.clear();
        for (auto& x : rf.uniforms) x.name.clear();
        for (auto& x : rf.uniformBlocks) x.name.clear();
        for (auto& x : rf.storageBlocks) x.name.clear();
        for (auto& x : rf.bufferVariables) x.name.clear();
        for (auto& x : rf.xfbVaryings) x.name.clear();
    }
    // METALGLDRV_DUMP_GLSL=<substring>: log the sources of linked programs containing it.
    if (const char* dump = getenv("METALGLDRV_DUMP_GLSL")) {
        std::string all;
        for (auto& sh : p->attached)
            for (auto& src : sh->sources) all += src;
        if (all.find(dump) != std::string::npos)
            for (auto& sh : p->attached) {
                std::string src;
                for (auto& part : sh->sources) src += part;
                MGL_INFO("program %u shader type %x GLSL:\n%s", p->name, sh->type, src.c_str());
            }
    }
    p->program = r.program;
    p->refl = std::move(r.reflection);
    setupLinkedProgram(p);
    // Compile the common variant now so the MSL compile cost lands at link time, not mid-frame.
    uint32_t sm = p->refl.stageMask;
    bool multiPass = sm & ((1u << (int)glsl::Stage::Geometry) | (1u << (int)glsl::Stage::TessEval));
    if (!multiPass && !envFlag("METALGLDRV_LAZY_COMPILE")) {
        glsl::VariantKey vk;
        vk.kind = (sm & (1u << (int)glsl::Stage::Compute)) ? glsl::PipelineKind::Compute : glsl::PipelineKind::Classic;
        ProgramVariant* v = ctx->pipelines_->variant(p, vk);
        if (v && !v->ok) {
            // Report backend failures through the link status so apps see them.
            p->linked = false;
            p->infoLog += "error: internal Metal shader compilation failed: " + v->error + "\n";
            MGL_DBG("program %u: %s", program, p->infoLog.c_str());
        }
    }
}

// Subroutine selections are reset to defaults when the program used for a stage changes (GL 4.6 7.10).
static void resetSubroutineSelection(Context* ctx, int st, ProgramObj* p) {
    ctx->subroutineSelection[st].clear();
    if (!p) return;
    const auto& ss = p->refl.subroutines[st];
    for (int l = 0; l < ss.numLocations; l++) {
        uint32_t def = 0;
        for (auto& u : ss.uniforms)
            if (l >= u.location && l < u.location + u.arraySize && !u.compatible.empty()) def = u.compatible[0];
        ctx->subroutineSelection[st].push_back(def);
    }
}

API void APIENTRY glUseProgram(GLuint program) {
    MGL_CTX();
    if (ctx->xfb->active && !ctx->xfb->paused) return ctx->setError(GL_INVALID_OPERATION);
    if (!program) {
        releaseCurrentProgram(ctx);
        ctx->program = nullptr;
        if (ctx->pipeline)
            for (int st = 0; st < (int)glsl::Stage::Count; st++) resetSubroutineSelection(ctx, st, ctx->pipeline->stages[st].get());
        return;
    }
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!p->linked) return ctx->setError(GL_INVALID_OPERATION, "program not linked");
    if (ctx->program.get() != p) releaseCurrentProgram(ctx);
    ctx->program = p;
    for (int st = 0; st < (int)glsl::Stage::Count; st++) resetSubroutineSelection(ctx, st, p);
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
    case GL_COMPLETION_STATUS_ARB: *params = GL_TRUE; break;
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
    // GL 4.6 7.13: stage-specific queries need a linked program containing that stage
    case GL_GEOMETRY_VERTICES_OUT: case GL_GEOMETRY_INPUT_TYPE: case GL_GEOMETRY_OUTPUT_TYPE: case GL_GEOMETRY_SHADER_INVOCATIONS:
    case GL_TESS_CONTROL_OUTPUT_VERTICES: case GL_TESS_GEN_MODE: case GL_TESS_GEN_SPACING: case GL_TESS_GEN_VERTEX_ORDER:
    case GL_TESS_GEN_POINT_MODE: {
        glsl::Stage st = pname == GL_TESS_CONTROL_OUTPUT_VERTICES ? glsl::Stage::TessControl
                       : (pname == GL_TESS_GEN_MODE || pname == GL_TESS_GEN_SPACING || pname == GL_TESS_GEN_VERTEX_ORDER ||
                          pname == GL_TESS_GEN_POINT_MODE) ? glsl::Stage::TessEval : glsl::Stage::Geometry;
        if (!p->linked || !(r.stageMask & (1u << (int)st))) return ctx->setError(GL_INVALID_OPERATION);
        switch (pname) {
        case GL_GEOMETRY_VERTICES_OUT: *params = r.gsMaxVertices; break;
        case GL_GEOMETRY_INPUT_TYPE: *params = r.gsInputPrimitive; break;
        case GL_GEOMETRY_OUTPUT_TYPE: *params = r.gsOutputPrimitive; break;
        case GL_GEOMETRY_SHADER_INVOCATIONS: *params = r.gsInvocations; break;
        case GL_TESS_CONTROL_OUTPUT_VERTICES: *params = r.tcsOutputVertices; break;
        case GL_TESS_GEN_MODE: *params = r.tesPrimitiveMode; break;
        case GL_TESS_GEN_SPACING: *params = r.tesSpacing; break;
        case GL_TESS_GEN_VERTEX_ORDER: *params = r.tesOrder; break;
        default: *params = r.tesPointMode; break;
        }
        break;
    }
    case GL_COMPUTE_WORK_GROUP_SIZE:
        if (!p->linked || !(r.stageMask & (1 << (int)glsl::Stage::Compute))) return ctx->setError(GL_INVALID_OPERATION);
        params[0] = r.localSize[0];
        params[1] = r.localSize[1];
        params[2] = r.localSize[2];
        break;
    case GL_PROGRAM_SEPARABLE: *params = p->linked && p->linkedSeparable; break;
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
    // "attr", "attr[0]" or an array element "attr[N]" (N < array size; each element
    // uses locationsUsed / arraySize consecutive locations).
    std::string n = name;
    for (auto& a : p->refl.attributes) {
        if (a.name == n) return a.location;
        if (n.size() > a.name.size() + 2 && n.compare(0, a.name.size(), a.name) == 0 && n[a.name.size()] == '[' && n.back() == ']') {
            const char* digits = n.c_str() + a.name.size() + 1;
            char* end = nullptr;
            long idx = strtol(digits, &end, 10);
            if (end == digits || *end != ']' || end + 1 != n.c_str() + n.size() || idx < 0 || idx >= a.arraySize) return -1;
            if (idx > 0 && digits[0] == '0') return -1;  // no leading zeros
            return a.location + (int)idx * (a.locationsUsed / std::max(1, a.arraySize));
        }
    }
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
        if (n == u.name || (isArray && n == ubase)) return u.location;
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
        // Block members and atomic counters (offset within the counter buffer) have a layout.
        case GL_UNIFORM_OFFSET: params[i] = u.blockIndex >= 0 || u.atomicCounterBufferIndex >= 0 ? u.offset : -1; break;
        case GL_UNIFORM_ARRAY_STRIDE: params[i] = u.blockIndex >= 0 || u.atomicCounterBufferIndex >= 0 ? u.arrayStride : -1; break;
        case GL_UNIFORM_MATRIX_STRIDE: params[i] = u.blockIndex >= 0 ? u.matrixStride : u.atomicCounterBufferIndex >= 0 ? 0 : -1; break;
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
            if (sl.kind != glsl::SlotKind::Texture && sl.kind != glsl::SlotKind::Sampler && sl.kind != glsl::SlotKind::Image &&
                sl.kind != glsl::SlotKind::LodBias &&
                sl.kind != glsl::SlotKind::SamplerBorder)
                continue;
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
            if (sl.kind != glsl::SlotKind::Texture && sl.kind != glsl::SlotKind::Image) continue;
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
            // floating point to integer queries round to nearest (GL 4.6 §2.2.2)
            case BaseType::Float:
            case BaseType::Double: {
                double d;
                if (ts.base == BaseType::Float) {
                    float f;
                    memcpy(&f, e, 4);
                    d = f;
                } else {
                    memcpy(&d, e, 8);
                }
                if (std::is_integral<T>::value) {
                    d = std::nearbyint(d);
                    if (std::is_unsigned<T>::value) d = std::max(d, 0.0);  // negative -> 0 for unsigned queries
                    d = std::clamp(d, (double)std::numeric_limits<T>::lowest(), (double)std::numeric_limits<T>::max());
                }
                v = (T)d;
                break;
            }
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
    int arraySize = 1, location = -1, locationIndex = -1, locationComponent = 0, blockIndex = -1, offset = -1, arrayStride = -1,
        matrixStride = -1;
    int isRowMajor = 0, bufferBinding = -1, bufferDataSize = 0, atomicIndex = -1, topLevelArraySize = 1, topLevelArrayStride = 0;
    int xfbBufferIndex = -1, xfbStride = 0;
    uint32_t stageMask = 0;
    std::vector<int> activeVariables;   // also COMPATIBLE_SUBROUTINES for subroutine uniforms
    bool isPerPatch = false;
};

// Subroutine interfaces: GL_VERTEX_SUBROUTINE.. / GL_VERTEX_SUBROUTINE_UNIFORM.. -> stage (or -1).
int subroutineStage(GLenum iface, bool* isUniform) {
    static const GLenum subs[] = {GL_VERTEX_SUBROUTINE, GL_TESS_CONTROL_SUBROUTINE, GL_TESS_EVALUATION_SUBROUTINE,
                                  GL_GEOMETRY_SUBROUTINE, GL_FRAGMENT_SUBROUTINE, GL_COMPUTE_SUBROUTINE};
    static const GLenum unis[] = {GL_VERTEX_SUBROUTINE_UNIFORM, GL_TESS_CONTROL_SUBROUTINE_UNIFORM,
                                  GL_TESS_EVALUATION_SUBROUTINE_UNIFORM, GL_GEOMETRY_SUBROUTINE_UNIFORM,
                                  GL_FRAGMENT_SUBROUTINE_UNIFORM, GL_COMPUTE_SUBROUTINE_UNIFORM};
    static const glsl::Stage stages[] = {glsl::Stage::Vertex, glsl::Stage::TessControl, glsl::Stage::TessEval,
                                         glsl::Stage::Geometry, glsl::Stage::Fragment, glsl::Stage::Compute};
    for (int i = 0; i < 6; i++) {
        if (iface == subs[i]) { *isUniform = false; return (int)stages[i]; }
        if (iface == unis[i]) { *isUniform = true; return (int)stages[i]; }
    }
    return -1;
}

bool validInterface(GLenum iface) {
    bool u;
    switch (iface) {
    case GL_UNIFORM: case GL_UNIFORM_BLOCK: case GL_ATOMIC_COUNTER_BUFFER: case GL_PROGRAM_INPUT: case GL_PROGRAM_OUTPUT:
    case GL_TRANSFORM_FEEDBACK_VARYING: case GL_TRANSFORM_FEEDBACK_BUFFER: case GL_BUFFER_VARIABLE: case GL_SHADER_STORAGE_BLOCK:
        return true;
    default: return subroutineStage(iface, &u) >= 0;
    }
}

// GL 4.6 table 7.2: which properties each interface supports. Returns -1 for a value that is not
// a property at all (INVALID_ENUM), 0 when unsupported by the interface (INVALID_OPERATION).
int propertySupported(GLenum iface, GLenum prop) {
    bool subUniform = false;
    int subStage = subroutineStage(iface, &subUniform);
    bool isSubUniform = subStage >= 0 && subUniform;
    bool uni = iface == GL_UNIFORM, bv = iface == GL_BUFFER_VARIABLE, in = iface == GL_PROGRAM_INPUT, out = iface == GL_PROGRAM_OUTPUT;
    bool tfv = iface == GL_TRANSFORM_FEEDBACK_VARYING, tfb = iface == GL_TRANSFORM_FEEDBACK_BUFFER;
    bool ub = iface == GL_UNIFORM_BLOCK, sb = iface == GL_SHADER_STORAGE_BLOCK, acb = iface == GL_ATOMIC_COUNTER_BUFFER;
    switch (prop) {
    case GL_NAME_LENGTH: return !acb && !tfb;
    case GL_TYPE: return uni || bv || in || out || tfv;
    case GL_ARRAY_SIZE: return uni || bv || in || out || tfv || isSubUniform;
    case GL_OFFSET: return uni || bv || tfv;
    case GL_BLOCK_INDEX: case GL_ARRAY_STRIDE: case GL_MATRIX_STRIDE: case GL_IS_ROW_MAJOR: return uni || bv;
    case GL_ATOMIC_COUNTER_BUFFER_INDEX: return uni;
    case GL_BUFFER_BINDING: case GL_NUM_ACTIVE_VARIABLES: case GL_ACTIVE_VARIABLES: return ub || sb || acb || tfb;
    case GL_BUFFER_DATA_SIZE: return ub || sb || acb;
    case GL_REFERENCED_BY_VERTEX_SHADER: case GL_REFERENCED_BY_TESS_CONTROL_SHADER: case GL_REFERENCED_BY_TESS_EVALUATION_SHADER:
    case GL_REFERENCED_BY_GEOMETRY_SHADER: case GL_REFERENCED_BY_FRAGMENT_SHADER: case GL_REFERENCED_BY_COMPUTE_SHADER:
        return uni || ub || acb || sb || bv || in || out;
    case GL_TOP_LEVEL_ARRAY_SIZE: case GL_TOP_LEVEL_ARRAY_STRIDE: return bv;
    case GL_NUM_COMPATIBLE_SUBROUTINES: case GL_COMPATIBLE_SUBROUTINES: return isSubUniform;
    case GL_LOCATION: return uni || in || out || isSubUniform;
    case GL_LOCATION_INDEX: return out;
    case GL_IS_PER_PATCH: case GL_LOCATION_COMPONENT: return in || out;
    case GL_TRANSFORM_FEEDBACK_BUFFER_INDEX: return tfv;
    case GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE: return tfb;
    default: return -1;
    }
}

// Locations taken by one element of an input/output of GL type `t`.
int locationsPerElement(GLenum t) {
    switch (t) {
    case GL_FLOAT_MAT2: case GL_FLOAT_MAT2x3: case GL_FLOAT_MAT2x4: case GL_DOUBLE_MAT2: return 2;
    case GL_FLOAT_MAT3: case GL_FLOAT_MAT3x2: case GL_FLOAT_MAT3x4: case GL_DOUBLE_MAT3x2: return 3;
    case GL_FLOAT_MAT4: case GL_FLOAT_MAT4x2: case GL_FLOAT_MAT4x3: case GL_DOUBLE_MAT4x2: return 4;
    case GL_DOUBLE_VEC3: case GL_DOUBLE_VEC4: return 2;
    case GL_DOUBLE_MAT2x3: case GL_DOUBLE_MAT2x4: return 4;
    case GL_DOUBLE_MAT3: case GL_DOUBLE_MAT3x4: return 6;
    case GL_DOUBLE_MAT4: case GL_DOUBLE_MAT4x3: return 8;
    default: return 1;
    }
}

Resource fromInterfaceVar(const glsl::InterfaceVar& v) {
    Resource x;
    x.name = v.name;
    x.type = v.type;
    x.arraySize = v.arraySize;
    x.location = v.location;
    x.locationIndex = v.locationIndex;
    x.locationComponent = v.component;
    x.isPerPatch = v.perPatch;
    x.stageMask = v.stageMask;
    return x;
}
}  // namespace

// The active resources of `iface` (an empty list when the program has no successful link).
static std::vector<Resource> resourcesFor(ProgramObj* p, GLenum iface) {
    std::vector<Resource> out;
    if (!p->linked) return out;
    const glsl::Reflection& r = p->refl;
    switch (iface) {
    case GL_UNIFORM:
        for (auto& u : r.uniforms) {
            Resource x;
            bool atomic = typeShape(u.type).base == BaseType::AtomicCounter;
            x.name = u.name;
            x.type = u.type;
            x.arraySize = u.arraySize;
            x.location = u.blockIndex < 0 && !atomic ? u.location : -1;
            x.blockIndex = u.blockIndex;
            x.offset = u.blockIndex >= 0 || atomic ? u.offset : -1;
            x.arrayStride = u.blockIndex >= 0 || atomic ? u.arrayStride : -1;
            x.matrixStride = u.blockIndex >= 0 ? u.matrixStride : -1;
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
            x.bufferBinding = i < bindings.size() ? bindings[i] : blocks[i].binding;
            x.bufferDataSize = blocks[i].dataSize;
            x.activeVariables = blocks[i].activeUniforms;
            x.stageMask = blocks[i].stageMask;
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
        for (auto& v : r.programInputs) out.push_back(fromInterfaceVar(v));
        break;
    case GL_PROGRAM_OUTPUT:
        for (auto& v : r.programOutputs) out.push_back(fromInterfaceVar(v));
        break;
    case GL_TRANSFORM_FEEDBACK_VARYING:
    case GL_TRANSFORM_FEEDBACK_BUFFER: {
        // TRANSFORM_FEEDBACK_BUFFER resources: the buffers holding captured varyings, in binding order
        int bufIndex[4] = {-1, -1, -1, -1};
        std::vector<Resource> buffers;
        bool used[4] = {false, false, false, false};
        for (const auto& v : r.xfbVaryings)
            if (v.type && v.buffer >= 0 && v.buffer < 4) used[v.buffer] = true;
        for (int b = 0; b < 4; b++)
            if (used[b]) {
                bufIndex[b] = (int)buffers.size();
                Resource x;
                x.bufferBinding = b;
                x.xfbStride = (int)r.multipass.xfbStride[b];
                buffers.push_back(x);
            }
        for (size_t i = 0; i < r.xfbVaryings.size(); i++) {
            const auto& v = r.xfbVaryings[i];
            Resource x;
            x.name = v.name;
            x.type = v.type;
            x.arraySize = v.arraySize;
            x.offset = v.type ? v.offset : -1;
            if (v.type && v.buffer >= 0 && v.buffer < 4) {
                x.xfbBufferIndex = bufIndex[v.buffer];
                buffers[bufIndex[v.buffer]].activeVariables.push_back((int)i);
            }
            out.push_back(x);
        }
        if (iface == GL_TRANSFORM_FEEDBACK_BUFFER) out = std::move(buffers);
        break;
    }
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
    default: {
        bool isUniform;
        int st = subroutineStage(iface, &isUniform);
        if (st < 0) break;
        const glsl::StageSubroutines& ss = r.subroutines[st];
        if (!isUniform) {
            for (auto& f : ss.functions) {
                Resource x;
                x.name = f;
                out.push_back(x);
            }
        } else {
            for (auto& u : ss.uniforms) {
                if (u.dims.size() <= 1) {
                    Resource x;
                    x.name = u.arraySize > 1 ? u.name + "[0]" : u.name;
                    x.arraySize = u.arraySize;
                    x.location = u.location;
                    x.activeVariables = u.compatible;
                    out.push_back(x);
                    continue;
                }
                // arrays of arrays: one resource per innermost array, "u[i][j][0]"
                int inner = u.dims.back();
                for (int o = 0; o < u.arraySize / inner; o++) {
                    Resource x;
                    std::string sub;
                    for (int k = (int)u.dims.size() - 2, r = o; k >= 0; r /= u.dims[k], k--)
                        sub = "[" + std::to_string(r % u.dims[k]) + "]" + sub;
                    x.name = u.name + sub + "[0]";
                    x.arraySize = inner;
                    x.location = u.location + o * inner;
                    x.activeVariables = u.compatible;
                    out.push_back(x);
                }
            }
        }
        break;
    }
    }
    return out;
}

// Index of the resource named `name`: an exact match, or `name` + "[0]" for arrays.
static int findResource(const std::vector<Resource>& res, const std::string& n) {
    for (size_t i = 0; i < res.size(); i++) {
        const std::string& rn = res[i].name;
        if (rn == n) return (int)i;
        if (rn.size() == n.size() + 3 && rn.compare(n.size(), 3, "[0]") == 0 && rn.compare(0, n.size(), n) == 0) return (int)i;
    }
    return -1;
}

API void APIENTRY glGetProgramInterfaceiv(GLuint program, GLenum iface, GLenum pname, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!validInterface(iface)) return ctx->setError(GL_INVALID_ENUM, "bad program interface 0x%x", iface);
    bool isSubUniform = false;
    bool isSub = subroutineStage(iface, &isSubUniform) >= 0;
    switch (pname) {
    case GL_ACTIVE_RESOURCES: case GL_MAX_NAME_LENGTH: case GL_MAX_NUM_ACTIVE_VARIABLES: case GL_MAX_NUM_COMPATIBLE_SUBROUTINES: break;
    default: return ctx->setError(GL_INVALID_ENUM, "bad program interface pname 0x%x", pname);
    }
    if (pname == GL_MAX_NAME_LENGTH && (iface == GL_ATOMIC_COUNTER_BUFFER || iface == GL_TRANSFORM_FEEDBACK_BUFFER))
        return ctx->setError(GL_INVALID_OPERATION);
    if (pname == GL_MAX_NUM_ACTIVE_VARIABLES && iface != GL_UNIFORM_BLOCK && iface != GL_SHADER_STORAGE_BLOCK &&
        iface != GL_ATOMIC_COUNTER_BUFFER && iface != GL_TRANSFORM_FEEDBACK_BUFFER)
        return ctx->setError(GL_INVALID_OPERATION);
    if (pname == GL_MAX_NUM_COMPATIBLE_SUBROUTINES && !(isSub && isSubUniform)) return ctx->setError(GL_INVALID_OPERATION);
    auto res = resourcesFor(p, iface);
    GLint m = 0;
    for (auto& x : res) {
        switch (pname) {
        case GL_MAX_NAME_LENGTH: m = std::max<GLint>(m, (GLint)x.name.size() + 1); break;
        case GL_MAX_NUM_ACTIVE_VARIABLES:
        case GL_MAX_NUM_COMPATIBLE_SUBROUTINES: m = std::max<GLint>(m, (GLint)x.activeVariables.size()); break;
        default: break;
        }
    }
    *params = pname == GL_ACTIVE_RESOURCES ? (GLint)res.size() : m;
}

API GLuint APIENTRY glGetProgramResourceIndex(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(GL_INVALID_INDEX);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return GL_INVALID_INDEX;
    if (!validInterface(iface) || iface == GL_ATOMIC_COUNTER_BUFFER || iface == GL_TRANSFORM_FEEDBACK_BUFFER) {
        ctx->setError(GL_INVALID_ENUM);
        return GL_INVALID_INDEX;
    }
    if (!name) return GL_INVALID_INDEX;
    auto res = resourcesFor(p, iface);
    int i = findResource(res, name);
    // gl_NextBuffer / gl_SkipComponentsN are listed but are not names of active resources
    if (i >= 0 && iface == GL_TRANSFORM_FEEDBACK_VARYING && !res[i].type) return GL_INVALID_INDEX;
    return i < 0 ? GL_INVALID_INDEX : (GLuint)i;
}

API void APIENTRY glGetProgramResourceName(GLuint program, GLenum iface, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!validInterface(iface) || iface == GL_ATOMIC_COUNTER_BUFFER || iface == GL_TRANSFORM_FEEDBACK_BUFFER)
        return ctx->setError(GL_INVALID_ENUM);
    auto res = resourcesFor(p, iface);
    if (index >= res.size()) return ctx->setError(GL_INVALID_VALUE, "resource index %u out of range", index);
    if (bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    copyLog(res[index].name, bufSize, length, name);
}

API void APIENTRY glGetProgramResourceiv(GLuint program, GLenum iface, GLuint index, GLsizei propCount, const GLenum* props, GLsizei bufSize,
                                         GLsizei* length, GLint* params) {
    MGL_CTX();
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return;
    if (!validInterface(iface)) return ctx->setError(GL_INVALID_ENUM);
    if (propCount <= 0 || bufSize < 0) return ctx->setError(GL_INVALID_VALUE);
    auto res = resourcesFor(p, iface);
    if (index >= res.size()) return ctx->setError(GL_INVALID_VALUE, "resource index %u out of range", index);
    for (int i = 0; i < propCount; i++) {
        int s = propertySupported(iface, props[i]);
        if (s < 0) return ctx->setError(GL_INVALID_ENUM, "bad resource property 0x%x", props[i]);
        if (s == 0) return ctx->setError(GL_INVALID_OPERATION, "property 0x%x not supported by interface 0x%x", props[i], iface);
    }
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
        case GL_NUM_ACTIVE_VARIABLES:
        case GL_NUM_COMPATIBLE_SUBROUTINES: vals.push_back((GLint)x.activeVariables.size()); break;
        case GL_ACTIVE_VARIABLES:
        case GL_COMPATIBLE_SUBROUTINES: for (int v : x.activeVariables) vals.push_back(v); break;
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
        case GL_LOCATION_COMPONENT: vals.push_back(x.locationComponent); break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_INDEX: vals.push_back(x.xfbBufferIndex); break;
        case GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE: vals.push_back(x.xfbStride); break;
        }
    }
    GLsizei n = std::min<GLsizei>(bufSize, (GLsizei)vals.size());
    for (int i = 0; i < n; i++) params[i] = vals[i];
    if (length) *length = n;
}

// Location of `name` among inputs/outputs/subroutine uniforms: "x", "x[0]" or "x[i]" of an array.
static GLint resourceLocation(const std::vector<Resource>& res, const std::string& n, bool ioStride, const Resource** match) {
    *match = nullptr;
    int i = findResource(res, n);
    if (i >= 0) {
        *match = &res[i];
        return res[i].location;
    }
    std::string base;
    int elem;
    bool had;
    if (!parseArrayName(n, base, elem, had) || !had) return -1;
    i = findResource(res, base);
    if (i < 0 || res[i].name != base + "[0]" || elem >= res[i].arraySize) return -1;
    *match = &res[i];
    if (res[i].location < 0) return -1;
    return res[i].location + elem * (ioStride ? locationsPerElement(res[i].type) : 1);
}

API GLint APIENTRY glGetProgramResourceLocation(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    bool isSubUniform = false;
    bool isSub = subroutineStage(iface, &isSubUniform) >= 0;
    if (iface != GL_UNIFORM && iface != GL_PROGRAM_INPUT && iface != GL_PROGRAM_OUTPUT && !(isSub && isSubUniform)) {
        ctx->setError(GL_INVALID_ENUM);
        return -1;
    }
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION, "program not linked");
        return -1;
    }
    if (!name) return -1;
    if (iface == GL_UNIFORM) return uniformLocation(p, name);
    if (isSub) {
        // subroutine uniform arrays (of arrays) take flattened per-element locations
        static const GLenum kShaderTypes[] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER,
                                              GL_GEOMETRY_SHADER, GL_FRAGMENT_SHADER, GL_COMPUTE_SHADER};
        return glGetSubroutineUniformLocation(program, kShaderTypes[subroutineStage(iface, &isSubUniform)], name);
    }
    const Resource* m;
    return resourceLocation(resourcesFor(p, iface), name, !isSub, &m);
}

API GLint APIENTRY glGetProgramResourceLocationIndex(GLuint program, GLenum iface, const GLchar* name) {
    MGL_CTX_RET(-1);
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return -1;
    if (iface != GL_PROGRAM_OUTPUT) {
        ctx->setError(GL_INVALID_ENUM);
        return -1;
    }
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION, "program not linked");
        return -1;
    }
    if (!name) return -1;
    const Resource* m;
    auto res = resourcesFor(p, iface);
    resourceLocation(res, name, true, &m);
    return m && m->location >= 0 ? m->locationIndex : -1;
}

// ---------------------------------------------------------------------------
// separable programs & pipelines
// ---------------------------------------------------------------------------
API GLuint APIENTRY glCreateShaderProgramv(GLenum type, GLsizei count, const GLchar* const* strings) {
    MGL_CTX_RET(0);
    if (count < 0) {
        ctx->setError(GL_INVALID_VALUE);
        return 0;
    }
    GLuint s = glCreateShader(type);  // INVALID_ENUM for a bad type
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

static ProgramPipelineObj* pipelineObj(Context* ctx, GLuint name, bool create, GLenum err = GL_INVALID_OPERATION) {
    ProgramPipelineObj* pp = ctx->pipelines.get(name);
    if (!pp && create && ctx->pipelines.isName(name)) pp = ctx->pipelines.create(name, new ProgramPipelineObj());
    if (!pp) ctx->setError(err);
    return pp;
}

// GL 4.6 §11.1.3.11 program pipeline validation; empty string when valid.
std::string pipelineValidationError(ProgramPipelineObj* pp) {
    for (int st = 0; st < (int)glsl::Stage::Count; st++) {
        ProgramObj* sp = pp->stages[st].get();
        if (!sp) continue;
        if (!sp->program || !sp->linkedSeparable) return "program " + std::to_string(sp->name) + " is not a linked separable program";
        if (st == (int)glsl::Stage::Compute) continue;
        for (int o = 0; o < (int)glsl::Stage::Count; o++)
            if (o != (int)glsl::Stage::Compute && (sp->refl.stageMask & (1u << o)) && pp->stages[o].get() != sp)
                return "program " + std::to_string(sp->name) + " is not active for all the stages it was linked with";
    }
    if (!pp->stages[(int)glsl::Stage::Vertex] &&
        (pp->stages[(int)glsl::Stage::Geometry] || pp->stages[(int)glsl::Stage::TessControl] || pp->stages[(int)glsl::Stage::TessEval]))
        return "pipeline has geometry or tessellation stages but no vertex program";
    if (!pp->stages[(int)glsl::Stage::Vertex] && !pp->stages[(int)glsl::Stage::Compute])
        return "pipeline has no vertex or compute program";
    return "";
}

API void APIENTRY glBindProgramPipeline(GLuint pipeline) {
    MGL_CTX();
    if (!pipeline) {
        ctx->pipeline = nullptr;
        return;
    }
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true);
    if (pp) {
        ctx->pipeline = pp;
        if (!ctx->program)
            for (int st = 0; st < (int)glsl::Stage::Count; st++) resetSubroutineSelection(ctx, st, pp->stages[st].get());
    }
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
        if (!p->linkedSeparable || !p->linked) return ctx->setError(GL_INVALID_OPERATION);
    }
    const std::pair<GLbitfield, glsl::Stage> map[] = {
        {GL_VERTEX_SHADER_BIT, glsl::Stage::Vertex}, {GL_TESS_CONTROL_SHADER_BIT, glsl::Stage::TessControl},
        {GL_TESS_EVALUATION_SHADER_BIT, glsl::Stage::TessEval}, {GL_GEOMETRY_SHADER_BIT, glsl::Stage::Geometry},
        {GL_FRAGMENT_SHADER_BIT, glsl::Stage::Fragment}, {GL_COMPUTE_SHADER_BIT, glsl::Stage::Compute}};
    for (auto& [bit, st] : map) {
        if (!(stages & bit)) continue;
        pp->stages[(int)st] = (p && (p->refl.stageMask & (1u << (int)st))) ? p : nullptr;
        if (ctx->pipeline == pp && !ctx->program) resetSubroutineSelection(ctx, (int)st, pp->stages[(int)st].get());
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
    std::string err = pipelineValidationError(pp);
    pp->validated = err.empty();
    pp->infoLog = err.empty() ? "" : err + "\n";
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
    ProgramPipelineObj* pp = pipelineObj(ctx, pipeline, true, GL_INVALID_VALUE);
    if (pp) copyLog(pp->infoLog, bufSize, length, infoLog);
}

// --- subroutines (ARB_shader_subroutine) ---
static int stageIndexForShaderType(GLenum t) {
    switch (t) {
    case GL_VERTEX_SHADER: return (int)glsl::Stage::Vertex;
    case GL_TESS_CONTROL_SHADER: return (int)glsl::Stage::TessControl;
    case GL_TESS_EVALUATION_SHADER: return (int)glsl::Stage::TessEval;
    case GL_GEOMETRY_SHADER: return (int)glsl::Stage::Geometry;
    case GL_FRAGMENT_SHADER: return (int)glsl::Stage::Fragment;
    case GL_COMPUTE_SHADER: return (int)glsl::Stage::Compute;
    default: return -1;
    }
}

static const glsl::StageSubroutines* stageSubs(Context* ctx, GLuint program, GLenum shadertype, ProgramObj** out) {
    int st = stageIndexForShaderType(shadertype);
    if (st < 0) {
        ctx->setError(GL_INVALID_ENUM);
        return nullptr;
    }
    ProgramObj* p = getProgram(ctx, program);
    if (!p) return nullptr;
    if (!p->linked) {
        ctx->setError(GL_INVALID_OPERATION);
        return nullptr;
    }
    if (out) *out = p;
    return &p->refl.subroutines[st];
}

API GLint APIENTRY glGetSubroutineUniformLocation(GLuint program, GLenum shadertype, const GLchar* name) {
    MGL_CTX_RET(-1);
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return -1;
    // name, name[i], name[i][j]... (arrays of arrays are flattened row-major)
    std::string n = name;
    std::vector<int> idx;
    while (!n.empty() && n.back() == ']') {
        size_t lb = n.rfind('[');
        if (lb == std::string::npos || lb + 2 > n.size() - 1) return -1;
        std::string d = n.substr(lb + 1, n.size() - lb - 2);
        if (d.empty() || d.find_first_not_of("0123456789") != std::string::npos) return -1;
        idx.insert(idx.begin(), std::atoi(d.c_str()));
        n.resize(lb);
    }
    for (auto& u : ss->uniforms) {
        if (u.name != n) continue;
        if (idx.empty()) return u.location;
        if (idx.size() > u.dims.size()) return -1;
        int flat = 0;
        for (size_t k = 0; k < u.dims.size(); k++) {
            int i = k < idx.size() ? idx[k] : 0;
            if (i >= u.dims[k]) return -1;
            flat = flat * u.dims[k] + i;
        }
        return u.location + flat;
    }
    return -1;
}

API GLuint APIENTRY glGetSubroutineIndex(GLuint program, GLenum shadertype, const GLchar* name) {
    MGL_CTX_RET(GL_INVALID_INDEX);
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return GL_INVALID_INDEX;
    for (size_t i = 0; i < ss->functions.size(); i++)
        if (ss->functions[i] == name) return (GLuint)i;
    return GL_INVALID_INDEX;
}

API void APIENTRY glGetActiveSubroutineName(GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei* length, GLchar* name) {
    MGL_CTX();
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return;
    if (index >= ss->functions.size()) return ctx->setError(GL_INVALID_VALUE);
    copyLog(ss->functions[index], bufSize, length, name);
}

API void APIENTRY glGetActiveSubroutineUniformName(GLuint program, GLenum shadertype, GLuint index, GLsizei bufSize, GLsizei* length,
                                                   GLchar* name) {
    MGL_CTX();
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return;
    if (index >= ss->uniforms.size()) return ctx->setError(GL_INVALID_VALUE);
    copyLog(ss->uniforms[index].name, bufSize, length, name);
}

API void APIENTRY glGetActiveSubroutineUniformiv(GLuint program, GLenum shadertype, GLuint index, GLenum pname, GLint* values) {
    MGL_CTX();
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return;
    if (index >= ss->uniforms.size()) return ctx->setError(GL_INVALID_VALUE);
    const auto& u = ss->uniforms[index];
    switch (pname) {
    case GL_NUM_COMPATIBLE_SUBROUTINES: *values = (GLint)u.compatible.size(); break;
    case GL_COMPATIBLE_SUBROUTINES: for (size_t i = 0; i < u.compatible.size(); i++) values[i] = u.compatible[i]; break;
    case GL_UNIFORM_SIZE: *values = u.arraySize; break;
    case GL_UNIFORM_NAME_LENGTH: *values = (GLint)u.name.size() + 1; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glUniformSubroutinesuiv(GLenum shadertype, GLsizei count, const GLuint* indices) {
    MGL_CTX();
    int st = stageIndexForShaderType(shadertype);
    if (st < 0) return ctx->setError(GL_INVALID_ENUM);
    ProgramObj* p = ctx->activeProgramForStage((glsl::Stage)st);
    if (!p || !(p->refl.stageMask & (1u << st))) return ctx->setError(GL_INVALID_OPERATION);
    const glsl::StageSubroutines& ss = p->refl.subroutines[st];
    if (count != ss.numLocations) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++)
        if (indices[i] >= ss.functions.size()) return ctx->setError(GL_INVALID_VALUE);
    for (auto& u : ss.uniforms)
        for (int i = 0; i < u.arraySize; i++) {
            int loc = u.location + i;
            if (loc < count && std::find(u.compatible.begin(), u.compatible.end(), (int)indices[loc]) == u.compatible.end())
                return ctx->setError(GL_INVALID_OPERATION);
        }
    ctx->subroutineSelection[st].assign(indices, indices + count);
}

API void APIENTRY glGetUniformSubroutineuiv(GLenum shadertype, GLint location, GLuint* params) {
    MGL_CTX();
    int st = stageIndexForShaderType(shadertype);
    if (st < 0) return ctx->setError(GL_INVALID_ENUM);
    ProgramObj* p = ctx->activeProgramForStage((glsl::Stage)st);
    if (!p || !(p->refl.stageMask & (1u << st))) return ctx->setError(GL_INVALID_OPERATION);
    if (location < 0 || location >= p->refl.subroutines[st].numLocations) return ctx->setError(GL_INVALID_VALUE);
    const auto& sel = ctx->subroutineSelection[st];
    *params = location < (int)sel.size() ? sel[location] : 0;
}

API void APIENTRY glGetProgramStageiv(GLuint program, GLenum shadertype, GLenum pname, GLint* values) {
    MGL_CTX();
    const glsl::StageSubroutines* ss = stageSubs(ctx, program, shadertype, nullptr);
    if (!ss) return;
    switch (pname) {
    case GL_ACTIVE_SUBROUTINE_UNIFORMS: *values = (GLint)ss->uniforms.size(); break;
    case GL_ACTIVE_SUBROUTINE_UNIFORM_LOCATIONS: *values = ss->numLocations; break;
    case GL_ACTIVE_SUBROUTINES: *values = (GLint)ss->functions.size(); break;
    case GL_ACTIVE_SUBROUTINE_UNIFORM_MAX_LENGTH: {
        GLint m = 0;
        for (auto& u : ss->uniforms) m = std::max<GLint>(m, (GLint)u.name.size() + 1);
        *values = m;
        break;
    }
    case GL_ACTIVE_SUBROUTINE_MAX_LENGTH: {
        GLint m = 0;
        for (auto& f : ss->functions) m = std::max<GLint>(m, (GLint)f.size() + 1);
        *values = m;
        break;
    }
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
