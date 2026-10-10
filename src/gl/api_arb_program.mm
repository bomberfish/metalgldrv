// GL_ARB_vertex_program / GL_ARB_fragment_program (and GL_EXT_gpu_program_parameters): program objects,
// parameters and queries. Programs are translated to GLSL when loaded (arb_program.cpp) and drawn through the
// compatibility profile's program cache (compat.mm, mglCompatPrepareDraw).
#import "gl/context.h"
#include <atomic>
#include <cmath>
#include "gl/app_profile.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

static bool arbTarget(GLenum target) { return target == GL_VERTEX_PROGRAM_ARB || target == GL_FRAGMENT_PROGRAM_ARB; }

// The program bound to `target` (the default object for 0, made on first use).
static ArbProgram* boundArb(Context* ctx, GLenum target) {
    bool vp = target == GL_VERTEX_PROGRAM_ARB;
    Ref<ArbProgram>& bound = vp ? ctx->arbVP : ctx->arbFP;
    if (!bound) {
        Ref<ArbProgram>& def = vp ? ctx->arbDefaultVP : ctx->arbDefaultFP;
        if (!def) {
            def = new ArbProgram();
            def->target = target;
        }
        bound = def;
    }
    return bound.get();
}

API void APIENTRY glGenProgramsARB(GLsizei n, GLuint* programs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) programs[i] = ctx->share->arbPrograms.genName();
}

API void APIENTRY glDeleteProgramsARB(GLsizei n, const GLuint* programs) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    for (int i = 0; i < n; i++) {
        if (!programs[i]) continue;
        ArbProgram* p = ctx->share->arbPrograms.get(programs[i]);
        if (p && ctx->arbVP.get() == p) ctx->arbVP = nullptr;  // back to the default program
        if (p && ctx->arbFP.get() == p) ctx->arbFP = nullptr;
        ctx->share->arbPrograms.remove(programs[i]);
    }
}

API GLboolean APIENTRY glIsProgramARB(GLuint program) {
    MGL_CTX_RET(GL_FALSE);
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    return program && ctx->share->arbPrograms.get(program) != nullptr;
}

API void APIENTRY glBindProgramARB(GLenum target, GLuint program) {
    MGL_CTX();
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    Ref<ArbProgram>& bound = target == GL_VERTEX_PROGRAM_ARB ? ctx->arbVP : ctx->arbFP;
    if (!program) {
        bound = nullptr;
        return;
    }
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    ArbProgram* p = ctx->share->arbPrograms.get(program);
    if (!p) {
        p = ctx->share->arbPrograms.create(program, new ArbProgram());  // binding an unused name creates it
        p->target = target;
    }
    if (p->target != target) return ctx->setError(GL_INVALID_OPERATION, "program %u is a 0x%x program", program, p->target);
    bound = p;
}

API void APIENTRY glProgramStringARB(GLenum target, GLenum format, GLsizei len, const void* string) {
    MGL_CTX();
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    if (format != GL_PROGRAM_FORMAT_ASCII_ARB) return ctx->setError(GL_INVALID_ENUM);
    if (len < 0 || !string) return ctx->setError(GL_INVALID_VALUE);
    std::string src((const char*)string, (size_t)len);
    ArbTranslation tr = translateArbProgram(target, src);
    ctx->arbErrorPos = tr.ok ? -1 : tr.errorPos;
    ctx->arbErrorString = tr.ok ? "" : tr.error;
    if (!tr.ok) {
        MGL_DBG("ARB program failed to load: %s", tr.error.c_str());
        if (appProfile().logUnsupported) MGL_WARN("glProgramStringARB: %s", tr.error.c_str());
        return ctx->setError(GL_INVALID_OPERATION);  // the program keeps what it had
    }
    ArbProgram* p = boundArb(ctx, target);
    p->source = std::move(src);
    p->tr = std::move(tr);
    static std::atomic<uint64_t> versions{0};  // unique across programs: the draw-time cache is keyed by it
    p->version = ++versions;
}

// --- parameters ---
static float (*envParams(Context* ctx, GLenum target))[4] { return target == GL_VERTEX_PROGRAM_ARB ? ctx->arbVPEnv : ctx->arbFPEnv; }

static void setParams(Context* ctx, GLenum target, bool env, GLuint index, GLsizei count, const float* v) {
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    if (count < 0) return ctx->setError(GL_INVALID_VALUE);
    int max = env ? kArbMaxEnvParams : kArbMaxLocalParams;
    if (index >= (GLuint)max || (GLuint)count > (GLuint)max - index) return ctx->setError(GL_INVALID_VALUE);
    float(*dst)[4] = env ? envParams(ctx, target) : boundArb(ctx, target)->local;
    memcpy(dst[index], v, (size_t)count * 16);
}

API void APIENTRY glProgramEnvParameter4fARB(GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    MGL_CTX();
    float v[4] = {x, y, z, w};
    setParams(ctx, target, true, index, 1, v);
}
API void APIENTRY glProgramEnvParameter4fvARB(GLenum target, GLuint index, const GLfloat* params) {
    MGL_CTX();
    setParams(ctx, target, true, index, 1, params);
}
API void APIENTRY glProgramEnvParameter4dARB(GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    glProgramEnvParameter4fARB(target, index, (float)x, (float)y, (float)z, (float)w);
}
API void APIENTRY glProgramEnvParameter4dvARB(GLenum target, GLuint index, const GLdouble* p) {
    glProgramEnvParameter4fARB(target, index, (float)p[0], (float)p[1], (float)p[2], (float)p[3]);
}
API void APIENTRY glProgramLocalParameter4fARB(GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    MGL_CTX();
    float v[4] = {x, y, z, w};
    setParams(ctx, target, false, index, 1, v);
}
API void APIENTRY glProgramLocalParameter4fvARB(GLenum target, GLuint index, const GLfloat* params) {
    MGL_CTX();
    setParams(ctx, target, false, index, 1, params);
}
API void APIENTRY glProgramLocalParameter4dARB(GLenum target, GLuint index, GLdouble x, GLdouble y, GLdouble z, GLdouble w) {
    glProgramLocalParameter4fARB(target, index, (float)x, (float)y, (float)z, (float)w);
}
API void APIENTRY glProgramLocalParameter4dvARB(GLenum target, GLuint index, const GLdouble* p) {
    glProgramLocalParameter4fARB(target, index, (float)p[0], (float)p[1], (float)p[2], (float)p[3]);
}
// GL_EXT_gpu_program_parameters: several at once
API void APIENTRY glProgramEnvParameters4fvEXT(GLenum target, GLuint index, GLsizei count, const GLfloat* params) {
    MGL_CTX();
    setParams(ctx, target, true, index, count, params);
}
API void APIENTRY glProgramLocalParameters4fvEXT(GLenum target, GLuint index, GLsizei count, const GLfloat* params) {
    MGL_CTX();
    setParams(ctx, target, false, index, count, params);
}

static const float* getParam(Context* ctx, GLenum target, bool env, GLuint index) {
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM), nullptr;
    if (index >= (GLuint)(env ? kArbMaxEnvParams : kArbMaxLocalParams)) return ctx->setError(GL_INVALID_VALUE), nullptr;
    return env ? envParams(ctx, target)[index] : boundArb(ctx, target)->local[index];
}
API void APIENTRY glGetProgramEnvParameterfvARB(GLenum target, GLuint index, GLfloat* params) {
    MGL_CTX();
    if (const float* v = getParam(ctx, target, true, index)) memcpy(params, v, 16);
}
API void APIENTRY glGetProgramEnvParameterdvARB(GLenum target, GLuint index, GLdouble* params) {
    MGL_CTX();
    if (const float* v = getParam(ctx, target, true, index))
        for (int k = 0; k < 4; k++) params[k] = v[k];
}
API void APIENTRY glGetProgramLocalParameterfvARB(GLenum target, GLuint index, GLfloat* params) {
    MGL_CTX();
    if (const float* v = getParam(ctx, target, false, index)) memcpy(params, v, 16);
}
API void APIENTRY glGetProgramLocalParameterdvARB(GLenum target, GLuint index, GLdouble* params) {
    MGL_CTX();
    if (const float* v = getParam(ctx, target, false, index))
        for (int k = 0; k < 4; k++) params[k] = v[k];
}

// --- queries ---
API void APIENTRY glGetProgramivARB(GLenum target, GLenum pname, GLint* params) {
    MGL_CTX();
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    ArbProgram* p = boundArb(ctx, target);
    const ArbTranslation& t = p->tr;
    bool fp = target == GL_FRAGMENT_PROGRAM_ARB;
    switch (pname) {
    case GL_PROGRAM_LENGTH_ARB: *params = (GLint)p->source.size(); break;
    case GL_PROGRAM_FORMAT_ARB: *params = GL_PROGRAM_FORMAT_ASCII_ARB; break;
    case GL_PROGRAM_BINDING_ARB: *params = (GLint)p->name; break;
    case GL_PROGRAM_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_INSTRUCTIONS_ARB: *params = t.instructions; break;
    case GL_PROGRAM_TEMPORARIES_ARB: case GL_PROGRAM_NATIVE_TEMPORARIES_ARB: *params = t.temps; break;
    case GL_PROGRAM_PARAMETERS_ARB: case GL_PROGRAM_NATIVE_PARAMETERS_ARB: *params = t.params; break;
    case GL_PROGRAM_ATTRIBS_ARB: case GL_PROGRAM_NATIVE_ATTRIBS_ARB: *params = t.attribs; break;
    case GL_PROGRAM_ADDRESS_REGISTERS_ARB: case GL_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB: *params = t.addressRegs; break;
    case GL_PROGRAM_ALU_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB: *params = t.aluInstructions; break;
    case GL_PROGRAM_TEX_INSTRUCTIONS_ARB: case GL_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB: *params = t.texInstructions; break;
    case GL_PROGRAM_TEX_INDIRECTIONS_ARB: case GL_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB: *params = t.texIndirections; break;
    case GL_MAX_PROGRAM_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_INSTRUCTIONS_ARB:
    case GL_MAX_PROGRAM_ALU_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_ALU_INSTRUCTIONS_ARB:
    case GL_MAX_PROGRAM_TEX_INSTRUCTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_TEX_INSTRUCTIONS_ARB:
    case GL_MAX_PROGRAM_TEX_INDIRECTIONS_ARB: case GL_MAX_PROGRAM_NATIVE_TEX_INDIRECTIONS_ARB:
        *params = kArbMaxInstructions;
        break;
    case GL_MAX_PROGRAM_TEMPORARIES_ARB: case GL_MAX_PROGRAM_NATIVE_TEMPORARIES_ARB: *params = kArbMaxTemps; break;
    case GL_MAX_PROGRAM_PARAMETERS_ARB: case GL_MAX_PROGRAM_NATIVE_PARAMETERS_ARB: *params = kArbMaxParams; break;
    case GL_MAX_PROGRAM_ATTRIBS_ARB: case GL_MAX_PROGRAM_NATIVE_ATTRIBS_ARB: *params = fp ? 10 : kArbMaxAttribs; break;
    case GL_MAX_PROGRAM_ADDRESS_REGISTERS_ARB: case GL_MAX_PROGRAM_NATIVE_ADDRESS_REGISTERS_ARB: *params = fp ? 0 : kArbMaxAddressRegs; break;
    case GL_MAX_PROGRAM_LOCAL_PARAMETERS_ARB: *params = kArbMaxLocalParams; break;
    case GL_MAX_PROGRAM_ENV_PARAMETERS_ARB: *params = kArbMaxEnvParams; break;
    case GL_PROGRAM_UNDER_NATIVE_LIMITS_ARB: *params = t.ok ? 1 : 0; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glGetProgramStringARB(GLenum target, GLenum pname, void* string) {
    MGL_CTX();
    if (!arbTarget(target)) return ctx->setError(GL_INVALID_ENUM);
    if (pname != GL_PROGRAM_STRING_ARB) return ctx->setError(GL_INVALID_ENUM);
    const std::string& s = boundArb(ctx, target)->source;
    memcpy(string, s.data(), s.size());
}

// For the compatibility draw path: the enabled programs' objects (nullptr for a stage without one).
ArbProgram* mglArbProgram(Context* ctx, bool vertex) {
    if (vertex ? !ctx->arbVertexProgram : !ctx->arbFragmentProgram) return nullptr;
    return boundArb(ctx, vertex ? GL_VERTEX_PROGRAM_ARB : GL_FRAGMENT_PROGRAM_ARB);
}
