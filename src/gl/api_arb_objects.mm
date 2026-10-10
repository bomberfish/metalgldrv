// GL_ARB_shader_objects / GL_ARB_vertex_shader's handle-based entry points, as the core functions they became.
// GLhandleARB is a pointer on Apple platforms and a 32-bit integer elsewhere, which is what Windows programs
// under Wine pass: handles are the object names, read from the low 32 bits.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

static GLuint name(GLhandleARB h) { return (GLuint)(uintptr_t)h; }
static GLhandleARB handle(GLuint n) { return (GLhandleARB)(uintptr_t)n; }

static bool isShader(Context* ctx, GLuint n) {
    std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
    return ctx->share->shader(n) != nullptr;
}

API void APIENTRY glDeleteObjectARB(GLhandleARB obj) {
    MGL_CTX();
    GLuint n = name(obj);
    if (isShader(ctx, n)) glDeleteShader(n);
    else glDeleteProgram(n);
}

API GLhandleARB APIENTRY glGetHandleARB(GLenum pname) {
    MGL_CTX_RET(nullptr);
    if (pname != GL_PROGRAM_OBJECT_ARB) return ctx->setError(GL_INVALID_ENUM), handle(0);
    GLint p = 0;
    glGetIntegerv(GL_CURRENT_PROGRAM, &p);
    return handle((GLuint)p);
}

API void APIENTRY glDetachObjectARB(GLhandleARB containerObj, GLhandleARB attachedObj) { glDetachShader(name(containerObj), name(attachedObj)); }
API GLhandleARB APIENTRY glCreateShaderObjectARB(GLenum shaderType) { return handle(glCreateShader(shaderType)); }
API void APIENTRY glShaderSourceARB(GLhandleARB shaderObj, GLsizei count, const GLcharARB** string, const GLint* length) {
    glShaderSource(name(shaderObj), count, string, length);
}
API void APIENTRY glCompileShaderARB(GLhandleARB shaderObj) { glCompileShader(name(shaderObj)); }
API GLhandleARB APIENTRY glCreateProgramObjectARB(void) { return handle(glCreateProgram()); }
API void APIENTRY glAttachObjectARB(GLhandleARB containerObj, GLhandleARB obj) { glAttachShader(name(containerObj), name(obj)); }
API void APIENTRY glLinkProgramARB(GLhandleARB programObj) { glLinkProgram(name(programObj)); }
API void APIENTRY glUseProgramObjectARB(GLhandleARB programObj) { glUseProgram(name(programObj)); }
API void APIENTRY glValidateProgramARB(GLhandleARB programObj) { glValidateProgram(name(programObj)); }

// OBJECT_TYPE / OBJECT_SUBTYPE are ARB-only; the other OBJECT_*_ARB parameters have the core values of
// glGetShaderiv / glGetProgramiv.
API void APIENTRY glGetObjectParameterivARB(GLhandleARB obj, GLenum pname, GLint* params) {
    MGL_CTX();
    GLuint n = name(obj);
    bool shader = isShader(ctx, n);
    if (pname == GL_OBJECT_TYPE_ARB) {
        *params = shader ? GL_SHADER_OBJECT_ARB : GL_PROGRAM_OBJECT_ARB;
        if (!shader && !glIsProgram(n)) ctx->setError(GL_INVALID_VALUE);
        return;
    }
    if (pname == GL_OBJECT_SUBTYPE_ARB) {
        if (!shader) return ctx->setError(GL_INVALID_OPERATION);
        return glGetShaderiv(n, GL_SHADER_TYPE, params);
    }
    if (shader) glGetShaderiv(n, pname, params);
    else glGetProgramiv(n, pname, params);
}

API void APIENTRY glGetObjectParameterfvARB(GLhandleARB obj, GLenum pname, GLfloat* params) {
    GLint v[4] = {};
    glGetObjectParameterivARB(obj, pname, v);
    *params = (GLfloat)v[0];
}

API void APIENTRY glGetInfoLogARB(GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB* infoLog) {
    MGL_CTX();
    GLuint n = name(obj);
    if (isShader(ctx, n)) glGetShaderInfoLog(n, maxLength, length, infoLog);
    else glGetProgramInfoLog(n, maxLength, length, infoLog);
}

API void APIENTRY glGetAttachedObjectsARB(GLhandleARB containerObj, GLsizei maxCount, GLsizei* count, GLhandleARB* obj) {
    MGL_CTX();
    if (maxCount < 0) return ctx->setError(GL_INVALID_VALUE);
    std::vector<GLuint> names((size_t)maxCount);
    GLsizei n = 0;
    glGetAttachedShaders(name(containerObj), maxCount, &n, names.data());
    for (GLsizei i = 0; i < n; i++) obj[i] = handle(names[(size_t)i]);
    if (count) *count = n;
}

API GLint APIENTRY glGetUniformLocationARB(GLhandleARB programObj, const GLcharARB* n) { return glGetUniformLocation(name(programObj), n); }
API void APIENTRY glGetActiveUniformARB(GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei* length, GLint* size,
                                        GLenum* type, GLcharARB* n) {
    glGetActiveUniform(name(programObj), index, maxLength, length, size, type, n);
}
API void APIENTRY glGetUniformfvARB(GLhandleARB programObj, GLint location, GLfloat* params) { glGetUniformfv(name(programObj), location, params); }
API void APIENTRY glGetUniformivARB(GLhandleARB programObj, GLint location, GLint* params) { glGetUniformiv(name(programObj), location, params); }
API void APIENTRY glGetShaderSourceARB(GLhandleARB obj, GLsizei maxLength, GLsizei* length, GLcharARB* source) {
    glGetShaderSource(name(obj), maxLength, length, source);
}

// GL_ARB_vertex_shader
API void APIENTRY glBindAttribLocationARB(GLhandleARB programObj, GLuint index, const GLcharARB* n) { glBindAttribLocation(name(programObj), index, n); }
API void APIENTRY glGetActiveAttribARB(GLhandleARB programObj, GLuint index, GLsizei maxLength, GLsizei* length, GLint* size, GLenum* type,
                                       GLcharARB* n) {
    glGetActiveAttrib(name(programObj), index, maxLength, length, size, type, n);
}
API GLint APIENTRY glGetAttribLocationARB(GLhandleARB programObj, const GLcharARB* n) { return glGetAttribLocation(name(programObj), n); }
