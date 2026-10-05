// Vertex array objects and generic vertex attributes.
#import "gl/context.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

API void APIENTRY glGenVertexArrays(GLsizei n, GLuint* arrays) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) arrays[i] = ctx->vaos.genName();
}

static VertexArray* newVao() {
    VertexArray* v = new VertexArray();
    for (int i = 0; i < kMaxVertexAttribs; i++) v->attribs[i].binding = i;
    return v;
}

API void APIENTRY glCreateVertexArrays(GLsizei n, GLuint* arrays) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        arrays[i] = ctx->vaos.genName();
        ctx->vaos.create(arrays[i], newVao());
    }
}

API void APIENTRY glDeleteVertexArrays(GLsizei n, const GLuint* arrays) {
    MGL_CTX();
    if (n < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < n; i++) {
        VertexArray* v = ctx->vaos.get(arrays[i]);
        if (v && ctx->vao.get() == v) ctx->vao = ctx->coreProfile ? nullptr : ctx->defaultVao;
        ctx->vaos.remove(arrays[i]);
    }
}

API GLboolean APIENTRY glIsVertexArray(GLuint array) {
    MGL_CTX_RET(GL_FALSE);
    return array && ctx->vaos.get(array) != nullptr;
}

API void APIENTRY glBindVertexArray(GLuint array) {
    MGL_CTX();
    if (!array) {
        ctx->vao = ctx->coreProfile ? nullptr : ctx->defaultVao;
        return;
    }
    VertexArray* v = ctx->vaos.get(array);
    if (!v) {
        if (!ctx->vaos.isName(array)) return ctx->setError(GL_INVALID_OPERATION);
        v = ctx->vaos.create(array, newVao());
    }
    ctx->vao = v;
}

static VertexArray* curVao(Context* ctx) {
    if (!ctx->vao) ctx->setError(GL_INVALID_OPERATION, "no vertex array bound");
    return ctx->vao.get();
}

static VertexArray* namedVao(Context* ctx, GLuint name) {
    VertexArray* v = name ? ctx->vaos.get(name) : nullptr;
    if (!v) ctx->setError(GL_INVALID_OPERATION);
    return v;
}

static bool validAttribType(GLenum type, int size, bool integer, bool isLong, bool normalized) {
    if (isLong) return type == GL_DOUBLE && size >= 1 && size <= 4;
    switch (type) {
    case GL_BYTE: case GL_UNSIGNED_BYTE: case GL_SHORT: case GL_UNSIGNED_SHORT: case GL_INT: case GL_UNSIGNED_INT:
        return (size >= 1 && size <= 4) || (!integer && size == GL_BGRA && type == GL_UNSIGNED_BYTE && normalized);
    case GL_FLOAT: case GL_HALF_FLOAT: case GL_DOUBLE: case GL_FIXED:
        return !integer && size >= 1 && size <= 4;
    case GL_INT_2_10_10_10_REV: case GL_UNSIGNED_INT_2_10_10_10_REV:
        return !integer && (size == 4 || size == GL_BGRA);
    case GL_UNSIGNED_INT_10F_11F_11F_REV:
        return !integer && size == 3;
    default: return false;
    }
}

static void attribFormat(Context* ctx, VertexArray* v, GLuint index, GLint size, GLenum type, bool normalized, bool integer, bool isLong,
                         GLuint relativeOffset) {
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    if (!((size >= 1 && size <= 4) || size == GL_BGRA)) return ctx->setError(GL_INVALID_VALUE);
    if (relativeOffset > 2047) return ctx->setError(GL_INVALID_VALUE);
    if (!validAttribType(type, size, integer, isLong, normalized)) {
        if (size == GL_BGRA && !normalized) return ctx->setError(GL_INVALID_OPERATION);
        if ((type == GL_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_2_10_10_10_REV) && size != 4 && size != GL_BGRA)
            return ctx->setError(GL_INVALID_OPERATION);
        if (type == GL_UNSIGNED_INT_10F_11F_11F_REV && size != 3) return ctx->setError(GL_INVALID_OPERATION);
        return ctx->setError(GL_INVALID_ENUM);
    }
    VertexAttrib& a = v->attribs[index];
    a.size = size;
    a.type = type;
    a.normalized = normalized;
    a.integer = integer;
    a.isLong = isLong;
    a.relativeOffset = relativeOffset;
    v->version++;
}

static void attribPointer(Context* ctx, GLuint index, GLint size, GLenum type, bool normalized, bool integer, bool isLong, GLsizei stride,
                          const void* pointer) {
    VertexArray* v = curVao(ctx);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    if (stride < 0 || stride > 2048) return ctx->setError(GL_INVALID_VALUE);
    if (!ctx->arrayBuffer && v != ctx->defaultVao.get() && pointer) return ctx->setError(GL_INVALID_OPERATION, "no array buffer bound");
    GLenum errBefore = ctx->error;
    attribFormat(ctx, v, index, size, type, normalized, integer, isLong, 0);
    if (ctx->error != errBefore && ctx->error != GL_NO_ERROR) return;
    VertexAttrib& a = v->attribs[index];
    a.binding = index;
    VertexBinding& b = v->bindings[index];
    b.buffer = ctx->arrayBuffer;
    b.offset = (GLintptr)pointer;
    a.clientPointer = ctx->arrayBuffer ? nullptr : pointer;
    int compSize = 0;
    switch (type) {
    case GL_BYTE: case GL_UNSIGNED_BYTE: compSize = 1; break;
    case GL_SHORT: case GL_UNSIGNED_SHORT: case GL_HALF_FLOAT: compSize = 2; break;
    case GL_DOUBLE: compSize = 8; break;
    default: compSize = 4; break;
    }
    int comps = size == GL_BGRA ? 4 : size;
    if (type == GL_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_10F_11F_11F_REV) {
        compSize = 4;
        comps = 1;
    }
    b.stride = stride ? stride : compSize * comps;
    v->version++;
}

API void APIENTRY glVertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer) {
    MGL_CTX();
    attribPointer(ctx, index, size, type, normalized, false, false, stride, pointer);
}

API void APIENTRY glVertexAttribIPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer) {
    MGL_CTX();
    attribPointer(ctx, index, size, type, false, true, false, stride, pointer);
}

API void APIENTRY glVertexAttribLPointer(GLuint index, GLint size, GLenum type, GLsizei stride, const void* pointer) {
    MGL_CTX();
    attribPointer(ctx, index, size, type, false, false, true, stride, pointer);
}

API void APIENTRY glEnableVertexAttribArray(GLuint index) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[index].enabled = true;
    v->version++;
}

API void APIENTRY glDisableVertexAttribArray(GLuint index) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[index].enabled = false;
    v->version++;
}

API void APIENTRY glEnableVertexArrayAttrib(GLuint vaobj, GLuint index) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[index].enabled = true;
    v->version++;
}

API void APIENTRY glDisableVertexArrayAttrib(GLuint vaobj, GLuint index) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[index].enabled = false;
    v->version++;
}

// --- vertex attrib binding (GL 4.3 / DSA) ---
static void bindVertexBuffer(Context* ctx, VertexArray* v, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) {
    if (bindingindex >= kMaxVertexBindings) return ctx->setError(GL_INVALID_VALUE);
    if (offset < 0 || stride < 0 || stride > 2048) return ctx->setError(GL_INVALID_VALUE);
    Buffer* b = nullptr;
    if (buffer) {
        b = ctx->share->buffers.get(buffer);
        if (!b) {
            if (!ctx->share->buffers.isName(buffer)) return ctx->setError(GL_INVALID_OPERATION);
            b = ctx->share->buffers.create(buffer, new Buffer());
        }
    }
    VertexBinding& vb = v->bindings[bindingindex];
    vb.buffer = b;
    vb.offset = offset;
    vb.stride = stride;
    v->version++;
}

API void APIENTRY glBindVertexBuffer(GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) bindVertexBuffer(ctx, v, bindingindex, buffer, offset, stride);
}

API void APIENTRY glVertexArrayVertexBuffer(GLuint vaobj, GLuint bindingindex, GLuint buffer, GLintptr offset, GLsizei stride) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) bindVertexBuffer(ctx, v, bindingindex, buffer, offset, stride);
}

API void APIENTRY glBindVertexBuffers(GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets, const GLsizei* strides) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (!v) return;
    if (first + count > kMaxVertexBindings) return ctx->setError(GL_INVALID_OPERATION);
    for (int i = 0; i < count; i++)
        bindVertexBuffer(ctx, v, first + i, buffers ? buffers[i] : 0, buffers ? offsets[i] : 0, buffers ? strides[i] : 16);
}

API void APIENTRY glVertexArrayVertexBuffers(GLuint vaobj, GLuint first, GLsizei count, const GLuint* buffers, const GLintptr* offsets,
                                             const GLsizei* strides) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    if (first + count > kMaxVertexBindings) return ctx->setError(GL_INVALID_OPERATION);
    for (int i = 0; i < count; i++)
        bindVertexBuffer(ctx, v, first + i, buffers ? buffers[i] : 0, buffers ? offsets[i] : 0, buffers ? strides[i] : 16);
}

API void APIENTRY glVertexAttribFormat(GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) attribFormat(ctx, v, attribindex, size, type, normalized, false, false, relativeoffset);
}
API void APIENTRY glVertexAttribIFormat(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) attribFormat(ctx, v, attribindex, size, type, false, true, false, relativeoffset);
}
API void APIENTRY glVertexAttribLFormat(GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) attribFormat(ctx, v, attribindex, size, type, false, false, true, relativeoffset);
}
API void APIENTRY glVertexArrayAttribFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLboolean normalized, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) attribFormat(ctx, v, attribindex, size, type, normalized, false, false, relativeoffset);
}
API void APIENTRY glVertexArrayAttribIFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) attribFormat(ctx, v, attribindex, size, type, false, true, false, relativeoffset);
}
API void APIENTRY glVertexArrayAttribLFormat(GLuint vaobj, GLuint attribindex, GLint size, GLenum type, GLuint relativeoffset) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) attribFormat(ctx, v, attribindex, size, type, false, false, true, relativeoffset);
}

static void attribBinding(Context* ctx, VertexArray* v, GLuint attribindex, GLuint bindingindex) {
    if (attribindex >= kMaxVertexAttribs || bindingindex >= kMaxVertexBindings) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[attribindex].binding = bindingindex;
    v->version++;
}

API void APIENTRY glVertexAttribBinding(GLuint attribindex, GLuint bindingindex) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) attribBinding(ctx, v, attribindex, bindingindex);
}
API void APIENTRY glVertexArrayAttribBinding(GLuint vaobj, GLuint attribindex, GLuint bindingindex) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) attribBinding(ctx, v, attribindex, bindingindex);
}

static void bindingDivisor(Context* ctx, VertexArray* v, GLuint bindingindex, GLuint divisor) {
    if (bindingindex >= kMaxVertexBindings) return ctx->setError(GL_INVALID_VALUE);
    v->bindings[bindingindex].divisor = divisor;
    v->version++;
}

API void APIENTRY glVertexBindingDivisor(GLuint bindingindex, GLuint divisor) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (v) bindingDivisor(ctx, v, bindingindex, divisor);
}
API void APIENTRY glVertexArrayBindingDivisor(GLuint vaobj, GLuint bindingindex, GLuint divisor) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (v) bindingDivisor(ctx, v, bindingindex, divisor);
}

API void APIENTRY glVertexAttribDivisor(GLuint index, GLuint divisor) {
    MGL_CTX();
    VertexArray* v = curVao(ctx);
    if (!v) return;
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    v->attribs[index].binding = index;
    v->bindings[index].divisor = divisor;
    v->version++;
}
API void APIENTRY glVertexAttribDivisorARB(GLuint index, GLuint divisor) { glVertexAttribDivisor(index, divisor); }

API void APIENTRY glVertexArrayElementBuffer(GLuint vaobj, GLuint buffer) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    Buffer* b = buffer ? ctx->share->buffers.get(buffer) : nullptr;
    if (buffer && !b) return ctx->setError(GL_INVALID_OPERATION);
    v->elementBuffer = b;
    v->version++;
}

// --- queries ---
static bool vaoAttribParam(Context* ctx, VertexArray* v, GLuint index, GLenum pname, int64_t* out) {
    if (index >= kMaxVertexAttribs) {
        ctx->setError(GL_INVALID_VALUE);
        return false;
    }
    const VertexAttrib& a = v->attribs[index];
    const VertexBinding& b = v->bindings[a.binding];
    switch (pname) {
    case GL_VERTEX_ATTRIB_ARRAY_ENABLED: *out = a.enabled; return true;
    case GL_VERTEX_ATTRIB_ARRAY_SIZE: *out = a.size; return true;
    case GL_VERTEX_ATTRIB_ARRAY_STRIDE: *out = b.stride; return true;
    case GL_VERTEX_ATTRIB_ARRAY_TYPE: *out = a.type; return true;
    case GL_VERTEX_ATTRIB_ARRAY_NORMALIZED: *out = a.normalized; return true;
    case GL_VERTEX_ATTRIB_ARRAY_INTEGER: *out = a.integer; return true;
    case GL_VERTEX_ATTRIB_ARRAY_LONG: *out = a.isLong; return true;
    case GL_VERTEX_ATTRIB_ARRAY_DIVISOR: *out = b.divisor; return true;
    case GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING: *out = b.buffer ? b.buffer->name : 0; return true;
    case GL_VERTEX_ATTRIB_BINDING: *out = a.binding; return true;
    case GL_VERTEX_ATTRIB_RELATIVE_OFFSET: *out = a.relativeOffset; return true;
    default: ctx->setError(GL_INVALID_ENUM); return false;
    }
}

API void APIENTRY glGetVertexAttribiv(GLuint index, GLenum pname, GLint* params) {
    MGL_CTX();
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
        for (int k = 0; k < 4; k++) params[k] = (GLint)ctx->currentAttrib[index][k];
        return;
    }
    VertexArray* v = curVao(ctx);
    int64_t val;
    if (v && vaoAttribParam(ctx, v, index, pname, &val)) *params = (GLint)val;
}

API void APIENTRY glGetVertexAttribfv(GLuint index, GLenum pname, GLfloat* params) {
    MGL_CTX();
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
        for (int k = 0; k < 4; k++) params[k] = ctx->currentAttrib[index][k];
        return;
    }
    VertexArray* v = curVao(ctx);
    int64_t val;
    if (v && vaoAttribParam(ctx, v, index, pname, &val)) *params = (GLfloat)val;
}

API void APIENTRY glGetVertexAttribdv(GLuint index, GLenum pname, GLdouble* params) {
    MGL_CTX();
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
        for (int k = 0; k < 4; k++) params[k] = ctx->currentAttrib[index][k];
        return;
    }
    VertexArray* v = curVao(ctx);
    int64_t val;
    if (v && vaoAttribParam(ctx, v, index, pname, &val)) *params = (GLdouble)val;
}

API void APIENTRY glGetVertexAttribIiv(GLuint index, GLenum pname, GLint* params) {
    MGL_CTX();
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
        memcpy(params, ctx->currentAttrib[index], 16);
        return;
    }
    glGetVertexAttribiv(index, pname, params);
}

API void APIENTRY glGetVertexAttribIuiv(GLuint index, GLenum pname, GLuint* params) {
    MGL_CTX();
    if (pname == GL_CURRENT_VERTEX_ATTRIB) {
        if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
        memcpy(params, ctx->currentAttrib[index], 16);
        return;
    }
    glGetVertexAttribiv(index, pname, (GLint*)params);
}

API void APIENTRY glGetVertexAttribLdv(GLuint index, GLenum pname, GLdouble* params) { glGetVertexAttribdv(index, pname, params); }

API void APIENTRY glGetVertexAttribPointerv(GLuint index, GLenum pname, void** pointer) {
    MGL_CTX();
    if (pname != GL_VERTEX_ATTRIB_ARRAY_POINTER) return ctx->setError(GL_INVALID_ENUM);
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    VertexArray* v = curVao(ctx);
    if (!v) return;
    const VertexAttrib& a = v->attribs[index];
    *pointer = a.clientPointer ? (void*)a.clientPointer : (void*)v->bindings[a.binding].offset;
}

API void APIENTRY glGetVertexArrayiv(GLuint vaobj, GLenum pname, GLint* param) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    if (pname != GL_ELEMENT_ARRAY_BUFFER_BINDING) return ctx->setError(GL_INVALID_ENUM);
    *param = v->elementBuffer ? v->elementBuffer->name : 0;
}

API void APIENTRY glGetVertexArrayIndexediv(GLuint vaobj, GLuint index, GLenum pname, GLint* param) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    int64_t val;
    if (vaoAttribParam(ctx, v, index, pname, &val)) *param = (GLint)val;
}

API void APIENTRY glGetVertexArrayIndexed64iv(GLuint vaobj, GLuint index, GLenum pname, GLint64* param) {
    MGL_CTX();
    VertexArray* v = namedVao(ctx, vaobj);
    if (!v) return;
    if (pname == GL_VERTEX_BINDING_OFFSET) {
        if (index >= kMaxVertexBindings) return ctx->setError(GL_INVALID_VALUE);
        *param = v->bindings[index].offset;
        return;
    }
    int64_t val;
    if (vaoAttribParam(ctx, v, index, pname, &val)) *param = val;
}

// --- generic attribute values ---
static void setAttrib(Context* ctx, GLuint index, float x, float y, float z, float w) {
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    float* a = ctx->currentAttrib[index];
    a[0] = x;
    a[1] = y;
    a[2] = z;
    a[3] = w;
    ctx->currentAttribType[index] = GL_FLOAT;
}

static void setAttribI(Context* ctx, GLuint index, const void* v, GLenum type) {
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE);
    memcpy(ctx->currentAttrib[index], v, 16);
    ctx->currentAttribType[index] = type;
}

#define VA_F(name, args, ...)                         \
    API void APIENTRY glVertexAttrib##name args {     \
        MGL_CTX();                                    \
        setAttrib(ctx, __VA_ARGS__);                  \
    }
VA_F(1f, (GLuint i, GLfloat x), i, x, 0, 0, 1)
VA_F(2f, (GLuint i, GLfloat x, GLfloat y), i, x, y, 0, 1)
VA_F(3f, (GLuint i, GLfloat x, GLfloat y, GLfloat z), i, x, y, z, 1)
VA_F(4f, (GLuint i, GLfloat x, GLfloat y, GLfloat z, GLfloat w), i, x, y, z, w)
VA_F(1fv, (GLuint i, const GLfloat* v), i, v[0], 0, 0, 1)
VA_F(2fv, (GLuint i, const GLfloat* v), i, v[0], v[1], 0, 1)
VA_F(3fv, (GLuint i, const GLfloat* v), i, v[0], v[1], v[2], 1)
VA_F(4fv, (GLuint i, const GLfloat* v), i, v[0], v[1], v[2], v[3])
VA_F(1d, (GLuint i, GLdouble x), i, (float)x, 0, 0, 1)
VA_F(2d, (GLuint i, GLdouble x, GLdouble y), i, (float)x, (float)y, 0, 1)
VA_F(3d, (GLuint i, GLdouble x, GLdouble y, GLdouble z), i, (float)x, (float)y, (float)z, 1)
VA_F(4d, (GLuint i, GLdouble x, GLdouble y, GLdouble z, GLdouble w), i, (float)x, (float)y, (float)z, (float)w)
VA_F(1dv, (GLuint i, const GLdouble* v), i, (float)v[0], 0, 0, 1)
VA_F(2dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], 0, 1)
VA_F(3dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], (float)v[2], 1)
VA_F(4dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], (float)v[2], (float)v[3])
VA_F(1s, (GLuint i, GLshort x), i, x, 0, 0, 1)
VA_F(2s, (GLuint i, GLshort x, GLshort y), i, x, y, 0, 1)
VA_F(3s, (GLuint i, GLshort x, GLshort y, GLshort z), i, x, y, z, 1)
VA_F(4s, (GLuint i, GLshort x, GLshort y, GLshort z, GLshort w), i, x, y, z, w)
VA_F(1sv, (GLuint i, const GLshort* v), i, v[0], 0, 0, 1)
VA_F(2sv, (GLuint i, const GLshort* v), i, v[0], v[1], 0, 1)
VA_F(3sv, (GLuint i, const GLshort* v), i, v[0], v[1], v[2], 1)
VA_F(4sv, (GLuint i, const GLshort* v), i, v[0], v[1], v[2], v[3])
VA_F(4bv, (GLuint i, const GLbyte* v), i, v[0], v[1], v[2], v[3])
VA_F(4iv, (GLuint i, const GLint* v), i, (float)v[0], (float)v[1], (float)v[2], (float)v[3])
VA_F(4ubv, (GLuint i, const GLubyte* v), i, v[0], v[1], v[2], v[3])
VA_F(4usv, (GLuint i, const GLushort* v), i, v[0], v[1], v[2], v[3])
VA_F(4uiv, (GLuint i, const GLuint* v), i, (float)v[0], (float)v[1], (float)v[2], (float)v[3])
VA_F(4Nub, (GLuint i, GLubyte x, GLubyte y, GLubyte z, GLubyte w), i, x / 255.f, y / 255.f, z / 255.f, w / 255.f)
VA_F(4Nubv, (GLuint i, const GLubyte* v), i, v[0] / 255.f, v[1] / 255.f, v[2] / 255.f, v[3] / 255.f)
VA_F(4Nbv, (GLuint i, const GLbyte* v), i, std::max(-1.f, v[0] / 127.f), std::max(-1.f, v[1] / 127.f), std::max(-1.f, v[2] / 127.f), std::max(-1.f, v[3] / 127.f))
VA_F(4Nsv, (GLuint i, const GLshort* v), i, std::max(-1.f, v[0] / 32767.f), std::max(-1.f, v[1] / 32767.f), std::max(-1.f, v[2] / 32767.f), std::max(-1.f, v[3] / 32767.f))
VA_F(4Niv, (GLuint i, const GLint* v), i, std::max(-1.f, v[0] / 2147483647.f), std::max(-1.f, v[1] / 2147483647.f), std::max(-1.f, v[2] / 2147483647.f), std::max(-1.f, v[3] / 2147483647.f))
VA_F(4Nusv, (GLuint i, const GLushort* v), i, v[0] / 65535.f, v[1] / 65535.f, v[2] / 65535.f, v[3] / 65535.f)
VA_F(4Nuiv, (GLuint i, const GLuint* v), i, v[0] / 4294967295.f, v[1] / 4294967295.f, v[2] / 4294967295.f, v[3] / 4294967295.f)
VA_F(L1d, (GLuint i, GLdouble x), i, (float)x, 0, 0, 1)
VA_F(L2d, (GLuint i, GLdouble x, GLdouble y), i, (float)x, (float)y, 0, 1)
VA_F(L3d, (GLuint i, GLdouble x, GLdouble y, GLdouble z), i, (float)x, (float)y, (float)z, 1)
VA_F(L4d, (GLuint i, GLdouble x, GLdouble y, GLdouble z, GLdouble w), i, (float)x, (float)y, (float)z, (float)w)
VA_F(L1dv, (GLuint i, const GLdouble* v), i, (float)v[0], 0, 0, 1)
VA_F(L2dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], 0, 1)
VA_F(L3dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], (float)v[2], 1)
VA_F(L4dv, (GLuint i, const GLdouble* v), i, (float)v[0], (float)v[1], (float)v[2], (float)v[3])

#define VA_I(name, T, Enum, args, ...)                  \
    API void APIENTRY glVertexAttribI##name args {      \
        MGL_CTX();                                      \
        T v[4] = {__VA_ARGS__};                         \
        setAttribI(ctx, i, v, Enum);                    \
    }
VA_I(1i, GLint, GL_INT, (GLuint i, GLint x), x, 0, 0, 1)
VA_I(2i, GLint, GL_INT, (GLuint i, GLint x, GLint y), x, y, 0, 1)
VA_I(3i, GLint, GL_INT, (GLuint i, GLint x, GLint y, GLint z), x, y, z, 1)
VA_I(4i, GLint, GL_INT, (GLuint i, GLint x, GLint y, GLint z, GLint w), x, y, z, w)
VA_I(1ui, GLuint, GL_UNSIGNED_INT, (GLuint i, GLuint x), x, 0, 0, 1)
VA_I(2ui, GLuint, GL_UNSIGNED_INT, (GLuint i, GLuint x, GLuint y), x, y, 0, 1)
VA_I(3ui, GLuint, GL_UNSIGNED_INT, (GLuint i, GLuint x, GLuint y, GLuint z), x, y, z, 1)
VA_I(4ui, GLuint, GL_UNSIGNED_INT, (GLuint i, GLuint x, GLuint y, GLuint z, GLuint w), x, y, z, w)
VA_I(1iv, GLint, GL_INT, (GLuint i, const GLint* p), p[0], 0, 0, 1)
VA_I(2iv, GLint, GL_INT, (GLuint i, const GLint* p), p[0], p[1], 0, 1)
VA_I(3iv, GLint, GL_INT, (GLuint i, const GLint* p), p[0], p[1], p[2], 1)
VA_I(4iv, GLint, GL_INT, (GLuint i, const GLint* p), p[0], p[1], p[2], p[3])
VA_I(1uiv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLuint* p), p[0], 0, 0, 1)
VA_I(2uiv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLuint* p), p[0], p[1], 0, 1)
VA_I(3uiv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLuint* p), p[0], p[1], p[2], 1)
VA_I(4uiv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLuint* p), p[0], p[1], p[2], p[3])
VA_I(4bv, GLint, GL_INT, (GLuint i, const GLbyte* p), p[0], p[1], p[2], p[3])
VA_I(4sv, GLint, GL_INT, (GLuint i, const GLshort* p), p[0], p[1], p[2], p[3])
VA_I(4ubv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLubyte* p), p[0], p[1], p[2], p[3])
VA_I(4usv, GLuint, GL_UNSIGNED_INT, (GLuint i, const GLushort* p), p[0], p[1], p[2], p[3])

// Packed attribute setters (GL 3.3)
static float unpackSnorm10(int v) { return std::max(-1.f, v / 511.f); }
API void APIENTRY glVertexAttribP4ui(GLuint index, GLenum type, GLboolean normalized, GLuint value) {
    MGL_CTX();
    float c[4];
    if (type == GL_UNSIGNED_INT_2_10_10_10_REV) {
        uint32_t r = value & 1023, g = (value >> 10) & 1023, b = (value >> 20) & 1023, a = value >> 30;
        if (normalized) {
            c[0] = r / 1023.f; c[1] = g / 1023.f; c[2] = b / 1023.f; c[3] = a / 3.f;
        } else {
            c[0] = r; c[1] = g; c[2] = b; c[3] = a;
        }
    } else if (type == GL_INT_2_10_10_10_REV) {
        auto sx = [](uint32_t v, int bits) { int s = 32 - bits; return (int)(v << s) >> s; };
        int r = sx(value & 1023, 10), g = sx((value >> 10) & 1023, 10), b = sx((value >> 20) & 1023, 10), a = sx(value >> 30, 2);
        if (normalized) {
            c[0] = unpackSnorm10(r); c[1] = unpackSnorm10(g); c[2] = unpackSnorm10(b); c[3] = std::max(-1.f, (float)a);
        } else {
            c[0] = r; c[1] = g; c[2] = b; c[3] = a;
        }
    } else {
        return ctx->setError(GL_INVALID_ENUM);
    }
    setAttrib(ctx, index, c[0], c[1], c[2], c[3]);
}
API void APIENTRY glVertexAttribP4uiv(GLuint index, GLenum type, GLboolean normalized, const GLuint* value) { glVertexAttribP4ui(index, type, normalized, *value); }
