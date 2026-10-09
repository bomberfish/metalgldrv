// Fixed-function state setters, enable/disable, clears.
#import "gl/context.h"
#include <cmath>
#include "gl/internal_shaders.h"
#include "util/log.h"

using namespace mgl;

#define API extern "C" __attribute__((visibility("default")))

static bool validBlendFactor(GLenum f) {
    switch (f) {
    case GL_ZERO: case GL_ONE: case GL_SRC_COLOR: case GL_ONE_MINUS_SRC_COLOR: case GL_DST_COLOR: case GL_ONE_MINUS_DST_COLOR:
    case GL_SRC_ALPHA: case GL_ONE_MINUS_SRC_ALPHA: case GL_DST_ALPHA: case GL_ONE_MINUS_DST_ALPHA: case GL_CONSTANT_COLOR:
    case GL_ONE_MINUS_CONSTANT_COLOR: case GL_CONSTANT_ALPHA: case GL_ONE_MINUS_CONSTANT_ALPHA: case GL_SRC_ALPHA_SATURATE:
    case GL_SRC1_COLOR: case GL_ONE_MINUS_SRC1_COLOR: case GL_SRC1_ALPHA: case GL_ONE_MINUS_SRC1_ALPHA:
        return true;
    default: return false;
    }
}

static bool validBlendEq(GLenum e) {
    return e == GL_FUNC_ADD || e == GL_FUNC_SUBTRACT || e == GL_FUNC_REVERSE_SUBTRACT || e == GL_MIN || e == GL_MAX;
}

// KHR_blend_equation_advanced: BlendEquation(i) only (BlendEquationSeparate(i) rejects them).
static bool validAdvancedBlendEq(Context* ctx, GLenum e) {
    if (!ctx->dev->caps.programmableBlending) return false;
    for (uint32_t a : glsl::kAdvancedBlendEquations)
        if (e == a) return true;
    return false;
}

static bool validCompare(GLenum f) { return f >= GL_NEVER && f <= GL_ALWAYS; }

static bool validStencilOp(GLenum op) {
    switch (op) {
    case GL_KEEP: case GL_ZERO: case GL_REPLACE: case GL_INCR: case GL_DECR: case GL_INVERT: case GL_INCR_WRAP: case GL_DECR_WRAP:
        return true;
    default: return false;
    }
}

// Returns pointer to the boolean for simple capabilities, or handles indexed ones.
static bool setCap(Context* ctx, GLenum cap, bool v, int index = -1) {
    auto all = [&](auto fn) {
        if (index >= 0) fn(index);
        else
            for (int i = 0; i < kMaxDrawBuffers; i++) fn(i);
    };
    switch (cap) {
    case GL_BLEND: all([&](int i) { ctx->blend[i].enabled = v; }); return true;
    case GL_SCISSOR_TEST:
        if (index >= 0) ctx->scissorEnabled = v ? ctx->scissorEnabled | (1u << index) : ctx->scissorEnabled & ~(1u << index);
        else ctx->scissorEnabled = v ? 0xffff : 0;
        return true;
    default: break;
    }
    if (index >= 0) return false;
    switch (cap) {
    case GL_DEPTH_TEST: ctx->depthTest = v; return true;
    case GL_STENCIL_TEST: ctx->stencilTest = v; return true;
    case GL_CULL_FACE: ctx->cullFace = v; return true;
    case GL_POLYGON_OFFSET_FILL: ctx->polygonOffsetFill = v; return true;
    case GL_POLYGON_OFFSET_LINE: ctx->polygonOffsetLine = v; return true;
    case GL_POLYGON_OFFSET_POINT: ctx->polygonOffsetPoint = v; return true;
    case GL_PROGRAM_POINT_SIZE: ctx->programPointSize = v; return true;
    case GL_PRIMITIVE_RESTART: ctx->primitiveRestart = v; return true;
    case GL_PRIMITIVE_RESTART_FIXED_INDEX: ctx->primitiveRestartFixed = v; return true;
    case GL_DEPTH_CLAMP: ctx->depthClamp = v; return true;
    case GL_RASTERIZER_DISCARD: ctx->rasterizerDiscard = v; return true;
    case GL_MULTISAMPLE: ctx->multisample = v; return true;
    case GL_SAMPLE_ALPHA_TO_COVERAGE: ctx->sampleAlphaToCoverage = v; return true;
    case GL_SAMPLE_ALPHA_TO_ONE: ctx->sampleAlphaToOne = v; return true;
    case GL_SAMPLE_COVERAGE: ctx->sampleCoverage = v; return true;
    case GL_SAMPLE_MASK: ctx->sampleMaskEnabled = v; return true;
    case GL_SAMPLE_SHADING: ctx->sampleShading = v; return true;
    case GL_COLOR_LOGIC_OP: ctx->colorLogicOp = v; return true;
    case GL_DITHER: ctx->dither = v; return true;
    case GL_BLEND_ADVANCED_COHERENT_KHR:
        if (!ctx->dev->caps.programmableBlending) return false;
        ctx->blendAdvancedCoherent = v;
        return true;
    case GL_FRAMEBUFFER_SRGB:
        if (ctx->framebufferSRGB != v) {
            ctx->applyClearNow();  // pending clears belong to the current attachment views
            ctx->framebufferSRGB = v;
            ctx->passDirty = true;
        }
        return true;
    case GL_LINE_SMOOTH: ctx->lineSmooth = v; return true;
    case GL_POLYGON_SMOOTH: ctx->polygonSmooth = v; return true;
    case GL_TEXTURE_CUBE_MAP_SEAMLESS: ctx->textureCubeSeamless = v; return true;
    case GL_DEBUG_OUTPUT: ctx->debugOutput = v; return true;
    case GL_DEBUG_OUTPUT_SYNCHRONOUS: ctx->debugOutputSync = v; return true;
    default: break;
    }
    if (cap >= GL_CLIP_DISTANCE0 && cap < GL_CLIP_DISTANCE0 + kMaxClipDistances) {
        uint32_t bit = 1u << (cap - GL_CLIP_DISTANCE0);
        ctx->clipDistanceMask = v ? ctx->clipDistanceMask | bit : ctx->clipDistanceMask & ~bit;
        return true;
    }
    return false;
}

bool mglCompatCap(Context* ctx, GLenum cap, bool v);  // compat.mm
bool mglCompatIsEnabled(Context* ctx, GLenum cap, bool* out);

API void APIENTRY glEnable(GLenum cap) {
    MGL_CTX();
    if (!setCap(ctx, cap, true) && !(!ctx->coreProfile && mglCompatCap(ctx, cap, true))) ctx->setError(GL_INVALID_ENUM, "glEnable(0x%x)", cap);
}

API void APIENTRY glDisable(GLenum cap) {
    MGL_CTX();
    if (!setCap(ctx, cap, false) && !(!ctx->coreProfile && mglCompatCap(ctx, cap, false))) ctx->setError(GL_INVALID_ENUM, "glDisable(0x%x)", cap);
}

API void APIENTRY glEnablei(GLenum cap, GLuint index) {
    MGL_CTX();
    int max = cap == GL_SCISSOR_TEST ? kMaxViewports : kMaxDrawBuffers;
    if (index >= (GLuint)max) return ctx->setError(GL_INVALID_VALUE);
    if (!setCap(ctx, cap, true, index)) ctx->setError(GL_INVALID_ENUM, "glEnablei(0x%x)", cap);
}

API void APIENTRY glDisablei(GLenum cap, GLuint index) {
    MGL_CTX();
    int max = cap == GL_SCISSOR_TEST ? kMaxViewports : kMaxDrawBuffers;
    if (index >= (GLuint)max) return ctx->setError(GL_INVALID_VALUE);
    if (!setCap(ctx, cap, false, index)) ctx->setError(GL_INVALID_ENUM, "glDisablei(0x%x)", cap);
}

bool mglIsEnabled(Context* ctx, GLenum cap, int index, bool* out) {
    switch (cap) {
    case GL_BLEND: *out = ctx->blend[index < 0 ? 0 : index].enabled; return true;
    case GL_SCISSOR_TEST: *out = ctx->scissorEnabled & (1u << (index < 0 ? 0 : index)); return true;
    default: break;
    }
    if (index >= 0) return false;
    switch (cap) {
    case GL_DEPTH_TEST: *out = ctx->depthTest; return true;
    case GL_STENCIL_TEST: *out = ctx->stencilTest; return true;
    case GL_CULL_FACE: *out = ctx->cullFace; return true;
    case GL_POLYGON_OFFSET_FILL: *out = ctx->polygonOffsetFill; return true;
    case GL_POLYGON_OFFSET_LINE: *out = ctx->polygonOffsetLine; return true;
    case GL_POLYGON_OFFSET_POINT: *out = ctx->polygonOffsetPoint; return true;
    case GL_PROGRAM_POINT_SIZE: *out = ctx->programPointSize; return true;
    case GL_PRIMITIVE_RESTART: *out = ctx->primitiveRestart; return true;
    case GL_PRIMITIVE_RESTART_FIXED_INDEX: *out = ctx->primitiveRestartFixed; return true;
    case GL_DEPTH_CLAMP: *out = ctx->depthClamp; return true;
    case GL_RASTERIZER_DISCARD: *out = ctx->rasterizerDiscard; return true;
    case GL_MULTISAMPLE: *out = ctx->multisample; return true;
    case GL_SAMPLE_ALPHA_TO_COVERAGE: *out = ctx->sampleAlphaToCoverage; return true;
    case GL_SAMPLE_ALPHA_TO_ONE: *out = ctx->sampleAlphaToOne; return true;
    case GL_SAMPLE_COVERAGE: *out = ctx->sampleCoverage; return true;
    case GL_SAMPLE_MASK: *out = ctx->sampleMaskEnabled; return true;
    case GL_SAMPLE_SHADING: *out = ctx->sampleShading; return true;
    case GL_COLOR_LOGIC_OP: *out = ctx->colorLogicOp; return true;
    case GL_DITHER: *out = ctx->dither; return true;
    case GL_BLEND_ADVANCED_COHERENT_KHR: *out = ctx->blendAdvancedCoherent; return ctx->dev->caps.programmableBlending;
    case GL_FRAMEBUFFER_SRGB: *out = ctx->framebufferSRGB; return true;
    case GL_LINE_SMOOTH: *out = ctx->lineSmooth; return true;
    case GL_POLYGON_SMOOTH: *out = ctx->polygonSmooth; return true;
    case GL_TEXTURE_CUBE_MAP_SEAMLESS: *out = ctx->textureCubeSeamless; return true;
    case GL_DEBUG_OUTPUT: *out = ctx->debugOutput; return true;
    case GL_DEBUG_OUTPUT_SYNCHRONOUS: *out = ctx->debugOutputSync; return true;
    default: break;
    }
    if (cap >= GL_CLIP_DISTANCE0 && cap < GL_CLIP_DISTANCE0 + kMaxClipDistances) {
        *out = ctx->clipDistanceMask & (1u << (cap - GL_CLIP_DISTANCE0));
        return true;
    }
    if (!ctx->coreProfile) return mglCompatIsEnabled(ctx, cap, out);
    return false;
}

API GLboolean APIENTRY glIsEnabled(GLenum cap) {
    MGL_CTX_RET(GL_FALSE);
    bool v = false;
    if (!mglIsEnabled(ctx, cap, -1, &v)) {
        ctx->setError(GL_INVALID_ENUM, "glIsEnabled(0x%x)", cap);
        return GL_FALSE;
    }
    return v;
}

API GLboolean APIENTRY glIsEnabledi(GLenum cap, GLuint index) {
    MGL_CTX_RET(GL_FALSE);
    bool v = false;
    if ((cap == GL_SCISSOR_TEST && index >= (GLuint)kMaxViewports) || (cap == GL_BLEND && index >= (GLuint)kMaxDrawBuffers)) {
        ctx->setError(GL_INVALID_VALUE);
        return GL_FALSE;
    }
    if (!mglIsEnabled(ctx, cap, index, &v)) {
        ctx->setError(GL_INVALID_ENUM, "glIsEnabledi(0x%x)", cap);
        return GL_FALSE;
    }
    return v;
}

// --- viewport / scissor / depth range ---
API void APIENTRY glViewport(GLint x, GLint y, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    {
        // Log distinct full-size viewport changes (diagnostics for window scaling issues).
        static int lastW = -1, lastH = -1, logged = 0;
        if ((w != lastW || h != lastH) && logged < 40 && (!ctx->drawFb || ctx->drawFb->isDefault)) {
            MGL_INFO("default-framebuffer viewport %dx%d at (%d,%d)", w, h, x, y);
            lastW = w;
            lastH = h;
            logged++;
        }
    }
    w = std::min(w, 16384);
    h = std::min(h, 16384);
    for (int i = 0; i < kMaxViewports; i++) {
        ctx->viewport[i][0] = x;
        ctx->viewport[i][1] = y;
        ctx->viewport[i][2] = w;
        ctx->viewport[i][3] = h;
    }
}

API void APIENTRY glViewportIndexedf(GLuint index, GLfloat x, GLfloat y, GLfloat w, GLfloat h) {
    MGL_CTX();
    if (index >= kMaxViewports) return ctx->setError(GL_INVALID_VALUE);
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    ctx->viewport[index][0] = std::clamp(x, -32768.f, 32767.f);
    ctx->viewport[index][1] = std::clamp(y, -32768.f, 32767.f);
    ctx->viewport[index][2] = std::min(w, 16384.f);
    ctx->viewport[index][3] = std::min(h, 16384.f);
}

API void APIENTRY glViewportIndexedfv(GLuint index, const GLfloat* v) { glViewportIndexedf(index, v[0], v[1], v[2], v[3]); }

API void APIENTRY glViewportArrayv(GLuint first, GLsizei count, const GLfloat* v) {
    MGL_CTX();
    if (count < 0 || first + count > kMaxViewports) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++) glViewportIndexedf(first + i, v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}

API void APIENTRY glScissor(GLint x, GLint y, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < kMaxViewports; i++) {
        ctx->scissor[i][0] = x;
        ctx->scissor[i][1] = y;
        ctx->scissor[i][2] = w;
        ctx->scissor[i][3] = h;
    }
}

API void APIENTRY glScissorIndexed(GLuint index, GLint x, GLint y, GLsizei w, GLsizei h) {
    MGL_CTX();
    if (index >= kMaxViewports || w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    ctx->scissor[index][0] = x;
    ctx->scissor[index][1] = y;
    ctx->scissor[index][2] = w;
    ctx->scissor[index][3] = h;
}

API void APIENTRY glScissorIndexedv(GLuint index, const GLint* v) { glScissorIndexed(index, v[0], v[1], v[2], v[3]); }

API void APIENTRY glScissorArrayv(GLuint first, GLsizei count, const GLint* v) {
    MGL_CTX();
    if (count < 0 || first + count > kMaxViewports) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++) glScissorIndexed(first + i, v[4 * i], v[4 * i + 1], v[4 * i + 2], v[4 * i + 3]);
}

API void APIENTRY glDepthRange(GLdouble n, GLdouble f) {
    MGL_CTX();
    for (int i = 0; i < kMaxViewports; i++) {
        ctx->depthRange[i][0] = std::clamp(n, 0.0, 1.0);
        ctx->depthRange[i][1] = std::clamp(f, 0.0, 1.0);
    }
}
API void APIENTRY glDepthRangef(GLfloat n, GLfloat f) { glDepthRange(n, f); }

API void APIENTRY glDepthRangeIndexed(GLuint index, GLdouble n, GLdouble f) {
    MGL_CTX();
    if (index >= kMaxViewports) return ctx->setError(GL_INVALID_VALUE);
    ctx->depthRange[index][0] = std::clamp(n, 0.0, 1.0);
    ctx->depthRange[index][1] = std::clamp(f, 0.0, 1.0);
}

API void APIENTRY glDepthRangeArrayv(GLuint first, GLsizei count, const GLdouble* v) {
    MGL_CTX();
    if (count < 0 || first + count > kMaxViewports) return ctx->setError(GL_INVALID_VALUE);
    for (int i = 0; i < count; i++) glDepthRangeIndexed(first + i, v[2 * i], v[2 * i + 1]);
}

API void APIENTRY glDepthRangeArraydvNV(GLuint first, GLsizei count, const GLdouble* v) { glDepthRangeArrayv(first, count, v); }

API void APIENTRY glClipControl(GLenum origin, GLenum depth) {
    MGL_CTX();
    if ((origin != GL_LOWER_LEFT && origin != GL_UPPER_LEFT) || (depth != GL_NEGATIVE_ONE_TO_ONE && depth != GL_ZERO_TO_ONE))
        return ctx->setError(GL_INVALID_ENUM);
    ctx->clipOrigin = origin;
    ctx->clipDepthMode = depth;
}

// --- blending ---
API void APIENTRY glBlendFunc(GLenum s, GLenum d) {
    MGL_CTX();
    if (!validBlendFactor(s) || !validBlendFactor(d)) return ctx->setError(GL_INVALID_ENUM);
    for (auto& b : ctx->blend) {
        b.srcRGB = b.srcA = s;
        b.dstRGB = b.dstA = d;
    }
}

API void APIENTRY glBlendFuncSeparate(GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    MGL_CTX();
    if (!validBlendFactor(sr) || !validBlendFactor(dr) || !validBlendFactor(sa) || !validBlendFactor(da))
        return ctx->setError(GL_INVALID_ENUM);
    for (auto& b : ctx->blend) {
        b.srcRGB = sr;
        b.dstRGB = dr;
        b.srcA = sa;
        b.dstA = da;
    }
}

API void APIENTRY glBlendFunci(GLuint buf, GLenum s, GLenum d) {
    MGL_CTX();
    if (buf >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (!validBlendFactor(s) || !validBlendFactor(d)) return ctx->setError(GL_INVALID_ENUM);
    auto& b = ctx->blend[buf];
    b.srcRGB = b.srcA = s;
    b.dstRGB = b.dstA = d;
}

API void APIENTRY glBlendFuncSeparatei(GLuint buf, GLenum sr, GLenum dr, GLenum sa, GLenum da) {
    MGL_CTX();
    if (buf >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (!validBlendFactor(sr) || !validBlendFactor(dr) || !validBlendFactor(sa) || !validBlendFactor(da))
        return ctx->setError(GL_INVALID_ENUM);
    auto& b = ctx->blend[buf];
    b.srcRGB = sr;
    b.dstRGB = dr;
    b.srcA = sa;
    b.dstA = da;
}

API void APIENTRY glBlendEquation(GLenum m) {
    MGL_CTX();
    if (!validBlendEq(m)) {
        if (!validAdvancedBlendEq(ctx, m)) return ctx->setError(GL_INVALID_ENUM);
        ctx->advancedBlendUsed = true;
    }
    for (auto& b : ctx->blend) b.eqRGB = b.eqA = m;
}

API void APIENTRY glBlendEquationSeparate(GLenum mr, GLenum ma) {
    MGL_CTX();
    if (!validBlendEq(mr) || !validBlendEq(ma)) return ctx->setError(GL_INVALID_ENUM);
    for (auto& b : ctx->blend) {
        b.eqRGB = mr;
        b.eqA = ma;
    }
}

API void APIENTRY glBlendEquationi(GLuint buf, GLenum m) {
    MGL_CTX();
    if (buf >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (!validBlendEq(m)) {
        if (!validAdvancedBlendEq(ctx, m)) return ctx->setError(GL_INVALID_ENUM);
        ctx->advancedBlendUsed = true;
    }
    ctx->blend[buf].eqRGB = ctx->blend[buf].eqA = m;
}

API void APIENTRY glBlendEquationSeparatei(GLuint buf, GLenum mr, GLenum ma) {
    MGL_CTX();
    if (buf >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    if (!validBlendEq(mr) || !validBlendEq(ma)) return ctx->setError(GL_INVALID_ENUM);
    ctx->blend[buf].eqRGB = mr;
    ctx->blend[buf].eqA = ma;
}

// KHR_blend_equation_advanced: advanced blending reads the destination with framebuffer fetch,
// which observes earlier fragments in order, so there is nothing to wait for.
API void APIENTRY glBlendBarrierKHR(void) {
    MGL_CTX();
    (void)ctx;
}

API void APIENTRY glBlendColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    MGL_CTX();
    ctx->blendColor[0] = r;
    ctx->blendColor[1] = g;
    ctx->blendColor[2] = b;
    ctx->blendColor[3] = a;
}

API void APIENTRY glColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    MGL_CTX();
    uint8_t m = (r ? 1 : 0) | (g ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
    for (auto& bl : ctx->blend) bl.colorMask = m;
}

API void APIENTRY glColorMaski(GLuint buf, GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    MGL_CTX();
    if (buf >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    ctx->blend[buf].colorMask = (r ? 1 : 0) | (g ? 2 : 0) | (b ? 4 : 0) | (a ? 8 : 0);
}

API void APIENTRY glLogicOp(GLenum op) {
    MGL_CTX();
    if (op < GL_CLEAR || op > GL_SET) return ctx->setError(GL_INVALID_ENUM);
    ctx->logicOp = op;
}

// --- depth / stencil ---
API void APIENTRY glDepthFunc(GLenum f) {
    MGL_CTX();
    if (!validCompare(f)) return ctx->setError(GL_INVALID_ENUM);
    ctx->depthFunc = f;
}

API void APIENTRY glDepthMask(GLboolean m) {
    MGL_CTX();
    ctx->depthWrite = m;
}

API void APIENTRY glStencilFuncSeparate(GLenum face, GLenum func, GLint ref, GLuint mask) {
    MGL_CTX();
    if (!validCompare(func)) return ctx->setError(GL_INVALID_ENUM);
    if (face != GL_FRONT && face != GL_BACK && face != GL_FRONT_AND_BACK) return ctx->setError(GL_INVALID_ENUM);
    for (int f = 0; f < 2; f++) {
        if ((f == 0 && face == GL_BACK) || (f == 1 && face == GL_FRONT)) continue;
        ctx->stencil[f].func = func;
        ctx->stencil[f].ref = ref;
        ctx->stencil[f].valueMask = mask;
    }
}

API void APIENTRY glStencilFunc(GLenum func, GLint ref, GLuint mask) { glStencilFuncSeparate(GL_FRONT_AND_BACK, func, ref, mask); }

API void APIENTRY glStencilOpSeparate(GLenum face, GLenum sfail, GLenum dpfail, GLenum dppass) {
    MGL_CTX();
    if (!validStencilOp(sfail) || !validStencilOp(dpfail) || !validStencilOp(dppass)) return ctx->setError(GL_INVALID_ENUM);
    if (face != GL_FRONT && face != GL_BACK && face != GL_FRONT_AND_BACK) return ctx->setError(GL_INVALID_ENUM);
    for (int f = 0; f < 2; f++) {
        if ((f == 0 && face == GL_BACK) || (f == 1 && face == GL_FRONT)) continue;
        ctx->stencil[f].sfail = sfail;
        ctx->stencil[f].dpfail = dpfail;
        ctx->stencil[f].dppass = dppass;
    }
}

API void APIENTRY glStencilOp(GLenum sfail, GLenum dpfail, GLenum dppass) { glStencilOpSeparate(GL_FRONT_AND_BACK, sfail, dpfail, dppass); }

API void APIENTRY glStencilMaskSeparate(GLenum face, GLuint mask) {
    MGL_CTX();
    if (face != GL_FRONT && face != GL_BACK && face != GL_FRONT_AND_BACK) return ctx->setError(GL_INVALID_ENUM);
    if (face != GL_BACK) ctx->stencil[0].writeMask = mask;
    if (face != GL_FRONT) ctx->stencil[1].writeMask = mask;
}

API void APIENTRY glStencilMask(GLuint mask) { glStencilMaskSeparate(GL_FRONT_AND_BACK, mask); }

// --- rasterization ---
API void APIENTRY glCullFace(GLenum mode) {
    MGL_CTX();
    if (mode != GL_FRONT && mode != GL_BACK && mode != GL_FRONT_AND_BACK) return ctx->setError(GL_INVALID_ENUM);
    ctx->cullMode = mode;
}

API void APIENTRY glFrontFace(GLenum mode) {
    MGL_CTX();
    if (mode != GL_CW && mode != GL_CCW) return ctx->setError(GL_INVALID_ENUM);
    ctx->frontFace = mode;
}

API void APIENTRY glPolygonMode(GLenum face, GLenum mode) {
    MGL_CTX();
    if (face != GL_FRONT_AND_BACK && (ctx->coreProfile || (face != GL_FRONT && face != GL_BACK))) return ctx->setError(GL_INVALID_ENUM);
    if (mode != GL_POINT && mode != GL_LINE && mode != GL_FILL) return ctx->setError(GL_INVALID_ENUM);
    ctx->polygonMode = mode;
}

API void APIENTRY glPolygonOffset(GLfloat factor, GLfloat units) {
    MGL_CTX();
    ctx->polygonOffsetFactor = factor;
    ctx->polygonOffsetUnits = units;
    ctx->polygonOffsetClamp = 0;
}

API void APIENTRY glPolygonOffsetClamp(GLfloat factor, GLfloat units, GLfloat clamp) {
    MGL_CTX();
    ctx->polygonOffsetFactor = factor;
    ctx->polygonOffsetUnits = units;
    ctx->polygonOffsetClamp = clamp;
}
API void APIENTRY glPolygonOffsetClampEXT(GLfloat f, GLfloat u, GLfloat c) { glPolygonOffsetClamp(f, u, c); }

API void APIENTRY glLineWidth(GLfloat w) {
    MGL_CTX();
    if (w <= 0) return ctx->setError(GL_INVALID_VALUE);
    if (ctx->coreProfile && ctx->forwardCompatible && w > 1) return ctx->setError(GL_INVALID_VALUE);
    ctx->lineWidth = w;
}

API void APIENTRY glPointSize(GLfloat s) {
    MGL_CTX();
    if (s <= 0) return ctx->setError(GL_INVALID_VALUE);
    ctx->pointSize = s;
}

API void APIENTRY glPointParameterf(GLenum pname, GLfloat param) {
    MGL_CTX();
    switch (pname) {
    case GL_POINT_FADE_THRESHOLD_SIZE:
        if (param < 0) return ctx->setError(GL_INVALID_VALUE);
        ctx->pointFadeThreshold = param;
        break;
    case GL_POINT_SPRITE_COORD_ORIGIN:
        if ((GLenum)param != GL_LOWER_LEFT && (GLenum)param != GL_UPPER_LEFT) return ctx->setError(GL_INVALID_ENUM);
        ctx->pointSpriteOrigin = (GLenum)param;
        break;
    default:
        if (!ctx->coreProfile && (pname == GL_POINT_SIZE_MIN || pname == GL_POINT_SIZE_MAX || pname == GL_POINT_DISTANCE_ATTENUATION))
            return;
        ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glPointParameteri(GLenum pname, GLint param) { glPointParameterf(pname, (GLfloat)param); }
API void APIENTRY glPointParameterfv(GLenum pname, const GLfloat* p) { glPointParameterf(pname, p[0]); }
API void APIENTRY glPointParameteriv(GLenum pname, const GLint* p) { glPointParameterf(pname, (GLfloat)p[0]); }

API void APIENTRY glPrimitiveRestartIndex(GLuint index) {
    MGL_CTX();
    ctx->primitiveRestartIndex = index;
}

API void APIENTRY glProvokingVertex(GLenum mode) {
    MGL_CTX();
    if (mode != GL_FIRST_VERTEX_CONVENTION && mode != GL_LAST_VERTEX_CONVENTION) return ctx->setError(GL_INVALID_ENUM);
    ctx->provokingVertex = mode;
}

API void APIENTRY glSampleCoverage(GLfloat value, GLboolean invert) {
    MGL_CTX();
    ctx->sampleCoverageValue = std::clamp(value, 0.f, 1.f);
    ctx->sampleCoverageInvert = invert;
}

API void APIENTRY glSampleMaski(GLuint index, GLbitfield mask) {
    MGL_CTX();
    if (index >= 1) return ctx->setError(GL_INVALID_VALUE);
    ctx->sampleMaskValue = mask;
}

API void APIENTRY glMinSampleShading(GLfloat v) {
    MGL_CTX();
    ctx->minSampleShading = std::clamp(v, 0.f, 1.f);
}

// --- EXT_fragment_shading_rate (METALGLDRV_STRICT; draw.mm fsrState) ---
static bool validShadingRate(GLenum r) {
    switch (r) {
    case GL_SHADING_RATE_1X1_PIXELS_EXT: case GL_SHADING_RATE_1X2_PIXELS_EXT: case GL_SHADING_RATE_1X4_PIXELS_EXT:
    case GL_SHADING_RATE_2X1_PIXELS_EXT: case GL_SHADING_RATE_2X2_PIXELS_EXT: case GL_SHADING_RATE_2X4_PIXELS_EXT:
    case GL_SHADING_RATE_4X1_PIXELS_EXT: case GL_SHADING_RATE_4X2_PIXELS_EXT: case GL_SHADING_RATE_4X4_PIXELS_EXT: return true;
    default: return false;
    }
}

API void APIENTRY glShadingRateEXT(GLenum rate) {
    MGL_CTX();
    if (!ctx->strict()) return ctx->setError(GL_INVALID_OPERATION, "EXT_fragment_shading_rate needs METALGLDRV_STRICT=1");
    if (!validShadingRate(rate)) return ctx->setError(GL_INVALID_ENUM);
    ctx->shadingRate = rate;
}

API void APIENTRY glShadingRateCombinerOpsEXT(GLenum combinerOp0, GLenum combinerOp1) {
    MGL_CTX();
    if (!ctx->strict()) return ctx->setError(GL_INVALID_OPERATION, "EXT_fragment_shading_rate needs METALGLDRV_STRICT=1");
    auto valid = [](GLenum op) { return op >= GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT && op <= GL_FRAGMENT_SHADING_RATE_COMBINER_OP_MUL_EXT; };
    if (!valid(combinerOp0) || !valid(combinerOp1)) return ctx->setError(GL_INVALID_ENUM);
    // non-trivial combiners, the primitive and the attachment rate are all supported
    ctx->shadingRateCombiner[0] = combinerOp0;
    ctx->shadingRateCombiner[1] = combinerOp1;
}

// The supported rates: 1x1, 1x2, 2x1 and 2x2 (the required ones) at every sample count, as a coarse
// fragment must fit in a 2x2 quad (see glsl msl.cpp, mgl_fsr_setup).
API void APIENTRY glGetFragmentShadingRatesEXT(GLsizei samples, GLsizei maxCount, GLsizei* count, GLenum* shadingRates) {
    MGL_CTX();
    if (!ctx->strict()) return ctx->setError(GL_INVALID_OPERATION, "EXT_fragment_shading_rate needs METALGLDRV_STRICT=1");
    if (maxCount < 0) return ctx->setError(GL_INVALID_VALUE);
    static const GLenum kRates[] = {GL_SHADING_RATE_2X2_PIXELS_EXT, GL_SHADING_RATE_2X1_PIXELS_EXT, GL_SHADING_RATE_1X2_PIXELS_EXT,
                                    GL_SHADING_RATE_1X1_PIXELS_EXT};
    GLsizei n = 0;
    if (samples == 0 || samples == 1 || samples == 2 || samples == 4)
        for (GLenum r : kRates)
            if (n < maxCount) shadingRates[n++] = r;
    if (count) *count = n;
}

API void APIENTRY glPatchParameteri(GLenum pname, GLint value) {
    MGL_CTX();
    if (pname != GL_PATCH_VERTICES) return ctx->setError(GL_INVALID_ENUM);
    if (value <= 0 || value > 32) return ctx->setError(GL_INVALID_VALUE);
    ctx->patchVertices = value;
}

API void APIENTRY glPatchParameterfv(GLenum pname, const GLfloat* values) {
    MGL_CTX();
    if (pname == GL_PATCH_DEFAULT_OUTER_LEVEL) memcpy(ctx->patchDefaultOuter, values, 16);
    else if (pname == GL_PATCH_DEFAULT_INNER_LEVEL) memcpy(ctx->patchDefaultInner, values, 8);
    else ctx->setError(GL_INVALID_ENUM);
}

API void APIENTRY glHint(GLenum target, GLenum mode) {
    MGL_CTX();
    if (mode != GL_FASTEST && mode != GL_NICEST && mode != GL_DONT_CARE) return ctx->setError(GL_INVALID_ENUM);
    switch (target) {
    case GL_LINE_SMOOTH_HINT: ctx->hints[0] = mode; break;
    case GL_POLYGON_SMOOTH_HINT: ctx->hints[1] = mode; break;
    case GL_TEXTURE_COMPRESSION_HINT: ctx->hints[2] = mode; break;
    case GL_FRAGMENT_SHADER_DERIVATIVE_HINT: ctx->hints[3] = mode; break;
    case GL_PERSPECTIVE_CORRECTION_HINT: case GL_POINT_SMOOTH_HINT: case GL_FOG_HINT: case GL_GENERATE_MIPMAP_HINT:
        if (ctx->coreProfile) return ctx->setError(GL_INVALID_ENUM, "glHint(0x%x)", target);
        ctx->hints[4] = mode;
        break;
    default: ctx->setError(GL_INVALID_ENUM, "glHint(0x%x)", target);
    }
}

API void APIENTRY glPixelStorei(GLenum pname, GLint param) {
    MGL_CTX();
    PixelStore* ps = nullptr;
    GLenum p = pname;
    switch (pname) {
    case GL_PACK_SWAP_BYTES: case GL_PACK_LSB_FIRST: case GL_PACK_ROW_LENGTH: case GL_PACK_IMAGE_HEIGHT: case GL_PACK_SKIP_ROWS:
    case GL_PACK_SKIP_PIXELS: case GL_PACK_SKIP_IMAGES: case GL_PACK_ALIGNMENT: case GL_PACK_COMPRESSED_BLOCK_WIDTH:
    case GL_PACK_COMPRESSED_BLOCK_HEIGHT: case GL_PACK_COMPRESSED_BLOCK_DEPTH: case GL_PACK_COMPRESSED_BLOCK_SIZE:
        ps = &ctx->pack;
        break;
    default: ps = &ctx->unpack; break;
    }
    if (param < 0 && pname != GL_PACK_SWAP_BYTES && pname != GL_UNPACK_SWAP_BYTES) return ctx->setError(GL_INVALID_VALUE);
    switch (p) {
    case GL_PACK_SWAP_BYTES: case GL_UNPACK_SWAP_BYTES: ps->swapBytes = param; break;
    case GL_PACK_LSB_FIRST: case GL_UNPACK_LSB_FIRST: ps->lsbFirst = param; break;
    case GL_PACK_ROW_LENGTH: case GL_UNPACK_ROW_LENGTH: ps->rowLength = param; break;
    case GL_PACK_IMAGE_HEIGHT: case GL_UNPACK_IMAGE_HEIGHT: ps->imageHeight = param; break;
    case GL_PACK_SKIP_ROWS: case GL_UNPACK_SKIP_ROWS: ps->skipRows = param; break;
    case GL_PACK_SKIP_PIXELS: case GL_UNPACK_SKIP_PIXELS: ps->skipPixels = param; break;
    case GL_PACK_SKIP_IMAGES: case GL_UNPACK_SKIP_IMAGES: ps->skipImages = param; break;
    case GL_PACK_ALIGNMENT: case GL_UNPACK_ALIGNMENT:
        if (param != 1 && param != 2 && param != 4 && param != 8) return ctx->setError(GL_INVALID_VALUE);
        ps->alignment = param;
        break;
    case GL_PACK_COMPRESSED_BLOCK_WIDTH: case GL_UNPACK_COMPRESSED_BLOCK_WIDTH: ps->compressedBlockWidth = param; break;
    case GL_PACK_COMPRESSED_BLOCK_HEIGHT: case GL_UNPACK_COMPRESSED_BLOCK_HEIGHT: ps->compressedBlockHeight = param; break;
    case GL_PACK_COMPRESSED_BLOCK_DEPTH: case GL_UNPACK_COMPRESSED_BLOCK_DEPTH: ps->compressedBlockDepth = param; break;
    case GL_PACK_COMPRESSED_BLOCK_SIZE: case GL_UNPACK_COMPRESSED_BLOCK_SIZE: ps->compressedBlockSize = param; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glPixelStoref(GLenum pname, GLfloat param) { glPixelStorei(pname, (GLint)lroundf(param)); }

// --- clears ---
API void APIENTRY glClearColor(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {
    MGL_CTX();
    ctx->clearColor[0] = r;
    ctx->clearColor[1] = g;
    ctx->clearColor[2] = b;
    ctx->clearColor[3] = a;
}

API void APIENTRY glClearDepth(GLdouble d) {
    MGL_CTX();
    ctx->clearDepth = std::clamp(d, 0.0, 1.0);
}
API void APIENTRY glClearDepthf(GLfloat d) { glClearDepth(d); }

API void APIENTRY glClearStencil(GLint s) {
    MGL_CTX();
    ctx->clearStencil = s;
}

bool mglFramebufferComplete(Context* ctx, Framebuffer* fb, GLenum* status);  // api_fbo.mm
const char* framebufferIncompleteReason(Context* ctx, Framebuffer* fb);  // api_fbo.mm
bool mglCheckDrawConditional(Context* ctx);  // queries: conditional rendering

// Shared clear implementation. colorDB: draw buffer index (-1: all), values per call.
static void doClear(Context* ctx, bool color, int colorDB, const float* fval, const uint32_t* ival, bool isIntClear, bool depth,
                    float dval, bool stencil, int sval) {
    if (ctx->rasterizerDiscard) return;
    if (!mglCheckDrawConditional(ctx)) return;
    Framebuffer* fb = ctx->drawFb.get();
    if (!fb) return;
    GLenum status;
    if (!mglFramebufferComplete(ctx, fb, &status))
        return ctx->setError(GL_INVALID_FRAMEBUFFER_OPERATION, "%s", framebufferIncompleteReason(ctx, fb));
    PassDesc p;
    if (!ctx->resolvePass(p)) return;
    // Which draw buffers are cleared
    uint8_t dbMask = 0;
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (!color || !p.color[i].tex) continue;
        if (colorDB >= 0 && i != colorDB) continue;
        dbMask |= 1u << i;
    }
    bool doDepth = depth && p.depth.tex && ctx->depthWrite;
    bool doStencil = stencil && p.stencil.tex && (ctx->stencil[0].writeMask & 0xff) != 0;
    if (!dbMask && !doDepth && !doStencil) return;
    // Scissor (index 0)
    bool scissored = false;
    MTLScissorRect sc = {0, 0, (NSUInteger)p.width, (NSUInteger)p.height};
    if (ctx->scissorEnabled & 1) {
        int x0 = std::max(0, ctx->scissor[0][0]), y0 = std::max(0, ctx->scissor[0][1]);
        int x1 = std::min(p.width, ctx->scissor[0][0] + ctx->scissor[0][2]);
        int y1 = std::min(p.height, ctx->scissor[0][1] + ctx->scissor[0][3]);
        if (x1 <= x0 || y1 <= y0) return;
        if (x0 > 0 || y0 > 0 || x1 < p.width || y1 < p.height) {
            scissored = true;
            sc = {(NSUInteger)x0, (NSUInteger)y0, (NSUInteger)(x1 - x0), (NSUInteger)(y1 - y0)};
        }
    }
    bool fullMasks = true;
    for (int i = 0; i < kMaxDrawBuffers; i++)
        if ((dbMask & (1u << i)) && ctx->blend[i].colorMask != 0xF) fullMasks = false;
    if (doStencil && (ctx->stencil[0].writeMask & 0xff) != 0xff) fullMasks = false;
    // Load-action path: only when no draws happened yet in this pass (or no pass open).
    // Fold into the next pass's load action when nothing has been drawn into this
    // framebuffer yet (no open pass, or the open pass belongs to another framebuffer).
    // (An open pass whose attachments no longer match the framebuffer will not be reused.)
    bool canFold = !scissored && fullMasks && (!ctx->passOpen || ctx->passFb_ != fb || !(p == ctx->curPass));
    if (canFold) {
        auto& pc = ctx->pendingClear;
        for (int i = 0; i < kMaxDrawBuffers; i++) {
            if (!(dbMask & (1u << i))) continue;
            pc.colorMask |= 1u << i;
            if (isIntClear) memcpy(pc.colorInt[i], ival, 16);
            else memcpy(pc.color[i], fval, 16);
            if (!isIntClear) {
                // float clear of an integer attachment is undefined; store something sensible
                for (int k = 0; k < 4; k++) pc.colorInt[i][k] = (uint32_t)(int32_t)fval[k];
            } else {
                for (int k = 0; k < 4; k++) pc.color[i][k] = (float)(int32_t)ival[k];
            }
        }
        if (doDepth) {
            pc.depth = true;
            pc.depthValue = dval;
        }
        if (doStencil) {
            pc.stencil = true;
            pc.stencilValue = sval & 0xff;
        }
        return;
    }
    QuadClear q;
    q.drawBufferMask = dbMask;
    for (int i = 0; i < kMaxDrawBuffers; i++) {
        if (isIntClear) {
            memcpy(q.colorInt[i], ival, 16);
            for (int k = 0; k < 4; k++) q.color[i][k] = (float)(int32_t)ival[k];
        } else {
            memcpy(q.color[i], fval, 16);
            // float clear of an integer attachment is undefined; store something sensible
            for (int k = 0; k < 4; k++) q.colorInt[i][k] = (uint32_t)(int32_t)fval[k];
        }
        q.colorWriteMask[i] = ctx->blend[i].colorMask;
    }
    q.depth = doDepth;
    q.depthValue = dval;
    q.stencil = doStencil;
    q.stencilValue = sval & 0xff;
    q.stencilWriteMask = ctx->stencil[0].writeMask & 0xff;
    q.scissored = scissored;
    q.scissor = sc;
    clearWithQuad(ctx, q);
}

API void APIENTRY glClear(GLbitfield mask) {
    MGL_CTX();
    if (mask & ~(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT | (ctx->coreProfile ? 0 : GL_ACCUM_BUFFER_BIT)))
        return ctx->setError(GL_INVALID_VALUE);
    float c[4] = {ctx->clearColor[0], ctx->clearColor[1], ctx->clearColor[2], ctx->clearColor[3]};
    doClear(ctx, mask & GL_COLOR_BUFFER_BIT, -1, c, nullptr, false, mask & GL_DEPTH_BUFFER_BIT, (float)ctx->clearDepth,
            mask & GL_STENCIL_BUFFER_BIT, ctx->clearStencil);
}

API void APIENTRY glClearBufferfv(GLenum buffer, GLint drawbuffer, const GLfloat* value) {
    MGL_CTX();
    if (buffer == GL_COLOR) {
        if (drawbuffer < 0 || drawbuffer >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
        doClear(ctx, true, drawbuffer, value, nullptr, false, false, 0, false, 0);
    } else if (buffer == GL_DEPTH) {
        if (drawbuffer != 0) return ctx->setError(GL_INVALID_VALUE);
        doClear(ctx, false, -1, nullptr, nullptr, false, true, std::clamp(value[0], 0.f, 1.f), false, 0);
    } else {
        ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glClearBufferiv(GLenum buffer, GLint drawbuffer, const GLint* value) {
    MGL_CTX();
    if (buffer == GL_COLOR) {
        if (drawbuffer < 0 || drawbuffer >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
        doClear(ctx, true, drawbuffer, nullptr, (const uint32_t*)value, true, false, 0, false, 0);
    } else if (buffer == GL_STENCIL) {
        if (drawbuffer != 0) return ctx->setError(GL_INVALID_VALUE);
        doClear(ctx, false, -1, nullptr, nullptr, false, false, 0, true, value[0]);
    } else {
        ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glClearBufferuiv(GLenum buffer, GLint drawbuffer, const GLuint* value) {
    MGL_CTX();
    if (buffer != GL_COLOR) return ctx->setError(GL_INVALID_ENUM);
    if (drawbuffer < 0 || drawbuffer >= kMaxDrawBuffers) return ctx->setError(GL_INVALID_VALUE);
    doClear(ctx, true, drawbuffer, nullptr, value, true, false, 0, false, 0);
}

API void APIENTRY glClearBufferfi(GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
    MGL_CTX();
    if (buffer != GL_DEPTH_STENCIL) return ctx->setError(GL_INVALID_ENUM);
    if (drawbuffer != 0) return ctx->setError(GL_INVALID_VALUE);
    doClear(ctx, false, -1, nullptr, nullptr, false, true, std::clamp(depth, 0.f, 1.f), true, stencil);
}

API void APIENTRY glClearNamedFramebufferfv(GLuint fbn, GLenum buffer, GLint drawbuffer, const GLfloat* value) {
    MGL_CTX();
    Ref<Framebuffer> saved = ctx->drawFb;
    Framebuffer* fb = fbn ? ctx->framebuffers.get(fbn) : ctx->defaultFramebuffer();
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    ctx->drawFb = fb;
    ctx->passDirty = true;
    glClearBufferfv(buffer, drawbuffer, value);
    ctx->drawFb = saved;
    ctx->passDirty = true;
}
API void APIENTRY glClearNamedFramebufferiv(GLuint fbn, GLenum buffer, GLint drawbuffer, const GLint* value) {
    MGL_CTX();
    Ref<Framebuffer> saved = ctx->drawFb;
    Framebuffer* fb = fbn ? ctx->framebuffers.get(fbn) : ctx->defaultFramebuffer();
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    ctx->drawFb = fb;
    ctx->passDirty = true;
    glClearBufferiv(buffer, drawbuffer, value);
    ctx->drawFb = saved;
    ctx->passDirty = true;
}
API void APIENTRY glClearNamedFramebufferuiv(GLuint fbn, GLenum buffer, GLint drawbuffer, const GLuint* value) {
    MGL_CTX();
    Ref<Framebuffer> saved = ctx->drawFb;
    Framebuffer* fb = fbn ? ctx->framebuffers.get(fbn) : ctx->defaultFramebuffer();
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    ctx->drawFb = fb;
    ctx->passDirty = true;
    glClearBufferuiv(buffer, drawbuffer, value);
    ctx->drawFb = saved;
    ctx->passDirty = true;
}
API void APIENTRY glClearNamedFramebufferfi(GLuint fbn, GLenum buffer, GLint drawbuffer, GLfloat depth, GLint stencil) {
    MGL_CTX();
    Ref<Framebuffer> saved = ctx->drawFb;
    Framebuffer* fb = fbn ? ctx->framebuffers.get(fbn) : ctx->defaultFramebuffer();
    if (!fb) return ctx->setError(GL_INVALID_OPERATION);
    ctx->drawFb = fb;
    ctx->passDirty = true;
    glClearBufferfi(buffer, drawbuffer, depth, stencil);
    ctx->drawFb = saved;
    ctx->passDirty = true;
}

// --- flush / finish / errors ---
API void APIENTRY glFlush(void) {
    MGL_CTX();
    ctx->applyClearNow();
    ctx->flush();
}

API void APIENTRY glFinish(void) {
    MGL_CTX();
    ctx->applyClearNow();
    ctx->finish();
}

API GLenum APIENTRY glGetError(void) {
    MGL_CTX_RET(GL_NO_ERROR);
    GLenum e = ctx->error;
    ctx->error = GL_NO_ERROR;
    return e;
}

API GLenum APIENTRY glGetGraphicsResetStatus(void) { return GL_NO_ERROR; }
API GLenum APIENTRY glGetGraphicsResetStatusARB(void) { return GL_NO_ERROR; }

API void APIENTRY glClampColor(GLenum target, GLenum clamp) {
    MGL_CTX();
    if (clamp != GL_TRUE && clamp != GL_FALSE && clamp != GL_FIXED_ONLY) return ctx->setError(GL_INVALID_ENUM);
    // Only CLAMP_READ_COLOR survives in core; vertex/fragment color clamping are compatibility-only no-ops here.
    if (target == GL_CLAMP_READ_COLOR) ctx->clampReadColor = clamp;
    else if (ctx->coreProfile || (target != GL_CLAMP_VERTEX_COLOR && target != GL_CLAMP_FRAGMENT_COLOR)) ctx->setError(GL_INVALID_ENUM);
}
