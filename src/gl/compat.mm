// Compatibility profile: fixed-function pipeline emulation.
//
// Legacy state (matrix stacks, lighting, fog, texenv, client arrays, immediate
// mode, display lists) lives in CompatState. Draws without a program use a
// generated GLSL "fixed-function" program compiled by our own compiler; any
// compatibility program (generated or user) gets the gl_* state uniforms
// written into its default uniform block before each draw.
#import "gl/context.h"
#include <cmath>
#include <functional>
#include <map>
#include <unordered_map>
#include "gl/gltypes.h"
#include "gl/pipeline.h"
#include "util/log.h"

using namespace mgl;
#define API extern "C" __attribute__((visibility("default")))

namespace mgl {
void setupLinkedProgram(ProgramObj* p);
}

namespace {

// ---------------------------------------------------------------------------
// 4x4 column-major matrices
// ---------------------------------------------------------------------------
struct Mat4 {
    float m[16];
    static Mat4 identity() {
        Mat4 r{};
        r.m[0] = r.m[5] = r.m[10] = r.m[15] = 1;
        return r;
    }
    float& at(int row, int col) { return m[col * 4 + row]; }
    float at(int row, int col) const { return m[col * 4 + row]; }
};

Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 r{};
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) {
            float s = 0;
            for (int k = 0; k < 4; k++) s += a.at(rr, k) * b.at(k, c);
            r.at(rr, c) = s;
        }
    return r;
}

Mat4 transpose(const Mat4& a) {
    Mat4 r;
    for (int c = 0; c < 4; c++)
        for (int rr = 0; rr < 4; rr++) r.at(rr, c) = a.at(c, rr);
    return r;
}

Mat4 inverse(const Mat4& a) {
    const float* m = a.m;
    float inv[16];
    inv[0] = m[5] * m[10] * m[15] - m[5] * m[11] * m[14] - m[9] * m[6] * m[15] + m[9] * m[7] * m[14] + m[13] * m[6] * m[11] - m[13] * m[7] * m[10];
    inv[4] = -m[4] * m[10] * m[15] + m[4] * m[11] * m[14] + m[8] * m[6] * m[15] - m[8] * m[7] * m[14] - m[12] * m[6] * m[11] + m[12] * m[7] * m[10];
    inv[8] = m[4] * m[9] * m[15] - m[4] * m[11] * m[13] - m[8] * m[5] * m[15] + m[8] * m[7] * m[13] + m[12] * m[5] * m[11] - m[12] * m[7] * m[9];
    inv[12] = -m[4] * m[9] * m[14] + m[4] * m[10] * m[13] + m[8] * m[5] * m[14] - m[8] * m[6] * m[13] - m[12] * m[5] * m[10] + m[12] * m[6] * m[9];
    inv[1] = -m[1] * m[10] * m[15] + m[1] * m[11] * m[14] + m[9] * m[2] * m[15] - m[9] * m[3] * m[14] - m[13] * m[2] * m[11] + m[13] * m[3] * m[10];
    inv[5] = m[0] * m[10] * m[15] - m[0] * m[11] * m[14] - m[8] * m[2] * m[15] + m[8] * m[3] * m[14] + m[12] * m[2] * m[11] - m[12] * m[3] * m[10];
    inv[9] = -m[0] * m[9] * m[15] + m[0] * m[11] * m[13] + m[8] * m[1] * m[15] - m[8] * m[3] * m[13] - m[12] * m[1] * m[11] + m[12] * m[3] * m[9];
    inv[13] = m[0] * m[9] * m[14] - m[0] * m[10] * m[13] - m[8] * m[1] * m[14] + m[8] * m[2] * m[13] + m[12] * m[1] * m[10] - m[12] * m[2] * m[9];
    inv[2] = m[1] * m[6] * m[15] - m[1] * m[7] * m[14] - m[5] * m[2] * m[15] + m[5] * m[3] * m[14] + m[13] * m[2] * m[7] - m[13] * m[3] * m[6];
    inv[6] = -m[0] * m[6] * m[15] + m[0] * m[7] * m[14] + m[4] * m[2] * m[15] - m[4] * m[3] * m[14] - m[12] * m[2] * m[7] + m[12] * m[3] * m[6];
    inv[10] = m[0] * m[5] * m[15] - m[0] * m[7] * m[13] - m[4] * m[1] * m[15] + m[4] * m[3] * m[13] + m[12] * m[1] * m[7] - m[12] * m[3] * m[5];
    inv[14] = -m[0] * m[5] * m[14] + m[0] * m[6] * m[13] + m[4] * m[1] * m[14] - m[4] * m[2] * m[13] - m[12] * m[1] * m[6] + m[12] * m[2] * m[5];
    inv[3] = -m[1] * m[6] * m[11] + m[1] * m[7] * m[10] + m[5] * m[2] * m[11] - m[5] * m[3] * m[10] - m[9] * m[2] * m[7] + m[9] * m[3] * m[6];
    inv[7] = m[0] * m[6] * m[11] - m[0] * m[7] * m[10] - m[4] * m[2] * m[11] + m[4] * m[3] * m[10] + m[8] * m[2] * m[7] - m[8] * m[3] * m[6];
    inv[11] = -m[0] * m[5] * m[11] + m[0] * m[7] * m[9] + m[4] * m[1] * m[11] - m[4] * m[3] * m[9] - m[8] * m[1] * m[7] + m[8] * m[3] * m[5];
    inv[15] = m[0] * m[5] * m[10] - m[0] * m[6] * m[9] - m[4] * m[1] * m[10] + m[4] * m[2] * m[9] + m[8] * m[1] * m[6] - m[8] * m[2] * m[5];
    float det = m[0] * inv[0] + m[1] * inv[4] + m[2] * inv[8] + m[3] * inv[12];
    Mat4 r;
    if (det == 0) return Mat4::identity();
    for (int i = 0; i < 16; i++) r.m[i] = inv[i] / det;
    return r;
}

Mat4 fromRows(const float* rows) {  // transposed load
    Mat4 r;
    for (int i = 0; i < 16; i++) r.m[i] = rows[i];
    return transpose(r);
}

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
struct LightState {
    bool enabled = false;
    float ambient[4] = {0, 0, 0, 1}, diffuse[4] = {0, 0, 0, 1}, specular[4] = {0, 0, 0, 1};
    float position[4] = {0, 0, 1, 0};  // eye space
    float spotDirection[3] = {0, 0, -1};  // eye space
    float spotExponent = 0, spotCutoff = 180;
    float constantAtt = 1, linearAtt = 0, quadraticAtt = 0;
};

struct MaterialState {
    float ambient[4] = {0.2f, 0.2f, 0.2f, 1}, diffuse[4] = {0.8f, 0.8f, 0.8f, 1}, specular[4] = {0, 0, 0, 1};
    float emission[4] = {0, 0, 0, 1};
    float shininess = 0;
};

struct TexEnv {
    GLenum mode = GL_MODULATE;
    float color[4] = {0, 0, 0, 0};
    GLenum combineRGB = GL_MODULATE, combineA = GL_MODULATE;
    GLenum srcRGB[3] = {GL_TEXTURE, GL_PREVIOUS, GL_CONSTANT}, srcA[3] = {GL_TEXTURE, GL_PREVIOUS, GL_CONSTANT};
    GLenum opRGB[3] = {GL_SRC_COLOR, GL_SRC_COLOR, GL_SRC_ALPHA}, opA[3] = {GL_SRC_ALPHA, GL_SRC_ALPHA, GL_SRC_ALPHA};
    float scaleRGB = 1, scaleA = 1;
    float lodBias = 0;
};

struct TexGen {
    bool enabled[4] = {};
    GLenum mode[4] = {GL_EYE_LINEAR, GL_EYE_LINEAR, GL_EYE_LINEAR, GL_EYE_LINEAR};
    float eyePlane[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    float objPlane[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
};

struct ClientArray {
    bool enabled = false;
    GLint size = 4;
    GLenum type = GL_FLOAT;
    GLsizei stride = 0;
    const void* pointer = nullptr;
    Ref<Buffer> buffer;
};

enum CompatAttr { CA_VERTEX, CA_NORMAL, CA_COLOR, CA_SECONDARY, CA_FOG, CA_TEX0, CA_COUNT = CA_TEX0 + 8 };
const char* kAttrNames[CA_COUNT] = {"gl_Vertex", "gl_Normal", "gl_Color", "gl_SecondaryColor", "gl_FogCoord",
                                    "gl_MultiTexCoord0", "gl_MultiTexCoord1", "gl_MultiTexCoord2", "gl_MultiTexCoord3",
                                    "gl_MultiTexCoord4", "gl_MultiTexCoord5", "gl_MultiTexCoord6", "gl_MultiTexCoord7"};

struct ImmVertex {
    float v[CA_COUNT][4];
};

using ListCmd = std::function<void()>;

}  // namespace

struct mgl::CompatState {
    GLenum matrixMode = GL_MODELVIEW;
    std::vector<Mat4> modelview{Mat4::identity()}, projection{Mat4::identity()}, color{Mat4::identity()};
    std::vector<Mat4> texture[8];
    uint64_t matrixVersion = 1;

    float current[CA_COUNT][4];
    bool lighting = false, colorMaterial = false, normalize = false, rescaleNormal = false, fog = false, alphaTest = false;
    bool pointSprite = false, colorSum = false;
    GLenum colorMaterialFace = GL_FRONT_AND_BACK, colorMaterialMode = GL_AMBIENT_AND_DIFFUSE;
    LightState lights[8];
    MaterialState material[2];
    float lightModelAmbient[4] = {0.2f, 0.2f, 0.2f, 1};
    bool localViewer = false, twoSide = false;
    GLenum colorControl = GL_SINGLE_COLOR;
    GLenum shadeModel = GL_SMOOTH;
    GLenum fogMode = GL_EXP, fogCoordSrc = GL_FRAGMENT_DEPTH;
    float fogDensity = 1, fogStart = 0, fogEnd = 1, fogColor[4] = {0, 0, 0, 0}, fogIndex = 0;
    GLenum alphaFunc = GL_ALWAYS;
    float alphaRef = 0;
    uint32_t texEnabled[8] = {};  // bit per target: 1D=1, 2D=2, 3D=4, CUBE=8, RECT=16
    TexEnv texEnv[8];
    TexGen texGen[8];
    float clipPlanes[8][4] = {};
    float pointSizeMin = 0, pointSizeMax = 511, pointAtt[3] = {1, 0, 0};

    ClientArray arrays[CA_COUNT];
    GLuint clientActiveTexture = 0;

    // immediate mode
    bool inBegin = false;
    GLenum beginMode = 0;
    std::vector<ImmVertex> imm;

    // display lists
    std::map<GLuint, std::vector<ListCmd>> lists;
    GLuint compilingList = 0;
    GLenum listMode = 0;
    std::vector<ListCmd> listBuf;
    GLuint listBase = 0;
    int callDepth = 0;

    // attribute stack (coarse)
    struct Saved {
        GLbitfield mask;
        Context* snapshot;  // shallow copies of POD state
    };
    std::vector<std::pair<GLbitfield, std::vector<uint8_t>>> attribStack;

    // generated fixed-function programs
    std::unordered_map<std::string, Ref<ProgramObj>> ffPrograms;
    // cached uniform indices per program: name -> index
    std::unordered_map<ProgramObj*, std::pair<uint32_t, std::unordered_map<std::string, int>>> uniformIdx;

    CompatState() {
        for (auto& t : texture) t.push_back(Mat4::identity());
        for (int i = 0; i < CA_COUNT; i++) {
            current[i][0] = current[i][1] = current[i][2] = 0;
            current[i][3] = 1;
        }
        current[CA_NORMAL][2] = 1;
        current[CA_COLOR][0] = current[CA_COLOR][1] = current[CA_COLOR][2] = 1;
        lights[0].diffuse[0] = lights[0].diffuse[1] = lights[0].diffuse[2] = 1;
        lights[0].specular[0] = lights[0].specular[1] = lights[0].specular[2] = 1;
    }
    std::vector<Mat4>& stack() {
        switch (matrixMode) {
        case GL_PROJECTION: return projection;
        case GL_TEXTURE: return texture[std::min<int>(activeUnitForMatrix, 7)];
        case GL_COLOR: return color;
        default: return modelview;
        }
    }
    int activeUnitForMatrix = 0;
};

static CompatState* compat(Context* ctx) {
    if (!ctx->compat) ctx->compat = new CompatState();
    ctx->compat->activeUnitForMatrix = ctx->activeTexture;
    return ctx->compat;
}

// Display-list recording: returns true if the command was only recorded.
#define RECORD(...)                                                             \
    do {                                                                        \
        CompatState* cs_ = compat(ctx);                                         \
        if (cs_->compilingList && cs_->callDepth == 0) {                        \
            cs_->listBuf.push_back([=]() { __VA_ARGS__; });                     \
            if (cs_->listMode == GL_COMPILE) return;                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// capabilities / queries (called from api_state / api_get)
// ---------------------------------------------------------------------------
static int texTargetBit(GLenum cap) {
    switch (cap) {
    case GL_TEXTURE_1D: return 1;
    case GL_TEXTURE_2D: return 2;
    case GL_TEXTURE_3D: return 4;
    case GL_TEXTURE_CUBE_MAP: return 8;
    case GL_TEXTURE_RECTANGLE: return 16;
    default: return 0;
    }
}

bool mglCompatCap(Context* ctx, GLenum cap, bool v) {
    CompatState* cs = compat(ctx);
    if (int bit = texTargetBit(cap)) {
        uint32_t& e = cs->texEnabled[std::min<int>(ctx->activeTexture, 7)];
        e = v ? (e | bit) : (e & ~bit);
        return true;
    }
    if (cap >= GL_LIGHT0 && cap < GL_LIGHT0 + 8) {
        cs->lights[cap - GL_LIGHT0].enabled = v;
        return true;
    }
    switch (cap) {
    case GL_LIGHTING: cs->lighting = v; return true;
    case GL_COLOR_MATERIAL: cs->colorMaterial = v; return true;
    case GL_NORMALIZE: cs->normalize = v; return true;
    case GL_RESCALE_NORMAL: cs->rescaleNormal = v; return true;
    case GL_FOG: cs->fog = v; return true;
    case GL_ALPHA_TEST: cs->alphaTest = v; return true;
    case GL_POINT_SPRITE: cs->pointSprite = v; return true;
    case GL_COLOR_SUM: cs->colorSum = v; return true;
    case GL_TEXTURE_GEN_S: case GL_TEXTURE_GEN_T: case GL_TEXTURE_GEN_R: case GL_TEXTURE_GEN_Q:
        cs->texGen[std::min<int>(ctx->activeTexture, 7)].enabled[cap - GL_TEXTURE_GEN_S] = v;
        return true;
    case GL_POINT_SMOOTH: case GL_LINE_STIPPLE: case GL_POLYGON_STIPPLE: case GL_AUTO_NORMAL: case GL_MAP1_VERTEX_3:
    case GL_MAP2_VERTEX_3: case GL_INDEX_LOGIC_OP: case GL_VERTEX_PROGRAM_TWO_SIDE:
        return true;  // accepted, no effect
    default: return false;
    }
}

bool mglCompatIsEnabled(Context* ctx, GLenum cap, bool* out) {
    CompatState* cs = compat(ctx);
    if (int bit = texTargetBit(cap)) {
        *out = cs->texEnabled[std::min<int>(ctx->activeTexture, 7)] & bit;
        return true;
    }
    if (cap >= GL_LIGHT0 && cap < GL_LIGHT0 + 8) {
        *out = cs->lights[cap - GL_LIGHT0].enabled;
        return true;
    }
    switch (cap) {
    case GL_LIGHTING: *out = cs->lighting; return true;
    case GL_COLOR_MATERIAL: *out = cs->colorMaterial; return true;
    case GL_NORMALIZE: *out = cs->normalize; return true;
    case GL_RESCALE_NORMAL: *out = cs->rescaleNormal; return true;
    case GL_FOG: *out = cs->fog; return true;
    case GL_ALPHA_TEST: *out = cs->alphaTest; return true;
    case GL_POINT_SPRITE: *out = cs->pointSprite; return true;
    case GL_COLOR_SUM: *out = cs->colorSum; return true;
    case GL_VERTEX_ARRAY: *out = cs->arrays[CA_VERTEX].enabled; return true;
    case GL_NORMAL_ARRAY: *out = cs->arrays[CA_NORMAL].enabled; return true;
    case GL_COLOR_ARRAY: *out = cs->arrays[CA_COLOR].enabled; return true;
    case GL_SECONDARY_COLOR_ARRAY: *out = cs->arrays[CA_SECONDARY].enabled; return true;
    case GL_FOG_COORD_ARRAY: *out = cs->arrays[CA_FOG].enabled; return true;
    case GL_TEXTURE_COORD_ARRAY: *out = cs->arrays[CA_TEX0 + std::min<int>(cs->clientActiveTexture, 7)].enabled; return true;
    case GL_TEXTURE_GEN_S: case GL_TEXTURE_GEN_T: case GL_TEXTURE_GEN_R: case GL_TEXTURE_GEN_Q:
        *out = cs->texGen[std::min<int>(ctx->activeTexture, 7)].enabled[cap - GL_TEXTURE_GEN_S];
        return true;
    default: return false;
    }
}

bool mglCompatGet(Context* ctx, GLenum pname, double* out, int* n, bool* isFloat) {
    CompatState* cs = compat(ctx);
    auto mat = [&](const Mat4& m) {
        for (int i = 0; i < 16; i++) out[i] = m.m[i];
        *n = 16;
        *isFloat = true;
        return true;
    };
    auto vec = [&](const float* v, int k) {
        for (int i = 0; i < k; i++) out[i] = v[i];
        *n = k;
        *isFloat = true;
        return true;
    };
    auto one = [&](double v, bool f) {
        out[0] = v;
        *n = 1;
        *isFloat = f;
        return true;
    };
    int unit = std::min<int>(ctx->activeTexture, 7);
    switch (pname) {
    case GL_MODELVIEW_MATRIX: return mat(cs->modelview.back());
    case GL_PROJECTION_MATRIX: return mat(cs->projection.back());
    case GL_TEXTURE_MATRIX: return mat(cs->texture[unit].back());
    case GL_COLOR_MATRIX: return mat(cs->color.back());
    case GL_TRANSPOSE_MODELVIEW_MATRIX: return mat(transpose(cs->modelview.back()));
    case GL_TRANSPOSE_PROJECTION_MATRIX: return mat(transpose(cs->projection.back()));
    case GL_TRANSPOSE_TEXTURE_MATRIX: return mat(transpose(cs->texture[unit].back()));
    case GL_MATRIX_MODE: return one(cs->matrixMode, false);
    case GL_MODELVIEW_STACK_DEPTH: return one((double)cs->modelview.size(), false);
    case GL_PROJECTION_STACK_DEPTH: return one((double)cs->projection.size(), false);
    case GL_TEXTURE_STACK_DEPTH: return one((double)cs->texture[unit].size(), false);
    case GL_MAX_MODELVIEW_STACK_DEPTH: return one(32, false);
    case GL_MAX_PROJECTION_STACK_DEPTH: case GL_MAX_TEXTURE_STACK_DEPTH: case GL_MAX_COLOR_MATRIX_STACK_DEPTH: return one(4, false);
    case GL_MAX_ATTRIB_STACK_DEPTH: case GL_MAX_CLIENT_ATTRIB_STACK_DEPTH: return one(16, false);
    case GL_MAX_LIGHTS: case GL_MAX_CLIP_PLANES: return one(8, false);
    case GL_MAX_LIST_NESTING: return one(64, false);
    case GL_CURRENT_COLOR: return vec(cs->current[CA_COLOR], 4);
    case GL_CURRENT_SECONDARY_COLOR: return vec(cs->current[CA_SECONDARY], 4);
    case GL_CURRENT_NORMAL: return vec(cs->current[CA_NORMAL], 3);
    case GL_CURRENT_TEXTURE_COORDS: return vec(cs->current[CA_TEX0 + unit], 4);
    case GL_CURRENT_FOG_COORD: return one(cs->current[CA_FOG][0], true);
    case GL_SHADE_MODEL: return one(cs->shadeModel, false);
    case GL_ALPHA_TEST_FUNC: return one(cs->alphaFunc, false);
    case GL_ALPHA_TEST_REF: return one(cs->alphaRef, true);
    case GL_FOG_MODE: return one(cs->fogMode, false);
    case GL_FOG_DENSITY: return one(cs->fogDensity, true);
    case GL_FOG_START: return one(cs->fogStart, true);
    case GL_FOG_END: return one(cs->fogEnd, true);
    case GL_FOG_COLOR: return vec(cs->fogColor, 4);
    case GL_FOG_COORD_SRC: return one(cs->fogCoordSrc, false);
    case GL_LIGHT_MODEL_AMBIENT: return vec(cs->lightModelAmbient, 4);
    case GL_LIGHT_MODEL_TWO_SIDE: return one(cs->twoSide, false);
    case GL_LIGHT_MODEL_LOCAL_VIEWER: return one(cs->localViewer, false);
    case GL_LIGHT_MODEL_COLOR_CONTROL: return one(cs->colorControl, false);
    case GL_COLOR_MATERIAL_FACE: return one(cs->colorMaterialFace, false);
    case GL_COLOR_MATERIAL_PARAMETER: return one(cs->colorMaterialMode, false);
    case GL_CLIENT_ACTIVE_TEXTURE: return one(GL_TEXTURE0 + cs->clientActiveTexture, false);
    case GL_LIST_BASE: return one(cs->listBase, false);
    case GL_LIST_INDEX: return one(cs->compilingList, false);
    case GL_LIST_MODE: return one(cs->listMode, false);
    case GL_ATTRIB_STACK_DEPTH: return one((double)cs->attribStack.size(), false);
    case GL_VERTEX_ARRAY_SIZE: return one(cs->arrays[CA_VERTEX].size, false);
    case GL_VERTEX_ARRAY_TYPE: return one(cs->arrays[CA_VERTEX].type, false);
    case GL_VERTEX_ARRAY_STRIDE: return one(cs->arrays[CA_VERTEX].stride, false);
    case GL_COLOR_ARRAY_SIZE: return one(cs->arrays[CA_COLOR].size, false);
    case GL_TEXTURE_COORD_ARRAY_SIZE: return one(cs->arrays[CA_TEX0 + cs->clientActiveTexture].size, false);
    case GL_RED_BITS: case GL_GREEN_BITS: case GL_BLUE_BITS: case GL_ALPHA_BITS: return one(8, false);
    case GL_DEPTH_BITS: return one(24, false);
    case GL_STENCIL_BITS: return one(8, false);
    case GL_MAX_TEXTURE_UNITS: return one(8, false);
    case GL_AUX_BUFFERS: case GL_ACCUM_RED_BITS: case GL_ACCUM_GREEN_BITS: case GL_ACCUM_BLUE_BITS: case GL_ACCUM_ALPHA_BITS:
    case GL_INDEX_BITS: return one(0, false);
    case GL_RGBA_MODE: return one(1, false);
    default: return false;
    }
}

// ---------------------------------------------------------------------------
// matrices
// ---------------------------------------------------------------------------
static void multMatrix(Context* ctx, const Mat4& m) {
    CompatState* cs = compat(ctx);
    auto& s = cs->stack();
    s.back() = mul(s.back(), m);
    cs->matrixVersion++;
}

API void APIENTRY glMatrixMode(GLenum mode) {
    MGL_CTX();
    RECORD(glMatrixMode(mode));
    if (mode != GL_MODELVIEW && mode != GL_PROJECTION && mode != GL_TEXTURE && mode != GL_COLOR) return ctx->setError(GL_INVALID_ENUM);
    compat(ctx)->matrixMode = mode;
}

API void APIENTRY glPushMatrix(void) {
    MGL_CTX();
    RECORD(glPushMatrix());
    CompatState* cs = compat(ctx);
    auto& s = cs->stack();
    size_t max = cs->matrixMode == GL_MODELVIEW ? 32 : 4;
    if (s.size() >= max) return ctx->setError(GL_STACK_OVERFLOW);
    s.push_back(s.back());
}

API void APIENTRY glPopMatrix(void) {
    MGL_CTX();
    RECORD(glPopMatrix());
    CompatState* cs = compat(ctx);
    auto& s = cs->stack();
    if (s.size() <= 1) return ctx->setError(GL_STACK_UNDERFLOW);
    s.pop_back();
    cs->matrixVersion++;
}

API void APIENTRY glLoadIdentity(void) {
    MGL_CTX();
    RECORD(glLoadIdentity());
    CompatState* cs = compat(ctx);
    cs->stack().back() = Mat4::identity();
    cs->matrixVersion++;
}

API void APIENTRY glLoadMatrixf(const GLfloat* m) {
    MGL_CTX();
    Mat4 mm;
    memcpy(mm.m, m, 64);
    RECORD(glLoadMatrixf(mm.m));
    CompatState* cs = compat(ctx);
    cs->stack().back() = mm;
    cs->matrixVersion++;
}

API void APIENTRY glLoadMatrixd(const GLdouble* m) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)m[i];
    glLoadMatrixf(f);
}

API void APIENTRY glMultMatrixf(const GLfloat* m) {
    MGL_CTX();
    Mat4 mm;
    memcpy(mm.m, m, 64);
    RECORD(multMatrix(ctx, mm));
    multMatrix(ctx, mm);
}

API void APIENTRY glMultMatrixd(const GLdouble* m) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)m[i];
    glMultMatrixf(f);
}

API void APIENTRY glLoadTransposeMatrixf(const GLfloat* m) {
    Mat4 t = fromRows(m);
    glLoadMatrixf(t.m);
}
API void APIENTRY glLoadTransposeMatrixd(const GLdouble* m) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)m[i];
    glLoadTransposeMatrixf(f);
}
API void APIENTRY glMultTransposeMatrixf(const GLfloat* m) {
    Mat4 t = fromRows(m);
    glMultMatrixf(t.m);
}
API void APIENTRY glMultTransposeMatrixd(const GLdouble* m) {
    float f[16];
    for (int i = 0; i < 16; i++) f[i] = (float)m[i];
    glMultTransposeMatrixf(f);
}

API void APIENTRY glTranslatef(GLfloat x, GLfloat y, GLfloat z) {
    MGL_CTX();
    Mat4 m = Mat4::identity();
    m.m[12] = x;
    m.m[13] = y;
    m.m[14] = z;
    RECORD(multMatrix(ctx, m));
    multMatrix(ctx, m);
}
API void APIENTRY glTranslated(GLdouble x, GLdouble y, GLdouble z) { glTranslatef((float)x, (float)y, (float)z); }

API void APIENTRY glScalef(GLfloat x, GLfloat y, GLfloat z) {
    MGL_CTX();
    Mat4 m = Mat4::identity();
    m.m[0] = x;
    m.m[5] = y;
    m.m[10] = z;
    RECORD(multMatrix(ctx, m));
    multMatrix(ctx, m);
}
API void APIENTRY glScaled(GLdouble x, GLdouble y, GLdouble z) { glScalef((float)x, (float)y, (float)z); }

API void APIENTRY glRotatef(GLfloat angle, GLfloat x, GLfloat y, GLfloat z) {
    MGL_CTX();
    float len = sqrtf(x * x + y * y + z * z);
    Mat4 m = Mat4::identity();
    if (len > 0) {
        x /= len;
        y /= len;
        z /= len;
        float a = angle * (float)M_PI / 180.0f, c = cosf(a), s = sinf(a), t = 1 - c;
        m.at(0, 0) = x * x * t + c;
        m.at(0, 1) = x * y * t - z * s;
        m.at(0, 2) = x * z * t + y * s;
        m.at(1, 0) = y * x * t + z * s;
        m.at(1, 1) = y * y * t + c;
        m.at(1, 2) = y * z * t - x * s;
        m.at(2, 0) = x * z * t - y * s;
        m.at(2, 1) = y * z * t + x * s;
        m.at(2, 2) = z * z * t + c;
    }
    RECORD(multMatrix(ctx, m));
    multMatrix(ctx, m);
}
API void APIENTRY glRotated(GLdouble a, GLdouble x, GLdouble y, GLdouble z) { glRotatef((float)a, (float)x, (float)y, (float)z); }

API void APIENTRY glOrtho(GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f) {
    MGL_CTX();
    if (l == r || b == t || n == f) return ctx->setError(GL_INVALID_VALUE);
    Mat4 m = Mat4::identity();
    m.at(0, 0) = (float)(2 / (r - l));
    m.at(1, 1) = (float)(2 / (t - b));
    m.at(2, 2) = (float)(-2 / (f - n));
    m.at(0, 3) = (float)(-(r + l) / (r - l));
    m.at(1, 3) = (float)(-(t + b) / (t - b));
    m.at(2, 3) = (float)(-(f + n) / (f - n));
    RECORD(multMatrix(ctx, m));
    multMatrix(ctx, m);
}
API void APIENTRY glOrthof(GLfloat l, GLfloat r, GLfloat b, GLfloat t, GLfloat n, GLfloat f) { glOrtho(l, r, b, t, n, f); }

API void APIENTRY glFrustum(GLdouble l, GLdouble r, GLdouble b, GLdouble t, GLdouble n, GLdouble f) {
    MGL_CTX();
    if (n <= 0 || f <= 0 || l == r || b == t || n == f) return ctx->setError(GL_INVALID_VALUE);
    Mat4 m{};
    m.at(0, 0) = (float)(2 * n / (r - l));
    m.at(1, 1) = (float)(2 * n / (t - b));
    m.at(0, 2) = (float)((r + l) / (r - l));
    m.at(1, 2) = (float)((t + b) / (t - b));
    m.at(2, 2) = (float)(-(f + n) / (f - n));
    m.at(3, 2) = -1;
    m.at(2, 3) = (float)(-2 * f * n / (f - n));
    RECORD(multMatrix(ctx, m));
    multMatrix(ctx, m);
}
API void APIENTRY glFrustumf(GLfloat l, GLfloat r, GLfloat b, GLfloat t, GLfloat n, GLfloat f) { glFrustum(l, r, b, t, n, f); }

// ---------------------------------------------------------------------------
// current values & immediate mode
// ---------------------------------------------------------------------------
static void emitVertex(Context* ctx, float x, float y, float z, float w);

static void setCurrent(Context* ctx, int attr, float x, float y, float z, float w) {
    CompatState* cs = compat(ctx);
    float* c = cs->current[attr];
    c[0] = x;
    c[1] = y;
    c[2] = z;
    c[3] = w;
}

#define CUR(attr, name, params, x, y, z, w)          \
    API void APIENTRY name params {                    \
        MGL_CTX();                                     \
        RECORD(setCurrent(ctx, attr, x, y, z, w));     \
        setCurrent(ctx, attr, x, y, z, w);             \
    }
static inline float ub(GLubyte v) { return v / 255.0f; }
static inline float us(GLushort v) { return v / 65535.0f; }
static inline float bn(GLbyte v) { return std::max(-1.0f, v / 127.0f); }
static inline float sn(GLshort v) { return std::max(-1.0f, v / 32767.0f); }
static inline float ui(GLuint v) { return (float)(v / 4294967295.0); }
static inline float in_(GLint v) { return std::max(-1.0f, (float)(v / 2147483647.0)); }

CUR(CA_COLOR, glColor3f, (GLfloat r, GLfloat g, GLfloat b), r, g, b, 1)
CUR(CA_COLOR, glColor4f, (GLfloat r, GLfloat g, GLfloat b, GLfloat a), r, g, b, a)
CUR(CA_COLOR, glColor3d, (GLdouble r, GLdouble g, GLdouble b), (float)r, (float)g, (float)b, 1)
CUR(CA_COLOR, glColor4d, (GLdouble r, GLdouble g, GLdouble b, GLdouble a), (float)r, (float)g, (float)b, (float)a)
CUR(CA_COLOR, glColor3ub, (GLubyte r, GLubyte g, GLubyte b), ub(r), ub(g), ub(b), 1)
CUR(CA_COLOR, glColor4ub, (GLubyte r, GLubyte g, GLubyte b, GLubyte a), ub(r), ub(g), ub(b), ub(a))
CUR(CA_COLOR, glColor3b, (GLbyte r, GLbyte g, GLbyte b), bn(r), bn(g), bn(b), 1)
CUR(CA_COLOR, glColor4b, (GLbyte r, GLbyte g, GLbyte b, GLbyte a), bn(r), bn(g), bn(b), bn(a))
CUR(CA_COLOR, glColor3s, (GLshort r, GLshort g, GLshort b), sn(r), sn(g), sn(b), 1)
CUR(CA_COLOR, glColor4s, (GLshort r, GLshort g, GLshort b, GLshort a), sn(r), sn(g), sn(b), sn(a))
CUR(CA_COLOR, glColor3i, (GLint r, GLint g, GLint b), in_(r), in_(g), in_(b), 1)
CUR(CA_COLOR, glColor4i, (GLint r, GLint g, GLint b, GLint a), in_(r), in_(g), in_(b), in_(a))
CUR(CA_COLOR, glColor3us, (GLushort r, GLushort g, GLushort b), us(r), us(g), us(b), 1)
CUR(CA_COLOR, glColor4us, (GLushort r, GLushort g, GLushort b, GLushort a), us(r), us(g), us(b), us(a))
CUR(CA_COLOR, glColor3ui, (GLuint r, GLuint g, GLuint b), ui(r), ui(g), ui(b), 1)
CUR(CA_COLOR, glColor4ui, (GLuint r, GLuint g, GLuint b, GLuint a), ui(r), ui(g), ui(b), ui(a))
CUR(CA_COLOR, glColor3fv, (const GLfloat* v), v[0], v[1], v[2], 1)
CUR(CA_COLOR, glColor4fv, (const GLfloat* v), v[0], v[1], v[2], v[3])
CUR(CA_COLOR, glColor3dv, (const GLdouble* v), (float)v[0], (float)v[1], (float)v[2], 1)
CUR(CA_COLOR, glColor4dv, (const GLdouble* v), (float)v[0], (float)v[1], (float)v[2], (float)v[3])
CUR(CA_COLOR, glColor3ubv, (const GLubyte* v), ub(v[0]), ub(v[1]), ub(v[2]), 1)
CUR(CA_COLOR, glColor4ubv, (const GLubyte* v), ub(v[0]), ub(v[1]), ub(v[2]), ub(v[3]))
CUR(CA_SECONDARY, glSecondaryColor3f, (GLfloat r, GLfloat g, GLfloat b), r, g, b, 1)
CUR(CA_SECONDARY, glSecondaryColor3fv, (const GLfloat* v), v[0], v[1], v[2], 1)
CUR(CA_SECONDARY, glSecondaryColor3ub, (GLubyte r, GLubyte g, GLubyte b), ub(r), ub(g), ub(b), 1)
CUR(CA_SECONDARY, glSecondaryColor3ubv, (const GLubyte* v), ub(v[0]), ub(v[1]), ub(v[2]), 1)
CUR(CA_NORMAL, glNormal3f, (GLfloat x, GLfloat y, GLfloat z), x, y, z, 1)
CUR(CA_NORMAL, glNormal3d, (GLdouble x, GLdouble y, GLdouble z), (float)x, (float)y, (float)z, 1)
CUR(CA_NORMAL, glNormal3fv, (const GLfloat* v), v[0], v[1], v[2], 1)
CUR(CA_NORMAL, glNormal3dv, (const GLdouble* v), (float)v[0], (float)v[1], (float)v[2], 1)
CUR(CA_NORMAL, glNormal3b, (GLbyte x, GLbyte y, GLbyte z), bn(x), bn(y), bn(z), 1)
CUR(CA_NORMAL, glNormal3s, (GLshort x, GLshort y, GLshort z), sn(x), sn(y), sn(z), 1)
CUR(CA_NORMAL, glNormal3i, (GLint x, GLint y, GLint z), in_(x), in_(y), in_(z), 1)
CUR(CA_FOG, glFogCoordf, (GLfloat f), f, 0, 0, 1)
CUR(CA_FOG, glFogCoordd, (GLdouble f), (float)f, 0, 0, 1)
CUR(CA_FOG, glFogCoordfv, (const GLfloat* f), f[0], 0, 0, 1)
CUR(CA_TEX0, glTexCoord1f, (GLfloat s), s, 0, 0, 1)
CUR(CA_TEX0, glTexCoord2f, (GLfloat s, GLfloat t), s, t, 0, 1)
CUR(CA_TEX0, glTexCoord3f, (GLfloat s, GLfloat t, GLfloat r), s, t, r, 1)
CUR(CA_TEX0, glTexCoord4f, (GLfloat s, GLfloat t, GLfloat r, GLfloat q), s, t, r, q)
CUR(CA_TEX0, glTexCoord1d, (GLdouble s), (float)s, 0, 0, 1)
CUR(CA_TEX0, glTexCoord2d, (GLdouble s, GLdouble t), (float)s, (float)t, 0, 1)
CUR(CA_TEX0, glTexCoord3d, (GLdouble s, GLdouble t, GLdouble r), (float)s, (float)t, (float)r, 1)
CUR(CA_TEX0, glTexCoord4d, (GLdouble s, GLdouble t, GLdouble r, GLdouble q), (float)s, (float)t, (float)r, (float)q)
CUR(CA_TEX0, glTexCoord1i, (GLint s), (float)s, 0, 0, 1)
CUR(CA_TEX0, glTexCoord2i, (GLint s, GLint t), (float)s, (float)t, 0, 1)
CUR(CA_TEX0, glTexCoord2s, (GLshort s, GLshort t), (float)s, (float)t, 0, 1)
CUR(CA_TEX0, glTexCoord2fv, (const GLfloat* v), v[0], v[1], 0, 1)
CUR(CA_TEX0, glTexCoord2dv, (const GLdouble* v), (float)v[0], (float)v[1], 0, 1)
CUR(CA_TEX0, glTexCoord3fv, (const GLfloat* v), v[0], v[1], v[2], 1)
CUR(CA_TEX0, glTexCoord4fv, (const GLfloat* v), v[0], v[1], v[2], v[3])

static void multiTex(Context* ctx, GLenum unit, float s, float t, float r, float q) {
    int u = (int)unit - GL_TEXTURE0;
    if (u < 0 || u >= 8) return ctx->setError(GL_INVALID_ENUM);
    RECORD(setCurrent(ctx, CA_TEX0 + u, s, t, r, q));
    setCurrent(ctx, CA_TEX0 + u, s, t, r, q);
}
API void APIENTRY glMultiTexCoord1f(GLenum u, GLfloat s) { MGL_CTX(); multiTex(ctx, u, s, 0, 0, 1); }
API void APIENTRY glMultiTexCoord2f(GLenum u, GLfloat s, GLfloat t) { MGL_CTX(); multiTex(ctx, u, s, t, 0, 1); }
API void APIENTRY glMultiTexCoord3f(GLenum u, GLfloat s, GLfloat t, GLfloat r) { MGL_CTX(); multiTex(ctx, u, s, t, r, 1); }
API void APIENTRY glMultiTexCoord4f(GLenum u, GLfloat s, GLfloat t, GLfloat r, GLfloat q) { MGL_CTX(); multiTex(ctx, u, s, t, r, q); }
API void APIENTRY glMultiTexCoord2fv(GLenum u, const GLfloat* v) { MGL_CTX(); multiTex(ctx, u, v[0], v[1], 0, 1); }
API void APIENTRY glMultiTexCoord4fv(GLenum u, const GLfloat* v) { MGL_CTX(); multiTex(ctx, u, v[0], v[1], v[2], v[3]); }
API void APIENTRY glMultiTexCoord2s(GLenum u, GLshort s, GLshort t) { MGL_CTX(); multiTex(ctx, u, s, t, 0, 1); }
API void APIENTRY glMultiTexCoord2i(GLenum u, GLint s, GLint t) { MGL_CTX(); multiTex(ctx, u, (float)s, (float)t, 0, 1); }
API void APIENTRY glMultiTexCoord2d(GLenum u, GLdouble s, GLdouble t) { MGL_CTX(); multiTex(ctx, u, (float)s, (float)t, 0, 1); }
API void APIENTRY glMultiTexCoord1fARB(GLenum u, GLfloat s) { glMultiTexCoord1f(u, s); }
API void APIENTRY glMultiTexCoord2fARB(GLenum u, GLfloat s, GLfloat t) { glMultiTexCoord2f(u, s, t); }

#define VTX(name, params, x, y, z, w)                  \
    API void APIENTRY name params {                    \
        MGL_CTX();                                     \
        RECORD(emitVertex(ctx, x, y, z, w));           \
        emitVertex(ctx, x, y, z, w);                   \
    }
VTX(glVertex2f, (GLfloat x, GLfloat y), x, y, 0, 1)
VTX(glVertex3f, (GLfloat x, GLfloat y, GLfloat z), x, y, z, 1)
VTX(glVertex4f, (GLfloat x, GLfloat y, GLfloat z, GLfloat w), x, y, z, w)
VTX(glVertex2d, (GLdouble x, GLdouble y), (float)x, (float)y, 0, 1)
VTX(glVertex3d, (GLdouble x, GLdouble y, GLdouble z), (float)x, (float)y, (float)z, 1)
VTX(glVertex4d, (GLdouble x, GLdouble y, GLdouble z, GLdouble w), (float)x, (float)y, (float)z, (float)w)
VTX(glVertex2i, (GLint x, GLint y), (float)x, (float)y, 0, 1)
VTX(glVertex3i, (GLint x, GLint y, GLint z), (float)x, (float)y, (float)z, 1)
VTX(glVertex2s, (GLshort x, GLshort y), x, y, 0, 1)
VTX(glVertex3s, (GLshort x, GLshort y, GLshort z), x, y, z, 1)
VTX(glVertex2fv, (const GLfloat* v), v[0], v[1], 0, 1)
VTX(glVertex3fv, (const GLfloat* v), v[0], v[1], v[2], 1)
VTX(glVertex4fv, (const GLfloat* v), v[0], v[1], v[2], v[3])
VTX(glVertex2dv, (const GLdouble* v), (float)v[0], (float)v[1], 0, 1)
VTX(glVertex3dv, (const GLdouble* v), (float)v[0], (float)v[1], (float)v[2], 1)
VTX(glVertex2iv, (const GLint* v), (float)v[0], (float)v[1], 0, 1)
VTX(glVertex3iv, (const GLint* v), (float)v[0], (float)v[1], (float)v[2], 1)

static void emitVertex(Context* ctx, float x, float y, float z, float w) {
    CompatState* cs = compat(ctx);
    if (!cs->inBegin) return;  // vertex outside Begin/End: undefined, ignore
    ImmVertex v;
    memcpy(v.v, cs->current, sizeof v.v);
    v.v[CA_VERTEX][0] = x;
    v.v[CA_VERTEX][1] = y;
    v.v[CA_VERTEX][2] = z;
    v.v[CA_VERTEX][3] = w;
    cs->imm.push_back(v);
}

static void drawImmediate(Context* ctx, GLenum mode, const std::vector<ImmVertex>& verts);

API void APIENTRY glBegin(GLenum mode) {
    MGL_CTX();
    RECORD(glBegin(mode));
    CompatState* cs = compat(ctx);
    if (cs->inBegin) return ctx->setError(GL_INVALID_OPERATION);
    if (mode > GL_POLYGON) return ctx->setError(GL_INVALID_ENUM);
    cs->inBegin = true;
    cs->beginMode = mode;
    cs->imm.clear();
}

API void APIENTRY glEnd(void) {
    MGL_CTX();
    RECORD(glEnd());
    CompatState* cs = compat(ctx);
    if (!cs->inBegin) return ctx->setError(GL_INVALID_OPERATION);
    cs->inBegin = false;
    if (!cs->imm.empty()) {
        std::vector<ImmVertex> verts;
        verts.swap(cs->imm);
        drawImmediate(ctx, cs->beginMode, verts);
    }
}

API void APIENTRY glRectf(GLfloat x1, GLfloat y1, GLfloat x2, GLfloat y2) {
    glBegin(GL_QUADS);
    glVertex2f(x1, y1);
    glVertex2f(x2, y1);
    glVertex2f(x2, y2);
    glVertex2f(x1, y2);
    glEnd();
}
API void APIENTRY glRecti(GLint x1, GLint y1, GLint x2, GLint y2) { glRectf((float)x1, (float)y1, (float)x2, (float)y2); }
API void APIENTRY glRectd(GLdouble x1, GLdouble y1, GLdouble x2, GLdouble y2) { glRectf((float)x1, (float)y1, (float)x2, (float)y2); }

// ---------------------------------------------------------------------------
// lighting, material, fog, alpha, texenv, texgen, clip planes
// ---------------------------------------------------------------------------
static void lightParam(Context* ctx, GLenum light, GLenum pname, const float* p) {
    if (light < GL_LIGHT0 || light >= GL_LIGHT0 + 8) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    LightState& L = cs->lights[light - GL_LIGHT0];
    const Mat4& mv = cs->modelview.back();
    switch (pname) {
    case GL_AMBIENT: memcpy(L.ambient, p, 16); break;
    case GL_DIFFUSE: memcpy(L.diffuse, p, 16); break;
    case GL_SPECULAR: memcpy(L.specular, p, 16); break;
    case GL_POSITION:
        for (int r = 0; r < 4; r++) L.position[r] = mv.at(r, 0) * p[0] + mv.at(r, 1) * p[1] + mv.at(r, 2) * p[2] + mv.at(r, 3) * p[3];
        break;
    case GL_SPOT_DIRECTION:
        for (int r = 0; r < 3; r++) L.spotDirection[r] = mv.at(r, 0) * p[0] + mv.at(r, 1) * p[1] + mv.at(r, 2) * p[2];
        break;
    case GL_SPOT_EXPONENT: L.spotExponent = p[0]; break;
    case GL_SPOT_CUTOFF: L.spotCutoff = p[0]; break;
    case GL_CONSTANT_ATTENUATION: L.constantAtt = p[0]; break;
    case GL_LINEAR_ATTENUATION: L.linearAtt = p[0]; break;
    case GL_QUADRATIC_ATTENUATION: L.quadraticAtt = p[0]; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

API void APIENTRY glLightfv(GLenum light, GLenum pname, const GLfloat* params) {
    MGL_CTX();
    float p[4] = {params[0], 0, 0, 0};
    int n = (pname == GL_AMBIENT || pname == GL_DIFFUSE || pname == GL_SPECULAR || pname == GL_POSITION) ? 4 : pname == GL_SPOT_DIRECTION ? 3 : 1;
    for (int i = 0; i < n; i++) p[i] = params[i];
    RECORD(lightParam(ctx, light, pname, p));
    lightParam(ctx, light, pname, p);
}
API void APIENTRY glLightf(GLenum light, GLenum pname, GLfloat param) { glLightfv(light, pname, &param); }
API void APIENTRY glLighti(GLenum light, GLenum pname, GLint param) {
    float f = (float)param;
    glLightfv(light, pname, &f);
}
API void APIENTRY glLightiv(GLenum light, GLenum pname, const GLint* params) {
    float f[4] = {(float)params[0], 0, 0, 0};
    if (pname == GL_AMBIENT || pname == GL_DIFFUSE || pname == GL_SPECULAR)
        for (int i = 0; i < 4; i++) f[i] = in_(params[i]);
    else if (pname == GL_POSITION || pname == GL_SPOT_DIRECTION)
        for (int i = 0; i < 4; i++) f[i] = (float)params[i];
    glLightfv(light, pname, f);
}

static void materialParam(Context* ctx, GLenum face, GLenum pname, const float* p) {
    CompatState* cs = compat(ctx);
    for (int f = 0; f < 2; f++) {
        if ((f == 0 && face == GL_BACK) || (f == 1 && face == GL_FRONT)) continue;
        MaterialState& m = cs->material[f];
        switch (pname) {
        case GL_AMBIENT: memcpy(m.ambient, p, 16); break;
        case GL_DIFFUSE: memcpy(m.diffuse, p, 16); break;
        case GL_SPECULAR: memcpy(m.specular, p, 16); break;
        case GL_EMISSION: memcpy(m.emission, p, 16); break;
        case GL_SHININESS: m.shininess = p[0]; break;
        case GL_AMBIENT_AND_DIFFUSE: memcpy(m.ambient, p, 16); memcpy(m.diffuse, p, 16); break;
        default: return ctx->setError(GL_INVALID_ENUM);
        }
    }
}

API void APIENTRY glMaterialfv(GLenum face, GLenum pname, const GLfloat* params) {
    MGL_CTX();
    float p[4] = {params[0], 0, 0, 0};
    if (pname != GL_SHININESS)
        for (int i = 0; i < 4; i++) p[i] = params[i];
    RECORD(materialParam(ctx, face, pname, p));
    materialParam(ctx, face, pname, p);
}
API void APIENTRY glMaterialf(GLenum face, GLenum pname, GLfloat param) { glMaterialfv(face, pname, &param); }
API void APIENTRY glMateriali(GLenum face, GLenum pname, GLint param) {
    float f = (float)param;
    glMaterialfv(face, pname, &f);
}
API void APIENTRY glMaterialiv(GLenum face, GLenum pname, const GLint* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = pname == GL_SHININESS ? (float)params[i] : in_(params[i]);
    glMaterialfv(face, pname, f);
}

API void APIENTRY glColorMaterial(GLenum face, GLenum mode) {
    MGL_CTX();
    RECORD(glColorMaterial(face, mode));
    CompatState* cs = compat(ctx);
    cs->colorMaterialFace = face;
    cs->colorMaterialMode = mode;
}

API void APIENTRY glLightModelfv(GLenum pname, const GLfloat* params) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    switch (pname) {
    case GL_LIGHT_MODEL_AMBIENT: memcpy(cs->lightModelAmbient, params, 16); break;
    case GL_LIGHT_MODEL_LOCAL_VIEWER: cs->localViewer = params[0] != 0; break;
    case GL_LIGHT_MODEL_TWO_SIDE: cs->twoSide = params[0] != 0; break;
    case GL_LIGHT_MODEL_COLOR_CONTROL: cs->colorControl = (GLenum)params[0]; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glLightModelf(GLenum pname, GLfloat param) { glLightModelfv(pname, &param); }
API void APIENTRY glLightModeli(GLenum pname, GLint param) {
    float f = (float)param;
    glLightModelfv(pname, &f);
}
API void APIENTRY glLightModeliv(GLenum pname, const GLint* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = pname == GL_LIGHT_MODEL_AMBIENT ? in_(params[i]) : (float)params[i];
    glLightModelfv(pname, f);
}

API void APIENTRY glShadeModel(GLenum mode) {
    MGL_CTX();
    RECORD(glShadeModel(mode));
    if (mode != GL_FLAT && mode != GL_SMOOTH) return ctx->setError(GL_INVALID_ENUM);
    compat(ctx)->shadeModel = mode;
}

API void APIENTRY glAlphaFunc(GLenum func, GLfloat ref) {
    MGL_CTX();
    RECORD(glAlphaFunc(func, ref));
    if (func < GL_NEVER || func > GL_ALWAYS) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    cs->alphaFunc = func;
    cs->alphaRef = std::clamp(ref, 0.0f, 1.0f);
}

API void APIENTRY glFogfv(GLenum pname, const GLfloat* params) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    switch (pname) {
    case GL_FOG_MODE: cs->fogMode = (GLenum)params[0]; break;
    case GL_FOG_DENSITY: cs->fogDensity = params[0]; break;
    case GL_FOG_START: cs->fogStart = params[0]; break;
    case GL_FOG_END: cs->fogEnd = params[0]; break;
    case GL_FOG_COLOR: memcpy(cs->fogColor, params, 16); break;
    case GL_FOG_INDEX: cs->fogIndex = params[0]; break;
    case GL_FOG_COORD_SRC: cs->fogCoordSrc = (GLenum)params[0]; break;
    case 0x855A /* GL_FOG_DISTANCE_MODE_NV */: break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glFogf(GLenum pname, GLfloat param) { glFogfv(pname, &param); }
API void APIENTRY glFogi(GLenum pname, GLint param) {
    float f = (float)param;
    glFogfv(pname, &f);
}
API void APIENTRY glFogiv(GLenum pname, const GLint* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = pname == GL_FOG_COLOR ? in_(params[i]) : (float)params[i];
    glFogfv(pname, f);
}

API void APIENTRY glTexEnvfv(GLenum target, GLenum pname, const GLfloat* params) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    int u = std::min<int>(ctx->activeTexture, 7);
    TexEnv& e = cs->texEnv[u];
    if (target == GL_TEXTURE_FILTER_CONTROL) {
        if (pname == GL_TEXTURE_LOD_BIAS) e.lodBias = params[0];
        return;
    }
    if (target == GL_POINT_SPRITE) return;  // GL_COORD_REPLACE: always on for sprites
    if (target != GL_TEXTURE_ENV) return ctx->setError(GL_INVALID_ENUM);
    GLenum v = (GLenum)params[0];
    switch (pname) {
    case GL_TEXTURE_ENV_MODE: e.mode = v; break;
    case GL_TEXTURE_ENV_COLOR: memcpy(e.color, params, 16); break;
    case GL_COMBINE_RGB: e.combineRGB = v; break;
    case GL_COMBINE_ALPHA: e.combineA = v; break;
    case GL_SRC0_RGB: case GL_SRC1_RGB: case GL_SRC2_RGB: e.srcRGB[pname - GL_SRC0_RGB] = v; break;
    case GL_SRC0_ALPHA: case GL_SRC1_ALPHA: case GL_SRC2_ALPHA: e.srcA[pname - GL_SRC0_ALPHA] = v; break;
    case GL_OPERAND0_RGB: case GL_OPERAND1_RGB: case GL_OPERAND2_RGB: e.opRGB[pname - GL_OPERAND0_RGB] = v; break;
    case GL_OPERAND0_ALPHA: case GL_OPERAND1_ALPHA: case GL_OPERAND2_ALPHA: e.opA[pname - GL_OPERAND0_ALPHA] = v; break;
    case GL_RGB_SCALE: e.scaleRGB = params[0]; break;
    case GL_ALPHA_SCALE: e.scaleA = params[0]; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glTexEnvf(GLenum target, GLenum pname, GLfloat param) { glTexEnvfv(target, pname, &param); }
API void APIENTRY glTexEnvi(GLenum target, GLenum pname, GLint param) {
    float f = (float)param;
    glTexEnvfv(target, pname, &f);
}
API void APIENTRY glTexEnviv(GLenum target, GLenum pname, const GLint* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = pname == GL_TEXTURE_ENV_COLOR ? in_(params[i]) : (float)params[i];
    glTexEnvfv(target, pname, f);
}

API void APIENTRY glGetTexEnvfv(GLenum target, GLenum pname, GLfloat* params) {
    MGL_CTX();
    TexEnv& e = compat(ctx)->texEnv[std::min<int>(ctx->activeTexture, 7)];
    if (pname == GL_TEXTURE_ENV_COLOR) memcpy(params, e.color, 16);
    else if (pname == GL_TEXTURE_ENV_MODE) params[0] = (float)e.mode;
    else if (pname == GL_TEXTURE_LOD_BIAS) params[0] = e.lodBias;
    else params[0] = 0;
}
API void APIENTRY glGetTexEnviv(GLenum target, GLenum pname, GLint* params) {
    float f[4] = {};
    glGetTexEnvfv(target, pname, f);
    for (int i = 0; i < (pname == GL_TEXTURE_ENV_COLOR ? 4 : 1); i++) params[i] = (GLint)f[i];
}

API void APIENTRY glTexGenfv(GLenum coord, GLenum pname, const GLfloat* params) {
    MGL_CTX();
    if (coord < GL_S || coord > GL_Q) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    TexGen& g = cs->texGen[std::min<int>(ctx->activeTexture, 7)];
    int c = coord - GL_S;
    switch (pname) {
    case GL_TEXTURE_GEN_MODE: g.mode[c] = (GLenum)params[0]; break;
    case GL_OBJECT_PLANE: memcpy(g.objPlane[c], params, 16); break;
    case GL_EYE_PLANE: {
        Mat4 inv = inverse(cs->modelview.back());
        for (int k = 0; k < 4; k++)
            g.eyePlane[c][k] = params[0] * inv.at(0, k) + params[1] * inv.at(1, k) + params[2] * inv.at(2, k) + params[3] * inv.at(3, k);
        break;
    }
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glTexGenf(GLenum coord, GLenum pname, GLfloat param) { glTexGenfv(coord, pname, &param); }
API void APIENTRY glTexGeni(GLenum coord, GLenum pname, GLint param) {
    float f = (float)param;
    glTexGenfv(coord, pname, &f);
}
API void APIENTRY glTexGend(GLenum coord, GLenum pname, GLdouble param) {
    float f = (float)param;
    glTexGenfv(coord, pname, &f);
}
API void APIENTRY glTexGeniv(GLenum coord, GLenum pname, const GLint* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = (float)params[i];
    glTexGenfv(coord, pname, f);
}
API void APIENTRY glTexGendv(GLenum coord, GLenum pname, const GLdouble* params) {
    float f[4];
    for (int i = 0; i < 4; i++) f[i] = (float)params[i];
    glTexGenfv(coord, pname, f);
}

API void APIENTRY glClipPlane(GLenum plane, const GLdouble* eq) {
    MGL_CTX();
    if (plane < GL_CLIP_PLANE0 || plane >= GL_CLIP_PLANE0 + 8) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    Mat4 inv = inverse(cs->modelview.back());
    float* p = cs->clipPlanes[plane - GL_CLIP_PLANE0];
    for (int k = 0; k < 4; k++) p[k] = (float)(eq[0] * inv.at(0, k) + eq[1] * inv.at(1, k) + eq[2] * inv.at(2, k) + eq[3] * inv.at(3, k));
}
API void APIENTRY glGetClipPlane(GLenum plane, GLdouble* eq) {
    MGL_CTX();
    if (plane < GL_CLIP_PLANE0 || plane >= GL_CLIP_PLANE0 + 8) return ctx->setError(GL_INVALID_ENUM);
    for (int k = 0; k < 4; k++) eq[k] = compat(ctx)->clipPlanes[plane - GL_CLIP_PLANE0][k];
}

// Accepted with no effect.
API void APIENTRY glLineStipple(GLint factor, GLushort pattern) {}
API void APIENTRY glPolygonStipple(const GLubyte* mask) {}
API void APIENTRY glEdgeFlag(GLboolean flag) {}
API void APIENTRY glEdgeFlagv(const GLboolean* flag) {}
API void APIENTRY glIndexi(GLint c) {}
API void APIENTRY glIndexf(GLfloat c) {}
API void APIENTRY glPassThrough(GLfloat token) {}
API void APIENTRY glInitNames(void) {}
API void APIENTRY glPushName(GLuint name) {}
API void APIENTRY glPopName(void) {}
API void APIENTRY glLoadName(GLuint name) {}
API GLint APIENTRY glRenderMode(GLenum mode) { return 0; }
API void APIENTRY glAccum(GLenum op, GLfloat value) {}
API void APIENTRY glClearAccum(GLfloat r, GLfloat g, GLfloat b, GLfloat a) {}
API void APIENTRY glClearIndex(GLfloat c) {}
API void APIENTRY glPixelTransferf(GLenum pname, GLfloat param) {}
API void APIENTRY glPixelTransferi(GLenum pname, GLint param) {}
API void APIENTRY glPixelZoom(GLfloat xfactor, GLfloat yfactor) {}
API void APIENTRY glRasterPos2f(GLfloat x, GLfloat y) {}
API void APIENTRY glRasterPos2i(GLint x, GLint y) {}
API void APIENTRY glRasterPos3f(GLfloat x, GLfloat y, GLfloat z) {}
API void APIENTRY glWindowPos2i(GLint x, GLint y) {}
API void APIENTRY glBitmap(GLsizei w, GLsizei h, GLfloat xo, GLfloat yo, GLfloat xm, GLfloat ym, const GLubyte* bitmap) {}
API void APIENTRY glDrawPixels(GLsizei w, GLsizei h, GLenum format, GLenum type, const void* pixels) {
    static bool warned = false;
    if (!warned) MGL_WARN("glDrawPixels not implemented");
    warned = true;
}
API void APIENTRY glPrioritizeTextures(GLsizei n, const GLuint* textures, const GLfloat* priorities) {}
API GLboolean APIENTRY glAreTexturesResident(GLsizei n, const GLuint* textures, GLboolean* residences) {
    for (int i = 0; i < n; i++) residences[i] = GL_TRUE;
    return GL_TRUE;
}

// ---------------------------------------------------------------------------
// client arrays
// ---------------------------------------------------------------------------
static int arrayForCap(Context* ctx, GLenum cap) {
    CompatState* cs = compat(ctx);
    switch (cap) {
    case GL_VERTEX_ARRAY: return CA_VERTEX;
    case GL_NORMAL_ARRAY: return CA_NORMAL;
    case GL_COLOR_ARRAY: return CA_COLOR;
    case GL_SECONDARY_COLOR_ARRAY: return CA_SECONDARY;
    case GL_FOG_COORD_ARRAY: return CA_FOG;
    case GL_TEXTURE_COORD_ARRAY: return CA_TEX0 + std::min<int>(cs->clientActiveTexture, 7);
    case GL_INDEX_ARRAY: case GL_EDGE_FLAG_ARRAY: return -2;
    default: return -1;
    }
}

API void APIENTRY glEnableClientState(GLenum cap) {
    MGL_CTX();
    int a = arrayForCap(ctx, cap);
    if (a == -1) return ctx->setError(GL_INVALID_ENUM);
    if (a >= 0) compat(ctx)->arrays[a].enabled = true;
}

API void APIENTRY glDisableClientState(GLenum cap) {
    MGL_CTX();
    int a = arrayForCap(ctx, cap);
    if (a == -1) return ctx->setError(GL_INVALID_ENUM);
    if (a >= 0) compat(ctx)->arrays[a].enabled = false;
}

API void APIENTRY glClientActiveTexture(GLenum texture) {
    MGL_CTX();
    int u = (int)texture - GL_TEXTURE0;
    if (u < 0 || u >= 8) return ctx->setError(GL_INVALID_ENUM);
    compat(ctx)->clientActiveTexture = u;
}

static void setArray(Context* ctx, int a, GLint size, GLenum type, GLsizei stride, const void* ptr) {
    if (stride < 0) return ctx->setError(GL_INVALID_VALUE);
    ClientArray& c = compat(ctx)->arrays[a];
    c.size = size;
    c.type = type;
    c.stride = stride;
    c.pointer = ptr;
    c.buffer = ctx->arrayBuffer;
}

API void APIENTRY glVertexPointer(GLint size, GLenum type, GLsizei stride, const void* ptr) { MGL_CTX(); setArray(ctx, CA_VERTEX, size, type, stride, ptr); }
API void APIENTRY glNormalPointer(GLenum type, GLsizei stride, const void* ptr) { MGL_CTX(); setArray(ctx, CA_NORMAL, 3, type, stride, ptr); }
API void APIENTRY glColorPointer(GLint size, GLenum type, GLsizei stride, const void* ptr) { MGL_CTX(); setArray(ctx, CA_COLOR, size, type, stride, ptr); }
API void APIENTRY glSecondaryColorPointer(GLint size, GLenum type, GLsizei stride, const void* ptr) { MGL_CTX(); setArray(ctx, CA_SECONDARY, size, type, stride, ptr); }
API void APIENTRY glFogCoordPointer(GLenum type, GLsizei stride, const void* ptr) { MGL_CTX(); setArray(ctx, CA_FOG, 1, type, stride, ptr); }
API void APIENTRY glTexCoordPointer(GLint size, GLenum type, GLsizei stride, const void* ptr) {
    MGL_CTX();
    setArray(ctx, CA_TEX0 + std::min<int>(compat(ctx)->clientActiveTexture, 7), size, type, stride, ptr);
}
API void APIENTRY glIndexPointer(GLenum type, GLsizei stride, const void* ptr) {}
API void APIENTRY glEdgeFlagPointer(GLsizei stride, const void* ptr) {}

API void APIENTRY glInterleavedArrays(GLenum format, GLsizei stride, const void* pointer) {
    MGL_CTX();
    const uint8_t* p = (const uint8_t*)pointer;
    struct Fmt { bool t, c, n; int tsz, csz, vsz; GLenum ctype; int coff, noff, voff, size; };
    Fmt f{};
    const int F = 4;
    switch (format) {
    case GL_V2F: f = {false, false, false, 0, 0, 2, 0, 0, 0, 0, 2 * F}; break;
    case GL_V3F: f = {false, false, false, 0, 0, 3, 0, 0, 0, 0, 3 * F}; break;
    case GL_C4UB_V2F: f = {false, true, false, 0, 4, 2, GL_UNSIGNED_BYTE, 0, 0, 4, 4 + 2 * F}; break;
    case GL_C4UB_V3F: f = {false, true, false, 0, 4, 3, GL_UNSIGNED_BYTE, 0, 0, 4, 4 + 3 * F}; break;
    case GL_C3F_V3F: f = {false, true, false, 0, 3, 3, GL_FLOAT, 0, 0, 3 * F, 6 * F}; break;
    case GL_N3F_V3F: f = {false, false, true, 0, 0, 3, 0, 0, 0, 3 * F, 6 * F}; break;
    case GL_C4F_N3F_V3F: f = {false, true, true, 0, 4, 3, GL_FLOAT, 0, 4 * F, 7 * F, 10 * F}; break;
    case GL_T2F_V3F: f = {true, false, false, 2, 0, 3, 0, 0, 0, 2 * F, 5 * F}; break;
    case GL_T4F_V4F: f = {true, false, false, 4, 0, 4, 0, 0, 0, 4 * F, 8 * F}; break;
    case GL_T2F_C4UB_V3F: f = {true, true, false, 2, 4, 3, GL_UNSIGNED_BYTE, 2 * F, 0, 2 * F + 4, 2 * F + 4 + 3 * F}; break;
    case GL_T2F_C3F_V3F: f = {true, true, false, 2, 3, 3, GL_FLOAT, 2 * F, 0, 5 * F, 8 * F}; break;
    case GL_T2F_N3F_V3F: f = {true, false, true, 2, 0, 3, 0, 0, 2 * F, 5 * F, 8 * F}; break;
    case GL_T2F_C4F_N3F_V3F: f = {true, true, true, 2, 4, 3, GL_FLOAT, 2 * F, 6 * F, 9 * F, 12 * F}; break;
    case GL_T4F_C4F_N3F_V4F: f = {true, true, true, 4, 4, 4, GL_FLOAT, 4 * F, 8 * F, 11 * F, 15 * F}; break;
    default: return ctx->setError(GL_INVALID_ENUM);
    }
    GLsizei s = stride ? stride : f.size;
    CompatState* cs = compat(ctx);
    cs->arrays[CA_COLOR].enabled = cs->arrays[CA_NORMAL].enabled = false;
    cs->arrays[CA_TEX0 + cs->clientActiveTexture].enabled = false;
    if (f.t) {
        setArray(ctx, CA_TEX0 + cs->clientActiveTexture, f.tsz, GL_FLOAT, s, p);
        cs->arrays[CA_TEX0 + cs->clientActiveTexture].enabled = true;
    }
    if (f.c) {
        setArray(ctx, CA_COLOR, f.csz, f.ctype, s, p + f.coff);
        cs->arrays[CA_COLOR].enabled = true;
    }
    if (f.n) {
        setArray(ctx, CA_NORMAL, 3, GL_FLOAT, s, p + f.noff);
        cs->arrays[CA_NORMAL].enabled = true;
    }
    setArray(ctx, CA_VERTEX, f.vsz, GL_FLOAT, s, p + f.voff);
    cs->arrays[CA_VERTEX].enabled = true;
}

// ---------------------------------------------------------------------------
// display lists
// ---------------------------------------------------------------------------
API GLuint APIENTRY glGenLists(GLsizei range) {
    MGL_CTX_RET(0);
    if (range < 0) return ctx->setError(GL_INVALID_VALUE), 0;
    CompatState* cs = compat(ctx);
    GLuint base = 1;
    while (true) {
        bool free = true;
        for (GLsizei i = 0; i < range; i++)
            if (cs->lists.count(base + i)) {
                free = false;
                base = base + i + 1;
                break;
            }
        if (free) break;
    }
    for (GLsizei i = 0; i < range; i++) cs->lists[base + i];
    return range ? base : 0;
}

API void APIENTRY glDeleteLists(GLuint list, GLsizei range) {
    MGL_CTX();
    if (range < 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    for (GLsizei i = 0; i < range; i++) cs->lists.erase(list + i);
}

API GLboolean APIENTRY glIsList(GLuint list) {
    MGL_CTX_RET(GL_FALSE);
    return compat(ctx)->lists.count(list) ? GL_TRUE : GL_FALSE;
}

API void APIENTRY glNewList(GLuint list, GLenum mode) {
    MGL_CTX();
    if (list == 0) return ctx->setError(GL_INVALID_VALUE);
    if (mode != GL_COMPILE && mode != GL_COMPILE_AND_EXECUTE) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    if (cs->compilingList) return ctx->setError(GL_INVALID_OPERATION);
    cs->compilingList = list;
    cs->listMode = mode;
    cs->listBuf.clear();
}

API void APIENTRY glEndList(void) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    if (!cs->compilingList) return ctx->setError(GL_INVALID_OPERATION);
    cs->lists[cs->compilingList] = std::move(cs->listBuf);
    cs->listBuf.clear();
    cs->compilingList = 0;
    cs->listMode = 0;
}

static void callList(Context* ctx, GLuint list) {
    CompatState* cs = compat(ctx);
    if (cs->callDepth >= 64) return;
    auto it = cs->lists.find(list);
    if (it == cs->lists.end()) return;
    cs->callDepth++;
    auto cmds = it->second;  // copy: lists may be redefined while executing
    for (auto& c : cmds) c();
    cs->callDepth--;
}

API void APIENTRY glCallList(GLuint list) {
    MGL_CTX();
    RECORD(callList(ctx, list));
    callList(ctx, list);
}

API void APIENTRY glCallLists(GLsizei n, GLenum type, const void* lists) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    for (GLsizei i = 0; i < n; i++) {
        GLuint id;
        switch (type) {
        case GL_BYTE: id = (GLuint)((const GLbyte*)lists)[i]; break;
        case GL_UNSIGNED_BYTE: id = ((const GLubyte*)lists)[i]; break;
        case GL_SHORT: id = (GLuint)((const GLshort*)lists)[i]; break;
        case GL_UNSIGNED_SHORT: id = ((const GLushort*)lists)[i]; break;
        case GL_INT: id = (GLuint)((const GLint*)lists)[i]; break;
        case GL_UNSIGNED_INT: id = ((const GLuint*)lists)[i]; break;
        case GL_FLOAT: id = (GLuint)((const GLfloat*)lists)[i]; break;
        default: return ctx->setError(GL_INVALID_ENUM);
        }
        GLuint l = cs->listBase + id;
        RECORD(callList(ctx, l));
        callList(ctx, l);
    }
}

API void APIENTRY glListBase(GLuint base) {
    MGL_CTX();
    compat(ctx)->listBase = base;
}

// ---------------------------------------------------------------------------
// attribute stack (coarse: saves the full fixed-function state blocks)
// ---------------------------------------------------------------------------
namespace {
struct AttribSnapshot {
    BlendState blend[kMaxDrawBuffers];
    float blendColor[4], clearColor[4];
    bool depthTest, depthWrite, stencilTest, cullFace, scissor, dither, polygonOffsetFill, blendOn;
    GLenum depthFunc, cullMode, frontFace, polygonMode;
    StencilFace stencil[2];
    float viewport[4];
    double depthRange[2];
    int scissorBox[4];
    uint32_t scissorEnabled;
    float polygonOffset[2];
    float lineWidth, pointSize;
    GLuint activeTexture;
    bool lighting, fog, alphaTest, colorMaterial, normalize;
    GLenum alphaFunc, shadeModel, fogMode;
    float alphaRef, fogDensity, fogStart, fogEnd, fogColor[4];
    float color[4];
    uint32_t texEnabled[8];
    bool lightsEnabled[8];
    double clearDepth;
};
}  // namespace

API void APIENTRY glPushAttrib(GLbitfield mask) {
    MGL_CTX();
    RECORD(glPushAttrib(mask));
    CompatState* cs = compat(ctx);
    if (cs->attribStack.size() >= 16) return ctx->setError(GL_STACK_OVERFLOW);
    AttribSnapshot s;
    memcpy(s.blend, ctx->blend, sizeof s.blend);
    memcpy(s.blendColor, ctx->blendColor, 16);
    memcpy(s.clearColor, ctx->clearColor, 16);
    s.depthTest = ctx->depthTest;
    s.depthWrite = ctx->depthWrite;
    s.stencilTest = ctx->stencilTest;
    s.cullFace = ctx->cullFace;
    s.dither = ctx->dither;
    s.polygonOffsetFill = ctx->polygonOffsetFill;
    s.depthFunc = ctx->depthFunc;
    s.cullMode = ctx->cullMode;
    s.frontFace = ctx->frontFace;
    s.polygonMode = ctx->polygonMode;
    memcpy(s.stencil, ctx->stencil, sizeof s.stencil);
    memcpy(s.viewport, ctx->viewport[0], 16);
    s.depthRange[0] = ctx->depthRange[0][0];
    s.depthRange[1] = ctx->depthRange[0][1];
    memcpy(s.scissorBox, ctx->scissor[0], 16);
    s.scissorEnabled = ctx->scissorEnabled;
    s.polygonOffset[0] = ctx->polygonOffsetFactor;
    s.polygonOffset[1] = ctx->polygonOffsetUnits;
    s.lineWidth = ctx->lineWidth;
    s.pointSize = ctx->pointSize;
    s.activeTexture = ctx->activeTexture;
    s.lighting = cs->lighting;
    s.fog = cs->fog;
    s.alphaTest = cs->alphaTest;
    s.colorMaterial = cs->colorMaterial;
    s.normalize = cs->normalize;
    s.alphaFunc = cs->alphaFunc;
    s.shadeModel = cs->shadeModel;
    s.fogMode = cs->fogMode;
    s.alphaRef = cs->alphaRef;
    s.fogDensity = cs->fogDensity;
    s.fogStart = cs->fogStart;
    s.fogEnd = cs->fogEnd;
    memcpy(s.fogColor, cs->fogColor, 16);
    memcpy(s.color, cs->current[CA_COLOR], 16);
    memcpy(s.texEnabled, cs->texEnabled, sizeof s.texEnabled);
    for (int i = 0; i < 8; i++) s.lightsEnabled[i] = cs->lights[i].enabled;
    s.clearDepth = ctx->clearDepth;
    std::vector<uint8_t> bytes(sizeof s);
    memcpy(bytes.data(), &s, sizeof s);
    cs->attribStack.push_back({mask, std::move(bytes)});
}

API void APIENTRY glPopAttrib(void) {
    MGL_CTX();
    RECORD(glPopAttrib());
    CompatState* cs = compat(ctx);
    if (cs->attribStack.empty()) return ctx->setError(GL_STACK_UNDERFLOW);
    auto [mask, bytes] = cs->attribStack.back();
    cs->attribStack.pop_back();
    AttribSnapshot s;
    memcpy(&s, bytes.data(), sizeof s);
    if (mask & (GL_COLOR_BUFFER_BIT | GL_ENABLE_BIT)) {
        memcpy(ctx->blend, s.blend, sizeof s.blend);
        memcpy(ctx->blendColor, s.blendColor, 16);
        ctx->dither = s.dither;
        cs->alphaTest = s.alphaTest;
        cs->alphaFunc = s.alphaFunc;
        cs->alphaRef = s.alphaRef;
    }
    if (mask & GL_COLOR_BUFFER_BIT) memcpy(ctx->clearColor, s.clearColor, 16);
    if (mask & (GL_DEPTH_BUFFER_BIT | GL_ENABLE_BIT)) {
        ctx->depthTest = s.depthTest;
        ctx->depthWrite = (mask & GL_DEPTH_BUFFER_BIT) ? s.depthWrite : ctx->depthWrite;
        if (mask & GL_DEPTH_BUFFER_BIT) {
            ctx->depthFunc = s.depthFunc;
            ctx->clearDepth = s.clearDepth;
        }
    }
    if (mask & (GL_STENCIL_BUFFER_BIT | GL_ENABLE_BIT)) {
        ctx->stencilTest = s.stencilTest;
        if (mask & GL_STENCIL_BUFFER_BIT) memcpy(ctx->stencil, s.stencil, sizeof s.stencil);
    }
    if (mask & (GL_POLYGON_BIT | GL_ENABLE_BIT)) {
        ctx->cullFace = s.cullFace;
        ctx->polygonOffsetFill = s.polygonOffsetFill;
        if (mask & GL_POLYGON_BIT) {
            ctx->cullMode = s.cullMode;
            ctx->frontFace = s.frontFace;
            ctx->polygonMode = s.polygonMode;
            ctx->polygonOffsetFactor = s.polygonOffset[0];
            ctx->polygonOffsetUnits = s.polygonOffset[1];
        }
    }
    if (mask & GL_VIEWPORT_BIT) {
        for (int i = 0; i < kMaxViewports; i++) {
            memcpy(ctx->viewport[i], s.viewport, 16);
            ctx->depthRange[i][0] = s.depthRange[0];
            ctx->depthRange[i][1] = s.depthRange[1];
        }
    }
    if (mask & (GL_SCISSOR_BIT | GL_ENABLE_BIT)) {
        ctx->scissorEnabled = s.scissorEnabled;
        if (mask & GL_SCISSOR_BIT)
            for (int i = 0; i < kMaxViewports; i++) memcpy(ctx->scissor[i], s.scissorBox, 16);
    }
    if (mask & GL_LINE_BIT) ctx->lineWidth = s.lineWidth;
    if (mask & GL_POINT_BIT) ctx->pointSize = s.pointSize;
    if (mask & (GL_LIGHTING_BIT | GL_ENABLE_BIT)) {
        cs->lighting = s.lighting;
        cs->colorMaterial = s.colorMaterial;
        for (int i = 0; i < 8; i++) cs->lights[i].enabled = s.lightsEnabled[i];
        if (mask & GL_LIGHTING_BIT) cs->shadeModel = s.shadeModel;
    }
    if (mask & (GL_FOG_BIT | GL_ENABLE_BIT)) {
        cs->fog = s.fog;
        if (mask & GL_FOG_BIT) {
            cs->fogMode = s.fogMode;
            cs->fogDensity = s.fogDensity;
            cs->fogStart = s.fogStart;
            cs->fogEnd = s.fogEnd;
            memcpy(cs->fogColor, s.fogColor, 16);
        }
    }
    if (mask & GL_TRANSFORM_BIT) cs->normalize = s.normalize;
    if (mask & GL_CURRENT_BIT) memcpy(cs->current[CA_COLOR], s.color, 16);
    if (mask & (GL_TEXTURE_BIT | GL_ENABLE_BIT)) memcpy(cs->texEnabled, s.texEnabled, sizeof s.texEnabled);
    if (mask & GL_TEXTURE_BIT) ctx->activeTexture = s.activeTexture;
}

API void APIENTRY glPushClientAttrib(GLbitfield mask) {}
API void APIENTRY glPopClientAttrib(void) {}

// ---------------------------------------------------------------------------
// fixed-function program generation
// ---------------------------------------------------------------------------
static std::string ffKey(Context* ctx, CompatState* cs, bool* usedUnits) {
    std::string k;
    auto add = [&](int v) { k += std::to_string(v) + ","; };
    add(cs->lighting);
    if (cs->lighting) {
        for (auto& l : cs->lights) add(l.enabled ? (l.position[3] == 0 ? 1 : 2) + (l.spotCutoff != 180 ? 2 : 0) : 0);
        add(cs->colorMaterial ? (int)cs->colorMaterialMode : 0);
        add(cs->twoSide);
        add(cs->localViewer);
        add(cs->colorControl == GL_SEPARATE_SPECULAR_COLOR);
    }
    add(cs->normalize * 2 + cs->rescaleNormal);
    add(cs->fog ? (int)cs->fogMode : 0);
    if (cs->fog) add(cs->fogCoordSrc == GL_FOG_COORD);
    add(cs->colorSum);
    add(cs->pointSprite);
    for (int u = 0; u < 8; u++) {
        uint32_t e = cs->texEnabled[u];
        int target = e & 8 ? 8 : e & 4 ? 4 : e & 16 ? 16 : e & 2 ? 2 : e & 1 ? 1 : 0;
        // texture must also be complete-ish (bound); unbound default textures behave as disabled
        usedUnits[u] = target != 0;
        add(target);
        if (!target) continue;
        const TexEnv& te = cs->texEnv[u];
        add(te.mode);
        if (te.mode == GL_COMBINE) {
            add(te.combineRGB);
            add(te.combineA);
            for (int i = 0; i < 3; i++) {
                add(te.srcRGB[i]);
                add(te.srcA[i]);
                add(te.opRGB[i]);
                add(te.opA[i]);
            }
        }
        // base format of the bound texture determines REPLACE/DECAL behavior
        int ti = texTargetIndex(target == 8 ? GL_TEXTURE_CUBE_MAP : target == 4 ? GL_TEXTURE_3D : target == 16 ? GL_TEXTURE_RECTANGLE : target == 1 ? GL_TEXTURE_1D : GL_TEXTURE_2D);
        Texture* t = ctx->textures[u][ti].get();
        const TexImageLevel& L = t->baseImage();
        add(L.fmt ? (int)L.fmt->baseFormat : 0);
        const TexGen& g = cs->texGen[u];
        for (int c = 0; c < 4; c++) add(g.enabled[c] ? (int)g.mode[c] : 0);
    }
    return k;
}

static std::string glslTexType(int target) {
    switch (target) {
    case 1: return "sampler1D";
    case 4: return "sampler3D";
    case 8: return "samplerCube";
    case 16: return "sampler2DRect";
    default: return "sampler2D";
    }
}

static std::string texLookup(int target, int u) {
    std::string t = "gl_TexCoord[" + std::to_string(u) + "]";
    std::string s = "tex" + std::to_string(u);
    switch (target) {
    case 1: return "texture1D(" + s + ", " + t + ".s / " + t + ".q)";
    case 4: return "texture3D(" + s + ", " + t + ".stp / " + t + ".q)";
    case 8: return "textureCube(" + s + ", " + t + ".stp)";
    case 16: return "texture2DRect(" + s + ", " + t + ".st / " + t + ".q)";
    default: return "texture2D(" + s + ", " + t + ".st / " + t + ".q)";
    }
}

static std::string combineSrc(GLenum src, GLenum op, bool alpha, int u) {
    std::string base;
    switch (src) {
    case GL_TEXTURE: base = "t"; break;
    case GL_CONSTANT: base = "gl_TextureEnvColor[" + std::to_string(u) + "]"; break;
    case GL_PRIMARY_COLOR: base = "gl_Color"; break;
    case GL_PREVIOUS: base = "c"; break;
    default:
        if (src >= GL_TEXTURE0 && src < GL_TEXTURE0 + 8) base = "tx" + std::to_string(src - GL_TEXTURE0);
        else base = "t";
    }
    std::string sw = alpha ? ".a" : ".rgb";
    switch (op) {
    case GL_ONE_MINUS_SRC_COLOR: return "(1.0 - " + base + sw + ")";
    case GL_SRC_ALPHA: return alpha ? base + ".a" : "vec3(" + base + ".a)";
    case GL_ONE_MINUS_SRC_ALPHA: return alpha ? "(1.0 - " + base + ".a)" : "vec3(1.0 - " + base + ".a)";
    default: return base + sw;
    }
}

static std::string combineExpr(GLenum fn, const std::string& a0, const std::string& a1, const std::string& a2, bool alpha) {
    std::string half = alpha ? "0.5" : "vec3(0.5)";
    switch (fn) {
    case GL_REPLACE: return a0;
    case GL_ADD: return a0 + " + " + a1;
    case GL_ADD_SIGNED: return a0 + " + " + a1 + " - " + half;
    case GL_INTERPOLATE: return "mix(" + a1 + ", " + a0 + ", " + a2 + ")";
    case GL_SUBTRACT: return a0 + " - " + a1;
    case GL_DOT3_RGB: case GL_DOT3_RGBA:
        return alpha ? a0 : "vec3(4.0 * dot(" + a0 + " - vec3(0.5), " + a1 + " - vec3(0.5)))";
    default: return a0 + " * " + a1;
    }
}

static Ref<ProgramObj> buildFFProgram(Context* ctx, CompatState* cs, const bool* units) {
    std::string vs = "#version 120\n", fs = "#version 120\n";
    int targets[8] = {};
    for (int u = 0; u < 8; u++) {
        uint32_t e = cs->texEnabled[u];
        targets[u] = e & 8 ? 8 : e & 4 ? 4 : e & 16 ? 16 : e & 2 ? 2 : e & 1 ? 1 : 0;
        if (targets[u]) fs += "uniform " + glslTexType(targets[u]) + " tex" + std::to_string(u) + ";\n";
    }
    // --- vertex ---
    vs += "void main() {\n  vec4 eye = gl_ModelViewMatrix * gl_Vertex;\n  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n";
    vs += "  gl_ClipVertex = eye;\n";
    if (cs->lighting) {
        vs += "  vec3 N = gl_NormalMatrix * gl_Normal;\n";
        if (cs->normalize) vs += "  N = normalize(N);\n";
        else if (cs->rescaleNormal) vs += "  N *= gl_NormalScale;\n";
        bool cm = cs->colorMaterial;
        GLenum cmm = cs->colorMaterialMode;
        auto matTerm = [&](const char* face, const char* term, GLenum mode) -> std::string {
            if (cm && (cmm == mode || (cmm == GL_AMBIENT_AND_DIFFUSE && (mode == GL_AMBIENT || mode == GL_DIFFUSE)))) return "gl_Color";
            return std::string(face) + "." + term;
        };
        for (int side = 0; side < (cs->twoSide ? 2 : 1); side++) {
            const char* mat = side == 0 ? "gl_FrontMaterial" : "gl_BackMaterial";
            std::string n = side == 0 ? "N" : "(-N)";
            std::string out = side == 0 ? "gl_FrontColor" : "gl_BackColor";
            std::string out2 = side == 0 ? "gl_FrontSecondaryColor" : "gl_BackSecondaryColor";
            vs += "  {\n  vec4 amb = " + matTerm(mat, "ambient", GL_AMBIENT) + ", dif = " + matTerm(mat, "diffuse", GL_DIFFUSE) +
                  ", spc = " + matTerm(mat, "specular", GL_SPECULAR) + ", emi = " + matTerm(mat, "emission", GL_EMISSION) + ";\n";
            vs += "  vec3 sum = emi.rgb + gl_LightModel.ambient.rgb * amb.rgb;\n  vec3 spec = vec3(0.0);\n";
            for (int i = 0; i < 8; i++) {
                const LightState& L = cs->lights[i];
                if (!L.enabled) continue;
                std::string ls = "gl_LightSource[" + std::to_string(i) + "]";
                vs += "  {\n";
                if (L.position[3] == 0) {
                    vs += "    vec3 Ld = normalize(" + ls + ".position.xyz); float att = 1.0;\n";
                } else {
                    vs += "    vec3 Lv = " + ls + ".position.xyz - eye.xyz; float d = length(Lv); vec3 Ld = Lv / d;\n";
                    vs += "    float att = 1.0 / (" + ls + ".constantAttenuation + " + ls + ".linearAttenuation * d + " + ls +
                          ".quadraticAttenuation * d * d);\n";
                    if (L.spotCutoff != 180)
                        vs += "    float sd = dot(-Ld, normalize(" + ls + ".spotDirection)); att *= sd < " + ls +
                              ".spotCosCutoff ? 0.0 : pow(max(sd, 0.0), " + ls + ".spotExponent);\n";
                }
                vs += "    float nl = max(dot(" + n + ", Ld), 0.0);\n";
                vs += "    vec3 H = normalize(Ld + " + std::string(cs->localViewer ? "normalize(-eye.xyz)" : "vec3(0.0, 0.0, 1.0)") + ");\n";
                vs += "    float sp = nl > 0.0 ? pow(max(dot(" + n + ", H), 0.0), " + std::string(mat) + ".shininess) : 0.0;\n";
                vs += "    sum += att * (" + ls + ".ambient.rgb * amb.rgb + nl * " + ls + ".diffuse.rgb * dif.rgb);\n";
                vs += "    spec += att * sp * " + ls + ".specular.rgb * spc.rgb;\n  }\n";
            }
            if (cs->colorControl == GL_SEPARATE_SPECULAR_COLOR) {
                vs += "  " + out + " = vec4(clamp(sum, 0.0, 1.0), dif.a);\n  " + out2 + " = vec4(clamp(spec, 0.0, 1.0), 1.0);\n";
            } else {
                vs += "  " + out + " = vec4(clamp(sum + spec, 0.0, 1.0), dif.a);\n  " + out2 + " = vec4(0.0);\n";
            }
            vs += "  }\n";
        }
    } else {
        vs += "  gl_FrontColor = gl_Color;\n  gl_BackColor = gl_Color;\n  gl_FrontSecondaryColor = gl_SecondaryColor;\n";
    }
    for (int u = 0; u < 8; u++) {
        if (!targets[u]) continue;
        std::string us = std::to_string(u);
        const TexGen& g = cs->texGen[u];
        vs += "  { vec4 tc = gl_MultiTexCoord" + us + ";\n";
        const char* comp = "stpq";
        bool sphere = false;
        for (int c = 0; c < 4; c++) {
            if (!g.enabled[c]) continue;
            std::string cc(1, comp[c]);
            std::string C(1, (char)toupper(comp[c]));
            switch (g.mode[c]) {
            case GL_OBJECT_LINEAR: vs += "    tc." + cc + " = dot(gl_ObjectPlane" + C + "[" + us + "], gl_Vertex);\n"; break;
            case GL_EYE_LINEAR: vs += "    tc." + cc + " = dot(gl_EyePlane" + C + "[" + us + "], eye);\n"; break;
            case GL_SPHERE_MAP: case GL_REFLECTION_MAP: case GL_NORMAL_MAP: sphere = true; break;
            }
        }
        if (sphere) {
            vs += "    vec3 n2 = normalize(gl_NormalMatrix * gl_Normal); vec3 u2 = normalize(eye.xyz); vec3 r2 = reflect(u2, n2);\n";
            for (int c = 0; c < 3; c++) {
                if (!g.enabled[c]) continue;
                std::string cc(1, comp[c]);
                if (g.mode[c] == GL_SPHERE_MAP) {
                    vs += "    { float m = 2.0 * sqrt(r2.x*r2.x + r2.y*r2.y + (r2.z+1.0)*(r2.z+1.0)); tc." + cc + " = r2." +
                          std::string(1, "xyz"[c]) + " / m + 0.5; }\n";
                } else if (g.mode[c] == GL_REFLECTION_MAP) {
                    vs += "    tc." + cc + " = r2." + std::string(1, "xyz"[c]) + ";\n";
                } else if (g.mode[c] == GL_NORMAL_MAP) {
                    vs += "    tc." + cc + " = n2." + std::string(1, "xyz"[c]) + ";\n";
                }
            }
        }
        vs += "    gl_TexCoord[" + us + "] = gl_TextureMatrix[" + us + "] * tc; }\n";
    }
    if (cs->fog) vs += cs->fogCoordSrc == GL_FOG_COORD ? "  gl_FogFragCoord = gl_FogCoord;\n" : "  gl_FogFragCoord = abs(eye.z);\n";
    vs += "}\n";
    // --- fragment ---
    fs += "void main() {\n  vec4 c = gl_Color;\n";
    for (int u = 0; u < 8; u++)
        if (targets[u]) fs += "  vec4 tx" + std::to_string(u) + " = " + texLookup(targets[u], u) + ";\n";
    for (int u = 0; u < 8; u++) {
        if (!targets[u]) continue;
        std::string us = std::to_string(u);
        const TexEnv& e = cs->texEnv[u];
        int ti = texTargetIndex(targets[u] == 8 ? GL_TEXTURE_CUBE_MAP : targets[u] == 4 ? GL_TEXTURE_3D : targets[u] == 16 ? GL_TEXTURE_RECTANGLE : targets[u] == 1 ? GL_TEXTURE_1D : GL_TEXTURE_2D);
        const TexImageLevel& L = ctx->textures[u][ti]->baseImage();
        GLenum base = L.fmt ? L.fmt->baseFormat : GL_RGBA;
        bool hasRGB = base != GL_ALPHA, hasA = base == GL_RGBA || base == GL_ALPHA || base == GL_LUMINANCE_ALPHA || base == GL_INTENSITY;
        fs += "  { vec4 t = tx" + us + ";\n";
        switch (e.mode) {
        case GL_REPLACE:
            if (hasRGB) fs += "    c.rgb = t.rgb;\n";
            if (hasA) fs += "    c.a = t.a;\n";
            break;
        case GL_DECAL:
            if (base == GL_RGBA) fs += "    c.rgb = mix(c.rgb, t.rgb, t.a);\n";
            else fs += "    c.rgb = t.rgb;\n";
            break;
        case GL_BLEND:
            if (hasRGB) fs += "    c.rgb = mix(c.rgb, gl_TextureEnvColor[" + us + "].rgb, t.rgb);\n";
            if (hasA) fs += base == GL_INTENSITY ? "    c.a = mix(c.a, gl_TextureEnvColor[" + us + "].a, t.a);\n" : "    c.a *= t.a;\n";
            break;
        case GL_ADD:
            if (hasRGB) fs += "    c.rgb += t.rgb;\n";
            if (hasA) fs += base == GL_INTENSITY ? "    c.a += t.a;\n" : "    c.a *= t.a;\n";
            break;
        case GL_COMBINE: {
            std::string r = combineExpr(e.combineRGB, combineSrc(e.srcRGB[0], e.opRGB[0], false, u), combineSrc(e.srcRGB[1], e.opRGB[1], false, u),
                                        combineSrc(e.srcRGB[2], e.opRGB[2], false, u), false);
            std::string a = combineExpr(e.combineA, combineSrc(e.srcA[0], e.opA[0], true, u), combineSrc(e.srcA[1], e.opA[1], true, u),
                                        combineSrc(e.srcA[2], e.opA[2], true, u), true);
            fs += "    vec3 rr = " + r + "; float aa = " + a + ";\n";
            fs += "    c = clamp(vec4(rr * " + std::to_string(e.scaleRGB) + ", aa * " + std::to_string(e.scaleA) + "), 0.0, 1.0);\n";
            break;
        }
        default:  // MODULATE
            if (hasRGB) fs += "    c.rgb *= t.rgb;\n";
            if (hasA) fs += "    c.a *= t.a;\n";
            break;
        }
        fs += "  }\n";
    }
    if (cs->colorSum || (cs->lighting && cs->colorControl == GL_SEPARATE_SPECULAR_COLOR)) fs += "  c.rgb += gl_SecondaryColor.rgb;\n";
    if (cs->fog) {
        switch (cs->fogMode) {
        case GL_LINEAR: fs += "  float f = (gl_Fog.end - gl_FogFragCoord) * gl_Fog.scale;\n"; break;
        case GL_EXP2: fs += "  float f = exp(-pow(gl_Fog.density * gl_FogFragCoord, 2.0));\n"; break;
        default: fs += "  float f = exp(-gl_Fog.density * gl_FogFragCoord);\n"; break;
        }
        fs += "  c.rgb = mix(gl_Fog.color.rgb, c.rgb, clamp(f, 0.0, 1.0));\n";
    }
    fs += "  gl_FragColor = c;\n}\n";

    glsl::CompileOptions opts;
    opts.compatibilityProfile = true;
    glsl::CompileResult v = glsl::compile(glsl::Stage::Vertex, {vs}, opts);
    glsl::CompileResult f = glsl::compile(glsl::Stage::Fragment, {fs}, opts);
    if (!v.shader || !f.shader) {
        MGL_ERR("fixed-function shader failed to compile:\n%s%s\n--- VS ---\n%s\n--- FS ---\n%s", v.infoLog.c_str(), f.infoLog.c_str(),
                vs.c_str(), fs.c_str());
        return nullptr;
    }
    glsl::LinkInput li;
    li.shaders = {v.shader, f.shader};
    glsl::LinkResult lr = glsl::link(li);
    if (!lr.program) {
        MGL_ERR("fixed-function program failed to link:\n%s", lr.infoLog.c_str());
        return nullptr;
    }
    Ref<ProgramObj> p = new ProgramObj();
    p->linked = true;
    p->program = lr.program;
    p->refl = std::move(lr.reflection);
    setupLinkedProgram(p.get());
    // sampler uniforms texN -> unit N
    for (size_t s = 0; s < p->refl.slots.size(); s++) {
        const auto& sl = p->refl.slots[s];
        if (sl.kind != glsl::SlotKind::Texture && sl.kind != glsl::SlotKind::Sampler && sl.kind != glsl::SlotKind::LodBias) continue;
        const std::string& n = p->refl.uniforms[sl.index].name;
        if (n.size() == 4 && n.compare(0, 3, "tex") == 0) p->samplerUnits[s] = n[3] - '0';
    }
    return p;
}

// ---------------------------------------------------------------------------
// gl_* uniforms
// ---------------------------------------------------------------------------
static int findUniform(CompatState* cs, ProgramObj* p, const std::string& name) {
    auto& entry = cs->uniformIdx[p];
    if (entry.first != p->linkGeneration + 1 || entry.second.empty()) {
        entry.first = p->linkGeneration + 1;
        entry.second.clear();
        for (size_t i = 0; i < p->refl.uniforms.size(); i++) {
            const auto& u = p->refl.uniforms[i];
            if (u.blockIndex >= 0 || u.name.compare(0, 3, "gl_") != 0) continue;
            entry.second[u.name] = (int)i;
            // "gl_TextureMatrix[0]" also reachable as "gl_TextureMatrix"
            if (u.name.size() > 3 && u.name.compare(u.name.size() - 3, 3, "[0]") == 0) entry.second[u.name.substr(0, u.name.size() - 3)] = (int)i;
        }
        entry.second["__init"] = -1;
    }
    auto it = entry.second.find(name);
    return it == entry.second.end() ? -1 : it->second;
}

// Writes `count` floats (vec/mat columns of `rows`) into a uniform, optionally an array element.
static void writeU(CompatState* cs, ProgramObj* p, const std::string& name, const float* data, int count, int element = 0) {
    int idx = findUniform(cs, p, name);
    if (idx < 0) return;
    const glsl::Uniform& u = p->refl.uniforms[idx];
    if (u.offset < 0) return;
    TypeShape ts = typeShape(u.type);
    size_t base = u.offset + (size_t)element * u.arrayStride;
    if (ts.cols > 1) {
        int ms = u.matrixStride ? u.matrixStride : ts.rows * 4;
        for (int c = 0; c < ts.cols; c++) {
            size_t off = base + (size_t)c * ms;
            if (off + ts.rows * 4 > p->uniformData.size()) return;
            for (int r = 0; r < ts.rows; r++) {
                float v = (ts.cols == 3 && ts.rows == 3) ? data[c * 3 + r] : data[c * 4 + r];
                memcpy(&p->uniformData[off + r * 4], &v, 4);
            }
        }
    } else {
        if (base + count * 4 > p->uniformData.size()) return;
        memcpy(&p->uniformData[base], data, count * 4);
    }
}

static void writeMat(CompatState* cs, ProgramObj* p, const std::string& name, const Mat4& m, int element = 0) {
    writeU(cs, p, name, m.m, 16, element);
}

static void feedUniforms(Context* ctx, CompatState* cs, ProgramObj* p) {
    if (findUniform(cs, p, "__init") == -2) return;
    const Mat4& mv = cs->modelview.back();
    const Mat4& pr = cs->projection.back();
    Mat4 mvp = mul(pr, mv);
    bool any = false;
    for (auto& [n, i] : cs->uniformIdx[p].second)
        if (i >= 0) any = true;
    if (!any) return;
    writeMat(cs, p, "gl_ModelViewMatrix", mv);
    writeMat(cs, p, "gl_ProjectionMatrix", pr);
    writeMat(cs, p, "gl_ModelViewProjectionMatrix", mvp);
    if (findUniform(cs, p, "gl_NormalMatrix") >= 0 || findUniform(cs, p, "gl_ModelViewMatrixInverse") >= 0 ||
        findUniform(cs, p, "gl_ModelViewMatrixInverseTranspose") >= 0) {
        Mat4 inv = inverse(mv);
        Mat4 it = transpose(inv);
        float nm[9];
        for (int c = 0; c < 3; c++)
            for (int r = 0; r < 3; r++) nm[c * 3 + r] = it.at(r, c);
        writeU(cs, p, "gl_NormalMatrix", nm, 9);
        writeMat(cs, p, "gl_ModelViewMatrixInverse", inv);
        writeMat(cs, p, "gl_ModelViewMatrixInverseTranspose", it);
    }
    writeMat(cs, p, "gl_ModelViewMatrixTranspose", transpose(mv));
    writeMat(cs, p, "gl_ProjectionMatrixTranspose", transpose(pr));
    writeMat(cs, p, "gl_ModelViewProjectionMatrixTranspose", transpose(mvp));
    if (findUniform(cs, p, "gl_ProjectionMatrixInverse") >= 0) writeMat(cs, p, "gl_ProjectionMatrixInverse", inverse(pr));
    if (findUniform(cs, p, "gl_ModelViewProjectionMatrixInverse") >= 0) writeMat(cs, p, "gl_ModelViewProjectionMatrixInverse", inverse(mvp));
    for (int u = 0; u < 8; u++) {
        const Mat4& tm = cs->texture[u].back();
        writeMat(cs, p, "gl_TextureMatrix", tm, u);
        if (findUniform(cs, p, "gl_TextureMatrixInverse") >= 0) writeMat(cs, p, "gl_TextureMatrixInverse", inverse(tm), u);
        writeMat(cs, p, "gl_TextureMatrixTranspose", transpose(tm), u);
        writeU(cs, p, "gl_TextureEnvColor", cs->texEnv[u].color, 4, u);
        const TexGen& g = cs->texGen[u];
        const char* C = "STRQ";
        for (int c = 0; c < 4; c++) {
            writeU(cs, p, std::string("gl_EyePlane") + C[c], g.eyePlane[c], 4, u);
            writeU(cs, p, std::string("gl_ObjectPlane") + C[c], g.objPlane[c], 4, u);
        }
    }
    float scale = 1;
    {
        float x = mv.at(0, 2), y = mv.at(1, 2), z = mv.at(2, 2);
        float l = sqrtf(x * x + y * y + z * z);
        if (l > 0) scale = 1 / l;
    }
    writeU(cs, p, "gl_NormalScale", &scale, 1);
    for (int i = 0; i < 8; i++) writeU(cs, p, "gl_ClipPlane", cs->clipPlanes[i], 4, i);
    float pt[7] = {ctx->pointSize, cs->pointSizeMin, cs->pointSizeMax, ctx->pointFadeThreshold, cs->pointAtt[0], cs->pointAtt[1], cs->pointAtt[2]};
    const char* ptn[7] = {"size", "sizeMin", "sizeMax", "fadeThresholdSize", "distanceConstantAttenuation", "distanceLinearAttenuation",
                          "distanceQuadraticAttenuation"};
    for (int k = 0; k < 7; k++) writeU(cs, p, std::string("gl_Point.") + ptn[k], &pt[k], 1);
    for (int f = 0; f < 2; f++) {
        std::string m = f ? "gl_BackMaterial." : "gl_FrontMaterial.";
        const MaterialState& ms = cs->material[f];
        writeU(cs, p, m + "emission", ms.emission, 4);
        writeU(cs, p, m + "ambient", ms.ambient, 4);
        writeU(cs, p, m + "diffuse", ms.diffuse, 4);
        writeU(cs, p, m + "specular", ms.specular, 4);
        writeU(cs, p, m + "shininess", &ms.shininess, 1);
        std::string lmp = f ? "gl_BackLightModelProduct.sceneColor" : "gl_FrontLightModelProduct.sceneColor";
        float sc[4];
        for (int k = 0; k < 3; k++) sc[k] = ms.emission[k] + ms.ambient[k] * cs->lightModelAmbient[k];
        sc[3] = ms.diffuse[3];
        writeU(cs, p, lmp, sc, 4);
        for (int i = 0; i < 8; i++) {
            const LightState& L = cs->lights[i];
            std::string lp = (f ? "gl_BackLightProduct[" : "gl_FrontLightProduct[") + std::to_string(i) + "].";
            float a[4], d[4], s[4];
            for (int k = 0; k < 4; k++) {
                a[k] = ms.ambient[k] * L.ambient[k];
                d[k] = ms.diffuse[k] * L.diffuse[k];
                s[k] = ms.specular[k] * L.specular[k];
            }
            writeU(cs, p, lp + "ambient", a, 4);
            writeU(cs, p, lp + "diffuse", d, 4);
            writeU(cs, p, lp + "specular", s, 4);
        }
    }
    writeU(cs, p, "gl_LightModel.ambient", cs->lightModelAmbient, 4);
    for (int i = 0; i < 8; i++) {
        const LightState& L = cs->lights[i];
        std::string ls = "gl_LightSource[" + std::to_string(i) + "].";
        writeU(cs, p, ls + "ambient", L.ambient, 4);
        writeU(cs, p, ls + "diffuse", L.diffuse, 4);
        writeU(cs, p, ls + "specular", L.specular, 4);
        writeU(cs, p, ls + "position", L.position, 4);
        float hv[4] = {0, 0, 1, 1};
        {
            float lx = L.position[0], ly = L.position[1], lz = L.position[2];
            float l = sqrtf(lx * lx + ly * ly + lz * lz);
            if (l > 0) {
                hv[0] = lx / l;
                hv[1] = ly / l;
                hv[2] = lz / l + 1;
                float h = sqrtf(hv[0] * hv[0] + hv[1] * hv[1] + hv[2] * hv[2]);
                if (h > 0)
                    for (int k = 0; k < 3; k++) hv[k] /= h;
            }
        }
        writeU(cs, p, ls + "halfVector", hv, 4);
        writeU(cs, p, ls + "spotDirection", L.spotDirection, 3);
        writeU(cs, p, ls + "spotExponent", &L.spotExponent, 1);
        writeU(cs, p, ls + "spotCutoff", &L.spotCutoff, 1);
        float cosc = cosf(L.spotCutoff * (float)M_PI / 180.0f);
        writeU(cs, p, ls + "spotCosCutoff", &cosc, 1);
        writeU(cs, p, ls + "constantAttenuation", &L.constantAtt, 1);
        writeU(cs, p, ls + "linearAttenuation", &L.linearAtt, 1);
        writeU(cs, p, ls + "quadraticAttenuation", &L.quadraticAtt, 1);
    }
    float fogScale = cs->fogEnd != cs->fogStart ? 1.0f / (cs->fogEnd - cs->fogStart) : 0.0f;
    writeU(cs, p, "gl_Fog.color", cs->fogColor, 4);
    writeU(cs, p, "gl_Fog.density", &cs->fogDensity, 1);
    writeU(cs, p, "gl_Fog.start", &cs->fogStart, 1);
    writeU(cs, p, "gl_Fog.end", &cs->fogEnd, 1);
    writeU(cs, p, "gl_Fog.scale", &fogScale, 1);
    p->uniformVersion++;
}

// Points compatibility attributes (gl_Vertex, gl_Color, ...) of `p` at client arrays or current values.
static void bindCompatAttribs(Context* ctx, CompatState* cs, ProgramObj* p) {
    VertexArray* vao = ctx->vao.get();
    bool isDefault = vao == ctx->defaultVao.get();
    for (const auto& a : p->refl.attributes) {
        if (a.name.compare(0, 3, "gl_") != 0 || a.location < 0 || a.location >= kMaxVertexAttribs) continue;
        int idx = -1;
        for (int k = 0; k < CA_COUNT; k++)
            if (a.name == kAttrNames[k]) idx = k;
        if (idx < 0) continue;
        int loc = a.location;
        const ClientArray& arr = cs->arrays[idx];
        if (isDefault) {
            VertexAttrib& va = vao->attribs[loc];
            VertexBinding& vb = vao->bindings[loc];
            if (arr.enabled) {
                va.enabled = true;
                va.size = arr.size;
                va.type = arr.type;
                va.normalized = idx == CA_COLOR || idx == CA_SECONDARY || (idx == CA_NORMAL && arr.type != GL_FLOAT);
                va.integer = false;
                va.relativeOffset = 0;
                va.binding = loc;
                int compSize = (arr.type == GL_UNSIGNED_BYTE || arr.type == GL_BYTE) ? 1 : (arr.type == GL_SHORT || arr.type == GL_UNSIGNED_SHORT || arr.type == GL_HALF_FLOAT) ? 2 : arr.type == GL_DOUBLE ? 8 : 4;
                vb.stride = arr.stride ? arr.stride : compSize * (arr.size == GL_BGRA ? 4 : arr.size);
                vb.divisor = 0;
                if (arr.buffer) {
                    vb.buffer = arr.buffer;
                    vb.offset = (GLintptr)arr.pointer;
                    va.clientPointer = nullptr;
                } else {
                    vb.buffer = nullptr;
                    vb.offset = 0;
                    va.clientPointer = arr.pointer;
                }
                vao->version++;
                continue;
            }
            if (va.enabled) {
                va.enabled = false;
                vao->version++;
            }
        }
        memcpy(ctx->currentAttrib[loc], cs->current[idx], 16);
    }
}

ProgramObj* mglCompatPrepareDraw(Context* ctx, ProgramObj* prog) {
    CompatState* cs = compat(ctx);
    if (!prog) {
        bool units[8] = {};
        std::string key = ffKey(ctx, cs, units);
        auto it = cs->ffPrograms.find(key);
        if (it == cs->ffPrograms.end()) it = cs->ffPrograms.emplace(key, buildFFProgram(ctx, cs, units)).first;
        prog = it->second.get();
        if (!prog) return nullptr;
    }
    feedUniforms(ctx, cs, prog);
    bindCompatAttribs(ctx, cs, prog);
    // alpha test (applied in compat fragment shaders via MGLSys)
    ctx->compatAlphaFunc = cs->alphaTest ? cs->alphaFunc : GL_ALWAYS;
    ctx->compatAlphaRef = cs->alphaRef;
    // flat shading: GL_FLAT on legacy color varyings
    return prog;
}

// ---------------------------------------------------------------------------
// immediate mode submission
// ---------------------------------------------------------------------------
static void drawImmediate(Context* ctx, GLenum mode, const std::vector<ImmVertex>& verts) {
    CompatState* cs = compat(ctx);
    // Temporarily route all compat arrays to the immediate buffer.
    ClientArray saved[CA_COUNT];
    for (int i = 0; i < CA_COUNT; i++) saved[i] = cs->arrays[i];
    Ref<Buffer> savedArrayBuffer = ctx->arrayBuffer;
    Ref<VertexArray> savedVao = ctx->vao;
    ctx->vao = ctx->defaultVao;
    ctx->arrayBuffer = nullptr;
    for (int i = 0; i < CA_COUNT; i++) {
        ClientArray& a = cs->arrays[i];
        a.enabled = true;
        a.size = i == CA_NORMAL ? 3 : i == CA_FOG ? 1 : 4;
        a.type = GL_FLOAT;
        a.stride = sizeof(ImmVertex);
        a.pointer = verts[0].v[i];
        a.buffer = nullptr;
    }
    extern void mglDrawArraysInternal(Context*, GLenum, GLint, GLsizei);
    mglDrawArraysInternal(ctx, mode, 0, (GLsizei)verts.size());
    for (int i = 0; i < CA_COUNT; i++) cs->arrays[i] = saved[i];
    ctx->arrayBuffer = savedArrayBuffer;
    ctx->vao = savedVao;
}
