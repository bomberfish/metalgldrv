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

// Shininess and spot exponents go up to this (GL_NV_light_max_exponent; GL itself stops at 128).
constexpr float kMaxLightExponent = 1024;

struct MaterialState {
    float ambient[4] = {0.2f, 0.2f, 0.2f, 1}, diffuse[4] = {0.8f, 0.8f, 0.8f, 1}, specular[4] = {0, 0, 0, 1};
    float emission[4] = {0, 0, 0, 1};
    float shininess = 0;
};

struct TexEnv {
    GLenum mode = GL_MODULATE;
    float color[4] = {0, 0, 0, 0};
    GLenum combineRGB = GL_MODULATE, combineA = GL_MODULATE;
    // [3]: GL_NV_texture_env_combine4's fourth argument
    GLenum srcRGB[4] = {GL_TEXTURE, GL_PREVIOUS, GL_CONSTANT, GL_ZERO}, srcA[4] = {GL_TEXTURE, GL_PREVIOUS, GL_CONSTANT, GL_ZERO};
    GLenum opRGB[4] = {GL_SRC_COLOR, GL_SRC_COLOR, GL_SRC_ALPHA, GL_SRC_COLOR}, opA[4] = {GL_SRC_ALPHA, GL_SRC_ALPHA, GL_SRC_ALPHA, GL_SRC_ALPHA};
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

// CA_POINTSIZE: GL_APPLE_vertex_point_size's per-vertex point size (an attribute of fixed-function programs).
enum CompatAttr { CA_VERTEX, CA_NORMAL, CA_COLOR, CA_SECONDARY, CA_FOG, CA_TEX0, CA_POINTSIZE = CA_TEX0 + 8, CA_COUNT };
const char* kAttrNames[CA_COUNT] = {"gl_Vertex", "gl_Normal", "gl_Color", "gl_SecondaryColor", "gl_FogCoord",
                                    "gl_MultiTexCoord0", "gl_MultiTexCoord1", "gl_MultiTexCoord2", "gl_MultiTexCoord3",
                                    "gl_MultiTexCoord4", "gl_MultiTexCoord5", "gl_MultiTexCoord6", "gl_MultiTexCoord7",
                                    "mgl_PointSizeAPPLE"};

// GL_APPLE_vertex_point_size (Apple's glext.h; not in the Khronos registry)
constexpr GLenum GL_RASTER_POSITION_UNCLIPPED_IBM = 0x19262;  // GL_IBM_rasterpos_clip (not in the Khronos registry)
constexpr GLenum GL_VERTEX_POINT_SIZE_APPLE = 0x8A26, GL_CURRENT_POINT_SIZE_APPLE = 0x8A27, GL_POINT_SIZE_ARRAY_APPLE = 0x8B9C,
                 GL_POINT_SIZE_ARRAY_TYPE_APPLE = 0x898A, GL_POINT_SIZE_ARRAY_STRIDE_APPLE = 0x898B,
                 GL_POINT_SIZE_ARRAY_POINTER_APPLE = 0x898C, GL_POINT_SIZE_ARRAY_BUFFER_BINDING_APPLE = 0x8B9F;

struct ImmVertex {
    float v[CA_COUNT][4];
    float g[kMaxVertexAttribs][4];  // generic attributes (glVertexAttrib between Begin and End)
};

using ListCmd = std::function<void()>;

// Evaluators (GL 2.1 §5.1). Maps by target index: COLOR_4, INDEX, NORMAL, TEXTURE_COORD_1..4, VERTEX_3, VERTEX_4
// (GL_MAP1_COLOR_4 + i, GL_MAP2_COLOR_4 + i).
constexpr int kEvalTargets = 9;
constexpr int kEvalDims[kEvalTargets] = {4, 1, 3, 1, 2, 3, 4, 3, 4};
constexpr int kMaxEvalOrder = 30;
struct EvalMap {
    float u1 = 0, u2 = 1, v1 = 0, v2 = 1;
    int uorder = 1, vorder = 1;
    std::vector<float> pts;  // uorder x vorder points of the target's dimension, u-major
};

}  // namespace

struct mgl::CompatState {
    GLenum matrixMode = GL_MODELVIEW;
    std::vector<Mat4> modelview{Mat4::identity()}, projection{Mat4::identity()}, color{Mat4::identity()};
    std::vector<Mat4> texture[8];
    uint64_t matrixVersion = 1;

    float current[CA_COUNT][4];
    bool lighting = false, colorMaterial = false, normalize = false, rescaleNormal = false, fog = false, alphaTest = false;
    bool pointSprite = false, colorSum = false;
    bool vertexPointSize = false;  // GL_VERTEX_POINT_SIZE_APPLE
    GLenum colorMaterialFace = GL_FRONT_AND_BACK, colorMaterialMode = GL_AMBIENT_AND_DIFFUSE;
    LightState lights[8];
    MaterialState material[2];
    float lightModelAmbient[4] = {0.2f, 0.2f, 0.2f, 1};
    bool localViewer = false, twoSide = false;
    bool specularVector = false;  // GL_APPLE_specular_vector: Phong (reflection vector) specular
    GLenum colorControl = GL_SINGLE_COLOR;
    GLenum shadeModel = GL_SMOOTH;
    GLenum fogMode = GL_EXP, fogCoordSrc = GL_FRAGMENT_DEPTH;
    GLenum fogDistance = GL_EYE_PLANE_ABSOLUTE_NV;  // GL_NV_fog_distance
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
    // raster position (window coordinates) and the values it carries; pixel zoom; internal objects for
    // glBitmap / glDrawPixels / glCopyPixels (created on first use)
    float rasterPos[4] = {0, 0, 0, 1}, rasterColor[4] = {1, 1, 1, 1}, rasterSecondary[4] = {0, 0, 0, 1}, rasterTex[4] = {0, 0, 0, 1};
    float rasterDistance = 0, zoomX = 1, zoomY = 1;
    bool rasterValid = true, rasterUnclippedIBM = false;
    GLuint metaProg = 0, metaVao = 0, metaVbo = 0, metaTex = 0;
    GLint metaColorLoc = -1, metaModeLoc = -1;
    // evaluators
    EvalMap map1[kEvalTargets], map2[kEvalTargets];
    bool map1Enabled[kEvalTargets] = {}, map2Enabled[kEvalTargets] = {}, autoNormal = false;
    // GL_APPLE_vertex_program_evaluators: maps for the generic attributes (size = components, 1..4)
    EvalMap attribMap1[kMaxVertexAttribs], attribMap2[kMaxVertexAttribs];
    int attribMap1Size[kMaxVertexAttribs] = {}, attribMap2Size[kMaxVertexAttribs] = {};
    bool attribMap1On[kMaxVertexAttribs] = {}, attribMap2On[kMaxVertexAttribs] = {};
    int grid1N = 1, grid2UN = 1, grid2VN = 1;
    float grid1U[2] = {0, 1}, grid2U[2] = {0, 1}, grid2V[2] = {0, 1};
    // GL_APPLE_vertex_array_range: client arrays are read at draw time (always current), so the range, its
    // storage hint and flushes are only recorded
    bool varEnabled = false;
    GLsizei varLength = 0;
    const void* varPointer = nullptr;
    GLenum varStorageHint = GL_STORAGE_SHARED_APPLE;
    // GL_EXT_compiled_vertex_array: the locked range (draws read the arrays as they are either way)
    GLint lockFirst = 0;
    GLsizei lockCount = 0;
    // GL_APPLE_element_array
    bool elementArrayEnabled = false;
    GLenum elementArrayType = GL_UNSIGNED_INT;
    const void* elementArrayPointer = nullptr;

    // immediate mode
    bool inBegin = false;
    GLenum beginMode = 0;
    uint32_t immGeneric = 0;  // generic attributes set between Begin and End (drawn per vertex)
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
        current[CA_POINTSIZE][0] = 1;
        // Initial maps: order 1, holding the default of what they produce.
        static const float defaults[kEvalTargets][4] = {{1, 1, 1, 1}, {1}, {0, 0, 1}, {0}, {0, 0}, {0, 0, 0}, {0, 0, 0, 1}, {0, 0, 0}, {0, 0, 0, 1}};
        for (int i = 0; i < kEvalTargets; i++) {
            map1[i].pts.assign(defaults[i], defaults[i] + kEvalDims[i]);
            map2[i].pts.assign(defaults[i], defaults[i] + kEvalDims[i]);
        }
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

void mglCompatDestroy(Context* ctx) {
    delete ctx->compat;
    ctx->compat = nullptr;
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
    if (cap >= GL_MAP1_COLOR_4 && cap < GL_MAP1_COLOR_4 + kEvalTargets) {
        cs->map1Enabled[cap - GL_MAP1_COLOR_4] = v;
        return true;
    }
    if (cap >= GL_MAP2_COLOR_4 && cap < GL_MAP2_COLOR_4 + kEvalTargets) {
        cs->map2Enabled[cap - GL_MAP2_COLOR_4] = v;
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
    case GL_VERTEX_POINT_SIZE_APPLE: cs->vertexPointSize = v; return true;
    case GL_RASTER_POSITION_UNCLIPPED_IBM: cs->rasterUnclippedIBM = v; return true;
    case GL_TEXTURE_GEN_S: case GL_TEXTURE_GEN_T: case GL_TEXTURE_GEN_R: case GL_TEXTURE_GEN_Q:
        cs->texGen[std::min<int>(ctx->activeTexture, 7)].enabled[cap - GL_TEXTURE_GEN_S] = v;
        return true;
    case GL_AUTO_NORMAL: cs->autoNormal = v; return true;
    case GL_POINT_SMOOTH: case GL_LINE_STIPPLE: case GL_POLYGON_STIPPLE: case GL_INDEX_LOGIC_OP: case GL_VERTEX_PROGRAM_TWO_SIDE:
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
    if (cap >= GL_MAP1_COLOR_4 && cap < GL_MAP1_COLOR_4 + kEvalTargets) {
        *out = cs->map1Enabled[cap - GL_MAP1_COLOR_4];
        return true;
    }
    if (cap >= GL_MAP2_COLOR_4 && cap < GL_MAP2_COLOR_4 + kEvalTargets) {
        *out = cs->map2Enabled[cap - GL_MAP2_COLOR_4];
        return true;
    }
    switch (cap) {
    case GL_AUTO_NORMAL: *out = cs->autoNormal; return true;
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
    case GL_ELEMENT_ARRAY_APPLE: *out = cs->elementArrayEnabled; return true;
    case GL_VERTEX_ARRAY_RANGE_APPLE: *out = cs->varEnabled; return true;
    case GL_VERTEX_POINT_SIZE_APPLE: *out = cs->vertexPointSize; return true;
    case GL_RASTER_POSITION_UNCLIPPED_IBM: *out = cs->rasterUnclippedIBM; return true;
    case GL_POINT_SIZE_ARRAY_APPLE: *out = cs->arrays[CA_POINTSIZE].enabled; return true;
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
    case GL_MAX_EVAL_ORDER: return one(kMaxEvalOrder, false);
    case GL_CURRENT_RASTER_POSITION: return vec(cs->rasterPos, 4);
    case GL_CURRENT_RASTER_POSITION_VALID: {
        out[0] = cs->rasterValid;
        *n = 1;
        *isFloat = false;
        return true;
    }
    case GL_CURRENT_RASTER_COLOR: return vec(cs->rasterColor, 4);
    case GL_CURRENT_RASTER_SECONDARY_COLOR: return vec(cs->rasterSecondary, 4);
    case GL_CURRENT_RASTER_TEXTURE_COORDS: return vec(cs->rasterTex, 4);
    case GL_CURRENT_RASTER_DISTANCE: return one(cs->rasterDistance, true);
    case GL_ZOOM_X: return one(cs->zoomX, true);
    case GL_ZOOM_Y: return one(cs->zoomY, true);
    case GL_MAX_SHININESS_NV: case GL_MAX_SPOT_EXPONENT_NV: return one(kMaxLightExponent, true);
    case GL_MAP1_GRID_SEGMENTS: return one(cs->grid1N, false);
    case GL_MAP1_GRID_DOMAIN: return vec(cs->grid1U, 2);
    case GL_MAP2_GRID_SEGMENTS: {
        out[0] = cs->grid2UN;
        out[1] = cs->grid2VN;
        *n = 2;
        *isFloat = false;
        return true;
    }
    case GL_MAP2_GRID_DOMAIN: {
        float d[4] = {cs->grid2U[0], cs->grid2U[1], cs->grid2V[0], cs->grid2V[1]};
        return vec(d, 4);
    }
    case GL_ELEMENT_ARRAY_TYPE_APPLE: return one(cs->elementArrayType, false);
    case GL_ARRAY_ELEMENT_LOCK_FIRST_EXT: return one(cs->lockFirst, false);
    case GL_ARRAY_ELEMENT_LOCK_COUNT_EXT: return one(cs->lockCount, false);
    case GL_POINT_SIZE_MIN: return one(cs->pointSizeMin, true);
    case GL_POINT_SIZE_MAX: return one(cs->pointSizeMax, true);
    case GL_POINT_DISTANCE_ATTENUATION: return vec(cs->pointAtt, 3);
    case GL_VERTEX_ARRAY_RANGE_LENGTH_APPLE: return one(cs->varLength, false);
    case GL_VERTEX_ARRAY_STORAGE_HINT_APPLE: return one(cs->varStorageHint, false);
    case GL_CURRENT_POINT_SIZE_APPLE: return one(cs->current[CA_POINTSIZE][0], true);
    case GL_POINT_SIZE_ARRAY_TYPE_APPLE: return one(cs->arrays[CA_POINTSIZE].type, false);
    case GL_POINT_SIZE_ARRAY_STRIDE_APPLE: return one(cs->arrays[CA_POINTSIZE].stride, false);
    case GL_POINT_SIZE_ARRAY_BUFFER_BINDING_APPLE: return one(cs->arrays[CA_POINTSIZE].buffer ? cs->arrays[CA_POINTSIZE].buffer->name : 0, false);
    case GL_ALPHA_TEST_FUNC: return one(cs->alphaFunc, false);
    case GL_ALPHA_TEST_REF: return one(cs->alphaRef, true);
    case GL_FOG_MODE: return one(cs->fogMode, false);
    case GL_FOG_DENSITY: return one(cs->fogDensity, true);
    case GL_FOG_START: return one(cs->fogStart, true);
    case GL_FOG_END: return one(cs->fogEnd, true);
    case GL_FOG_COLOR: return vec(cs->fogColor, 4);
    case GL_FOG_COORD_SRC: return one(cs->fogCoordSrc, false);
    case GL_FOG_DISTANCE_MODE_NV: return one(cs->fogDistance, false);
    case GL_LIGHT_MODEL_AMBIENT: return vec(cs->lightModelAmbient, 4);
    case GL_LIGHT_MODEL_TWO_SIDE: return one(cs->twoSide, false);
    case GL_LIGHT_MODEL_LOCAL_VIEWER: return one(cs->localViewer, false);
    case GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE: return one(cs->specularVector, false);
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
CUR(CA_POINTSIZE, glVertexPointSizefAPPLE, (GLfloat s), s, 0, 0, 1)
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
    memcpy(v.g, ctx->currentAttrib, sizeof v.g);
    v.v[CA_VERTEX][0] = x;
    v.v[CA_VERTEX][1] = y;
    v.v[CA_VERTEX][2] = z;
    v.v[CA_VERTEX][3] = w;
    cs->imm.push_back(v);
}

static void drawImmediate(Context* ctx, GLenum mode, const std::vector<ImmVertex>& verts);

static void immBegin(Context* ctx, GLenum mode) {
    CompatState* cs = compat(ctx);
    if (cs->inBegin) return ctx->setError(GL_INVALID_OPERATION);
    if (mode > GL_POLYGON) return ctx->setError(GL_INVALID_ENUM);
    cs->inBegin = true;
    cs->beginMode = mode;
    cs->immGeneric = 0;
    cs->imm.clear();
}

static void immEnd(Context* ctx) {
    CompatState* cs = compat(ctx);
    if (!cs->inBegin) return ctx->setError(GL_INVALID_OPERATION);
    cs->inBegin = false;
    if (!cs->imm.empty()) {
        std::vector<ImmVertex> verts;
        verts.swap(cs->imm);
        drawImmediate(ctx, cs->beginMode, verts);
    }
}

API void APIENTRY glBegin(GLenum mode) {
    MGL_CTX();
    RECORD(glBegin(mode));
    immBegin(ctx, mode);
}

API void APIENTRY glEnd(void) {
    MGL_CTX();
    RECORD(glEnd());
    immEnd(ctx);
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
    case GL_SPOT_EXPONENT:
        if (p[0] < 0 || p[0] > kMaxLightExponent) return ctx->setError(GL_INVALID_VALUE);
        L.spotExponent = p[0];
        break;
    case GL_SPOT_CUTOFF:
        if ((p[0] < 0 || p[0] > 90) && p[0] != 180) return ctx->setError(GL_INVALID_VALUE);
        L.spotCutoff = p[0];
        break;
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
    if (pname == GL_SHININESS && (p[0] < 0 || p[0] > kMaxLightExponent)) return ctx->setError(GL_INVALID_VALUE);
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
    case GL_LIGHT_MODEL_SPECULAR_VECTOR_APPLE: cs->specularVector = params[0] != 0; break;
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
    case GL_FOG_DISTANCE_MODE_NV: {
        GLenum m = (GLenum)params[0];
        if (m != GL_EYE_RADIAL_NV && m != GL_EYE_PLANE && m != GL_EYE_PLANE_ABSOLUTE_NV) return ctx->setError(GL_INVALID_ENUM);
        cs->fogDistance = m;
        break;
    }
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
    case GL_SOURCE3_RGB_NV: e.srcRGB[3] = v; break;
    case GL_SOURCE3_ALPHA_NV: e.srcA[3] = v; break;
    case GL_OPERAND3_RGB_NV: e.opRGB[3] = v; break;
    case GL_OPERAND3_ALPHA_NV: e.opA[3] = v; break;
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
    switch (pname) {
    case GL_TEXTURE_ENV_COLOR: memcpy(params, e.color, 16); break;
    case GL_TEXTURE_ENV_MODE: params[0] = (float)e.mode; break;
    case GL_TEXTURE_LOD_BIAS: params[0] = e.lodBias; break;
    case GL_COMBINE_RGB: params[0] = (float)e.combineRGB; break;
    case GL_COMBINE_ALPHA: params[0] = (float)e.combineA; break;
    case GL_SRC0_RGB: case GL_SRC1_RGB: case GL_SRC2_RGB: params[0] = (float)e.srcRGB[pname - GL_SRC0_RGB]; break;
    case GL_SRC0_ALPHA: case GL_SRC1_ALPHA: case GL_SRC2_ALPHA: params[0] = (float)e.srcA[pname - GL_SRC0_ALPHA]; break;
    case GL_OPERAND0_RGB: case GL_OPERAND1_RGB: case GL_OPERAND2_RGB: params[0] = (float)e.opRGB[pname - GL_OPERAND0_RGB]; break;
    case GL_OPERAND0_ALPHA: case GL_OPERAND1_ALPHA: case GL_OPERAND2_ALPHA: params[0] = (float)e.opA[pname - GL_OPERAND0_ALPHA]; break;
    case GL_RGB_SCALE: params[0] = e.scaleRGB; break;
    case GL_ALPHA_SCALE: params[0] = e.scaleA; break;
    case GL_SOURCE3_RGB_NV: params[0] = (float)e.srcRGB[3]; break;
    case GL_SOURCE3_ALPHA_NV: params[0] = (float)e.srcA[3]; break;
    case GL_OPERAND3_RGB_NV: params[0] = (float)e.opRGB[3]; break;
    case GL_OPERAND3_ALPHA_NV: params[0] = (float)e.opA[3]; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
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
static void getTexGen(Context* ctx, GLenum coord, GLenum pname, double* out) {
    if (coord < GL_S || coord > GL_Q) return ctx->setError(GL_INVALID_ENUM);
    const TexGen& g = compat(ctx)->texGen[std::min<int>(ctx->activeTexture, 7)];
    int c = coord - GL_S;
    switch (pname) {
    case GL_TEXTURE_GEN_MODE: out[0] = g.mode[c]; break;
    case GL_OBJECT_PLANE: for (int k = 0; k < 4; k++) out[k] = g.objPlane[c][k]; break;
    case GL_EYE_PLANE: for (int k = 0; k < 4; k++) out[k] = g.eyePlane[c][k]; break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glGetTexGendv(GLenum coord, GLenum pname, GLdouble* params) {
    MGL_CTX();
    double v[4] = {};
    getTexGen(ctx, coord, pname, v);
    for (int k = 0; k < (pname == GL_TEXTURE_GEN_MODE ? 1 : 4); k++) params[k] = v[k];
}
API void APIENTRY glGetTexGenfv(GLenum coord, GLenum pname, GLfloat* params) {
    MGL_CTX();
    double v[4] = {};
    getTexGen(ctx, coord, pname, v);
    for (int k = 0; k < (pname == GL_TEXTURE_GEN_MODE ? 1 : 4); k++) params[k] = (GLfloat)v[k];
}
API void APIENTRY glGetTexGeniv(GLenum coord, GLenum pname, GLint* params) {
    MGL_CTX();
    double v[4] = {};
    getTexGen(ctx, coord, pname, v);
    for (int k = 0; k < (pname == GL_TEXTURE_GEN_MODE ? 1 : 4); k++) params[k] = (GLint)lround(v[k]);
}

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
    case GL_POINT_SIZE_ARRAY_APPLE: return CA_POINTSIZE;
    case GL_TEXTURE_COORD_ARRAY: return CA_TEX0 + std::min<int>(cs->clientActiveTexture, 7);
    case GL_INDEX_ARRAY: case GL_EDGE_FLAG_ARRAY: return -2;
    default: return -1;
    }
}

API void APIENTRY glEnableClientState(GLenum cap) {
    MGL_CTX();
    if (cap == GL_ELEMENT_ARRAY_APPLE) {
        compat(ctx)->elementArrayEnabled = true;
        return;
    }
    if (cap == GL_VERTEX_ARRAY_RANGE_APPLE) {
        compat(ctx)->varEnabled = true;
        return;
    }
    int a = arrayForCap(ctx, cap);
    if (a == -1) return ctx->setError(GL_INVALID_ENUM);
    if (a >= 0) compat(ctx)->arrays[a].enabled = true;
}

API void APIENTRY glDisableClientState(GLenum cap) {
    MGL_CTX();
    if (cap == GL_ELEMENT_ARRAY_APPLE) {
        compat(ctx)->elementArrayEnabled = false;
        return;
    }
    if (cap == GL_VERTEX_ARRAY_RANGE_APPLE) {
        compat(ctx)->varEnabled = false;
        return;
    }
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
// GL_APPLE_vertex_point_size
API void APIENTRY glPointSizePointerAPPLE(GLenum type, GLsizei stride, const void* ptr) {
    MGL_CTX();
    setArray(ctx, CA_POINTSIZE, 1, type, stride, ptr);
}
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
// Distance attenuation or size bounds other than the defaults: fixed-function shaders compute the point size.
static bool pointParamsActive(const CompatState* cs) {
    return cs->pointAtt[0] != 1 || cs->pointAtt[1] != 0 || cs->pointAtt[2] != 0 || cs->pointSizeMin != 0 || cs->pointSizeMax != 511;
}

static std::string ffKey(Context* ctx, CompatState* cs, bool* usedUnits) {
    std::string k;
    auto add = [&](int v) { k += std::to_string(v) + ","; };
    add(cs->lighting);
    if (cs->lighting) {
        for (auto& l : cs->lights) add(l.enabled ? (l.position[3] == 0 ? 1 : 2) + (l.spotCutoff != 180 ? 2 : 0) : 0);
        add(cs->colorMaterial ? (int)cs->colorMaterialMode : 0);
        add(cs->twoSide);
        add(cs->localViewer);
        add(cs->specularVector);
        add(cs->colorControl == GL_SEPARATE_SPECULAR_COLOR);
    }
    add(cs->normalize * 2 + cs->rescaleNormal);
    add(cs->fog ? (int)cs->fogMode : 0);
    if (cs->fog) add(cs->fogCoordSrc == GL_FOG_COORD);
    if (cs->fog) add(cs->fogDistance);
    add(cs->colorSum);
    add(cs->pointSprite);
    add(cs->vertexPointSize);
    add(pointParamsActive(cs));
    for (int u = 0; u < 8; u++) {
        uint32_t e = cs->texEnabled[u];
        int target = e & 8 ? 8 : e & 4 ? 4 : e & 16 ? 16 : e & 2 ? 2 : e & 1 ? 1 : 0;
        // texture must also be complete-ish (bound); unbound default textures behave as disabled
        usedUnits[u] = target != 0;
        add(target);
        if (!target) continue;
        const TexEnv& te = cs->texEnv[u];
        add(te.mode);
        if (te.mode == GL_COMBINE || te.mode == GL_COMBINE4_NV) {
            add(te.combineRGB);
            add(te.combineA);
            for (int i = 0; i < 4; i++) {
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
    case GL_ZERO: base = "vec4(0.0)"; break;  // NV_texture_env_combine4
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
    // GL_ATI_texture_env_combine3
    case GL_MODULATE_ADD_ATI: return a0 + " * " + a2 + " + " + a1;
    case GL_MODULATE_SIGNED_ADD_ATI: return a0 + " * " + a2 + " + " + a1 + " - " + half;
    case GL_MODULATE_SUBTRACT_ATI: return a0 + " * " + a2 + " - " + a1;
    case GL_DOT3_RGB: case GL_DOT3_RGBA: case GL_DOT3_RGB_EXT: case GL_DOT3_RGBA_EXT:
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
    if (cs->vertexPointSize) vs += "attribute float mgl_PointSizeAPPLE;\n";
    vs += "void main() {\n  vec4 eye = gl_ModelViewMatrix * gl_Vertex;\n  gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;\n";
    if (cs->vertexPointSize || pointParamsActive(cs)) {
        vs += std::string("  float psize = ") + (cs->vertexPointSize ? "mgl_PointSizeAPPLE" : "gl_Point.size") + ";\n";
        if (pointParamsActive(cs)) {
            vs += "  float pd = length(eye.xyz);\n";
            vs += "  psize = clamp(psize * inversesqrt(gl_Point.distanceConstantAttenuation + gl_Point.distanceLinearAttenuation * pd +"
                  " gl_Point.distanceQuadraticAttenuation * pd * pd), gl_Point.sizeMin, gl_Point.sizeMax);\n";
        }
        vs += "  gl_PointSize = psize;\n";
    }
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
                std::string view = cs->localViewer ? "normalize(-eye.xyz)" : "vec3(0.0, 0.0, 1.0)";
                if (cs->specularVector) {
                    // GL_APPLE_specular_vector: the light reflected about the normal, against the view vector
                    vs += "    float sp = nl > 0.0 ? pow(max(dot(reflect(-Ld, " + n + "), " + view + "), 0.0), " + std::string(mat) +
                          ".shininess) : 0.0;\n";
                } else {
                    vs += "    vec3 H = normalize(Ld + " + view + ");\n";
                    vs += "    float sp = nl > 0.0 ? pow(max(dot(" + n + ", H), 0.0), " + std::string(mat) + ".shininess) : 0.0;\n";
                }
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
    if (cs->fog) {
        // fragment depth per GL_NV_fog_distance: |z| (the default), radial distance, or signed -z
        const char* dist = cs->fogDistance == GL_EYE_RADIAL_NV ? "length(eye.xyz)" : cs->fogDistance == GL_EYE_PLANE ? "-eye.z" : "abs(eye.z)";
        vs += cs->fogCoordSrc == GL_FOG_COORD ? std::string("  gl_FogFragCoord = gl_FogCoord;\n") : std::string("  gl_FogFragCoord = ") + dist + ";\n";
    }
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
            // DOT3_RGBA puts the dot product in alpha too; EXT_texture_env_dot3's forms ignore the scales.
            bool dotAlpha = e.combineRGB == GL_DOT3_RGBA || e.combineRGB == GL_DOT3_RGBA_EXT;
            bool extDot = e.combineRGB == GL_DOT3_RGB_EXT || e.combineRGB == GL_DOT3_RGBA_EXT;
            float scaleRGB = extDot ? 1.0f : e.scaleRGB, scaleA = extDot && dotAlpha ? 1.0f : e.scaleA;
            fs += "    vec3 rr = " + r + "; float aa = " + (dotAlpha ? std::string("rr.r") : a) + ";\n";
            fs += "    c = clamp(vec4(rr * " + std::to_string(scaleRGB) + ", aa * " + std::to_string(scaleA) + "), 0.0, 1.0);\n";
            break;
        }
        case GL_COMBINE4_NV: {  // NV_texture_env_combine4: Arg0 * Arg1 + Arg2 * Arg3 (- 0.5 for ADD_SIGNED)
            auto four = [&](bool alpha) {
                const GLenum* src = alpha ? e.srcA : e.srcRGB;
                const GLenum* op = alpha ? e.opA : e.opRGB;
                std::string x = combineSrc(src[0], op[0], alpha, u) + " * " + combineSrc(src[1], op[1], alpha, u) + " + " +
                                combineSrc(src[2], op[2], alpha, u) + " * " + combineSrc(src[3], op[3], alpha, u);
                if ((alpha ? e.combineA : e.combineRGB) == GL_ADD_SIGNED) x += alpha ? " - 0.5" : " - vec3(0.5)";
                return x;
            };
            fs += "    vec3 rr = " + four(false) + "; float aa = " + four(true) + ";\n";
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
        if (sl.kind != glsl::SlotKind::Texture && sl.kind != glsl::SlotKind::Sampler && sl.kind != glsl::SlotKind::LodBias &&
            sl.kind != glsl::SlotKind::SamplerBorder) continue;
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
        if ((a.name.compare(0, 3, "gl_") != 0 && a.name != kAttrNames[CA_POINTSIZE]) || a.location < 0 || a.location >= kMaxVertexAttribs)
            continue;
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
    const bool userProg = prog != nullptr;
    if (!prog) {
        bool units[8] = {};
        std::string key = ffKey(ctx, cs, units);
        auto it = cs->ffPrograms.find(key);
        if (it == cs->ffPrograms.end()) it = cs->ffPrograms.emplace(key, buildFFProgram(ctx, cs, units)).first;
        prog = it->second.get();
        if (!prog) return nullptr;
    }
    // GL_APPLE_vertex_point_size: fixed-function programs then write gl_PointSize, which the draw must use.
    ctx->compatVertexPointSize = !userProg && (cs->vertexPointSize || pointParamsActive(cs));
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
        a.size = i == CA_NORMAL ? 3 : i == CA_FOG || i == CA_POINTSIZE ? 1 : 4;
        a.type = GL_FLOAT;
        a.stride = sizeof(ImmVertex);
        a.pointer = verts[0].v[i];
        a.buffer = nullptr;
    }
    // Generic attributes given per vertex: arrays of the default vertex array object for this draw.
    VertexArray* vao = ctx->defaultVao.get();
    VertexAttrib savedAttr[kMaxVertexAttribs];
    VertexBinding savedBind[kMaxVertexAttribs];
    for (int i = 0; i < kMaxVertexAttribs; i++) {
        if (!(cs->immGeneric & (1u << i))) continue;
        savedAttr[i] = vao->attribs[i];
        savedBind[i] = vao->bindings[i];
        VertexAttrib& va = vao->attribs[i];
        va = VertexAttrib();
        va.enabled = true;
        va.binding = i;
        va.clientPointer = i == 0 ? verts[0].v[CA_VERTEX] : verts[0].g[i];
        vao->bindings[i] = VertexBinding();
        vao->bindings[i].stride = sizeof(ImmVertex);
    }
    if (cs->immGeneric) vao->version++;
    extern void mglDrawArraysInternal(Context*, GLenum, GLint, GLsizei);
    mglDrawArraysInternal(ctx, mode, 0, (GLsizei)verts.size());
    if (cs->immGeneric) {
        for (int i = 0; i < kMaxVertexAttribs; i++)
            if (cs->immGeneric & (1u << i)) {
                vao->attribs[i] = savedAttr[i];
                vao->bindings[i] = savedBind[i];
            }
        vao->version++;
    }
    for (int i = 0; i < CA_COUNT; i++) cs->arrays[i] = saved[i];
    ctx->arrayBuffer = savedArrayBuffer;
    ctx->vao = savedVao;
}

// Client array pointers (glGetPointerv).
bool mglCompatGetPointer(Context* ctx, GLenum pname, void** out) {
    CompatState* cs = compat(ctx);
    switch (pname) {
    case GL_VERTEX_ARRAY_POINTER: *out = (void*)cs->arrays[CA_VERTEX].pointer; return true;
    case GL_NORMAL_ARRAY_POINTER: *out = (void*)cs->arrays[CA_NORMAL].pointer; return true;
    case GL_COLOR_ARRAY_POINTER: *out = (void*)cs->arrays[CA_COLOR].pointer; return true;
    case GL_SECONDARY_COLOR_ARRAY_POINTER: *out = (void*)cs->arrays[CA_SECONDARY].pointer; return true;
    case GL_FOG_COORD_ARRAY_POINTER: *out = (void*)cs->arrays[CA_FOG].pointer; return true;
    case GL_TEXTURE_COORD_ARRAY_POINTER: *out = (void*)cs->arrays[CA_TEX0 + std::min<int>(cs->clientActiveTexture, 7)].pointer; return true;
    case GL_ELEMENT_ARRAY_POINTER_APPLE: *out = (void*)cs->elementArrayPointer; return true;
    case GL_VERTEX_ARRAY_RANGE_POINTER_APPLE: *out = (void*)cs->varPointer; return true;
    case GL_POINT_SIZE_ARRAY_POINTER_APPLE: *out = (void*)cs->arrays[CA_POINTSIZE].pointer; return true;
    default: return false;
    }
}

// ---------------------------------------------------------------------------
// GL_APPLE_element_array: an index array set like the other client arrays (an offset into the bound
// element array buffer, if any); the draws are glDrawElements on it from index `first`.
// ---------------------------------------------------------------------------
static size_t indexSize(GLenum type) { return type == GL_UNSIGNED_BYTE ? 1 : type == GL_UNSIGNED_SHORT ? 2 : 4; }

static const void* elementsAt(CompatState* cs, GLint first) {
    return (const uint8_t*)cs->elementArrayPointer + (ptrdiff_t)first * (ptrdiff_t)indexSize(cs->elementArrayType);
}

API void APIENTRY glElementPointerAPPLE(GLenum type, const void* pointer) {
    MGL_CTX();
    if (type != GL_UNSIGNED_BYTE && type != GL_UNSIGNED_SHORT && type != GL_UNSIGNED_INT) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    cs->elementArrayType = type;
    cs->elementArrayPointer = pointer;
}

API void APIENTRY glDrawElementArrayAPPLE(GLenum mode, GLint first, GLsizei count) {
    MGL_CTX();
    if (first < 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    glDrawElements(mode, count, cs->elementArrayType, elementsAt(cs, first));
}

API void APIENTRY glDrawRangeElementArrayAPPLE(GLenum mode, GLuint start, GLuint end, GLint first, GLsizei count) {
    MGL_CTX();
    if (first < 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    glDrawRangeElements(mode, start, end, count, cs->elementArrayType, elementsAt(cs, first));
}

API void APIENTRY glMultiDrawElementArrayAPPLE(GLenum mode, const GLint* first, const GLsizei* count, GLsizei primcount) {
    MGL_CTX();
    if (primcount < 0) return ctx->setError(GL_INVALID_VALUE);
    for (GLsizei i = 0; i < primcount; i++) glDrawElementArrayAPPLE(mode, first[i], count[i]);
}

API void APIENTRY glMultiDrawRangeElementArrayAPPLE(GLenum mode, GLuint start, GLuint end, const GLint* first, const GLsizei* count,
                                                    GLsizei primcount) {
    MGL_CTX();
    if (primcount < 0) return ctx->setError(GL_INVALID_VALUE);
    for (GLsizei i = 0; i < primcount; i++) glDrawRangeElementArrayAPPLE(mode, start, end, first[i], count[i]);
}

// ---------------------------------------------------------------------------
// GL_APPLE_vertex_array_range
// ---------------------------------------------------------------------------
API void APIENTRY glVertexArrayRangeAPPLE(GLsizei length, void* pointer) {
    MGL_CTX();
    if (length < 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    cs->varLength = length;
    cs->varPointer = pointer;
}

API void APIENTRY glFlushVertexArrayRangeAPPLE(GLsizei length, void* pointer) {
    MGL_CTX();
    if (length < 0) return ctx->setError(GL_INVALID_VALUE);
    (void)pointer;  // draws read client arrays as they are; nothing is cached to update
}

API void APIENTRY glVertexArrayParameteriAPPLE(GLenum pname, GLint param) {
    MGL_CTX();
    if (pname != GL_VERTEX_ARRAY_STORAGE_HINT_APPLE) return ctx->setError(GL_INVALID_ENUM);
    if (param != GL_STORAGE_CLIENT_APPLE && param != GL_STORAGE_CACHED_APPLE && param != GL_STORAGE_SHARED_APPLE)
        return ctx->setError(GL_INVALID_ENUM);
    compat(ctx)->varStorageHint = (GLenum)param;
}

// ---------------------------------------------------------------------------
// Evaluators (GL 2.1 §5.1): maps are evaluated on the CPU into immediate-mode vertices.
// ---------------------------------------------------------------------------
static int evalTarget(GLenum target, bool* two) {
    if (target >= GL_MAP1_COLOR_4 && target < GL_MAP1_COLOR_4 + kEvalTargets) {
        *two = false;
        return target - GL_MAP1_COLOR_4;
    }
    if (target >= GL_MAP2_COLOR_4 && target < GL_MAP2_COLOR_4 + kEvalTargets) {
        *two = true;
        return target - GL_MAP2_COLOR_4;
    }
    return -1;
}

// De Casteljau on `order` points of `dim` floats spaced `step` floats apart: the value at t and d/dt.
static void bezier(const float* p, int step, int order, int dim, float t, float* out, float* deriv) {
    float tmp[kMaxEvalOrder][4];
    for (int i = 0; i < order; i++)
        for (int k = 0; k < dim; k++) tmp[i][k] = p[i * step + k];
    if (order == 1) {
        for (int k = 0; k < dim; k++) {
            out[k] = tmp[0][k];
            if (deriv) deriv[k] = 0;
        }
        return;
    }
    for (int r = order - 1; r > 1; r--)
        for (int i = 0; i < r; i++)
            for (int k = 0; k < dim; k++) tmp[i][k] = (1 - t) * tmp[i][k] + t * tmp[i + 1][k];
    for (int k = 0; k < dim; k++) {
        out[k] = (1 - t) * tmp[0][k] + t * tmp[1][k];
        if (deriv) deriv[k] = (order - 1) * (tmp[1][k] - tmp[0][k]);
    }
}

static void evalMap1(const EvalMap& m, int dim, float u, float* out) {
    float t = m.u2 != m.u1 ? (u - m.u1) / (m.u2 - m.u1) : 0;
    bezier(m.pts.data(), dim, m.uorder, dim, t, out, nullptr);
}

// Value and partial derivatives (per unit of u and v) of a 2D map.
static void evalMap2(const EvalMap& m, int dim, float u, float v, float* out, float* du, float* dv) {
    float s = m.u2 != m.u1 ? (u - m.u1) / (m.u2 - m.u1) : 0, t = m.v2 != m.v1 ? (v - m.v1) / (m.v2 - m.v1) : 0;
    float col[kMaxEvalOrder][4], colDv[kMaxEvalOrder][4];
    for (int i = 0; i < m.uorder; i++)  // along v for each u row
        bezier(m.pts.data() + (size_t)i * m.vorder * dim, dim, m.vorder, dim, t, col[i], colDv[i]);
    bezier(&col[0][0], 4, m.uorder, dim, s, out, du);
    if (dv) bezier(&colDv[0][0], 4, m.uorder, dim, s, dv, nullptr);
    if (du && m.u2 != m.u1)
        for (int k = 0; k < dim; k++) du[k] /= (m.u2 - m.u1);
    if (dv && m.v2 != m.v1)
        for (int k = 0; k < dim; k++) dv[k] /= (m.v2 - m.v1);
}

// The vertex an EvalCoord produces: evaluated attributes for this vertex only (the current values stay).
static void evalEmit(Context* ctx, bool two, float u, float v) {
    CompatState* cs = compat(ctx);
    const EvalMap* maps = two ? cs->map2 : cs->map1;
    const bool* on = two ? cs->map2Enabled : cs->map1Enabled;
    float saved[CA_COUNT][4];
    memcpy(saved, cs->current, sizeof saved);
    auto eval = [&](int i, float* out, float* du, float* dv) {
        if (two) evalMap2(maps[i], kEvalDims[i], u, v, out, du, dv);
        else evalMap1(maps[i], kEvalDims[i], u, out);
    };
    float r[4];
    if (on[0]) {
        eval(0, r, nullptr, nullptr);
        memcpy(cs->current[CA_COLOR], r, 16);
    }
    for (int i = 6; i >= 3; i--)  // the highest enabled texture coordinate map
        if (on[i]) {
            float tc[4] = {0, 0, 0, 1};
            eval(i, tc, nullptr, nullptr);
            memcpy(cs->current[CA_TEX0], tc, 16);
            break;
        }
    if (on[2]) {
        eval(2, r, nullptr, nullptr);
        memcpy(cs->current[CA_NORMAL], r, 12);
    }
    // Generic attribute maps (GL_APPLE_vertex_program_evaluators); attribute 0 is the vertex.
    float savedGeneric[kMaxVertexAttribs][4];
    memcpy(savedGeneric, ctx->currentAttrib, sizeof savedGeneric);
    const EvalMap* amaps = two ? cs->attribMap2 : cs->attribMap1;
    const int* asize = two ? cs->attribMap2Size : cs->attribMap1Size;
    const bool* aon = two ? cs->attribMap2On : cs->attribMap1On;
    bool attrib0 = false;
    float a0[4] = {0, 0, 0, 1};
    for (int i = 0; i < kMaxVertexAttribs; i++) {
        if (!aon[i] || !asize[i]) continue;
        float r4[4] = {0, 0, 0, 1};
        if (two) evalMap2(amaps[i], asize[i], u, v, r4, nullptr, nullptr);
        else evalMap1(amaps[i], asize[i], u, r4);
        if (i == 0) {
            attrib0 = true;
            memcpy(a0, r4, 16);
        } else {
            memcpy(ctx->currentAttrib[i], r4, 16);
            cs->immGeneric |= 1u << i;
        }
    }
    int vi = on[8] ? 8 : on[7] ? 7 : -1;
    if (attrib0) {
        cs->immGeneric |= 1u;
        emitVertex(ctx, a0[0], a0[1], a0[2], a0[3]);
    } else if (vi >= 0) {
        float p[4] = {0, 0, 0, 1}, du[4] = {}, dv[4] = {};
        eval(vi, p, two ? du : nullptr, two ? dv : nullptr);
        if (two && cs->autoNormal) {
            if (vi == 8) {  // homogeneous: derivatives of x/w
                for (int k = 0; k < 3; k++) {
                    du[k] = du[k] * p[3] - p[k] * du[3];
                    dv[k] = dv[k] * p[3] - p[k] * dv[3];
                }
            }
            float nx = du[1] * dv[2] - du[2] * dv[1], ny = du[2] * dv[0] - du[0] * dv[2], nz = du[0] * dv[1] - du[1] * dv[0];
            float l = sqrtf(nx * nx + ny * ny + nz * nz);
            if (l > 0) {
                cs->current[CA_NORMAL][0] = nx / l;
                cs->current[CA_NORMAL][1] = ny / l;
                cs->current[CA_NORMAL][2] = nz / l;
            }
        }
        emitVertex(ctx, p[0], p[1], p[2], vi == 8 ? p[3] : 1.0f);
    }
    memcpy(cs->current, saved, sizeof saved);
    memcpy(ctx->currentAttrib, savedGeneric, sizeof savedGeneric);
}

static void defineMap(Context* ctx, GLenum target, float u1, float u2, int ustride, int uorder, float v1, float v2, int vstride,
                      int vorder, const std::vector<float>& src, bool two) {
    bool isTwo;
    int i = evalTarget(target, &isTwo);
    if (i < 0 || isTwo != two) return ctx->setError(GL_INVALID_ENUM);
    int dim = kEvalDims[i];
    if (u1 == u2 || (two && v1 == v2)) return ctx->setError(GL_INVALID_VALUE);
    if (uorder < 1 || uorder > kMaxEvalOrder || (two && (vorder < 1 || vorder > kMaxEvalOrder))) return ctx->setError(GL_INVALID_VALUE);
    if (ustride < dim || (two && vstride < dim)) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    EvalMap& m = two ? cs->map2[i] : cs->map1[i];
    m.u1 = u1;
    m.u2 = u2;
    m.v1 = v1;
    m.v2 = v2;
    m.uorder = uorder;
    m.vorder = two ? vorder : 1;
    m.pts.resize((size_t)m.uorder * m.vorder * dim);
    for (int a = 0; a < m.uorder; a++)
        for (int b = 0; b < m.vorder; b++)
            for (int k = 0; k < dim; k++) m.pts[((size_t)a * m.vorder + b) * dim + k] = src[(size_t)a * ustride + (size_t)b * vstride + k];
}

// The client points as floats: enough for `order` points `stride` apart (and `vorder` rows `vstride` apart).
template <typename T>
static std::vector<float> mapPoints(const T* p, int ustride, int uorder, int vstride, int vorder, int dim) {
    if (!p || uorder < 1 || vorder < 1) return {};
    size_t n = (size_t)(uorder - 1) * std::max(ustride, 0) + (size_t)(vorder - 1) * std::max(vstride, 0) + dim;
    return std::vector<float>(p, p + n);
}

static int mapDim(GLenum target) {
    bool two;
    int i = evalTarget(target, &two);
    return i < 0 ? 4 : kEvalDims[i];
}

#define MAP1_IMPL(Name, T)                                                                                   \
    API void APIENTRY Name(GLenum target, T u1, T u2, GLint stride, GLint order, const T* points) {          \
        MGL_CTX();                                                                                           \
        std::vector<float> pts = mapPoints(points, stride, order, 0, 1, mapDim(target));                     \
        float a = (float)u1, b = (float)u2;                                                                  \
        RECORD(defineMap(ctx, target, a, b, stride, order, 0, 1, 0, 1, pts, false));                         \
        defineMap(ctx, target, a, b, stride, order, 0, 1, 0, 1, pts, false);                                 \
    }
MAP1_IMPL(glMap1f, GLfloat)
MAP1_IMPL(glMap1d, GLdouble)

#define MAP2_IMPL(Name, T)                                                                                               \
    API void APIENTRY Name(GLenum target, T u1, T u2, GLint ustride, GLint uorder, T v1, T v2, GLint vstride, GLint vorder, \
                           const T* points) {                                                                            \
        MGL_CTX();                                                                                                       \
        std::vector<float> pts = mapPoints(points, ustride, uorder, vstride, vorder, mapDim(target));                     \
        float a = (float)u1, b = (float)u2, c = (float)v1, d = (float)v2;                                                \
        RECORD(defineMap(ctx, target, a, b, ustride, uorder, c, d, vstride, vorder, pts, true));                         \
        defineMap(ctx, target, a, b, ustride, uorder, c, d, vstride, vorder, pts, true);                                 \
    }
MAP2_IMPL(glMap2f, GLfloat)
MAP2_IMPL(glMap2d, GLdouble)

static void getMap(Context* ctx, GLenum target, GLenum query, double* out, int* n) {
    *n = 0;
    bool two;
    int i = evalTarget(target, &two);
    if (i < 0) return ctx->setError(GL_INVALID_ENUM);
    CompatState* cs = compat(ctx);
    const EvalMap& m = two ? cs->map2[i] : cs->map1[i];
    switch (query) {
    case GL_COEFF:
        for (float f : m.pts) out[(*n)++] = f;
        break;
    case GL_ORDER:
        out[(*n)++] = m.uorder;
        if (two) out[(*n)++] = m.vorder;
        break;
    case GL_DOMAIN:
        out[(*n)++] = m.u1;
        out[(*n)++] = m.u2;
        if (two) {
            out[(*n)++] = m.v1;
            out[(*n)++] = m.v2;
        }
        break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}

#define GETMAP_IMPL(Name, T)                                            \
    API void APIENTRY Name(GLenum target, GLenum query, T* v) {         \
        MGL_CTX();                                                      \
        double out[kMaxEvalOrder * kMaxEvalOrder * 4];                  \
        int n;                                                          \
        getMap(ctx, target, query, out, &n);                            \
        for (int k = 0; k < n; k++) v[k] = (T)out[k];                   \
    }
GETMAP_IMPL(glGetMapfv, GLfloat)
GETMAP_IMPL(glGetMapdv, GLdouble)
API void APIENTRY glGetMapiv(GLenum target, GLenum query, GLint* v) {
    MGL_CTX();
    double out[kMaxEvalOrder * kMaxEvalOrder * 4];
    int n;
    getMap(ctx, target, query, out, &n);
    for (int k = 0; k < n; k++) v[k] = (GLint)lround(out[k]);
}

static void mapGrid1(Context* ctx, int un, float u1, float u2) {
    if (un <= 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    cs->grid1N = un;
    cs->grid1U[0] = u1;
    cs->grid1U[1] = u2;
}
static void mapGrid2(Context* ctx, int un, float u1, float u2, int vn, float v1, float v2) {
    if (un <= 0 || vn <= 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    cs->grid2UN = un;
    cs->grid2U[0] = u1;
    cs->grid2U[1] = u2;
    cs->grid2VN = vn;
    cs->grid2V[0] = v1;
    cs->grid2V[1] = v2;
}
API void APIENTRY glMapGrid1f(GLint un, GLfloat u1, GLfloat u2) { MGL_CTX(); RECORD(mapGrid1(ctx, un, u1, u2)); mapGrid1(ctx, un, u1, u2); }
API void APIENTRY glMapGrid1d(GLint un, GLdouble u1, GLdouble u2) { glMapGrid1f(un, (float)u1, (float)u2); }
API void APIENTRY glMapGrid2f(GLint un, GLfloat u1, GLfloat u2, GLint vn, GLfloat v1, GLfloat v2) {
    MGL_CTX();
    RECORD(mapGrid2(ctx, un, u1, u2, vn, v1, v2));
    mapGrid2(ctx, un, u1, u2, vn, v1, v2);
}
API void APIENTRY glMapGrid2d(GLint un, GLdouble u1, GLdouble u2, GLint vn, GLdouble v1, GLdouble v2) {
    glMapGrid2f(un, (float)u1, (float)u2, vn, (float)v1, (float)v2);
}

#define EVAL1(Name, Params, U)                         \
    API void APIENTRY Name Params {                    \
        MGL_CTX();                                     \
        float u_ = (float)(U);                         \
        RECORD(evalEmit(ctx, false, u_, 0));           \
        evalEmit(ctx, false, u_, 0);                   \
    }
#define EVAL2(Name, Params, U, V)                      \
    API void APIENTRY Name Params {                    \
        MGL_CTX();                                     \
        float u_ = (float)(U), v_ = (float)(V);        \
        RECORD(evalEmit(ctx, true, u_, v_));           \
        evalEmit(ctx, true, u_, v_);                   \
    }
EVAL1(glEvalCoord1f, (GLfloat u), u)
EVAL1(glEvalCoord1d, (GLdouble u), u)
EVAL1(glEvalCoord1fv, (const GLfloat* u), u[0])
EVAL1(glEvalCoord1dv, (const GLdouble* u), u[0])
EVAL2(glEvalCoord2f, (GLfloat u, GLfloat v), u, v)
EVAL2(glEvalCoord2d, (GLdouble u, GLdouble v), u, v)
EVAL2(glEvalCoord2fv, (const GLfloat* u), u[0], u[1])
EVAL2(glEvalCoord2dv, (const GLdouble* u), u[0], u[1])

// Grid point i of n over [a, b], exactly b at i == n.
static float gridAt(int i, int n, const float* d) { return i == n ? d[1] : d[0] + i * (d[1] - d[0]) / n; }

static void evalPoint1(Context* ctx, int i) {
    CompatState* cs = compat(ctx);
    evalEmit(ctx, false, gridAt(i, cs->grid1N, cs->grid1U), 0);
}
static void evalPoint2(Context* ctx, int i, int j) {
    CompatState* cs = compat(ctx);
    evalEmit(ctx, true, gridAt(i, cs->grid2UN, cs->grid2U), gridAt(j, cs->grid2VN, cs->grid2V));
}
API void APIENTRY glEvalPoint1(GLint i) { MGL_CTX(); RECORD(evalPoint1(ctx, i)); evalPoint1(ctx, i); }
API void APIENTRY glEvalPoint2(GLint i, GLint j) { MGL_CTX(); RECORD(evalPoint2(ctx, i, j)); evalPoint2(ctx, i, j); }

static void evalMesh1(Context* ctx, GLenum mode, int i1, int i2) {
    if (mode != GL_POINT && mode != GL_LINE) return ctx->setError(GL_INVALID_ENUM);
    immBegin(ctx, mode == GL_POINT ? GL_POINTS : GL_LINE_STRIP);
    for (int i = i1; i <= i2; i++) evalPoint1(ctx, i);
    immEnd(ctx);
}

static void evalMesh2(Context* ctx, GLenum mode, int i1, int i2, int j1, int j2) {
    switch (mode) {
    case GL_FILL:
        for (int j = j1; j < j2; j++) {
            immBegin(ctx, GL_QUAD_STRIP);
            for (int i = i1; i <= i2; i++) {
                evalPoint2(ctx, i, j);
                evalPoint2(ctx, i, j + 1);
            }
            immEnd(ctx);
        }
        break;
    case GL_LINE:
        for (int j = j1; j <= j2; j++) {
            immBegin(ctx, GL_LINE_STRIP);
            for (int i = i1; i <= i2; i++) evalPoint2(ctx, i, j);
            immEnd(ctx);
        }
        for (int i = i1; i <= i2; i++) {
            immBegin(ctx, GL_LINE_STRIP);
            for (int j = j1; j <= j2; j++) evalPoint2(ctx, i, j);
            immEnd(ctx);
        }
        break;
    case GL_POINT:
        immBegin(ctx, GL_POINTS);
        for (int j = j1; j <= j2; j++)
            for (int i = i1; i <= i2; i++) evalPoint2(ctx, i, j);
        immEnd(ctx);
        break;
    default: ctx->setError(GL_INVALID_ENUM);
    }
}
API void APIENTRY glEvalMesh1(GLenum mode, GLint i1, GLint i2) { MGL_CTX(); RECORD(evalMesh1(ctx, mode, i1, i2)); evalMesh1(ctx, mode, i1, i2); }
API void APIENTRY glEvalMesh2(GLenum mode, GLint i1, GLint i2, GLint j1, GLint j2) {
    MGL_CTX();
    RECORD(evalMesh2(ctx, mode, i1, i2, j1, j2));
    evalMesh2(ctx, mode, i1, i2, j1, j2);
}

// glVertexAttrib between Begin and End (compatibility profile): attribute 0 is a vertex, others are recorded
// per vertex. Returns true when the call made a vertex.
bool mglCompatImmAttrib(Context* ctx, GLuint index, float x, float y, float z, float w) {
    CompatState* cs = ctx->compat;
    if (!cs || !cs->inBegin) return false;
    cs->immGeneric |= 1u << index;
    if (index != 0) return false;
    emitVertex(ctx, x, y, z, w);
    return true;
}

// ---------------------------------------------------------------------------
// GL_APPLE_vertex_program_evaluators: evaluator maps for generic vertex attributes, evaluated by
// glEvalCoord / glEvalMesh with the classic maps (attribute 0 produces the vertex).
// ---------------------------------------------------------------------------
static void defineAttribMap(Context* ctx, GLuint index, int size, float u1, float u2, int ustride, int uorder, float v1, float v2,
                            int vstride, int vorder, const std::vector<float>& src, bool two) {
    if (index >= kMaxVertexAttribs || size < 1 || size > 4) return ctx->setError(GL_INVALID_VALUE);
    if (u1 == u2 || (two && v1 == v2)) return ctx->setError(GL_INVALID_VALUE);
    if (uorder < 1 || uorder > kMaxEvalOrder || (two && (vorder < 1 || vorder > kMaxEvalOrder))) return ctx->setError(GL_INVALID_VALUE);
    if (ustride < size || (two && vstride < size)) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    EvalMap& m = two ? cs->attribMap2[index] : cs->attribMap1[index];
    (two ? cs->attribMap2Size : cs->attribMap1Size)[index] = size;
    m.u1 = u1;
    m.u2 = u2;
    m.v1 = v1;
    m.v2 = v2;
    m.uorder = uorder;
    m.vorder = two ? vorder : 1;
    m.pts.resize((size_t)m.uorder * m.vorder * size);
    for (int a = 0; a < m.uorder; a++)
        for (int b = 0; b < m.vorder; b++)
            for (int k = 0; k < size; k++) m.pts[((size_t)a * m.vorder + b) * size + k] = src[(size_t)a * ustride + (size_t)b * vstride + k];
}

#define MAPATTRIB1_IMPL(Name, T)                                                                                  \
    API void APIENTRY Name(GLuint index, GLuint size, T u1, T u2, GLint stride, GLint order, const T* points) {   \
        MGL_CTX();                                                                                                \
        std::vector<float> pts = mapPoints(points, stride, order, 0, 1, (int)std::min(size, 4u));                 \
        float a = (float)u1, b = (float)u2;                                                                       \
        RECORD(defineAttribMap(ctx, index, (int)size, a, b, stride, order, 0, 1, 0, 1, pts, false));              \
        defineAttribMap(ctx, index, (int)size, a, b, stride, order, 0, 1, 0, 1, pts, false);                      \
    }
MAPATTRIB1_IMPL(glMapVertexAttrib1fAPPLE, GLfloat)
MAPATTRIB1_IMPL(glMapVertexAttrib1dAPPLE, GLdouble)

#define MAPATTRIB2_IMPL(Name, T)                                                                                        \
    API void APIENTRY Name(GLuint index, GLuint size, T u1, T u2, GLint ustride, GLint uorder, T v1, T v2, GLint vstride, \
                           GLint vorder, const T* points) {                                                             \
        MGL_CTX();                                                                                                      \
        std::vector<float> pts = mapPoints(points, ustride, uorder, vstride, vorder, (int)std::min(size, 4u));          \
        float a = (float)u1, b = (float)u2, c = (float)v1, d = (float)v2;                                               \
        RECORD(defineAttribMap(ctx, index, (int)size, a, b, ustride, uorder, c, d, vstride, vorder, pts, true));        \
        defineAttribMap(ctx, index, (int)size, a, b, ustride, uorder, c, d, vstride, vorder, pts, true);                \
    }
MAPATTRIB2_IMPL(glMapVertexAttrib2fAPPLE, GLfloat)
MAPATTRIB2_IMPL(glMapVertexAttrib2dAPPLE, GLdouble)

static bool* attribMapEnable(Context* ctx, GLuint index, GLenum pname) {
    if (index >= kMaxVertexAttribs) return ctx->setError(GL_INVALID_VALUE), nullptr;
    CompatState* cs = compat(ctx);
    if (pname == GL_VERTEX_ATTRIB_MAP1_APPLE) return &cs->attribMap1On[index];
    if (pname == GL_VERTEX_ATTRIB_MAP2_APPLE) return &cs->attribMap2On[index];
    ctx->setError(GL_INVALID_ENUM);
    return nullptr;
}
API void APIENTRY glEnableVertexAttribAPPLE(GLuint index, GLenum pname) {
    MGL_CTX();
    if (bool* b = attribMapEnable(ctx, index, pname)) *b = true;
}
API void APIENTRY glDisableVertexAttribAPPLE(GLuint index, GLenum pname) {
    MGL_CTX();
    if (bool* b = attribMapEnable(ctx, index, pname)) *b = false;
}
API GLboolean APIENTRY glIsVertexAttribEnabledAPPLE(GLuint index, GLenum pname) {
    MGL_CTX_RET(GL_FALSE);
    bool* b = attribMapEnable(ctx, index, pname);
    return b && *b;
}

// glGetVertexAttrib*(index, VERTEX_ATTRIB_MAP{1,2}_{SIZE,COEFF,ORDER,DOMAIN}_APPLE); false for other pnames.
bool mglCompatGetAttribMap(Context* ctx, GLuint index, GLenum pname, double* out, int* n) {
    bool two;
    switch (pname) {
    case GL_VERTEX_ATTRIB_MAP1_SIZE_APPLE: case GL_VERTEX_ATTRIB_MAP1_COEFF_APPLE: case GL_VERTEX_ATTRIB_MAP1_ORDER_APPLE:
    case GL_VERTEX_ATTRIB_MAP1_DOMAIN_APPLE:
        two = false;
        break;
    case GL_VERTEX_ATTRIB_MAP2_SIZE_APPLE: case GL_VERTEX_ATTRIB_MAP2_COEFF_APPLE: case GL_VERTEX_ATTRIB_MAP2_ORDER_APPLE:
    case GL_VERTEX_ATTRIB_MAP2_DOMAIN_APPLE:
        two = true;
        break;
    default: return false;
    }
    *n = 0;
    if (index >= kMaxVertexAttribs) {
        ctx->setError(GL_INVALID_VALUE);
        return true;
    }
    CompatState* cs = compat(ctx);
    const EvalMap& m = two ? cs->attribMap2[index] : cs->attribMap1[index];
    int size = two ? cs->attribMap2Size[index] : cs->attribMap1Size[index];
    switch (pname) {
    case GL_VERTEX_ATTRIB_MAP1_SIZE_APPLE: case GL_VERTEX_ATTRIB_MAP2_SIZE_APPLE: out[(*n)++] = size; break;
    case GL_VERTEX_ATTRIB_MAP1_COEFF_APPLE: case GL_VERTEX_ATTRIB_MAP2_COEFF_APPLE:
        if (size)
            for (float f : m.pts) out[(*n)++] = f;
        break;
    case GL_VERTEX_ATTRIB_MAP1_ORDER_APPLE: case GL_VERTEX_ATTRIB_MAP2_ORDER_APPLE:
        out[(*n)++] = size ? m.uorder : 0;
        if (two) out[(*n)++] = size ? m.vorder : 0;
        break;
    default:  // domain
        out[(*n)++] = m.u1;
        out[(*n)++] = m.u2;
        if (two) {
            out[(*n)++] = m.v1;
            out[(*n)++] = m.v2;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// GL_ARB_vertex_type_2_10_10_10_rev: packed immediate-mode values (positions and texture coordinates
// unnormalized, normals and colors normalized)
// ---------------------------------------------------------------------------
bool mglDecodePacked(GLenum type, int n, bool normalized, GLuint value, float* c);  // api_vertex.mm

static bool packedType(Context* ctx, GLenum type, bool allowFloat11) {
    if (type == GL_INT_2_10_10_10_REV || type == GL_UNSIGNED_INT_2_10_10_10_REV) return true;
    if (allowFloat11 && type == GL_UNSIGNED_INT_10F_11F_11F_REV) return true;
    ctx->setError(GL_INVALID_ENUM);
    return false;
}

#define PACKED_IMPL(Suffix, N, Norm, Float11, Call)                                         \
    API void APIENTRY gl##Suffix##ui(GLenum type, GLuint value) {                          \
        MGL_CTX();                                                                         \
        float c[4];                                                                        \
        if (!packedType(ctx, type, Float11) || !mglDecodePacked(type, N, Norm, value, c))  \
            return;                                                                        \
        Call;                                                                              \
    }                                                                                      \
    API void APIENTRY gl##Suffix##uiv(GLenum type, const GLuint* value) { gl##Suffix##ui(type, *value); }
PACKED_IMPL(VertexP2, 2, false, false, glVertex4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(VertexP3, 3, false, false, glVertex4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(VertexP4, 4, false, false, glVertex4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(TexCoordP1, 1, false, false, glTexCoord4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(TexCoordP2, 2, false, false, glTexCoord4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(TexCoordP3, 3, false, false, glTexCoord4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(TexCoordP4, 4, false, false, glTexCoord4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(NormalP3, 3, true, false, glNormal3f(c[0], c[1], c[2]))
PACKED_IMPL(ColorP3, 3, true, false, glColor4f(c[0], c[1], c[2], 1.0f))
PACKED_IMPL(ColorP4, 4, true, false, glColor4f(c[0], c[1], c[2], c[3]))
PACKED_IMPL(SecondaryColorP3, 3, true, false, glSecondaryColor3f(c[0], c[1], c[2]))

#define MTPACKED_IMPL(N)                                                                            \
    API void APIENTRY glMultiTexCoordP##N##ui(GLenum texture, GLenum type, GLuint value) {         \
        MGL_CTX();                                                                                 \
        float c[4];                                                                                \
        if (!packedType(ctx, type, false) || !mglDecodePacked(type, N, false, value, c)) return;   \
        glMultiTexCoord4f(texture, c[0], c[1], c[2], c[3]);                                        \
    }                                                                                              \
    API void APIENTRY glMultiTexCoordP##N##uiv(GLenum texture, GLenum type, const GLuint* value) { \
        glMultiTexCoordP##N##ui(texture, type, *value);                                            \
    }
MTPACKED_IMPL(1)
MTPACKED_IMPL(2)
MTPACKED_IMPL(3)
MTPACKED_IMPL(4)

// GL_ARB_robustness / GL 4.5: glGetMap with the size of the output buffer (bytes).
template <typename T>
static void getnMap(GLenum target, GLenum query, GLsizei bufSize, T* v) {
    MGL_CTX();
    double out[kMaxEvalOrder * kMaxEvalOrder * 4];
    int n;
    getMap(ctx, target, query, out, &n);
    if ((size_t)n * sizeof(T) > (size_t)std::max(bufSize, 0)) return ctx->setError(GL_INVALID_OPERATION);
    for (int k = 0; k < n; k++) v[k] = std::is_integral_v<T> ? (T)lround(out[k]) : (T)out[k];
}
API void APIENTRY glGetnMapdv(GLenum target, GLenum query, GLsizei bufSize, GLdouble* v) { getnMap(target, query, bufSize, v); }
API void APIENTRY glGetnMapfv(GLenum target, GLenum query, GLsizei bufSize, GLfloat* v) { getnMap(target, query, bufSize, v); }
API void APIENTRY glGetnMapiv(GLenum target, GLenum query, GLsizei bufSize, GLint* v) { getnMap(target, query, bufSize, v); }
API void APIENTRY glGetnMapdvARB(GLenum target, GLenum query, GLsizei bufSize, GLdouble* v) { getnMap(target, query, bufSize, v); }
API void APIENTRY glGetnMapfvARB(GLenum target, GLenum query, GLsizei bufSize, GLfloat* v) { getnMap(target, query, bufSize, v); }
API void APIENTRY glGetnMapivARB(GLenum target, GLenum query, GLsizei bufSize, GLint* v) { getnMap(target, query, bufSize, v); }

// ---------------------------------------------------------------------------
// GL_NV_primitive_restart: the enable and index are core primitive restart's; glPrimitiveRestartNV ends the
// immediate-mode primitive and starts another of the same mode.
// ---------------------------------------------------------------------------
API void APIENTRY glPrimitiveRestartIndexNV(GLuint index) { glPrimitiveRestartIndex(index); }

static void primitiveRestart(Context* ctx) {
    CompatState* cs = compat(ctx);
    if (!cs->inBegin) return ctx->setError(GL_INVALID_OPERATION);
    GLenum mode = cs->beginMode;
    immEnd(ctx);
    immBegin(ctx, mode);
}
API void APIENTRY glPrimitiveRestartNV(void) {
    MGL_CTX();
    RECORD(primitiveRestart(ctx));
    primitiveRestart(ctx);
}

// ---------------------------------------------------------------------------
// Immediate-mode type variants (generated from gl.xml): converted to the float entry points; color and
// normal integers normalize as GL 4.6 compatibility §10.2.1 / Table 10.3 specifies.
// ---------------------------------------------------------------------------
API void APIENTRY glColor3bv(const GLbyte * v) { glColor4f(bn(v[0]), bn(v[1]), bn(v[2]), 1.0f); }
API void APIENTRY glColor3iv(const GLint * v) { glColor4f(in_(v[0]), in_(v[1]), in_(v[2]), 1.0f); }
API void APIENTRY glColor3sv(const GLshort * v) { glColor4f(sn(v[0]), sn(v[1]), sn(v[2]), 1.0f); }
API void APIENTRY glColor3uiv(const GLuint * v) { glColor4f(ui(v[0]), ui(v[1]), ui(v[2]), 1.0f); }
API void APIENTRY glColor3usv(const GLushort * v) { glColor4f(us(v[0]), us(v[1]), us(v[2]), 1.0f); }
API void APIENTRY glColor4bv(const GLbyte * v) { glColor4f(bn(v[0]), bn(v[1]), bn(v[2]), bn(v[3])); }
API void APIENTRY glColor4iv(const GLint * v) { glColor4f(in_(v[0]), in_(v[1]), in_(v[2]), in_(v[3])); }
API void APIENTRY glColor4sv(const GLshort * v) { glColor4f(sn(v[0]), sn(v[1]), sn(v[2]), sn(v[3])); }
API void APIENTRY glColor4uiv(const GLuint * v) { glColor4f(ui(v[0]), ui(v[1]), ui(v[2]), ui(v[3])); }
API void APIENTRY glColor4usv(const GLushort * v) { glColor4f(us(v[0]), us(v[1]), us(v[2]), us(v[3])); }
API void APIENTRY glFogCoorddv(const GLdouble * coord) { glFogCoordf((GLfloat)coord[0]); }
API void APIENTRY glMultiTexCoord1d(GLenum target, GLdouble s) { glMultiTexCoord4f(target, (GLfloat)s, 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1dv(GLenum target, const GLdouble * v) { glMultiTexCoord4f(target, (GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1fv(GLenum target, const GLfloat * v) { glMultiTexCoord4f(target, (GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1i(GLenum target, GLint s) { glMultiTexCoord4f(target, (GLfloat)s, 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1iv(GLenum target, const GLint * v) { glMultiTexCoord4f(target, (GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1s(GLenum target, GLshort s) { glMultiTexCoord4f(target, (GLfloat)s, 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1sv(GLenum target, const GLshort * v) { glMultiTexCoord4f(target, (GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord2dv(GLenum target, const GLdouble * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord2iv(GLenum target, const GLint * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord2sv(GLenum target, const GLshort * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord3d(GLenum target, GLdouble s, GLdouble t, GLdouble r) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, 1.0f); }
API void APIENTRY glMultiTexCoord3dv(GLenum target, const GLdouble * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glMultiTexCoord3fv(GLenum target, const GLfloat * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glMultiTexCoord3i(GLenum target, GLint s, GLint t, GLint r) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, 1.0f); }
API void APIENTRY glMultiTexCoord3iv(GLenum target, const GLint * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glMultiTexCoord3s(GLenum target, GLshort s, GLshort t, GLshort r) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, 1.0f); }
API void APIENTRY glMultiTexCoord3sv(GLenum target, const GLshort * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glMultiTexCoord4d(GLenum target, GLdouble s, GLdouble t, GLdouble r, GLdouble q) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, (GLfloat)q); }
API void APIENTRY glMultiTexCoord4dv(GLenum target, const GLdouble * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glMultiTexCoord4i(GLenum target, GLint s, GLint t, GLint r, GLint q) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, (GLfloat)q); }
API void APIENTRY glMultiTexCoord4iv(GLenum target, const GLint * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glMultiTexCoord4s(GLenum target, GLshort s, GLshort t, GLshort r, GLshort q) { glMultiTexCoord4f(target, (GLfloat)s, (GLfloat)t, (GLfloat)r, (GLfloat)q); }
API void APIENTRY glMultiTexCoord4sv(GLenum target, const GLshort * v) { glMultiTexCoord4f(target, (GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glNormal3bv(const GLbyte * v) { glNormal3f(bn(v[0]), bn(v[1]), bn(v[2])); }
API void APIENTRY glNormal3iv(const GLint * v) { glNormal3f(in_(v[0]), in_(v[1]), in_(v[2])); }
API void APIENTRY glNormal3sv(const GLshort * v) { glNormal3f(sn(v[0]), sn(v[1]), sn(v[2])); }
API void APIENTRY glSecondaryColor3b(GLbyte red, GLbyte green, GLbyte blue) { glSecondaryColor3f(bn(red), bn(green), bn(blue)); }
API void APIENTRY glSecondaryColor3bv(const GLbyte * v) { glSecondaryColor3f(bn(v[0]), bn(v[1]), bn(v[2])); }
API void APIENTRY glSecondaryColor3d(GLdouble red, GLdouble green, GLdouble blue) { glSecondaryColor3f((GLfloat)red, (GLfloat)green, (GLfloat)blue); }
API void APIENTRY glSecondaryColor3dv(const GLdouble * v) { glSecondaryColor3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]); }
API void APIENTRY glSecondaryColor3i(GLint red, GLint green, GLint blue) { glSecondaryColor3f(in_(red), in_(green), in_(blue)); }
API void APIENTRY glSecondaryColor3iv(const GLint * v) { glSecondaryColor3f(in_(v[0]), in_(v[1]), in_(v[2])); }
API void APIENTRY glSecondaryColor3s(GLshort red, GLshort green, GLshort blue) { glSecondaryColor3f(sn(red), sn(green), sn(blue)); }
API void APIENTRY glSecondaryColor3sv(const GLshort * v) { glSecondaryColor3f(sn(v[0]), sn(v[1]), sn(v[2])); }
API void APIENTRY glSecondaryColor3ui(GLuint red, GLuint green, GLuint blue) { glSecondaryColor3f(ui(red), ui(green), ui(blue)); }
API void APIENTRY glSecondaryColor3uiv(const GLuint * v) { glSecondaryColor3f(ui(v[0]), ui(v[1]), ui(v[2])); }
API void APIENTRY glSecondaryColor3us(GLushort red, GLushort green, GLushort blue) { glSecondaryColor3f(us(red), us(green), us(blue)); }
API void APIENTRY glSecondaryColor3usv(const GLushort * v) { glSecondaryColor3f(us(v[0]), us(v[1]), us(v[2])); }
API void APIENTRY glTexCoord1dv(const GLdouble * v) { glTexCoord4f((GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord1fv(const GLfloat * v) { glTexCoord4f((GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord1iv(const GLint * v) { glTexCoord4f((GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord1s(GLshort s) { glTexCoord4f((GLfloat)s, 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord1sv(const GLshort * v) { glTexCoord4f((GLfloat)v[0], 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord2iv(const GLint * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glTexCoord2sv(const GLshort * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glTexCoord3dv(const GLdouble * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glTexCoord3i(GLint s, GLint t, GLint r) { glTexCoord4f((GLfloat)s, (GLfloat)t, (GLfloat)r, 1.0f); }
API void APIENTRY glTexCoord3iv(const GLint * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glTexCoord3s(GLshort s, GLshort t, GLshort r) { glTexCoord4f((GLfloat)s, (GLfloat)t, (GLfloat)r, 1.0f); }
API void APIENTRY glTexCoord3sv(const GLshort * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glTexCoord4dv(const GLdouble * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glTexCoord4i(GLint s, GLint t, GLint r, GLint q) { glTexCoord4f((GLfloat)s, (GLfloat)t, (GLfloat)r, (GLfloat)q); }
API void APIENTRY glTexCoord4iv(const GLint * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glTexCoord4s(GLshort s, GLshort t, GLshort r, GLshort q) { glTexCoord4f((GLfloat)s, (GLfloat)t, (GLfloat)r, (GLfloat)q); }
API void APIENTRY glTexCoord4sv(const GLshort * v) { glTexCoord4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glVertex2sv(const GLshort * v) { glVertex4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glVertex3sv(const GLshort * v) { glVertex4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glVertex4dv(const GLdouble * v) { glVertex4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glVertex4i(GLint x, GLint y, GLint z, GLint w) { glVertex4f((GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
API void APIENTRY glVertex4iv(const GLint * v) { glVertex4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glVertex4s(GLshort x, GLshort y, GLshort z, GLshort w) { glVertex4f((GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
API void APIENTRY glVertex4sv(const GLshort * v) { glVertex4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }

// GL 1.4 / ARB_point_parameters (compatibility): point size bounds and distance attenuation, applied by
// fixed-function vertex shaders. Returns false for other parameters.
bool mglCompatPointParam(Context* ctx, GLenum pname, const GLfloat* v) {
    CompatState* cs = compat(ctx);
    switch (pname) {
    case GL_POINT_SIZE_MIN:
    case GL_POINT_SIZE_MAX:
        if (v[0] < 0) return ctx->setError(GL_INVALID_VALUE), true;
        (pname == GL_POINT_SIZE_MIN ? cs->pointSizeMin : cs->pointSizeMax) = v[0];
        return true;
    case GL_POINT_DISTANCE_ATTENUATION: memcpy(cs->pointAtt, v, 12); return true;
    default: return false;
    }
}

// ---------------------------------------------------------------------------
// GL_EXT_compiled_vertex_array: locking promises the arrays won't change, which lets drivers transform them
// once; draws here read client arrays as they are, so the lock is only recorded.
// ---------------------------------------------------------------------------
API void APIENTRY glLockArraysEXT(GLint first, GLsizei count) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    if (first < 0 || count <= 0) return ctx->setError(GL_INVALID_VALUE);
    if (cs->lockCount) return ctx->setError(GL_INVALID_OPERATION);  // already locked
    cs->lockFirst = first;
    cs->lockCount = count;
}

API void APIENTRY glUnlockArraysEXT(void) {
    MGL_CTX();
    CompatState* cs = compat(ctx);
    if (!cs->lockCount) return ctx->setError(GL_INVALID_OPERATION);
    cs->lockFirst = 0;
    cs->lockCount = 0;
}

// ---------------------------------------------------------------------------
// glArrayElement: the vertex at index i of the enabled arrays (fixed-function and generic), inside Begin/End.
// ---------------------------------------------------------------------------
static float halfBits(uint16_t h) {
    uint32_t sign = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v = e == 0 ? ldexpf((float)m, -24) : e == 31 ? (m ? NAN : INFINITY) : ldexpf(1.0f + m / 1024.0f, (int)e - 15);
    return sign ? -v : v;
}

// Reads `size` components of `type` at p (GL_BGRA size: four, blue first); normalized integers map to [0,1] / [-1,1].
static void readElement(const uint8_t* p, GLint size, GLenum type, bool normalized, float* out) {
    int n = size == GL_BGRA ? 4 : size;
    out[0] = out[1] = out[2] = 0;
    out[3] = 1;
    for (int k = 0; k < n; k++) {
        float v = 0;
        switch (type) {
        case GL_BYTE: { int8_t x; memcpy(&x, p + k, 1); v = normalized ? std::max(-1.0f, x / 127.0f) : x; break; }
        case GL_UNSIGNED_BYTE: v = normalized ? p[k] / 255.0f : p[k]; break;
        case GL_SHORT: { int16_t x; memcpy(&x, p + 2 * k, 2); v = normalized ? std::max(-1.0f, x / 32767.0f) : x; break; }
        case GL_UNSIGNED_SHORT: { uint16_t x; memcpy(&x, p + 2 * k, 2); v = normalized ? x / 65535.0f : x; break; }
        case GL_INT: { int32_t x; memcpy(&x, p + 4 * k, 4); v = normalized ? std::max(-1.0f, (float)(x / 2147483647.0)) : (float)x; break; }
        case GL_UNSIGNED_INT: { uint32_t x; memcpy(&x, p + 4 * k, 4); v = normalized ? (float)(x / 4294967295.0) : (float)x; break; }
        case GL_FLOAT: memcpy(&v, p + 4 * k, 4); break;
        case GL_DOUBLE: { double x; memcpy(&x, p + 8 * k, 8); v = (float)x; break; }
        case GL_HALF_FLOAT: { uint16_t x; memcpy(&x, p + 2 * k, 2); v = halfBits(x); break; }
        default: break;
        }
        out[k] = v;
    }
    if (size == GL_BGRA) std::swap(out[0], out[2]);
}

static int typeBytes(GLenum type) {
    switch (type) {
    case GL_BYTE: case GL_UNSIGNED_BYTE: return 1;
    case GL_SHORT: case GL_UNSIGNED_SHORT: case GL_HALF_FLOAT: return 2;
    case GL_DOUBLE: return 8;
    default: return 4;
    }
}

static const uint8_t* clientElement(const ClientArray& a, GLint i) {
    const uint8_t* base = a.buffer ? (a.buffer->cpu ? a.buffer->cpu + (uintptr_t)a.pointer : nullptr) : (const uint8_t*)a.pointer;
    if (!base) return nullptr;
    int comps = a.size == GL_BGRA ? 4 : a.size;
    size_t stride = a.stride ? (size_t)a.stride : (size_t)comps * typeBytes(a.type);
    return base + (size_t)i * stride;
}

static void arrayElement(Context* ctx, GLint i) {
    CompatState* cs = compat(ctx);
    for (int a = CA_COUNT - 1; a >= 0; a--) {  // the vertex last: it emits
        if (a == CA_VERTEX) continue;
        const ClientArray& arr = cs->arrays[a];
        if (!arr.enabled) continue;
        const uint8_t* p = clientElement(arr, i);
        if (!p) continue;
        bool norm = a == CA_COLOR || a == CA_SECONDARY || (a == CA_NORMAL && arr.type != GL_FLOAT && arr.type != GL_DOUBLE);
        float v[4];
        readElement(p, arr.size, arr.type, norm, v);
        if (a == CA_NORMAL) v[3] = 1;
        memcpy(cs->current[a], v, 16);
    }
    // generic attribute arrays of the bound vertex array object (client memory or buffers)
    VertexArray* vao = ctx->vao.get();
    for (int g = kMaxVertexAttribs - 1; g >= 1; g--) {
        const VertexAttrib& va = vao->attribs[g];
        if (!va.enabled) continue;
        const VertexBinding& vb = vao->bindings[va.binding];
        const uint8_t* base = va.clientPointer ? (const uint8_t*)va.clientPointer
                              : vb.buffer && vb.buffer->cpu ? vb.buffer->cpu + vb.offset + va.relativeOffset
                                                            : nullptr;
        if (!base) continue;
        int comps = va.size == GL_BGRA ? 4 : va.size;
        size_t stride = vb.stride ? (size_t)vb.stride : (size_t)comps * typeBytes(va.type);
        float v[4];
        readElement(base + (size_t)i * stride, va.size, va.type, va.normalized, v);
        glVertexAttrib4f((GLuint)g, v[0], v[1], v[2], v[3]);
    }
    if (cs->arrays[CA_VERTEX].enabled) {
        if (const uint8_t* p = clientElement(cs->arrays[CA_VERTEX], i)) {
            float v[4];
            readElement(p, cs->arrays[CA_VERTEX].size, cs->arrays[CA_VERTEX].type, false, v);
            emitVertex(ctx, v[0], v[1], v[2], v[3]);
        }
    }
}

API void APIENTRY glArrayElement(GLint i) {
    MGL_CTX();
    if (i < 0) return ctx->setError(GL_INVALID_VALUE);
    CompatState* cs = compat(ctx);
    if (cs->compilingList && cs->callDepth == 0) {
        // Display lists hold the array values (client data is read when the list is compiled).
        struct Value {
            int attr;
            float v[4];
        };
        std::vector<Value> vals;
        for (int a = 0; a < CA_COUNT; a++) {
            const ClientArray& arr = cs->arrays[a];
            const uint8_t* p = arr.enabled ? clientElement(arr, i) : nullptr;
            if (!p) continue;
            Value val{a, {}};
            bool norm = a == CA_COLOR || a == CA_SECONDARY || (a == CA_NORMAL && arr.type != GL_FLOAT && arr.type != GL_DOUBLE);
            readElement(p, arr.size, arr.type, norm, val.v);
            vals.push_back(val);
        }
        cs->listBuf.push_back([vals]() {
            Context* c = currentContext();
            CompatState* s = compat(c);
            const float* pos = nullptr;
            for (const Value& val : vals) {
                if (val.attr == CA_VERTEX) pos = val.v;
                else memcpy(s->current[val.attr], val.v, 16);
            }
            if (pos) emitVertex(c, pos[0], pos[1], pos[2], pos[3]);
        });
        if (cs->listMode == GL_COMPILE) return;
    }
    arrayElement(ctx, i);
}

// ---------------------------------------------------------------------------
// GL_EXT_vertex_array: GL 1.1 vertex arrays before they were core (the pointer calls carry a count, which only
// bounds the array); GL_EXT_texture_object: texture objects likewise.
// ---------------------------------------------------------------------------
API void APIENTRY glVertexPointerEXT(GLint size, GLenum type, GLsizei stride, GLsizei, const void* p) { glVertexPointer(size, type, stride, p); }
API void APIENTRY glNormalPointerEXT(GLenum type, GLsizei stride, GLsizei, const void* p) { glNormalPointer(type, stride, p); }
API void APIENTRY glColorPointerEXT(GLint size, GLenum type, GLsizei stride, GLsizei, const void* p) { glColorPointer(size, type, stride, p); }
API void APIENTRY glTexCoordPointerEXT(GLint size, GLenum type, GLsizei stride, GLsizei, const void* p) { glTexCoordPointer(size, type, stride, p); }
API void APIENTRY glIndexPointerEXT(GLenum type, GLsizei stride, GLsizei, const void* p) { glIndexPointer(type, stride, p); }
API void APIENTRY glEdgeFlagPointerEXT(GLsizei stride, GLsizei, const GLboolean* p) { glEdgeFlagPointer(stride, p); }

API void APIENTRY glGenTexturesEXT(GLsizei n, GLuint* textures) { glGenTextures(n, textures); }
API void APIENTRY glDeleteTexturesEXT(GLsizei n, const GLuint* textures) { glDeleteTextures(n, textures); }
API GLboolean APIENTRY glIsTextureEXT(GLuint texture) { return glIsTexture(texture); }
API GLboolean APIENTRY glAreTexturesResidentEXT(GLsizei n, const GLuint* textures, GLboolean* residences) {
    return glAreTexturesResident(n, textures, residences);
}

// ---------------------------------------------------------------------------
// GL_NV_half_float: half-precision immediate-mode values (generated from gl.xml; the vertex weight one
// belongs to EXT_vertex_weighting, which isn't supported). Arrays of HALF_FLOAT_NV are core half floats.
// ---------------------------------------------------------------------------
static float h(GLhalfNV v) { return halfBits(v); }
API void APIENTRY glVertex2hNV(GLhalfNV x, GLhalfNV y) { glVertex4f(h(x), h(y), 0.0f, 1.0f); }
API void APIENTRY glVertex2hvNV(const GLhalfNV * v) { glVertex4f(h(v[0]), h(v[1]), 0.0f, 1.0f); }
API void APIENTRY glVertex3hNV(GLhalfNV x, GLhalfNV y, GLhalfNV z) { glVertex4f(h(x), h(y), h(z), 1.0f); }
API void APIENTRY glVertex3hvNV(const GLhalfNV * v) { glVertex4f(h(v[0]), h(v[1]), h(v[2]), 1.0f); }
API void APIENTRY glVertex4hNV(GLhalfNV x, GLhalfNV y, GLhalfNV z, GLhalfNV w) { glVertex4f(h(x), h(y), h(z), h(w)); }
API void APIENTRY glVertex4hvNV(const GLhalfNV * v) { glVertex4f(h(v[0]), h(v[1]), h(v[2]), h(v[3])); }
API void APIENTRY glNormal3hNV(GLhalfNV nx, GLhalfNV ny, GLhalfNV nz) { glNormal3f(h(nx), h(ny), h(nz)); }
API void APIENTRY glNormal3hvNV(const GLhalfNV * v) { glNormal3f(h(v[0]), h(v[1]), h(v[2])); }
API void APIENTRY glColor3hNV(GLhalfNV red, GLhalfNV green, GLhalfNV blue) { glColor4f(h(red), h(green), h(blue), 1.0f); }
API void APIENTRY glColor3hvNV(const GLhalfNV * v) { glColor4f(h(v[0]), h(v[1]), h(v[2]), 1.0f); }
API void APIENTRY glColor4hNV(GLhalfNV red, GLhalfNV green, GLhalfNV blue, GLhalfNV alpha) { glColor4f(h(red), h(green), h(blue), h(alpha)); }
API void APIENTRY glColor4hvNV(const GLhalfNV * v) { glColor4f(h(v[0]), h(v[1]), h(v[2]), h(v[3])); }
API void APIENTRY glTexCoord1hNV(GLhalfNV s) { glTexCoord4f(h(s), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord1hvNV(const GLhalfNV * v) { glTexCoord4f(h(v[0]), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glTexCoord2hNV(GLhalfNV s, GLhalfNV t) { glTexCoord4f(h(s), h(t), 0.0f, 1.0f); }
API void APIENTRY glTexCoord2hvNV(const GLhalfNV * v) { glTexCoord4f(h(v[0]), h(v[1]), 0.0f, 1.0f); }
API void APIENTRY glTexCoord3hNV(GLhalfNV s, GLhalfNV t, GLhalfNV r) { glTexCoord4f(h(s), h(t), h(r), 1.0f); }
API void APIENTRY glTexCoord3hvNV(const GLhalfNV * v) { glTexCoord4f(h(v[0]), h(v[1]), h(v[2]), 1.0f); }
API void APIENTRY glTexCoord4hNV(GLhalfNV s, GLhalfNV t, GLhalfNV r, GLhalfNV q) { glTexCoord4f(h(s), h(t), h(r), h(q)); }
API void APIENTRY glTexCoord4hvNV(const GLhalfNV * v) { glTexCoord4f(h(v[0]), h(v[1]), h(v[2]), h(v[3])); }
API void APIENTRY glMultiTexCoord1hNV(GLenum target, GLhalfNV s) { glMultiTexCoord4f(target, h(s), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord1hvNV(GLenum target, const GLhalfNV * v) { glMultiTexCoord4f(target, h(v[0]), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord2hNV(GLenum target, GLhalfNV s, GLhalfNV t) { glMultiTexCoord4f(target, h(s), h(t), 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord2hvNV(GLenum target, const GLhalfNV * v) { glMultiTexCoord4f(target, h(v[0]), h(v[1]), 0.0f, 1.0f); }
API void APIENTRY glMultiTexCoord3hNV(GLenum target, GLhalfNV s, GLhalfNV t, GLhalfNV r) { glMultiTexCoord4f(target, h(s), h(t), h(r), 1.0f); }
API void APIENTRY glMultiTexCoord3hvNV(GLenum target, const GLhalfNV * v) { glMultiTexCoord4f(target, h(v[0]), h(v[1]), h(v[2]), 1.0f); }
API void APIENTRY glMultiTexCoord4hNV(GLenum target, GLhalfNV s, GLhalfNV t, GLhalfNV r, GLhalfNV q) { glMultiTexCoord4f(target, h(s), h(t), h(r), h(q)); }
API void APIENTRY glMultiTexCoord4hvNV(GLenum target, const GLhalfNV * v) { glMultiTexCoord4f(target, h(v[0]), h(v[1]), h(v[2]), h(v[3])); }
API void APIENTRY glVertexAttrib1hNV(GLuint index, GLhalfNV x) { glVertexAttrib4f(index, h(x), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glVertexAttrib1hvNV(GLuint index, const GLhalfNV * v) { glVertexAttrib4f(index, h(v[0]), 0.0f, 0.0f, 1.0f); }
API void APIENTRY glVertexAttrib2hNV(GLuint index, GLhalfNV x, GLhalfNV y) { glVertexAttrib4f(index, h(x), h(y), 0.0f, 1.0f); }
API void APIENTRY glVertexAttrib2hvNV(GLuint index, const GLhalfNV * v) { glVertexAttrib4f(index, h(v[0]), h(v[1]), 0.0f, 1.0f); }
API void APIENTRY glVertexAttrib3hNV(GLuint index, GLhalfNV x, GLhalfNV y, GLhalfNV z) { glVertexAttrib4f(index, h(x), h(y), h(z), 1.0f); }
API void APIENTRY glVertexAttrib3hvNV(GLuint index, const GLhalfNV * v) { glVertexAttrib4f(index, h(v[0]), h(v[1]), h(v[2]), 1.0f); }
API void APIENTRY glVertexAttrib4hNV(GLuint index, GLhalfNV x, GLhalfNV y, GLhalfNV z, GLhalfNV w) { glVertexAttrib4f(index, h(x), h(y), h(z), h(w)); }
API void APIENTRY glVertexAttrib4hvNV(GLuint index, const GLhalfNV * v) { glVertexAttrib4f(index, h(v[0]), h(v[1]), h(v[2]), h(v[3])); }
API void APIENTRY glVertexAttribs1hvNV(GLuint index, GLsizei n, const GLhalfNV * v) {
    for (GLsizei i = 0; i < n; i++) glVertexAttrib1hvNV(index + (GLuint)i, v + (size_t)i * 1);
}
API void APIENTRY glVertexAttribs2hvNV(GLuint index, GLsizei n, const GLhalfNV * v) {
    for (GLsizei i = 0; i < n; i++) glVertexAttrib2hvNV(index + (GLuint)i, v + (size_t)i * 2);
}
API void APIENTRY glVertexAttribs3hvNV(GLuint index, GLsizei n, const GLhalfNV * v) {
    for (GLsizei i = 0; i < n; i++) glVertexAttrib3hvNV(index + (GLuint)i, v + (size_t)i * 3);
}
API void APIENTRY glVertexAttribs4hvNV(GLuint index, GLsizei n, const GLhalfNV * v) {
    for (GLsizei i = 0; i < n; i++) glVertexAttrib4hvNV(index + (GLuint)i, v + (size_t)i * 4);
}
API void APIENTRY glFogCoordhNV(GLhalfNV fog) { glFogCoordf(h(fog)); }
API void APIENTRY glFogCoordhvNV(const GLhalfNV * fog) { glFogCoordf(h(fog[0])); }
API void APIENTRY glSecondaryColor3hNV(GLhalfNV red, GLhalfNV green, GLhalfNV blue) { glSecondaryColor3f(h(red), h(green), h(blue)); }
API void APIENTRY glSecondaryColor3hvNV(const GLhalfNV * v) { glSecondaryColor3f(h(v[0]), h(v[1]), h(v[2])); }

// ---------------------------------------------------------------------------
// glGetLight / glGetMaterial (positions and spot directions are in eye coordinates, as stored)
// ---------------------------------------------------------------------------
static int getLight(Context* ctx, GLenum light, GLenum pname, float* out) {
    if (light < GL_LIGHT0 || light >= GL_LIGHT0 + 8) return ctx->setError(GL_INVALID_ENUM), 0;
    const LightState& L = compat(ctx)->lights[light - GL_LIGHT0];
    switch (pname) {
    case GL_AMBIENT: memcpy(out, L.ambient, 16); return 4;
    case GL_DIFFUSE: memcpy(out, L.diffuse, 16); return 4;
    case GL_SPECULAR: memcpy(out, L.specular, 16); return 4;
    case GL_POSITION: memcpy(out, L.position, 16); return 4;
    case GL_SPOT_DIRECTION: memcpy(out, L.spotDirection, 12); return 3;
    case GL_SPOT_EXPONENT: out[0] = L.spotExponent; return 1;
    case GL_SPOT_CUTOFF: out[0] = L.spotCutoff; return 1;
    case GL_CONSTANT_ATTENUATION: out[0] = L.constantAtt; return 1;
    case GL_LINEAR_ATTENUATION: out[0] = L.linearAtt; return 1;
    case GL_QUADRATIC_ATTENUATION: out[0] = L.quadraticAtt; return 1;
    default: ctx->setError(GL_INVALID_ENUM); return 0;
    }
}
API void APIENTRY glGetLightfv(GLenum light, GLenum pname, GLfloat* params) {
    MGL_CTX();
    float v[4];
    int n = getLight(ctx, light, pname, v);
    memcpy(params, v, n * 4);
}
API void APIENTRY glGetLightiv(GLenum light, GLenum pname, GLint* params) {
    MGL_CTX();
    float v[4];
    int n = getLight(ctx, light, pname, v);
    bool color = pname == GL_AMBIENT || pname == GL_DIFFUSE || pname == GL_SPECULAR;
    for (int k = 0; k < n; k++) params[k] = color ? (GLint)lround(std::clamp(v[k], -1.0f, 1.0f) * 2147483647.0) : (GLint)lroundf(v[k]);
}

static int getMaterial(Context* ctx, GLenum face, GLenum pname, float* out) {
    if (face != GL_FRONT && face != GL_BACK) return ctx->setError(GL_INVALID_ENUM), 0;
    const MaterialState& m = compat(ctx)->material[face == GL_BACK];
    switch (pname) {
    case GL_AMBIENT: memcpy(out, m.ambient, 16); return 4;
    case GL_DIFFUSE: memcpy(out, m.diffuse, 16); return 4;
    case GL_SPECULAR: memcpy(out, m.specular, 16); return 4;
    case GL_EMISSION: memcpy(out, m.emission, 16); return 4;
    case GL_SHININESS: out[0] = m.shininess; return 1;
    default: ctx->setError(GL_INVALID_ENUM); return 0;
    }
}
API void APIENTRY glGetMaterialfv(GLenum face, GLenum pname, GLfloat* params) {
    MGL_CTX();
    float v[4];
    int n = getMaterial(ctx, face, pname, v);
    memcpy(params, v, n * 4);
}
API void APIENTRY glGetMaterialiv(GLenum face, GLenum pname, GLint* params) {
    MGL_CTX();
    float v[4];
    int n = getMaterial(ctx, face, pname, v);
    for (int k = 0; k < n; k++) params[k] = pname == GL_SHININESS ? (GLint)lroundf(v[k]) : (GLint)lround(std::clamp(v[k], -1.0f, 1.0f) * 2147483647.0);
}

// ---------------------------------------------------------------------------
// Raster position, glBitmap, glDrawPixels, glCopyPixels (GL 4.6 compatibility §14.7, §18.1)
// ---------------------------------------------------------------------------
static void setRasterFromCurrent(CompatState* cs) {
    memcpy(cs->rasterColor, cs->current[CA_COLOR], 16);
    memcpy(cs->rasterSecondary, cs->current[CA_SECONDARY], 16);
    memcpy(cs->rasterTex, cs->current[CA_TEX0], 16);
}

static void rasterPos(Context* ctx, float x, float y, float z, float w) {
    CompatState* cs = compat(ctx);
    const Mat4& mv = cs->modelview.back();
    const Mat4& pr = cs->projection.back();
    float eye[4], clip[4];
    for (int r = 0; r < 4; r++) eye[r] = mv.at(r, 0) * x + mv.at(r, 1) * y + mv.at(r, 2) * z + mv.at(r, 3) * w;
    for (int r = 0; r < 4; r++) clip[r] = pr.at(r, 0) * eye[0] + pr.at(r, 1) * eye[1] + pr.at(r, 2) * eye[2] + pr.at(r, 3) * eye[3];
    bool inside = clip[3] > 0;
    for (int k = 0; k < 3 && inside; k++) inside = clip[k] >= -clip[3] && clip[k] <= clip[3];
    // GL_IBM_rasterpos_clip: RASTER_POSITION_UNCLIPPED_IBM keeps positions outside the view volume (in front of the eye)
    cs->rasterValid = inside || (cs->rasterUnclippedIBM && clip[3] > 0);
    if (!cs->rasterValid) return;
    const float* vp = ctx->viewport[0];
    const float* dr = ctx->depthRange[0];
    float nx = clip[0] / clip[3], ny = clip[1] / clip[3], nz = clip[2] / clip[3];
    cs->rasterPos[0] = vp[0] + (nx + 1) * vp[2] / 2;
    cs->rasterPos[1] = vp[1] + (ny + 1) * vp[3] / 2;
    cs->rasterPos[2] = dr[0] + (nz + 1) / 2 * (dr[1] - dr[0]);
    cs->rasterPos[3] = clip[3];
    cs->rasterDistance = sqrtf(eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]);
    setRasterFromCurrent(cs);
}

static void windowPos(Context* ctx, float x, float y, float z) {
    CompatState* cs = compat(ctx);
    const float* dr = ctx->depthRange[0];
    cs->rasterPos[0] = x;
    cs->rasterPos[1] = y;
    cs->rasterPos[2] = dr[0] + std::clamp(z, 0.0f, 1.0f) * (dr[1] - dr[0]);
    cs->rasterPos[3] = 1;
    cs->rasterValid = true;
    cs->rasterDistance = 0;
    setRasterFromCurrent(cs);
}

API void APIENTRY glRasterPos4f(GLfloat x, GLfloat y, GLfloat z, GLfloat w) {
    MGL_CTX();
    RECORD(rasterPos(ctx, x, y, z, w));
    rasterPos(ctx, x, y, z, w);
}
API void APIENTRY glWindowPos3f(GLfloat x, GLfloat y, GLfloat z) {
    MGL_CTX();
    RECORD(windowPos(ctx, x, y, z));
    windowPos(ctx, x, y, z);
}
API void APIENTRY glPixelZoom(GLfloat xfactor, GLfloat yfactor) {
    MGL_CTX();
    RECORD(glPixelZoom(xfactor, yfactor));
    compat(ctx)->zoomX = xfactor;
    compat(ctx)->zoomY = yfactor;
}

// The internal program and objects that draw rectangles of pixels at window coordinates. Rendering state
// (depth / stencil tests, blending, alpha test, scissor, write masks) applies; the app's bindings and program
// are restored afterwards.
static bool metaEnsure(Context* ctx) {
    CompatState* cs = compat(ctx);
    if (cs->metaProg) return true;
    const char* vs = "#version 330 core\nlayout(location = 0) in vec3 p;\nlayout(location = 1) in vec2 t;\nout vec2 uv;\n"
                     "void main() { uv = t; gl_Position = vec4(p, 1.0); }\n";
    const char* fs = "#version 330 core\nuniform sampler2D tex;\nuniform vec4 color;\nuniform int mode;\nin vec2 uv;\nout vec4 o;\n"
                     "void main() { vec4 s = texture(tex, uv); if (mode == 0) { if (s.r == 0.0) discard; o = color; } else o = s; }\n";
    GLuint v = glCreateShader(GL_VERTEX_SHADER), f = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(v, 1, &vs, nullptr);
    glShaderSource(f, 1, &fs, nullptr);
    glCompileShader(v);
    glCompileShader(f);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    glDeleteShader(v);
    glDeleteShader(f);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        MGL_ERR("glBitmap / glDrawPixels: internal program failed to link");
        glDeleteProgram(p);
        return false;
    }
    cs->metaProg = p;
    cs->metaColorLoc = glGetUniformLocation(p, "color");
    cs->metaModeLoc = glGetUniformLocation(p, "mode");
    glGenVertexArrays(1, &cs->metaVao);
    glGenBuffers(1, &cs->metaVbo);
    glGenTextures(1, &cs->metaTex);
    return true;
}

// Uploads `w` x `h` texels of `internalFormat` from `format` / `type` data with default unpacking.
static void metaUpload(Context* ctx, GLint internalFormat, int w, int h, GLenum format, GLenum type, const void* data) {
    CompatState* cs = compat(ctx);
    PixelStore saved = ctx->unpack;
    Ref<Buffer> savedPbo = ctx->pixelUnpackBuffer;
    ctx->unpack = PixelStore();
    ctx->unpack.alignment = 1;
    ctx->pixelUnpackBuffer = nullptr;
    GLint savedActive = 0, savedTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActive);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);
    glBindTexture(GL_TEXTURE_2D, cs->metaTex);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, w, h, 0, format, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
    glActiveTexture((GLenum)savedActive);
    ctx->unpack = saved;
    ctx->pixelUnpackBuffer = savedPbo;
}

// Draws the meta texture over window rectangle (x0, y0)-(x1, y1) at window depth z: as a mask in `color`
// (mode 0, glBitmap) or its colors (mode 1).
static void metaDraw(Context* ctx, float x0, float y0, float x1, float y1, float z, int mode, const float* color) {
    CompatState* cs = compat(ctx);
    const float* vp = ctx->viewport[0];
    const float* dr = ctx->depthRange[0];
    if (vp[2] <= 0 || vp[3] <= 0) return;
    auto nx = [&](float x) { return (x - vp[0]) / vp[2] * 2 - 1; };
    auto ny = [&](float y) { return (y - vp[1]) / vp[3] * 2 - 1; };
    float nz = dr[1] != dr[0] ? (z - dr[0]) / (dr[1] - dr[0]) * 2 - 1 : 0;
    float quad[4][5] = {{nx(x0), ny(y0), nz, 0, 0}, {nx(x1), ny(y0), nz, 1, 0}, {nx(x0), ny(y1), nz, 0, 1}, {nx(x1), ny(y1), nz, 1, 1}};
    // save
    Ref<ProgramObj> savedProg = ctx->program;
    GLint savedVao = 0, savedBuf = 0, savedActive = 0, savedTex = 0, savedSampler = 0, savedPoly[2] = {};
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &savedVao);
    glGetIntegerv(GL_ARRAY_BUFFER_BINDING, &savedBuf);
    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActive);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);
    glGetIntegerv(GL_SAMPLER_BINDING, &savedSampler);
    glGetIntegerv(GL_POLYGON_MODE, savedPoly);
    GLboolean cull = glIsEnabled(GL_CULL_FACE);
    // draw (the internal program replaces the current one directly, so the app's isn't released or reset)
    {
        std::lock_guard<std::recursive_mutex> g(ctx->share->mutex);
        ctx->program = ctx->share->program(cs->metaProg);
    }
    glUniform4fv(cs->metaColorLoc, 1, color);
    glUniform1i(cs->metaModeLoc, mode);
    glBindVertexArray(cs->metaVao);
    glBindBuffer(GL_ARRAY_BUFFER, cs->metaVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STREAM_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 20, (const void*)0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 20, (const void*)12);
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glBindTexture(GL_TEXTURE_2D, cs->metaTex);
    glBindSampler(0, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glDisable(GL_CULL_FACE);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    // restore
    if (cull) glEnable(GL_CULL_FACE);
    glPolygonMode(GL_FRONT_AND_BACK, (GLenum)savedPoly[0]);
    glBindSampler(0, (GLuint)savedSampler);
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
    glActiveTexture((GLenum)savedActive);
    glBindBuffer(GL_ARRAY_BUFFER, (GLuint)savedBuf);
    glBindVertexArray((GLuint)savedVao);
    ctx->program = savedProg;
}

// glBitmap with the bits already unpacked into one byte per pixel (0 / 1), rows bottom to top.
static void bitmapDraw(Context* ctx, int w, int h, float xo, float yo, float xm, float ym, const std::vector<uint8_t>& bits) {
    CompatState* cs = compat(ctx);
    if (!cs->rasterValid) return;
    if (w > 0 && h > 0 && metaEnsure(ctx)) {
        metaUpload(ctx, GL_R8, w, h, GL_RED, GL_UNSIGNED_BYTE, bits.data());
        float x0 = floorf(cs->rasterPos[0] - xo), y0 = floorf(cs->rasterPos[1] - yo);
        metaDraw(ctx, x0, y0, x0 + w, y0 + h, cs->rasterPos[2], 0, cs->rasterColor);
    }
    cs->rasterPos[0] += xm;
    cs->rasterPos[1] += ym;
}

// Client bitmap bits (pixel unpack state: alignment, row length, skips, LSB_FIRST; or the bound unpack buffer) as
// 0 / 255 bytes.
static std::vector<uint8_t> unpackBitmap(Context* ctx, int w, int h, const GLubyte* bitmap) {
    std::vector<uint8_t> bits((size_t)std::max(w, 0) * std::max(h, 0));
    const PixelStore& ps = ctx->unpack;
    const uint8_t* base = (const uint8_t*)bitmap;
    if (Buffer* pbo = ctx->pixelUnpackBuffer.get()) base = pbo->cpu ? pbo->cpu + (uintptr_t)bitmap : nullptr;
    if (!base || bits.empty()) return bits;
    int rowLen = ps.rowLength > 0 ? ps.rowLength : w;
    size_t rowBytes = ((size_t)rowLen + 7) / 8;
    rowBytes = (rowBytes + ps.alignment - 1) / ps.alignment * ps.alignment;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            size_t bit = (size_t)ps.skipPixels + x;
            uint8_t byte = base[(size_t)(ps.skipRows + y) * rowBytes + bit / 8];
            bool on = ps.lsbFirst ? (byte >> (bit % 8)) & 1 : (byte >> (7 - bit % 8)) & 1;
            bits[(size_t)y * w + x] = on ? 255 : 0;
        }
    return bits;
}

API void APIENTRY glBitmap(GLsizei w, GLsizei h, GLfloat xo, GLfloat yo, GLfloat xm, GLfloat ym, const GLubyte* bitmap) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    std::vector<uint8_t> bits = unpackBitmap(ctx, w, h, bitmap);
    RECORD(bitmapDraw(ctx, w, h, xo, yo, xm, ym, bits));  // lists keep the bits (wglUseFontBitmaps)
    bitmapDraw(ctx, w, h, xo, yo, xm, ym, bits);
}

// glDrawPixels with the client data already converted to RGBA floats.
static void drawPixelsRGBA(Context* ctx, int w, int h, const std::vector<float>& rgba) {
    CompatState* cs = compat(ctx);
    if (!cs->rasterValid || w <= 0 || h <= 0 || !metaEnsure(ctx)) return;
    metaUpload(ctx, GL_RGBA32F, w, h, GL_RGBA, GL_FLOAT, rgba.data());
    float x0 = cs->rasterPos[0], y0 = cs->rasterPos[1];
    metaDraw(ctx, x0, y0, x0 + w * cs->zoomX, y0 + h * cs->zoomY, cs->rasterPos[2], 1, cs->rasterColor);
}

API void APIENTRY glDrawPixels(GLsizei w, GLsizei h, GLenum format, GLenum type, const void* pixels) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    if (format == GL_DEPTH_COMPONENT || format == GL_STENCIL_INDEX || format == GL_DEPTH_STENCIL || format == GL_COLOR_INDEX) {
        static bool warned = false;
        if (!warned) MGL_WARN("glDrawPixels: depth / stencil / color index data is not supported");
        warned = true;
        return;
    }
    if (GLenum e = pixelFormatTypeError(format, type)) return ctx->setError(e);
    std::vector<float> rgba((size_t)w * h * 4);
    const uint8_t* src = (const uint8_t*)pixels;
    if (Buffer* pbo = ctx->pixelUnpackBuffer.get()) src = pbo->cpu ? pbo->cpu + (uintptr_t)pixels : nullptr;
    if (src && w && h) {
        ClientLayout l = clientLayout(format, type, w, h, ctx->unpack);
        convertToStorage(formatInfo(GL_RGBA32F), format, type, src + l.skipBytes, l.rowStride, l.imageStride, rgba.data(), (size_t)w * 16,
                         (size_t)w * h * 16, w, h, 1, ctx->unpack.swapBytes);
    }
    RECORD(drawPixelsRGBA(ctx, w, h, rgba));
    drawPixelsRGBA(ctx, w, h, rgba);
}

static void copyPixels(Context* ctx, int x, int y, int w, int h, GLenum type) {
    CompatState* cs = compat(ctx);
    if (type != GL_COLOR) {
        static bool warned = false;
        if (!warned) MGL_WARN("glCopyPixels: depth / stencil copies are not supported");
        warned = true;
        return;
    }
    if (!cs->rasterValid || w <= 0 || h <= 0 || !metaEnsure(ctx)) return;
    GLint savedActive = 0, savedTex = 0;
    glGetIntegerv(GL_ACTIVE_TEXTURE, &savedActive);
    glActiveTexture(GL_TEXTURE0);
    glGetIntegerv(GL_TEXTURE_BINDING_2D, &savedTex);
    glBindTexture(GL_TEXTURE_2D, cs->metaTex);
    glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, x, y, w, h, 0);  // from the read framebuffer
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, (GLuint)savedTex);
    glActiveTexture((GLenum)savedActive);
    float x0 = cs->rasterPos[0], y0 = cs->rasterPos[1];
    metaDraw(ctx, x0, y0, x0 + w * cs->zoomX, y0 + h * cs->zoomY, cs->rasterPos[2], 1, cs->rasterColor);
}

API void APIENTRY glCopyPixels(GLint x, GLint y, GLsizei w, GLsizei h, GLenum type) {
    MGL_CTX();
    if (w < 0 || h < 0) return ctx->setError(GL_INVALID_VALUE);
    if (type != GL_COLOR && type != GL_DEPTH && type != GL_STENCIL && type != GL_DEPTH_STENCIL_TO_RGBA_NV) return ctx->setError(GL_INVALID_ENUM);
    RECORD(copyPixels(ctx, x, y, w, h, type));
    copyPixels(ctx, x, y, w, h, type);
}

// glRasterPos / glWindowPos type variants (generated from gl.xml)
API void APIENTRY glRasterPos2d(GLdouble x, GLdouble y) { glRasterPos4f((GLfloat)x, (GLfloat)y, 0.0f, 1.0f); }
API void APIENTRY glRasterPos2dv(const GLdouble * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glRasterPos2f(GLfloat x, GLfloat y) { glRasterPos4f((GLfloat)x, (GLfloat)y, 0.0f, 1.0f); }
API void APIENTRY glRasterPos2fv(const GLfloat * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glRasterPos2i(GLint x, GLint y) { glRasterPos4f((GLfloat)x, (GLfloat)y, 0.0f, 1.0f); }
API void APIENTRY glRasterPos2iv(const GLint * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glRasterPos2s(GLshort x, GLshort y) { glRasterPos4f((GLfloat)x, (GLfloat)y, 0.0f, 1.0f); }
API void APIENTRY glRasterPos2sv(const GLshort * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], 0.0f, 1.0f); }
API void APIENTRY glRasterPos3d(GLdouble x, GLdouble y, GLdouble z) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, 1.0f); }
API void APIENTRY glRasterPos3dv(const GLdouble * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glRasterPos3f(GLfloat x, GLfloat y, GLfloat z) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, 1.0f); }
API void APIENTRY glRasterPos3fv(const GLfloat * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glRasterPos3i(GLint x, GLint y, GLint z) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, 1.0f); }
API void APIENTRY glRasterPos3iv(const GLint * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glRasterPos3s(GLshort x, GLshort y, GLshort z) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, 1.0f); }
API void APIENTRY glRasterPos3sv(const GLshort * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], 1.0f); }
API void APIENTRY glRasterPos4d(GLdouble x, GLdouble y, GLdouble z, GLdouble w) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
API void APIENTRY glRasterPos4dv(const GLdouble * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glRasterPos4fv(const GLfloat * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glRasterPos4i(GLint x, GLint y, GLint z, GLint w) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
API void APIENTRY glRasterPos4iv(const GLint * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glRasterPos4s(GLshort x, GLshort y, GLshort z, GLshort w) { glRasterPos4f((GLfloat)x, (GLfloat)y, (GLfloat)z, (GLfloat)w); }
API void APIENTRY glRasterPos4sv(const GLshort * v) { glRasterPos4f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2], (GLfloat)v[3]); }
API void APIENTRY glWindowPos2d(GLdouble x, GLdouble y) { glWindowPos3f((GLfloat)x, (GLfloat)y, 0.0f); }
API void APIENTRY glWindowPos2dv(const GLdouble * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], 0.0f); }
API void APIENTRY glWindowPos2f(GLfloat x, GLfloat y) { glWindowPos3f((GLfloat)x, (GLfloat)y, 0.0f); }
API void APIENTRY glWindowPos2fv(const GLfloat * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], 0.0f); }
API void APIENTRY glWindowPos2i(GLint x, GLint y) { glWindowPos3f((GLfloat)x, (GLfloat)y, 0.0f); }
API void APIENTRY glWindowPos2iv(const GLint * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], 0.0f); }
API void APIENTRY glWindowPos2s(GLshort x, GLshort y) { glWindowPos3f((GLfloat)x, (GLfloat)y, 0.0f); }
API void APIENTRY glWindowPos2sv(const GLshort * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], 0.0f); }
API void APIENTRY glWindowPos3d(GLdouble x, GLdouble y, GLdouble z) { glWindowPos3f((GLfloat)x, (GLfloat)y, (GLfloat)z); }
API void APIENTRY glWindowPos3dv(const GLdouble * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]); }
API void APIENTRY glWindowPos3fv(const GLfloat * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]); }
API void APIENTRY glWindowPos3i(GLint x, GLint y, GLint z) { glWindowPos3f((GLfloat)x, (GLfloat)y, (GLfloat)z); }
API void APIENTRY glWindowPos3iv(const GLint * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]); }
API void APIENTRY glWindowPos3s(GLshort x, GLshort y, GLshort z) { glWindowPos3f((GLfloat)x, (GLfloat)y, (GLfloat)z); }
API void APIENTRY glWindowPos3sv(const GLshort * v) { glWindowPos3f((GLfloat)v[0], (GLfloat)v[1], (GLfloat)v[2]); }
