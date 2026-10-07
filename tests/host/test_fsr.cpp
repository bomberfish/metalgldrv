// EXT_fragment_shading_rate (METALGLDRV_STRICT only): coarse shading emulated in the fragment shader.
// Run mgltest with METALGLDRV_STRICT=1 to cover it; without it the tests check the extension is absent.
#include <cmath>
#include "harness.h"

namespace {
bool hasFsr() {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (!strcmp((const char*)glGetStringi(GL_EXTENSIONS, i), "GL_EXT_fragment_shading_rate")) return true;
    return false;
}

const int W = 16, H = 16;

// A W x H framebuffer with RGBA32F color attachments 0 and 1.
struct Target {
    GLuint fbo = 0, tex[2] = {0, 0};
    Target() {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glGenTextures(2, tex);
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, tex[i]);
            glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, W, H);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_2D, tex[i], 0);
        }
        const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
        glDrawBuffers(2, bufs);
        glViewport(0, 0, 2 * W, 2 * H);
        const float z[4] = {-1, -1, -1, -1};
        glClearBufferfv(GL_COLOR, 0, z);
        glClearBufferfv(GL_COLOR, 1, z);
    }
    ~Target() {
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(2, tex);
    }
    std::vector<float> read(int att) {
        std::vector<float> v(W * H * 4);
        glReadBuffer(GL_COLOR_ATTACHMENT0 + att);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_FLOAT, v.data());
        return v;
    }
};

// Atomic counter at binding 0, reset to 0.
struct Counter {
    GLuint buf = 0;
    Counter() {
        glGenBuffers(1, &buf);
        glBindBufferBase(GL_ATOMIC_COUNTER_BUFFER, 0, buf);
        GLuint z = 0;
        glBufferData(GL_ATOMIC_COUNTER_BUFFER, 4, &z, GL_DYNAMIC_READ);
    }
    ~Counter() { glDeleteBuffers(1, &buf); }
    GLuint value() {
        glMemoryBarrier(GL_ATOMIC_COUNTER_BARRIER_BIT);
        GLuint v = 0;
        glBindBuffer(GL_ATOMIC_COUNTER_BUFFER, buf);
        glGetBufferSubData(GL_ATOMIC_COUNTER_BUFFER, 0, 4, &v);
        return v;
    }
};

// One triangle covering the window (from gl_VertexID) with a 32x32 viewport, inside the clip volume: Metal
// splits clipped polygons into triangles, and coarse fragments along the split would be shaded once per
// triangle. `wpos` is the window position.
const char* kQuadVS = R"(#version 450
#extension GL_EXT_fragment_shading_rate : enable
out vec2 wpos;
uniform int u_primRate;
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 1.0 : -1.0, gl_VertexID == 2 ? 1.0 : -1.0);
    wpos = (p * 0.5 + 0.5) * vec2(32.0);
    gl_Position = vec4(p, 0.0, 1.0);
    gl_PrimitiveShadingRateEXT = u_primRate;
})";

GLuint quadVao() {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    return vao;
}

void resetFsr() {
    glShadingRateEXT(GL_SHADING_RATE_1X1_PIXELS_EXT);
    glShadingRateCombinerOpsEXT(GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT);
}
}  // namespace

TEST(fsr_extension_strict_only) {
    const char* s = getenv("METALGLDRV_STRICT");
    bool strict = s && atoi(s) != 0;
    CHECK_EQ(hasFsr(), strict);
    if (!hasFsr()) return;
    GLenum rates[16];
    GLsizei n = -1;
    glGetFragmentShadingRatesEXT(4, 16, &n, rates);
    CHECK_EQ(n, 4);
    GLint v = 0;
    glGetIntegerv(GL_SHADING_RATE_EXT, &v);
    CHECK_EQ(v, GL_SHADING_RATE_1X1_PIXELS_EXT);
    glShadingRateEXT(GL_SAMPLE_SHADING);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_ENUM);
    CHECK_GL();
}

// One invocation per coarse fragment: gl_FragCoord is the fragment's centre, inputs are evaluated there,
// derivatives are per pixel, every pixel of the fragment gets its outputs; 4x4 is clamped to 2x2.
TEST_SIZE(fsr_coarse_quad, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
in vec2 wpos;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    atomicCounterIncrement(cnt);
    o0 = vec4(gl_FragCoord.xy, wpos);
    o1 = vec4(dFdx(wpos.x), dFdy(wpos.y), float(gl_ShadingRateEXT), float(bitCount(gl_SampleMaskIn[0])));
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    struct Case {
        GLenum rate;
        int w, h, packed;
    } cases[] = {{GL_SHADING_RATE_1X1_PIXELS_EXT, 1, 1, 0}, {GL_SHADING_RATE_2X2_PIXELS_EXT, 2, 2, 5},
                 {GL_SHADING_RATE_2X1_PIXELS_EXT, 2, 1, 4}, {GL_SHADING_RATE_1X2_PIXELS_EXT, 1, 2, 1},
                 {GL_SHADING_RATE_4X4_PIXELS_EXT, 2, 2, 5}, {GL_SHADING_RATE_4X1_PIXELS_EXT, 2, 1, 4}};
    for (const Case& c : cases) {
        Target t;
        Counter cnt;
        glShadingRateEXT(c.rate);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        CHECK_EQ(cnt.value(), (GLuint)(W * H / (c.w * c.h)));
        std::vector<float> a = t.read(0), b = t.read(1);
        int bad = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) {
                const float* o0 = &a[(y * W + x) * 4];
                const float* o1 = &b[(y * W + x) * 4];
                float cx = (x / c.w) * c.w + c.w * 0.5f, cy = (y / c.h) * c.h + c.h * 0.5f;
                bool ok = o0[0] == cx && o0[1] == cy && fabsf(o0[2] - cx) < 1e-4f && fabsf(o0[3] - cy) < 1e-4f;
                ok = ok && fabsf(o1[0] - 1.0f) < 1e-4f && fabsf(o1[1] - 1.0f) < 1e-4f && o1[2] == c.packed && o1[3] == c.w * c.h;
                if (!ok && bad++ < 3)
                    fprintf(stderr, "  rate %dx%d pixel (%d,%d): %g %g %g %g / %g %g %g %g\n", c.w, c.h, x, y, o0[0], o0[1], o0[2], o0[3],
                            o1[0], o1[1], o1[2], o1[3]);
            }
        CHECK_EQ(bad, 0);
    }
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Perspective-correct inputs at the fragment centre and their derivatives (differences to the neighbouring
// coarse fragments, per pixel), against the interpolation formula; partial coverage: one invocation per coarse
// fragment with a covered pixel (2x1 and 1x2 quads whose rows/columns are covered in different columns/rows
// shade from lanes that are not in a 2x2 arrangement).
TEST_SIZE(fsr_perspective_partial, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(R"(#version 450
layout(location = 0) in vec4 pos;
layout(location = 1) in float attr;
out float v;
void main() { v = attr; gl_Position = pos; })",
                              R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
in float v;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    atomicCounterIncrement(cnt);
    o0 = vec4(gl_FragCoord.xy, v, 1.0);
    o1 = vec4(float(gl_ShadingRateEXT), dFdx(v), dFdy(v), fwidth(v));
})");
    glUseProgram(p);
    // Window-space vertices (x, y) with clip w and attribute, on the rasterizer's subpixel grid so that the
    // formula below is what the GPU interpolates: a large triangle, and a sliver along the diagonal (quads
    // covered at two diagonal pixels).
    const float geometry[2][3][4] = {{{1.25f, 0.75f, 1.0f, 10.0f}, {15.125f, 3.25f, 4.0f, -6.0f}, {5.375f, 14.625f, 2.0f, 3.0f}},
                                     {{0.25f, 0.0f, 1.0f, 2.0f}, {16.0f, 15.75f, 3.0f, 9.0f}, {0.0f, 0.25f, 2.0f, -4.0f}}};
    int diagonalQuads = 0;
    for (int g = 0; g < 2; g++) {
        const float (*vx)[4] = geometry[g];
        float data[3][5];
        for (int i = 0; i < 3; i++) {
            float w = vx[i][2];
            data[i][0] = (vx[i][0] / (2 * W) * 2 - 1) * w;  // the viewport is 2W x 2H (Target)
            data[i][1] = (vx[i][1] / (2 * H) * 2 - 1) * w;
            data[i][2] = 0;
            data[i][3] = w;
            data[i][4] = vx[i][3];
        }
        GLuint vao, vbo;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 20, nullptr);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 20, (void*)16);
        glEnableVertexAttribArray(0);
        glEnableVertexAttribArray(1);
        auto expected = [&](float px, float py) {
            auto edge = [](const float* a, const float* b, float x, float y) { return (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]); };
            float area = edge(vx[0], vx[1], vx[2][0], vx[2][1]);
            float l0 = edge(vx[1], vx[2], px, py) / area, l1 = edge(vx[2], vx[0], px, py) / area, l2 = edge(vx[0], vx[1], px, py) / area;
            float num = l0 * vx[0][3] / vx[0][2] + l1 * vx[1][3] / vx[1][2] + l2 * vx[2][3] / vx[2][2];
            float den = l0 / vx[0][2] + l1 / vx[1][2] + l2 / vx[2][2];
            return num / den;
        };
        std::vector<bool> covered(W * H);
        resetFsr();
        {
            Target t;
            Counter cnt;
            glDrawArrays(GL_TRIANGLES, 0, 3);
            std::vector<float> a = t.read(0);
            GLuint n = 0;
            for (int i = 0; i < W * H; i++) n += covered[i] = a[i * 4 + 3] == 1.0f;
            CHECK_EQ(cnt.value(), n);
        }
        // quads whose two covered pixels are diagonal (the 2x1 / 1x2 case above)
        int diagonal = 0;
        for (int y = 0; y < H; y += 2)
            for (int x = 0; x < W; x += 2) {
                bool c00 = covered[y * W + x], c10 = covered[y * W + x + 1], c01 = covered[(y + 1) * W + x], c11 = covered[(y + 1) * W + x + 1];
                diagonal += (c00 && c11 && !c10 && !c01) || (c10 && c01 && !c00 && !c11);
            }
        diagonalQuads += diagonal;
        struct Case {
            GLenum rate;
            int w, h, packed;
        } cases[] = {{GL_SHADING_RATE_2X2_PIXELS_EXT, 2, 2, 5}, {GL_SHADING_RATE_2X1_PIXELS_EXT, 2, 1, 4}, {GL_SHADING_RATE_1X2_PIXELS_EXT, 1, 2, 1}};
        auto near = [](float a, float e, float tol) { return fabsf(a - e) <= tol * std::max(1.0f, fabsf(e)); };
        for (const Case& c : cases) {
            glShadingRateEXT(c.rate);
            Target t;
            Counter cnt;
            glDrawArrays(GL_TRIANGLES, 0, 3);
            std::vector<float> a = t.read(0), b = t.read(1);
            GLuint blocks = 0;
            int bad = 0;
            for (int by = 0; by < H; by += c.h)
                for (int bx = 0; bx < W; bx += c.w) {
                    bool any = false;
                    float cx = bx + c.w * 0.5f, cy = by + c.h * 0.5f, e = expected(cx, cy);
                    // forward or backward differences to the neighbouring fragment, per pixel
                    float dxf = (expected(cx + c.w, cy) - e) / c.w, dxb = (e - expected(cx - c.w, cy)) / c.w;
                    float dyf = (expected(cx, cy + c.h) - e) / c.h, dyb = (e - expected(cx, cy - c.h)) / c.h;
                    for (int y = by; y < by + c.h; y++)
                        for (int x = bx; x < bx + c.w; x++) {
                            const float* o = &a[(y * W + x) * 4];
                            const float* d = &b[(y * W + x) * 4];
                            if ((o[3] == 1.0f) != covered[y * W + x]) bad++;  // per-pixel coverage is kept
                            if (!covered[y * W + x]) continue;
                            any = true;
                            bool ok = o[0] == cx && o[1] == cy && near(o[2], e, 1e-4f) && d[0] == c.packed;
                            bool okx = near(d[1], dxf, 1e-3f) || near(d[1], dxb, 1e-3f), oky = near(d[2], dyf, 1e-3f) || near(d[2], dyb, 1e-3f);
                            ok = ok && okx && oky && near(d[3], fabsf(d[1]) + fabsf(d[2]), 1e-5f);
                            if (!ok && bad++ < 3)
                                fprintf(stderr, "  %dx%d pixel (%d,%d): %g %g v %g expected %g; dx %g (%g|%g) dy %g (%g|%g)\n", c.w, c.h, x, y, o[0], o[1],
                                        o[2], e, d[1], dxf, dxb, d[2], dyf, dyb);
                        }
                    blocks += any;
                }
            CHECK_EQ(bad, 0);
            CHECK_EQ(cnt.value(), blocks);
        }
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
    CHECK(diagonalQuads > 0);
    resetFsr();
    glDeleteProgram(p);
    CHECK_GL();
}

// discard drops the whole coarse fragment, and its side effects.
TEST_SIZE(fsr_discard, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    if (gl_FragCoord.x < 8.0) discard;
    int i = 0;
    while (true) {  // left through the discard, or after 10 iterations
        if (++i > 3 && gl_FragCoord.y < 4.0) discard;
        if (i > 10) break;
    }
    atomicCounterIncrement(cnt);
    o0 = vec4(1.0);
    o1 = vec4(1.0);
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    Target t;
    Counter cnt;
    glDrawArrays(GL_TRIANGLES, 0, 3);
    // fragment centres x = 1, 3, ..., 15: kept from 9 (pixels 8..15), y >= 4 kept from centre 5 (pixels 4..15)
    CHECK_EQ(cnt.value(), 4u * 6u);
    std::vector<float> a = t.read(0);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) bad += (a[(y * W + x) * 4] == 1.0f) != (x >= 8 && y >= 4);
    CHECK_EQ(bad, 0);
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Implicit-LOD lookups and textureQueryLod use the differences between coarse fragments (the LOD
// scales with the fragment size).
TEST_SIZE(fsr_texture_lod, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
in vec2 wpos;
uniform sampler2D u_tex;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    vec2 uv = wpos / 16.0;
    o0 = texture(u_tex, uv);
    o1 = vec4(textureQueryLod(u_tex, uv), textureLod(u_tex, uv, 0.0).r, 0.0);
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexStorage2D(GL_TEXTURE_2D, 5, GL_RGBA32F, 16, 16);
    for (int l = 0; l < 5; l++) {
        int s = 16 >> l;
        std::vector<float> d(s * s * 4, (float)l);
        glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, s, s, GL_RGBA, GL_FLOAT, d.data());
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint vao = quadVao();
    for (int coarse = 0; coarse < 2; coarse++) {
        glShadingRateEXT(coarse ? GL_SHADING_RATE_2X2_PIXELS_EXT : GL_SHADING_RATE_1X1_PIXELS_EXT);
        Target t;
        glBindTexture(GL_TEXTURE_2D, tex);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        std::vector<float> a = t.read(0), b = t.read(1);
        int bad = 0;
        for (int i = 0; i < W * H; i++)
            if (a[i * 4] != coarse || fabsf(b[i * 4] - coarse) > 1e-3f || fabsf(b[i * 4 + 1] - coarse) > 1e-3f || b[i * 4 + 2] != 0.0f) {
                if (bad++ < 3) fprintf(stderr, "  coarse %d pixel %d: level %g lod %g %g\n", coarse, i, a[i * 4], b[i * 4], b[i * 4 + 1]);
            }
        CHECK_EQ(bad, 0);
    }
    resetFsr();
    glDeleteTextures(1, &tex);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// The attachment rate (2x2-pixel texels) with REPLACE, and the primitive rate written by the vertex shader.
TEST_SIZE(fsr_attachment_and_primitive, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    atomicCounterIncrement(cnt);
    o0 = vec4(float(gl_ShadingRateEXT));
    o1 = vec4(gl_FragCoord.xy, 0.0, 0.0);
})");
    glUseProgram(p);
    GLuint vao = quadVao();
    GLuint sr;
    glGenTextures(1, &sr);
    glBindTexture(GL_TEXTURE_2D, sr);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8UI, W / 2, H / 2);
    std::vector<uint8_t> rates(W / 2 * H / 2);
    for (int y = 0; y < H / 2; y++)
        for (int x = 0; x < W / 2; x++) rates[y * (W / 2) + x] = x < 4 ? 5 : y < 4 ? 4 : 0;  // 2x2 | 2x1 below, 1x1 above
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, W / 2, H / 2, GL_RED_INTEGER, GL_UNSIGNED_BYTE, rates.data());
    {
        Target t;
        Counter cnt;
        glFramebufferShadingRateEXT(GL_FRAMEBUFFER, GL_SHADING_RATE_ATTACHMENT_EXT, sr, 0, 1, 2, 2);
        GLint name = 0;
        glGetFramebufferAttachmentParameteriv(GL_FRAMEBUFFER, GL_SHADING_RATE_ATTACHMENT_EXT, GL_FRAMEBUFFER_ATTACHMENT_OBJECT_NAME, &name);
        CHECK_EQ((GLuint)name, sr);
        glShadingRateCombinerOpsEXT(GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_EXT);
        glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        CHECK_EQ(cnt.value(), (GLuint)(8 * 16 / 4 + 8 * 8 / 2 + 8 * 8));
        std::vector<float> a = t.read(0);
        int bad = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) bad += a[(y * W + x) * 4] != (x < 8 ? 5 : y < 8 ? 4 : 0);
        CHECK_EQ(bad, 0);
    }
    {
        // primitive rate 1x2 (MAX with the 1x1 draw rate), attachment ignored (KEEP)
        Target t;
        Counter cnt;
        glShadingRateCombinerOpsEXT(GL_FRAGMENT_SHADING_RATE_COMBINER_OP_MAX_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT);
        glUniform1i(glGetUniformLocation(p, "u_primRate"), 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        CHECK_EQ(cnt.value(), (GLuint)(W * H / 2));
        std::vector<float> a = t.read(0), b = t.read(1);
        int bad = 0;
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++) bad += a[(y * W + x) * 4] != 1 || b[(y * W + x) * 4 + 1] != (y & ~1) + 1;
        CHECK_EQ(bad, 0);
    }
    resetFsr();
    glDeleteTextures(1, &sr);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Fragment shaders that write gl_FragDepth (or read gl_FragCoord without enabling the extension) shade at 1x1.
TEST_SIZE(fsr_forced_1x1, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    atomicCounterIncrement(cnt);
    o0 = gl_FragCoord;
    o1 = vec4(0.0);
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    Target t;
    Counter cnt;
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_EQ(cnt.value(), (GLuint)(W * H));
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

namespace {
GLuint linkStages(std::initializer_list<std::pair<GLenum, const char*>> stages, bool separable = false) {
    GLuint p = glCreateProgram();
    if (separable) glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    for (auto& [type, src] : stages) {
        GLuint s = glCreateShader(type);
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        GLint ok = 0;
        glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            char buf[4096] = {};
            glGetShaderInfoLog(s, sizeof buf, nullptr, buf);
            fprintf(stderr, "  shader compile failed: %s\n", buf);
        }
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    CHECK(ok);
    return p;
}

// Fragment shader of the pipeline tests: the rate, the fragment centre, and an invocation count.
const char* kRateFS = R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
layout(binding = 0, offset = 0) uniform atomic_uint cnt;
void main() {
    atomicCounterIncrement(cnt);
    o0 = vec4(float(gl_ShadingRateEXT));
    o1 = vec4(gl_FragCoord.xy, 0.0, 0.0);
})";

// Every pixel shows rate `packed` (w x h) and its fragment's centre; one invocation per fragment.
void checkRate(Target& t, Counter& cnt, int packed, int w, int h) {
    CHECK_EQ(cnt.value(), (GLuint)(W * H / (w * h)));
    std::vector<float> a = t.read(0), b = t.read(1);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float* c = &b[(y * W + x) * 4];
            bad += a[(y * W + x) * 4] != packed || c[0] != (x / w) * w + w * 0.5f || c[1] != (y / h) * h + h * 0.5f;
        }
    CHECK_EQ(bad, 0);
}
}  // namespace

// interpolateAt*: offsets from the fragment's centre; centroid and sample locations within a covered pixel.
TEST_SIZE(fsr_interpolate_at, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
in vec2 wpos;
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    o0 = vec4(interpolateAtOffset(wpos, vec2(0.25, -0.375)), gl_FragCoord.xy);
    o1 = vec4(interpolateAtCentroid(wpos), interpolateAtSample(wpos, 0));
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    Target t;
    glDrawArrays(GL_TRIANGLES, 0, 3);
    std::vector<float> a = t.read(0), b = t.read(1);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float* o = &a[(y * W + x) * 4];
            const float* c = &b[(y * W + x) * 4];
            float cx = (x & ~1) + 1.0f, cy = (y & ~1) + 1.0f;
            bool ok = o[2] == cx && o[3] == cy && fabsf(o[0] - (cx + 0.25f)) < 1e-4f && fabsf(o[1] - (cy - 0.375f)) < 1e-4f;
            // pixel centres of the fragment
            for (int k = 0; k < 4; k += 2) ok = ok && fabsf(fabsf(c[k] - cx) - 0.5f) < 1e-4f && fabsf(fabsf(c[k + 1] - cy) - 0.5f) < 1e-4f;
            if (!ok && bad++ < 3) fprintf(stderr, "  pixel (%d,%d): %g %g %g %g / %g %g %g %g\n", x, y, o[0], o[1], o[2], o[3], c[0], c[1], c[2], c[3]);
        }
    CHECK_EQ(bad, 0);
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Subgroup quad operations work on the 2x2 block of coarse fragments.
TEST_SIZE(fsr_subgroup_quad, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
#extension GL_KHR_shader_subgroup_quad : require
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    o0 = vec4(subgroupQuadSwapHorizontal(gl_FragCoord.xy), subgroupQuadSwapVertical(gl_FragCoord.xy));
    o1 = vec4(subgroupQuadSwapDiagonal(gl_FragCoord.xy), gl_FragCoord.xy);
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    Target t;
    glDrawArrays(GL_TRIANGLES, 0, 3);
    std::vector<float> a = t.read(0), b = t.read(1);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float* o = &a[(y * W + x) * 4];
            const float* c = &b[(y * W + x) * 4];
            float cx = c[2], cy = c[3];
            bool ok = fabsf(o[0] - cx) == 2.0f && o[1] == cy && o[2] == cx && fabsf(o[3] - cy) == 2.0f && fabsf(c[0] - cx) == 2.0f &&
                      fabsf(c[1] - cy) == 2.0f;
            if (!ok && bad++ < 3) fprintf(stderr, "  pixel (%d,%d): %g %g %g %g / %g %g %g %g\n", x, y, o[0], o[1], o[2], o[3], c[0], c[1], c[2], c[3]);
        }
    CHECK_EQ(bad, 0);
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// The primitive rate from a geometry shader, and from a separable vertex program in a program pipeline.
TEST_SIZE(fsr_primitive_rate_paths, 16, 16) {
    if (!hasFsr()) return;
    const char* vs = R"(#version 450
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 1.0 : -1.0, gl_VertexID == 2 ? 1.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
})";
    const char* gs = R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;
void main() {
    for (int i = 0; i < 3; i++) {
        gl_Position = gl_in[i].gl_Position;
        gl_PrimitiveShadingRateEXT = gl_ShadingRateFlag2HorizontalPixelsEXT;
        EmitVertex();
    }
})";
    GLuint vao = quadVao();
    glShadingRateCombinerOpsEXT(GL_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT);
    {
        GLuint p = linkStages({{GL_VERTEX_SHADER, vs}, {GL_GEOMETRY_SHADER, gs}, {GL_FRAGMENT_SHADER, kRateFS}});
        glUseProgram(p);
        Target t;
        Counter cnt;
        glDrawArrays(GL_TRIANGLES, 0, 3);
        checkRate(t, cnt, 4, 2, 1);
        glUseProgram(0);
        glDeleteProgram(p);
    }
    {
        GLuint pv = linkStages({{GL_VERTEX_SHADER, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
out gl_PerVertex { vec4 gl_Position; };
void main() {
    vec2 p = vec2(gl_VertexID == 1 ? 1.0 : -1.0, gl_VertexID == 2 ? 1.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
    gl_PrimitiveShadingRateEXT = gl_ShadingRateFlag2VerticalPixelsEXT;
})"}}, true);
        GLuint pf = linkStages({{GL_FRAGMENT_SHADER, kRateFS}}, true);
        GLuint pipe;
        glGenProgramPipelines(1, &pipe);
        glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, pv);
        glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, pf);
        glBindProgramPipeline(pipe);
        Target t;
        Counter cnt;
        glDrawArrays(GL_TRIANGLES, 0, 3);
        checkRate(t, cnt, 1, 1, 2);
        glBindProgramPipeline(0);
        glDeleteProgramPipelines(1, &pipe);
        glDeleteProgram(pv);
        glDeleteProgram(pf);
    }
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

// Mesh shaders: the per-primitive rate.
TEST_SIZE(fsr_mesh_primitive_rate, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = linkStages({{GL_MESH_SHADER_EXT, R"(#version 460
#extension GL_EXT_mesh_shader : require
#extension GL_EXT_fragment_shading_rate : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    SetMeshOutputsEXT(3u, 1u);
    for (int i = 0; i < 3; i++)
        gl_MeshVerticesEXT[i].gl_Position = vec4(i == 1 ? 1.0 : -1.0, i == 2 ? 1.0 : -1.0, 0.0, 1.0);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
    gl_MeshPrimitivesEXT[0].gl_PrimitiveShadingRateEXT = gl_ShadingRateFlag2HorizontalPixelsEXT | gl_ShadingRateFlag2VerticalPixelsEXT;
})"},
                           {GL_FRAGMENT_SHADER, kRateFS}});
    glUseProgram(p);
    glShadingRateCombinerOpsEXT(GL_FRAGMENT_SHADING_RATE_COMBINER_OP_REPLACE_EXT, GL_FRAGMENT_SHADING_RATE_COMBINER_OP_KEEP_EXT);
    Target t;
    Counter cnt;
    glDrawMeshTasksEXT(1, 1, 1);
    checkRate(t, cnt, 5, 2, 2);
    resetFsr();
    glDeleteProgram(p);
    CHECK_GL();
}

// 4x multisampling: the fragment's coverage mask holds the samples of all its pixels.
TEST_SIZE(fsr_msaa, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, R"(#version 450
#extension GL_EXT_fragment_shading_rate : require
layout(location = 0) out vec4 o0;
layout(location = 1) out vec4 o1;
void main() {
    o0 = vec4(float(gl_ShadingRateEXT), float(bitCount(gl_SampleMaskIn[0])), 0.0, 0.0);
    o1 = vec4(gl_FragCoord.xy, 0.0, 0.0);
})");
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    GLuint ms[2], fbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(2, ms);
    for (int i = 0; i < 2; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, ms[i]);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_RGBA32F, W, H);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, ms[i]);
    }
    const GLenum bufs[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, bufs);
    glViewport(0, 0, 2 * W, 2 * H);
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    Target t;  // resolve
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fbo);
    for (int i = 0; i < 2; i++) {
        glReadBuffer(GL_COLOR_ATTACHMENT0 + i);
        glDrawBuffer(GL_COLOR_ATTACHMENT0 + i);
        glBlitFramebuffer(0, 0, W, H, 0, 0, W, H, GL_COLOR_BUFFER_BIT, GL_NEAREST);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
    std::vector<float> a = t.read(0), b = t.read(1);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            const float* o = &a[(y * W + x) * 4];
            const float* c = &b[(y * W + x) * 4];
            bool ok = o[0] == 5.0f && o[1] == 16.0f && c[0] == (x & ~1) + 1.0f && c[1] == (y & ~1) + 1.0f;
            if (!ok && bad++ < 3) fprintf(stderr, "  pixel (%d,%d): %g %g / %g %g\n", x, y, o[0], o[1], c[0], c[1]);
        }
    CHECK_EQ(bad, 0);
    resetFsr();
    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(2, ms);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// The scissor test applies per pixel: a fragment with one pixel inside the box is shaded, and writes only that pixel.
TEST_SIZE(fsr_scissor, 16, 16) {
    if (!hasFsr()) return;
    GLuint p = compileProgram(kQuadVS, kRateFS);
    glUseProgram(p);
    glUniform1i(glGetUniformLocation(p, "u_primRate"), 0);
    GLuint vao = quadVao();
    glShadingRateEXT(GL_SHADING_RATE_2X2_PIXELS_EXT);
    Target t;
    Counter cnt;
    glEnable(GL_SCISSOR_TEST);
    glScissor(5, 3, 6, 9);  // pixels x 5..10, y 3..11: fragment columns 4..10, rows 2..10
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glDisable(GL_SCISSOR_TEST);
    CHECK_EQ(cnt.value(), 4u * 5u);
    std::vector<float> a = t.read(0), b = t.read(1);
    int bad = 0;
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++) {
            bool in = x >= 5 && x <= 10 && y >= 3 && y <= 11;
            const float* c = &b[(y * W + x) * 4];
            bad += in ? (a[(y * W + x) * 4] != 5 || c[0] != (x & ~1) + 1 || c[1] != (y & ~1) + 1) : a[(y * W + x) * 4] != -1;
        }
    CHECK_EQ(bad, 0);
    resetFsr();
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}
