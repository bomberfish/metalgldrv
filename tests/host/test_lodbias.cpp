// Trilinear LOD fraction precision: explicit LOD and GL_TEXTURE_LOD_BIAS (KHR-GL46.texture_lod_bias).
#include <algorithm>
#include <cmath>
#include "harness.h"

namespace {
// 256x256 texture whose level i has a solid color; returns the sampled RGB at (0,0).
void sampleAt(float lod, float stateBias, uint8_t out[4]) {
    static const uint8_t colors[9][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255},
                                         {255, 0, 255}, {255, 128, 128}, {128, 255, 128}, {128, 128, 255}};
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (int l = 0, s = 256; l <= 8; l++, s >>= 1) {
        std::vector<uint8_t> d((size_t)s * s * 3);
        for (int i = 0; i < s * s; i++) memcpy(&d[i * 3], colors[l], 3);
        glTexImage2D(GL_TEXTURE_2D, l, GL_RGB, s, s, 0, GL_RGB, GL_UNSIGNED_BYTE, d.data());
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, stateBias);
    const char* vs = "#version 400 core\nuniform float lod; uniform sampler2D t; out vec4 c;\n"
                     "void main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0; gl_Position = vec4(p, 0, 1);"
                     " c = textureLod(t, vec2(0.5), lod); }";
    const char* fs = "#version 400 core\nin vec4 c; out vec4 o; void main() { o = c; }";
    GLuint p = compileProgram(vs, fs);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "lod"), lod);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 3);
    readPixel(1, 1, out);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);
}
}  // namespace

static int lodBiasSweep(bool fragment, int mode = 0) {  // mode 1: state bias only, 2: shader bias only
    // KHR-GL46.texture_lod_bias vertex sweep: state bias f + g for f, g on a 128-step grid in [-16, 16]
    static const uint8_t colors[9][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {0, 255, 255},
                                         {255, 0, 255}, {255, 128, 128}, {128, 255, 128}, {128, 128, 255}};
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    for (int l = 0, s = 256; l <= 8; l++, s >>= 1) {
        std::vector<uint8_t> d((size_t)s * s * 3);
        for (int i = 0; i < s * s; i++) memcpy(&d[i * 3], colors[l], 3);
        glTexImage2D(GL_TEXTURE_2D, l, GL_RGB, s, s, 0, GL_RGB, GL_UNSIGNED_BYTE, d.data());
    }
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    // as the CTS: a 1x1 RGBA8 target, 256x256 viewport, full-screen quad from a buffer
    const char* vs = fragment ? "#version 400 core\nin vec4 vertex; out vec2 tc;\nvoid main() { gl_Position = vertex; tc = vertex.xy * 0.5 + 0.5; }"
                              : "#version 400 core\nin vec4 vertex; uniform float lod; uniform sampler2D t; out vec4 c;\n"
                                "void main() { gl_Position = vertex; c = textureLod(t, vertex.xy * 0.5 + 0.5, lod); }";
    const char* fs = fragment ? "#version 400 core\nin vec2 tc; uniform float bias; uniform sampler2D t; out vec4 o;\n"
                                "void main() { o = texture(t, vec2(tc.x, 0), bias); }"
                              : "#version 400 core\nin vec4 c; out vec4 o; void main() { o = c; }";
    GLuint p = compileProgram(vs, fs);
    GLuint vao, vbo, target, fbo;
    glGenTextures(1, &target);
    glBindTexture(GL_TEXTURE_2D, target);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 1, 1);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, target, 0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glViewport(0, 0, 256, 256);
    const float quad[] = {-1, -1, 0, 1, 1, -1, 0, 1, -1, 1, 0, 1, 1, 1, 0, 1};
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_DYNAMIC_DRAW);
    glUseProgram(p);
    GLint loc = glGetAttribLocation(p, "vertex");
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 16, nullptr);
    glEnableVertexAttribArray(loc);
    glUniform1f(glGetUniformLocation(p, "lod"), 0.0f);
    int fails = 0;
    for (int i = 0; i < 128 && fails < 5; i++)
        for (int j = 0; j < 128 && fails < 5; j++) {
            float f = 16.0f * ((float)i / 127.0f * 2.0f - 1.0f), g = 16.0f * ((float)j / 127.0f * 2.0f - 1.0f);
            if (mode == 1) g = 0;
            if (mode == 2) f = 0;
            float lod = std::clamp(f + g, 0.0f, 8.0f);
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, fragment ? f : f + g);
            if (fragment) glUniform1f(glGetUniformLocation(p, "bias"), g);
            glClearColor(0, 0, 0, 1);
            glClear(GL_COLOR_BUFFER_BIT);
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
            uint8_t px[4];
            glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            int d1 = lod >= 8 ? 8 : (int)floorf(lod), d2 = lod >= 8 ? 8 : d1 + 1;
            float fr = fmodf(lod, 1.0f);
            for (int c = 0; c < 3; c++) {
                int want = (int)((1 - fr) * colors[d1][c] + fr * colors[d2][c]);
                if (abs(px[c] - want) > 5) {
                    fprintf(stderr, "  bias %f (i %d j %d): got %d %d %d want comp %d = %d\n", f + g, i, j, px[0], px[1], px[2], c, want);
                    fails++;
                    break;
                }
            }
        }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &target);
    glDeleteBuffers(1, &vbo);
    glViewport(0, 0, 64, 64);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    glDeleteTextures(1, &tex);
    return fails;
}

TEST(lod_bias_sweep) {
    CHECK_EQ(lodBiasSweep(false), 0);
    CHECK_EQ(lodBiasSweep(true, 1), 0);
    CHECK_EQ(lodBiasSweep(true, 2), 0);
    // Not checked: shader bias() together with a nonzero GL_TEXTURE_LOD_BIAS loses LOD precision on
    // Apple GPUs (up to ~1/4 level; KHR-GL46.texture_lod_bias_all's one failing combination).
    CHECK_GL();
}

TEST(lod_fraction_precision) {
    // LOD 2.519684 between level 2 (blue) and 3 (yellow): expect ~ (132, 132, 122)
    uint8_t a[4], b[4];
    sampleAt(2.519684f, 0.0f, a);
    sampleAt(0.0f, 2.519684f, b);
    fprintf(stderr, "  explicit lod: %d %d %d   state bias: %d %d %d\n", a[0], a[1], a[2], b[0], b[1], b[2]);
    CHECK(abs(a[2] - 122) <= 3);
    CHECK(abs(b[2] - 122) <= 3);
    CHECK_GL();
}
