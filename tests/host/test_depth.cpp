// Depth textures as Iris uses them: render, sample, copy (CopyImageSubData / CopyTexSubImage / Blit).
#include "harness.h"
#include <vector>

static GLuint prog(const char* vs, const char* fs) {
    GLuint p = glCreateProgram();
    for (int i = 0; i < 2; i++) {
        GLuint s = glCreateShader(i ? GL_FRAGMENT_SHADER : GL_VERTEX_SHADER);
        const char* src = i ? fs : vs;
        glShaderSource(s, 1, &src, nullptr);
        glCompileShader(s);
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    glLinkProgram(p);
    return p;
}

static const char* kQuadVS = R"(#version 330 core
uniform float z;
out vec2 uv;
void main() {
    vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0 - 1.0;
    uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, z, 1);
})";

static GLuint depthTex(GLenum ifmt, GLenum fmt, GLenum type) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, 64, 64, 0, fmt, type, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    return t;
}

// Shows texture `t`'s red channel (depth) as gray in the default framebuffer.
static void showDepth(GLuint t, GLuint vao) {
    GLuint p = prog(kQuadVS, R"(#version 330 core
uniform sampler2D tex; in vec2 uv; out vec4 o; void main() { o = vec4(texture(tex, uv).rrr, 1); })");
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDisable(GL_DEPTH_TEST);
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, t);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDeleteProgram(p);
}

static void runDepthCase(GLenum ifmt, GLenum fmt, GLenum type, GLenum attach) {
    GLuint vao;
    glGenVertexArrays(1, &vao);
    GLuint color;
    glGenTextures(1, &color);
    glBindTexture(GL_TEXTURE_2D, color);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    GLuint d = depthTex(ifmt, fmt, type);
    GLuint fb;
    glGenFramebuffers(1, &fb);
    glBindFramebuffer(GL_FRAMEBUFFER, fb);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, color, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, attach, GL_TEXTURE_2D, d, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glClearDepth(1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    GLuint p = prog(kQuadVS, R"(#version 330 core
out vec4 o; void main() { o = vec4(1); })");
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);  // window depth 0.5
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDisable(GL_DEPTH_TEST);

    showDepth(d, vao);
    CHECK_PIXEL(32, 32, 128, 128, 128, 255);

    // glCopyImageSubData depth -> depth
    GLuint d2 = depthTex(ifmt, fmt, type);
    glCopyImageSubData(d, GL_TEXTURE_2D, 0, 0, 0, 0, d2, GL_TEXTURE_2D, 0, 0, 0, 0, 64, 64, 1);
    showDepth(d2, vao);
    CHECK_PIXEL(32, 32, 128, 128, 128, 255);

    // glCopyTexSubImage2D from the FBO's depth
    GLuint d3 = depthTex(ifmt, fmt, type);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
    glBindTexture(GL_TEXTURE_2D, d3);
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, 64, 64);
    showDepth(d3, vao);
    CHECK_PIXEL(32, 32, 128, 128, 128, 255);

    // glBlitFramebuffer depth into another FBO
    GLuint d4 = depthTex(ifmt, fmt, type);
    GLuint fb2;
    glGenFramebuffers(1, &fb2);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fb2);
    glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, attach, GL_TEXTURE_2D, d4, 0);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb);
    glBlitFramebuffer(0, 0, 64, 64, 0, 0, 64, 64, GL_DEPTH_BUFFER_BIT, GL_NEAREST);
    showDepth(d4, vao);
    CHECK_PIXEL(32, 32, 128, 128, 128, 255);

    GLuint ts[4] = {d, d2, d3, d4};
    glDeleteTextures(4, ts);
    glDeleteTextures(1, &color);
    GLuint fbs[2] = {fb, fb2};
    glDeleteFramebuffers(2, fbs);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

TEST(depth32f_sample_and_copy) { runDepthCase(GL_DEPTH_COMPONENT32F, GL_DEPTH_COMPONENT, GL_FLOAT, GL_DEPTH_ATTACHMENT); }
TEST(depth24_sample_and_copy) { runDepthCase(GL_DEPTH_COMPONENT24, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, GL_DEPTH_ATTACHMENT); }
TEST(depth_unsized_sample_and_copy) { runDepthCase(GL_DEPTH_COMPONENT, GL_DEPTH_COMPONENT, GL_FLOAT, GL_DEPTH_ATTACHMENT); }
TEST(depth24s8_sample_and_copy) { runDepthCase(GL_DEPTH24_STENCIL8, GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, GL_DEPTH_STENCIL_ATTACHMENT); }

// Minecraft's atlas pattern: allocate every mip level with glTexImage2D(NULL), then
// upload each level with glTexSubImage2D; sample with explicit LODs.
TEST(mip_levels_texsubimage_upload) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 4);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, 4);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 0.0f);
    for (int l = 0; l <= 4; l++) glTexImage2D(GL_TEXTURE_2D, l, GL_RGBA8, 64 >> l, 64 >> l, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    for (int l = 0; l <= 4; l++) {
        int s = 64 >> l;
        std::vector<uint32_t> px((size_t)s * s, 0xff000000u | (uint32_t)(l * 50));  // red = 50*level, alpha 255
        glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, s, s, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    }
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint p = prog(kQuadVS, R"(#version 330 core
uniform sampler2D tex; uniform float lod; in vec2 uv; out vec4 o; void main() { o = textureLod(tex, uv, lod); })");
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);
    for (int l = 0; l <= 4; l++) {
        glUniform1f(glGetUniformLocation(p, "lod"), (float)l);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        CHECK_PIXEL(32, 32, l * 50, 0, 0, 255);
    }
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

// Iris terrain atlas state: 5 uploaded levels, MAX_LEVEL 0, sampler object with a mipmap
// filter, implicit-LOD texture() feeding an alpha test.
TEST(atlas_max_level0_sampler_object_alpha) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 4);
    for (int l = 0; l <= 4; l++) glTexImage2D(GL_TEXTURE_2D, l, GL_RGBA8, 64 >> l, 64 >> l, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    for (int l = 0; l <= 4; l++) {
        int s = 64 >> l;
        std::vector<uint32_t> px((size_t)s * s, 0xff00c000u);  // opaque green
        glTexSubImage2D(GL_TEXTURE_2D, l, 0, 0, s, s, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    GLuint smp;
    glGenSamplers(1, &smp);
    glSamplerParameteri(smp, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_LINEAR);
    glSamplerParameteri(smp, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glSamplerParameteri(smp, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glSamplerParameteri(smp, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glBindSampler(0, smp);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint p = prog(kQuadVS, R"(#version 330 core
uniform sampler2D tex; in vec2 uv; out vec4 o;
void main() { vec4 c = texture(tex, uv * 0.05); if (c.a <= 0.5) discard; o = c; })");
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(32, 32, 0, 192, 0, 255);
    glBindSampler(0, 0);
    glDeleteSamplers(1, &smp);
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

// A draw where some fragments discard must keep the others (color and depth).
TEST(partial_discard_keeps_other_fragments) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    std::vector<uint32_t> px(16 * 16, 0xff00ff00u);
    for (int y = 0; y < 16; y++)
        for (int x = 8; x < 16; x++) px[y * 16 + x] = 0x0000ff00u;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint p = prog(kQuadVS, R"(#version 330 core
uniform sampler2D tex; in vec2 uv; out vec4 o;
void main() { vec4 c = texture(tex, uv); if (c.a <= 0.5) discard; o = c; })");
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);  // opaque half drawn
    CHECK_PIXEL(48, 32, 255, 0, 0, 255);  // transparent half discarded
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

// Shape of Complementary's gbuffers_terrain: alpha-test discard, then a function sampling
// in a loop (shadow filter) that feeds the output.
TEST(partial_discard_with_sampling_function) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    std::vector<uint32_t> px(16 * 16, 0xff00ff00u);
    for (int y = 0; y < 16; y++)
        for (int x = 8; x < 16; x++) px[y * 16 + x] = 0x0000ff00u;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    GLuint p = prog(kQuadVS, R"(#version 330 core
uniform sampler2D tex; uniform sampler2D other; uniform int n; in vec2 uv; out vec4 o;
vec3 sampleOther(vec2 c) {
    float a = texture(other, c).r;
    vec3 r = vec3(0.0);
    if (a < 1.0) r = texture(other, c * 0.5).rgb;
    return r * (1.0 - a) + a;
}
vec3 filtered(vec2 c) {
    vec3 s = vec3(0.0);
    for (int i = 0; i < n; i++) s += sampleOther(c + float(i) * 0.01);
    return s / float(n);
}
void main() {
    vec4 c = texture(tex, uv);
    if (c.a <= 0.5) discard;
    c.rgb *= filtered(uv);
    o = c;
})");
    glUseProgram(p);
    glUniform1f(glGetUniformLocation(p, "z"), 0.0f);
    glUniform1i(glGetUniformLocation(p, "tex"), 0);
    glUniform1i(glGetUniformLocation(p, "other"), 1);
    glUniform1i(glGetUniformLocation(p, "n"), 2);
    GLuint white;
    glGenTextures(1, &white);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, white);
    uint32_t wpx = 0xffffffffu;
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_RGBA, GL_UNSIGNED_BYTE, &wpx);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glActiveTexture(GL_TEXTURE0);
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(16, 32, 0, 255, 0, 255);
    CHECK_PIXEL(48, 32, 255, 0, 0, 255);
    glDeleteProgram(p);
    glDeleteTextures(1, &t);
    glDeleteTextures(1, &white);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}
