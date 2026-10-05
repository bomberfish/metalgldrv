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
