// Geometry shader / tessellation / transform feedback paths (multi-pass pipelines).
#include "harness.h"
#include <vector>

namespace {

GLuint mkShader(GLenum type, const char* src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetShaderInfoLog(s, sizeof log, nullptr, log);
        fprintf(stderr, "  compile failed: %s\n", log);
    }
    return s;
}

GLuint mkProgram(std::initializer_list<std::pair<GLenum, const char*>> stages, std::initializer_list<const char*> xfb = {},
                 GLenum xfbMode = GL_INTERLEAVED_ATTRIBS) {
    GLuint p = glCreateProgram();
    std::vector<GLuint> sh;
    for (auto& [t, src] : stages) {
        sh.push_back(mkShader(t, src));
        glAttachShader(p, sh.back());
    }
    if (xfb.size()) {
        std::vector<const char*> v(xfb);
        glTransformFeedbackVaryings(p, (GLsizei)v.size(), v.data(), xfbMode);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link failed: %s\n", log);
    }
    for (GLuint s : sh) glDeleteShader(s);
    return p;
}

struct Fbo {
    GLuint fbo = 0, tex = 0;
    int w, h;
    Fbo(int w_, int h_) : w(w_), h(h_) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, w, h);
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
        glViewport(0, 0, w, h);
    }
    ~Fbo() {
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
        glDeleteTextures(1, &tex);
    }
    std::vector<uint8_t> read() {
        std::vector<uint8_t> d(w * h * 4);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, d.data());
        return d;
    }
    void dump() {
        auto d = read();
        for (int y = h - 1; y >= 0; y--) {
            for (int x = 0; x < w; x++) {
                const uint8_t* p = &d[(y * w + x) * 4];
                fprintf(stderr, "%c", (p[0] | p[1] | p[2] | p[3]) ? '#' : '.');
            }
            fprintf(stderr, "\n");
        }
    }
};

}  // namespace

// geometry_shader.rendering points -> points (9 points per input point, 38x3 target)
TEST(gs_points_to_points) {
    const char* vs = R"(#version 400
in vec4 position;
out vec4 vs_gs_color;
uniform ivec2 renderingTargetSize;
void main() {
    gl_PointSize = 1.0;
    if (renderingTargetSize.y == 3) gl_Position = position;
    else gl_Position = vec4(position.x, -1.0 + (1.5 + float(5 * gl_InstanceID)) / float(renderingTargetSize.y) * 2.0, position.zw);
    switch (gl_VertexID) {
        case 0: vs_gs_color = vec4(1, 0, 0, 0); break;
        case 1: vs_gs_color = vec4(0, 1, 0, 0); break;
        case 2: vs_gs_color = vec4(0, 0, 1, 0); break;
        case 3: vs_gs_color = vec4(0, 0, 0, 1); break;
        case 4: vs_gs_color = vec4(1, 1, 0, 0); break;
        case 5: vs_gs_color = vec4(1, 0, 1, 0); break;
        case 6: vs_gs_color = vec4(1, 0, 0, 1); break;
        case 7: vs_gs_color = vec4(1, 1, 1, 0); break;
        case 8: vs_gs_color = vec4(1, 1, 0, 1); break;
    }
})";
    const char* gs = R"(#version 400
layout(points) in;
layout(points, max_vertices=9) out;
in vec4 vs_gs_color[1];
out vec4 gs_fs_color;
uniform ivec2 renderingTargetSize;
void main() {
    float dx = 2.0 / float(renderingTargetSize.x);
    float dy = 2.0 / float(renderingTargetSize.y);
    for (int j = -1; j <= 1; j++)
        for (int i = -1; i <= 1; i++) {
            gs_fs_color = vs_gs_color[0];
            gl_Position = gl_in[0].gl_Position + vec4(float(i) * dx, float(j) * dy, 0, 0);
            EmitVertex();
        }
})";
    const char* fs = R"(#version 400
in vec4 gs_fs_color; out vec4 result; void main() { result = gs_fs_color; })";
    GLuint p = mkProgram({{GL_VERTEX_SHADER, vs}, {GL_GEOMETRY_SHADER, gs}, {GL_FRAGMENT_SHADER, fs}});
    Fbo fb(38, 3);
    float data[8 * 4];
    for (int n = 0; n < 8; n++) {
        data[n * 4 + 0] = -1 + ((float(1 + 5 * n) + 0.5f) / 38.0f) * 2.0f;
        data[n * 4 + 1] = -1 + (2.0f / 3.0f) * 2.0f;
        data[n * 4 + 2] = 0;
        data[n * 4 + 3] = 1;
    }
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
    GLint loc = glGetAttribLocation(p, "position");
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(loc);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glUniform2i(glGetUniformLocation(p, "renderingTargetSize"), 38, 3);
    glDrawArrays(GL_POINTS, 0, 8);
    static const uint8_t ref[8][4] = {{255, 0, 0, 0}, {0, 255, 0, 0}, {0, 0, 255, 0}, {0, 0, 0, 255},
                                      {255, 255, 0, 0}, {255, 0, 255, 0}, {255, 0, 0, 255}, {255, 255, 255, 0}};
    for (int n = 0; n < 8; n++) CHECK_PIXEL(n * 5 + 1, 1, ref[n][0], ref[n][1], ref[n][2], ref[n][3]);
    if (g_failures) fb.dump();
    // instanced: 4 instances into a 38x18 target
    {
        Fbo fi(38, 18);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2i(glGetUniformLocation(p, "renderingTargetSize"), 38, 18);
        glDrawArraysInstanced(GL_POINTS, 0, 8, 4);
        for (int i = 0; i < 4; i++)
            for (int n = 0; n < 8; n++) CHECK_PIXEL(n * 5 + 1, i * 5 + 1, ref[n][0], ref[n][1], ref[n][2], ref[n][3]);
        if (g_failures) fi.dump();
    }
    // indexed (reversed vertex data, unordered ubyte indices)
    {
        float rev[8 * 4];
        for (int n = 0; n < 8; n++) memcpy(rev + n * 4, data + (7 - n) * 4, 16);
        uint8_t idx[8];
        for (int n = 0; n < 8; n++) idx[n] = (uint8_t)((n * 2) % 8 + n / 4);
        GLuint vb2, ib;
        glGenBuffers(1, &vb2);
        glBindBuffer(GL_ARRAY_BUFFER, vb2);
        glBufferData(GL_ARRAY_BUFFER, sizeof rev, rev, GL_STATIC_DRAW);
        glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glGenBuffers(1, &ib);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ib);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        Fbo fe(38, 3);
        glClear(GL_COLOR_BUFFER_BIT);
        glUniform2i(glGetUniformLocation(p, "renderingTargetSize"), 38, 3);
        glDrawElements(GL_POINTS, 8, GL_UNSIGNED_BYTE, nullptr);
        for (int n = 0; n < 8; n++) {
            int index = idx[n];
            const uint8_t* r = ref[7 - index];
            CHECK_PIXEL(index * 5 + 1, 1, r[0], r[1], r[2], r[3]);
        }
        if (g_failures) fe.dump();
        glDeleteBuffers(1, &vb2);
        glDeleteBuffers(1, &ib);
    }
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Layered framebuffer without a geometry shader: only layer 0 is rendered (GL 4.6 9.8).
TEST(layered_fbo_no_gs_renders_layer0) {
    const char* vs = R"(#version 400
in vec4 pos; void main() { gl_Position = pos; })";
    const char* fs = R"(#version 400
out vec4 o; void main() { o = vec4(1.0); })";
    GLuint p = mkProgram({{GL_VERTEX_SHADER, vs}, {GL_FRAGMENT_SHADER, fs}});
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_3D, tex);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA8, 4, 4, 2);
    std::vector<uint8_t> zero(4 * 4 * 2 * 4, 0);
    glTexSubImage3D(GL_TEXTURE_3D, 0, 0, 0, 0, 4, 4, 2, GL_RGBA, GL_UNSIGNED_BYTE, zero.data());
    GLuint fbo, rfbo;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_DRAW_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    float v[] = {-1, -1, 0, 1, 3, -1, 0, 1, -1, 3, 0, 1};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGenFramebuffers(1, &rfbo);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, rfbo);
    for (int layer = 0; layer < 2; layer++) {
        glFramebufferTextureLayer(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, layer);
        uint8_t px[4];
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
        CHECK_EQ((int)px[0], layer == 0 ? 255 : 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteFramebuffers(1, &rfbo);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}
