// Private Metal API (src/mtl/spi.h): wide lines and exact border colors. Without it (MGL_PRIVATE_API off,
// METALGLDRV_MORALLY_WRONG=0) the driver advertises 1-pixel lines only and the tests check that instead.
#include "harness.h"
#include <cmath>

namespace {
GLuint simpleProgram(const char* vs, const char* fs) {
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
}  // namespace

// glLineWidth: a horizontal line through the middle covers `width` rows (GL 4.6 14.5.2.1, aliased wide lines).
TEST(wide_lines) {
    GLfloat range[2] = {0, 0};
    glGetFloatv(GL_ALIASED_LINE_WIDTH_RANGE, range);
    CHECK(range[0] == 1.0f);
    GLuint p = simpleProgram("#version 410 core\nlayout(location = 0) in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0.0, 1.0); }\n",
                             "#version 410 core\nout vec4 o;\nvoid main() { o = vec4(1.0); }\n");
    glUseProgram(p);
    const float v[4] = {-0.8f, 0.01f, 0.8f, 0.01f};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    for (float w : {1.0f, 3.0f, 8.0f, 13.0f, 64.0f}) {
        glLineWidth(w);
        glClearColor(0, 0, 0, 0);
        glClear(GL_COLOR_BUFFER_BIT);
        glDrawArrays(GL_LINES, 0, 2);
        std::vector<uint8_t> px(64 * 64 * 4);
        glReadPixels(0, 0, 64, 64, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        int rows = 0;
        for (int y = 0; y < 64; y++) rows += px[(y * 64 + 32) * 4] > 128;
        int expect = (int)std::min(w, range[1]);
        if (getenv("SPI_VERBOSE")) fprintf(stderr, "  width %g: %d rows (range max %g)\n", w, rows, range[1]);
        CHECK(std::abs(rows - expect) <= 1);
    }
    glLineWidth(1);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(p);
    CHECK_GL();
}

// Cube maps without GL_TEXTURE_CUBE_MAP_SEAMLESS (GL's default): linear filtering stays inside the selected face,
// through the sampler (spi::forceCubeSeams) or the shader's clamping. Random texels, directions across every face
// up to its edges and corners, several LODs. CUBE_SWEEP_OUT=<file> saves the image (to compare both ways).
TEST(cube_nonseamless_sweep) {
    GLuint cube;
    glGenTextures(1, &cube);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    uint32_t rng = 1;
    for (int lvl = 0; lvl < 4; lvl++)
        for (int f = 0; f < 6; f++) {
            int n = 8 >> lvl;
            std::vector<uint32_t> px(n * n);
            for (uint32_t& p : px) p = (rng = rng * 1664525u + 1013904223u) | 0xff000000u;
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + f, lvl, GL_RGBA8, n, n, 0, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
        }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    GLuint prog = simpleProgram("#version 430 core\nvoid main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0; gl_Position = vec4(p, 0, 1); }",
                                R"(#version 430 core
uniform samplerCube s;
out vec4 c;
void main() {
    // x: direction over a face and across its edges; y: face, LOD
    vec2 q = gl_FragCoord.xy;
    float u = (q.x / 64.0) * 2.4 - 1.2, v = fract(q.y / 8.0) * 2.4 - 1.2;
    int face = int(q.y / 8.0) % 6;
    float lod = floor(q.y / 48.0) * 0.75;
    vec3 d = face == 0 ? vec3(1, v, u) : face == 1 ? vec3(-1, v, u) : face == 2 ? vec3(u, 1, v) : face == 3 ? vec3(u, -1, v) : face == 4 ? vec3(u, v, 1) : vec3(u, v, -1);
    c = textureLod(s, d, lod);
})");
    GLuint vao, fbo, rt;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 64, 192);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glViewport(0, 0, 64, 192);
    glUseProgram(prog);
    glBindTexture(GL_TEXTURE_CUBE_MAP, cube);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    std::vector<uint8_t> out(64 * 192 * 4);
    glReadPixels(0, 0, 64, 192, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
    if (const char* f = getenv("CUBE_SWEEP_OUT"))
        if (FILE* fp = fopen(f, "wb")) {
            fwrite(out.data(), 1, out.size(), fp);
            fclose(fp);
        }
    // seamless filtering blends neighbouring faces near the edges: a different image
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    std::vector<uint8_t> seamless(out.size());
    glReadPixels(0, 0, 64, 192, GL_RGBA, GL_UNSIGNED_BYTE, seamless.data());
    glDisable(GL_TEXTURE_CUBE_MAP_SEAMLESS);
    int differ = 0;
    for (size_t i = 0; i < out.size(); i += 4) differ += memcmp(&out[i], &seamless[i], 4) != 0;
    CHECK(differ > 500);
    CHECK_GL();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    GLuint ts[2] = {rt, cube};
    glDeleteTextures(2, ts);
    glDeleteVertexArrays(1, &vao);
}
