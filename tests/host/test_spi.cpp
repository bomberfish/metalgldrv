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
