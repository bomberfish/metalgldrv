// Geometry/tessellation/transform feedback/subroutines/provoking vertex through the GL runtime.
#include "harness.h"
#include <vector>

static GLuint makeShader(GLenum type, const char* src) {
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

static GLuint linkShaders(std::initializer_list<GLuint> shaders, std::initializer_list<const char*> xfb = {}) {
    GLuint p = glCreateProgram();
    for (GLuint s : shaders) glAttachShader(p, s);
    if (xfb.size()) {
        std::vector<const char*> v(xfb);
        glTransformFeedbackVaryings(p, (GLsizei)v.size(), v.data(), GL_INTERLEAVED_ATTRIBS);
    }
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link failed: %s\n", log);
    }
    for (GLuint s : shaders) glDeleteShader(s);
    return p;
}

static GLuint fullscreenVao() {
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    float v[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof v, v, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    return vao;
}

TEST(flat_provoking_vertex) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 p;
flat out vec4 c;
void main() {
    c = gl_VertexID == 0 ? vec4(1,0,0,1) : gl_VertexID == 1 ? vec4(0,1,0,1) : vec4(0,0,1,1);
    gl_Position = vec4(p, 0, 1);
})";
    const char* fs = R"(#version 330 core
flat in vec4 c; out vec4 o; void main() { o = c; })";
    GLuint p = linkShaders({makeShader(GL_VERTEX_SHADER, vs), makeShader(GL_FRAGMENT_SHADER, fs)});
    GLuint vao = fullscreenVao();
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_PIXEL(4, 4, 0, 0, 255, 255);  // last vertex (GL default)
    glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    CHECK_PIXEL(4, 4, 255, 0, 0, 255);
    glProvokingVertex(GL_LAST_VERTEX_CONVENTION);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

TEST(geometry_shader) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 p; void main() { gl_Position = vec4(p * 0.5, 0, 1); })";
    // Emits the input triangle twice: shifted left (red) and right (green).
    const char* gs = R"(#version 330 core
layout(triangles) in;
layout(triangle_strip, max_vertices = 6) out;
out vec4 col;
void main() {
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            gl_Position = gl_in[i].gl_Position + vec4(k == 0 ? -0.5 : 0.5, 0, 0, 0);
            col = k == 0 ? vec4(1,0,0,1) : vec4(0,1,0,1);
            EmitVertex();
        }
        EndPrimitive();
    }
})";
    const char* fs = R"(#version 330 core
in vec4 col; out vec4 o; void main() { o = col; })";
    GLuint p = linkShaders({makeShader(GL_VERTEX_SHADER, vs), makeShader(GL_GEOMETRY_SHADER, gs), makeShader(GL_FRAGMENT_SHADER, fs)});
    GLuint vao = fullscreenVao();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);  // quad (-0.5..0.5) -> two copies
    CHECK_PIXEL(10, 32, 255, 0, 0, 255);   // x ~ -0.69
    CHECK_PIXEL(54, 32, 0, 255, 0, 255);   // x ~ +0.69
    CHECK_PIXEL(32, 60, 0, 0, 0, 255);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

TEST(tessellation_quad) {
    const char* vs = R"(#version 400 core
layout(location = 0) in vec2 p; void main() { gl_Position = vec4(p, 0, 1); })";
    const char* tcs = R"(#version 400 core
layout(vertices = 4) out;
void main() {
    gl_out[gl_InvocationID].gl_Position = gl_in[gl_InvocationID].gl_Position;
    if (gl_InvocationID == 0) {
        gl_TessLevelOuter[0] = gl_TessLevelOuter[1] = gl_TessLevelOuter[2] = gl_TessLevelOuter[3] = 4.0;
        gl_TessLevelInner[0] = gl_TessLevelInner[1] = 4.0;
    }
})";
    // Maps the unit square domain onto the bottom-left quadrant of the screen.
    const char* tes = R"(#version 400 core
layout(quads, equal_spacing, ccw) in;
out vec2 uv;
void main() {
    uv = gl_TessCoord.xy;
    gl_Position = vec4(gl_TessCoord.xy - 1.0, 0, 1);
})";
    const char* fs = R"(#version 400 core
in vec2 uv; out vec4 o; void main() { o = vec4(uv, 1, 1); })";
    GLuint p = linkShaders({makeShader(GL_VERTEX_SHADER, vs), makeShader(GL_TESS_CONTROL_SHADER, tcs),
                            makeShader(GL_TESS_EVALUATION_SHADER, tes), makeShader(GL_FRAGMENT_SHADER, fs)});
    GLuint vao = fullscreenVao();
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    glPatchParameteri(GL_PATCH_VERTICES, 4);
    glDrawArrays(GL_PATCHES, 0, 4);
    uint8_t px[4];
    readPixel(8, 8, px);
    CHECK(px[2] == 255 && px[0] > 30 && px[0] < 100);  // uv ~ (0.25, 0.25)
    CHECK_PIXEL(48, 48, 0, 0, 0, 255);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

TEST(transform_feedback_capture) {
    const char* vs = R"(#version 330 core
layout(location = 0) in vec2 p;
out vec4 outv;
void main() { outv = vec4(p * 2.0, float(gl_VertexID), 1.0); gl_Position = vec4(p, 0, 1); })";
    const char* fs = R"(#version 330 core
out vec4 o; void main() { o = vec4(1); })";
    GLuint p = linkShaders({makeShader(GL_VERTEX_SHADER, vs), makeShader(GL_FRAGMENT_SHADER, fs)}, {"outv"});
    GLuint vao = fullscreenVao();
    GLuint xb;
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, 3 * 16, nullptr, GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_TRIANGLES);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    float out[12] = {};
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, sizeof out, out);
    CHECK(out[0] == -2 && out[1] == -2 && out[2] == 0);
    CHECK(out[4] == 2 && out[5] == -2 && out[6] == 1);
    CHECK(out[8] == -2 && out[9] == 2 && out[10] == 2);
    glDeleteBuffers(1, &xb);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}

TEST(subroutines) {
    const char* vs = R"(#version 400 core
layout(location = 0) in vec2 p; void main() { gl_Position = vec4(p, 0, 1); })";
    const char* fs = R"(#version 400 core
subroutine vec4 Colorer();
subroutine(Colorer) vec4 red() { return vec4(1,0,0,1); }
subroutine(Colorer) vec4 blue() { return vec4(0,0,1,1); }
subroutine uniform Colorer pick;
out vec4 o;
void main() { o = pick(); })";
    GLuint p = linkShaders({makeShader(GL_VERTEX_SHADER, vs), makeShader(GL_FRAGMENT_SHADER, fs)});
    GLuint vao = fullscreenVao();
    glUseProgram(p);
    GLuint iBlue = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "blue");
    GLuint iRed = glGetSubroutineIndex(p, GL_FRAGMENT_SHADER, "red");
    CHECK(iBlue != GL_INVALID_INDEX && iRed != GL_INVALID_INDEX);
    GLint loc = glGetSubroutineUniformLocation(p, GL_FRAGMENT_SHADER, "pick");
    CHECK_EQ(loc, 0);
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &iBlue);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(30, 30, 0, 0, 255, 255);
    glUniformSubroutinesuiv(GL_FRAGMENT_SHADER, 1, &iRed);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    CHECK_PIXEL(30, 30, 255, 0, 0, 255);
    glDeleteProgram(p);
    glDeleteVertexArrays(1, &vao);
    CHECK_GL();
}
