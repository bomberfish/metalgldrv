// GL_ARB_enhanced_layouts through the GL runtime: transform feedback qualifiers, component-packed vertex inputs.
#include "harness.h"
#include <vector>

static GLuint buildProgram(std::initializer_list<std::pair<GLenum, const char*>> stages, std::vector<const char*> xfb = {},
                           GLenum xfbMode = GL_INTERLEAVED_ATTRIBS, bool separable = false) {
    GLuint p = glCreateProgram();
    if (separable) glProgramParameteri(p, GL_PROGRAM_SEPARABLE, GL_TRUE);
    for (auto& [type, src] : stages) {
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
        glAttachShader(p, s);
        glDeleteShader(s);
    }
    if (!xfb.empty()) glTransformFeedbackVaryings(p, (GLsizei)xfb.size(), xfb.data(), xfbMode);
    glLinkProgram(p);
    GLint ok = 0;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[4096];
        glGetProgramInfoLog(p, sizeof log, nullptr, log);
        fprintf(stderr, "  link failed: %s\n", log);
    }
    return p;
}

static GLint xfbProp(GLuint p, GLenum iface, GLuint index, GLenum prop) {
    GLint v = -12345;
    glGetProgramResourceiv(p, iface, index, 1, &prop, 1, nullptr, &v);
    return v;
}

TEST(xfb_qualifier_resources) {
    const char* vs = R"(#version 430 core
const uint type_size = 8;
layout (xfb_buffer = 1, xfb_stride = 4 * type_size) out;
layout (xfb_buffer = 0, xfb_offset = 1 * type_size) out vec2 b0_v1;
layout (xfb_buffer = 1, xfb_offset = 1 * type_size) out vec2 b1_v1;
layout (xfb_buffer = 0, xfb_offset = 3 * type_size) out vec2 b0_v3;
layout (xfb_buffer = 0, xfb_offset = 0 * type_size) out vec2 b0_v0;
in vec4 in_vs;
void main() { b0_v1 = vec2(1); b1_v1 = vec2(2); b0_v3 = vec2(3); b0_v0 = in_vs.xy; })";
    GLuint p = buildProgram({{GL_VERTEX_SHADER, vs}}, {}, GL_INTERLEAVED_ATTRIBS, true);
    GLuint i00 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b0_v0");
    GLuint i03 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b0_v3");
    GLuint i11 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b1_v1");
    CHECK(i00 != GL_INVALID_INDEX && i03 != GL_INVALID_INDEX && i11 != GL_INVALID_INDEX);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i03, GL_OFFSET), 24);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i03, GL_TYPE), GL_FLOAT_VEC2);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(xfb_api_skip_components_resources) {
    const char* vs = R"(#version 430 core
out vec2 b0_v1; out vec2 b1_v1; out vec2 b0_v3; out vec2 b0_v0;
in vec4 in_vs;
void main() { b0_v1 = vec2(1); b1_v1 = vec2(2); b0_v3 = vec2(3); b0_v0 = in_vs.xy; })";
    GLuint p = buildProgram({{GL_VERTEX_SHADER, vs}},
                            {"b0_v0", "b0_v1", "gl_SkipComponents2", "b0_v3", "gl_NextBuffer", "gl_SkipComponents2", "b1_v1",
                             "gl_SkipComponents4"},
                            GL_INTERLEAVED_ATTRIBS, true);
    GLuint i03 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b0_v3");
    GLuint i11 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b1_v1");
    CHECK(i03 != GL_INVALID_INDEX && i11 != GL_INVALID_INDEX);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i03, GL_OFFSET), 24);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i11, GL_OFFSET), 8);
    glDeleteProgram(p);
    CHECK_GL();
}

TEST(xfb_qualifier_resources_separable_gs) {
    const char* gs = R"(#version 430 core
layout(points) in;
layout(triangle_strip, max_vertices = 4) out;
layout (xfb_buffer = 1, xfb_stride = 16) out;
layout (xfb_buffer = 0, xfb_offset = 4) out float b0_v1[1];
layout (xfb_buffer = 1, xfb_offset = 4) out float b1_v1[1];
layout (xfb_buffer = 0, xfb_offset = 0) out float b0_v0[1];
out gl_PerVertex { vec4 gl_Position; };
in vec4 tes_gs[];
out vec4 gs_fs;
void main() {
    b0_v1[0] = 0; b1_v1[0] = 1; b0_v0[0] = 1;
    gs_fs = tes_gs[0];
    gl_Position = vec4(-1, -1, 0, 1);
    EmitVertex();
})";
    GLuint p = buildProgram({{GL_GEOMETRY_SHADER, gs}}, {}, GL_INTERLEAVED_ATTRIBS, true);
    GLuint i11 = glGetProgramResourceIndex(p, GL_TRANSFORM_FEEDBACK_VARYING, "b1_v1");
    CHECK(i11 != GL_INVALID_INDEX);
    glDeleteProgram(p);
    CHECK_GL();
}

// Captures one vertex through transform feedback buffer 0 and returns the first `n` floats.
static std::vector<float> captureVertex(GLuint p, const float attrib[4], int n) {
    GLuint vao, vbo, xb;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 16, attrib, GL_STATIC_DRAW);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(0);
    glGenBuffers(1, &xb);
    glBindBuffer(GL_TRANSFORM_FEEDBACK_BUFFER, xb);
    std::vector<float> out(n, -1.0f);
    glBufferData(GL_TRANSFORM_FEEDBACK_BUFFER, n * 4, out.data(), GL_STREAM_READ);
    glBindBufferBase(GL_TRANSFORM_FEEDBACK_BUFFER, 0, xb);
    glUseProgram(p);
    glEnable(GL_RASTERIZER_DISCARD);
    glBeginTransformFeedback(GL_POINTS);
    glDrawArrays(GL_POINTS, 0, 1);
    glEndTransformFeedback();
    glDisable(GL_RASTERIZER_DISCARD);
    glGetBufferSubData(GL_TRANSFORM_FEEDBACK_BUFFER, 0, n * 4, out.data());
    glDeleteBuffers(1, &xb);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    return out;
}

// layout(component) vertex inputs sharing one attribute location; xfb_offset on block members.
TEST(component_packed_vertex_inputs_and_xfb_block) {
    const char* vs = R"(#version 440 core
layout(location = 0, component = 0) in float a;
layout(location = 0, component = 1) in vec2 b;
layout(location = 0, component = 3) in float c;
layout(xfb_offset = 4) out Blk { float x; vec2 y; layout(xfb_offset = 0) float w; } blk;
void main() { blk.x = c; blk.y = b; blk.w = a; gl_Position = vec4(0); })";
    GLuint p = buildProgram({{GL_VERTEX_SHADER, vs}});
    const float attrib[4] = {1, 2, 3, 4};
    std::vector<float> out = captureVertex(p, attrib, 4);
    CHECK(out[0] == 1 && out[1] == 4 && out[2] == 2 && out[3] == 3);
    glDeleteProgram(p);
    CHECK_GL();
}
