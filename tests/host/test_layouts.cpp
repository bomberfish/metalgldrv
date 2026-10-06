// GL_ARB_enhanced_layouts through the GL runtime: transform feedback qualifiers and their program interface.
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
    GLint b0 = xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i00, GL_TRANSFORM_FEEDBACK_BUFFER_INDEX);
    GLint b1 = xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i11, GL_TRANSFORM_FEEDBACK_BUFFER_INDEX);
    CHECK_EQ(b0, 0);
    CHECK_EQ(b1, 1);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, b0, GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE), 32);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, b1, GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE), 32);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, b1, GL_BUFFER_BINDING), 1);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, b0, GL_NUM_ACTIVE_VARIABLES), 3);
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
    GLint b1 = xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i11, GL_TRANSFORM_FEEDBACK_BUFFER_INDEX);
    CHECK_EQ(b1, 1);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, 0, GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE), 32);
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_BUFFER, b1, GL_TRANSFORM_FEEDBACK_BUFFER_STRIDE), 32);
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
    CHECK_EQ(xfbProp(p, GL_TRANSFORM_FEEDBACK_VARYING, i11, GL_TRANSFORM_FEEDBACK_BUFFER_INDEX), 1);
    glDeleteProgram(p);
    CHECK_GL();
}
