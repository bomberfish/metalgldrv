// Optional extensions: ARB_shader_viewport_layer_array, ...
#include <cmath>
#include "harness.h"

namespace {
bool hasExtension(const char* name) {
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    for (GLint i = 0; i < n; i++)
        if (!strcmp((const char*)glGetStringi(GL_EXTENSIONS, i), name)) return true;
    return false;
}

void drawQuad(GLuint prog) {
    static const float pos[] = {-1, -1, 1, -1, -1, 1, 1, 1};
    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
    GLint loc = glGetAttribLocation(prog, "pos");
    glVertexAttribPointer(loc, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
    glEnableVertexAttribArray(loc);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
}
}  // namespace

// gl_ViewportIndex written by the vertex shader selects the viewport.
TEST(viewport_index_from_vertex_shader) {
    if (!hasExtension("GL_ARB_shader_viewport_layer_array")) return;
    const char* vs = "#version 450 core\n#extension GL_ARB_shader_viewport_layer_array : require\n"
                     "in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); gl_ViewportIndex = 2; }";
    const char* fs = "#version 450 core\nout vec4 c;\nvoid main() { c = vec4(1, 0, 0, 1); }";
    GLuint p = compileProgram(vs, fs);
    glUseProgram(p);
    glViewportIndexedf(0, 0, 0, 16, 16);
    glViewportIndexedf(1, 16, 0, 16, 16);
    glViewportIndexedf(2, 32, 0, 16, 16);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    drawQuad(p);
    CHECK_PIXEL(8, 8, 0, 0, 0, 255);
    CHECK_PIXEL(40, 8, 255, 0, 0, 255);
    CHECK_GL();
}

// gl_Layer written by the vertex shader selects the layer of a layered framebuffer.
TEST(layer_from_vertex_shader) {
    if (!hasExtension("GL_ARB_shader_viewport_layer_array")) return;
    const char* vs = "#version 450 core\n#extension GL_ARB_shader_viewport_layer_array : require\n"
                     "in vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); gl_Layer = 2; }";
    const char* fs = "#version 450 core\nout vec4 c;\nvoid main() { c = vec4(0, 1, 0, 1); }";
    GLuint p = compileProgram(vs, fs);
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D_ARRAY, tex);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, 16, 16, 4);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0);
    glViewport(0, 0, 16, 16);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glUseProgram(p);
    drawQuad(p);
    for (int l = 0; l < 4; l++) {
        glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, tex, 0, l);
        CHECK_PIXEL(8, 8, 0, l == 2 ? 255 : 0, 0, 255);
    }
    CHECK_GL();
}

// Mirror of KHR-GL46.shader_viewport_layer_array.ShaderLayerFramebufferNonLayeredTestCase:
// gl_Layer from VS and from VS+tess into a non-layered 2D attachment.
TEST(layer_written_nonlayered_fbo) {
    if (!hasExtension("GL_ARB_shader_viewport_layer_array")) return;
    const char* vs = "#version 450 core\n#extension GL_ARB_shader_viewport_layer_array: require\n"
                     "in highp vec2 inPosition;\nin int inLayer;\nin highp vec4 inColor;\nout int vsLayer;\n"
                     "out highp vec3 vsPosition;\nout highp vec4 vsColor;\nvoid main()\n{\n"
                     "    gl_Position = vec4(inPosition, 0.0, 1.0);\n    gl_Layer = (inLayer + 0) % 1;\n"
                     "    vsLayer = inLayer;\n    vsPosition = vec3(inPosition, 0.0);\n    vsColor = inColor;\n}\n";
    const char* tcs = "#version 450 core\nlayout(vertices = 3) out;\nin highp vec3 vsPosition[];\nin highp vec4 vsColor[];\n"
                      "in int vsLayer[];\nout highp vec3 tcsPosition[];\nout highp vec4 tcsColor[];\nout int tcsLayer[];\n"
                      "void main()\n{\n    tcsPosition[gl_InvocationID] = vsPosition[gl_InvocationID];\n"
                      "    tcsColor[gl_InvocationID] = vsColor[gl_InvocationID];\n    tcsLayer[gl_InvocationID] = vsLayer[gl_InvocationID];\n"
                      "    gl_TessLevelInner[0] = 3;\n    gl_TessLevelOuter[0] = 3;\n    gl_TessLevelOuter[1] = 3;\n    gl_TessLevelOuter[2] = 3;\n}\n";
    const char* tes = "#version 450 core\n#extension GL_ARB_shader_viewport_layer_array: require\n"
                      "layout(triangles, equal_spacing, cw) in;\nin highp vec3 tcsPosition[];\nin highp vec4 tcsColor[];\n"
                      "in int tcsLayer[];\nout highp vec4 tesColor;\nout highp int tesLayer;\nvoid main()\n{\n"
                      "    vec3 p0 = gl_TessCoord.x * tcsPosition[0];\n    vec3 p1 = gl_TessCoord.y * tcsPosition[1];\n"
                      "    vec3 p2 = gl_TessCoord.z * tcsPosition[2];\n    tesColor = tcsColor[0];\n    tesLayer = tcsLayer[0];\n"
                      "    gl_Layer = (tcsLayer[0] + 1) % 1;\n    gl_Position = vec4(normalize(p0 + p1 + p2), 1.0);\n}\n";
    const char* fsv = "#version 450 core\nin highp vec4 vsColor;\nout vec4 finalOutColor;\nvoid main() { finalOutColor = vsColor; }\n";
    const char* fst = "#version 450 core\nin highp vec4 tesColor;\nout vec4 finalOutColor;\nvoid main() { finalOutColor = tesColor; }\n";
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 64, 64, 0, GL_RGBA, GL_FLOAT, nullptr);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    glViewport(0, 0, 64, 64);
    for (int tess = 0; tess < 2; tess++) {
        const char* srcs[4] = {vs, tess ? tcs : nullptr, tess ? tes : nullptr, tess ? fst : fsv};
        GLenum types[4] = {GL_VERTEX_SHADER, GL_TESS_CONTROL_SHADER, GL_TESS_EVALUATION_SHADER, GL_FRAGMENT_SHADER};
        GLuint p = glCreateProgram();
        for (int i = 0; i < 4; i++) {
            if (!srcs[i]) continue;
            GLuint s = glCreateShader(types[i]);
            glShaderSource(s, 1, &srcs[i], nullptr);
            glCompileShader(s);
            glAttachShader(p, s);
        }
        glLinkProgram(p);
        GLint ok = 0;
        glGetProgramiv(p, GL_LINK_STATUS, &ok);
        CHECK(ok);
        glUseProgram(p);
        static const float pos[] = {-1, -1, -1, 1, 1, -1, 1, 1};
        static const int layer[] = {0, 0, 0, 0};
        static const float col[] = {1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1};
        static const uint16_t idx[] = {0, 1, 2, 2, 1, 3};
        GLuint vao, vb[4];
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(4, vb);
        glBindBuffer(GL_ARRAY_BUFFER, vb[0]);
        glBufferData(GL_ARRAY_BUFFER, sizeof pos, pos, GL_STATIC_DRAW);
        GLint l = glGetAttribLocation(p, "inPosition");
        glVertexAttribPointer(l, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(l);
        glBindBuffer(GL_ARRAY_BUFFER, vb[1]);
        glBufferData(GL_ARRAY_BUFFER, sizeof layer, layer, GL_STATIC_DRAW);
        l = glGetAttribLocation(p, "inLayer");
        glVertexAttribIPointer(l, 1, GL_INT, 0, nullptr);
        glEnableVertexAttribArray(l);
        glBindBuffer(GL_ARRAY_BUFFER, vb[2]);
        glBufferData(GL_ARRAY_BUFFER, sizeof col, col, GL_STATIC_DRAW);
        l = glGetAttribLocation(p, "inColor");
        glVertexAttribPointer(l, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
        glEnableVertexAttribArray(l);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vb[3]);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof idx, idx, GL_STATIC_DRAW);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        if (tess) glPatchParameteri(GL_PATCH_VERTICES, 3);
        glDrawElements(tess ? GL_PATCHES : GL_TRIANGLE_STRIP, 6, GL_UNSIGNED_SHORT, nullptr);
        CHECK_PIXEL(32, 32, 255, 0, 0, 255);
        CHECK_GL();
    }
}

// ARB_texture_filter_minmax: a 2x2 R8 texture {0, 255, 255, 255} sampled at its center with
// GL_LINEAR gives the average, the minimum or the maximum of the four texels.
TEST(texture_filter_minmax) {
    if (!hasExtension("GL_ARB_texture_filter_minmax")) return;
    const char* vs = "#version 330 core\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); }";
    const char* fs = "#version 330 core\nuniform sampler2D s;\nout vec4 c;\nvoid main() { c = texture(s, vec2(0.5)); }";
    GLuint p = compileProgram(vs, fs);
    glUseProgram(p);
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    static const uint8_t texels[] = {0, 255, 255, 255};
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, 2, 2, 0, GL_RED, GL_UNSIGNED_BYTE, texels);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    const GLenum modes[] = {GL_WEIGHTED_AVERAGE_ARB, GL_MIN, GL_MAX};
    const int expect[] = {191, 0, 255};
    for (int m = 0; m < 3; m++) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_REDUCTION_MODE_ARB, modes[m]);
        GLint q = 0;
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_REDUCTION_MODE_ARB, &q);
        CHECK_EQ(q, (GLint)modes[m]);
        drawQuad(p);
        uint8_t px[4];
        readPixel(8, 8, px);
        if (abs(px[0] - expect[m]) > 2) {
            testFail(__FILE__, __LINE__, "reduction 0x" + std::to_string(modes[m]) + ": got " + std::to_string(px[0]) +
                                             " expected " + std::to_string(expect[m]));
        }
    }
    CHECK_GL();
}
