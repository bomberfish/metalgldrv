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

// KHR_blend_equation_advanced: GL_MULTIPLY_KHR and GL_SCREEN_KHR into a single-sample and a 4x
// multisample framebuffer (the shader reads the destination per sample), then resolved.
TEST(blend_equation_advanced_msaa) {
    if (!hasExtension("GL_KHR_blend_equation_advanced")) return;
    const char* vs = "#version 430 core\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); }";
    const char* fs = "#version 430 core\n#extension GL_KHR_blend_equation_advanced : require\n"
                     "layout(blend_support_multiply, blend_support_screen) out;\nlayout(location = 0) out vec4 c;\n"
                     "uniform vec4 col;\nvoid main() { c = col; }";
    GLuint p = compileProgram(vs, fs);
    glUseProgram(p);
    GLuint rbs[2], fbos[2];
    glGenRenderbuffers(2, rbs);
    glGenFramebuffers(2, fbos);
    for (int i = 0; i < 2; i++) {
        glBindRenderbuffer(GL_RENDERBUFFER, rbs[i]);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, i ? 4 : 0, GL_RGBA8, 16, 16);
        glBindFramebuffer(GL_FRAMEBUFFER, fbos[i]);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbs[i]);
    }
    for (int ms = 1; ms >= 0; ms--) {
        glBindFramebuffer(GL_FRAMEBUFFER, fbos[ms]);
        glViewport(0, 0, 16, 16);
        glDisable(GL_BLEND);
        glClearColor(0.5f, 1.0f, 0.25f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendEquation(GL_MULTIPLY_KHR);
        glUniform4f(glGetUniformLocation(p, "col"), 0.5f, 0.5f, 1.0f, 1.0f);
        drawQuad(p);  // (0.25, 0.5, 0.25)
        glBlendEquation(GL_SCREEN_KHR);
        glUniform4f(glGetUniformLocation(p, "col"), 0.5f, 0.0f, 0.0f, 1.0f);
        drawQuad(p);  // (0.625, 0.5, 0.25)
        glDisable(GL_BLEND);
        if (ms) {
            glBindFramebuffer(GL_READ_FRAMEBUFFER, fbos[1]);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbos[0]);
            glBlitFramebuffer(0, 0, 16, 16, 0, 0, 16, 16, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, fbos[0]);
        CHECK_PIXEL(8, 8, 159, 128, 64, 255);
    }
    // An advanced equation needs a matching blend_support qualifier.
    glEnable(GL_BLEND);
    glBlendEquation(GL_DARKEN_KHR);
    drawQuad(p);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    glBlendEquation(GL_FUNC_ADD);
    glDisable(GL_BLEND);
    CHECK_GL();
}

// GL_RGB9_E5 is color-renderable (and blendable): HDR values survive a draw with additive blending.
TEST(rgb9e5_renderbuffer) {
    const char* vs = "#version 330 core\nin vec2 pos;\nvoid main() { gl_Position = vec4(pos, 0, 1); }";
    const char* fs = "#version 330 core\nout vec4 c;\nvoid main() { c = vec4(0.25, 0.5, 2.0, 1.0); }";
    GLuint p = compileProgram(vs, fs);
    GLuint rb, fbo;
    glGenRenderbuffers(1, &rb);
    glBindRenderbuffer(GL_RENDERBUFFER, rb);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB9_E5, 16, 16);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rb);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 16, 16);
    glClearColor(1, 1, 1, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE);
    glUseProgram(p);
    drawQuad(p);
    glDisable(GL_BLEND);
    float px[3] = {};
    glReadPixels(8, 8, 1, 1, GL_RGB, GL_FLOAT, px);
    CHECK(fabsf(px[0] - 1.25f) < 0.01f && fabsf(px[1] - 1.5f) < 0.01f && fabsf(px[2] - 3.0f) < 0.02f);
    CHECK_GL();
}

// GL_APPLE_vertex_array_object: binding a name that wasn't generated creates the object.
TEST(apple_vertex_array_object) {
    CHECK(hasExtension("GL_APPLE_vertex_array_object"));
    auto bind = (void (*)(GLuint))eglGetProcAddress("glBindVertexArrayAPPLE");
    auto isVao = (GLboolean(*)(GLuint))eglGetProcAddress("glIsVertexArrayAPPLE");
    CHECK(bind && isVao);
    if (!bind || !isVao) return;
    bind(4242);
    CHECK_GL();
    CHECK(isVao(4242));
    GLint b = 0;
    glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &b);
    CHECK_EQ(b, 4242);
    glEnableVertexAttribArray(3);
    bind(0);
    GLint en = 1;
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en);
    CHECK_EQ(en, 0);  // state stayed with VAO 4242
    bind(4242);
    glGetVertexAttribiv(3, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &en);
    CHECK_EQ(en, 1);
    bind(0);
    GLuint name = 4242;
    glDeleteVertexArrays(1, &name);
    CHECK(!isVao(4242));
    CHECK_GL();
}

// GL_APPLE_fence: fences and object tests complete once the GPU has run the commands before them.
TEST(apple_fence) {
    CHECK(hasExtension("GL_APPLE_fence"));
    auto gen = (void (*)(GLsizei, GLuint*))eglGetProcAddress("glGenFencesAPPLE");
    auto del = (void (*)(GLsizei, const GLuint*))eglGetProcAddress("glDeleteFencesAPPLE");
    auto set = (void (*)(GLuint))eglGetProcAddress("glSetFenceAPPLE");
    auto isFence = (GLboolean(*)(GLuint))eglGetProcAddress("glIsFenceAPPLE");
    auto test = (GLboolean(*)(GLuint))eglGetProcAddress("glTestFenceAPPLE");
    auto finish = (void (*)(GLuint))eglGetProcAddress("glFinishFenceAPPLE");
    auto testObject = (GLboolean(*)(GLenum, GLuint))eglGetProcAddress("glTestObjectAPPLE");
    auto finishObject = (void (*)(GLenum, GLint))eglGetProcAddress("glFinishObjectAPPLE");
    GLuint f = 0;
    gen(1, &f);
    CHECK(f != 0);
    CHECK(!isFence(f));  // not set yet
    glClearColor(1, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    set(f);
    CHECK(isFence(f));
    finish(f);
    CHECK(test(f));
    CHECK(testObject(0x8A0B /* GL_FENCE_APPLE */, f));
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    uint32_t px[16] = {};
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    finishObject(GL_TEXTURE, (GLint)tex);
    CHECK(testObject(GL_TEXTURE, tex));
    CHECK_GL();
    testObject(GL_TEXTURE, 999999);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    del(1, &f);
    CHECK(!isFence(f));
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_APPLE_flush_buffer_range: with BUFFER_FLUSHING_UNMAP_APPLE off, glMapBuffer writes only the flushed ranges.
TEST(apple_flush_buffer_range) {
    CHECK(hasExtension("GL_APPLE_flush_buffer_range"));
    auto param = (void (*)(GLenum, GLenum, GLint))eglGetProcAddress("glBufferParameteriAPPLE");
    auto flush = (void (*)(GLenum, GLintptr, GLsizeiptr))eglGetProcAddress("glFlushMappedBufferRangeAPPLE");
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    uint8_t init[16] = {};
    glBufferData(GL_ARRAY_BUFFER, 16, init, GL_DYNAMIC_DRAW);
    param(GL_ARRAY_BUFFER, 0x8A13 /* GL_BUFFER_FLUSHING_UNMAP_APPLE */, GL_FALSE);
    param(GL_ARRAY_BUFFER, 0x8A12 /* GL_BUFFER_SERIALIZED_MODIFY_APPLE */, GL_FALSE);
    GLint v = 1;
    glGetBufferParameteriv(GL_ARRAY_BUFFER, 0x8A13, &v);
    CHECK_EQ(v, 0);
    auto* p = (uint8_t*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    CHECK(p != nullptr);
    if (!p) return;
    for (int i = 0; i < 16; i++) p[i] = 0x40 + i;
    flush(GL_ARRAY_BUFFER, 4, 4);
    CHECK(glUnmapBuffer(GL_ARRAY_BUFFER));
    uint8_t out[16];
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, 16, out);
    CHECK_EQ(out[4], 0x44);
    CHECK_EQ(out[7], 0x47);
    CHECK_GL();
    // Defaults: the whole mapping is written at unmap.
    param(GL_ARRAY_BUFFER, 0x8A13, GL_TRUE);
    p = (uint8_t*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
    p[15] = 0x99;
    flush(GL_ARRAY_BUFFER, 0, 4);  // allowed, nothing to do
    glUnmapBuffer(GL_ARRAY_BUFFER);
    glGetBufferSubData(GL_ARRAY_BUFFER, 0, 16, out);
    CHECK_EQ(out[15], 0x99);
    glDeleteBuffers(1, &b);
    CHECK_GL();
}

// GL_APPLE_packed_pixels (the GL 1.2 packed pixel types): BGRA UNSIGNED_INT_8_8_8_8_REV in, 5_6_5 and 8_8_8_8 out.
TEST(apple_packed_pixels) {
    CHECK(hasExtension("GL_APPLE_packed_pixels"));
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    uint32_t px = 0x80FF0000;  // ARGB in a uint: alpha 0x80, red 0xFF
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 1, 1, 0, GL_BGRA, GL_UNSIGNED_INT_8_8_8_8_REV, &px);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    uint16_t rgb565 = 0;
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, 1, 1, GL_RGB, GL_UNSIGNED_SHORT_5_6_5, &rgb565);
    CHECK_EQ(rgb565, 0xF800);
    uint32_t rgba = 0;
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, &rgba);
    CHECK_EQ(rgba, 0xFF000080u);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_APPLE_row_bytes: row strides in bytes for uploads and readbacks, overriding row length and alignment.
TEST(apple_row_bytes) {
    CHECK(hasExtension("GL_APPLE_row_bytes"));
    const GLenum UNPACK_ROW_BYTES = 0x8A16, PACK_ROW_BYTES = 0x8A15;
    // 2x2 RGBA8 with a 12-byte stride (4 bytes of padding per row)
    uint8_t src[24] = {1, 2, 3, 4, 5, 6, 7, 8, 0xEE, 0xEE, 0xEE, 0xEE, 9, 10, 11, 12, 13, 14, 15, 16, 0xEE, 0xEE, 0xEE, 0xEE};
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 7);  // ignored while ROW_BYTES is set
    glPixelStorei(UNPACK_ROW_BYTES, 12);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
    glPixelStorei(UNPACK_ROW_BYTES, 0);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    GLint v = -1;
    glGetIntegerv(UNPACK_ROW_BYTES, &v);
    CHECK_EQ(v, 0);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    uint8_t out[32];
    memset(out, 0xAA, sizeof out);
    glPixelStorei(PACK_ROW_BYTES, 16);
    glReadPixels(0, 0, 2, 2, GL_RGBA, GL_UNSIGNED_BYTE, out);
    glPixelStorei(PACK_ROW_BYTES, 0);
    CHECK_EQ(out[0], 1);
    CHECK_EQ(out[7], 8);
    CHECK_EQ(out[8], 0xAA);  // padding untouched
    CHECK_EQ(out[16], 9);
    CHECK_EQ(out[23], 16);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_APPLE_flush_render: the entry points resolve and submit work.
TEST(apple_flush_render) {
    CHECK(hasExtension("GL_APPLE_flush_render"));
    auto flushRender = (void (*)())eglGetProcAddress("glFlushRenderAPPLE");
    auto finishRender = (void (*)())eglGetProcAddress("glFinishRenderAPPLE");
    auto swap = (void (*)())eglGetProcAddress("glSwapAPPLE");
    CHECK(flushRender && finishRender && swap);
    if (!flushRender || !finishRender || !swap) return;
    glClearColor(0, 1, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    flushRender();
    finishRender();
    swap();  // a pbuffer: just a flush
    CHECK_PIXEL(4, 4, 0, 255, 0, 255);
    CHECK_GL();
}

// GL_APPLE_object_purgeable: volatile objects keep their contents unless the system reclaims them; released
// ones come back undefined.
TEST(apple_object_purgeable) {
    CHECK(hasExtension("GL_APPLE_object_purgeable"));
    auto purgeable = (GLenum(*)(GLenum, GLuint, GLenum))eglGetProcAddress("glObjectPurgeableAPPLE");
    auto unpurgeable = (GLenum(*)(GLenum, GLuint, GLenum))eglGetProcAddress("glObjectUnpurgeableAPPLE");
    auto getParam = (void (*)(GLenum, GLuint, GLenum, GLint*))eglGetProcAddress("glGetObjectParameterivAPPLE");
    const GLenum BUFFER_OBJECT = 0x85B3, VOLATILE = 0x8A1A, RELEASED = 0x8A19, RETAINED = 0x8A1B, UNDEFINED = 0x8A1C,
                 PURGEABLE = 0x8A1D;
    GLuint b;
    glGenBuffers(1, &b);
    glBindBuffer(GL_ARRAY_BUFFER, b);
    uint32_t data[256];
    for (int i = 0; i < 256; i++) data[i] = i;
    glBufferData(GL_ARRAY_BUFFER, sizeof data, data, GL_STATIC_DRAW);
    CHECK_EQ(purgeable(BUFFER_OBJECT, b, VOLATILE), VOLATILE);
    GLint p = 0;
    getParam(BUFFER_OBJECT, b, PURGEABLE, &p);
    CHECK_EQ(p, GL_TRUE);
    purgeable(BUFFER_OBJECT, b, VOLATILE);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);  // already purgeable
    GLenum r = unpurgeable(BUFFER_OBJECT, b, RETAINED);
    CHECK(r == RETAINED || r == UNDEFINED);
    if (r == RETAINED) {
        uint32_t out[4];
        glGetBufferSubData(GL_ARRAY_BUFFER, 4 * 100, sizeof out, out);
        CHECK_EQ(out[0], 100u);
    }
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
    CHECK_EQ(purgeable(GL_TEXTURE, tex, RELEASED), RELEASED);
    CHECK_EQ(unpurgeable(GL_TEXTURE, tex, UNDEFINED), UNDEFINED);
    getParam(GL_TEXTURE, tex, PURGEABLE, &p);
    CHECK_EQ(p, GL_FALSE);
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &b);
    CHECK_GL();
}

// GL_APPLE_client_storage: the pixel store flag is accepted and uploads behave as usual.
TEST(apple_client_storage) {
    CHECK(hasExtension("GL_APPLE_client_storage"));
    const GLenum UNPACK_CLIENT_STORAGE = 0x85B2;
    glPixelStorei(UNPACK_CLIENT_STORAGE, GL_TRUE);
    GLboolean on = GL_FALSE;
    glGetBooleanv(UNPACK_CLIENT_STORAGE, &on);
    CHECK(on);
    uint32_t px[4] = {0xFF0000FF, 0xFF0000FF, 0xFF0000FF, 0xFF0000FF};
    GLuint tex, fbo;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2, 2, 0, GL_RGBA, GL_UNSIGNED_BYTE, px);
    memset(px, 0, sizeof px);  // the app may not do this with client storage, but a copy doesn't mind
    glPixelStorei(UNPACK_CLIENT_STORAGE, GL_FALSE);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    CHECK_PIXEL(1, 1, 255, 0, 0, 255);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}

// GL_APPLE_texture_range: the range and the storage hint are recorded per texture.
TEST(apple_texture_range) {
    CHECK(hasExtension("GL_APPLE_texture_range"));
    auto range = (void (*)(GLenum, GLsizei, const void*))eglGetProcAddress("glTextureRangeAPPLE");
    auto getPtr = (void (*)(GLenum, GLenum, void**))eglGetProcAddress("glGetTexParameterPointervAPPLE");
    const GLenum RANGE_LENGTH = 0x85B7, RANGE_POINTER = 0x85B8, STORAGE_HINT = 0x85BC, STORAGE_CACHED = 0x85BE;
    static uint8_t mem[4096];
    GLuint tex;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    range(GL_TEXTURE_2D, sizeof mem, mem);
    glTexParameteri(GL_TEXTURE_2D, STORAGE_HINT, STORAGE_CACHED);
    GLint v = 0;
    glGetTexParameteriv(GL_TEXTURE_2D, RANGE_LENGTH, &v);
    CHECK_EQ(v, (GLint)sizeof mem);
    glGetTexParameteriv(GL_TEXTURE_2D, STORAGE_HINT, &v);
    CHECK_EQ(v, (GLint)STORAGE_CACHED);
    void* p = nullptr;
    getPtr(GL_TEXTURE_2D, RANGE_POINTER, &p);
    CHECK(p == mem);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 16, 0, GL_RGBA, GL_UNSIGNED_BYTE, mem);
    glDeleteTextures(1, &tex);
    CHECK_GL();
}
