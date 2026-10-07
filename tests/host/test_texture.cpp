// Texture copy / pixel storage / format tests.
#include "harness.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

// RGB9_E5 data must round-trip bit-exactly (non-canonical encodings included)
// through mutable multi-level 3D textures and glCopyImageSubData.
TEST(copy_image_rgb9e5_3d_bitexact) {
    const uint32_t v = 0xf8fc0000u;  // non-canonical encoding of 8064.0
    std::vector<uint32_t> l0(4 * 4 * 4, v), l1(2 * 2 * 2, v);
    GLuint t[2];
    glGenTextures(2, t);
    for (GLuint name : t) {
        glBindTexture(GL_TEXTURE_3D, name);
        glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB9_E5, 4, 4, 4, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l0.data());
        glTexImage3D(GL_TEXTURE_3D, 1, GL_RGB9_E5, 2, 2, 2, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l1.data());
        glTexImage3D(GL_TEXTURE_3D, 2, GL_RGB9_E5, 1, 1, 1, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, l1.data());
    }
    CHECK_GL();
    std::vector<uint32_t> out(64, 0);
    glBindTexture(GL_TEXTURE_3D, t[0]);
    glGetTexImage(GL_TEXTURE_3D, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
    CHECK_EQ(out[0], v);
    CHECK_EQ(out[63], v);
    glCopyImageSubData(t[0], GL_TEXTURE_3D, 0, 0, 0, 0, t[1], GL_TEXTURE_3D, 1, 0, 0, 0, 2, 2, 2);
    CHECK_GL();
    glBindTexture(GL_TEXTURE_3D, t[1]);
    glGetTexImage(GL_TEXTURE_3D, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
    CHECK_EQ(out[0], v);
    CHECK_EQ(out[7], v);
    glDeleteTextures(2, t);
}

static GLuint makeMip3D(GLenum target, int w0, int h0, int d0, const std::vector<uint32_t>& data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(target, t);
    for (int l = 0; l < 3; l++) {
        int w = std::max(1, w0 >> l), h = std::max(1, h0 >> l), d = target == GL_TEXTURE_3D ? std::max(1, d0 >> l) : d0;
        glTexImage3D(target, l, GL_RGB9_E5, w, h, d, 0, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, data.data());
        glTexParameteri(target, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(target, GL_TEXTURE_MAX_LEVEL, 2);
    }
    return t;
}

TEST(copy_image_3d_depth_region) {
    for (GLenum target : {GL_TEXTURE_3D, GL_TEXTURE_2D_ARRAY}) {
        std::vector<uint32_t> a(30 * 30 * 24, 0x11111111u), b(30 * 30 * 24, 0x22222222u);
        GLuint s = makeMip3D(target, 7, 7, 12, a);
        GLuint d = makeMip3D(target, 30, 30, target == GL_TEXTURE_3D ? 24 : 12, b);
        glCopyImageSubData(s, target, 0, 0, 0, 0, d, target, 1, 14, 14, 0, 1, 1, 12);
        CHECK_GL();
        std::vector<uint32_t> out(15 * 15 * 12, 0);
        glBindTexture(target, d);
        glGetTexImage(target, 1, GL_RGB, GL_UNSIGNED_INT_5_9_9_9_REV, out.data());
        CHECK_EQ(out[14 * 15 + 14], 0x11111111u);
        CHECK_EQ(out[11 * 225 + 14 * 15 + 14], 0x11111111u);
        CHECK_EQ(out[0], 0x22222222u);
        GLuint ts[2] = {s, d};
        glDeleteTextures(2, ts);
    }
}

static GLuint tex2D(GLenum ifmt, int w, int h, GLenum fmt, GLenum type, const void* data) {
    GLuint t;
    glGenTextures(1, &t);
    glBindTexture(GL_TEXTURE_2D, t);
    glTexImage2D(GL_TEXTURE_2D, 0, ifmt, w, h, 0, fmt, type, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    return t;
}

// imageSize()/imageStore() from graphics stages of a separable program pipeline.
static void imageSizeStage(GLenum stageBit, bool pipeline) {
    const char* vsPlain = "#version 430 core\nout gl_PerVertex { vec4 gl_Position; float gl_PointSize; };\n"
                          "void main() { gl_Position = vec4(0.0, 0.0, 0.0, 1.0); gl_PointSize = 1.0; }";
    const char* body =
        "layout(binding = 0, rgba32i) writeonly uniform iimage2D g_result;\n"
        "layout(binding = 1, rgba32f) uniform image2D g_image_2d;\n"
        "layout(binding = 2, rgba32f) uniform image3D g_image_3d;\n"
        "void main() {\n"
        "  imageStore(g_result, ivec2(0, 0), ivec4(imageSize(g_image_2d), 0, 0));\n"
        "  imageStore(g_result, ivec2(0, 1), ivec4(imageSize(g_image_3d), 0));\n";
    std::string vs = std::string("#version 430 core\n") + body + "  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);\n}\n";
    std::string fs = std::string("#version 430 core\n") + body + "}\n";
    GLuint t[3];
    glGenTextures(3, t);
    glBindTexture(GL_TEXTURE_2D, t[1]);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, 32, 16);
    glBindImageTexture(1, t[1], 0, GL_FALSE, 0, GL_READ_ONLY, GL_RGBA32F);
    glBindTexture(GL_TEXTURE_3D, t[2]);
    glTexStorage3D(GL_TEXTURE_3D, 1, GL_RGBA32F, 8, 4, 2);
    glBindImageTexture(2, t[2], 0, GL_TRUE, 0, GL_READ_ONLY, GL_RGBA32F);
    int init[8];
    for (int& v : init) v = 100000;
    glBindTexture(GL_TEXTURE_2D, t[0]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32I, 1, 2, 0, GL_RGBA_INTEGER, GL_INT, init);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glBindImageTexture(0, t[0], 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32I);
    GLuint vao, pipe = 0, p1 = 0, p2 = 0;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    if (pipeline) {
        glGenProgramPipelines(1, &pipe);
        const char* s1 = stageBit == GL_VERTEX_SHADER_BIT ? vs.c_str() : vsPlain;
        p1 = glCreateShaderProgramv(GL_VERTEX_SHADER, 1, &s1);
        glUseProgramStages(pipe, GL_VERTEX_SHADER_BIT, p1);
        if (stageBit == GL_FRAGMENT_SHADER_BIT) {
            const char* s2 = fs.c_str();
            p2 = glCreateShaderProgramv(GL_FRAGMENT_SHADER, 1, &s2);
            glUseProgramStages(pipe, GL_FRAGMENT_SHADER_BIT, p2);
        }
        glBindProgramPipeline(pipe);
    } else {
        std::string fsPlain = "#version 430 core\nout vec4 c; void main() { c = vec4(1.0); }";
        p1 = stageBit == GL_VERTEX_SHADER_BIT ? compileProgram(vs.c_str(), fsPlain.c_str()) : compileProgram(vsPlain, fs.c_str());
        glUseProgram(p1);
    }
    if (stageBit == GL_VERTEX_SHADER_BIT) glEnable(GL_RASTERIZER_DISCARD);
    glDrawArrays(GL_POINTS, 0, 1);
    glDisable(GL_RASTERIZER_DISCARD);
    CHECK_GL();
    glMemoryBarrier(GL_TEXTURE_UPDATE_BARRIER_BIT);
    int out[8] = {};
    glBindTexture(GL_TEXTURE_2D, t[0]);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_INT, out);
    CHECK_EQ(out[0], 32);
    CHECK_EQ(out[1], 16);
    CHECK_EQ(out[4], 8);
    CHECK_EQ(out[6], 2);
    glUseProgram(0);
    glBindProgramPipeline(0);
    if (pipe) glDeleteProgramPipelines(1, &pipe);
    glDeleteProgram(p1);
    if (p2) glDeleteProgram(p2);
    glDeleteVertexArrays(1, &vao);
    glDeleteTextures(3, t);
}

TEST(image_size_vs_program) { imageSizeStage(GL_VERTEX_SHADER_BIT, false); }
TEST(image_size_fs_program) { imageSizeStage(GL_FRAGMENT_SHADER_BIT, false); }
TEST(image_size_vs_pipeline) { imageSizeStage(GL_VERTEX_SHADER_BIT, true); }

// Rendering into an sRGB multisample texture without FRAMEBUFFER_SRGB stores the value
// as is; texelFetch decodes it.
TEST(srgb_multisample_texel_fetch) {
    GLuint ms, fbo[2], out;
    glGenTextures(1, &ms);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_SRGB8_ALPHA8, 4, 4, GL_FALSE);
    glGenFramebuffers(2, fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[0]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, ms, 0);
    CHECK_EQ(glCheckFramebufferStatus(GL_FRAMEBUFFER), (GLenum)GL_FRAMEBUFFER_COMPLETE);
    glViewport(0, 0, 4, 4);
    const char* vs = "#version 430 core\nvoid main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0; gl_Position = vec4(p, 0, 1); }";
    GLuint fill = compileProgram(vs, "#version 430 core\nout vec4 c; void main() { c = vec4(0.5, 0.25, 0.75, 1.0); }");
    GLuint vao;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(fill);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glGenTextures(1, &out);
    glBindTexture(GL_TEXTURE_2D, out);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 4, 4);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo[1]);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, out, 0);
    GLuint rd = compileProgram(vs, "#version 430 core\nuniform sampler2DMS s; out vec4 c;\n"
                                   "void main() { c = texelFetch(s, ivec2(gl_FragCoord.xy), 1); }");
    glUseProgram(rd);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, ms);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    uint8_t px[4] = {};
    glReadPixels(1, 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    // sRGB decode of 128/255, 64/255, 191/255
    CHECK(abs(px[0] - 55) <= 2);
    CHECK(abs(px[1] - 13) <= 2);
    CHECK(abs(px[2] - 133) <= 2);
    CHECK_GL();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glDeleteProgram(fill);
    glDeleteProgram(rd);
    glDeleteFramebuffers(2, fbo);
    glDeleteTextures(1, &ms);
    glDeleteTextures(1, &out);
    glDeleteVertexArrays(1, &vao);
}

// Raw bit reinterpretation between same-size formats of one view class.
TEST(copy_image_reinterpret) {
    uint32_t a[4] = {0x3ff003ffu, 0x12345678u, 0xc0000001u, 0xdeadbeefu}, out[4] = {};
    GLuint s = tex2D(GL_RGB10_A2, 2, 2, GL_RGBA, GL_UNSIGNED_INT_2_10_10_10_REV, a);
    GLuint d = tex2D(GL_R11F_G11F_B10F, 2, 2, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, nullptr);
    glCopyImageSubData(s, GL_TEXTURE_2D, 0, 0, 0, 0, d, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    CHECK_GL();
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_INT_10F_11F_11F_REV, out);
    CHECK(memcmp(a, out, 16) == 0);
    GLuint r = tex2D(GL_RGBA8UI, 2, 2, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, nullptr);
    glCopyImageSubData(d, GL_TEXTURE_2D, 0, 0, 0, 0, r, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, out);
    CHECK(memcmp(a, out, 16) == 0);
    // different view classes / sizes
    GLuint x = tex2D(GL_RGBA16F, 2, 2, GL_RGBA, GL_HALF_FLOAT, nullptr);
    glCopyImageSubData(s, GL_TEXTURE_2D, 0, 0, 0, 0, x, GL_TEXTURE_2D, 0, 0, 0, 0, 2, 2, 1);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_OPERATION);
    GLuint ts[4] = {s, d, r, x};
    glDeleteTextures(4, ts);
}

// Compressed <-> uncompressed: one 128-bit block <-> one RGBA32UI texel.
TEST(copy_image_compressed_uncompressed) {
    uint32_t blocks[2 * 2 * 4];
    for (int i = 0; i < 16; i++) blocks[i] = 0x01020304u * (i + 1);
    GLuint c;
    glGenTextures(1, &c);
    glBindTexture(GL_TEXTURE_2D, c);
    glCompressedTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RG_RGTC2, 8, 8, 0, sizeof blocks, blocks);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    GLuint u = tex2D(GL_RGBA32UI, 2, 2, GL_RGBA_INTEGER, GL_UNSIGNED_INT, nullptr);
    glCopyImageSubData(c, GL_TEXTURE_2D, 0, 0, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 8, 8, 1);
    CHECK_GL();
    uint32_t out[16] = {};
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA_INTEGER, GL_UNSIGNED_INT, out);
    CHECK(memcmp(blocks, out, sizeof out) == 0);
    // and back, into the second block row
    glCopyImageSubData(u, GL_TEXTURE_2D, 0, 0, 0, 0, c, GL_TEXTURE_2D, 0, 0, 4, 0, 2, 1, 1);
    CHECK_GL();
    uint32_t back[16] = {};
    glBindTexture(GL_TEXTURE_2D, c);
    glGetCompressedTexImage(GL_TEXTURE_2D, 0, back);
    CHECK(memcmp(back + 8, blocks, 32) == 0);
    // compressed storage specified through glTexImage2D
    GLuint c2;
    glGenTextures(1, &c2);
    glBindTexture(GL_TEXTURE_2D, c2);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_COMPRESSED_RG_RGTC2, 16, 16, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    CHECK_GL();
    glCopyImageSubData(c2, GL_TEXTURE_2D, 0, 0, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    CHECK_GL();
    glDeleteTextures(1, &c2);
    // unaligned compressed region
    glCopyImageSubData(c, GL_TEXTURE_2D, 0, 2, 0, 0, u, GL_TEXTURE_2D, 0, 0, 0, 0, 4, 4, 1);
    CHECK_EQ(glGetError(), (GLenum)GL_INVALID_VALUE);
    GLuint ts[2] = {c, u};
    glDeleteTextures(2, ts);
}

// Uncompressed data specified for compressed internal formats is CPU-encoded: the level
// reports the compressed format and samples back close to the source.
TEST(compressed_from_uncompressed_data) {
    struct Case { GLenum fmt; int channels; bool sgn; bool hdr; float tol; };
    const Case cases[] = {
        {GL_COMPRESSED_RGB_S3TC_DXT1_EXT, 3, false, false, 0.08f}, {GL_COMPRESSED_RGBA_S3TC_DXT1_EXT, 3, false, false, 0.08f},
        {GL_COMPRESSED_RGBA_S3TC_DXT3_EXT, 4, false, false, 0.08f}, {GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, 4, false, false, 0.08f},
        {GL_COMPRESSED_RED_RGTC1, 1, false, false, 0.03f}, {GL_COMPRESSED_SIGNED_RED_RGTC1, 1, true, false, 0.06f},
        {GL_COMPRESSED_RG_RGTC2, 2, false, false, 0.03f}, {GL_COMPRESSED_SIGNED_RG_RGTC2, 2, true, false, 0.06f},
        {GL_COMPRESSED_RGBA_BPTC_UNORM, 4, false, false, 0.05f}, {GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT, 3, false, true, 0.15f},
        {GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, 3, true, true, 0.15f}, {GL_COMPRESSED_RGB8_ETC2, 3, false, false, 0.12f},
        {GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2, 3, false, false, 0.2f}, {GL_COMPRESSED_RGBA8_ETC2_EAC, 4, false, false, 0.12f},
        {GL_COMPRESSED_R11_EAC, 1, false, false, 0.03f}, {GL_COMPRESSED_SIGNED_R11_EAC, 1, true, false, 0.03f},
        {GL_COMPRESSED_RG11_EAC, 2, false, false, 0.03f}, {GL_COMPRESSED_SIGNED_RG11_EAC, 2, true, false, 0.03f},
        {GL_COMPRESSED_RGBA, 4, false, false, 0.08f},
    };
    const int N = 8;
    // Smooth per-block content compresses near-losslessly.
    std::vector<float> src(N * N * 4);
    for (int y = 0; y < N; y++)
        for (int x = 0; x < N; x++) {
            float* p = &src[(y * N + x) * 4];
            p[0] = x / 7.0f;
            p[1] = x / 14.0f + 0.25f;
            p[2] = 0.6f - x / 28.0f;
            p[3] = 1.0f;
        }
    const char* vs = "#version 430 core\nvoid main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0; gl_Position = vec4(p, 0, 1); }";
    GLuint prog = compileProgram(vs, "#version 430 core\nuniform sampler2D s; out vec4 c;\n"
                                     "void main() { c = texelFetch(s, ivec2(gl_FragCoord.xy), 0); }");
    GLuint vao, fbo, rt;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA32F, N, N);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glViewport(0, 0, N, N);
    glUseProgram(prog);
    for (const Case& c : cases) {
        std::vector<float> in = src;
        for (size_t i = 0; i < in.size(); i++) {
            if (i % 4 == 3) continue;
            if (c.sgn) in[i] = in[i] * 2 - 1;
            if (c.hdr) in[i] *= 4;
        }
        GLuint t;
        glGenTextures(1, &t);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, t);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, c.fmt, N, N, 0, GL_RGBA, GL_FLOAT, in.data());
        CHECK_GL();
        GLint compressed = 0, size = 0;
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED, &compressed);
        glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_COMPRESSED_IMAGE_SIZE, &size);
        CHECK_EQ(compressed, 1);
        CHECK(size > 0);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        std::vector<float> out(N * N * 4);
        glReadPixels(0, 0, N, N, GL_RGBA, GL_FLOAT, out.data());
        float worst = 0;
        for (int i = 0; i < N * N; i++)
            for (int k = 0; k < c.channels; k++)
                worst = std::max(worst, std::fabs(out[i * 4 + k] - in[i * 4 + k]) / (c.hdr ? 4.0f : 1.0f));
        if (worst > c.tol) testFail(__FILE__, __LINE__, "format 0x" + std::to_string(c.fmt) + " max error " + std::to_string(worst));
        glDeleteTextures(1, &t);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    glDeleteTextures(1, &rt);
    glDeleteVertexArrays(1, &vao);
}

// A cube-map view of a 2D array texture samples the selected layers and levels.
TEST(texture_view_cube_from_array) {
    GLuint arr, view;
    glGenTextures(1, &arr);
    glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
    glTexStorage3D(GL_TEXTURE_2D_ARRAY, 3, GL_RGBA8, 4, 4, 7);
    glGenTextures(1, &view);
    glTextureView(view, GL_TEXTURE_CUBE_MAP, arr, GL_RGBA8, 1, 2, 1, 6);
    CHECK_GL();
    // Parent contents written after the view exists: uploads for even layers, FBO clears for odd ones.
    GLuint fill;
    glGenFramebuffers(1, &fill);
    glBindFramebuffer(GL_FRAMEBUFFER, fill);
    for (int l = 0; l < 3; l++)
        for (int s = 0; s < 7; s++) {
            int w = 4 >> l;
            if (s % 2 == 0) {
                std::vector<uint32_t> px(w * w, 0xff000000u | (uint32_t)(s * 30) | (uint32_t)(l * 80) << 8);
                glBindTexture(GL_TEXTURE_2D_ARRAY, arr);
                glTexSubImage3D(GL_TEXTURE_2D_ARRAY, l, 0, 0, s, w, w, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
            } else {
                glFramebufferTextureLayer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, arr, l, s);
                glClearColor(s * 30 / 255.0f, l * 80 / 255.0f, 0, 1);
                glClear(GL_COLOR_BUFFER_BIT);
            }
        }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glDeleteFramebuffers(1, &fill);
    glBindTexture(GL_TEXTURE_CUBE_MAP, view);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST_MIPMAP_NEAREST);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    const char* vs = "#version 430 core\nvoid main() { vec2 p = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 4.0 - 1.0; gl_Position = vec4(p, 0, 1); }";
    GLuint prog = compileProgram(vs, "#version 430 core\nuniform samplerCube s; out vec4 c;\n"
                                     "void main() { int i = int(gl_FragCoord.x); vec3 d[6] = vec3[6](vec3(1,0,0), vec3(-1,0,0), vec3(0,1,0),\n"
                                     "  vec3(0,-1,0), vec3(0,0,1), vec3(0,0,-1)); c = textureLod(s, d[i % 6], float(i / 6)); }");
    GLuint vao, fbo, rt;
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glGenTextures(1, &rt);
    glBindTexture(GL_TEXTURE_2D, rt);
    glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, 12, 1);
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt, 0);
    glViewport(0, 0, 12, 1);
    glUseProgram(prog);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_CUBE_MAP, view);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    uint32_t out[12] = {};
    glReadPixels(0, 0, 12, 1, GL_RGBA, GL_UNSIGNED_BYTE, out);
    for (int i = 0; i < 12; i++) {
        uint32_t expect = 0xff000000u | (uint32_t)((1 + i % 6) * 30) | (uint32_t)((1 + i / 6) * 80) << 8;
        if (out[i] != expect) testFail(__FILE__, __LINE__, "texel " + std::to_string(i) + " = " + std::to_string(out[i]) + " expected " + std::to_string(expect));
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glUseProgram(0);
    glDeleteProgram(prog);
    glDeleteFramebuffers(1, &fbo);
    GLuint ts[3] = {rt, view, arr};
    glDeleteTextures(3, ts);
    glDeleteVertexArrays(1, &vao);
}
