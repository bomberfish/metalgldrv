// Texture copy / pixel storage / format tests.
#include "harness.h"
#include <algorithm>
#include <cstdint>
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
